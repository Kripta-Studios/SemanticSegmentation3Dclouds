from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.classes import CLASS_NAMES, IGNORE_INDEX
from src.data.blocks import assign_split, make_blocks_from_pair
from src.data.pnoa import find_tile_pairs
from src.utils.progress import eta_line


def tile_signature(args: dict) -> dict:
    return {
        "tile_size": args["tile_size"],
        "stride": args["stride"],
        "points_per_block": args["points_per_block"],
        "min_points": args["min_points"],
        "val_ratio": args["val_ratio"],
        "test_ratio": args["test_ratio"],
        "split_mode": args["split_mode"],
        "seed": args["seed"],
        "force_split": args.get("force_split") or "",
    }


def marker_path(out_root: str, tile_id: str) -> Path:
    return Path(out_root) / "_tile_done" / f"{tile_id}.json"


def load_done_marker(out_root: str, tile_id: str, signature: dict) -> dict | None:
    path = marker_path(out_root, tile_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("signature") != signature:
        return None
    stats = payload.get("stats")
    if not isinstance(stats, dict):
        return None
    stats = dict(stats)
    stats["tile_skipped_by_marker"] = True
    return stats


def write_done_marker(out_root: str, stats: dict, signature: dict) -> None:
    path = marker_path(out_root, stats["tile_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps({"signature": signature, "stats": stats}, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def process_pair(payload: dict) -> dict:
    pair = payload["pair"]
    args = payload["args"]
    signature = tile_signature(args)
    if not args["no_skip_existing"]:
        marker = load_done_marker(args["out"], pair.tile_id, signature)
        if marker is not None:
            return marker
    stats = make_blocks_from_pair(
        pair.col_path,
        pair.cir_path,
        args["out"],
        tile_size=args["tile_size"],
        stride=args["stride"],
        points_per_block=args["points_per_block"],
        min_points=args["min_points"],
        val_ratio=args["val_ratio"],
        test_ratio=args["test_ratio"],
        split_mode=args["split_mode"],
        seed=args["seed"],
        skip_existing=not args["no_skip_existing"],
        force_split=args.get("force_split") or None,
    )
    if not args["no_skip_existing"]:
        write_done_marker(args["out"], stats, signature)
    return stats


def update_summary(summary: dict, stats: dict) -> None:
    summary["tiles_ok"] += 1
    split = stats["split"]
    summary["tiles_by_split"][split] += 1
    if stats.get("tile_skipped_by_marker"):
        summary["tiles_skipped_by_marker"] += 1
    summary["blocks_by_split"][split] += stats["blocks"]
    summary["points_by_split"][split] += stats["points_written"]
    summary["written_blocks"] += int(stats.get("written_blocks", 0))
    summary["skipped_blocks"] += int(stats.get("skipped_blocks", 0))
    for cls, count in stats["class_counts"].items():
        summary["class_counts"][cls] += count
        summary["class_counts_by_split"][split][cls] += count
    summary["tile_stats"].append(stats)


def class_distribution_rows(counts: dict | Counter) -> list[dict]:
    total_all = sum(int(counts.get(str(cls), counts.get(cls, 0))) for cls in range(7))
    total_reliable = sum(int(counts.get(str(cls), counts.get(cls, 0))) for cls in range(IGNORE_INDEX))
    rows = []
    for cls in range(7):
        points = int(counts.get(str(cls), counts.get(cls, 0)))
        denom = total_reliable if cls != IGNORE_INDEX else total_all
        rows.append(
            {
                "class_id": cls,
                "class_name": CLASS_NAMES[cls],
                "points": points,
                "pct_reliable": float(points / total_reliable * 100.0) if cls != IGNORE_INDEX and total_reliable else 0.0,
                "pct_all": float(points / total_all * 100.0) if total_all else 0.0,
            }
        )
    return rows


def split_balance_warnings(class_counts_by_split: dict) -> list[str]:
    warnings = []
    train_counts = class_counts_by_split.get("train", {})
    for split, counts in sorted(class_counts_by_split.items()):
        if split == "train":
            continue
        for cls in range(IGNORE_INDEX):
            split_points = int(counts.get(str(cls), counts.get(cls, 0)))
            train_points = int(train_counts.get(str(cls), train_counts.get(cls, 0)))
            if split_points > 0 and train_points == 0:
                warnings.append(f"{CLASS_NAMES[cls]} appears in {split} but has zero training points")
            elif train_points > 0 and split_points == 0:
                warnings.append(f"{CLASS_NAMES[cls]} appears in train but has zero {split} points")
    for cls in range(IGNORE_INDEX):
        train_points = int(train_counts.get(str(cls), train_counts.get(cls, 0)))
        if 0 < train_points < 10_000:
            warnings.append(f"{CLASS_NAMES[cls]} has only {train_points} sampled training points")
    return warnings


def select_balanced_pairs(pairs: list, max_tiles: int, split_mode: str, val_ratio: float, test_ratio: float) -> list:
    if max_tiles <= 0 or len(pairs) <= max_tiles:
        return pairs
    groups = defaultdict(list)
    for pair in pairs:
        split = assign_split(pair.tile_id, pair.campaign, val_ratio=val_ratio, test_ratio=test_ratio, split_mode=split_mode)
        groups[(pair.campaign, split)].append(pair)
    selected = []
    keys = sorted(groups)
    index = 0
    while len(selected) < max_tiles:
        added = False
        for key in keys:
            group = groups[key]
            if index < len(group):
                selected.append(group[index])
                added = True
                if len(selected) >= max_tiles:
                    break
        if not added:
            break
        index += 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert paired PNOA COL/CIR LAZ files into train/val/test block tensors.")
    parser.add_argument("--raw", default="data/raw/pnoa_galicia")
    parser.add_argument("--out", default="data/processed/galicia_blocks")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--tile-size", type=float, default=50.0)
    parser.add_argument("--stride", type=float, default=50.0)
    parser.add_argument("--points-per-block", type=int, default=8192)
    parser.add_argument("--min-points", type=int, default=100)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--split-mode", choices=["mixed", "campaign", "hash"], default="mixed")
    parser.add_argument("--force-split", choices=["train", "val", "test"], default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-tiles", type=int, default=0)
    parser.add_argument("--tile-contains", default="")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    pairs = find_tile_pairs(args.raw)
    if args.tile_contains:
        pairs = [pair for pair in pairs if args.tile_contains in pair.tile_id]
    if args.max_tiles > 0:
        pairs = select_balanced_pairs(pairs, args.max_tiles, args.split_mode, args.val_ratio, args.test_ratio)
    if not pairs:
        raise SystemExit(f"No COL/CIR pairs found in {args.raw}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    summary = {
        "tiles_requested": len(pairs),
        "tiles_ok": 0,
        "tiles_skipped_by_marker": 0,
        "tiles_error": 0,
        "errors": [],
        "written_blocks": 0,
        "skipped_blocks": 0,
        "tiles_by_split": Counter(),
        "blocks_by_split": Counter(),
        "points_by_split": Counter(),
        "class_counts": Counter(),
        "class_counts_by_split": defaultdict(Counter),
        "tile_stats": [],
    }
    worker_args = {
        "out": args.out,
        "tile_size": args.tile_size,
        "stride": args.stride,
        "points_per_block": args.points_per_block,
        "min_points": args.min_points,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "split_mode": args.split_mode,
        "force_split": args.force_split,
        "seed": args.seed,
        "no_skip_existing": args.no_skip_existing,
    }
    start_time = time.perf_counter()
    if args.num_workers <= 1:
        for i, pair in enumerate(pairs, 1):
            print(f"[{i}/{len(pairs)}] {pair.tile_id}")
            status = "ok"
            try:
                update_summary(summary, process_pair({"pair": pair, "args": worker_args}))
            except Exception as exc:
                summary["tiles_error"] += 1
                summary["errors"].append({"tile_id": pair.tile_id, "error": f"{type(exc).__name__}: {exc}"})
                print(f"ERROR {pair.tile_id}: {exc}")
                status = "error"
            print(f"{eta_line('prepare blocks', start_time, i, len(pairs))} | {status} {pair.tile_id}")
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_pair, {"pair": pair, "args": worker_args}): pair for pair in pairs}
            for i, future in enumerate(as_completed(futures), 1):
                pair = futures[future]
                status = "ok"
                try:
                    update_summary(summary, future.result())
                except Exception as exc:
                    summary["tiles_error"] += 1
                    summary["errors"].append({"tile_id": pair.tile_id, "error": f"{type(exc).__name__}: {exc}"})
                    print(f"ERROR {pair.tile_id}: {exc}")
                    status = "error"
                print(f"{eta_line('prepare blocks', start_time, i, len(pairs))} | {status} {pair.tile_id}")

    payload = {
        **summary,
        "tiles_by_split": dict(summary["tiles_by_split"]),
        "blocks_by_split": dict(summary["blocks_by_split"]),
        "points_by_split": dict(summary["points_by_split"]),
        "class_counts": dict(summary["class_counts"]),
        "class_counts_by_split": {split: dict(counts) for split, counts in summary["class_counts_by_split"].items()},
    }
    payload["split_percentages"] = {
        split: {
            "tiles_pct": float(count / max(sum(payload["tiles_by_split"].values()), 1) * 100.0),
            "blocks_pct": float(payload["blocks_by_split"].get(split, 0) / max(sum(payload["blocks_by_split"].values()), 1) * 100.0),
            "points_pct": float(payload["points_by_split"].get(split, 0) / max(sum(payload["points_by_split"].values()), 1) * 100.0),
        }
        for split, count in payload["tiles_by_split"].items()
    }
    payload["class_distribution_by_split"] = {
        split: class_distribution_rows(counts)
        for split, counts in payload["class_counts_by_split"].items()
    }
    payload["balance_warnings"] = split_balance_warnings(payload["class_counts_by_split"])
    (reports / "prepare_blocks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (Path(args.out) / "_prepare_complete.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["tiles_ok", "tiles_skipped_by_marker", "tiles_error", "written_blocks", "skipped_blocks", "tiles_by_split", "blocks_by_split", "points_by_split", "split_percentages", "balance_warnings"]}, indent=2))
    print(json.dumps(payload["class_distribution_by_split"], indent=2))


if __name__ == "__main__":
    main()
