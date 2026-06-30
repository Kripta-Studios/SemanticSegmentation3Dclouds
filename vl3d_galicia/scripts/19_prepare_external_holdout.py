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
from src.data.blocks import make_blocks_from_pair
from src.data.pnoa import find_tile_pairs
from src.utils.progress import eta_line


def class_rows(counts: Counter) -> list[dict]:
    total_reliable = sum(int(counts.get(str(cls), counts.get(cls, 0))) for cls in range(IGNORE_INDEX))
    total_all = sum(int(counts.get(str(cls), counts.get(cls, 0))) for cls in range(7))
    rows = []
    for cls in range(7):
        points = int(counts.get(str(cls), counts.get(cls, 0)))
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


def process_pair(payload: dict) -> dict:
    pair = payload["pair"]
    args = payload["args"]
    return make_blocks_from_pair(
        pair.col_path,
        pair.cir_path,
        args["out"],
        tile_size=args["tile_size"],
        stride=args["stride"],
        points_per_block=args["points_per_block"],
        min_points=args["min_points"],
        val_ratio=0.0,
        test_ratio=1.0,
        split_mode="hash",
        seed=args["seed"],
        skip_existing=not args["no_skip_existing"],
        force_split="test",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare non-training external CCAA LAZ data as a pure test holdout.")
    parser.add_argument("--raw", default="data/raw/pnoa_varias_ccaa")
    parser.add_argument("--out", default="data/processed/pnoa_varias_ccaa_holdout")
    parser.add_argument("--reports", default="reports/external_holdout")
    parser.add_argument("--include-campaign-prefix", action="append", default=[])
    parser.add_argument("--exclude-campaign-prefix", action="append", default=["GAL"])
    parser.add_argument(
        "--tile-list",
        default="",
        help="Optional text file with one tile_id per line. If provided, only those paired tiles are prepared.",
    )
    parser.add_argument("--max-tiles", type=int, default=0)
    parser.add_argument("--tile-size", type=float, default=50.0)
    parser.add_argument("--stride", type=float, default=50.0)
    parser.add_argument("--points-per-block", type=int, default=8192)
    parser.add_argument("--min-points", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    pairs = find_tile_pairs(args.raw)
    include_prefixes = tuple(args.include_campaign_prefix)
    exclude_prefixes = tuple(args.exclude_campaign_prefix)
    if include_prefixes:
        pairs = [pair for pair in pairs if pair.campaign.startswith(include_prefixes)]
    if exclude_prefixes:
        pairs = [pair for pair in pairs if not pair.campaign.startswith(exclude_prefixes)]
    requested_tile_ids: list[str] = []
    if args.tile_list:
        tile_list_path = Path(args.tile_list)
        requested_tile_ids = [
            line.strip()
            for line in tile_list_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        pair_by_id = {pair.tile_id: pair for pair in pairs}
        missing = [tile_id for tile_id in requested_tile_ids if tile_id not in pair_by_id]
        if missing:
            raise SystemExit(f"{len(missing)} requested tile IDs are missing after campaign filters: {missing[:10]}")
        pairs = [pair_by_id[tile_id] for tile_id in requested_tile_ids]
    if args.max_tiles > 0:
        pairs = pairs[: args.max_tiles]
    if not pairs:
        raise SystemExit(
            "No external holdout pairs selected. "
            "Check --include-campaign-prefix/--exclude-campaign-prefix."
        )

    out = Path(args.out)
    reports = Path(args.reports)
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    worker_args = {
        "out": args.out,
        "tile_size": args.tile_size,
        "stride": args.stride,
        "points_per_block": args.points_per_block,
        "min_points": args.min_points,
        "seed": args.seed,
        "no_skip_existing": args.no_skip_existing,
    }
    summary = {
        "raw": str(Path(args.raw).resolve()),
        "out": str(out.resolve()),
        "force_split": "test",
        "excluded_campaign_prefixes": list(exclude_prefixes),
        "included_campaign_prefixes": list(include_prefixes),
        "tile_list": str(Path(args.tile_list).resolve()) if args.tile_list else "",
        "requested_tile_ids": requested_tile_ids,
        "tiles_requested": len(pairs),
        "tiles_ok": 0,
        "tiles_error": 0,
        "errors": [],
        "campaigns": Counter(),
        "blocks_by_campaign": Counter(),
        "points_by_campaign": Counter(),
        "class_counts": Counter(),
        "class_counts_by_campaign": defaultdict(Counter),
        "tile_stats": [],
    }
    start = time.perf_counter()
    if args.num_workers <= 1:
        for done, pair in enumerate(pairs, 1):
            try:
                stats = process_pair({"pair": pair, "args": worker_args})
                summary["tiles_ok"] += 1
                summary["campaigns"][stats["campaign"]] += 1
                summary["blocks_by_campaign"][stats["campaign"]] += int(stats["blocks"])
                summary["points_by_campaign"][stats["campaign"]] += int(stats["points_written"])
                for cls, count in stats["class_counts"].items():
                    summary["class_counts"][cls] += int(count)
                    summary["class_counts_by_campaign"][stats["campaign"]][cls] += int(count)
                summary["tile_stats"].append(stats)
                status = "ok"
            except Exception as exc:
                summary["tiles_error"] += 1
                summary["errors"].append({"tile_id": pair.tile_id, "campaign": pair.campaign, "error": f"{type(exc).__name__}: {exc}"})
                status = "error"
            print(f"{eta_line('external holdout prepare', start, done, len(pairs))} | {status} {pair.tile_id}")
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_pair, {"pair": pair, "args": worker_args}): pair for pair in pairs}
            for done, future in enumerate(as_completed(futures), 1):
                pair = futures[future]
                try:
                    stats = future.result()
                    summary["tiles_ok"] += 1
                    summary["campaigns"][stats["campaign"]] += 1
                    summary["blocks_by_campaign"][stats["campaign"]] += int(stats["blocks"])
                    summary["points_by_campaign"][stats["campaign"]] += int(stats["points_written"])
                    for cls, count in stats["class_counts"].items():
                        summary["class_counts"][cls] += int(count)
                        summary["class_counts_by_campaign"][stats["campaign"]][cls] += int(count)
                    summary["tile_stats"].append(stats)
                    status = "ok"
                except Exception as exc:
                    summary["tiles_error"] += 1
                    summary["errors"].append({"tile_id": pair.tile_id, "campaign": pair.campaign, "error": f"{type(exc).__name__}: {exc}"})
                    status = "error"
                print(f"{eta_line('external holdout prepare', start, done, len(pairs))} | {status} {pair.tile_id}")

    payload = {
        **summary,
        "campaigns": dict(summary["campaigns"]),
        "blocks_by_campaign": dict(summary["blocks_by_campaign"]),
        "points_by_campaign": dict(summary["points_by_campaign"]),
        "class_counts": dict(summary["class_counts"]),
        "class_distribution": class_rows(summary["class_counts"]),
        "class_distribution_by_campaign": {
            campaign: class_rows(counts)
            for campaign, counts in summary["class_counts_by_campaign"].items()
        },
    }
    (reports / "external_holdout_prepare.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "_external_holdout_complete.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["tiles_ok", "tiles_error", "campaigns", "blocks_by_campaign", "points_by_campaign", "class_distribution"]}, indent=2))


if __name__ == "__main__":
    main()
