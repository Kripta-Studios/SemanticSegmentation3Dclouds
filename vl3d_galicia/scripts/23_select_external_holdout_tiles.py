from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import laspy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.classes import CLASS_NAMES, IGNORE_INDEX, map_labels
from src.data.pnoa import find_tile_pairs
from src.utils.progress import eta_line


CLASS_PRIORITY = {
    0: 0.5,  # ground is abundant.
    1: 3.0,  # low vegetation is hard and under-represented.
    2: 3.0,  # medium vegetation is hard and under-represented.
    3: 1.5,
    4: 5.0,  # building is rare in the external CAT subset.
    5: 6.0,  # water is often absent; include it if available.
}


def count_project_classes(col_path: str, chunk_size: int) -> tuple[int, list[int]]:
    counts = np.zeros(IGNORE_INDEX + 1, dtype=np.int64)
    with laspy.open(col_path) as reader:
        total = int(reader.header.point_count)
        dims = {dim.name for dim in reader.header.point_format.dimensions}
        if "classification" not in dims:
            counts[IGNORE_INDEX] = total
            return total, counts.tolist()
        for chunk in reader.chunk_iterator(chunk_size):
            labels = map_labels(np.asarray(chunk.classification, dtype=np.uint8))
            counts += np.bincount(labels, minlength=IGNORE_INDEX + 1).astype(np.int64)
    return total, counts.tolist()


def audit_pair(payload: dict) -> dict:
    pair = payload["pair"]
    chunk_size = int(payload["chunk_size"])
    total, counts = count_project_classes(str(pair.col_path), chunk_size=chunk_size)
    file_bytes = int(pair.col_path.stat().st_size + pair.cir_path.stat().st_size)
    reliable = int(sum(counts[:IGNORE_INDEX]))
    row = {
        "tile_id": pair.tile_id,
        "campaign": pair.campaign,
        "col_path": str(pair.col_path),
        "cir_path": str(pair.cir_path),
        "file_bytes": file_bytes,
        "point_count": int(total),
        "reliable_points": reliable,
        "unreliable_points": int(counts[IGNORE_INDEX]),
    }
    for cls in range(IGNORE_INDEX + 1):
        row[f"class_{cls}"] = int(counts[cls])
        row[f"class_{cls}_name"] = CLASS_NAMES[cls]
    return row


def class_count(row: dict, cls: int) -> int:
    return int(row.get(f"class_{cls}", 0))


def select_tiles(rows: list[dict], target_tiles: int) -> list[dict]:
    if target_tiles <= 0 or target_tiles >= len(rows):
        return rows
    selected: list[dict] = []
    selected_ids: set[str] = set()

    def add(row: dict) -> None:
        if row["tile_id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["tile_id"])

    present_classes = [cls for cls in range(IGNORE_INDEX) if sum(class_count(row, cls) for row in rows) > 0]
    for cls in sorted(present_classes, key=lambda item: CLASS_PRIORITY.get(item, 1.0), reverse=True):
        if len(selected) >= target_tiles:
            break
        best = max(
            (row for row in rows if row["tile_id"] not in selected_ids),
            key=lambda row: (class_count(row, cls), class_count(row, cls) / max(int(row["reliable_points"]), 1)),
            default=None,
        )
        if best is not None and class_count(best, cls) > 0:
            add(best)

    running = Counter()
    for row in selected:
        for cls in range(IGNORE_INDEX):
            running[cls] += class_count(row, cls)

    while len(selected) < target_tiles:
        best_row = None
        best_score = -1.0
        for row in rows:
            if row["tile_id"] in selected_ids:
                continue
            reliable = max(int(row["reliable_points"]), 1)
            score = 0.0
            for cls in range(IGNORE_INDEX):
                count = class_count(row, cls)
                if count <= 0:
                    continue
                rarity_gain = math.log1p(count) / math.sqrt(float(running[cls] + 1))
                ratio_gain = count / reliable
                score += CLASS_PRIORITY.get(cls, 1.0) * (rarity_gain + ratio_gain)
            score += math.log1p(reliable) * 0.01
            if score > best_score:
                best_score = score
                best_row = row
        if best_row is None:
            break
        add(best_row)
        for cls in range(IGNORE_INDEX):
            running[cls] += class_count(best_row, cls)
    return selected


def distribution(rows: list[dict]) -> list[dict]:
    counts = Counter()
    total_all = 0
    total_reliable = 0
    for row in rows:
        total_all += int(row["point_count"])
        total_reliable += int(row["reliable_points"])
        for cls in range(IGNORE_INDEX + 1):
            counts[cls] += class_count(row, cls)
    out = []
    for cls in range(IGNORE_INDEX + 1):
        points = int(counts[cls])
        out.append(
            {
                "class_id": cls,
                "class_name": CLASS_NAMES[cls],
                "points": points,
                "pct_reliable": float(points / total_reliable * 100.0) if cls != IGNORE_INDEX and total_reliable else 0.0,
                "pct_all": float(points / total_all * 100.0) if total_all else 0.0,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit external PNOA CCAA tiles and select a class-stratified holdout subset.")
    parser.add_argument("--raw", default="data/raw/pnoa_varias_ccaa")
    parser.add_argument("--reports", default="reports/external_holdout_selection")
    parser.add_argument("--include-campaign-prefix", action="append", default=[])
    parser.add_argument("--exclude-campaign-prefix", action="append", default=["GAL"])
    parser.add_argument("--target-tiles", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--manifest-in",
        default="",
        help="Optional existing tile_class_manifest.json to reuse instead of auditing raw LAZ files again.",
    )
    parser.add_argument(
        "--exclude-tile-list",
        default="",
        help="Optional text file with tile_id values that must not be selected.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    manifest_json = reports / "tile_class_manifest.json"
    manifest_csv = reports / "tile_class_manifest.csv"

    if args.manifest_in:
        rows = json.loads(Path(args.manifest_in).read_text(encoding="utf-8"))
        write_csv(manifest_csv, rows)
        manifest_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    elif manifest_json.exists() and not args.force:
        rows = json.loads(manifest_json.read_text(encoding="utf-8"))
    else:
        pairs = find_tile_pairs(args.raw)
        include_prefixes = tuple(args.include_campaign_prefix)
        exclude_prefixes = tuple(args.exclude_campaign_prefix)
        if include_prefixes:
            pairs = [pair for pair in pairs if pair.campaign.startswith(include_prefixes)]
        if exclude_prefixes:
            pairs = [pair for pair in pairs if not pair.campaign.startswith(exclude_prefixes)]
        if not pairs:
            raise SystemExit("No pairs selected for audit.")
        rows = []
        start = time.perf_counter()
        if args.num_workers <= 1:
            for done, pair in enumerate(pairs, 1):
                row = audit_pair({"pair": pair, "chunk_size": args.chunk_size})
                rows.append(row)
                print(f"{eta_line('external class audit', start, done, len(pairs))} | {pair.tile_id}")
        else:
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                futures = {executor.submit(audit_pair, {"pair": pair, "chunk_size": args.chunk_size}): pair for pair in pairs}
                for done, future in enumerate(as_completed(futures), 1):
                    pair = futures[future]
                    row = future.result()
                    rows.append(row)
                    print(f"{eta_line('external class audit', start, done, len(pairs))} | {pair.tile_id}")
        rows = sorted(rows, key=lambda row: row["tile_id"])
        manifest_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        write_csv(manifest_csv, rows)

    excluded_tile_ids: set[str] = set()
    if args.exclude_tile_list:
        excluded_tile_ids = {
            line.strip()
            for line in Path(args.exclude_tile_list).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        rows = [row for row in rows if row["tile_id"] not in excluded_tile_ids]
        if not rows:
            raise SystemExit("No candidate rows remain after applying --exclude-tile-list.")

    selected = select_tiles(rows, target_tiles=args.target_tiles)
    selected = sorted(selected, key=lambda row: row["tile_id"])
    selected_ids = [row["tile_id"] for row in selected]
    (reports / "selected_tiles.txt").write_text("\n".join(selected_ids) + "\n", encoding="utf-8")
    selection_payload = {
        "raw": str(Path(args.raw).resolve()),
        "target_tiles": args.target_tiles,
        "manifest_in": str(Path(args.manifest_in).resolve()) if args.manifest_in else "",
        "exclude_tile_list": str(Path(args.exclude_tile_list).resolve()) if args.exclude_tile_list else "",
        "excluded_tile_count": len(excluded_tile_ids),
        "selected_tiles": selected_ids,
        "selected_tile_count": len(selected),
        "selected_file_bytes": int(sum(int(row["file_bytes"]) for row in selected)),
        "selected_point_count": int(sum(int(row["point_count"]) for row in selected)),
        "selected_distribution": distribution(selected),
        "full_tile_count": len(rows),
        "full_file_bytes": int(sum(int(row["file_bytes"]) for row in rows)),
        "full_point_count": int(sum(int(row["point_count"]) for row in rows)),
        "full_distribution": distribution(rows),
    }
    (reports / "selected_tiles.json").write_text(json.dumps(selection_payload, indent=2), encoding="utf-8")
    print(json.dumps(selection_payload, indent=2))


if __name__ == "__main__":
    main()
