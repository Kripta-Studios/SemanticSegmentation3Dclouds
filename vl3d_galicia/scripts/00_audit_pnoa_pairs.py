from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import laspy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.classes import ASPRS_TO_PAPER, CLASS_NAMES
from src.data.pnoa import class_count_rows, count_classes_in_laz, find_tile_pairs
from src.utils.progress import eta_line


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PNOA Galicia COL/CIR pairs and raw ASPRS class balance.")
    parser.add_argument("--raw", default="data/raw/pnoa_galicia")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--manifest", default="data/manifests/pnoa_galicia_tiles.csv")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--no-class-scan", action="store_true")
    args = parser.parse_args()

    pairs = find_tile_pairs(args.raw)
    if args.max_files > 0:
        pairs = pairs[: args.max_files]
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    raw_counts = Counter()
    paper_counts = Counter()
    campaign_counts = defaultdict(Counter)
    rows = []
    errors = []
    point_total = 0
    header_total = 0
    pair_mismatches = 0
    start_time = time.perf_counter()
    for i, pair in enumerate(pairs, 1):
        row = {
            "tile_id": pair.tile_id,
            "campaign": pair.campaign,
            "col_path": str(pair.col_path),
            "cir_path": str(pair.cir_path),
            "col_points": 0,
            "cir_points": 0,
            "pair_status": "ok",
            "read_status": "not_scanned" if args.no_class_scan else "ok",
        }
        try:
            with laspy.open(pair.col_path) as col_reader:
                row["col_points"] = int(col_reader.header.point_count)
            with laspy.open(pair.cir_path) as cir_reader:
                row["cir_points"] = int(cir_reader.header.point_count)
            header_total += row["col_points"]
            if row["col_points"] != row["cir_points"]:
                row["pair_status"] = "point_count_mismatch"
                pair_mismatches += 1
            if not args.no_class_scan:
                total, counts = count_classes_in_laz(pair.col_path, chunk_size=args.chunk_size)
                point_total += total
                raw_counts.update(counts)
                campaign_counts[pair.campaign].update(counts)
                for asprs_class, count in counts.items():
                    paper_counts[ASPRS_TO_PAPER.get(asprs_class, 6)] += count
        except Exception as exc:
            row["read_status"] = f"error:{type(exc).__name__}"
            errors.append({"tile_id": pair.tile_id, "file": str(pair.col_path), "error": str(exc)})
        rows.append(row)
        if i % 25 == 0 or i == len(pairs):
            print(f"{eta_line('audit raw pairs', start_time, i, len(pairs))} | errors={len(errors)}")

    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "raw_dir": str(Path(args.raw).resolve()),
        "tile_pairs": len(pairs),
        "complete_pairs": len(pairs),
        "pair_point_count_mismatches": pair_mismatches,
        "header_points_total": header_total,
        "read_points_total": point_total,
        "errors": errors,
        "raw_asprs_distribution": class_count_rows(dict(raw_counts), point_total),
        "paper_class_distribution": [
            {
                "class_id": cls,
                "class_name": CLASS_NAMES[cls],
                "points": int(paper_counts.get(cls, 0)),
                "pct": float(paper_counts.get(cls, 0) / point_total * 100.0) if point_total else 0.0,
            }
            for cls in range(7)
        ],
        "campaign_asprs_distribution": {
            campaign: class_count_rows(dict(counts), sum(counts.values()))
            for campaign, counts in campaign_counts.items()
        },
    }
    (reports / "pnoa_raw_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["paper_class_distribution"], indent=2))
    if errors:
        print(f"Errors detected: {len(errors)}. See {reports / 'pnoa_raw_audit.json'}")


if __name__ == "__main__":
    main()
