from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import laspy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.geographic_split import (
    OFFICIAL_SPLITS,
    SPLIT_MANIFEST_SCHEMA_VERSION,
    bbox_distance_m,
    compute_split_hash,
    file_sha256,
    galicia_campaign_north_val_split,
    projected_epsg_from_header,
    tile_grid_xy,
    validate_split_manifest,
)
from src.data.pnoa import PNOA_FEATURE_SCHEMA, find_tile_pairs


POLICY_NAME = "galicia-campaign-test-north-cluster-val-v1"


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def probe_laz(path: Path, points: int = 1024) -> tuple[bool, str | None]:
    try:
        with laspy.open(path) as reader:
            next(reader.chunk_iterator(points), None)
    except Exception as exc:  # source audit must preserve the concrete codec error
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def build_row(pair, repo_root: Path, val_y_min: int, buffer_y_min: int, hash_sources: bool) -> dict:
    with laspy.open(pair.col_path) as col_reader:
        col_header = col_reader.header
        bounds = [
            float(col_header.mins[0]),
            float(col_header.mins[1]),
            float(col_header.maxs[0]),
            float(col_header.maxs[1]),
        ]
        col_points = int(col_header.point_count)
        epsg = projected_epsg_from_header(col_header)
    with laspy.open(pair.cir_path) as cir_reader:
        cir_points = int(cir_reader.header.point_count)
    col_ok, col_error = probe_laz(pair.col_path)
    cir_ok, cir_error = probe_laz(pair.cir_path)
    split = galicia_campaign_north_val_split(
        pair.tile_id,
        pair.campaign,
        val_y_min=val_y_min,
        buffer_y_min=buffer_y_min,
    )
    if not col_ok or not cir_ok:
        split = "excluded_source_error"
    grid_x, grid_y = tile_grid_xy(pair.tile_id)
    return {
        "tile_id": pair.tile_id,
        "campaign": pair.campaign,
        "grid_x_km": grid_x,
        "grid_y_km": grid_y,
        "col_path": relative_path(pair.col_path, repo_root),
        "cir_path": relative_path(pair.cir_path, repo_root),
        "col_size_bytes": pair.col_path.stat().st_size,
        "cir_size_bytes": pair.cir_path.stat().st_size,
        "col_sha256": file_sha256(pair.col_path) if hash_sources else None,
        "cir_sha256": file_sha256(pair.cir_path) if hash_sources else None,
        "col_points": col_points,
        "cir_points": cir_points,
        "point_count_match": col_points == cir_points,
        "bounds": bounds,
        "crs": f"EPSG:{epsg}" if epsg is not None else "unknown",
        "split": split,
        "source_readable": bool(col_ok and cir_ok),
        "source_read_error": " | ".join(item for item in (col_error, cir_error) if item) or None,
    }


def add_isolation(rows: list[dict]) -> None:
    official = [row for row in rows if row["split"] in OFFICIAL_SPLITS]
    for row in rows:
        for split in OFFICIAL_SPLITS:
            candidates = [other for other in official if other["split"] == split and other is not row]
            key = f"min_distance_to_{split}_m"
            row[key] = min((bbox_distance_m(row["bounds"], other["bounds"]) for other in candidates), default=None)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened = []
    for row in rows:
        item = dict(row)
        item["bounds"] = json.dumps(item["bounds"], separators=(",", ":"))
        flattened.append(item)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and audit the frozen label-blind Galicia geographic split.")
    parser.add_argument("--raw", default="data/raw/pnoa_galicia")
    parser.add_argument("--out-json", default="protocols/galicia_geographic_split_v1.json")
    parser.add_argument("--out-csv", default="protocols/galicia_geographic_split_v1.csv")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--val-y-min", type=int, default=4804)
    parser.add_argument("--buffer-y-min", type=int, default=4800)
    parser.add_argument("--no-source-hashes", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    pairs = find_tile_pairs(args.raw)
    if not pairs:
        raise SystemExit(f"No COL/CIR pairs found under {args.raw}")
    rows = []
    started = time.time()
    for index, pair in enumerate(pairs, 1):
        rows.append(
            build_row(
                pair,
                repo_root,
                val_y_min=args.val_y_min,
                buffer_y_min=args.buffer_y_min,
                hash_sources=not args.no_source_hashes,
            )
        )
        if index % 25 == 0 or index == len(pairs):
            elapsed = max(time.time() - started, 1e-6)
            print(f"split manifest {index}/{len(pairs)} ({index / elapsed:.2f} tiles/s)")
    add_isolation(rows)
    manifest = {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "policy": POLICY_NAME,
        "policy_parameters": {
            "test_campaign": "GAL-E-2016",
            "validation_campaign": "GAL-W-2015",
            "val_y_min": args.val_y_min,
            "buffer_y_min": args.buffer_y_min,
            "label_blind_membership": True,
        },
        "seed": args.seed,
        "feature_schema": PNOA_FEATURE_SCHEMA.as_dict(),
        "git_commit": git_head(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_root": relative_path(Path(args.raw), repo_root),
        "tiles": rows,
    }
    manifest["split_hash"] = compute_split_hash(manifest)
    manifest["audit"] = validate_split_manifest(manifest)
    manifest["excluded_tiles"] = {
        status: sum(row["split"] == status for row in rows)
        for status in sorted({row["split"] for row in rows if row["split"] not in OFFICIAL_SPLITS})
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_csv(Path(args.out_csv), rows)
    digest = file_sha256(out_json)
    out_json.with_suffix(out_json.suffix + ".sha256").write_text(f"{digest}  {out_json.name}\n", encoding="utf-8")
    print(json.dumps({"split_hash": manifest["split_hash"], "manifest_sha256": digest, **manifest["audit"]}, indent=2))


if __name__ == "__main__":
    main()

