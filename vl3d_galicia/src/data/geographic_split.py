from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable


SPLIT_MANIFEST_SCHEMA_VERSION = "galicia-geographic-split-v1"
OFFICIAL_SPLITS = ("train", "val", "test")
TILE_GRID_RE = re.compile(r"_(?P<x>\d+)-(?P<y>\d+)_ORT")


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tile_grid_xy(tile_id: str) -> tuple[int, int]:
    match = TILE_GRID_RE.search(tile_id)
    if match is None:
        raise ValueError(f"Cannot parse PNOA grid coordinates from tile_id={tile_id!r}")
    return int(match.group("x")), int(match.group("y"))


def galicia_campaign_north_val_split(
    tile_id: str,
    campaign: str,
    *,
    val_y_min: int = 4804,
    buffer_y_min: int = 4800,
) -> str:
    """Label-blind Galicia split with complete campaigns/contiguous regions.

    GAL-E-2016 is the external geographic test campaign.  GAL-W-2015 tiles in
    the northern cluster form validation.  Any intervening grid row is an
    explicit geographic buffer and is never used for fitting or scoring.
    """

    if campaign == "GAL-E-2016":
        return "test"
    if campaign != "GAL-W-2015":
        return "excluded_unknown_campaign"
    _, y = tile_grid_xy(tile_id)
    if y >= val_y_min:
        return "val"
    if y >= buffer_y_min:
        return "excluded_buffer"
    return "train"


def projected_epsg_from_header(header) -> int | None:
    """Read ProjectedCSTypeGeoKey (3072) without requiring pyproj."""

    for vlr in getattr(header, "vlrs", []):
        for key in getattr(vlr, "geo_keys", []):
            if int(getattr(key, "id", -1)) == 3072 and int(getattr(key, "tiff_tag_location", -1)) == 0:
                value = int(getattr(key, "value_offset", 0))
                return value if value > 0 else None
    return None


def bbox_distance_m(a: Iterable[float], b: Iterable[float]) -> float:
    ax0, ay0, ax1, ay1 = (float(value) for value in a)
    bx0, by0, bx1, by1 = (float(value) for value in b)
    dx = max(ax0 - bx1, bx0 - ax1, 0.0)
    dy = max(ay0 - by1, by0 - ay1, 0.0)
    return float(math.hypot(dx, dy))


def bbox_overlap_area_m2(a: Iterable[float], b: Iterable[float]) -> float:
    ax0, ay0, ax1, ay1 = (float(value) for value in a)
    bx0, by0, bx1, by1 = (float(value) for value in b)
    width = max(min(ax1, bx1) - max(ax0, bx0), 0.0)
    height = max(min(ay1, by1) - max(ay0, by0), 0.0)
    return float(width * height)


def split_hash_payload(manifest: dict) -> dict:
    rows = []
    for row in sorted(manifest.get("tiles", []), key=lambda item: item["tile_id"]):
        rows.append(
            {
                key: row.get(key)
                for key in (
                    "tile_id",
                    "campaign",
                    "split",
                    "col_path",
                    "cir_path",
                    "col_sha256",
                    "cir_sha256",
                    "bounds",
                    "crs",
                    "col_points",
                    "cir_points",
                )
            }
        )
    return {
        "schema_version": manifest.get("schema_version"),
        "policy": manifest.get("policy"),
        "seed": manifest.get("seed"),
        "tiles": rows,
    }


def compute_split_hash(manifest: dict) -> str:
    return canonical_sha256(split_hash_payload(manifest))


def validate_split_manifest(manifest: dict, *, overlap_tolerance_m2: float = 1.0) -> dict:
    if manifest.get("schema_version") != SPLIT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unknown split manifest schema {manifest.get('schema_version')!r}; "
            f"expected {SPLIT_MANIFEST_SCHEMA_VERSION!r}"
        )
    rows = list(manifest.get("tiles", []))
    if not rows:
        raise ValueError("Split manifest contains no tiles")
    tile_ids = [str(row["tile_id"]) for row in rows]
    if len(tile_ids) != len(set(tile_ids)):
        raise ValueError("Split manifest contains duplicate tile_id values")

    by_split = {split: [row for row in rows if row.get("split") == split] for split in OFFICIAL_SPLITS}
    source_sets: dict[str, set[str]] = {}
    tile_sets: dict[str, set[str]] = {}
    for split, split_rows in by_split.items():
        tile_sets[split] = {str(row["tile_id"]) for row in split_rows}
        source_sets[split] = {
            str(row[key])
            for row in split_rows
            for key in ("col_path", "cir_path")
        }
        if not split_rows:
            raise ValueError(f"Split {split!r} contains no tiles")

    intersections: dict[str, list[str]] = {}
    overlap_pairs: list[dict] = []
    min_distances: dict[str, float] = {}
    for index, split_a in enumerate(OFFICIAL_SPLITS):
        for split_b in OFFICIAL_SPLITS[index + 1 :]:
            key = f"{split_a}_{split_b}"
            tile_intersection = sorted(tile_sets[split_a] & tile_sets[split_b])
            source_intersection = sorted(source_sets[split_a] & source_sets[split_b])
            intersections[f"{key}_tile_ids"] = tile_intersection
            intersections[f"{key}_source_laz"] = source_intersection
            if tile_intersection or source_intersection:
                raise ValueError(f"Split leakage detected for {key}")
            best = math.inf
            for row_a in by_split[split_a]:
                for row_b in by_split[split_b]:
                    area = bbox_overlap_area_m2(row_a["bounds"], row_b["bounds"])
                    if area > overlap_tolerance_m2:
                        overlap_pairs.append(
                            {
                                "split_a": split_a,
                                "tile_a": row_a["tile_id"],
                                "split_b": split_b,
                                "tile_b": row_b["tile_id"],
                                "overlap_area_m2": area,
                            }
                        )
                    best = min(best, bbox_distance_m(row_a["bounds"], row_b["bounds"]))
            min_distances[f"{key}_min_distance_m"] = float(best)
    if overlap_pairs:
        raise ValueError(f"Cross-split tile bounds overlap: {overlap_pairs[:3]}")

    expected_hash = compute_split_hash(manifest)
    actual_hash = manifest.get("split_hash")
    if actual_hash is not None and actual_hash != expected_hash:
        raise ValueError(f"Split hash mismatch: manifest={actual_hash}, computed={expected_hash}")
    return {
        "tile_counts": {split: len(by_split[split]) for split in OFFICIAL_SPLITS},
        "tile_intersections": intersections,
        "cross_split_bounds_overlap_count": 0,
        "min_distances": min_distances,
        "split_hash": expected_hash,
    }


def load_split_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    audit = validate_split_manifest(manifest)
    manifest["split_hash"] = audit["split_hash"]
    return manifest

