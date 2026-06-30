from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import laspy
import numpy as np

from src.data.classes import ASPRS_NAMES, map_labels


@dataclass(frozen=True)
class TilePair:
    tile_id: str
    col_path: Path
    cir_path: Path
    campaign: str


def tile_id_from_col(path: Path) -> str:
    return path.stem


def campaign_from_name(name: str) -> str:
    normalized = name.replace("_", "-")
    match = re.match(r"PNOA-(\d{4})-([A-Z]+(?:-[A-Z])?)", normalized)
    if not match:
        return "unknown"
    year, region = match.groups()
    return f"{region}-{year}"


def find_tile_pairs(raw_dir: str | Path) -> list[TilePair]:
    raw = Path(raw_dir)
    pairs: list[TilePair] = []
    for col in sorted(raw.rglob("*COL.laz")):
        cir = Path(str(col).replace("-COL.laz", "-CIR.laz"))
        if not cir.exists():
            continue
        pairs.append(TilePair(tile_id_from_col(col), col, cir, campaign_from_name(col.name)))
    return pairs


def point_format_dimensions(las) -> set[str]:
    return {dim.name for dim in las.point_format.dimensions}


def normalize_intensity(intensity: np.ndarray) -> np.ndarray:
    values = intensity.astype(np.float32, copy=False)
    p99 = np.percentile(values, 99) if values.size else 0.0
    if p99 <= 0:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / p99, 0.0, 1.0).astype(np.float32)


def _hashed_xyz(coords: np.ndarray, quantization_mm: float = 1.0) -> np.ndarray:
    scale = 1000.0 / quantization_mm
    q = np.rint(coords.astype(np.float64, copy=False) * scale).astype(np.int64, copy=False)
    return (q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791)


def align_nir_by_xyz(
    col_coords: np.ndarray,
    cir_coords: np.ndarray,
    cir_red: np.ndarray,
    quantization_mm: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    col_key = _hashed_xyz(col_coords, quantization_mm)
    cir_key = _hashed_xyz(cir_coords, quantization_mm)
    order = np.argsort(cir_key, kind="mergesort")
    sorted_key = cir_key[order]
    pos = np.searchsorted(sorted_key, col_key)
    in_bounds = pos < sorted_key.size
    matched = np.zeros(col_key.shape[0], dtype=bool)
    matched[in_bounds] = sorted_key[pos[in_bounds]] == col_key[in_bounds]
    nir = np.zeros(col_key.shape[0], dtype=np.float32)
    if matched.any():
        nir[matched] = cir_red[order[pos[matched]]].astype(np.float32, copy=False)
    return nir, ~matched


def read_col_cir_pair(
    col_path: str | Path,
    cir_path: str | Path,
    quantization_mm: float = 1.0,
) -> dict:
    col_path = Path(col_path)
    cir_path = Path(cir_path)
    with laspy.open(col_path) as f_col:
        las_col = f_col.read()
    dims_col = point_format_dimensions(las_col)
    coords = np.column_stack([las_col.x, las_col.y, las_col.z]).astype(np.float64, copy=False)
    n_points = coords.shape[0]

    classification = (
        np.asarray(las_col.classification, dtype=np.uint8)
        if "classification" in dims_col
        else np.zeros(n_points, dtype=np.uint8)
    )
    labels = map_labels(classification)

    intensity = (
        normalize_intensity(np.asarray(las_col.intensity))
        if "intensity" in dims_col
        else np.zeros(n_points, dtype=np.float32)
    )
    if {"red", "green", "blue"}.issubset(dims_col):
        rgb = np.column_stack([las_col.red, las_col.green, las_col.blue]).astype(np.float32) / 65535.0
    else:
        rgb = np.zeros((n_points, 3), dtype=np.float32)

    with laspy.open(cir_path) as f_cir:
        las_cir = f_cir.read()
    cir_coords = np.column_stack([las_cir.x, las_cir.y, las_cir.z]).astype(np.float64, copy=False)
    cir_red = np.asarray(las_cir.red, dtype=np.float32)
    if cir_coords.shape[0] == n_points and np.allclose(coords, cir_coords, atol=quantization_mm / 1000.0):
        nir = cir_red
        missing_nir = np.zeros(n_points, dtype=bool)
        nir_join = "direct"
    else:
        nir, missing_nir = align_nir_by_xyz(coords, cir_coords, cir_red, quantization_mm=quantization_mm)
        nir_join = "spatial_hash"

    features = np.column_stack([intensity, rgb, nir / 65535.0]).astype(np.float32)
    return {
        "coords": coords.astype(np.float32),
        "features": features,
        "labels": labels,
        "classification": classification,
        "missing_nir_mask": missing_nir,
        "tile_id": tile_id_from_col(col_path),
        "campaign": campaign_from_name(col_path.name),
        "nir_join": nir_join,
        "col_points": int(n_points),
        "cir_points": int(cir_coords.shape[0]),
    }


def count_classes_in_laz(col_path: str | Path, chunk_size: int = 1_000_000) -> tuple[int, dict[int, int]]:
    counts: dict[int, int] = {}
    with laspy.open(col_path) as reader:
        total = int(reader.header.point_count)
        dims = {dim.name for dim in reader.header.point_format.dimensions}
        if "classification" not in dims:
            return total, {-1: total}
        for chunk in reader.chunk_iterator(chunk_size):
            values, freqs = np.unique(np.asarray(chunk.classification, dtype=np.uint8), return_counts=True)
            for value, freq in zip(values, freqs):
                counts[int(value)] = counts.get(int(value), 0) + int(freq)
    return total, counts


def class_count_rows(counts: dict[int, int], total: int) -> list[dict]:
    rows = []
    for cls, count in sorted(counts.items()):
        rows.append(
            {
                "class_id": cls,
                "class_name": ASPRS_NAMES.get(cls, f"asprs_{cls}") if cls >= 0 else "missing_classification",
                "points": int(count),
                "pct": float(count / total * 100.0) if total else 0.0,
            }
        )
    return rows
