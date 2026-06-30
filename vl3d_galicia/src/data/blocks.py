from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.data.classes import IGNORE_INDEX
from src.data.pnoa import read_col_cir_pair


def torch_save_atomic(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def stable_unit_interval(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(16**12)


def assign_split(
    tile_id: str,
    campaign: str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    split_mode: str = "mixed",
) -> str:
    if split_mode == "campaign" and campaign.startswith("GAL-E"):
        return "test"
    score = stable_unit_interval(tile_id)
    if split_mode in {"mixed", "hash"} and score < test_ratio:
        return "test"
    val_threshold = test_ratio + val_ratio if split_mode in {"mixed", "hash"} else val_ratio
    return "val" if score < val_threshold else "train"


def iter_block_masks(coords: np.ndarray, tile_size: float, stride: float):
    x = coords[:, 0]
    y = coords[:, 1]
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    for x0 in np.arange(x_min, x_max, stride):
        x1 = x0 + tile_size
        x_mask = (x >= x0) & (x < x1)
        if not x_mask.any():
            continue
        for y0 in np.arange(y_min, y_max, stride):
            y1 = y0 + tile_size
            mask = x_mask & (y >= y0) & (y < y1)
            if mask.any():
                yield float(x0), float(y0), float(x1), float(y1), mask


def sample_block_indices(labels: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    n_points = labels.shape[0]
    if n_points <= max_points:
        return np.arange(n_points)
    reliable = np.flatnonzero(labels != IGNORE_INDEX)
    if reliable.size == 0:
        return rng.choice(n_points, max_points, replace=False)
    per_class: list[np.ndarray] = []
    budget = max_points // 2
    classes = [c for c in range(IGNORE_INDEX) if np.any(labels == c)]
    if classes:
        per_cls = max(1, budget // len(classes))
        for cls in classes:
            cls_idx = np.flatnonzero(labels == cls)
            take = min(per_cls, cls_idx.size)
            per_class.append(rng.choice(cls_idx, take, replace=False))
    chosen = np.concatenate(per_class) if per_class else np.array([], dtype=np.int64)
    if chosen.size < max_points:
        remaining_mask = np.ones(n_points, dtype=bool)
        remaining_mask[chosen] = False
        remaining = np.flatnonzero(remaining_mask)
        fill = rng.choice(remaining, max_points - chosen.size, replace=False)
        chosen = np.concatenate([chosen, fill])
    rng.shuffle(chosen)
    return chosen[:max_points]


def make_blocks_from_pair(
    col_path: str | Path,
    cir_path: str | Path,
    out_root: str | Path,
    tile_size: float,
    stride: float,
    points_per_block: int,
    min_points: int,
    val_ratio: float,
    test_ratio: float,
    split_mode: str,
    seed: int,
    skip_existing: bool = True,
    force_split: str | None = None,
) -> dict:
    tile = read_col_cir_pair(col_path, cir_path)
    split = force_split or assign_split(tile["tile_id"], tile["campaign"], val_ratio=val_ratio, test_ratio=test_ratio, split_mode=split_mode)
    out_dir = Path(out_root) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + int(stable_unit_interval(tile["tile_id"]) * 1_000_000))
    stats = {
        "tile_id": tile["tile_id"],
        "campaign": tile["campaign"],
        "split": split,
        "blocks": 0,
        "written_blocks": 0,
        "skipped_blocks": 0,
        "points_written": 0,
        "class_counts": {str(i): 0 for i in range(7)},
        "missing_nir_points": int(tile["missing_nir_mask"].sum()),
        "nir_join": tile["nir_join"],
        "col_points": tile["col_points"],
        "cir_points": tile["cir_points"],
    }
    coords = tile["coords"]
    labels = tile["labels"]
    features = tile["features"]
    missing_nir = tile["missing_nir_mask"]
    for block_idx, (x0, y0, x1, y1, mask) in enumerate(iter_block_masks(coords, tile_size, stride)):
        point_count = int(mask.sum())
        if point_count < min_points:
            continue
        idx = np.flatnonzero(mask)
        sampled = idx[sample_block_indices(labels[idx], points_per_block, rng)]
        block_coords = coords[sampled].astype(np.float32, copy=True)
        local = block_coords.copy()
        local[:, 0] -= (x0 + x1) * 0.5
        local[:, 1] -= (y0 + y1) * 0.5
        local[:, 2] -= float(np.min(local[:, 2]))
        block_labels = labels[sampled]
        out_path = out_dir / f"{tile['tile_id']}_block_{block_idx:05d}.pt"
        if skip_existing and out_path.exists():
            try:
                existing = torch.load(out_path, weights_only=False, map_location="cpu")
                existing_labels = existing["labels"].numpy()
                counts = np.bincount(existing_labels, minlength=7)
                for cls, count in enumerate(counts):
                    stats["class_counts"][str(cls)] += int(count)
                stats["blocks"] += 1
                stats["skipped_blocks"] += 1
                stats["points_written"] += int(existing_labels.size)
            except Exception:
                out_path.unlink(missing_ok=True)
            else:
                continue
        counts = np.bincount(block_labels, minlength=7)
        for cls, count in enumerate(counts):
            stats["class_counts"][str(cls)] += int(count)
        stats["blocks"] += 1
        stats["written_blocks"] += 1
        stats["points_written"] += int(sampled.size)
        torch_save_atomic(
            {
                "coords": torch.from_numpy(local),
                "global_coords": torch.from_numpy(block_coords),
                "features": torch.from_numpy(features[sampled].astype(np.float32, copy=False)),
                "labels": torch.from_numpy(block_labels.astype(np.int64, copy=False)),
                "reliable_mask": torch.from_numpy(block_labels != IGNORE_INDEX),
                "missing_nir_mask": torch.from_numpy(missing_nir[sampled]),
                "tile_id": tile["tile_id"],
                "campaign": tile["campaign"],
                "split": split,
                "bbox": [x0, y0, x1, y1],
            },
            out_path,
        )
    return stats


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
