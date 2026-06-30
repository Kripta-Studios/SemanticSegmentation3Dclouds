from __future__ import annotations

import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


def block_features(data: dict, use_tw_input: bool = False) -> torch.Tensor:
    coords = data["coords"].float()
    if use_tw_input:
        if "features_with_tw" in data:
            feats = data["features_with_tw"].float()
        elif "tw_features" in data:
            feats = torch.cat([data.get("features_original", data["features"]).float(), data["tw_features"].float()], dim=1)
        else:
            feats = data.get("features_original", data["features"]).float()
    else:
        feats = data.get("features_original", data["features"]).float()
    return torch.cat([coords, feats], dim=1)


class JepaBlockMasker:
    def __init__(
        self,
        mask_type: str = "spatial_block",
        context_ratio: float = 0.7,
        target_ratio: float = 0.2,
        min_context_points: int = 64,
        min_target_points: int = 32,
        max_retries: int = 20,
        seed: int | None = None,
    ):
        self.mask_type = mask_type
        self.context_ratio = context_ratio
        self.target_ratio = target_ratio
        self.min_context_points = min_context_points
        self.min_target_points = min_target_points
        self.max_retries = max_retries
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def _random_point_mask(self, num_points: int):
        indices = np.random.permutation(num_points)
        n_ctx = max(self.min_context_points, int(num_points * self.context_ratio))
        n_tgt = max(self.min_target_points, int(num_points * self.target_ratio))
        if n_ctx + n_tgt > num_points:
            scale = num_points / max(n_ctx + n_tgt, 1)
            n_ctx = max(1, int(n_ctx * scale))
            n_tgt = max(1, int(n_tgt * scale))
        return indices[:n_ctx], indices[n_ctx : n_ctx + n_tgt]

    def _height_band_mask(self, coords: np.ndarray):
        z = coords[:, 2]
        q0 = np.random.uniform(0.15, 0.65)
        q1 = min(q0 + self.target_ratio, 0.95)
        lo, hi = np.quantile(z, [q0, q1])
        tgt = np.flatnonzero((z >= lo) & (z <= hi))
        ctx = np.flatnonzero((z < lo) | (z > hi))
        if tgt.size < self.min_target_points or ctx.size < self.min_context_points:
            return self._random_point_mask(coords.shape[0])
        np.random.shuffle(ctx)
        return ctx[: max(self.min_context_points, int(coords.shape[0] * self.context_ratio))], tgt

    def _spatial_block_mask(self, coords: np.ndarray):
        num_points = coords.shape[0]
        n_tgt = max(self.min_target_points, int(num_points * self.target_ratio))
        for _ in range(self.max_retries):
            center = coords[np.random.randint(0, num_points)]
            dists = np.linalg.norm(coords[:, :3] - center[:3], axis=1)
            if n_tgt >= num_points - self.min_context_points:
                n_tgt = max(1, num_points - self.min_context_points)
            order = np.argpartition(dists, n_tgt)
            tgt = order[:n_tgt]
            ctx = order[n_tgt:]
            if tgt.size >= self.min_target_points and ctx.size >= self.min_context_points:
                np.random.shuffle(ctx)
                n_ctx = min(ctx.size, max(self.min_context_points, int(num_points * self.context_ratio)))
                return ctx[:n_ctx], tgt
        return self._random_point_mask(num_points)

    def generate_mask(self, coords: np.ndarray):
        if self.mask_type in {"random", "random_point_mask"}:
            return self._random_point_mask(coords.shape[0])
        if self.mask_type in {"spatial", "spatial_block", "spatial_block_mask"}:
            return self._spatial_block_mask(coords)
        if self.mask_type in {"height", "height_band"}:
            return self._height_band_mask(coords)
        raise ValueError(f"Unknown mask_type: {self.mask_type}")


class GeoPointJepaDataset(Dataset):
    def __init__(
        self,
        blocks_dir: str,
        max_blocks: int | None = None,
        masker: JepaBlockMasker | None = None,
        use_tw_input: bool = False,
        return_tw_target: bool = False,
    ):
        self.files = sorted(glob.glob(os.path.join(blocks_dir, "*.pt")))
        if max_blocks is not None and max_blocks > 0:
            self.files = self.files[:max_blocks]
        self.masker = masker or JepaBlockMasker()
        self.use_tw_input = use_tw_input
        self.return_tw_target = return_tw_target

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        file_path = self.files[idx]
        data = torch.load(file_path, weights_only=False, map_location="cpu")
        x = block_features(data, use_tw_input=self.use_tw_input)
        coords = data["coords"].float()
        ctx_idx, tgt_idx = self.masker.generate_mask(coords.numpy())
        out = {
            "x_context": x[ctx_idx],
            "x_target": x[tgt_idx],
            "coords_context": coords[ctx_idx],
            "coords_target": coords[tgt_idx],
            "labels_context": data["labels"][ctx_idx],
            "labels_target": data["labels"][tgt_idx],
            "file_path": file_path,
        }
        if self.return_tw_target:
            if "tw_features" not in data:
                raise KeyError(f"{file_path} does not contain tw_features")
            tw = data["tw_features"].float()
            valid = data.get("tw_valid_mask", tw[:, -1] > 0)
            out["tw_target"] = tw[tgt_idx]
            out["tw_valid_target"] = valid[tgt_idx]
        return out


def _pad(items: list[torch.Tensor], value: float = 0.0, dtype=None) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(item.shape[0] for item in items)
    shape = (len(items), max_len, *items[0].shape[1:])
    out = torch.full(shape, value, dtype=dtype or items[0].dtype)
    mask = torch.zeros((len(items), max_len), dtype=torch.bool)
    for i, item in enumerate(items):
        out[i, : item.shape[0]] = item
        mask[i, : item.shape[0]] = True
    return out, mask


def jepa_collate_fn(batch: list[dict]) -> dict:
    x_ctx, mask_ctx = _pad([item["x_context"] for item in batch])
    x_tgt, mask_tgt = _pad([item["x_target"] for item in batch])
    coords_ctx, _ = _pad([item["coords_context"] for item in batch])
    coords_tgt, _ = _pad([item["coords_target"] for item in batch])
    labels_ctx, _ = _pad([item["labels_context"] for item in batch], value=6, dtype=torch.long)
    labels_tgt, _ = _pad([item["labels_target"] for item in batch], value=6, dtype=torch.long)
    out = {
        "x_context": x_ctx,
        "x_target": x_tgt,
        "coords_context": coords_ctx,
        "coords_target": coords_tgt,
        "mask_context": mask_ctx,
        "mask_target": mask_tgt,
        "labels_context": labels_ctx,
        "labels_target": labels_tgt,
    }
    if "tw_target" in batch[0]:
        tw_tgt, _ = _pad([item["tw_target"] for item in batch])
        tw_valid, _ = _pad([item["tw_valid_target"] for item in batch], value=0, dtype=torch.bool)
        out["tw_target"] = tw_tgt
        out["tw_valid_target"] = tw_valid
    return out

