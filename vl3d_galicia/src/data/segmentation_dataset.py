from __future__ import annotations

import glob
import os
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.classes import IGNORE_INDEX
from src.data.jepa_dataset import block_features


def parse_class_boost(spec: dict[int, float] | None = None) -> dict[int, float]:
    return spec or {}


def _common_parent(files: list[str]) -> Path | None:
    if not files:
        return None
    try:
        return Path(os.path.commonpath(files))
    except ValueError:
        return None


def label_counts_for_files(files: list[str]) -> list[np.ndarray] | None:
    parent = _common_parent(files)
    if parent is None:
        return None
    cache_path = parent / "class_counts_cache.csv"
    if not cache_path.exists():
        return None
    rows: dict[str, np.ndarray] = {}
    with cache_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row.get("file") or ""
            if not name:
                continue
            rows[name] = np.asarray([float(row.get(f"c{i}", 0.0) or 0.0) for i in range(7)], dtype=np.float64)
    counts = []
    for path in files:
        cached = rows.get(Path(path).name)
        if cached is None:
            return None
        counts.append(cached)
    return counts


def select_block_files(
    files: list[str],
    max_blocks: int | None = None,
    mode: str = "sorted",
    seed: int = 42,
    class_boost: dict[int, float] | None = None,
) -> list[str]:
    if max_blocks is None or max_blocks <= 0 or len(files) <= max_blocks:
        return files
    if mode == "sorted":
        return files[:max_blocks]
    rng = np.random.default_rng(int(seed))
    if mode == "random":
        indices = np.arange(len(files))
        rng.shuffle(indices)
        return [files[int(idx)] for idx in indices[:max_blocks]]
    if mode != "class_balanced":
        raise ValueError(f"Unknown block selection mode: {mode}")
    boost = parse_class_boost(class_boost)
    cached_counts = label_counts_for_files(files)
    if cached_counts is not None:
        per_block_counts = [counts[:IGNORE_INDEX] for counts in cached_counts]
        global_counts = np.sum(np.stack(per_block_counts, axis=0), axis=0)
    else:
        per_block_counts = []
        global_counts = np.zeros(IGNORE_INDEX, dtype=np.float64)
        for path in files:
            labels = torch.load(path, weights_only=False, map_location="cpu")["labels"].long()
            labels = labels[labels != IGNORE_INDEX]
            counts = torch.bincount(labels, minlength=IGNORE_INDEX).double().numpy()[:IGNORE_INDEX]
            per_block_counts.append(counts)
            global_counts += counts
    safe = np.maximum(global_counts, 1.0)
    class_weights = 1.0 / np.sqrt(safe)
    class_weights = class_weights / max(float(np.mean(class_weights)), 1e-12)
    for cls, value in boost.items():
        if 0 <= int(cls) < IGNORE_INDEX:
            class_weights[int(cls)] *= float(value)
    scores = []
    for counts in per_block_counts:
        total = float(np.sum(counts))
        if total <= 0:
            scores.append(1e-6)
            continue
        freq = counts / total
        present = np.count_nonzero(counts > 0)
        rare_presence = np.sum((counts > 0).astype(np.float64) * class_weights)
        scores.append(float(np.dot(freq, class_weights) + 0.05 * present + 0.02 * rare_presence))
    probs = np.asarray(scores, dtype=np.float64)
    probs = np.maximum(probs, 1e-12)
    probs = probs / probs.sum()
    indices = rng.choice(len(files), size=int(max_blocks), replace=False, p=probs)
    return [files[int(idx)] for idx in indices]


class SegmentationBlockDataset(Dataset):
    def __init__(
        self,
        blocks_dir: str,
        max_blocks: int | None = None,
        use_tw_input: bool = False,
        external_feature_dir: str | None = None,
        external_feature_key: str = "dino_features",
        require_external_features: bool = True,
        selection_mode: str = "sorted",
        selection_seed: int = 42,
        selection_class_boost: dict[int, float] | None = None,
        coordinate_normalization: str = "none",
        spectral_normalization: str = "none",
        external_feature_normalization: str = "none",
        spectral_jitter_std: float = 0.0,
        spectral_dropout_prob: float = 0.0,
        external_spectral_jitter_std: float = 0.0,
        external_spectral_dropout_prob: float = 0.0,
    ):
        self.files = sorted(glob.glob(os.path.join(blocks_dir, "*.pt")))
        self.files = select_block_files(
            self.files,
            max_blocks=max_blocks,
            mode=selection_mode,
            seed=selection_seed,
            class_boost=selection_class_boost,
        )
        self.use_tw_input = use_tw_input
        self.blocks_dir = Path(blocks_dir)
        self.external_feature_dir = Path(external_feature_dir) if external_feature_dir else None
        self.external_feature_key = external_feature_key
        self.require_external_features = require_external_features
        self.selection_mode = selection_mode
        self.selection_seed = int(selection_seed)
        if coordinate_normalization not in {"none", "xy_unit_z_robust"}:
            raise ValueError(f"Unknown coordinate_normalization: {coordinate_normalization}")
        if spectral_normalization not in {"none", "block_robust"}:
            raise ValueError(f"Unknown spectral_normalization: {spectral_normalization}")
        if external_feature_normalization not in {"none", "block_robust", "spectral_block_robust"}:
            raise ValueError(f"Unknown external_feature_normalization: {external_feature_normalization}")
        self.coordinate_normalization = coordinate_normalization
        self.spectral_normalization = spectral_normalization
        self.external_feature_normalization = external_feature_normalization
        self.spectral_jitter_std = max(float(spectral_jitter_std), 0.0)
        self.spectral_dropout_prob = min(max(float(spectral_dropout_prob), 0.0), 1.0)
        self.external_spectral_jitter_std = max(float(external_spectral_jitter_std), 0.0)
        self.external_spectral_dropout_prob = min(max(float(external_spectral_dropout_prob), 0.0), 1.0)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        file_path = self.files[idx]
        data = torch.load(file_path, weights_only=False, map_location="cpu")
        x = block_features(data, use_tw_input=self.use_tw_input)
        x = self._normalize_base_features(x)
        x = self._augment_base_spectral(x)
        external = self._load_external_features(file_path)
        if external is not None:
            if external.shape[0] != x.shape[0]:
                raise ValueError(
                    f"External feature point count mismatch for {file_path}: "
                    f"{external.shape[0]} vs {x.shape[0]}"
                )
            x = torch.cat([x, external.float()], dim=1)
        labels = data["labels"].long().clone()
        reliable = data.get("reliable_mask", labels != IGNORE_INDEX)
        labels[~reliable] = IGNORE_INDEX
        return {
            "features": x.float(),
            "labels": labels,
            "reliable_mask": reliable.bool(),
            "file_path": file_path,
        }

    @staticmethod
    def _robust_minmax(values: torch.Tensor, lo_q: float = 0.05, hi_q: float = 0.95) -> torch.Tensor:
        if values.numel() == 0:
            return values
        lo = torch.quantile(values.float(), lo_q, dim=0)
        hi = torch.quantile(values.float(), hi_q, dim=0)
        scale = torch.clamp(hi - lo, min=1e-6)
        return torch.clamp((values.float() - lo) / scale, min=-1.0, max=2.0)

    @staticmethod
    def _robust_center_scale(values: torch.Tensor, lo_q: float = 0.05, hi_q: float = 0.95) -> torch.Tensor:
        if values.numel() == 0:
            return values
        lo = torch.quantile(values.float(), lo_q, dim=0)
        hi = torch.quantile(values.float(), hi_q, dim=0)
        mid = 0.5 * (lo + hi)
        scale = torch.clamp(hi - lo, min=1e-6)
        return torch.clamp((values.float() - mid) / scale, min=-3.0, max=3.0)

    def _normalize_base_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.coordinate_normalization == "none" and self.spectral_normalization == "none":
            return x
        x = x.clone()
        if self.coordinate_normalization == "xy_unit_z_robust" and x.shape[1] >= 3:
            xy_scale = torch.clamp(torch.max(torch.abs(x[:, :2])), min=1.0)
            x[:, :2] = x[:, :2] / xy_scale
            z = x[:, 2:3]
            x[:, 2:3] = self._robust_minmax(z, lo_q=0.05, hi_q=0.95).clamp(-2.0, 4.0)
        if self.spectral_normalization == "block_robust" and x.shape[1] >= 8:
            x[:, 3:8] = self._robust_minmax(x[:, 3:8], lo_q=0.05, hi_q=0.95)
        return x

    def _augment_base_spectral(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < 8 or (self.spectral_jitter_std <= 0.0 and self.spectral_dropout_prob <= 0.0):
            return x
        x = x.clone()
        cols = x[:, 3:8]
        if self.spectral_jitter_std > 0.0:
            scale = 1.0 + torch.randn((1, cols.shape[1]), dtype=cols.dtype) * self.spectral_jitter_std
            shift = torch.randn((1, cols.shape[1]), dtype=cols.dtype) * (0.5 * self.spectral_jitter_std)
            cols = cols * scale + shift
        if self.spectral_dropout_prob > 0.0:
            keep = torch.rand((1, cols.shape[1])) >= self.spectral_dropout_prob
            fill = torch.full_like(cols, 0.5 if self.spectral_normalization == "block_robust" else 0.0)
            cols = torch.where(keep, cols, fill)
        x[:, 3:8] = torch.clamp(cols, min=-1.0, max=2.0)
        return x

    def _normalize_external_features(self, external: torch.Tensor, names: list[str] | None) -> torch.Tensor:
        mode = self.external_feature_normalization
        if mode == "none":
            return external.float()
        external = external.float().clone()
        if mode == "block_robust":
            return self._robust_center_scale(external, lo_q=0.05, hi_q=0.95)
        if names:
            spectral_tokens = ("intensity", "red", "green", "blue", "nir", "ndvi", "ndwi", "brightness", "exg")
            indices = [
                idx
                for idx, name in enumerate(names[: external.shape[1]])
                if any(token in str(name).lower() for token in spectral_tokens)
            ]
            if indices:
                cols = torch.tensor(indices, dtype=torch.long)
                external[:, cols] = self._robust_center_scale(external[:, cols], lo_q=0.05, hi_q=0.95)
                external = self._augment_external_spectral(external, indices)
        return external

    def _augment_external_spectral(self, external: torch.Tensor, indices: list[int]) -> torch.Tensor:
        if not indices or (self.external_spectral_jitter_std <= 0.0 and self.external_spectral_dropout_prob <= 0.0):
            return external
        cols_idx = torch.tensor(indices, dtype=torch.long)
        cols = external[:, cols_idx]
        if self.external_spectral_jitter_std > 0.0:
            scale = 1.0 + torch.randn((1, cols.shape[1]), dtype=cols.dtype) * self.external_spectral_jitter_std
            shift = torch.randn((1, cols.shape[1]), dtype=cols.dtype) * (0.5 * self.external_spectral_jitter_std)
            cols = cols * scale + shift
        if self.external_spectral_dropout_prob > 0.0:
            keep = torch.rand((1, cols.shape[1])) >= self.external_spectral_dropout_prob
            cols = torch.where(keep, cols, torch.zeros_like(cols))
        external[:, cols_idx] = torch.clamp(cols, min=-3.0, max=3.0)
        return external

    def _load_external_features(self, file_path: str) -> torch.Tensor | None:
        if self.external_feature_dir is None:
            return None
        source = Path(file_path)
        candidates = [
            self.external_feature_dir / self.blocks_dir.name / source.name,
            self.external_feature_dir / source.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                payload = torch.load(candidate, weights_only=False, map_location="cpu")
                if self.external_feature_key not in payload:
                    raise KeyError(f"{candidate} does not contain '{self.external_feature_key}'")
                names = payload.get("feature_names")
                return self._normalize_external_features(payload[self.external_feature_key].float(), names)
        if self.require_external_features:
            raise FileNotFoundError(
                f"External features for {source.name} not found under {self.external_feature_dir}"
            )
        return None


def segmentation_collate_fn(batch: list[dict]) -> dict:
    batch_size = len(batch)
    max_len = max(item["features"].shape[0] for item in batch)
    feat_dim = batch[0]["features"].shape[1]
    features = torch.zeros(batch_size, max_len, feat_dim)
    labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for i, item in enumerate(batch):
        n_points = item["features"].shape[0]
        features[i, :n_points] = item["features"]
        labels[i, :n_points] = item["labels"]
        mask[i, :n_points] = True
    return {"features": features, "labels": labels, "mask": mask}
