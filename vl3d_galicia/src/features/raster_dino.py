from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
SAT493M_MEAN = torch.tensor([0.430, 0.411, 0.296], dtype=torch.float32).view(1, 3, 1, 1)
SAT493M_STD = torch.tensor([0.213, 0.156, 0.143], dtype=torch.float32).view(1, 3, 1, 1)


@dataclass(frozen=True)
class RasterizedBlock:
    image: torch.Tensor
    raster: torch.Tensor
    point_cells: torch.Tensor
    channel_names: list[str]


def _as_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.as_tensor(value, dtype=torch.float32)


def _normalize_01(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    if values.numel() == 0:
        return values
    finite = torch.isfinite(values)
    if not finite.any():
        return torch.zeros_like(values)
    valid = values[finite]
    lo = torch.quantile(valid, 0.01)
    hi = torch.quantile(valid, 0.99)
    if float((hi - lo).abs()) < 1e-8:
        return torch.zeros_like(values)
    return ((values - lo) / (hi - lo)).clamp(0.0, 1.0)


def _safe_feature_column(features: torch.Tensor, idx: int, default: torch.Tensor) -> torch.Tensor:
    if features.ndim == 2 and features.shape[1] > idx:
        return features[:, idx].float()
    return default.float()


def _cell_indices(coords: torch.Tensor, grid_size: int) -> torch.Tensor:
    xy = coords[:, :2].float()
    mins = xy.min(dim=0).values
    maxs = xy.max(dim=0).values
    span = (maxs - mins).clamp_min(1e-6)
    norm = (xy - mins) / span
    x = torch.clamp((norm[:, 0] * (grid_size - 1)).long(), 0, grid_size - 1)
    y = torch.clamp((norm[:, 1] * (grid_size - 1)).long(), 0, grid_size - 1)
    return torch.stack([y, x], dim=1)


def aggregate_points_to_raster(values: torch.Tensor, cells: torch.Tensor, grid_size: int) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError(f"Expected values as [N, C], got {tuple(values.shape)}")
    n_points, channels = values.shape
    raster = torch.zeros(channels, grid_size, grid_size, dtype=torch.float32)
    counts = torch.zeros(1, grid_size, grid_size, dtype=torch.float32)
    if n_points == 0:
        return torch.cat([raster, counts], dim=0)
    flat = cells[:, 0].clamp(0, grid_size - 1) * grid_size + cells[:, 1].clamp(0, grid_size - 1)
    raster_flat = raster.view(channels, -1)
    counts_flat = counts.view(1, -1)
    raster_flat.index_add_(1, flat, values.t().contiguous())
    ones = torch.ones(1, n_points, dtype=torch.float32)
    counts_flat.index_add_(1, flat, ones)
    raster = raster / counts.clamp_min(1.0)
    density = _normalize_01(torch.log1p(counts))
    return torch.cat([raster, density], dim=0)


def make_multichannel_raster(
    block: dict[str, Any],
    grid_size: int = 128,
    tw_channels: int = 8,
) -> RasterizedBlock:
    coords = _as_float_tensor(block["coords"])
    base = _as_float_tensor(block.get("features_original", block["features"]))
    z = coords[:, 2].float()
    z_norm = _normalize_01(z)
    intensity_default = torch.zeros_like(z_norm)
    nir_default = _safe_feature_column(base, 1, intensity_default)

    columns = [
        _safe_feature_column(base, 0, intensity_default),
        _safe_feature_column(base, 1, intensity_default),
        _safe_feature_column(base, 2, intensity_default),
        _safe_feature_column(base, 3, intensity_default),
        _safe_feature_column(base, 4, nir_default),
        z_norm,
    ]
    names = ["red", "green", "blue", "intensity", "nir", "z_norm"]

    if "tw_features" in block and tw_channels > 0:
        tw = _as_float_tensor(block["tw_features"])
        keep = min(int(tw_channels), int(tw.shape[1]))
        for idx in range(keep):
            columns.append(_normalize_01(tw[:, idx]))
            names.append(f"tw_{idx:02d}")

    values = torch.stack(columns, dim=1)
    cells = _cell_indices(coords, int(grid_size))
    raster = aggregate_points_to_raster(values, cells, int(grid_size))
    names = [*names, "density"]
    image = raster_to_image(raster, names, mode="rgb_nir_height")
    return RasterizedBlock(image=image, raster=raster, point_cells=cells, channel_names=names)


def raster_to_image(raster: torch.Tensor, channel_names: list[str], mode: str = "rgb_nir_height") -> torch.Tensor:
    by_name = {name: idx for idx, name in enumerate(channel_names)}

    def channel(name: str, fallback: str = "density") -> torch.Tensor:
        idx = by_name.get(name, by_name.get(fallback, 0))
        return raster[idx]

    if mode == "rgb":
        channels = [channel("red"), channel("green"), channel("blue")]
    elif mode == "cir":
        channels = [channel("nir"), channel("red"), channel("green")]
    elif mode == "height":
        channels = [channel("z_norm"), channel("density"), channel("intensity")]
    elif mode == "nir_height_density":
        channels = [channel("nir"), channel("z_norm"), channel("density")]
    elif mode == "rgb_nir_height":
        channels = [channel("red"), channel("nir"), channel("z_norm")]
    else:
        raise ValueError(f"Unknown raster image mode: {mode}")
    img = torch.stack([_normalize_01(c) for c in channels], dim=0)
    return img.clamp(0.0, 1.0)


def sample_raster_at_points(raster: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    y = cells[:, 0].clamp(0, raster.shape[1] - 1)
    x = cells[:, 1].clamp(0, raster.shape[2] - 1)
    return raster[:, y, x].t().contiguous()


def stat_dense_features(raster: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    base = raster.unsqueeze(0)
    pooled3 = F.avg_pool2d(base, kernel_size=3, stride=1, padding=1).squeeze(0)
    pooled7 = F.avg_pool2d(base, kernel_size=7, stride=1, padding=3).squeeze(0)
    local = sample_raster_at_points(raster, cells)
    neigh3 = sample_raster_at_points(pooled3, cells)
    neigh7 = sample_raster_at_points(pooled7, cells)
    return torch.cat([local, neigh3, neigh7], dim=1)


def deterministic_projection(features: torch.Tensor, out_dim: int, seed: int = 13) -> torch.Tensor:
    if out_dim <= 0 or features.shape[1] == out_dim:
        return features.float()
    if features.shape[1] < out_dim:
        pad = torch.zeros(features.shape[0], out_dim - features.shape[1], dtype=features.dtype)
        return torch.cat([features, pad], dim=1).float()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    proj = torch.randn(features.shape[1], out_dim, generator=generator, dtype=torch.float32)
    proj = F.normalize(proj, dim=0)
    return (features.float() @ proj).float()


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (features - mean) / std


class DinoDenseExtractor:
    def __init__(
        self,
        backend: str = "stat",
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        repo_dir: str | None = None,
        weights: str | None = None,
        device: str = "cuda",
        normalize: str = "imagenet",
    ):
        self.backend = backend
        self.model_name = model_name
        self.repo_dir = repo_dir
        self.weights = weights
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.normalize = normalize
        self.model = None
        self.processor = None
        self.real_backend = "stat"
        if backend != "stat":
            self._load_model()

    @property
    def uses_real_dino(self) -> bool:
        return self.real_backend != "stat"

    def _load_model(self) -> None:
        errors = []
        for candidate in ([self.backend] if self.backend != "auto" else ["hf", "timm", "dinov2", "torchhub"]):
            try:
                if candidate == "dinov2":
                    name = self.model_name if self.model_name.startswith("dinov2_") else "dinov2_vits14"
                    self.model = torch.hub.load("facebookresearch/dinov2", name).to(self.device).eval()
                    self.real_backend = "dinov2"
                    self.model_name = name
                    return
                if candidate == "hf":
                    from transformers import AutoModel

                    self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
                    self.real_backend = "hf"
                    return
                if candidate == "timm":
                    import timm

                    self.model = timm.create_model(self.model_name, pretrained=True, num_classes=0).to(self.device).eval()
                    self.real_backend = "timm"
                    return
                if candidate == "torchhub":
                    if not self.repo_dir:
                        raise ValueError("--repo-dir is required for torchhub DINOv3 loading")
                    kwargs = {}
                    if self.weights:
                        kwargs["weights"] = self.weights
                    self.model = torch.hub.load(str(Path(self.repo_dir)), self.model_name, source="local", **kwargs).to(self.device).eval()
                    self.real_backend = "torchhub"
                    return
            except Exception as exc:  # noqa: BLE001 - preserve backend fallback diagnostics
                errors.append(f"{candidate}: {exc}")
        raise RuntimeError("Could not load a DINO backend. Tried: " + " | ".join(errors))

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        batch = image.unsqueeze(0).float().to(self.device)
        if self.normalize == "sat493m" or "sat493m" in self.model_name.lower():
            mean, std = SAT493M_MEAN.to(self.device), SAT493M_STD.to(self.device)
        else:
            mean, std = IMAGENET_MEAN.to(self.device), IMAGENET_STD.to(self.device)
        return (batch - mean) / std

    @staticmethod
    def _first_tensor(mapping: dict, keys: tuple[str, ...]) -> torch.Tensor | None:
        for key in keys:
            value = mapping.get(key)
            if isinstance(value, torch.Tensor):
                return value
        return None

    @torch.inference_mode()
    def image_feature_map(self, image: torch.Tensor) -> torch.Tensor:
        if self.real_backend == "stat":
            raise RuntimeError("Stat backend does not expose DINO image features")
        pixels = self._normalize_image(image)
        if self.real_backend == "hf":
            outputs = self.model(pixel_values=pixels, output_hidden_states=False)
            tokens = outputs.last_hidden_state
            if tokens.shape[1] > 1:
                tokens = tokens[:, 1:, :]
            patch = int(getattr(self.model.config, "patch_size", 16) or 16)
            h = max(int(image.shape[1]) // patch, 1)
            w = max(int(image.shape[2]) // patch, 1)
            if h * w != tokens.shape[1]:
                side = int(tokens.shape[1] ** 0.5)
                h = max(side, 1)
                w = max(tokens.shape[1] // h, 1)
            return tokens[:, : h * w, :].reshape(1, h, w, -1).permute(0, 3, 1, 2).squeeze(0).cpu()
        if self.real_backend in {"torchhub", "dinov2"}:
            if hasattr(self.model, "forward_features"):
                out = self.model.forward_features(pixels)
                if isinstance(out, dict):
                    tokens = self._first_tensor(out, ("x_norm_patchtokens", "patchtokens", "tokens"))
                    if tokens is not None:
                        h = w = int(tokens.shape[1] ** 0.5)
                        return tokens[:, : h * w, :].reshape(1, h, w, -1).permute(0, 3, 1, 2).squeeze(0).cpu()
            out = self.model(pixels)
            if isinstance(out, torch.Tensor) and out.ndim == 4:
                return out.squeeze(0).cpu()
            if isinstance(out, torch.Tensor) and out.ndim == 3:
                tokens = out[:, 1:, :] if out.shape[1] > 1 else out
                h = w = int(tokens.shape[1] ** 0.5)
                return tokens[:, : h * w, :].reshape(1, h, w, -1).permute(0, 3, 1, 2).squeeze(0).cpu()
            raise RuntimeError("Unsupported torchhub DINO output shape")
        if self.real_backend == "timm":
            if hasattr(self.model, "forward_features"):
                out = self.model.forward_features(pixels)
            else:
                out = self.model(pixels)
            if isinstance(out, dict):
                out = self._first_tensor(out, ("x_norm_patchtokens", "features", "last_hidden_state"))
            if isinstance(out, torch.Tensor) and out.ndim == 4:
                return out.squeeze(0).cpu()
            if isinstance(out, torch.Tensor) and out.ndim == 3:
                tokens = out[:, 1:, :] if out.shape[1] > 1 else out
                h = w = int(tokens.shape[1] ** 0.5)
                return tokens[:, : h * w, :].reshape(1, h, w, -1).permute(0, 3, 1, 2).squeeze(0).cpu()
            raise RuntimeError("Unsupported timm DINO output shape")
        raise RuntimeError(f"Unsupported backend: {self.real_backend}")

    def point_features(
        self,
        rasterized: RasterizedBlock,
        out_dim: int = 64,
        projection_seed: int = 13,
        include_stat_features: bool = True,
    ) -> torch.Tensor:
        if self.real_backend == "stat":
            features = stat_dense_features(rasterized.raster, rasterized.point_cells)
        else:
            fmap = self.image_feature_map(rasterized.image)
            fmap = F.interpolate(
                fmap.unsqueeze(0),
                size=(rasterized.raster.shape[1], rasterized.raster.shape[2]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            features = sample_raster_at_points(fmap, rasterized.point_cells)
            if include_stat_features:
                features = torch.cat([features, stat_dense_features(rasterized.raster, rasterized.point_cells)], dim=1)
        features = normalize_features(features.float())
        return deterministic_projection(features, int(out_dim), seed=int(projection_seed))
