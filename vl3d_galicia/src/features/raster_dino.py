from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.data.pnoa import PNOA_FEATURE_SCHEMA


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
    projection_view: str = "top_xy"


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


def _named_feature_columns(block: dict[str, Any], features: torch.Tensor) -> dict[str, torch.Tensor]:
    names = block.get("feature_names")
    if names is None and isinstance(block.get("feature_schema"), dict):
        names = block["feature_schema"].get("names")
    if names is None:
        raise ValueError(
            "Block has no feature_names/feature_schema. Rebuild the PNOA block cache with "
            f"schema {PNOA_FEATURE_SCHEMA.version}; positional spectral columns are ambiguous."
        )
    version = block.get("feature_schema_version")
    if version is None and isinstance(block.get("feature_schema"), dict):
        version = block["feature_schema"].get("version")
    if version != PNOA_FEATURE_SCHEMA.version:
        raise ValueError(
            f"Unknown PNOA feature schema version {version!r}; expected {PNOA_FEATURE_SCHEMA.version!r}. "
            "Rebuild the block cache."
        )
    names = [str(name) for name in names]
    if features.ndim != 2 or features.shape[1] != len(names):
        raise ValueError(
            f"Feature schema has {len(names)} names but tensor shape is {tuple(features.shape)}"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"Feature names must be unique, got {names}")
    missing = [name for name in PNOA_FEATURE_SCHEMA.names if name not in names]
    if missing:
        raise ValueError(f"Feature schema is missing required PNOA channels: {missing}")
    return {name: features[:, idx].float() for idx, name in enumerate(names)}


PROJECTION_AXES = {
    "top_xy": (0, 1),
    "side_xz": (0, 2),
    "side_yz": (1, 2),
}


def _cell_indices(coords: torch.Tensor, grid_size: int, projection_view: str = "top_xy") -> torch.Tensor:
    if projection_view not in PROJECTION_AXES:
        raise ValueError(f"Unknown projection view: {projection_view}")
    axes = PROJECTION_AXES[projection_view]
    plane = coords[:, list(axes)].float()
    mins = plane.min(dim=0).values
    maxs = plane.max(dim=0).values
    span = (maxs - mins).clamp_min(1e-6)
    norm = (plane - mins) / span
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
    projection_view: str = "top_xy",
) -> RasterizedBlock:
    coords = _as_float_tensor(block["coords"])
    base = _as_float_tensor(block.get("features_original", block["features"]))
    by_name = _named_feature_columns(block, base)
    z = coords[:, 2].float()
    z_norm = _normalize_01(z)
    columns = [
        by_name["red"],
        by_name["green"],
        by_name["blue"],
        by_name["intensity"],
        by_name["nir"],
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
    cells = _cell_indices(coords, int(grid_size), projection_view=projection_view)
    raster = aggregate_points_to_raster(values, cells, int(grid_size))
    names = [*names, "density"]
    image = raster_to_image(raster, names, mode="rgb_nir_height")
    return RasterizedBlock(
        image=image,
        raster=raster,
        point_cells=cells,
        channel_names=names,
        projection_view=projection_view,
    )


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
        hf_repo_id: str | None = None,
        hf_revision: str | None = None,
        dtype: str = "float16",
    ):
        self.backend = backend
        self.hf_repo_id = hf_repo_id
        self.hf_revision = hf_revision
        self.requested_model_name = hf_repo_id or model_name
        self.model_name = self.requested_model_name
        env_model_path = os.environ.get("DINOV3_MODEL_PATH")
        self.model_source = env_model_path if env_model_path and "dinov3" in self.requested_model_name.lower() else model_name
        self.repo_dir = repo_dir
        self.weights = weights
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.normalize = normalize
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(f"Unknown DINO dtype: {dtype}")
        self.dtype = dtype
        self.model = None
        self.processor = None
        self.real_backend = "stat"
        if backend != "stat":
            self._load_model()

    @property
    def uses_real_dino(self) -> bool:
        return self.real_backend != "stat"

    def _backend_candidates(self) -> list[str]:
        if self.backend != "auto":
            return [self.backend]
        requested = self.requested_model_name.lower()
        if requested.startswith("dinov2_"):
            return ["dinov2"]
        if "/" in self.requested_model_name:
            return ["hf"]
        if self.repo_dir:
            return ["torchhub"]
        return ["timm"]

    def _load_model(self) -> None:
        errors = []
        for candidate in self._backend_candidates():
            try:
                if candidate == "dinov2":
                    if not self.model_name.startswith("dinov2_"):
                        raise ValueError("backend=dinov2 requires an explicit dinov2_* model; family substitution is forbidden")
                    name = self.model_name
                    source = str(Path(self.repo_dir)) if self.repo_dir else "facebookresearch/dinov2"
                    kwargs = {"source": "local"} if self.repo_dir else {}
                    self.model = torch.hub.load(source, name, **kwargs).to(self.device).eval()
                    self.real_backend = "dinov2"
                    self.model_name = name
                    return
                if candidate == "hf":
                    from transformers import AutoModel

                    torch_dtype = {
                        "float32": torch.float32,
                        "float16": torch.float16,
                        "bfloat16": torch.bfloat16,
                    }[self.dtype]
                    local_only = Path(str(self.model_source)).exists() or os.environ.get("HF_HUB_OFFLINE") == "1"
                    self.model = AutoModel.from_pretrained(
                        self.model_source,
                        revision=self.hf_revision,
                        local_files_only=local_only,
                        dtype=torch_dtype,
                        low_cpu_mem_usage=True,
                    ).to(self.device).eval()
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
        normalized = (batch - mean) / std
        if self.model is not None:
            model_dtype = next(self.model.parameters()).dtype
            normalized = normalized.to(dtype=model_dtype)
        return normalized

    @property
    def checkpoint_path(self) -> Path | None:
        if self.weights and Path(self.weights).is_file():
            return Path(self.weights)
        source = Path(str(self.model_source))
        candidate = source / "model.safetensors"
        return candidate if candidate.is_file() else None

    @property
    def config_path(self) -> Path | None:
        source = Path(str(self.model_source))
        candidate = source / "config.json"
        return candidate if candidate.is_file() else None

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
            outputs = self.model(
                pixel_values=pixels,
                output_hidden_states=False,
                interpolate_pos_encoding=True,
            )
            tokens = outputs.last_hidden_state
            patch = int(getattr(self.model.config, "patch_size", 16) or 16)
            register_tokens = int(getattr(self.model.config, "num_register_tokens", 0) or 0)
            h = max(int(image.shape[1]) // patch, 1)
            w = max(int(image.shape[2]) // patch, 1)
            prefix_tokens = 1 + register_tokens
            expected = prefix_tokens + h * w
            if tokens.shape[1] != expected:
                raise ValueError(
                    "Unexpected Hugging Face DINO token layout: "
                    f"got {tokens.shape[1]}, expected CLS(1)+register({register_tokens})+patches({h * w})"
                )
            patch_tokens = tokens[:, prefix_tokens:, :]
            return patch_tokens.reshape(1, h, w, -1).permute(0, 3, 1, 2).squeeze(0).cpu()
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
