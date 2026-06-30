from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


EPS = 1e-6


@dataclass(frozen=True)
class GeomContextConfig:
    cell_sizes: tuple[float, ...] = (2.5, 5.0, 10.0)
    include_tw_summary: bool = True
    include_metric_height: bool = True


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_std(values: np.ndarray) -> float:
    value = float(np.std(values)) if values.size else 0.0
    return value if value > EPS else 1.0


def _normalize_block_z(z: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    z = z.astype(np.float32, copy=False)
    if z.size == 0:
        return z, 0.0, 0.0, 1.0
    z_p05 = float(np.percentile(z, 5.0))
    z_p95 = float(np.percentile(z, 95.0))
    z_scale = max(z_p95 - z_p05, 1.0)
    z_norm = np.clip((z - z_p05) / z_scale, -2.0, 4.0).astype(np.float32)
    return z_norm, z_p05, z_p95, z_scale


def _spectral_indices(base: np.ndarray) -> dict[str, np.ndarray]:
    n = base.shape[0]
    zeros = np.zeros(n, dtype=np.float32)
    intensity = base[:, 0].astype(np.float32, copy=False) if base.shape[1] > 0 else zeros
    red = base[:, 1].astype(np.float32, copy=False) if base.shape[1] > 1 else zeros
    green = base[:, 2].astype(np.float32, copy=False) if base.shape[1] > 2 else zeros
    blue = base[:, 3].astype(np.float32, copy=False) if base.shape[1] > 3 else zeros
    nir = base[:, 4].astype(np.float32, copy=False) if base.shape[1] > 4 else zeros
    ndvi = (nir - red) / np.maximum(nir + red, EPS)
    ndwi = (green - nir) / np.maximum(green + nir, EPS)
    brightness = (red + green + blue) / 3.0
    exg = 2.0 * green - red - blue
    nir_red = nir - red
    return {
        "intensity": intensity,
        "red": red,
        "green": green,
        "blue": blue,
        "nir": nir,
        "ndvi": ndvi.astype(np.float32),
        "ndwi": ndwi.astype(np.float32),
        "brightness": brightness.astype(np.float32),
        "exg": exg.astype(np.float32),
        "nir_red": nir_red.astype(np.float32),
    }


def _group_stats(values: np.ndarray, group_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(group_ids, kind="mergesort")
    sorted_groups = group_ids[order]
    sorted_values = values[order].astype(np.float32, copy=False)
    unique, starts, counts = np.unique(sorted_groups, return_index=True, return_counts=True)
    sums = np.add.reduceat(sorted_values, starts, axis=0)
    means = sums / counts[:, None].astype(np.float32)
    sums_sq = np.add.reduceat(sorted_values * sorted_values, starts, axis=0)
    variances = np.maximum(sums_sq / counts[:, None].astype(np.float32) - means * means, 0.0)
    stds = np.sqrt(variances).astype(np.float32)
    mins = np.minimum.reduceat(sorted_values, starts, axis=0)
    maxs = np.maximum.reduceat(sorted_values, starts, axis=0)
    inverse_sorted = np.empty(group_ids.shape[0], dtype=np.int64)
    inverse_sorted[order] = np.repeat(np.arange(unique.shape[0], dtype=np.int64), counts)
    return means[inverse_sorted], stds[inverse_sorted], mins[inverse_sorted], maxs[inverse_sorted], counts[inverse_sorted]


def _cell_features(
    coords: np.ndarray,
    z: np.ndarray,
    z_norm: np.ndarray,
    spectral: dict[str, np.ndarray],
    cell_size: float,
    z_scale: float,
    include_metric_height: bool = True,
) -> np.ndarray:
    xy = coords[:, :2].astype(np.float32, copy=False)
    mins = xy.min(axis=0)
    cell_xy = np.floor((xy - mins[None, :]) / max(float(cell_size), EPS)).astype(np.int64)
    x_bins = int(cell_xy[:, 0].max()) + 1 if cell_xy.size else 1
    group_ids = cell_xy[:, 1] * max(x_bins, 1) + cell_xy[:, 0]
    values = np.stack(
        [
            z.astype(np.float32),
            z_norm.astype(np.float32),
            spectral["intensity"],
            spectral["nir"],
            spectral["ndvi"],
            spectral["brightness"],
        ],
        axis=1,
    )
    means, stds, mins_v, maxs_v, counts = _group_stats(values, group_ids)
    z_mean = means[:, 0]
    z_std = stds[:, 0]
    z_min = mins_v[:, 0]
    z_max = maxs_v[:, 0]
    z_range = z_max - z_min
    log_density = np.log1p(counts.astype(np.float32))
    density_norm = log_density / max(float(np.log1p(max(counts.max(), 1))), 1.0)
    columns = [
        density_norm,
        log_density / 8.0,
        np.clip((z - z_min) / z_scale, -2.0, 4.0),
        np.clip((z - z_mean) / z_scale, -2.0, 4.0),
        np.clip((z_max - z) / z_scale, -2.0, 4.0),
        np.clip(z_range / z_scale, 0.0, 6.0),
        np.clip(z_std / z_scale, 0.0, 3.0),
        means[:, 1],
        stds[:, 1],
        means[:, 2],
        means[:, 3],
        means[:, 4],
        means[:, 5],
    ]
    if include_metric_height:
        columns.extend(
            [
                np.clip((z - z_min) / 2.0, 0.0, 8.0),
                np.clip((z - z_mean) / 2.0, -8.0, 8.0),
                np.clip((z_max - z) / 2.0, 0.0, 8.0),
                np.clip(z_range / 10.0, 0.0, 8.0),
                np.clip(z_std / 2.0, 0.0, 8.0),
            ]
        )
    out = np.stack(columns, axis=1)
    return out.astype(np.float32, copy=False)


def build_geom_context_features(block: dict, config: GeomContextConfig | None = None) -> tuple[torch.Tensor, list[str]]:
    config = config or GeomContextConfig()
    coords = _to_numpy(block["coords"]).astype(np.float32, copy=False)
    base = _to_numpy(block.get("features_original", block["features"])).astype(np.float32, copy=False)
    z = coords[:, 2].astype(np.float32, copy=False)
    z_norm, z_p05, z_p95, z_scale = _normalize_block_z(z)
    spectral = _spectral_indices(base)
    block_z_mean = float(np.mean(z)) if z.size else 0.0
    block_z_std = _safe_std(z)
    direct = [
        z_norm,
        np.clip((z - block_z_mean) / block_z_std, -4.0, 4.0).astype(np.float32),
        np.clip((z - z_p05) / z_scale, -2.0, 4.0).astype(np.float32),
        np.clip((z_p95 - z) / z_scale, -4.0, 4.0).astype(np.float32),
        spectral["ndvi"],
        spectral["ndwi"],
        spectral["brightness"],
        spectral["exg"],
        spectral["nir_red"],
    ]
    names = [
        "z_norm_p05_p95",
        "z_standard_block",
        "height_above_block_p05",
        "height_below_block_p95",
        "ndvi",
        "ndwi",
        "brightness",
        "exg",
        "nir_minus_red",
    ]
    if config.include_metric_height:
        direct.extend(
            [
                np.clip((z - z_p05) / 2.0, -2.0, 8.0).astype(np.float32),
                np.clip((z_p95 - z) / 2.0, -8.0, 8.0).astype(np.float32),
            ]
        )
        names.extend(["height_above_block_p05_m2", "height_below_block_p95_m2"])
    chunks = [np.stack(direct, axis=1).astype(np.float32, copy=False)]
    for cell_size in config.cell_sizes:
        cell = _cell_features(
            coords,
            z,
            z_norm,
            spectral,
            cell_size=cell_size,
            z_scale=z_scale,
            include_metric_height=config.include_metric_height,
        )
        prefix = f"cell_{cell_size:g}m"
        cell_names = [
            f"{prefix}_density_norm",
            f"{prefix}_log_density_scaled",
            f"{prefix}_height_above_min",
            f"{prefix}_height_from_mean",
            f"{prefix}_height_below_max",
            f"{prefix}_height_range",
            f"{prefix}_z_std",
            f"{prefix}_z_norm_mean",
            f"{prefix}_z_norm_std",
            f"{prefix}_intensity_mean",
            f"{prefix}_nir_mean",
            f"{prefix}_ndvi_mean",
            f"{prefix}_brightness_mean",
        ]
        if config.include_metric_height:
            cell_names.extend(
                [
                    f"{prefix}_height_above_min_m2",
                    f"{prefix}_height_from_mean_m2",
                    f"{prefix}_height_below_max_m2",
                    f"{prefix}_height_range_m10",
                    f"{prefix}_z_std_m2",
                ]
            )
        names.extend(cell_names)
        chunks.append(cell)
    if config.include_tw_summary and "tw_features" in block:
        tw = _to_numpy(block["tw_features"]).astype(np.float32, copy=False)
        keep = min(8, tw.shape[1])
        if keep:
            tw_keep = np.nan_to_num(tw[:, :keep], nan=0.0, posinf=0.0, neginf=0.0)
            tw_std = tw_keep.std(axis=0, keepdims=True)
            tw_std = np.maximum(tw_std, EPS)
            tw_norm = np.clip((tw_keep - tw_keep.mean(axis=0, keepdims=True)) / tw_std, -5.0, 5.0)
            chunks.append(tw_norm.astype(np.float32, copy=False))
            names.extend([f"tw_context_norm_{idx:02d}" for idx in range(keep)])
    features = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(features), names
