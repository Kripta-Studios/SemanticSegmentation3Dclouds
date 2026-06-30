from __future__ import annotations

import glob
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
import yaml

from src.data.classes import IGNORE_INDEX, class_weights_from_counts


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def merge_cli(defaults: dict, **overrides) -> dict:
    out = dict(defaults)
    for key, value in overrides.items():
        if value is not None:
            out[key] = value
    return out


def infer_in_channels(
    block_dir: str,
    use_tw_input: bool = False,
    external_feature_dir: str | None = None,
    external_feature_key: str = "dino_features",
) -> int:
    return infer_channel_layout(
        block_dir,
        use_tw_input=use_tw_input,
        external_feature_dir=external_feature_dir,
        external_feature_key=external_feature_key,
    )["in_channels"]


def infer_channel_layout(
    block_dir: str,
    use_tw_input: bool = False,
    external_feature_dir: str | None = None,
    external_feature_key: str = "dino_features",
) -> dict[str, int]:
    from src.data.jepa_dataset import block_features

    files = sorted(glob.glob(str(Path(block_dir) / "*.pt")))
    if not files:
        raise FileNotFoundError(f"No .pt blocks in {block_dir}")
    data = torch.load(files[0], weights_only=False, map_location="cpu")
    base_channels = int(block_features(data, use_tw_input=use_tw_input).shape[1])
    external_channels = 0
    if external_feature_dir:
        source = Path(files[0])
        root = Path(external_feature_dir)
        candidates = [root / Path(block_dir).name / source.name, root / source.name]
        for candidate in candidates:
            if candidate.exists():
                payload = torch.load(candidate, weights_only=False, map_location="cpu")
                if external_feature_key not in payload:
                    raise KeyError(f"{candidate} does not contain '{external_feature_key}'")
                external_channels = int(payload[external_feature_key].shape[1])
                break
        else:
            raise FileNotFoundError(f"External features for {source.name} not found under {external_feature_dir}")
    return {
        "base_in_channels": base_channels,
        "external_in_channels": external_channels,
        "in_channels": base_channels + external_channels,
    }


def class_counts_from_blocks(block_dir: str, max_blocks: int = 0) -> Counter:
    files = sorted(glob.glob(str(Path(block_dir) / "*.pt")))
    if max_blocks > 0:
        files = files[:max_blocks]
    return class_counts_from_files(files)


def class_counts_from_files(files: list[str]) -> Counter:
    from src.data.segmentation_dataset import label_counts_for_files

    cached_counts = label_counts_for_files(files)
    if cached_counts is not None:
        counts = Counter()
        for row in cached_counts:
            for i, value in enumerate(row.tolist()):
                counts[i] += int(value)
        return counts
    counts = Counter()
    for path in files:
        labels = torch.load(path, weights_only=False, map_location="cpu")["labels"]
        bincount = torch.bincount(labels.long(), minlength=7)
        for i, value in enumerate(bincount.tolist()):
            counts[i] += int(value)
    return counts


def class_weights_from_blocks(block_dir: str, max_blocks: int = 0, mode: str = "inverse_sqrt", max_weight: float = 20.0) -> torch.Tensor:
    return class_weights_from_counts(dict(class_counts_from_blocks(block_dir, max_blocks=max_blocks)), mode=mode, max_weight=max_weight)


def class_weights_from_files(files: list[str], mode: str = "inverse_sqrt", max_weight: float = 20.0) -> torch.Tensor:
    return class_weights_from_counts(dict(class_counts_from_files(files)), mode=mode, max_weight=max_weight)


def parse_class_boost(spec: str | None) -> dict[int, float]:
    if not spec:
        return {}
    out: dict[int, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split(":", 1)
        out[int(key)] = float(value)
    return out


def balanced_block_sampler(
    dataset,
    mode: str = "inverse_sqrt",
    max_weight: float = 20.0,
    alpha: float = 1.0,
    sampler_max_weight: float = 12.0,
    class_boost: dict[int, float] | None = None,
    generator: torch.Generator | None = None,
) -> tuple[WeightedRandomSampler, dict]:
    from src.data.segmentation_dataset import label_counts_for_files

    class_boost = class_boost or {}
    per_block_counts = []
    global_counts = Counter()
    cached_counts = label_counts_for_files(dataset.files)
    iterable = [torch.tensor(row, dtype=torch.double) for row in cached_counts] if cached_counts is not None else None
    for idx, path in enumerate(dataset.files):
        if iterable is not None:
            counts = iterable[idx]
        else:
            labels = torch.load(path, weights_only=False, map_location="cpu")["labels"].long()
            valid = labels != IGNORE_INDEX
            labels = labels[valid]
            counts = torch.bincount(labels, minlength=7).double()
        per_block_counts.append(counts)
        for i, value in enumerate(counts.tolist()):
            global_counts[i] += int(value)
    class_weights = class_weights_from_counts(dict(global_counts), mode=mode, max_weight=max_weight).double()
    for cls, boost in class_boost.items():
        if 0 <= cls < len(class_weights) and cls != IGNORE_INDEX:
            class_weights[cls] *= float(boost)
    raw_weights = []
    for counts in per_block_counts:
        total = float(counts[:IGNORE_INDEX].sum().item())
        if total <= 0:
            raw_weights.append(1.0)
            continue
        freq = counts / total
        score = float((freq * class_weights).sum().item())
        raw_weights.append(max(score, 1e-6))
    weights = torch.tensor(raw_weights, dtype=torch.double)
    weights = weights / max(float(weights.mean().item()), 1e-12)
    if alpha != 1.0:
        weights = torch.pow(weights, float(alpha))
    weights = torch.clamp(weights, min=1.0 / max(float(sampler_max_weight), 1.0), max=float(sampler_max_weight))
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
    summary = {
        "sampler_mode": mode,
        "sampler_alpha": float(alpha),
        "sampler_max_weight": float(sampler_max_weight),
        "sampler_class_boost": {str(k): float(v) for k, v in class_boost.items()},
        "sampler_weight_min": float(weights.min().item()),
        "sampler_weight_mean": float(weights.mean().item()),
        "sampler_weight_max": float(weights.max().item()),
    }
    return sampler, summary


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def set_global_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def model_parameter_summary(model) -> dict:
    if hasattr(model, "parameter_summary"):
        return model.parameter_summary()
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}


def segmentation_optimizer_groups(model, lr: float, encoder_lr_scale: float = 1.0) -> tuple[list[dict], dict]:
    encoder_params = [param for param in model.encoder.parameters() if param.requires_grad]
    encoder_param_ids = {id(param) for param in model.encoder.parameters()}
    head_params = [
        param
        for param in model.parameters()
        if param.requires_grad and id(param) not in encoder_param_ids
    ]
    groups = []
    encoder_lr = float(lr) * float(encoder_lr_scale)
    if encoder_params:
        groups.append({"params": encoder_params, "lr": encoder_lr, "name": "encoder"})
    if head_params:
        groups.append({"params": head_params, "lr": float(lr), "name": "head"})
    if not groups:
        raise ValueError("No trainable parameters for segmentation optimizer")
    return groups, {"encoder_lr": encoder_lr if encoder_params else 0.0, "head_lr": float(lr) if head_params else 0.0}
