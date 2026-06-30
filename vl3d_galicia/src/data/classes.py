from __future__ import annotations

import numpy as np
import torch

IGNORE_INDEX = 6

CLASS_NAMES = {
    0: "ground",
    1: "low_vegetation",
    2: "medium_vegetation",
    3: "high_vegetation",
    4: "building",
    5: "water",
    6: "unreliable",
}

ASPRS_NAMES = {
    0: "created_never_classified",
    1: "unclassified",
    2: "ground",
    3: "low_vegetation",
    4: "medium_vegetation",
    5: "high_vegetation",
    6: "building",
    7: "low_point_noise",
    8: "model_key_point",
    9: "water",
    12: "overlap",
}

ASPRS_TO_PAPER = {
    2: 0,
    3: 1,
    4: 2,
    5: 3,
    6: 4,
    9: 5,
}


def map_labels(classification: np.ndarray) -> np.ndarray:
    labels = np.full(classification.shape, IGNORE_INDEX, dtype=np.int64)
    for asprs_class, paper_class in ASPRS_TO_PAPER.items():
        labels[classification == asprs_class] = paper_class
    return labels


def effective_number_class_weights(
    counts: dict[int, int] | np.ndarray,
    num_classes: int = 7,
    ignore_index: int = IGNORE_INDEX,
    beta: float = 0.9999,
    max_weight: float = 12.0,
) -> torch.Tensor:
    return class_weights_from_counts(
        counts,
        num_classes=num_classes,
        ignore_index=ignore_index,
        mode="effective",
        beta=beta,
        max_weight=max_weight,
    )


def class_weights_from_counts(
    counts: dict[int, int] | np.ndarray,
    num_classes: int = 7,
    ignore_index: int = IGNORE_INDEX,
    mode: str = "inverse_sqrt",
    beta: float = 0.9999,
    max_weight: float = 20.0,
) -> torch.Tensor:
    if isinstance(counts, dict):
        arr = np.array([counts.get(i, 0) for i in range(num_classes)], dtype=np.float64)
    else:
        arr = np.asarray(counts, dtype=np.float64)
    weights = np.ones(num_classes, dtype=np.float64)
    valid = np.arange(num_classes) != ignore_index
    safe_counts = np.maximum(arr[valid], 1.0)
    if mode == "effective":
        effective = 1.0 - np.power(beta, safe_counts)
        weights[valid] = (1.0 - beta) / np.maximum(effective, 1e-12)
    elif mode == "inverse":
        weights[valid] = 1.0 / safe_counts
    elif mode == "inverse_sqrt":
        weights[valid] = 1.0 / np.sqrt(safe_counts)
    elif mode == "median_frequency":
        weights[valid] = np.median(safe_counts) / safe_counts
    elif mode == "none":
        weights[valid] = 1.0
    else:
        raise ValueError(f"Unknown class weight mode: {mode}")
    weights[valid] = weights[valid] / np.mean(weights[valid])
    weights[valid] = np.minimum(weights[valid], max_weight)
    weights[ignore_index] = 0.0
    return torch.tensor(weights, dtype=torch.float32)
