from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, f1_score, matthews_corrcoef, precision_recall_fscore_support


METRIC_PROTOCOL_VERSION = "segmentation-metrics-v2-pred-ignore-is-fn"


def _empty_metrics(labels: list[int], prediction_labels: list[int]) -> dict:
    return {
        "OA": 0.0,
        "AA": 0.0,
        "macro_f1": 0.0,
        "macro_F1": 0.0,
        "weighted_f1": 0.0,
        "macro_iou": 0.0,
        "mIoU": 0.0,
        "balanced_accuracy": 0.0,
        "coverage": 0.0,
        "ignored_prediction_rate": 0.0,
        "predicted_ignore_count": 0,
        "evaluated_points": 0,
        "ignored_target_points": 0,
        "class_precision": {},
        "class_recall": {},
        "class_f1": {},
        "class_iou": {},
        "class_support": {},
        "mcc": 0.0,
        "kappa": 0.0,
        "confusion_matrix": np.zeros((len(labels), len(prediction_labels)), dtype=np.int64).tolist(),
        "confusion_matrix_true_labels": labels,
        "confusion_matrix_prediction_labels": prediction_labels,
        "metric_protocol_version": METRIC_PROTOCOL_VERSION,
    }


def compute_segmentation_metrics(preds, targets, num_classes: int = 6, ignore_index: int = 6):
    """Compute metrics while ignoring only *target* ignore labels.

    A prediction equal to ``ignore_index`` for a valid target remains an error and
    contributes a false negative to the true class.  The returned confusion
    matrix is therefore ``n_valid_classes x num_classes``; its last column makes
    abstain/ignore predictions auditable instead of silently dropping them.
    """

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    preds = np.asarray(preds).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    if preds.shape != targets.shape:
        raise ValueError(f"preds and targets must have the same shape, got {preds.shape} and {targets.shape}")

    labels = [idx for idx in range(num_classes) if idx != ignore_index]
    prediction_labels = list(range(num_classes))
    if ignore_index not in prediction_labels:
        prediction_labels.append(ignore_index)
    mask = targets != ignore_index
    preds_valid = preds[mask].astype(np.int64, copy=False)
    targets_valid = targets[mask].astype(np.int64, copy=False)
    if targets_valid.size == 0:
        return _empty_metrics(labels, prediction_labels)
    invalid_targets = ~np.isin(targets_valid, labels)
    invalid_preds = ~np.isin(preds_valid, prediction_labels)
    if invalid_targets.any() or invalid_preds.any():
        raise ValueError(
            "Labels outside the declared schema: "
            f"targets={np.unique(targets_valid[invalid_targets]).tolist()}, "
            f"predictions={np.unique(preds_valid[invalid_preds]).tolist()}"
        )

    # sklearn correctly counts predictions outside ``labels`` as false negatives
    # for recall/F1.  IoU needs the explicit rectangular matrix below.
    macro_f1 = f1_score(targets_valid, preds_valid, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(targets_valid, preds_valid, labels=labels, average="weighted", zero_division=0)
    precision_arr, recall_arr, class_f1_arr, support_arr = precision_recall_fscore_support(
        targets_valid,
        preds_valid,
        labels=labels,
        zero_division=0,
    )

    target_to_row = {label: row for row, label in enumerate(labels)}
    prediction_to_col = {label: col for col, label in enumerate(prediction_labels)}
    cm = np.zeros((len(labels), len(prediction_labels)), dtype=np.int64)
    rows = np.fromiter((target_to_row[int(value)] for value in targets_valid), dtype=np.int64)
    cols = np.fromiter((prediction_to_col[int(value)] for value in preds_valid), dtype=np.int64)
    np.add.at(cm, (rows, cols), 1)

    intersections = np.asarray([cm[target_to_row[label], prediction_to_col[label]] for label in labels])
    ground_truth_set = cm.sum(axis=1)
    predicted_set = np.asarray([cm[:, prediction_to_col[label]].sum() for label in labels])
    union = ground_truth_set + predicted_set - intersections
    iou_arr = np.divide(
        intersections,
        union,
        out=np.zeros_like(intersections, dtype=np.float64),
        where=union != 0,
    )

    class_precision = {label: float(value) for label, value in zip(labels, precision_arr)}
    class_recall = {label: float(value) for label, value in zip(labels, recall_arr)}
    class_f1 = {label: float(value) for label, value in zip(labels, class_f1_arr)}
    class_support = {label: int(value) for label, value in zip(labels, support_arr)}
    class_iou = {label: float(value) for label, value in zip(labels, iou_arr)}
    macro_iou = float(np.mean(iou_arr))
    predicted_ignore_count = int(np.sum(preds_valid == ignore_index))
    evaluated_points = int(targets_valid.size)
    coverage = float(1.0 - predicted_ignore_count / evaluated_points) if evaluated_points else 0.0

    return {
        "OA": float(np.mean(preds_valid == targets_valid)),
        "AA": float(np.mean(recall_arr)),
        "balanced_accuracy": float(np.mean(recall_arr)),
        "macro_f1": float(macro_f1),
        "macro_F1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_iou": macro_iou,
        "mIoU": macro_iou,
        "class_precision": class_precision,
        "class_recall": class_recall,
        "class_f1": class_f1,
        "class_iou": class_iou,
        "class_support": class_support,
        "mcc": float(matthews_corrcoef(targets_valid, preds_valid)),
        "kappa": float(cohen_kappa_score(targets_valid, preds_valid)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_true_labels": labels,
        "confusion_matrix_prediction_labels": prediction_labels,
        "coverage": coverage,
        "ignored_prediction_rate": float(1.0 - coverage),
        "predicted_ignore_count": predicted_ignore_count,
        "evaluated_points": evaluated_points,
        "ignored_target_points": int(np.sum(~mask)),
        "metric_protocol_version": METRIC_PROTOCOL_VERSION,
    }
