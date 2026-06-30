from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from src.data.classes import CLASS_NAMES

GROUND = 0
LOW_VEGETATION = 1
MEDIUM_VEGETATION = 2
HIGH_VEGETATION = 3
BUILDING = 4
WATER = 5
IGNORE = 6

VEGETATION_CLASSES = [LOW_VEGETATION, MEDIUM_VEGETATION, HIGH_VEGETATION]
FOREST_CORE_CLASSES = [MEDIUM_VEGETATION, HIGH_VEGETATION]
HIGH_VEGETATION_CLASS = HIGH_VEGETATION


def binary_metrics(target: np.ndarray, pred: np.ndarray, positive_classes: list[int]) -> dict:
    valid = target != IGNORE
    target_pos = np.isin(target[valid], positive_classes)
    pred_pos = np.isin(pred[valid], positive_classes)
    tp = int(np.sum(target_pos & pred_pos))
    fp = int(np.sum(~target_pos & pred_pos))
    fn = int(np.sum(target_pos & ~pred_pos))
    tn = int(np.sum(~target_pos & ~pred_pos))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "support": int(np.sum(target_pos)),
    }


def forest_classification_metrics(target: np.ndarray, pred: np.ndarray, model_name: str) -> list[dict]:
    tasks = {
        "vegetation": VEGETATION_CLASSES,
        "forest_core": FOREST_CORE_CLASSES,
        "high_vegetation": [HIGH_VEGETATION_CLASS],
    }
    rows = []
    for task, classes in tasks.items():
        row = {"model_name": model_name, "task": task}
        row.update(binary_metrics(target, pred, classes))
        rows.append(row)
    return rows


def confusion_count(target: np.ndarray, pred: np.ndarray, true_cls: int, pred_cls: int) -> int:
    valid = target != IGNORE
    return int(np.sum(valid & (target == true_cls) & (pred == pred_cls)))


def forest_confusions(target: np.ndarray, pred: np.ndarray, model_name: str) -> list[dict]:
    pairs = [
        ("high_as_building", HIGH_VEGETATION, BUILDING),
        ("high_as_ground", HIGH_VEGETATION, GROUND),
        ("medium_as_low", MEDIUM_VEGETATION, LOW_VEGETATION),
        ("low_as_ground", LOW_VEGETATION, GROUND),
        ("building_as_high", BUILDING, HIGH_VEGETATION),
    ]
    return [
        {
            "model_name": model_name,
            "confusion": name,
            "true_class": CLASS_NAMES[true_cls],
            "pred_class": CLASS_NAMES[pred_cls],
            "count": confusion_count(target, pred, true_cls, pred_cls),
        }
        for name, true_cls, pred_cls in pairs
    ]


def ratio(labels: np.ndarray, classes: list[int]) -> float:
    valid = labels != IGNORE
    if not np.any(valid):
        return 0.0
    return float(np.mean(np.isin(labels[valid], classes)))


def per_tile_rows(records: list[dict], model_name: str, pred_key: str) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["tile_id"]].append(record)
    rows = []
    for tile_id, items in grouped.items():
        target = np.concatenate([item["labels"] for item in items])
        pred = np.concatenate([item[pred_key] for item in items])
        veg = binary_metrics(target, pred, VEGETATION_CLASSES)
        high = binary_metrics(target, pred, [HIGH_VEGETATION])
        row = {
            "model_name": model_name,
            "tile_id": tile_id,
            "n_points": int(target.size),
            "gt_vegetation_pct": ratio(target, VEGETATION_CLASSES) * 100.0,
            "pred_vegetation_pct": ratio(pred, VEGETATION_CLASSES) * 100.0,
            "gt_high_vegetation_pct": ratio(target, [HIGH_VEGETATION]) * 100.0,
            "pred_high_vegetation_pct": ratio(pred, [HIGH_VEGETATION]) * 100.0,
            "vegetation_f1": veg["f1"],
            "vegetation_iou": veg["iou"],
            "high_vegetation_f1": high["f1"],
            "high_vegetation_iou": high["iou"],
        }
        row["vegetation_pct_abs_error"] = abs(row["pred_vegetation_pct"] - row["gt_vegetation_pct"])
        row["high_vegetation_pct_abs_error"] = abs(row["pred_high_vegetation_pct"] - row["gt_high_vegetation_pct"])
        rows.append(row)
    return rows


def dominant_class(labels: np.ndarray) -> int:
    valid = labels[labels != IGNORE]
    if valid.size == 0:
        return IGNORE
    return int(Counter(valid.tolist()).most_common(1)[0][0])


def grid_rows(
    records: list[dict],
    grid_size: float = 2.0,
    gap_threshold: float = 0.2,
    min_points_per_cell: int = 10,
) -> list[dict]:
    buckets = defaultdict(list)
    for record in records:
        coords = record["coords"]
        cx = np.floor(coords[:, 0] / grid_size).astype(np.int64)
        cy = np.floor(coords[:, 1] / grid_size).astype(np.int64)
        for cell in np.unique(np.stack([cx, cy], axis=1), axis=0):
            mask = (cx == cell[0]) & (cy == cell[1])
            buckets[(int(cell[0]), int(cell[1]))].append((record, mask))

    rows = []
    for (cell_x, cell_y), items in buckets.items():
        coords = np.concatenate([record["coords"][mask] for record, mask in items])
        labels = np.concatenate([record["labels"][mask] for record, mask in items])
        baseline = np.concatenate([record["baseline_pred"][mask] for record, mask in items])
        jepa = np.concatenate([record["jepa_pred"][mask] for record, mask in items])
        n = int(labels.size)
        z = coords[:, 2]
        ground_mask = labels == GROUND
        veg_mask = np.isin(labels, VEGETATION_CLASSES)
        ground_z = float(np.percentile(z[ground_mask], 5)) if np.any(ground_mask) else float(np.min(z))
        veg_z_p95 = float(np.percentile(z[veg_mask], 95)) if np.any(veg_mask) else float(np.percentile(z, 95))
        canopy_height = float(max(0.0, veg_z_p95 - ground_z))
        high_cover = ratio(jepa, [HIGH_VEGETATION])
        med_high_cover = ratio(jepa, FOREST_CORE_CLASSES)
        gt_veg = ratio(labels, VEGETATION_CLASSES)
        baseline_veg = ratio(baseline, VEGETATION_CLASSES)
        jepa_veg = ratio(jepa, VEGETATION_CLASSES)
        gt_high = ratio(labels, [HIGH_VEGETATION])
        baseline_high = ratio(baseline, [HIGH_VEGETATION])
        jepa_high = ratio(jepa, [HIGH_VEGETATION])
        baseline_err = float(np.mean((baseline != labels) & (labels != IGNORE))) if n else 0.0
        jepa_err = float(np.mean((jepa != labels) & (labels != IGNORE))) if n else 0.0
        rows.append(
            {
                "cell_id": f"{cell_x}_{cell_y}",
                "cell_x": cell_x,
                "cell_y": cell_y,
                "x": float((cell_x + 0.5) * grid_size),
                "y": float((cell_y + 0.5) * grid_size),
                "n_points": n,
                "z_min": float(np.min(z)),
                "z_mean": float(np.mean(z)),
                "z_max": float(np.max(z)),
                "z_p95": float(np.percentile(z, 95)),
                "height_range": float(np.percentile(z, 95) - np.min(z)),
                "ground_z_proxy": ground_z,
                "veg_z_p95": veg_z_p95,
                "canopy_height_proxy": canopy_height,
                "vegetation_ratio": jepa_veg,
                "vegetation_ratio_gt": gt_veg,
                "vegetation_ratio_baseline": baseline_veg,
                "vegetation_ratio_jepa": jepa_veg,
                "high_vegetation_ratio": high_cover,
                "high_vegetation_ratio_gt": gt_high,
                "high_vegetation_ratio_baseline": baseline_high,
                "high_vegetation_ratio_jepa": jepa_high,
                "medium_high_vegetation_ratio": med_high_cover,
                "high_canopy_cover_proxy": high_cover,
                "medium_high_canopy_cover_proxy": med_high_cover,
                "canopy_gap_proxy": 1.0 - med_high_cover,
                "biomass_proxy": canopy_height * med_high_cover,
                "is_canopy_gap": bool(n >= min_points_per_cell and med_high_cover < gap_threshold),
                "dominant_class_gt": dominant_class(labels),
                "dominant_class_baseline": dominant_class(baseline),
                "dominant_class_jepa": dominant_class(jepa),
                "baseline_error_rate": baseline_err,
                "jepa_error_rate": jepa_err,
                "error_difference_baseline_minus_jepa": baseline_err - jepa_err,
            }
        )
    return rows


def anomaly_rows(grid: list[dict], gap_threshold: float = 0.2, sparse_threshold: int = 10) -> list[dict]:
    rows = []
    for row in grid:
        base = {
            "cell_id": row["cell_id"],
            "x": row["x"],
            "y": row["y"],
            "height_proxy": row["canopy_height_proxy"],
            "vegetation_ratio": row["vegetation_ratio"],
            "high_vegetation_ratio": row["high_vegetation_ratio"],
        }
        def add(kind, score, explanation, baseline_value="", jepa_value=""):
            rows.append({**base, "anomaly_type": kind, "score": float(score), "explanation": explanation, "baseline_value": baseline_value, "jepa_value": jepa_value})

        if row["high_vegetation_ratio"] > 0.4 and row["canopy_height_proxy"] < 1.0:
            add("pred_high_forest_but_low_height", row["high_vegetation_ratio"], "High vegetation prediction with low canopy height proxy", "", row["high_vegetation_ratio"])
        if row["vegetation_ratio"] < 0.1 and row["canopy_height_proxy"] > 3.0:
            add("height_high_but_pred_no_forest", row["canopy_height_proxy"], "High canopy height proxy but low predicted vegetation", "", row["vegetation_ratio"])
        disagreement = abs(row["baseline_error_rate"] - row["jepa_error_rate"])
        if disagreement > 0.25:
            add("baseline_jepa_disagreement", disagreement, "Large baseline vs JEPA error-rate disagreement", row["baseline_error_rate"], row["jepa_error_rate"])
        if row["n_points"] < sparse_threshold:
            add("sparse_points_cell", sparse_threshold - row["n_points"], "Low point density cell", row["n_points"], "")
        if row["is_canopy_gap"]:
            add("canopy_gap", 1.0 - row["medium_high_canopy_cover_proxy"], "Medium/high canopy cover below threshold", gap_threshold, row["medium_high_canopy_cover_proxy"])
    return rows


def forest_mvp_checklist(paths: dict, has_comparison: bool) -> dict:
    checks = {
        "explicit_forest_task": True,
        "vegetation_metrics": bool(paths.get("classification_metrics")),
        "high_vegetation_metrics": bool(paths.get("classification_metrics")),
        "tile_evaluation": bool(paths.get("by_tile")),
        "canopy_height_proxy": bool(paths.get("grid")),
        "canopy_cover_proxy": bool(paths.get("grid")),
        "canopy_gaps": bool(paths.get("canopy_gaps_map")),
        "forest_maps_exported": bool(paths.get("maps_dir")),
        "baseline_vs_jepa_comparison": has_comparison,
        "reproducible_report": bool(paths.get("report")),
    }
    passed = sum(bool(value) for value in checks.values())
    if passed >= 8:
        verdict = "Forest-JEPA MVP inicial"
    elif passed >= 5:
        verdict = "Geo-JEPA con modulo forestal parcial"
    else:
        verdict = "No Forest-JEPA"
    return {"checks": checks, "passed": passed, "total": len(checks), "verdict": verdict}


def compare_forest_grids_t1_t2(grid_t1: list[dict], grid_t2: list[dict]) -> list[dict]:
    """Future interface for multitemporal forest change detection.

    The current MVP does not call this function because the pipeline has no
    validated paired T1/T2 forest dataset yet.
    """
    by_id_t1 = {row["cell_id"]: row for row in grid_t1}
    by_id_t2 = {row["cell_id"]: row for row in grid_t2}
    rows = []
    for cell_id in sorted(set(by_id_t1) & set(by_id_t2)):
        a = by_id_t1[cell_id]
        b = by_id_t2[cell_id]
        rows.append(
            {
                "cell_id": cell_id,
                "delta_canopy_cover": b.get("medium_high_canopy_cover_proxy", 0.0) - a.get("medium_high_canopy_cover_proxy", 0.0),
                "delta_high_vegetation": b.get("high_vegetation_ratio", 0.0) - a.get("high_vegetation_ratio", 0.0),
                "delta_height_proxy": b.get("canopy_height_proxy", 0.0) - a.get("canopy_height_proxy", 0.0),
            }
        )
    return rows
