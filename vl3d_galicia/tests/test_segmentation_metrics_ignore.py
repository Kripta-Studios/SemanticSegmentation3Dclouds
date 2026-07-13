from __future__ import annotations

import pytest

from src.eval.segmentation_metrics import METRIC_PROTOCOL_VERSION, compute_segmentation_metrics


def test_prediction_ignore_is_a_false_negative_for_valid_target():
    metrics = compute_segmentation_metrics([6, 1], [0, 1], num_classes=7, ignore_index=6)

    assert metrics["OA"] == pytest.approx(0.5)
    assert metrics["class_iou"][0] == pytest.approx(0.0)
    assert metrics["class_recall"][0] == pytest.approx(0.0)
    assert metrics["confusion_matrix"][0][6] == 1
    assert metrics["predicted_ignore_count"] == 1
    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["ignored_prediction_rate"] == pytest.approx(0.5)
    assert metrics["evaluated_points"] == 2
    assert metrics["metric_protocol_version"] == METRIC_PROTOCOL_VERSION


def test_target_ignore_is_excluded_but_prediction_ignore_is_not():
    metrics = compute_segmentation_metrics([0, 6, 2], [6, 1, 2], num_classes=7, ignore_index=6)

    assert metrics["OA"] == pytest.approx(0.5)
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == 2
    assert metrics["confusion_matrix"][1][6] == 1


def test_out_of_schema_predictions_fail_loudly():
    with pytest.raises(ValueError, match="outside the declared schema"):
        compute_segmentation_metrics([7], [0], num_classes=7, ignore_index=6)


def test_miou_cannot_improve_by_hiding_an_error_as_ignore():
    hidden = compute_segmentation_metrics([6, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], num_classes=7, ignore_index=6)
    incorrectly_filtered = compute_segmentation_metrics([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], num_classes=7, ignore_index=6)
    assert hidden["mIoU"] <= incorrectly_filtered["mIoU"]
    assert hidden["class_iou"][0] == pytest.approx(0.0)


def test_seven_trainable_logits_are_rejected():
    from src.data.classes import validate_num_output_classes

    with pytest.raises(ValueError, match="target-only"):
        validate_num_output_classes(7)
    assert validate_num_output_classes(6) == 6
