import numpy as np

from scripts.train_common import segmentation_optimizer_groups
from src.eval.forest_metrics import HIGH_VEGETATION, LOW_VEGETATION, MEDIUM_VEGETATION, forest_mvp_checklist, grid_rows, ratio
from src.models.segmentation.heads import PointSegmentationNet


def test_freeze_encoder_disables_encoder_gradients():
    model = PointSegmentationNet(in_channels=8, probe_type="linear")
    model.freeze_encoder()
    assert not any(param.requires_grad for param in model.encoder.parameters())
    assert any(param.requires_grad for param in model.head.parameters())
    model.train()
    assert not model.encoder.training


def test_optimizer_groups_encoder_lr_scale():
    model = PointSegmentationNet(in_channels=8, probe_type="mlp")
    groups, lr_info = segmentation_optimizer_groups(model, lr=1e-3, encoder_lr_scale=0.01)
    lrs = sorted(group["lr"] for group in groups)
    assert lrs == [1e-5, 1e-3]
    assert lr_info["encoder_lr"] == 1e-5
    assert lr_info["head_lr"] == 1e-3


def test_forest_ratios_toy():
    labels = np.array([0, LOW_VEGETATION, MEDIUM_VEGETATION, HIGH_VEGETATION, 4, 6])
    assert ratio(labels, [LOW_VEGETATION, MEDIUM_VEGETATION, HIGH_VEGETATION]) == 3 / 5
    assert ratio(labels, [HIGH_VEGETATION]) == 1 / 5


def test_canopy_proxy_grid_toy():
    record = {
        "coords": np.array([[0.1, 0.1, 10.0], [0.2, 0.1, 11.0], [0.3, 0.2, 15.0], [0.4, 0.2, 20.0]]),
        "labels": np.array([0, 0, MEDIUM_VEGETATION, HIGH_VEGETATION]),
        "baseline_pred": np.array([0, 0, MEDIUM_VEGETATION, HIGH_VEGETATION]),
        "jepa_pred": np.array([0, 0, MEDIUM_VEGETATION, HIGH_VEGETATION]),
    }
    rows = grid_rows([record], grid_size=1.0, min_points_per_cell=1)
    assert len(rows) == 1
    row = rows[0]
    assert row["z_min"] == 10.0
    assert row["z_p95"] > 19.0
    assert row["canopy_height_proxy"] > 8.0
    assert row["medium_high_canopy_cover_proxy"] == 0.5


def test_forest_mvp_checklist_verdicts():
    full = {
        "classification_metrics": "x",
        "by_tile": "x",
        "grid": "x",
        "canopy_gaps_map": "x",
        "maps_dir": "x",
        "report": "x",
    }
    result = forest_mvp_checklist(full, has_comparison=True)
    assert result["passed"] >= 8
    assert result["verdict"] == "Forest-JEPA MVP inicial"
