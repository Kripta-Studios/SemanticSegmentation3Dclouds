from __future__ import annotations

from pathlib import Path
import importlib.util
import json

import torch

from src.data.segmentation_dataset import SegmentationBlockDataset
from src.data.pnoa import PNOA_FEATURE_SCHEMA
from src.features.raster_dino import DinoDenseExtractor, _normalize_01, make_multichannel_raster, raster_to_image
from src.models.segmentation.heads import GatedExternalPointSegmentationNet


ROOT = Path(__file__).resolve().parents[1]


def _load_script(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _toy_block(n: int = 32) -> dict:
    coords = torch.stack(
        [
            torch.linspace(-5, 5, n),
            torch.linspace(5, -5, n),
            torch.linspace(0, 10, n),
        ],
        dim=1,
    )
    features = torch.rand(n, 5)
    labels = torch.arange(n) % 6
    tw = torch.rand(n, 25)
    return {
        "coords": coords,
        "features": features,
        "feature_names": list(PNOA_FEATURE_SCHEMA.names),
        "feature_schema": PNOA_FEATURE_SCHEMA.as_dict(),
        "feature_schema_version": PNOA_FEATURE_SCHEMA.version,
        "labels": labels.long(),
        "reliable_mask": torch.ones(n, dtype=torch.bool),
        "tw_features": tw,
    }


def test_raster_dino_stat_features_shape():
    block = _toy_block(40)
    rasterized = make_multichannel_raster(block, grid_size=16, tw_channels=4)
    extractor = DinoDenseExtractor(backend="stat", device="cpu")
    features = extractor.point_features(rasterized, out_dim=12)
    assert rasterized.image.shape == (3, 16, 16)
    assert features.shape == (40, 12)
    assert torch.isfinite(features).all()


def test_segmentation_dataset_external_features(tmp_path: Path):
    data_root = tmp_path / "blocks"
    feat_root = tmp_path / "dino"
    (data_root / "train").mkdir(parents=True)
    (feat_root / "train").mkdir(parents=True)
    block = _toy_block(10)
    block_path = data_root / "train" / "tile_000.pt"
    torch.save(block, block_path)
    torch.save({"dino_features": torch.ones(10, 7)}, feat_root / "train" / "tile_000.pt")
    ds = SegmentationBlockDataset(
        str(data_root / "train"),
        use_tw_input=True,
        external_feature_dir=str(feat_root),
    )
    item = ds[0]
    expected = 3 + 5 + 25 + 7
    assert item["features"].shape == (10, expected)


def test_geom_base_then_dino_external_channel_order(tmp_path: Path):
    data_root = tmp_path / "blocks"
    geom_root = tmp_path / "geom"
    dino_root = tmp_path / "dino"
    for root in (data_root, geom_root, dino_root):
        (root / "train").mkdir(parents=True)
    block = _toy_block(10)
    torch.save(block, data_root / "train" / "tile_000.pt")
    (geom_root / "feature_config.json").write_text(json.dumps({"feature_schema_sha256": "geom"}), encoding="utf-8")
    (dino_root / "feature_config.json").write_text(json.dumps({"feature_schema_sha256": "dino"}), encoding="utf-8")
    torch.save({"geom_features": torch.full((10, 5), 2.0), "feature_schema_sha256": "geom"}, geom_root / "train" / "tile_000.pt")
    torch.save({"dino_features": torch.full((10, 7), 3.0), "feature_schema_sha256": "dino"}, dino_root / "train" / "tile_000.pt")
    ds = SegmentationBlockDataset(
        str(data_root / "train"),
        use_tw_input=True,
        base_feature_dir=str(geom_root),
        external_feature_dir=str(dino_root),
    )
    features = ds[0]["features"]
    assert features.shape[1] == 3 + 5 + 25 + 5 + 7
    assert torch.all(features[:, -12:-7] == 2.0)
    assert torch.all(features[:, -7:] == 3.0)


def test_external_payload_schema_hash_mismatch_fails_closed(tmp_path: Path):
    data_root = tmp_path / "blocks"
    feat_root = tmp_path / "dino"
    (data_root / "train").mkdir(parents=True)
    (feat_root / "train").mkdir(parents=True)
    block_path = data_root / "train" / "tile_000.pt"
    torch.save(_toy_block(10), block_path)
    (feat_root / "feature_config.json").write_text(
        json.dumps({"feature_schema_sha256": "expected-hash"}), encoding="utf-8"
    )
    torch.save(
        {"dino_features": torch.ones(10, 7), "feature_schema_sha256": "tampered-hash"},
        feat_root / "train" / "tile_000.pt",
    )
    ds = SegmentationBlockDataset(str(data_root / "train"), external_feature_dir=str(feat_root))
    try:
        _ = ds[0]
    except ValueError as exc:
        assert "schema mismatch" in str(exc)
    else:
        raise AssertionError("tampered external feature payload must fail")


def test_external_manifest_schema_hash_mismatch_fails_closed(tmp_path: Path):
    evaluator = _load_script("scripts/20_evaluate_segmentation_model.py", "evaluate_schema_test")
    (tmp_path / "feature_config.json").write_text(
        json.dumps({"feature_schema_sha256": "tampered-hash"}), encoding="utf-8"
    )
    try:
        evaluator.validate_external_feature_manifest(tmp_path, "checkpoint-hash")
    except ValueError as exc:
        assert "differs from the training checkpoint" in str(exc)
    else:
        raise AssertionError("tampered external feature manifest must fail")


def test_stat_artifact_is_explicitly_not_dino_promotion_candidate():
    builder = _load_script("scripts/14_build_dino_features.py", "dino_builder_schema_test")
    extractor = DinoDenseExtractor(backend="stat", device="cpu")
    schema = builder.external_feature_schema(extractor, 16, "rgb", 0, 12, 13, True)
    assert schema["requested_backbone"] == "stat_features"
    assert schema["actual_backbone"] == "stat_features"
    assert schema["used_real_dino"] is False
    assert schema["promotion_eligible"] is False
    changed = builder.external_feature_schema(extractor, 16, "cir", 0, 12, 13, True)
    assert changed["sha256"] != schema["sha256"]


def test_auto_backend_never_substitutes_backbone_family():
    hf = DinoDenseExtractor.__new__(DinoDenseExtractor)
    hf.backend = "auto"
    hf.requested_model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
    hf.repo_dir = None
    assert hf._backend_candidates() == ["hf"]
    dino2 = DinoDenseExtractor.__new__(DinoDenseExtractor)
    dino2.backend = "auto"
    dino2.requested_model_name = "dinov2_vits14"
    dino2.repo_dir = None
    assert dino2._backend_candidates() == ["dinov2"]


def test_dinov2_loader_uses_the_declared_local_checkpoint(monkeypatch, tmp_path: Path):
    weights = tmp_path / "dinov2.pth"
    weights.write_bytes(b"declared checkpoint")
    captured = {}

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    def fake_load(repo_or_dir, name, **kwargs):
        captured.update({"repo_or_dir": repo_or_dir, "name": name, **kwargs})
        return DummyModel()

    monkeypatch.setattr(torch.hub, "load", fake_load)
    extractor = DinoDenseExtractor(
        backend="dinov2",
        model_name="dinov2_vits14",
        repo_dir=str(tmp_path / "repo"),
        weights=str(weights),
        device="cpu",
        dtype="float32",
    )
    assert extractor.uses_real_dino
    assert captured["weights"] == str(weights)
    assert captured["repo_or_dir"] == str(tmp_path / "repo")
    assert captured["source"] == "local"


def test_gated_external_segmentation_backward_smoke():
    torch.manual_seed(17)
    model = GatedExternalPointSegmentationNet(
        base_in_channels=33,
        external_in_channels=64,
        num_classes=6,
        hidden_dim=32,
        embed_dim=48,
    )
    features = torch.randn(2, 64, 97)
    labels = torch.randint(0, 6, (2, 64))
    mask = torch.ones(2, 64, dtype=torch.bool)
    logits = model(features, mask)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 6), labels.reshape(-1))
    loss.backward()
    assert logits.shape == (2, 64, 6)
    assert torch.isfinite(loss)
    assert any(param.grad is not None for param in model.external_adapter.parameters())


def test_pnoa_sentinel_channels_and_all_image_modes():
    block = _toy_block(4)
    block["coords"] = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 2.0], [1.0, 1.0, 3.0]]
    )
    # Stored order is intensity, red, green, blue, nir.
    block["features"] = torch.tensor(
        [
            [0.10, 0.11, 0.21, 0.31, 0.41],
            [0.20, 0.12, 0.22, 0.32, 0.42],
            [0.30, 0.13, 0.23, 0.33, 0.43],
            [0.40, 0.14, 0.24, 0.34, 0.44],
        ]
    )
    rasterized = make_multichannel_raster(block, grid_size=2, tw_channels=0)
    by_name = {name: rasterized.raster[idx] for idx, name in enumerate(rasterized.channel_names)}

    assert torch.allclose(by_name["red"].flatten(), block["features"][:, 1])
    assert torch.allclose(by_name["green"].flatten(), block["features"][:, 2])
    assert torch.allclose(by_name["blue"].flatten(), block["features"][:, 3])
    assert torch.allclose(by_name["intensity"].flatten(), block["features"][:, 0])
    assert torch.allclose(by_name["nir"].flatten(), block["features"][:, 4])

    expected = {
        "rgb": ("red", "green", "blue"),
        "cir": ("nir", "red", "green"),
        "height": ("z_norm", "density", "intensity"),
        "rgb_nir_height": ("red", "nir", "z_norm"),
        "nir_height_density": ("nir", "z_norm", "density"),
    }
    for mode, channels in expected.items():
        image = raster_to_image(rasterized.raster, rasterized.channel_names, mode=mode)
        for idx, name in enumerate(channels):
            assert torch.allclose(image[idx], _normalize_01(by_name[name]))


def test_dino_raster_fails_when_semantic_channels_are_missing():
    block = _toy_block(4)
    block["feature_names"] = ["intensity", "red", "green", "blue", "not_nir"]
    try:
        make_multichannel_raster(block, grid_size=2, tw_channels=0)
    except ValueError as exc:
        assert "missing required PNOA channels" in str(exc)
    else:
        raise AssertionError("missing NIR channel must fail")


def test_dino_raster_rejects_unknown_channel_schema():
    block = _toy_block(4)
    block["feature_schema_version"] = "unknown-v99"
    block["feature_schema"] = {"version": "unknown-v99", "names": block["feature_names"]}
    try:
        make_multichannel_raster(block, grid_size=2, tw_channels=0)
    except ValueError as exc:
        assert "Unknown PNOA feature schema" in str(exc)
    else:
        raise AssertionError("unknown channel schemas must fail closed")


def test_hf_dinov3_register_tokens_are_not_reshaped_as_patches():
    class Config:
        patch_size = 2
        num_register_tokens = 4

    class DummyModel(torch.nn.Module):
        config = Config()

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, **kwargs):
            # CLS=-1, registers=-2, four spatial patch values=0..3.
            values = torch.tensor([-1, -2, -2, -2, -2, 0, 1, 2, 3], dtype=torch.float32)
            return type("Output", (), {"last_hidden_state": values.view(1, 9, 1)})()

    extractor = DinoDenseExtractor.__new__(DinoDenseExtractor)
    extractor.real_backend = "hf"
    extractor.model = DummyModel()
    extractor.device = torch.device("cpu")
    extractor.normalize = "imagenet"
    extractor.model_name = "facebook/dinov3-vit-test"
    feature_map = extractor.image_feature_map(torch.zeros(3, 4, 4))
    assert feature_map.shape == (1, 2, 2)
    assert torch.equal(feature_map.flatten(), torch.arange(4, dtype=torch.float32))


def test_multiview_rasters_change_projection_and_preserve_point_mapping():
    block = _toy_block(40)
    top = make_multichannel_raster(block, grid_size=16, tw_channels=0, projection_view="top_xy")
    side = make_multichannel_raster(block, grid_size=16, tw_channels=0, projection_view="side_xz")
    assert top.point_cells.shape == side.point_cells.shape == (40, 2)
    assert not torch.equal(top.point_cells, side.point_cells)


def test_multiview_mean_keeps_parameter_matched_feature_dimension():
    builder = _load_script("scripts/14_build_dino_features.py", "dino_builder_multiview_schema_test")
    extractor = DinoDenseExtractor(backend="stat", device="cpu")
    views = ("top_xy", "side_xz", "side_yz")
    concat = builder.external_feature_schema(
        extractor, 16, "rgb", 0, 12, 13, False, projection_views=views, view_fusion="concat"
    )
    mean = builder.external_feature_schema(
        extractor, 16, "rgb", 0, 12, 13, False, projection_views=views, view_fusion="mean"
    )
    assert concat["feature_dim"] == 36
    assert mean["feature_dim"] == 12
    assert concat["sha256"] != mean["sha256"]


def test_dino_cache_schema_changes_with_split_or_checkpoint(tmp_path: Path):
    builder = _load_script("scripts/14_build_dino_features.py", "dino_builder_provenance_test")

    class Config:
        patch_size = 16
        hidden_size = 32
        num_register_tokens = 4

    class DummyModel(torch.nn.Module):
        config = Config()

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    weights_a = tmp_path / "a.safetensors"
    weights_b = tmp_path / "b.safetensors"
    weights_a.write_bytes(b"checkpoint-a")
    weights_b.write_bytes(b"checkpoint-b")
    extractor = DinoDenseExtractor.__new__(DinoDenseExtractor)
    extractor.backend = "hf"
    extractor.requested_model_name = "facebook/dinov3-test"
    extractor.model_name = "facebook/dinov3-test"
    extractor.real_backend = "hf"
    extractor.model = DummyModel()
    extractor.model_source = tmp_path
    extractor.hf_repo_id = "facebook/dinov3-test"
    extractor.hf_revision = "rev-a"
    extractor.normalize = "imagenet"
    extractor.weights = str(weights_a)
    schema_a = builder.external_feature_schema(extractor, 16, "rgb", 0, 8, 13, False, split_hash="split-a")
    schema_split = builder.external_feature_schema(extractor, 16, "rgb", 0, 8, 13, False, split_hash="split-b")
    extractor.weights = str(weights_b)
    schema_checkpoint = builder.external_feature_schema(extractor, 16, "rgb", 0, 8, 13, False, split_hash="split-a")
    assert schema_a["sha256"] != schema_split["sha256"]
    assert schema_a["sha256"] != schema_checkpoint["sha256"]
    assert schema_a["checkpoint_sha256"] != schema_checkpoint["checkpoint_sha256"]
