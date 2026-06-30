from __future__ import annotations

from pathlib import Path

import torch

from src.data.segmentation_dataset import SegmentationBlockDataset
from src.features.raster_dino import DinoDenseExtractor, make_multichannel_raster
from src.models.segmentation.heads import GatedExternalPointSegmentationNet


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


def test_gated_external_segmentation_backward_smoke():
    torch.manual_seed(17)
    model = GatedExternalPointSegmentationNet(
        base_in_channels=33,
        external_in_channels=64,
        num_classes=7,
        hidden_dim=32,
        embed_dim=48,
    )
    features = torch.randn(2, 64, 97)
    labels = torch.randint(0, 6, (2, 64))
    mask = torch.ones(2, 64, dtype=torch.bool)
    logits = model(features, mask)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 7), labels.reshape(-1))
    loss.backward()
    assert logits.shape == (2, 64, 7)
    assert torch.isfinite(loss)
    assert any(param.grad is not None for param in model.external_adapter.parameters())
