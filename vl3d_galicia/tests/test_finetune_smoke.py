import torch

from src.models.segmentation.heads import PointSegmentationNet


def test_segmentation_backward_smoke():
    torch.manual_seed(7)
    model = PointSegmentationNet(in_channels=8, num_classes=7, hidden_dim=32, embed_dim=48)
    features = torch.randn(2, 128, 8)
    labels = torch.randint(0, 6, (2, 128))
    mask = torch.ones(2, 128, dtype=torch.bool)
    logits = model(features, mask)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 7), labels.reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    assert any(param.grad is not None for param in model.parameters())
