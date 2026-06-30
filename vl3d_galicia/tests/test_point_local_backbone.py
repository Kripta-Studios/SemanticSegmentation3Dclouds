import pytest
import torch
from src.models.point_local_backbone import PointLocalBackbone, get_knn_indices
from src.eval.segmentation_metrics import compute_segmentation_metrics

def test_point_local_backbone_forward():
    B, N, C = 2, 1024, 8
    model = PointLocalBackbone(in_channels=C, out_classes=7, k=16)
    
    x = torch.randn(B, N, C)
    coords = torch.randn(B, N, 3)
    
    out = model(x, coords)
    
    assert out.shape == (B, N, 7), "Output shape should be [B, N, num_classes]"
    assert not torch.isnan(out).any(), "Forward pass produced NaNs"

def test_point_local_backbone_knn():
    B, N = 2, 256
    coords = torch.randn(B, N, 3)
    k = 16
    
    idx = get_knn_indices(coords, k=k)
    
    assert idx.shape == (B, N, k), "KNN indices shape incorrect"
    assert (idx >= 0).all() and (idx < N).all(), "KNN indices out of bounds"

def test_jepa_embedding_injection():
    B, N, C = 2, 512, 8
    D = 128
    # We pass in_channels as C + D because the model expects concatenated features
    model = PointLocalBackbone(in_channels=C + D, out_classes=7, k=8)
    
    x = torch.randn(B, N, C)
    coords = torch.randn(B, N, 3)
    jepa_emb = torch.randn(B, D)
    
    out = model(x, coords, jepa_emb=jepa_emb)
    
    assert out.shape == (B, N, 7), "Output shape with JEPA embedding incorrect"
    assert not torch.isnan(out).any(), "JEPA embedding injection produced NaNs"

def test_phase5_metrics():
    B, N = 2, 100
    preds = torch.randint(0, 7, (B*N,))
    targets = torch.randint(0, 7, (B*N,))
    
    # Inject some ignore indices
    targets[0:10] = 6
    
    metrics = compute_segmentation_metrics(preds, targets, num_classes=7, ignore_index=6)
    
    assert 'macro_f1' in metrics
    assert 'class_f1' in metrics
    assert 6 not in metrics['class_f1'], "Ignore index should not be in class metrics"
