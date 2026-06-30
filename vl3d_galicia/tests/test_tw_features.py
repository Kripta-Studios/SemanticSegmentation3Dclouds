import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features.taubin_weingarten import compute_tw_lite_features
from src.features.taubin_weingarten import TW_FEATURE_DIM, TW_FEATURE_NAMES

@pytest.fixture
def random_coords():
    np.random.seed(42)
    return np.random.rand(500, 3).astype(np.float32)

def test_tw_shapes(random_coords):
    features = compute_tw_lite_features(random_coords, k_neighbors=16)
    assert features.shape == (500, TW_FEATURE_DIM), f"Expected TW feature shape, got {features.shape}"

def test_tw_no_nan(random_coords):
    features = compute_tw_lite_features(random_coords, k_neighbors=16)
    assert np.all(np.isfinite(features)), "Features contain NaNs or Infs"

def test_tw_deterministic(random_coords):
    feat1 = compute_tw_lite_features(random_coords, k_neighbors=16)
    feat2 = compute_tw_lite_features(random_coords, k_neighbors=16)
    assert np.allclose(feat1, feat2), "Features computation is not deterministic"

def test_tw_valid_mask(random_coords):
    features = compute_tw_lite_features(random_coords, k_neighbors=16)
    valid_mask = features[:, TW_FEATURE_NAMES.index("tw_valid")]
    
    assert np.all(np.isin(valid_mask, [0.0, 1.0])), "Valid mask should be binary (0.0 or 1.0)"
    assert np.sum(valid_mask) > 0, "Expected at least some points to be valid"

def test_degenerate_handling():
    # Only 3 points (less than min_neighbors=5 by default)
    coords = np.array([[0,0,0], [1,0,0], [0,1,0]], dtype=np.float32)
    features = compute_tw_lite_features(coords, min_neighbors=5)
    
    assert features.shape == (3, TW_FEATURE_DIM)
    assert np.all(features == 0), "Degenerate small point cloud should return all zeros"

def test_coplanar_handling():
    # Exactly coplanar points on a grid to trigger potentially ill-conditioned matrices
    x, y = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
    coords = np.column_stack([x.ravel(), y.ravel(), np.zeros_like(x.ravel())])
    
    # Due to ridge_lambda, this should not crash, but it might give tw_valid=1 or 0 safely
    features = compute_tw_lite_features(coords, k_neighbors=10)
    assert np.all(np.isfinite(features)), "Coplanar points caused NaNs"
