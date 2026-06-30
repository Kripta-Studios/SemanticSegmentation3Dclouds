import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features.taubin_weingarten import compute_tw_lite_features
from src.features.taubin_weingarten import TW_FEATURE_NAMES

KMIN = TW_FEATURE_NAMES.index("kappa_min")
KMAX = TW_FEATURE_NAMES.index("kappa_max")
VALID = TW_FEATURE_NAMES.index("tw_valid")

def test_tw_plane_curvature_zero():
    # Create a grid of points on Z=0
    x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    coords = np.column_stack([x.ravel(), y.ravel(), np.zeros_like(x.ravel())])
    
    # Add a bit of jitter to avoid exactly singular matrices, though ridge handles it
    coords += np.random.normal(0, 1e-6, coords.shape)
    
    features = compute_tw_lite_features(coords, k_neighbors=16, ridge_lambda=1e-6)
    
    # We expect kappa_min and kappa_max to be very close to 0
    k_min = features[:, KMIN]
    k_max = features[:, KMAX]
    
    assert np.allclose(k_min, 0.0, atol=1e-3), "Plane should have zero minimum curvature"
    assert np.allclose(k_max, 0.0, atol=1e-3), "Plane should have zero maximum curvature"

def test_tw_sphere_curvature():
    # Sphere of radius 2
    R = 2.0
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    U, V = np.meshgrid(u, v)
    X = R * np.cos(U) * np.sin(V)
    Y = R * np.sin(U) * np.sin(V)
    Z = R * np.cos(V)
    coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Add very small noise
    coords += np.random.normal(0, 1e-5, coords.shape)
    
    # Compute features
    features = compute_tw_lite_features(coords, k_neighbors=32, ridge_lambda=1e-6)
    
    k_min = features[:, KMIN]
    k_max = features[:, KMAX]
    valid = features[:, VALID].astype(bool)
    
    assert np.any(valid), "Expected some valid points on sphere"
    
    # Exclude poles where sampling might be degenerate
    k_min_valid = k_min[valid]
    k_max_valid = k_max[valid]
    
    # The normal can point inwards or outwards, so curvatures are either both 1/R or both -1/R
    # Taking absolute value should yield 1/R
    k_min_abs = np.abs(k_min_valid)
    k_max_abs = np.abs(k_max_valid)
    
    expected_curvature = 1.0 / R
    
    # Median absolute error should be small
    assert np.median(np.abs(k_min_abs - expected_curvature)) < 0.1, "Sphere k_min deviates too much from 1/R"
    assert np.median(np.abs(k_max_abs - expected_curvature)) < 0.1, "Sphere k_max deviates too much from 1/R"

def test_tw_paraboloid_curvature():
    # Paraboloid z = 1.0 * x^2 + 2.0 * y^2
    # At origin (0,0), curvatures should be 2.0 and 4.0
    x, y = np.meshgrid(np.linspace(-0.5, 0.5, 20), np.linspace(-0.5, 0.5, 20))
    z = 1.0 * x**2 + 2.0 * y**2
    coords = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    features = compute_tw_lite_features(coords, k_neighbors=24, ridge_lambda=1e-6)
    
    # Find the point closest to the origin
    dists = np.linalg.norm(coords, axis=1)
    origin_idx = np.argmin(dists)
    
    k_min = features[origin_idx, KMIN]
    k_max = features[origin_idx, KMAX]
    valid = features[origin_idx, VALID]
    
    assert valid == 1.0, "Origin point should be valid"
    
    # Depending on normal direction, they are either (-4, -2) or (2, 4)
    # So absolute values should be 2 and 4
    k1, k2 = np.abs(k_min), np.abs(k_max)
    
    assert np.isclose(min(k1, k2), 2.0, atol=0.2), f"Expected min curvature ~2.0, got {min(k1, k2)}"
    assert np.isclose(max(k1, k2), 4.0, atol=0.2), f"Expected max curvature ~4.0, got {max(k1, k2)}"
