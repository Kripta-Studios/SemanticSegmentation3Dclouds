import os
import torch
import glob
import pytest

def test_prepared_tiles_format():
    # Find any .pt file in the processed directory
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
    pt_files = glob.glob(os.path.join(root, "**", "*.pt"), recursive=True)
    
    if not pt_files:
        pytest.skip("No processed .pt files found. Run 01_prepare_tiles.py first.")
        
    sample_file = pt_files[0]
    data = torch.load(sample_file, weights_only=False)
    
    assert 'coords' in data
    assert 'features' in data
    assert 'labels' in data
    assert 'reliable_mask' in data or 'mask' in data
    
    coords = data['coords']
    features = data['features']
    labels = data['labels']
    
    assert coords.ndim == 2
    assert coords.shape[1] == 3
    assert coords.shape[0] == features.shape[0]
    assert coords.shape[0] == labels.shape[0]
    
    # Assert labels are within our mapped range [0, 6]
    assert torch.all(labels >= 0)
    assert torch.all(labels <= 6)
