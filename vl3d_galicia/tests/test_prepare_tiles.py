import os
import torch
import glob
import pytest
from src.data.blocks import BLOCK_SCHEMA_VERSION

@pytest.mark.data
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


@pytest.mark.integration
@pytest.mark.data
def test_prepare_tiles_integration_smoke(tmp_path):
    from src.data.pnoa import find_tile_pairs
    from src.data.blocks import make_blocks_from_pair

    # Find raw files
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "pnoa_galicia"))
    pairs = find_tile_pairs(raw_dir)
    assert len(pairs) > 0, "No raw tile pairs found in data/raw/pnoa_galicia"

    # Select the first pair (which is COL and CIR)
    pair = pairs[0]

    # Process block from this pair to tmp_path
    stats = make_blocks_from_pair(
        pair.col_path,
        pair.cir_path,
        tmp_path,
        tile_size=50.0,
        stride=50.0,
        points_per_block=100,  # small count for quick run
        min_points=10,
        val_ratio=0.1,
        test_ratio=0.1,
        split_mode="mixed",
        seed=42,
        skip_existing=False,
    )

    # Check stats
    assert stats["blocks"] > 0
    assert stats["tile_id"] == pair.tile_id

    # Check that blocks are written to the correct split subfolder
    split = stats["split"]
    written_files = glob.glob(os.path.join(str(tmp_path), split, "*.pt"))
    assert len(written_files) > 0

    # Load one written block and check metadata
    data = torch.load(written_files[0], weights_only=False)
    assert 'coords' in data
    assert 'features' in data
    assert 'labels' in data
    assert 'reliable_mask' in data
    assert 'tile_id' in data
    assert data['tile_id'] == pair.tile_id

    # Coordinates and labels shapes
    coords = data['coords']
    features = data['features']
    labels = data['labels']
    assert coords.ndim == 2
    assert coords.shape[1] == 3
    assert coords.shape[0] == features.shape[0]
    assert coords.shape[0] == labels.shape[0]
    assert torch.all(labels >= 0)
    assert torch.all(labels <= 6)
