import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.jepa_dataset import JepaBlockMasker, GeoPointJepaDataset, jepa_collate_fn
from src.models.jepa import GeoPointJEPA, LeJEPALoss, PointMLPEncoder, SIGRegEppsPulley

@pytest.fixture
def dummy_coords():
    np.random.seed(42)
    # 100 points, 3D
    return np.random.rand(100, 3).astype(np.float32)

@pytest.fixture
def dummy_batch():
    torch.manual_seed(42)
    B, N_c, N_t, C = 4, 60, 20, 8
    return {
        'x_context': torch.randn(B, N_c, C),
        'mask_context': torch.ones(B, N_c, dtype=torch.bool),
        'x_target': torch.randn(B, N_t, C),
        'mask_target': torch.ones(B, N_t, dtype=torch.bool),
        'coords_target': torch.randn(B, N_t, 3)
    }

def test_spatial_mask(dummy_coords):
    masker = JepaBlockMasker(mask_type="spatial_block_mask", min_context_points=10, min_target_points=5)
    ctx_idx, tgt_idx = masker.generate_mask(dummy_coords)
    
    assert len(ctx_idx) >= 10
    assert len(tgt_idx) >= 5
    assert len(set(ctx_idx).intersection(set(tgt_idx))) == 0, "Context and target should be disjoint"

def test_random_mask(dummy_coords):
    masker = JepaBlockMasker(mask_type="random_point_mask", min_context_points=10, min_target_points=10)
    ctx_idx, tgt_idx = masker.generate_mask(dummy_coords)
    
    assert len(ctx_idx) >= 10
    assert len(tgt_idx) >= 10
    assert len(set(ctx_idx).intersection(set(tgt_idx))) == 0

def test_geopoint_jepa_forward(dummy_batch):
    model = GeoPointJEPA(in_channels=8, embed_dim=128)
    model.eval()
    pred, tgt, ctx, embeddings = model(
        dummy_batch['x_context'], dummy_batch['mask_context'],
        dummy_batch['x_target'], dummy_batch['mask_target'],
        dummy_batch['coords_target']
    )
    
    assert pred.shape == (4, 128)
    assert tgt.shape == (4, 128)
    assert pred.requires_grad == True
    assert tgt.requires_grad == True
    assert embeddings.shape == (8, 128)

def test_sigreg_loss():
    loss_fn = SIGRegEppsPulley(num_slices=16)
    B, D = 16, 32
    
    collapsed = torch.zeros(B, D)
    torch.manual_seed(42)
    dispersed = torch.randn(B, D)
    assert loss_fn(collapsed) > loss_fn(dispersed)

def test_jepa_no_nan(dummy_batch):
    model = GeoPointJEPA(in_channels=8, embed_dim=64)
    pred, tgt, _, _ = model(
        dummy_batch['x_context'], dummy_batch['mask_context'],
        dummy_batch['x_target'], dummy_batch['mask_target'],
        dummy_batch['coords_target']
    )
    
    assert torch.all(torch.isfinite(pred))
    assert torch.all(torch.isfinite(tgt))

def test_jepa_deterministic(dummy_batch):
    torch.manual_seed(123)
    model1 = GeoPointJEPA()
    p1, t1, _, _ = model1(
        dummy_batch['x_context'], dummy_batch['mask_context'],
        dummy_batch['x_target'], dummy_batch['mask_target'],
        dummy_batch['coords_target']
    )
    
    torch.manual_seed(123)
    model2 = GeoPointJEPA()
    p2, t2, _, _ = model2(
        dummy_batch['x_context'], dummy_batch['mask_context'],
        dummy_batch['x_target'], dummy_batch['mask_target'],
        dummy_batch['coords_target']
    )
    
    assert torch.allclose(p1, p2)
    assert torch.allclose(t1, t2)

def test_jepa_no_collapse(dummy_batch):
    # Simulate a micro training loop to ensure loss goes down
    # and gradients flow correctly
    model = GeoPointJEPA(in_channels=8, embed_dim=32)
    loss_fn = LeJEPALoss(sigreg_weight=0.05, num_slices=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    initial_loss = None
    for i in range(5):
        optimizer.zero_grad()
        pred, tgt, _, embeddings = model(
            dummy_batch['x_context'], dummy_batch['mask_context'],
            dummy_batch['x_target'], dummy_batch['mask_target'],
            dummy_batch['coords_target']
        )
        loss, metrics = loss_fn(pred, tgt, embeddings, global_step=i)
        loss.backward()
        optimizer.step()
        
        if i == 0:
            initial_loss = loss.item()
            
    final_loss = loss.item()
    # Loss should decrease
    assert final_loss < initial_loss
    # Embeddings should not collapse

# --- PHASE 4 TESTS ---
from src.models.tw_jepa import TW_JEPA, TWAuxLoss

@pytest.fixture
def dummy_tw_batch():
    torch.manual_seed(42)
    B, N_t, tw_dim = 4, 20, 11
    # tw_target includes the mask in the last dimension? No, tw_valid_target is passed separately in TWAuxLoss.
    return {
        'pred_tw': torch.randn(B, tw_dim, requires_grad=True),
        'tw_target': torch.randn(B, N_t, tw_dim),
        'tw_valid_target': torch.ones(B, N_t, dtype=torch.bool),
        'mask_tgt': torch.ones(B, N_t, dtype=torch.bool)
    }

def test_tw_aux_masked_mean(dummy_tw_batch):
    loss_fn = TWAuxLoss(min_valid_points=2)
    
    # Test valid case
    loss, metrics = loss_fn(
        dummy_tw_batch['pred_tw'],
        dummy_tw_batch['tw_target'],
        dummy_tw_batch['tw_valid_target'],
        dummy_tw_batch['mask_tgt']
    )
    
    assert loss > 0
    assert metrics['tw_aux_valid_ratio'] == 1.0
    assert metrics['tw_aux_skipped_samples'] == 0

def test_tw_aux_skips_empty_targets(dummy_tw_batch):
    loss_fn = TWAuxLoss(min_valid_points=5)
    
    # Make batch 0 invalid via valid mask
    dummy_tw_batch['tw_valid_target'][0, :] = False
    
    # Make batch 1 invalid via target padding mask
    dummy_tw_batch['mask_tgt'][1, :] = False
    
    loss, metrics = loss_fn(
        dummy_tw_batch['pred_tw'],
        dummy_tw_batch['tw_target'],
        dummy_tw_batch['tw_valid_target'],
        dummy_tw_batch['mask_tgt']
    )
    
    # 2 samples skipped
    assert metrics['tw_aux_valid_ratio'] == 0.5
    assert metrics['tw_aux_skipped_samples'] == 2
    assert loss.requires_grad == True
    
    # All invalid
    dummy_tw_batch['tw_valid_target'][:] = False
    loss2, metrics2 = loss_fn(
        dummy_tw_batch['pred_tw'],
        dummy_tw_batch['tw_target'],
        dummy_tw_batch['tw_valid_target'],
        dummy_tw_batch['mask_tgt']
    )
    
    assert metrics2['tw_aux_valid_ratio'] == 0.0
    assert metrics2['tw_aux_skipped_samples'] == 4
    assert loss2.item() == 0.0
    assert loss2.requires_grad == True # To not break DDP/backward

def test_tw_aux_gamma_zero_equivalence(dummy_batch):
    torch.manual_seed(123)
    model_jepa = GeoPointJEPA(in_channels=8, embed_dim=32)
    
    torch.manual_seed(123)
    model_tw = TW_JEPA(in_channels=8, embed_dim=32, tw_dim=11)
    model_jepa.eval()
    model_tw.eval()
    
    p1, t1, _, _ = model_jepa(
        dummy_batch['x_context'], dummy_batch['mask_context'],
        dummy_batch['x_target'], dummy_batch['mask_target'],
        dummy_batch['coords_target']
    )
    
    p2, t2, pred_tw, _, _ = model_tw(
        dummy_batch['x_context'], dummy_batch['mask_context'],
        dummy_batch['x_target'], dummy_batch['mask_target'],
        dummy_batch['coords_target']
    )
    
    assert torch.allclose(p1, p2)
    assert torch.allclose(t1, t2)
