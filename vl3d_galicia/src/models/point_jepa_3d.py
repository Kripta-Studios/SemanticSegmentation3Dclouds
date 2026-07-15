import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def batched_index_select(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    b, n, c = x.shape
    flat = x.reshape(b * n, c)
    offset_shape = (b,) + (1,) * (idx.ndim - 1)
    offsets = torch.arange(b, device=x.device).reshape(offset_shape) * n
    gathered = flat[(idx + offsets).reshape(-1)]
    return gathered.reshape(*idx.shape, c)

class MiniPointNet(nn.Module):
    """Encoder for a single 3D patch (local point cloud)."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels // 2),
            nn.LayerNorm(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 2, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_patches, K, C)
        B, M, K, C = x.shape
        x = x.reshape(B * M * K, C)
        feat = self.mlp(x)
        feat = feat.reshape(B, M, K, -1)
        return torch.max(feat, dim=2)[0]  # (B, M, out_channels)


class PointJEPA3D(nn.Module):
    """
    True Point-JEPA implementation with 3D patches.
    - FPS + kNN for patch extraction
    - MiniPointNet patch encoder
    - Transformer Context Encoder
    - Transformer Predictor
    - EMA Teacher
    """
    def __init__(
        self, 
        in_channels: int = 8, 
        embed_dim: int = 256, 
        num_patches: int = 256, 
        points_per_patch: int = 32,
        encoder_depth: int = 4,
        predictor_depth: int = 2,
        ema_decay: float = 0.996
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.points_per_patch = points_per_patch
        self.ema_decay = ema_decay
        
        # Patch Extractor / Encoder
        self.patch_encoder = MiniPointNet(in_channels, embed_dim)
        
        # Positional Embedding (based on patch centers)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Transformer Context Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=embed_dim*4, batch_first=True, norm_first=True)
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)
        
        # Transformer Predictor
        predictor_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=embed_dim*4, batch_first=True, norm_first=True)
        self.predictor = nn.TransformerEncoder(predictor_layer, num_layers=predictor_depth)
        
        self.predictor_embed = nn.Linear(embed_dim, embed_dim)
        
        # Target Encoder (EMA)
        self.target_patch_encoder = copy.deepcopy(self.patch_encoder)
        self.target_context_encoder = copy.deepcopy(self.context_encoder)
        
        for param in self.target_patch_encoder.parameters():
            param.requires_grad_(False)
        for param in self.target_context_encoder.parameters():
            param.requires_grad_(False)
            
    @torch.no_grad()
    def update_target_encoder(self) -> None:
        for target_param, source_param in zip(self.target_patch_encoder.parameters(), self.patch_encoder.parameters()):
            target_param.data.mul_(self.ema_decay).add_(source_param.data, alpha=1.0 - self.ema_decay)
        for target_param, source_param in zip(self.target_context_encoder.parameters(), self.context_encoder.parameters()):
            target_param.data.mul_(self.ema_decay).add_(source_param.data, alpha=1.0 - self.ema_decay)

    def extract_patches(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, N, C), mask: (B, N)
        B, N, C = x.shape
        device = x.device
        coords = x[..., :3]
        
        # Farthest Point Sampling for patch centers (simplified as uniform sampling for speed if N is large)
        # For a true FPS, we should use a custom op or Open3D. Here we use an approximation or random sampling.
        # Let's use linspace for deterministic reproducible centers per block.
        idx = torch.linspace(0, N - 1, self.num_patches, device=device).round().long()
        anchor_coords = coords[:, idx]  # (B, M, 3)
        
        # kNN for patch points
        dist = torch.cdist(anchor_coords.float(), coords.float())
        dist = dist.masked_fill(~mask[:, None, :], 1e6)
        k = min(self.points_per_patch, N)
        nn_idx = torch.topk(dist, k=k, dim=-1, largest=False).indices  # (B, M, K)
        
        patch_points = batched_index_select(x, nn_idx)  # (B, M, K, C)
        patch_coords = patch_points[..., :3]
        # Normalize patch coordinates relative to anchor
        patch_points[..., :3] = patch_coords - anchor_coords.unsqueeze(2)
        
        return patch_points, anchor_coords

    def forward(self, x: torch.Tensor, mask: torch.Tensor, context_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, N, C)
        mask: (B, N) valid points mask
        context_mask: (B, M) bool mask indicating which patches are context (True) and which are target (False)
        """
        B = x.shape[0]
        patch_points, anchor_coords = self.extract_patches(x, mask)
        
        # Generate random mask if not provided (e.g. 50% context)
        if context_mask is None:
            context_mask = torch.rand(B, self.num_patches, device=x.device) > 0.5
            # Ensure at least one context and one target
            context_mask[:, 0] = True
            context_mask[:, 1] = False
            
        pos_emb = self.pos_embed(anchor_coords)  # (B, M, D)
        
        # Encode context patches
        ctx_patches = self.patch_encoder(patch_points) # (B, M, D)
        ctx_tokens = ctx_patches + pos_emb
        
        # Mask out target patches for the context encoder
        # We can either drop them or zero them. Transformers usually work better if we drop or use padding mask.
        # Let's zero them out for simplicity, but a better way is to only pass context tokens.
        ctx_tokens = ctx_tokens.masked_fill(~context_mask.unsqueeze(-1), 0.0)
        ctx_encoded = self.context_encoder(ctx_tokens) # (B, M, D)
        
        # Encode target patches with EMA teacher
        with torch.no_grad():
            tgt_patches = self.target_patch_encoder(patch_points)
            tgt_tokens = tgt_patches + pos_emb
            tgt_encoded = self.target_context_encoder(tgt_tokens)
            
        # Predict target tokens from context tokens and target positions
        pred_input = self.predictor_embed(ctx_encoded)
        # Inject positional embedding for targets
        pred_input = pred_input + pos_emb
        predicted_tgt = self.predictor(pred_input)
        
        # Extract only the target patches for loss computation
        target_mask = ~context_mask
        predicted_targets = predicted_tgt[target_mask]
        true_targets = tgt_encoded[target_mask]
        
        # Return predicted, true, and full context embeddings
        return predicted_targets, true_targets, ctx_encoded
