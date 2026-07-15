from __future__ import annotations

import torch
import torch.nn as nn

from src.models.jepa import PointEncoder, masked_max, masked_mean
from src.models.point_jepa_3d import PointJEPA3D


class PointJepa3DSegmentationNet(nn.Module):
    """Segmentation head taking PointJEPA3D as encoder."""
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 6,
        embed_dim: int = 256,
        dropout: float = 0.2,
        num_patches: int = 256,
        points_per_patch: int = 32,
    ):
        super().__init__()
        self.encoder_frozen = False
        self.encoder = PointJEPA3D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_patches=num_patches,
            points_per_patch=points_per_patch
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            
        B, N, _ = x.shape
        
        # We need per-point features, but PointJEPA3D gives us patch features (ctx_encoded).
        # We can interpolate patch features to all points using inverse distance weighting.
        patch_points, anchor_coords = self.encoder.extract_patches(x, mask)
        
        # Encode all patches as context
        context_mask = torch.ones(B, self.encoder.num_patches, dtype=torch.bool, device=x.device)
        pos_emb = self.encoder.pos_embed(anchor_coords)
        ctx_patches = self.encoder.patch_encoder(patch_points)
        ctx_encoded = self.encoder.context_encoder(ctx_patches + pos_emb)
        
        # Interpolate from anchor_coords (M patches) to all coords (N points)
        coords = x[..., :3]
        dist = torch.cdist(coords.float(), anchor_coords.float())
        # k=3 interpolation
        interp_dist, interp_idx = torch.topk(dist, k=3, dim=-1, largest=False)
        weights = 1.0 / interp_dist.clamp_min(1e-4)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        
        # ctx_encoded: (B, M, D) -> gather to (B, N, 3, D)
        B, M, D = ctx_encoded.shape
        flat_ctx = ctx_encoded.reshape(B * M, D)
        offsets = (torch.arange(B, device=x.device) * M).unsqueeze(-1).unsqueeze(-1)
        gathered = flat_ctx[(interp_idx + offsets).reshape(-1)].reshape(B, N, 3, D)
        
        propagated = (gathered * weights.unsqueeze(-1)).sum(dim=2)
        logits = self.head(propagated)
        return logits

    def load_jepa_encoder(self, checkpoint: dict, strict: bool = False) -> None:
        state = checkpoint.get("model", checkpoint)
        encoder_state = {
            key: value for key, value in state.items()
        }
        self.encoder.load_state_dict(encoder_state, strict=strict)

    def freeze_encoder(self) -> None:
        self.encoder_frozen = True
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self


class GatedExternalPointJepa3DSegmentationNet(nn.Module):
    """PointJEPA3D encoder with a gated residual adapter for raster/DINO features."""
    def __init__(
        self,
        base_in_channels: int,
        external_in_channels: int,
        num_classes: int = 6,
        embed_dim: int = 256,
        dropout: float = 0.2,
        num_patches: int = 256,
        points_per_patch: int = 32,
    ):
        super().__init__()
        if external_in_channels <= 0:
            raise ValueError("external_in_channels must be positive for gated fusion")
        self.base_in_channels = int(base_in_channels)
        self.external_in_channels = int(external_in_channels)
        self.encoder_frozen = False
        
        self.encoder = PointJEPA3D(
            in_channels=base_in_channels,
            embed_dim=embed_dim,
            num_patches=num_patches,
            points_per_patch=points_per_patch
        )
        
        # We need to project external features into embed_dim to fuse with PointJEPA patch features
        self.external_adapter = nn.Sequential(
            nn.Linear(external_in_channels, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(embed_dim)
        self._init_conservative_gate()
        
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def _init_conservative_gate(self) -> None:
        final_linear = self.fusion_gate[-2]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            nn.init.constant_(final_linear.bias, -2.0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            
        base_x = x[..., : self.base_in_channels]
        external_x = x[..., self.base_in_channels : self.base_in_channels + self.external_in_channels]
            
        B, N, _ = base_x.shape
        
        # 1. PointJEPA patch extraction and context encoding
        patch_points, anchor_coords = self.encoder.extract_patches(base_x, mask)
        context_mask = torch.ones(B, self.encoder.num_patches, dtype=torch.bool, device=x.device)
        pos_emb = self.encoder.pos_embed(anchor_coords)
        ctx_patches = self.encoder.patch_encoder(patch_points)
        ctx_encoded = self.encoder.context_encoder(ctx_patches + pos_emb)
        
        # 2. Interpolate JEPA patch features (M) to all points (N)
        coords = base_x[..., :3]
        dist = torch.cdist(coords.float(), anchor_coords.float())
        interp_dist, interp_idx = torch.topk(dist, k=3, dim=-1, largest=False)
        weights = 1.0 / interp_dist.clamp_min(1e-4)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        
        B, M, D = ctx_encoded.shape
        flat_ctx = ctx_encoded.reshape(B * M, D)
        offsets = (torch.arange(B, device=x.device) * M).unsqueeze(-1).unsqueeze(-1)
        gathered = flat_ctx[(interp_idx + offsets).reshape(-1)].reshape(B, N, 3, D)
        
        base_feat = (gathered * weights.unsqueeze(-1)).sum(dim=2)  # (B, N, D)
        
        # 3. Fuse with External features
        external_feat = self.external_adapter(external_x)  # (B, N, D)
        gate = self.fusion_gate(torch.cat([base_feat, external_feat], dim=-1))
        fused_feat = self.fusion_norm(base_feat + gate * external_feat)
        
        logits = self.head(fused_feat)
        return logits

    def load_jepa_encoder(self, checkpoint: dict, strict: bool = False) -> None:
        state = checkpoint.get("model", checkpoint)
        encoder_state = {key: value for key, value in state.items()}
        self.encoder.load_state_dict(encoder_state, strict=strict)

    def freeze_encoder(self) -> None:
        self.encoder_frozen = True
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self


class PointSegmentationNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 6,
        hidden_dim: int = 192,
        embed_dim: int = 256,
        dropout: float = 0.2,
        probe_type: str = "mlp",
    ):
        super().__init__()
        if probe_type not in {"linear", "mlp"}:
            raise ValueError(f"Unknown probe_type: {probe_type}")
        self.probe_type = probe_type
        self.encoder_frozen = False
        self.encoder = PointEncoder(in_channels, hidden_dim=hidden_dim, embed_dim=embed_dim, dropout=dropout)
        if probe_type == "linear":
            self.head = nn.Linear(hidden_dim + embed_dim, num_classes)
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim + embed_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes),
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        point_feat = self.encoder.point_features(x)
        pooled = torch.cat([masked_mean(point_feat, mask), masked_max(point_feat, mask)], dim=1)
        block_emb = self.encoder.projector(pooled)
        block_emb = block_emb.unsqueeze(1).expand(-1, x.shape[1], -1)
        logits = self.head(torch.cat([point_feat, block_emb], dim=-1))
        return logits

    def load_jepa_encoder(self, checkpoint: dict, strict: bool = False) -> None:
        state = checkpoint.get("model", checkpoint)
        if any(key.startswith("jepa.encoder.") for key in state):
            prefix = "jepa.encoder."
        else:
            prefix = "encoder."
        encoder_state = {
            key.replace(prefix, "", 1): value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not encoder_state:
            raise KeyError("No JEPA encoder weights found in checkpoint")
        self.encoder.load_state_dict(encoder_state, strict=strict)

    def freeze_encoder(self) -> None:
        self.encoder_frozen = True
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def parameter_summary(self) -> dict:
        total = sum(param.numel() for param in self.parameters())
        trainable = sum(param.numel() for param in self.parameters() if param.requires_grad)
        encoder_total = sum(param.numel() for param in self.encoder.parameters())
        encoder_trainable = sum(param.numel() for param in self.encoder.parameters() if param.requires_grad)
        return {
            "total_params": int(total),
            "trainable_params": int(trainable),
            "encoder_params": int(encoder_total),
            "encoder_trainable_params": int(encoder_trainable),
            "head_params": int(total - encoder_total),
            "head_trainable_params": int(trainable - encoder_trainable),
        }


class GatedExternalPointSegmentationNet(nn.Module):
    """Point/TW encoder with a gated residual adapter for raster/DINO features."""

    def __init__(
        self,
        base_in_channels: int,
        external_in_channels: int,
        num_classes: int = 6,
        hidden_dim: int = 192,
        embed_dim: int = 256,
        dropout: float = 0.2,
        probe_type: str = "mlp",
    ):
        super().__init__()
        if external_in_channels <= 0:
            raise ValueError("external_in_channels must be positive for gated fusion")
        if probe_type not in {"linear", "mlp"}:
            raise ValueError(f"Unknown probe_type: {probe_type}")
        self.base_in_channels = int(base_in_channels)
        self.external_in_channels = int(external_in_channels)
        self.probe_type = probe_type
        self.encoder_frozen = False
        self.encoder = PointEncoder(base_in_channels, hidden_dim=hidden_dim, embed_dim=embed_dim, dropout=dropout)
        self.external_adapter = nn.Sequential(
            nn.Linear(external_in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self._init_conservative_gate()
        if probe_type == "linear":
            self.head = nn.Linear(hidden_dim + embed_dim, num_classes)
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim + embed_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes),
            )

    def _init_conservative_gate(self) -> None:
        final_linear = self.fusion_gate[-2]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            nn.init.constant_(final_linear.bias, -2.0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.shape[-1] < self.base_in_channels + self.external_in_channels:
            raise ValueError(
                "Input does not contain the expected base and external features: "
                f"got {x.shape[-1]}, expected {self.base_in_channels + self.external_in_channels}"
            )
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        base_x = x[..., : self.base_in_channels]
        external_x = x[..., self.base_in_channels : self.base_in_channels + self.external_in_channels]
        base_feat = self.encoder.point_features(base_x)
        external_feat = self.external_adapter(external_x)
        gate = self.fusion_gate(torch.cat([base_feat, external_feat], dim=-1))
        point_feat = self.fusion_norm(base_feat + gate * external_feat)
        pooled = torch.cat([masked_mean(point_feat, mask), masked_max(point_feat, mask)], dim=1)
        block_emb = self.encoder.projector(pooled)
        block_emb = block_emb.unsqueeze(1).expand(-1, x.shape[1], -1)
        logits = self.head(torch.cat([point_feat, block_emb], dim=-1))
        return logits

    def load_jepa_encoder(self, checkpoint: dict, strict: bool = False) -> None:
        state = checkpoint.get("model", checkpoint)
        if any(key.startswith("jepa.encoder.") for key in state):
            prefix = "jepa.encoder."
        else:
            prefix = "encoder."
        encoder_state = {
            key.replace(prefix, "", 1): value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not encoder_state:
            raise KeyError("No JEPA encoder weights found in checkpoint")
        self.encoder.load_state_dict(encoder_state, strict=strict)

    def freeze_encoder(self) -> None:
        self.encoder_frozen = True
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def parameter_summary(self) -> dict:
        total = sum(param.numel() for param in self.parameters())
        trainable = sum(param.numel() for param in self.parameters() if param.requires_grad)
        encoder_total = sum(param.numel() for param in self.encoder.parameters())
        encoder_trainable = sum(param.numel() for param in self.encoder.parameters() if param.requires_grad)
        return {
            "total_params": int(total),
            "trainable_params": int(trainable),
            "encoder_params": int(encoder_total),
            "encoder_trainable_params": int(encoder_trainable),
            "head_params": int(total - encoder_total),
            "head_trainable_params": int(trainable - encoder_trainable),
        }
