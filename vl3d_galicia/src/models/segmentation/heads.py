from __future__ import annotations

import torch
import torch.nn as nn

from src.models.jepa import PointEncoder, masked_max, masked_mean


class PointSegmentationNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 7,
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
        num_classes: int = 7,
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
