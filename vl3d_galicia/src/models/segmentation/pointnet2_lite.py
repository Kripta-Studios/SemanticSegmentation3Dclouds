from __future__ import annotations

import torch
import torch.nn as nn

from src.models.jepa import masked_mean


def batched_index_select(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Select points/features with batched indices.

    x: [B, N, C]
    idx: [B, M, K] or [B, M]
    """
    b, n, c = x.shape
    flat = x.reshape(b * n, c)
    offset_shape = (b,) + (1,) * (idx.ndim - 1)
    offsets = torch.arange(b, device=x.device).reshape(offset_shape) * n
    gathered = flat[(idx + offsets).reshape(-1)]
    return gathered.reshape(*idx.shape, c)


class PointNet2LiteSegmentationNet(nn.Module):
    """A lightweight PointNet++/KPConv-style local aggregation baseline.

    The model uses fixed anchor points inside each block, learns local features
    around those anchors, and interpolates the anchor features back to all
    points. It avoids the O(N^2) full point graph used by the older EdgeConv
    prototype while still giving the network explicit local 3D neighborhoods.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 7,
        hidden_dim: int = 160,
        embed_dim: int = 256,
        dropout: float = 0.2,
        anchor_count: int = 384,
        neighbors: int = 16,
        interp_neighbors: int = 3,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.embed_dim = int(embed_dim)
        self.anchor_count = int(anchor_count)
        self.neighbors = int(neighbors)
        self.interp_neighbors = int(interp_neighbors)
        self.encoder = nn.ModuleDict(
            {
                "point_mlp": nn.Sequential(
                    nn.Linear(in_channels, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                ),
                "local_mlp": nn.Sequential(
                    nn.Linear(hidden_dim + 3, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.GELU(),
                ),
                "anchor_mlp": nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.GELU(),
                ),
            }
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim + embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _anchor_indices(self, n_points: int, device: torch.device) -> torch.Tensor:
        m = min(self.anchor_count, n_points)
        if m == n_points:
            return torch.arange(n_points, device=device)
        return torch.linspace(0, n_points - 1, m, device=device).round().long()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        coords = x[..., :3]
        point_feat = self.encoder["point_mlp"](x)
        b, n, _ = point_feat.shape
        anchor_idx = self._anchor_indices(n, x.device)
        anchor_coords = coords[:, anchor_idx]
        anchor_valid = mask[:, anchor_idx]
        m = anchor_idx.numel()
        k = min(self.neighbors, n)

        # Anchor-to-point neighborhoods.
        dist_anchor_point = torch.cdist(anchor_coords.float(), coords.float())
        dist_anchor_point = dist_anchor_point.masked_fill(~mask[:, None, :], 1e6)
        nn_idx = torch.topk(dist_anchor_point, k=k, dim=-1, largest=False).indices
        nn_feat = batched_index_select(point_feat, nn_idx)
        nn_coords = batched_index_select(coords, nn_idx)
        rel_coords = nn_coords - anchor_coords[:, :, None, :]
        edge = torch.cat([nn_feat, rel_coords], dim=-1)
        local = self.encoder["local_mlp"](edge)
        anchor_feat = local.max(dim=2).values
        anchor_feat = self.encoder["anchor_mlp"](anchor_feat)
        anchor_feat = anchor_feat * anchor_valid.unsqueeze(-1).to(anchor_feat.dtype)

        # Interpolate anchor features back to points.
        ik = min(self.interp_neighbors, m)
        dist_point_anchor = torch.cdist(coords.float(), anchor_coords.float())
        dist_point_anchor = dist_point_anchor.masked_fill(~anchor_valid[:, None, :], 1e6)
        interp_dist, interp_idx = torch.topk(dist_point_anchor, k=ik, dim=-1, largest=False)
        interp_feat = batched_index_select(anchor_feat, interp_idx)
        weights = 1.0 / interp_dist.clamp_min(1e-4)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        propagated = (interp_feat * weights.unsqueeze(-1)).sum(dim=2)
        propagated = propagated.masked_fill(~mask.unsqueeze(-1), 0.0)

        global_anchor = masked_mean(anchor_feat, anchor_valid)
        global_anchor = global_anchor[:, None, :].expand(b, n, -1)
        logits = self.head(torch.cat([point_feat, propagated, global_anchor], dim=-1))
        return logits

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
