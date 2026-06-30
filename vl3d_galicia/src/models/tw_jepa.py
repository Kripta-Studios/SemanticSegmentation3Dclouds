from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.jepa import GeoPointJEPA


class TW_JEPA(nn.Module):
    def __init__(self, in_channels: int = 8, hidden_dim: int = 192, embed_dim: int = 256, tw_dim: int = 24):
        super().__init__()
        self.jepa = GeoPointJEPA(in_channels=in_channels, hidden_dim=hidden_dim, embed_dim=embed_dim)
        self.tw_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, tw_dim),
        )

    def forward(self, x_ctx, mask_ctx, x_tgt, mask_tgt, coords_tgt):
        pred_emb, tgt_emb, ctx_emb, encoder_embeddings = self.jepa(x_ctx, mask_ctx, x_tgt, mask_tgt, coords_tgt)
        pred_tw = self.tw_head(pred_emb)
        return pred_emb, tgt_emb, pred_tw, ctx_emb, encoder_embeddings


class TWAuxLoss(nn.Module):
    def __init__(self, min_valid_points: int = 5, tw_dim: int | None = None):
        super().__init__()
        self.min_valid_points = min_valid_points
        self.tw_dim = tw_dim

    def forward(self, pred_tw, tw_target, tw_valid_target, mask_tgt):
        tw_dim = self.tw_dim or pred_tw.shape[1]
        valid = mask_tgt & tw_valid_target
        losses = []
        for i in range(pred_tw.shape[0]):
            if int(valid[i].sum()) < self.min_valid_points:
                continue
            target_mean = tw_target[i, valid[i], :tw_dim].mean(dim=0)
            losses.append(F.smooth_l1_loss(pred_tw[i], target_mean))
        if not losses:
            loss = pred_tw.sum() * 0.0
            ratio = 0.0
        else:
            loss = torch.stack(losses).mean()
            ratio = len(losses) / pred_tw.shape[0]
        return loss, {
            "tw_aux_loss": float(loss.detach().cpu()),
            "tw_aux_valid_ratio": ratio,
            "tw_aux_skipped_samples": pred_tw.shape[0] - len(losses),
        }

