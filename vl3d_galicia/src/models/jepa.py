from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(x.dtype)
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values


class PointEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 192, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.point_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def point_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.point_mlp(x)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        feats = self.point_features(x)
        pooled = torch.cat([masked_mean(feats, mask), masked_max(feats, mask)], dim=1)
        return self.projector(pooled)


class JepaPredictor(nn.Module):
    def __init__(self, embed_dim: int = 256, pos_dim: int = 6, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + pos_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, context_emb: torch.Tensor, target_summary: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([context_emb, target_summary], dim=1))


def masked_coord_summary(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mean = masked_mean(coords, mask)
    centered = (coords - mean.unsqueeze(1)).masked_fill(~mask.unsqueeze(-1), 0.0)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(coords.dtype)
    std = torch.sqrt((centered.square().sum(dim=1) / denom).clamp_min(1e-8))
    return torch.cat([mean, std], dim=1)


class SIGRegEppsPulley(nn.Module):
    def __init__(
        self,
        num_slices: int = 256,
        num_knots: int = 17,
        t_min: float = -5.0,
        t_max: float = 5.0,
        seed: int = 13,
    ):
        super().__init__()
        self.num_slices = num_slices
        self.num_knots = num_knots
        self.t_min = t_min
        self.t_max = t_max
        self.seed = seed

    def forward(self, embeddings: torch.Tensor, global_step: int | None = None) -> torch.Tensor:
        x = embeddings.reshape(-1, embeddings.shape[-1])
        if x.shape[0] < 2:
            return x.new_tensor(0.0)
        generator = torch.Generator(device=x.device)
        generator.manual_seed(self.seed if global_step is None else self.seed + int(global_step))
        directions = torch.randn(x.shape[1], self.num_slices, generator=generator, device=x.device, dtype=x.dtype)
        directions = F.normalize(directions, dim=0)
        t = torch.linspace(self.t_min, self.t_max, self.num_knots, device=x.device, dtype=x.dtype)
        target_cf = torch.exp(-0.5 * t.square())
        projected = (x @ directions).unsqueeze(-1) * t
        real = torch.cos(projected).mean(dim=0)
        imag = torch.sin(projected).mean(dim=0)
        err = ((real - target_cf).square() + imag.square()) * target_cf
        stat = torch.trapezoid(err, t, dim=1) * x.shape[0]
        return stat.mean()


class LeJEPALoss(nn.Module):
    def __init__(self, sigreg_weight: float = 0.1, num_slices: int = 256, pred_loss: str = "smooth_l1"):
        super().__init__()
        self.sigreg_weight = sigreg_weight
        self.sigreg = SIGRegEppsPulley(num_slices=num_slices)
        self.pred_loss = pred_loss

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        encoder_embeddings: torch.Tensor,
        global_step: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if self.pred_loss == "mse":
            loss_pred = F.mse_loss(pred, target)
        else:
            loss_pred = F.smooth_l1_loss(pred, target)
        loss_sigreg = self.sigreg(encoder_embeddings, global_step=global_step)
        loss = (1.0 - self.sigreg_weight) * loss_pred + self.sigreg_weight * loss_sigreg
        with torch.no_grad():
            std = encoder_embeddings.reshape(-1, encoder_embeddings.shape[-1]).std(dim=0).mean()
            cosine = F.cosine_similarity(pred, target, dim=1).mean()
        return loss, {
            "loss_pred": float(loss_pred.detach().cpu()),
            "loss_sigreg": float(loss_sigreg.detach().cpu()),
            "embedding_std_mean": float(std.detach().cpu()),
            "cosine_sim": float(cosine.detach().cpu()),
        }


class GeoPointJEPA(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        hidden_dim: int = 192,
        embed_dim: int = 256,
        dropout: float = 0.1,
        target_mode: str = "shared",
        ema_decay: float = 0.996,
    ):
        super().__init__()
        self.encoder = PointEncoder(in_channels, hidden_dim=hidden_dim, embed_dim=embed_dim, dropout=dropout)
        self.predictor = JepaPredictor(embed_dim=embed_dim, pos_dim=6, hidden_dim=embed_dim, dropout=dropout)
        self.target_mode = target_mode
        self.ema_decay = ema_decay
        self.target_encoder = copy.deepcopy(self.encoder) if target_mode == "ema" else None
        if self.target_encoder is not None:
            for param in self.target_encoder.parameters():
                param.requires_grad_(False)

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        if self.target_encoder is None:
            return
        for target_param, source_param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target_param.data.mul_(self.ema_decay).add_(source_param.data, alpha=1.0 - self.ema_decay)

    def forward(
        self,
        x_ctx: torch.Tensor,
        mask_ctx: torch.Tensor,
        x_tgt: torch.Tensor,
        mask_tgt: torch.Tensor,
        coords_tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx_emb = self.encoder(x_ctx, mask_ctx)
        if self.target_encoder is None:
            tgt_emb = self.encoder(x_tgt, mask_tgt)
        else:
            with torch.no_grad():
                tgt_emb = self.target_encoder(x_tgt, mask_tgt)
        summary = masked_coord_summary(coords_tgt, mask_tgt)
        pred = self.predictor(ctx_emb, summary)
        encoder_embeddings = torch.cat([ctx_emb, tgt_emb], dim=0)
        return pred, tgt_emb, ctx_emb, encoder_embeddings


SIGRegLoss = LeJEPALoss
PointMLPEncoder = PointEncoder

