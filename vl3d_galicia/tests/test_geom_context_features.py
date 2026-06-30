from __future__ import annotations

import torch

from src.features.geom_context import GeomContextConfig, build_geom_context_features


def test_geom_context_features_are_label_free_and_finite():
    n = 24
    block = {
        "coords": torch.stack(
            [
                torch.linspace(-5, 5, n),
                torch.linspace(5, -5, n),
                torch.cat([torch.zeros(8), torch.ones(8) * 2.0, torch.ones(8) * 7.0]),
            ],
            dim=1,
        ),
        "features": torch.rand(n, 5),
        "labels": torch.arange(n) % 6,
        "tw_features": torch.rand(n, 25),
    }
    features_a, names = build_geom_context_features(block, GeomContextConfig(cell_sizes=(2.5, 5.0)))
    block["labels"] = torch.zeros(n, dtype=torch.long)
    features_b, _ = build_geom_context_features(block, GeomContextConfig(cell_sizes=(2.5, 5.0)))
    assert features_a.shape[0] == n
    assert features_a.shape[1] == len(names)
    assert torch.isfinite(features_a).all()
    assert torch.allclose(features_a, features_b)
