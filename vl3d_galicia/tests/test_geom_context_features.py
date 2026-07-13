from __future__ import annotations

import torch

import pytest

from src.features.geom_context import (
    GeomContextConfig,
    build_geom_context_features,
    geom_context_schema,
    validate_feature_schema,
)


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


def test_56_and_73_schemas_are_deterministic_and_incompatible():
    cfg_56 = GeomContextConfig(include_metric_height=False)
    cfg_73 = GeomContextConfig(include_metric_height=True)
    schema_56_a = geom_context_schema(cfg_56, tw_feature_count=25)
    schema_56_b = geom_context_schema(cfg_56, tw_feature_count=25)
    schema_73 = geom_context_schema(cfg_73, tw_feature_count=25)

    assert schema_56_a.dimension == 56
    assert schema_73.dimension == 73
    assert schema_56_a.names == schema_56_b.names
    assert schema_56_a.sha256 == schema_56_b.sha256
    assert schema_56_a.sha256 != schema_73.sha256
    with pytest.raises(ValueError, match="56-dimensional"):
        validate_feature_schema(schema_56_a.sha256, schema_73.sha256, context="checkpoint")
