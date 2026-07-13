from __future__ import annotations

import copy

import pytest

from src.data.geographic_split import (
    SPLIT_MANIFEST_SCHEMA_VERSION,
    bbox_distance_m,
    compute_split_hash,
    galicia_campaign_north_val_split,
    validate_split_manifest,
)


def _manifest() -> dict:
    rows = [
        {"tile_id": "train", "campaign": "GAL-W-2015", "split": "train", "col_path": "t-col", "cir_path": "t-cir", "col_sha256": "1", "cir_sha256": "2", "bounds": [0, 0, 10, 10], "crs": "EPSG:25829", "col_points": 1, "cir_points": 1},
        {"tile_id": "val", "campaign": "GAL-W-2015", "split": "val", "col_path": "v-col", "cir_path": "v-cir", "col_sha256": "3", "cir_sha256": "4", "bounds": [20, 0, 30, 10], "crs": "EPSG:25829", "col_points": 1, "cir_points": 1},
        {"tile_id": "test", "campaign": "GAL-E-2016", "split": "test", "col_path": "e-col", "cir_path": "e-cir", "col_sha256": "5", "cir_sha256": "6", "bounds": [40, 0, 50, 10], "crs": "EPSG:25829", "col_points": 1, "cir_points": 1},
    ]
    manifest = {"schema_version": SPLIT_MANIFEST_SCHEMA_VERSION, "policy": "test", "seed": 7, "tiles": rows}
    manifest["split_hash"] = compute_split_hash(manifest)
    return manifest


def test_official_geographic_policy_is_label_blind_and_grouped():
    assert galicia_campaign_north_val_split("PNOA_2016_GAL_E_620-4760_ORT", "GAL-E-2016") == "test"
    assert galicia_campaign_north_val_split("PNOA_2015_GAL-W_584-4810_ORT", "GAL-W-2015") == "val"
    assert galicia_campaign_north_val_split("PNOA_2015_GAL-W_500-4802_ORT", "GAL-W-2015") == "excluded_buffer"
    assert galicia_campaign_north_val_split("PNOA_2015_GAL-W_500-4700_ORT", "GAL-W-2015") == "train"


def test_split_manifest_proves_tile_laz_and_bounds_disjoint():
    audit = validate_split_manifest(_manifest())
    assert audit["cross_split_bounds_overlap_count"] == 0
    assert audit["tile_intersections"]["train_test_source_laz"] == []
    assert audit["min_distances"]["train_val_min_distance_m"] == pytest.approx(10.0)


def test_split_manifest_rejects_shared_laz_and_changed_hash():
    manifest = _manifest()
    manifest["tiles"][2]["col_path"] = manifest["tiles"][0]["col_path"]
    manifest["split_hash"] = compute_split_hash(manifest)
    with pytest.raises(ValueError, match="leakage"):
        validate_split_manifest(manifest)

    changed = copy.deepcopy(_manifest())
    changed["tiles"][0]["col_sha256"] = "changed"
    with pytest.raises(ValueError, match="Split hash mismatch"):
        validate_split_manifest(changed)


def test_bbox_distance_treats_touching_as_zero_not_overlap():
    assert bbox_distance_m([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0

