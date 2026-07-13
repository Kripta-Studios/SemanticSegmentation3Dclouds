from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _selector_module():
    path = ROOT / "scripts/23_select_external_holdout_tiles.py"
    spec = importlib.util.spec_from_file_location("external_holdout_selector_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_label_blind_holdout_invariant_to_class_count_mutation():
    selector = _selector_module()
    rows = []
    for idx in range(12):
        row = {"tile_id": f"tile-{idx:02d}", "reliable_points": 100, "point_count": 110}
        row.update({f"class_{cls}": idx + cls for cls in range(7)})
        rows.append(row)
    mutated = []
    for idx, row in enumerate(rows):
        changed = dict(row)
        changed["reliable_points"] = 10_000_000 - idx
        changed["point_count"] = 20_000_000 - idx
        for cls in range(7):
            changed[f"class_{cls}"] = (idx + 1) * (cls + 3) * 99991
        mutated.append(changed)
    original_ids = [row["tile_id"] for row in selector.select_tiles_label_blind(rows, 5, seed=20260713)]
    mutated_ids = [row["tile_id"] for row in selector.select_tiles_label_blind(mutated, 5, seed=20260713)]
    assert original_ids == mutated_ids
