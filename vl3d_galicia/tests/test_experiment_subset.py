from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "27_materialize_experiment_subset.py"
    spec = importlib.util.spec_from_file_location("experiment_subset", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_subset_selection_is_deterministic_and_has_no_label_input(tmp_path: Path):
    module = _module()
    files = []
    for index in range(20):
        path = tmp_path / f"tile_block_{index:05d}.pt"
        path.write_bytes(bytes([index]))
        files.append(path)
    selected_a = module.selected_files(files, 5, seed=20260714)
    selected_b = module.selected_files(list(reversed(files)), 5, seed=20260714)
    assert [path.name for path in selected_a] == [path.name for path in selected_b]
