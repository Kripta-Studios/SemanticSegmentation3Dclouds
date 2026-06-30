from __future__ import annotations

from pathlib import Path

import torch

from src.data.segmentation_dataset import select_block_files


def test_class_balanced_block_selection_includes_rare_class(tmp_path: Path):
    files = []
    for idx in range(12):
        path = tmp_path / f"block_{idx:03d}.pt"
        labels = torch.zeros(64, dtype=torch.long)
        if idx >= 10:
            labels[:16] = 4
        torch.save({"labels": labels}, path)
        files.append(str(path))
    selected = select_block_files(
        files,
        max_blocks=4,
        mode="class_balanced",
        seed=7,
        class_boost={4: 8.0},
    )
    assert len(selected) == 4
    assert any("block_010" in path or "block_011" in path for path in selected)
