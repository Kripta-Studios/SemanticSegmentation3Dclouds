# SemanticSegmentation3Dclouds

Workspace for 3D point-cloud semantic segmentation experiments.

## Layout

- `vl3d_galicia/`: Galicia LiDAR semantic segmentation pipeline, training scripts, configs, docs, tests, and presentation material.
- `twalg_scripts/`: Taubin-Weingarten algorithm scripts and experiment configs.
- `docs/PLAN.md`: project plan and working notes.
- `references/papers/`: research papers used as project references.

## Local-only artifacts

Large run artifacts, local datasets, virtual environments, caches, logs, trained weights, and generated reports are intentionally ignored. The previous `data/processed`, `outputs`, `reports`, `models`, `logs`, `.venv`, pytest temp folders, and Python cache folders were removed before the initial root commit.

The original nested repositories were:

- `twalg_scripts`: `https://github.com/albertoesmp/twalg_scripts.git` at `ff5392717c5664e9ddd1bce80c333834932d05ab`
- `vl3d_galicia`: `https://github.com/albertoesmp/vl3d_galicia.git` at `559e2d9ac868a0197c1c38b6d03f71e684eaf896`
