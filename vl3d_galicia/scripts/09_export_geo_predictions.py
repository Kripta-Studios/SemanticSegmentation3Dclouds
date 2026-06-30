from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.geo_inference import block_files, error_rgb, labels_to_rgb, load_segmentation_model, predict_block


def scatter_rgb(coords: np.ndarray, rgb: np.ndarray, out: Path, title: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 7))
    plt.scatter(coords[:, 0], coords[:, 1], c=rgb / 255.0, s=1)
    plt.title(title)
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def input_rgb(data: dict) -> np.ndarray:
    feats = data["features"].numpy()
    if feats.shape[1] >= 4:
        rgb = np.clip(feats[:, 1:4], 0.0, 1.0)
        return (rgb * 255).astype(np.uint8)
    z = data["coords"][:, 2].numpy()
    zn = (z - z.min()) / max(float(z.max() - z.min()), 1e-6)
    gray = (zn * 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Geo-JEPA visual prediction maps for test blocks.")
    parser.add_argument("--data", default="data/processed/galicia_blocks_pilot_tw")
    parser.add_argument("--split", default="test")
    parser.add_argument("--baseline-checkpoint", default="outputs/pilot/baseline/best_model.pt")
    parser.add_argument("--jepa-checkpoint", default="outputs/pilot/jepa_full_finetune/best_model.pt")
    parser.add_argument("--baseline-run-config", default="outputs/pilot/baseline/run_config.json")
    parser.add_argument("--jepa-run-config", default="outputs/pilot/jepa_full_finetune/run_config.json")
    parser.add_argument("--out", default="outputs/pilot_demo/maps")
    parser.add_argument("--max-tiles", type=int, default=5)
    args = parser.parse_args()

    baseline_model, baseline_cfg = load_segmentation_model(args.baseline_checkpoint, args.data, split=args.split, run_config=args.baseline_run_config)
    jepa_model, jepa_cfg = load_segmentation_model(args.jepa_checkpoint, args.data, split=args.split, run_config=args.jepa_run_config)
    files = block_files(args.data, args.split)
    if not files:
        raise SystemExit(f"No blocks found in {Path(args.data) / args.split}")

    seen = set()
    exported = 0
    out = Path(args.out)
    for path in files:
        data = torch.load(path, weights_only=False, map_location="cpu")
        tile_id = data.get("tile_id", path.stem)
        if tile_id in seen:
            continue
        seen.add(tile_id)
        exported += 1
        prefix = f"tile_{exported:03d}"
        coords = data.get("global_coords", data["coords"]).numpy()
        target = data["labels"].numpy()
        baseline_pred = predict_block(baseline_model, data, baseline_cfg["use_tw_input"], baseline_cfg["device"])
        jepa_pred = predict_block(jepa_model, data, jepa_cfg["use_tw_input"], jepa_cfg["device"])
        baseline_err = (baseline_pred != target) & (target != 6)
        jepa_err = (jepa_pred != target) & (target != 6)
        diff_rgb = np.zeros((target.shape[0], 3), dtype=np.uint8)
        diff_rgb[baseline_err & ~jepa_err] = (0, 180, 0)
        diff_rgb[jepa_err & ~baseline_err] = (220, 30, 30)
        diff_rgb[baseline_err & jepa_err] = (240, 180, 0)
        diff_rgb[~baseline_err & ~jepa_err & (target != 6)] = (210, 210, 210)

        scatter_rgb(coords, input_rgb(data), out / f"{prefix}_input.png", f"{tile_id} input")
        scatter_rgb(coords, labels_to_rgb(target), out / f"{prefix}_ground_truth.png", f"{tile_id} ground truth")
        scatter_rgb(coords, labels_to_rgb(baseline_pred), out / f"{prefix}_baseline_prediction.png", f"{tile_id} baseline")
        scatter_rgb(coords, labels_to_rgb(jepa_pred), out / f"{prefix}_jepa_prediction.png", f"{tile_id} JEPA")
        scatter_rgb(coords, error_rgb(baseline_pred, target), out / f"{prefix}_baseline_error.png", f"{tile_id} baseline error")
        scatter_rgb(coords, error_rgb(jepa_pred, target), out / f"{prefix}_jepa_error.png", f"{tile_id} JEPA error")
        scatter_rgb(coords, diff_rgb, out / f"{prefix}_error_difference.png", f"{tile_id} error difference")
        if exported >= args.max_tiles:
            break
    print(f"Exported {exported} tile map sets to {out}")


if __name__ == "__main__":
    main()
