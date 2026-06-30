from __future__ import annotations

import argparse
import sys
from pathlib import Path

import laspy
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.geo_inference import CLASS_COLORS, block_files, load_segmentation_model, predict_block


def write_laz(path: Path, coords: np.ndarray, labels: np.ndarray, pred_class: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = coords.min(axis=0)
    las = laspy.LasData(header)
    las.x = coords[:, 0]
    las.y = coords[:, 1]
    las.z = coords[:, 2]
    las.classification = labels.astype(np.uint8)
    if "pred_class" not in {dim.name for dim in las.point_format.extra_dimensions}:
        las.add_extra_dim(laspy.ExtraBytesParams(name="pred_class", type=np.uint8))
    las.pred_class = pred_class.astype(np.uint8)
    colors = np.zeros((coords.shape[0], 3), dtype=np.uint16)
    for cls, color in CLASS_COLORS.items():
        colors[pred_class == cls] = np.array(color, dtype=np.uint16) * 257
    las.red = colors[:, 0]
    las.green = colors[:, 1]
    las.blue = colors[:, 2]
    las.write(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export colored LAZ predictions from processed test blocks.")
    parser.add_argument("--data", default="data/processed/galicia_blocks_pilot_tw")
    parser.add_argument("--split", default="test")
    parser.add_argument("--baseline-checkpoint", default="outputs/pilot/baseline/best_model.pt")
    parser.add_argument("--jepa-checkpoint", default="outputs/pilot/jepa_full_finetune/best_model.pt")
    parser.add_argument("--baseline-run-config", default="outputs/pilot/baseline/run_config.json")
    parser.add_argument("--jepa-run-config", default="outputs/pilot/jepa_full_finetune/run_config.json")
    parser.add_argument("--out", default="outputs/pilot_demo/laz_exports")
    parser.add_argument("--max-tiles", type=int, default=3)
    args = parser.parse_args()

    baseline_model, baseline_cfg = load_segmentation_model(args.baseline_checkpoint, args.data, split=args.split, run_config=args.baseline_run_config)
    jepa_model, jepa_cfg = load_segmentation_model(args.jepa_checkpoint, args.data, split=args.split, run_config=args.jepa_run_config)
    files = block_files(args.data, args.split)
    out = Path(args.out)
    exported = 0
    seen = set()
    for path in files:
        data = torch.load(path, weights_only=False, map_location="cpu")
        tile_id = data.get("tile_id", path.stem)
        if tile_id in seen:
            continue
        seen.add(tile_id)
        exported += 1
        prefix = f"tile_{exported:03d}"
        coords = data.get("global_coords", data["coords"]).numpy()
        labels = data["labels"].numpy()
        baseline_pred = predict_block(baseline_model, data, baseline_cfg["use_tw_input"], baseline_cfg["device"])
        jepa_pred = predict_block(jepa_model, data, jepa_cfg["use_tw_input"], jepa_cfg["device"])
        write_laz(out / f"{prefix}_ground_truth_colored.laz", coords, labels, labels)
        write_laz(out / f"{prefix}_baseline_colored.laz", coords, labels, baseline_pred)
        write_laz(out / f"{prefix}_jepa_colored.laz", coords, labels, jepa_pred)
        if exported >= args.max_tiles:
            break
    print(f"Exported {exported} LAZ triplets to {out}")


if __name__ == "__main__":
    main()
