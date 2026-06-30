from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_common import infer_in_channels
from src.data.segmentation_dataset import SegmentationBlockDataset, segmentation_collate_fn
from src.models.segmentation.heads import PointSegmentationNet
from src.training.segmentation_trainer import SegmentationTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a segmentation checkpoint on a prepared split.")
    parser.add_argument("--data", default="data/processed/galicia_blocks_tw")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="outputs/eval_metrics.json")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-tw-input", action="store_true")
    args = parser.parse_args()

    split_dir = str(Path(args.data) / args.split)
    in_channels = infer_in_channels(split_dir, use_tw_input=args.use_tw_input)
    model = PointSegmentationNet(in_channels=in_channels)
    state = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    model.load_state_dict(state.get("model", state), strict=False)
    ds = SegmentationBlockDataset(split_dir, use_tw_input=args.use_tw_input)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=segmentation_collate_fn)
    trainer = SegmentationTrainer(model, loader, loader, device="cuda" if torch.cuda.is_available() else "cpu", epochs=0)
    metrics = trainer.evaluate(loader)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

