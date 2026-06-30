from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_common import infer_channel_layout, save_json
from src.data.classes import IGNORE_INDEX
from src.data.segmentation_dataset import SegmentationBlockDataset, segmentation_collate_fn
from src.eval.segmentation_metrics import compute_segmentation_metrics
from src.models.segmentation.heads import GatedExternalPointSegmentationNet, PointSegmentationNet
from src.models.segmentation.pointnet2_lite import PointNet2LiteSegmentationNet


CLASS_NAMES = {
    0: "ground",
    1: "low_vegetation",
    2: "medium_vegetation",
    3: "high_vegetation",
    4: "building",
    5: "water",
}


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_model(config: dict, channel_layout: dict) -> torch.nn.Module:
    model_type = config.get("model_type", "point_mlp")
    hidden_dim = int(config.get("hidden_dim", 192))
    embed_dim = int(config.get("embed_dim", 256))
    dropout = float(config.get("dropout", 0.2))
    probe_type = config.get("probe_type", "mlp")
    if model_type == "pointnet2_lite":
        return PointNet2LiteSegmentationNet(
            in_channels=channel_layout["in_channels"],
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            dropout=dropout,
            anchor_count=int(config.get("anchor_count", 384)),
            neighbors=int(config.get("local_neighbors", 16)),
            interp_neighbors=int(config.get("interp_neighbors", 3)),
        )
    if config.get("fusion_type") == "gated":
        return GatedExternalPointSegmentationNet(
            base_in_channels=channel_layout["base_in_channels"],
            external_in_channels=channel_layout["external_in_channels"],
            probe_type=probe_type,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            dropout=dropout,
        )
    return PointSegmentationNet(
        in_channels=channel_layout["in_channels"],
        probe_type=probe_type,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        dropout=dropout,
    )


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> dict:
    model.eval()
    preds_all = []
    labels_all = []
    for batch in tqdm(loader, desc="External eval"):
        features = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        logits = model(features, mask)
        preds_all.append(logits.argmax(dim=-1).reshape(-1).cpu())
        labels_all.append(labels.reshape(-1).cpu())
    preds = torch.cat(preds_all)
    labels = torch.cat(labels_all)
    return compute_segmentation_metrics(preds, labels, num_classes=7, ignore_index=IGNORE_INDEX)


def write_reports(out: Path, metrics: dict, run_config: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "run_config.json", run_config)
    save_json(out / "metrics.json", metrics)
    save_json(out / "test_metrics.json", metrics)
    rows = []
    for cls, name in CLASS_NAMES.items():
        precision = metrics.get("class_precision", {})
        recall = metrics.get("class_recall", {})
        f1 = metrics.get("class_f1", {})
        iou = metrics.get("class_iou", {})
        support = metrics.get("class_support", {})
        rows.append(
            {
                "class_id": cls,
                "class_name": name,
                "precision": float(precision.get(str(cls), precision.get(cls, 0.0))),
                "recall": float(recall.get(str(cls), recall.get(cls, 0.0))),
                "f1": float(f1.get(str(cls), f1.get(cls, 0.0))),
                "iou": float(iou.get(str(cls), iou.get(cls, 0.0))),
                "support": int(support.get(str(cls), support.get(cls, 0))),
            }
        )
    pd.DataFrame(rows).to_csv(out / "per_class_metrics.csv", index=False)
    cm = np.asarray(metrics["confusion_matrix"])
    np.savetxt(out / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model on a separate data root.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-tw-input", action="store_true")
    parser.add_argument("--external-feature-dir", default="")
    parser.add_argument("--external-feature-key", default="geom_features")
    parser.add_argument("--coordinate-normalization", choices=["", "none", "xy_unit_z_robust"], default="")
    parser.add_argument("--spectral-normalization", choices=["", "none", "block_robust"], default="")
    parser.add_argument("--external-feature-normalization", choices=["", "none", "block_robust", "spectral_block_robust"], default="")
    parser.add_argument("--max-blocks", type=int, default=0)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    train_config = load_json(model_dir / "run_config.json") or load_json(model_dir / "config.json")
    if not train_config:
        raise SystemExit(f"No run_config.json/config.json found in {model_dir}")
    use_tw = bool(args.use_tw_input or train_config.get("use_tw_input", False))
    external_dir = args.external_feature_dir or train_config.get("external_feature_dir", "")
    external_key = args.external_feature_key or train_config.get("external_feature_key", "geom_features")
    coordinate_normalization = args.coordinate_normalization or train_config.get("coordinate_normalization", "none")
    spectral_normalization = args.spectral_normalization or train_config.get("spectral_normalization", "none")
    external_feature_normalization = args.external_feature_normalization or train_config.get("external_feature_normalization", "none")
    split_dir = str(Path(args.data) / args.split)
    channel_layout = infer_channel_layout(
        split_dir,
        use_tw_input=use_tw,
        external_feature_dir=external_dir,
        external_feature_key=external_key,
    )
    model = build_model(train_config, channel_layout)
    checkpoint = Path(args.checkpoint) if args.checkpoint else model_dir / "best_model.pt"
    state = torch.load(checkpoint, weights_only=False, map_location="cpu")
    model.load_state_dict(state.get("model", state), strict=True)
    dataset = SegmentationBlockDataset(
        split_dir,
        max_blocks=args.max_blocks,
        use_tw_input=use_tw,
        external_feature_dir=external_dir or None,
        external_feature_key=external_key,
        coordinate_normalization=coordinate_normalization,
        spectral_normalization=spectral_normalization,
        external_feature_normalization=external_feature_normalization,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=segmentation_collate_fn,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    metrics = evaluate(model, loader, device)
    run_config = {
        "model_name": Path(args.out).name,
        "experiment_type": "external_holdout_eval",
        "source_model_dir": str(model_dir),
        "checkpoint": str(checkpoint),
        "data_root": args.data,
        "split": args.split,
        "use_tw_input": use_tw,
        "external_feature_dir": external_dir,
        "external_feature_key": external_key,
        "coordinate_normalization": coordinate_normalization,
        "spectral_normalization": spectral_normalization,
        "external_feature_normalization": external_feature_normalization,
        "max_blocks": int(args.max_blocks),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        **channel_layout,
        "train_config": train_config,
    }
    write_reports(Path(args.out), metrics, run_config)
    print(json.dumps({k: metrics[k] for k in ["OA", "macro_f1", "macro_iou", "weighted_f1"]}, indent=2))


if __name__ == "__main__":
    main()
