from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_common import balanced_block_sampler, class_weights_from_blocks, infer_in_channels, load_config, model_parameter_summary, parse_class_boost, save_json, segmentation_optimizer_groups, set_global_seed
from src.data.segmentation_dataset import SegmentationBlockDataset, segmentation_collate_fn
from src.models.segmentation.heads import PointSegmentationNet
from src.training.segmentation_trainer import SegmentationTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune segmentation from a pretrained GeoPoint JEPA encoder.")
    parser.add_argument("--config")
    parser.add_argument("--data", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--use-tw-input", action="store_true")
    parser.add_argument("--max-train-blocks", type=int)
    parser.add_argument("--max-val-blocks", type=int)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--encoder-lr-scale", type=float, default=1.0)
    parser.add_argument("--probe-type", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weight-mode", choices=["none", "effective", "inverse", "inverse_sqrt", "median_frequency"], default="inverse_sqrt")
    parser.add_argument("--max-class-weight", type=float, default=20.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--loss-type", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--sampler-alpha", type=float, default=1.0)
    parser.add_argument("--sampler-max-weight", type=float, default=12.0)
    parser.add_argument("--sampler-class-boost", default="")
    parser.add_argument("--save-test-arrays", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_root = args.data or cfg.get("data", {}).get("blocks_dir", "data/processed/galicia_blocks_tw")
    out = Path(args.out or cfg.get("output", {}).get("dir", "outputs/jepa_finetune"))
    epochs = args.epochs or cfg.get("training", {}).get("epochs", 60)
    batch_size = args.batch_size or cfg.get("training", {}).get("batch_size", 24)
    workers = args.num_workers if args.num_workers is not None else cfg.get("training", {}).get("num_workers", 4)
    lr = args.lr or cfg.get("training", {}).get("learning_rate", 3e-4)
    use_tw = args.use_tw_input or cfg.get("data", {}).get("use_tw_input", False)
    max_train = args.max_train_blocks or cfg.get("data", {}).get("max_train_blocks", 0)
    max_val = args.max_val_blocks or cfg.get("data", {}).get("max_val_blocks", 0)
    generator = set_global_seed(args.seed)

    complete_path = out / "training_complete.json"
    best_model_path = out / "best_model.pt"
    checkpoint_path = out / "last_checkpoint.pt"
    if complete_path.exists() and best_model_path.exists() and not args.no_resume:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if int(complete.get("epochs_completed", 0)) >= epochs or bool(complete.get("early_stopped", False)):
            print(f"JEPA fine-tuning already complete at {complete_path}; skipping training.")
            return

    train_dir = str(Path(data_root) / "train")
    val_dir = str(Path(data_root) / "val")
    test_dir = str(Path(data_root) / "test")
    in_channels = infer_in_channels(train_dir, use_tw_input=use_tw)
    model = PointSegmentationNet(in_channels=in_channels, probe_type=args.probe_type)
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    model.load_jepa_encoder(checkpoint, strict=False)
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    if args.freeze_encoder:
        model.freeze_encoder()
    train_ds = SegmentationBlockDataset(train_dir, max_blocks=max_train, use_tw_input=use_tw)
    val_ds = SegmentationBlockDataset(val_dir, max_blocks=max_val, use_tw_input=use_tw)
    test_ds = SegmentationBlockDataset(test_dir, use_tw_input=use_tw) if Path(test_dir).exists() else None
    sampler = None
    sampler_summary = {}
    if args.balanced_sampler:
        sampler, sampler_summary = balanced_block_sampler(
            train_ds,
            mode=args.class_weight_mode,
            max_weight=args.max_class_weight,
            alpha=args.sampler_alpha,
            sampler_max_weight=args.sampler_max_weight,
            class_boost=parse_class_boost(args.sampler_class_boost),
            generator=generator,
        )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, num_workers=workers, collate_fn=segmentation_collate_fn, pin_memory=True, generator=generator if sampler is None else None)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=segmentation_collate_fn, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=segmentation_collate_fn) if test_ds else None
    weights = class_weights_from_blocks(train_dir, max_blocks=max_train, mode=args.class_weight_mode, max_weight=args.max_class_weight)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer_groups, lr_info = segmentation_optimizer_groups(model, lr=lr, encoder_lr_scale=args.encoder_lr_scale)
    run_config = {
        "model_name": out.name,
        "experiment_type": "jepa_downstream",
        "checkpoint": args.checkpoint,
        "checkpoint_config": checkpoint_config,
        "sigreg_weight": checkpoint_config.get("sigreg_weight"),
        "sigreg_slices": checkpoint_config.get("sigreg_slices"),
        "target_mode": checkpoint_config.get("target_mode", "shared"),
        "data_root": data_root,
        "use_tw_input": use_tw,
        "in_channels": in_channels,
        "probe_type": args.probe_type,
        "frozen_encoder": bool(args.freeze_encoder),
        "encoder_lr_scale": float(args.encoder_lr_scale),
        "encoder_lr": lr_info["encoder_lr"],
        "head_lr": lr_info["head_lr"],
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": workers,
        "seed": args.seed,
        "class_weight_mode": args.class_weight_mode,
        "max_class_weight": args.max_class_weight,
        "class_weights": weights.tolist(),
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "loss_type": args.loss_type,
        "focal_gamma": args.focal_gamma,
        "balanced_sampler": bool(args.balanced_sampler),
        **sampler_summary,
        "save_test_arrays": bool(args.save_test_arrays),
        **model_parameter_summary(model),
    }
    print(json.dumps(run_config, indent=2))
    save_json(out / "run_config.json", run_config)
    trainer = SegmentationTrainer(
        model,
        train_loader,
        val_loader,
        device=device,
        lr=lr,
        epochs=epochs,
        class_weights=weights,
        optimizer_param_groups=optimizer_groups,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
    )
    start_epoch = 1
    if checkpoint_path.exists() and not args.no_resume:
        start_epoch = trainer.load_training_checkpoint(checkpoint_path)
        print(f"Resuming JEPA fine-tuning from {checkpoint_path} at epoch {start_epoch}/{epochs}")
    trainer.fit(
        start_epoch=start_epoch,
        checkpoint_path=checkpoint_path,
        best_model_path=best_model_path,
        progress_label="JEPA fine-tuning",
    )
    if test_loader:
        if args.save_test_arrays:
            test_metrics, test_predictions, test_labels = trainer.evaluate(test_loader, return_predictions=True)
        else:
            test_metrics = trainer.evaluate(test_loader, return_predictions=False)
            test_predictions, test_labels = None, None
    else:
        test_metrics, test_predictions, test_labels = None, None, None
    trainer.save_reports(
        out,
        test_metrics=test_metrics,
        config=run_config,
        test_predictions=test_predictions,
        test_labels=test_labels,
    )
    complete_path.write_text(
        json.dumps(
            {
                "epochs_requested": epochs,
                "epochs_completed": trainer.completed_epoch,
                "early_stopped": trainer.early_stopped,
                "best_macro_f1": trainer.best_macro_f1,
                "pretrain_checkpoint": args.checkpoint,
                "checkpoint": str(checkpoint_path),
                "best_model": str(best_model_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
