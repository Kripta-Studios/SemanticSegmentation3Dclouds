from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_common import balanced_block_sampler, class_weights_from_files, infer_channel_layout, load_config, model_parameter_summary, parse_class_boost, save_json, segmentation_optimizer_groups, set_global_seed
from src.data.segmentation_dataset import SegmentationBlockDataset, segmentation_collate_fn
from src.models.segmentation.heads import GatedExternalPointSegmentationNet, PointSegmentationNet
from src.models.segmentation.pointnet2_lite import PointNet2LiteSegmentationNet
from src.training.segmentation_trainer import SegmentationTrainer
from src.data.classes import IGNORE_INDEX, validate_num_output_classes
from src.eval.segmentation_metrics import METRIC_PROTOCOL_VERSION


def main() -> None:
    parser = argparse.ArgumentParser(description="Train supervised PointNet-style baseline on prepared Galicia blocks.")
    parser.add_argument("--config")
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--embed-dim", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--model-type", choices=["point_mlp", "pointnet2_lite"], default="point_mlp")
    parser.add_argument("--anchor-count", type=int, default=384)
    parser.add_argument("--local-neighbors", type=int, default=16)
    parser.add_argument("--interp-neighbors", type=int, default=3)
    parser.add_argument("--use-tw-input", action="store_true")
    parser.add_argument("--base-feature-dir", default=None)
    parser.add_argument("--base-feature-key", default="geom_features")
    parser.add_argument("--external-feature-dir", default=None)
    parser.add_argument("--external-feature-key", default="dino_features")
    parser.add_argument(
        "--allow-stat-baseline",
        action="store_true",
        help="Explicitly allow stat-raster external features for smoke/diagnostics; never promotes as DINO.",
    )
    parser.add_argument("--fusion-type", choices=["concat", "gated"], default="concat")
    parser.add_argument("--coordinate-normalization", choices=["none", "xy_unit_z_robust"], default="none")
    parser.add_argument("--spectral-normalization", choices=["none", "block_robust"], default="none")
    parser.add_argument("--external-feature-normalization", choices=["none", "block_robust", "spectral_block_robust"], default="none")
    parser.add_argument("--spectral-jitter-std", type=float, default=0.0)
    parser.add_argument("--spectral-dropout-prob", type=float, default=0.0)
    parser.add_argument("--external-spectral-jitter-std", type=float, default=0.0)
    parser.add_argument("--external-spectral-dropout-prob", type=float, default=0.0)
    parser.add_argument("--max-train-blocks", type=int)
    parser.add_argument("--max-val-blocks", type=int)
    parser.add_argument("--max-test-blocks", type=int, default=0)
    parser.add_argument("--train-block-selection", choices=["sorted", "random", "class_balanced"], default="sorted")
    parser.add_argument("--val-block-selection", choices=["sorted", "random"], default="random")
    parser.add_argument("--test-block-selection", choices=["sorted", "random"], default="random")
    parser.add_argument(
        "--data-selection-seed",
        type=int,
        default=20260714,
        help="Fixed independently of the model seed so every seed sees identical blocks.",
    )
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
    parser.add_argument("--num-output-classes", type=int, default=IGNORE_INDEX)
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_root = args.data or cfg.get("data", {}).get("blocks_dir", "data/processed/galicia_blocks")
    out = Path(args.out or cfg.get("output", {}).get("dir", "outputs/baseline"))
    epochs = args.epochs or cfg.get("training", {}).get("epochs", 40)
    batch_size = args.batch_size or cfg.get("training", {}).get("batch_size", 24)
    workers = args.num_workers if args.num_workers is not None else cfg.get("training", {}).get("num_workers", 4)
    lr = args.lr or cfg.get("training", {}).get("learning_rate", 5e-4)
    weight_decay = args.weight_decay if args.weight_decay is not None else cfg.get("training", {}).get("weight_decay", 1e-4)
    hidden_dim = args.hidden_dim or cfg.get("model", {}).get("hidden_dim", 192)
    embed_dim = args.embed_dim or cfg.get("model", {}).get("embed_dim", 256)
    dropout = args.dropout if args.dropout is not None else cfg.get("model", {}).get("dropout", 0.2)
    use_tw = args.use_tw_input or cfg.get("data", {}).get("use_tw_input", False)
    max_train = args.max_train_blocks or cfg.get("data", {}).get("max_train_blocks", 0)
    max_val = args.max_val_blocks or cfg.get("data", {}).get("max_val_blocks", 0)
    max_test = args.max_test_blocks or cfg.get("data", {}).get("max_test_blocks", 0)
    external_feature_config = {}
    base_feature_config = {}
    if args.base_feature_dir:
        base_config_path = Path(args.base_feature_dir) / "feature_config.json"
        if not base_config_path.exists():
            raise ValueError(f"Base external features require a manifest: {base_config_path}")
        base_feature_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    if args.external_feature_dir:
        feature_config_path = Path(args.external_feature_dir) / "feature_config.json"
        if feature_config_path.exists():
            external_feature_config = json.loads(feature_config_path.read_text(encoding="utf-8"))
        elif args.external_feature_key == "dino_features":
            raise ValueError(f"DINO external features require a manifest: {feature_config_path}")
    if args.external_feature_key == "dino_features" and args.external_feature_dir:
        used_real_dino = bool(external_feature_config.get("used_real_dino", False))
        if not used_real_dino and not args.allow_stat_baseline:
            raise ValueError(
                "External cache is not backed by a real DINO model. Use --allow-stat-baseline only for "
                "explicit smoke/diagnostic runs; it is not promotion-eligible."
            )
    run_started = time.perf_counter()
    generator = set_global_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    complete_path = out / "training_complete.json"
    best_model_path = out / "best_model.pt"
    checkpoint_path = out / "last_checkpoint.pt"
    if complete_path.exists() and best_model_path.exists() and not args.no_resume:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if int(complete.get("epochs_completed", 0)) >= epochs or bool(complete.get("early_stopped", False)):
            run_config_path = out / "run_config.json"
            if not run_config_path.exists():
                raise ValueError(
                    f"Cannot reuse completed run without {run_config_path}; use a new --out directory "
                    "or --no-resume to regenerate it under the corrected metric/output schema."
                )
            previous_run = json.loads(run_config_path.read_text(encoding="utf-8"))
            compatible = (
                int(previous_run.get("num_output_classes", -1)) == int(args.num_output_classes)
                and previous_run.get("metric_protocol_version") == METRIC_PROTOCOL_VERSION
            )
            if not compatible:
                raise ValueError(
                    f"Completed run at {out} predates the corrected output/metric protocol; "
                    "use a new --out directory or --no-resume to retrain."
                )
            print(f"Baseline already complete at {complete_path}; skipping training.")
            return

    train_dir = str(Path(data_root) / "train")
    val_dir = str(Path(data_root) / "val")
    test_dir = str(Path(data_root) / "test")
    channel_layout = infer_channel_layout(
        train_dir,
        use_tw_input=use_tw,
        base_feature_dir=args.base_feature_dir,
        base_feature_key=args.base_feature_key,
        external_feature_dir=args.external_feature_dir,
        external_feature_key=args.external_feature_key,
    )
    in_channels = channel_layout["in_channels"]
    train_ds = SegmentationBlockDataset(
        train_dir,
        max_blocks=max_train,
        use_tw_input=use_tw,
        base_feature_dir=args.base_feature_dir,
        base_feature_key=args.base_feature_key,
        external_feature_dir=args.external_feature_dir,
        external_feature_key=args.external_feature_key,
        selection_mode=args.train_block_selection,
        selection_seed=args.data_selection_seed,
        selection_class_boost=parse_class_boost(args.sampler_class_boost),
        coordinate_normalization=args.coordinate_normalization,
        spectral_normalization=args.spectral_normalization,
        external_feature_normalization=args.external_feature_normalization,
        spectral_jitter_std=args.spectral_jitter_std,
        spectral_dropout_prob=args.spectral_dropout_prob,
        external_spectral_jitter_std=args.external_spectral_jitter_std,
        external_spectral_dropout_prob=args.external_spectral_dropout_prob,
    )
    val_ds = SegmentationBlockDataset(
        val_dir,
        max_blocks=max_val,
        use_tw_input=use_tw,
        base_feature_dir=args.base_feature_dir,
        base_feature_key=args.base_feature_key,
        external_feature_dir=args.external_feature_dir,
        external_feature_key=args.external_feature_key,
        selection_mode=args.val_block_selection,
        selection_seed=args.data_selection_seed,
        selection_class_boost=parse_class_boost(args.sampler_class_boost),
        coordinate_normalization=args.coordinate_normalization,
        spectral_normalization=args.spectral_normalization,
        external_feature_normalization=args.external_feature_normalization,
    )
    test_ds = (
        SegmentationBlockDataset(
            test_dir,
            max_blocks=max_test,
            use_tw_input=use_tw,
            base_feature_dir=args.base_feature_dir,
            base_feature_key=args.base_feature_key,
            external_feature_dir=args.external_feature_dir,
            external_feature_key=args.external_feature_key,
            selection_mode=args.test_block_selection,
            selection_seed=args.data_selection_seed,
            selection_class_boost=parse_class_boost(args.sampler_class_boost),
            coordinate_normalization=args.coordinate_normalization,
            spectral_normalization=args.spectral_normalization,
            external_feature_normalization=args.external_feature_normalization,
        )
        if Path(test_dir).exists()
        else None
    )
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
    weights = class_weights_from_files(train_ds.files, mode=args.class_weight_mode, max_weight=args.max_class_weight)
    validate_num_output_classes(args.num_output_classes)
    weights = weights[: args.num_output_classes] if weights is not None else None
    dataset_manifest = {}
    for manifest_name in ("_dataset_manifest.json", "_prepare_complete.json"):
        candidate = Path(data_root) / manifest_name
        if candidate.exists():
            dataset_manifest = json.loads(candidate.read_text(encoding="utf-8"))
            break
    dataset_run = dataset_manifest.get("run", dataset_manifest)
    dataset_split_hash = dataset_run.get("split_hash") or dataset_manifest.get("split_hash")
    for role, feature_config in (("base", base_feature_config), ("external", external_feature_config)):
        if feature_config and feature_config.get("split_hash") != dataset_split_hash:
            raise ValueError(
                f"{role} feature cache split hash differs from blocks: "
                f"{feature_config.get('split_hash')} != {dataset_split_hash}"
            )
    selected_files_payload = {
        "train": [Path(path).name for path in train_ds.files],
        "val": [Path(path).name for path in val_ds.files],
        "test": [Path(path).name for path in test_ds.files] if test_ds else [],
    }
    selected_blocks_hash = hashlib.sha256(
        json.dumps(selected_files_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if args.model_type == "pointnet2_lite" and args.fusion_type == "gated":
        raise ValueError("pointnet2_lite currently supports concat/no external fusion only")
    if args.model_type == "pointnet2_lite":
        model = PointNet2LiteSegmentationNet(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            dropout=dropout,
            anchor_count=args.anchor_count,
            neighbors=args.local_neighbors,
            interp_neighbors=args.interp_neighbors,
            num_classes=args.num_output_classes,
        )
    elif args.fusion_type == "gated":
        if not args.external_feature_dir:
            raise ValueError("--fusion-type gated requires --external-feature-dir")
        model = GatedExternalPointSegmentationNet(
            base_in_channels=channel_layout["base_in_channels"],
            external_in_channels=channel_layout["external_in_channels"],
            probe_type=args.probe_type,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            dropout=dropout,
            num_classes=args.num_output_classes,
        )
    else:
        model = PointSegmentationNet(
            in_channels=in_channels,
            probe_type=args.probe_type,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            dropout=dropout,
            num_classes=args.num_output_classes,
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer_groups, lr_info = segmentation_optimizer_groups(model, lr=lr, encoder_lr_scale=1.0)
    run_config = {
        "model_name": out.name,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": " ".join(shlex.quote(item) for item in sys.argv),
        "experiment_type": "baseline_supervised",
        "data_root": data_root,
        "use_tw_input": use_tw,
        "base_feature_dir": args.base_feature_dir,
        "base_feature_key": args.base_feature_key,
        "base_feature_config": base_feature_config,
        "base_feature_cache_hash": base_feature_config.get("cache_hash") or base_feature_config.get("feature_schema_sha256"),
        "external_feature_dir": args.external_feature_dir,
        "external_feature_key": args.external_feature_key,
        "external_feature_config": external_feature_config,
        "external_feature_backend": external_feature_config.get("backend_used", ""),
        "external_feature_model": external_feature_config.get("model", ""),
        "used_real_dino": bool(external_feature_config.get("used_real_dino", False)),
        "artifact_kind": external_feature_config.get("artifact_kind", "point_baseline"),
        "promotion_eligible": bool(
            not args.external_feature_dir or external_feature_config.get("promotion_eligible", False)
        ),
        "fusion_mode": f"point_tw_external_{args.fusion_type}" if args.external_feature_dir else "point_tw" if use_tw else "point",
        "fusion_type": args.fusion_type,
        "coordinate_normalization": args.coordinate_normalization,
        "spectral_normalization": args.spectral_normalization,
        "external_feature_normalization": args.external_feature_normalization,
        "spectral_jitter_std": float(args.spectral_jitter_std),
        "spectral_dropout_prob": float(args.spectral_dropout_prob),
        "external_spectral_jitter_std": float(args.external_spectral_jitter_std),
        "external_spectral_dropout_prob": float(args.external_spectral_dropout_prob),
        "model_type": args.model_type,
        "anchor_count": args.anchor_count,
        "local_neighbors": args.local_neighbors,
        "interp_neighbors": args.interp_neighbors,
        **channel_layout,
        "in_channels": in_channels,
        "num_output_classes": int(args.num_output_classes),
        "target_ignore_index": int(IGNORE_INDEX),
        "metric_protocol_version": METRIC_PROTOCOL_VERSION,
        "probe_type": args.probe_type,
        "frozen_encoder": False,
        "encoder_lr_scale": 1.0,
        "encoder_lr": lr_info["encoder_lr"],
        "head_lr": lr_info["head_lr"],
        "weight_decay": weight_decay,
        "hidden_dim": hidden_dim,
        "embed_dim": embed_dim,
        "dropout": dropout,
        "checkpoint": None,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": workers,
        "seed": args.seed,
        "data_selection_seed": args.data_selection_seed,
        "selected_blocks_hash": selected_blocks_hash,
        "split_hash": dataset_split_hash,
        "cache_hash": external_feature_config.get("feature_schema_sha256") if external_feature_config else None,
        "class_weight_mode": args.class_weight_mode,
        "max_class_weight": args.max_class_weight,
        "max_train_blocks": max_train,
        "max_val_blocks": max_val,
        "max_test_blocks": max_test,
        "train_block_selection": args.train_block_selection,
        "val_block_selection": args.val_block_selection,
        "test_block_selection": args.test_block_selection,
        "selected_train_blocks": len(train_ds.files),
        "selected_val_blocks": len(val_ds.files),
        "selected_test_blocks": len(test_ds.files) if test_ds else 0,
        "selected_test_tiles": len(
            {Path(path).stem.rsplit("_block_", 1)[0] for path in test_ds.files}
        ) if test_ds else 0,
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
    run_config["config_hash"] = hashlib.sha256(
        json.dumps(run_config, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
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
        weight_decay=weight_decay,
        optimizer_param_groups=optimizer_groups,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
    )
    start_epoch = 1
    if checkpoint_path.exists() and not args.no_resume:
        start_epoch = trainer.load_training_checkpoint(checkpoint_path)
        print(f"Resuming baseline from {checkpoint_path} at epoch {start_epoch}/{epochs}")
    trainer.fit(
        start_epoch=start_epoch,
        checkpoint_path=checkpoint_path,
        best_model_path=best_model_path,
        progress_label="baseline training",
    )
    if test_loader:
        if args.save_test_arrays:
            test_metrics, test_predictions, test_labels = trainer.evaluate(test_loader, return_predictions=True)
        else:
            test_metrics = trainer.evaluate(test_loader, return_predictions=False)
            test_predictions, test_labels = None, None
    else:
        test_metrics, test_predictions, test_labels = None, None, None
    if test_metrics is not None:
        test_metrics["tiles_evaluated"] = run_config["selected_test_tiles"]
        test_metrics["blocks_evaluated"] = run_config["selected_test_blocks"]
    run_config["duration_seconds"] = float(time.perf_counter() - run_started)
    run_config["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    run_config["selection_criterion"] = "validation_macro_f1"
    run_config["best_validation_macro_f1"] = float(trainer.best_macro_f1)
    save_json(out / "run_config.json", run_config)
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
                "checkpoint": str(checkpoint_path),
                "best_model": str(best_model_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
