from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run(cmd: list[str], dry_run: bool) -> None:
    print("\n> " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def baseline_cmd(args, name: str, seed: int, extra: list[str]) -> list[str]:
    return [
        args.python,
        "scripts/03_train_baseline.py",
        "--data",
        args.data,
        "--out",
        str(Path(args.out_root) / f"{name}_seed{seed}"),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(seed),
        "--max-train-blocks",
        str(args.max_train_blocks),
        "--max-val-blocks",
        str(args.max_val_blocks),
        "--train-block-selection",
        args.block_selection,
        "--val-block-selection",
        args.block_selection,
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--early-stopping-min-delta",
        "0.001",
        "--hidden-dim",
        "256",
        "--embed-dim",
        "384",
        "--dropout",
        "0.15",
        *extra,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible ablations for Galicia SOTA protocol.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data", default="data/processed/galicia_blocks_medium_tw")
    parser.add_argument("--geom-feature-dir", default="data/processed/galicia_blocks_medium_geom_context")
    parser.add_argument("--jepa-checkpoint", default="outputs/medium/tw_jepa_pretrain/best_jepa.pt")
    parser.add_argument("--out-root", default="outputs/sota_ablation")
    parser.add_argument("--compare-root", action="append", default=["outputs/medium_plus"])
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("42,1337,2026"))
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-train-blocks", type=int, default=12000)
    parser.add_argument("--max-val-blocks", type=int, default=2000)
    parser.add_argument("--block-selection", choices=["sorted", "random", "class_balanced"], default="class_balanced")
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--only", default="", help="Comma-separated ablation names to run.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    selected = {item.strip() for item in args.only.split(",") if item.strip()}

    ablations: dict[str, list[str]] = {
        "no_tw_ce": [
            "--class-weight-mode",
            "inverse_sqrt",
            "--max-class-weight",
            "20.0",
            "--loss-type",
            "ce",
        ],
        "tw_ce": [
            "--use-tw-input",
            "--class-weight-mode",
            "inverse_sqrt",
            "--max-class-weight",
            "20.0",
            "--loss-type",
            "ce",
        ],
        "tw_balanced_ce": [
            "--use-tw-input",
            "--class-weight-mode",
            "inverse_sqrt",
            "--max-class-weight",
            "20.0",
            "--loss-type",
            "ce",
            "--balanced-sampler",
            "--sampler-alpha",
            "1.2",
            "--sampler-max-weight",
            "10.0",
            "--sampler-class-boost",
            "1:1.5,2:2.0,4:4.0",
        ],
        "tw_balanced_focal": [
            "--use-tw-input",
            "--class-weight-mode",
            "inverse_sqrt",
            "--max-class-weight",
            "20.0",
            "--loss-type",
            "focal",
            "--focal-gamma",
            "1.5",
            "--balanced-sampler",
            "--sampler-alpha",
            "1.2",
            "--sampler-max-weight",
            "10.0",
            "--sampler-class-boost",
            "1:1.5,2:2.0,4:4.0",
        ],
        "geom_context_full": [
            "--use-tw-input",
            "--external-feature-dir",
            args.geom_feature_dir,
            "--external-feature-key",
            "geom_features",
            "--fusion-type",
            "concat",
            "--class-weight-mode",
            "inverse_sqrt",
            "--max-class-weight",
            "20.0",
            "--loss-type",
            "focal",
            "--focal-gamma",
            "1.5",
            "--balanced-sampler",
            "--sampler-alpha",
            "1.2",
            "--sampler-max-weight",
            "10.0",
            "--sampler-class-boost",
            "1:1.5,2:2.0,4:4.0",
        ],
        "pointnet2_lite_tw": [
            "--model-type",
            "pointnet2_lite",
            "--anchor-count",
            "384",
            "--local-neighbors",
            "16",
            "--interp-neighbors",
            "3",
            "--use-tw-input",
            "--class-weight-mode",
            "inverse_sqrt",
            "--max-class-weight",
            "20.0",
            "--loss-type",
            "focal",
            "--focal-gamma",
            "1.5",
            "--balanced-sampler",
            "--sampler-alpha",
            "1.2",
            "--sampler-max-weight",
            "10.0",
            "--sampler-class-boost",
            "1:1.5,2:2.0,4:4.0",
        ],
        "pointnet2_lite_geom": [
            "--model-type",
            "pointnet2_lite",
            "--anchor-count",
            "384",
            "--local-neighbors",
            "16",
            "--interp-neighbors",
            "3",
            "--use-tw-input",
            "--external-feature-dir",
            args.geom_feature_dir,
            "--external-feature-key",
            "geom_features",
            "--class-weight-mode",
            "inverse_sqrt",
            "--max-class-weight",
            "20.0",
            "--loss-type",
            "focal",
            "--focal-gamma",
            "1.5",
            "--balanced-sampler",
            "--sampler-alpha",
            "1.2",
            "--sampler-max-weight",
            "10.0",
            "--sampler-class-boost",
            "1:1.5,2:2.0,4:4.0",
        ],
    }

    for seed in args.seeds:
        for name, extra in ablations.items():
            if selected and name not in selected:
                continue
            out_dir = Path(args.out_root) / f"{name}_seed{seed}"
            complete = out_dir / "training_complete.json"
            if complete.exists():
                print(f"Skipping completed {out_dir}")
                continue
            run(baseline_cmd(args, name, seed, extra), dry_run=args.dry_run)

    if not selected or "jepa_frozen_mlp" in selected:
        for seed in args.seeds:
            out_dir = Path(args.out_root) / f"jepa_frozen_mlp_seed{seed}"
            if out_dir.joinpath("training_complete.json").exists():
                print(f"Skipping completed {out_dir}")
                continue
            cmd = [
                args.python,
                "scripts/05_finetune_jepa.py",
                "--data",
                args.data,
                "--checkpoint",
                args.jepa_checkpoint,
                "--out",
                str(out_dir),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--seed",
                str(seed),
                "--freeze-encoder",
                "--probe-type",
                "mlp",
                "--use-tw-input",
                "--class-weight-mode",
                "inverse_sqrt",
                "--max-class-weight",
                "20.0",
                "--loss-type",
                "focal",
                "--focal-gamma",
                "1.5",
                "--balanced-sampler",
                "--sampler-alpha",
                "1.2",
                "--sampler-max-weight",
                "10.0",
                "--sampler-class-boost",
                "1:1.5,2:2.0,4:4.0",
                "--early-stopping-patience",
                str(args.early_stopping_patience),
                "--early-stopping-min-delta",
                "0.001",
                "--max-train-blocks",
                str(args.max_train_blocks),
                "--max-val-blocks",
                str(args.max_val_blocks),
            ]
            run(cmd, dry_run=args.dry_run)

    compare_cmd = [args.python, "scripts/07_compare_results.py"]
    for root in args.compare_root:
        compare_cmd.extend(["--experiments-root", root])
    compare_cmd.extend(
        [
            "--experiments-root",
            args.out_root,
            "--out-csv",
            str(Path(args.out_root) / "comparison" / "test_comparison.csv"),
            "--out-md",
            str(Path(args.out_root) / "comparison" / "test_comparison.md"),
        ]
    )
    run(compare_cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
