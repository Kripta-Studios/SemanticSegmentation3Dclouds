from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standard Geo-JEPA MVP downstream pilot experiments.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data", default="data/processed/galicia_blocks_pilot_tw")
    parser.add_argument("--out-root", default="outputs/pilot")
    parser.add_argument("--pretrain-out", default=None)
    parser.add_argument("--epochs-baseline", type=int, default=5)
    parser.add_argument("--epochs-pretrain", type=int, default=10)
    parser.add_argument("--epochs-probe", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--jepa-batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-train-blocks", type=int, default=1500)
    parser.add_argument("--max-val-blocks", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weight-mode", choices=["none", "effective", "inverse", "inverse_sqrt", "median_frequency"], default="inverse_sqrt")
    parser.add_argument("--max-class-weight", type=float, default=20.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--pretrain-early-stopping-patience", type=int, default=0)
    parser.add_argument("--pretrain-early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    pretrain_out = Path(args.pretrain_out) if args.pretrain_out else out_root / "tw_jepa_pretrain"
    checkpoint = pretrain_out / "best_jepa.pt"

    train_limits = [
        "--max-train-blocks",
        str(args.max_train_blocks),
        "--max-val-blocks",
        str(args.max_val_blocks),
    ]
    jepa_limits = ["--max-blocks", str(args.max_train_blocks)]
    class_weight_args = ["--class-weight-mode", args.class_weight_mode, "--max-class-weight", str(args.max_class_weight)]
    early_stopping_args = [
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--early-stopping-min-delta",
        str(args.early_stopping_min_delta),
    ]
    pretrain_early_stopping_args = [
        "--early-stopping-patience",
        str(args.pretrain_early_stopping_patience),
        "--early-stopping-min-delta",
        str(args.pretrain_early_stopping_min_delta),
    ]

    resume_args = ["--no-resume"] if args.no_resume else []

    if not args.skip_pretrain and (args.no_resume or not checkpoint.exists()):
        run(
            [
                args.python,
                "scripts/04_pretrain_jepa.py",
                "--data",
                args.data,
                "--out",
                str(pretrain_out),
                "--epochs",
                str(args.epochs_pretrain),
                "--batch-size",
                str(args.jepa_batch_size),
                "--num-workers",
                str(args.num_workers),
                "--tw-target",
                "--seed",
                str(args.seed),
                *jepa_limits,
                *pretrain_early_stopping_args,
                *resume_args,
            ]
        )
    if not checkpoint.exists():
        raise SystemExit(f"Missing JEPA checkpoint: {checkpoint}")

    run(
        [
            args.python,
            "scripts/03_train_baseline.py",
            "--data",
            args.data,
            "--out",
            str(out_root / "baseline"),
            "--epochs",
            str(args.epochs_baseline),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--probe-type",
            "mlp",
            "--seed",
            str(args.seed),
            *class_weight_args,
            *early_stopping_args,
            *train_limits,
            *resume_args,
        ]
    )

    experiments = [
        ("jepa_frozen_linear", ["--freeze-encoder", "--probe-type", "linear"]),
        ("jepa_frozen_mlp", ["--freeze-encoder", "--probe-type", "mlp"]),
        ("jepa_semifrozen", ["--encoder-lr-scale", "0.01", "--probe-type", "mlp"]),
        ("jepa_full_finetune", ["--encoder-lr-scale", "1.0", "--probe-type", "mlp"]),
    ]
    for name, extra in experiments:
        run(
            [
                args.python,
                "scripts/05_finetune_jepa.py",
                "--data",
                args.data,
                "--checkpoint",
                str(checkpoint),
                "--out",
                str(out_root / name),
                "--epochs",
                str(args.epochs_probe),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--seed",
                str(args.seed),
                *extra,
                *class_weight_args,
                *early_stopping_args,
                *train_limits,
                *resume_args,
            ]
        )

    run(
        [
            args.python,
            "scripts/07_compare_results.py",
            "--experiments-root",
            str(out_root),
            "--out-csv",
            str(out_root / "comparison" / "test_comparison.csv"),
            "--out-md",
            str(out_root / "comparison" / "test_comparison.md"),
        ]
    )


if __name__ == "__main__":
    main()
