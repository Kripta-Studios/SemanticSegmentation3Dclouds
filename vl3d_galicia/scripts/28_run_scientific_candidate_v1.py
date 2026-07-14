from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


CLASS_NAMES = ["ground", "low_vegetation", "medium_vegetation", "high_vegetation", "building", "water"]
T_CRITICAL_95 = {1: 0.0, 2: 12.7062047364, 3: 4.3026527299}


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return seeds


def run(command: list[str], dry_run: bool) -> None:
    print("\n> " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def metric_value(metrics: dict, key: str) -> float:
    if key.startswith("class_"):
        family, class_id = key.rsplit("_", 1)
        return float(metrics[family][class_id])
    return float(metrics[key])


def summarize(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    critical = T_CRITICAL_95.get(len(values), 1.96)
    half = critical * std / (len(values) ** 0.5) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def aggregate(out_root: Path, models: list[str], seeds: list[int]) -> None:
    scalar_metrics = [
        "OA",
        "macro_f1",
        "macro_iou",
        "balanced_accuracy",
        "weighted_f1",
        "coverage",
        "ignored_prediction_rate",
        "evaluated_points",
    ]
    class_metrics = [
        f"{family}_{class_id}"
        for family in ("class_precision", "class_recall", "class_f1", "class_iou")
        for class_id in range(6)
    ]
    metric_keys = scalar_metrics + class_metrics
    individual_rows: list[dict] = []
    aggregate_payload: dict[str, dict] = {}
    aggregate_rows: list[dict] = []
    for model in models:
        by_seed = []
        for seed in seeds:
            run_dir = out_root / f"{model}_seed{seed}"
            metrics_path = run_dir / "test_metrics.json"
            config_path = run_dir / "run_config.json"
            if not metrics_path.exists() or not config_path.exists():
                raise FileNotFoundError(f"Incomplete run: {run_dir}")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            config = json.loads(config_path.read_text(encoding="utf-8"))
            row = {
                "model": model,
                "seed": seed,
                "metrics_path": str(metrics_path),
                "config_hash": config.get("config_hash"),
                "split_hash": config.get("split_hash"),
                "selected_blocks_hash": config.get("selected_blocks_hash"),
                "git_commit": config.get("git_commit"),
                "duration_seconds": config.get("duration_seconds"),
                "peak_vram_bytes": config.get("peak_vram_bytes"),
                "parameter_count": config.get("parameter_count"),
            }
            for key in metric_keys:
                row[key] = metric_value(metrics, key)
            individual_rows.append(row)
            by_seed.append((seed, metrics, config))
        split_hashes = {item[2].get("split_hash") for item in by_seed}
        block_hashes = {item[2].get("selected_blocks_hash") for item in by_seed}
        if len(split_hashes) != 1 or len(block_hashes) != 1:
            raise ValueError(f"Seeds for {model} do not share split/block hashes")
        model_summary = {
            "seeds": seeds,
            "split_hash": next(iter(split_hashes)),
            "selected_blocks_hash": next(iter(block_hashes)),
            "metrics": {},
        }
        for key in metric_keys:
            stats = summarize([metric_value(item[1], key) for item in by_seed])
            model_summary["metrics"][key] = stats
            aggregate_rows.append({"model": model, "metric": key, **stats})
        aggregate_payload[model] = model_summary

    out_root.mkdir(parents=True, exist_ok=True)
    if individual_rows:
        with (out_root / "individual_runs.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(individual_rows[0]))
            writer.writeheader()
            writer.writerows(individual_rows)
    with (out_root / "aggregate_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "metric", "n", "mean", "std", "ci95_low", "ci95_high"])
        writer.writeheader()
        writer.writerows(aggregate_rows)
    (out_root / "aggregate_metrics.json").write_text(json.dumps(aggregate_payload, indent=2), encoding="utf-8")
    lines = ["# Scientific candidate v1", "", "Mean ± sample SD over the frozen seeds; CI95 uses Student's t.", ""]
    lines.append("| model | OA | macro-F1 | mIoU | balanced accuracy | coverage |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for model in models:
        summary = aggregate_payload[model]["metrics"]
        cells = []
        for key in ("OA", "macro_f1", "macro_iou", "balanced_accuracy", "coverage"):
            item = summary[key]
            cells.append(f"{item['mean']:.4f} ± {item['std']:.4f}")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    (out_root / "aggregate_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Galicia M0-M4 scientific-candidate ablation.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data", default="data/processed/galicia_experiment_v1_tw")
    parser.add_argument("--geom", default="data/processed/galicia_experiment_v1_geom")
    parser.add_argument("--dinov2", default="data/processed/galicia_experiment_v1_dinov2_top")
    parser.add_argument("--dinov3", default="data/processed/galicia_experiment_v1_dinov3_top")
    parser.add_argument("--dinov3-multiview", default="data/processed/galicia_experiment_v1_dinov3_multiview")
    parser.add_argument("--out-root", default="outputs/scientific_candidate_v1")
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("7,13,21"))
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    model_features = {
        "M0_geometric": (None, "concat"),
        "M1_dinov2_concat": (args.dinov2, "concat"),
        "M2_dinov3_concat": (args.dinov3, "concat"),
        "M3_dinov3_gated": (args.dinov3, "gated"),
        "M4_dinov3_multiview": (args.dinov3_multiview, "concat"),
    }
    out_root = Path(args.out_root)
    if not args.aggregate_only:
        for model, (external, fusion) in model_features.items():
            for seed in args.seeds:
                out = out_root / f"{model}_seed{seed}"
                command = [
                    args.python,
                    "scripts/03_train_baseline.py",
                    "--data", args.data,
                    "--out", str(out),
                    "--base-feature-dir", args.geom,
                    "--base-feature-key", "geom_features",
                    "--use-tw-input",
                    "--fusion-type", fusion,
                    "--epochs", str(args.epochs),
                    "--batch-size", str(args.batch_size),
                    "--num-workers", str(args.num_workers),
                    "--hidden-dim", "128",
                    "--embed-dim", "192",
                    "--dropout", "0.15",
                    "--lr", "0.0005",
                    "--weight-decay", "0.0001",
                    "--seed", str(seed),
                    "--data-selection-seed", "20260714",
                    "--coordinate-normalization", "xy_unit_z_robust",
                    "--spectral-normalization", "block_robust",
                    "--class-weight-mode", "inverse_sqrt",
                    "--max-class-weight", "20",
                    "--loss-type", "focal",
                    "--focal-gamma", "1.5",
                    "--balanced-sampler",
                    "--sampler-alpha", "1.2",
                    "--sampler-max-weight", "10",
                    "--sampler-class-boost", "1:1.5,2:2.0,4:4.0",
                    "--early-stopping-patience", "5",
                    "--early-stopping-min-delta", "0.001",
                    "--train-block-selection", "sorted",
                    "--val-block-selection", "sorted",
                    "--test-block-selection", "sorted",
                    "--num-output-classes", "6",
                ]
                if external:
                    command.extend(["--external-feature-dir", external, "--external-feature-key", "dino_features"])
                run(command, args.dry_run)
    if not args.dry_run:
        aggregate(out_root, list(model_features), args.seeds)


if __name__ == "__main__":
    main()
