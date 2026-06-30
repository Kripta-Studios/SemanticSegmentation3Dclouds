from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


def parse_seeds(value: str) -> list[int]:
    seeds = []
    for item in value.split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return seeds


def run(cmd: list[str], dry_run: bool = False) -> None:
    print("\n> " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def metrics_for(exp_dir: Path) -> dict:
    path = exp_dir / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def config_for(exp_dir: Path) -> dict:
    for name in ("run_config.json", "config.json"):
        path = exp_dir / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def collect_rows(out_root: Path) -> list[dict]:
    rows = []
    for exp_dir in sorted(path for path in out_root.iterdir() if path.is_dir()):
        metrics = metrics_for(exp_dir)
        if not metrics:
            continue
        cfg = config_for(exp_dir)
        row = {
            "model_name": cfg.get("model_name", exp_dir.name),
            "seed": cfg.get("seed", ""),
            "OA": float(metrics.get("OA", 0.0)),
            "macro_F1": float(metrics.get("macro_f1", metrics.get("macro_F1", 0.0))),
            "mIoU": float(metrics.get("macro_iou", metrics.get("mIoU", 0.0))),
            "terrain_IoU": float(metrics.get("class_iou", {}).get("0", 0.0)),
            "low_vegetation_IoU": float(metrics.get("class_iou", {}).get("1", 0.0)),
            "medium_vegetation_IoU": float(metrics.get("class_iou", {}).get("2", 0.0)),
            "high_vegetation_IoU": float(metrics.get("class_iou", {}).get("3", 0.0)),
            "building_IoU": float(metrics.get("class_iou", {}).get("4", 0.0)),
            "water_IoU": float(metrics.get("class_iou", {}).get("5", 0.0)),
            "metrics_path": str(exp_dir / "metrics.json"),
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict]) -> None:
    summary: dict[str, float | int] = {"n_runs": len(rows)}
    for metric in (
        "OA",
        "macro_F1",
        "mIoU",
        "terrain_IoU",
        "low_vegetation_IoU",
        "medium_vegetation_IoU",
        "high_vegetation_IoU",
        "building_IoU",
        "water_IoU",
    ):
        values = [float(row[metric]) for row in rows]
        if values:
            summary[f"{metric}_mean"] = statistics.fmean(values)
            summary[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
            summary[f"{metric}_min"] = min(values)
            summary[f"{metric}_max"] = max(values)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the best medium local-context model with multiple seeds.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("42,1337,2026"))
    parser.add_argument("--data", default="data/processed/galicia_blocks_medium_tw")
    parser.add_argument("--feature-dir", default="data/processed/galicia_blocks_medium_geom_context")
    parser.add_argument("--out-root", default="outputs/local_context_multiseed")
    parser.add_argument("--compare-root", action="append", default=["outputs/medium_plus"])
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-train-blocks", type=int, default=12000)
    parser.add_argument("--max-val-blocks", type=int, default=2000)
    parser.add_argument("--force", action="store_true", help="Restart completed runs with --no-resume.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    for seed in args.seeds:
        exp_dir = out_root / f"geom_concat_h256_balancedval_seed{seed}"
        complete = exp_dir / "training_complete.json"
        if complete.exists() and not args.force:
            print(f"Skipping completed seed {seed}: {complete}", flush=True)
            continue
        cmd = [
            args.python,
            "scripts/03_train_baseline.py",
            "--data",
            args.data,
            "--out",
            str(exp_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--use-tw-input",
            "--external-feature-dir",
            args.feature_dir,
            "--external-feature-key",
            "geom_features",
            "--fusion-type",
            "concat",
            "--probe-type",
            "mlp",
            "--hidden-dim",
            "256",
            "--embed-dim",
            "384",
            "--dropout",
            "0.15",
            "--seed",
            str(seed),
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
            "10",
            "--early-stopping-min-delta",
            "0.001",
            "--max-train-blocks",
            str(args.max_train_blocks),
            "--max-val-blocks",
            str(args.max_val_blocks),
            "--train-block-selection",
            "class_balanced",
            "--val-block-selection",
            "class_balanced",
        ]
        if args.force:
            cmd.append("--no-resume")
        run(cmd, dry_run=args.dry_run)

    rows = collect_rows(out_root)
    write_csv(out_root / "multiseed_metrics.csv", rows)
    write_summary(out_root / "multiseed_summary.json", rows)

    compare_cmd = [
        args.python,
        "scripts/07_compare_results.py",
    ]
    for root in args.compare_root:
        compare_cmd.extend(["--experiments-root", root])
    compare_cmd.extend(
        [
            "--experiments-root",
            str(out_root),
            "--out-csv",
            str(out_root / "comparison" / "test_comparison.csv"),
            "--out-md",
            str(out_root / "comparison" / "test_comparison.md"),
        ]
    )
    run(compare_cmd, dry_run=args.dry_run)

    report_cmd = [
        args.python,
        "scripts/16_sota_benchmark_report.py",
        "--comparison-csv",
        str(out_root / "comparison" / "test_comparison.csv"),
        "--out-dir",
        str(out_root / "benchmark"),
        "--multiseed-root",
        str(out_root),
    ]
    run(report_cmd, dry_run=args.dry_run)
    elapsed = time.perf_counter() - started
    print(f"Multi-seed pipeline finished in {elapsed / 60.0:.1f} min", flush=True)


if __name__ == "__main__":
    main()
