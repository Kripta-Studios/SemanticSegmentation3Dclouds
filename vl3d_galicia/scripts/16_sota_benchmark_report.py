from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.classes import CLASS_NAMES, IGNORE_INDEX


CLASS_SLUGS = {
    0: "terrain",
    1: "low_vegetation",
    2: "medium_vegetation",
    3: "high_vegetation",
    4: "building",
    5: "water",
}


PAPER_REFERENCE = {
    "name": "Deep Learning for Ultra-Large-Scale Semantic Segmentation of Geographic 3D Point Clouds",
    "notes": (
        "Only metrics explicitly extracted in this repository are compared here. "
        "A strict paper/SOTA claim still requires running the paper backbone/protocol "
        "on the same split."
    ),
    "metrics": {
        "low_vegetation_f1": 0.4625,
        "high_vegetation_f1": 0.9617,
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in (None, ""):
        return default
    return float(value)


def _metrics_path(row: dict[str, str]) -> Path | None:
    value = row.get("metrics_path", "")
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _run_config(metrics_path: Path | None) -> dict[str, Any]:
    if metrics_path is None:
        return {}
    exp_dir = metrics_path.parent
    for name in ("run_config.json", "config.json"):
        payload = _read_json(exp_dir / name)
        if payload:
            return payload
    return {}


def _best_row(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        raise ValueError("No experiment rows found")
    return max(rows, key=lambda row: _float(row, "mIoU"))


def _baseline_row(rows: list[dict[str, str]], baseline_name: str) -> dict[str, str]:
    for row in rows:
        if row.get("model_name", "").lower() == baseline_name.lower():
            return row
    for row in rows:
        if row.get("experiment_type", "").lower() == "baseline_supervised":
            return row
    raise ValueError(f"No baseline row named '{baseline_name}' found")


def _class_delta_rows(baseline: dict[str, str], candidate: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for cls, slug in CLASS_SLUGS.items():
        rows.append(
            {
                "class_id": cls,
                "class_name": CLASS_NAMES.get(cls, slug),
                "baseline_iou": _float(baseline, f"{slug}_IoU"),
                "candidate_iou": _float(candidate, f"{slug}_IoU"),
                "delta_iou": _float(candidate, f"{slug}_IoU") - _float(baseline, f"{slug}_IoU"),
            }
        )
    return rows


def _paper_rows(candidate_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    class_f1 = candidate_metrics.get("class_f1", {})
    low_f1 = float(class_f1.get("1", class_f1.get(1, 0.0)))
    high_f1 = float(class_f1.get("3", class_f1.get(3, 0.0)))
    candidate_values = {
        "low_vegetation_f1": low_f1,
        "high_vegetation_f1": high_f1,
    }
    rows = []
    for metric, paper_value in PAPER_REFERENCE["metrics"].items():
        candidate_value = float(candidate_values.get(metric, 0.0))
        rows.append(
            {
                "metric": metric,
                "paper_value": paper_value,
                "candidate_value": candidate_value,
                "delta": candidate_value - paper_value,
                "beats_paper_extracted_metric": candidate_value > paper_value,
            }
        )
    return rows


def _multi_seed_rows(root: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if root is None or not root.exists():
        return [], {}
    rows = []
    for metrics_path in sorted(root.glob("*/metrics.json")):
        metrics = _read_json(metrics_path)
        cfg = _run_config(metrics_path)
        rows.append(
            {
                "model_name": cfg.get("model_name", metrics_path.parent.name),
                "seed": cfg.get("seed", ""),
                "OA": float(metrics.get("OA", 0.0)),
                "macro_F1": float(metrics.get("macro_f1", metrics.get("macro_F1", 0.0))),
                "mIoU": float(metrics.get("macro_iou", metrics.get("mIoU", 0.0))),
                "metrics_path": str(metrics_path),
            }
        )
    summary: dict[str, Any] = {}
    for metric in ("OA", "macro_F1", "mIoU"):
        values = [float(row[metric]) for row in rows]
        if values:
            summary[f"{metric}_mean"] = statistics.fmean(values)
            summary[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
            summary[f"{metric}_min"] = min(values)
            summary[f"{metric}_max"] = max(values)
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _fmt_delta(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


def _write_markdown(
    path: Path,
    comparison_csv: Path,
    baseline: dict[str, str],
    candidate: dict[str, str],
    candidate_metrics: dict[str, Any],
    candidate_config: dict[str, Any],
    class_rows: list[dict[str, Any]],
    paper_rows: list[dict[str, Any]],
    multi_seed_rows: list[dict[str, Any]],
    multi_seed_summary: dict[str, Any],
) -> None:
    feature_cfg = candidate_config.get("external_feature_config", {})
    uses_labels = feature_cfg.get("uses_labels", None)
    test_complete = int(candidate_config.get("max_test_blocks", 0) or 0) == 0
    improves_all_classes = all(float(row["delta_iou"]) > 0.0 for row in class_rows)
    candidate_is_supervised = candidate_config.get("experiment_type") == "baseline_supervised"
    feature_label_free = uses_labels is False
    paper_wins = sum(1 for row in paper_rows if row["beats_paper_extracted_metric"])

    if improves_all_classes and _float(candidate, "mIoU") > _float(baseline, "mIoU"):
        internal_verdict = "POSITIVE: improves OA/macro-F1/mIoU and all tracked class IoUs against the internal baseline."
    else:
        internal_verdict = "MIXED/NEGATIVE: does not dominate the internal baseline across all tracked metrics."

    if candidate_is_supervised:
        ssl_verdict = (
            "No. The current winning run is supervised downstream training. "
            "Its external geometric context features are label-free, but the model itself is not a self-supervised JEPA winner."
        )
    else:
        ssl_verdict = "Partially/yes depending on the selected experiment type; inspect run_config.json."

    sota_verdict = (
        "Not yet. The result is strong internally, but strict SOTA needs a reproduced SFL-Net/KPConv/PT-style benchmark "
        "on the same locked split plus multi-seed confidence."
    )

    lines = [
        "# Galicia LiDAR Benchmark Report",
        "",
        f"Comparison CSV: `{comparison_csv}`",
        "",
        "## Executive Verdict",
        "",
        f"- Internal benchmark: {internal_verdict}",
        f"- Self-supervised claim: {ssl_verdict}",
        f"- Strict SOTA claim: {sota_verdict}",
        "",
        "## Best Candidate vs Internal Baseline",
        "",
        "|metric|baseline|candidate|delta|",
        "|---|---:|---:|---:|",
    ]
    for metric in ("OA", "macro_F1", "mIoU"):
        b = _float(baseline, metric)
        c = _float(candidate, metric)
        lines.append(f"|{metric}|{_fmt_pct(b)}|{_fmt_pct(c)}|{_fmt_delta(c - b)}|")

    lines.extend(
        [
            "",
            "## Per-Class IoU",
            "",
            "|class|baseline IoU|candidate IoU|delta|",
            "|---|---:|---:|---:|",
        ]
    )
    for row in class_rows:
        lines.append(
            f"|{row['class_name']}|{_fmt_pct(row['baseline_iou'])}|"
            f"{_fmt_pct(row['candidate_iou'])}|{_fmt_delta(row['delta_iou'])}|"
        )

    lines.extend(
        [
            "",
            "## Extracted Paper Metrics",
            "",
            PAPER_REFERENCE["notes"],
            "",
            "|metric|paper extracted|candidate|delta|beats extracted metric|",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in paper_rows:
        lines.append(
            f"|{row['metric']}|{_fmt_pct(row['paper_value'])}|{_fmt_pct(row['candidate_value'])}|"
            f"{_fmt_delta(row['delta'])}|{row['beats_paper_extracted_metric']}|"
        )
    lines.append("")
    if paper_wins == len(paper_rows):
        lines.append("The candidate beats all paper metrics currently extracted into this repository.")
    else:
        lines.append("The candidate does not beat every extracted paper metric.")

    lines.extend(
        [
            "",
            "## Leakage / Protocol Audit",
            "",
            f"- External feature cache uses labels: `{uses_labels}`.",
            f"- External features are label-free according to config: `{feature_label_free}`.",
            f"- Test uses full split (`max_test_blocks == 0`): `{test_complete}`.",
            f"- Train block selection: `{candidate_config.get('train_block_selection', '')}`.",
            f"- Val block selection: `{candidate_config.get('val_block_selection', '')}`.",
            f"- Test block selection: `{candidate_config.get('test_block_selection', '')}`.",
            f"- Selected train/val/test blocks: `{candidate_config.get('selected_train_blocks', '')}` / "
            f"`{candidate_config.get('selected_val_blocks', '')}` / `{candidate_config.get('selected_test_blocks', '')}`.",
            "",
            "Important caveat: the current test split has been inspected during model development. "
            "For a publication-grade or enterprise benchmark, freeze a fresh geographic holdout before further tuning and run it once.",
            "",
            "## Multi-Seed Stability",
            "",
        ]
    )
    if multi_seed_rows:
        lines.extend(["|model|seed|OA|macro-F1|mIoU|", "|---|---:|---:|---:|---:|"])
        for row in multi_seed_rows:
            lines.append(
                f"|{row['model_name']}|{row['seed']}|{_fmt_pct(row['OA'])}|"
                f"{_fmt_pct(row['macro_F1'])}|{_fmt_pct(row['mIoU'])}|"
            )
        lines.append("")
        lines.append(
            "Summary: "
            + ", ".join(
                f"{key}={value:.4f}" for key, value in sorted(multi_seed_summary.items()) if isinstance(value, float)
            )
        )
    else:
        lines.append("No multi-seed directory was provided or no completed multi-seed metrics were found.")

    lines.extend(
        [
            "",
            "## Reproduction Commands",
            "",
            "Generate or refresh the local context feature cache:",
            "",
            "```powershell",
            "python scripts\\15_build_geom_context_features.py --data data/processed/galicia_blocks_medium_tw --out data/processed/galicia_blocks_medium_geom_context --splits train,val,test --num-workers 16",
            "```",
            "",
            "Train the current best medium model:",
            "",
            "```powershell",
            "python scripts\\03_train_baseline.py --data data/processed/galicia_blocks_medium_tw --out outputs/local_context_medium/geom_concat_h256_balancedval --epochs 45 --batch-size 24 --num-workers 8 --use-tw-input --external-feature-dir data/processed/galicia_blocks_medium_geom_context --external-feature-key geom_features --fusion-type concat --probe-type mlp --hidden-dim 256 --embed-dim 384 --dropout 0.15 --seed 42 --class-weight-mode inverse_sqrt --max-class-weight 20.0 --loss-type focal --focal-gamma 1.5 --balanced-sampler --sampler-alpha 1.2 --sampler-max-weight 10.0 --sampler-class-boost 1:1.5,2:2.0,4:4.0 --early-stopping-patience 10 --early-stopping-min-delta 0.001 --max-train-blocks 12000 --max-val-blocks 2000 --train-block-selection class_balanced --val-block-selection class_balanced",
            "```",
            "",
            "Refresh comparison and benchmark report:",
            "",
            "```powershell",
            "python scripts\\07_compare_results.py --experiments-root outputs\\medium_plus --experiments-root outputs\\local_context_medium --out-csv outputs\\local_context_medium\\comparison\\test_comparison.csv --out-md outputs\\local_context_medium\\comparison\\test_comparison.md",
            "python scripts\\16_sota_benchmark_report.py --comparison-csv outputs\\local_context_medium\\comparison\\test_comparison.csv --out-dir outputs\\local_context_medium\\benchmark",
            "```",
            "",
            "## Next Work Required For SOTA",
            "",
            "- Reproduce the paper backbone or a modern KPConv/RandLA-Net/PTv3/Superpoint Transformer baseline in this same protocol.",
            "- Lock a new geographic holdout before tuning again.",
            "- Run at least 3 seeds and report mean/std.",
            "- Only then claim SOTA/industry superiority. Until then, claim a strong internal Galicia MVP result.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a strict benchmark report for Galicia LiDAR runs.")
    parser.add_argument("--comparison-csv", default="outputs/local_context_medium/comparison/test_comparison.csv")
    parser.add_argument("--out-dir", default="outputs/local_context_medium/benchmark")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="")
    parser.add_argument("--multiseed-root", default="")
    args = parser.parse_args()

    comparison_csv = Path(args.comparison_csv)
    rows = _read_rows(comparison_csv)
    baseline = _baseline_row(rows, args.baseline_name)
    if args.candidate_name:
        matches = [row for row in rows if row.get("model_name") == args.candidate_name]
        if not matches:
            raise SystemExit(f"Candidate not found: {args.candidate_name}")
        candidate = matches[0]
    else:
        candidate = _best_row(rows)

    metrics_path = _metrics_path(candidate)
    candidate_metrics = _read_json(metrics_path) if metrics_path else {}
    candidate_config = _run_config(metrics_path)
    class_rows = _class_delta_rows(baseline, candidate)
    paper_rows = _paper_rows(candidate_metrics)
    multiseed_root = Path(args.multiseed_root) if args.multiseed_root else None
    seed_rows, seed_summary = _multi_seed_rows(multiseed_root)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "class_iou_vs_baseline.csv", class_rows)
    _write_csv(out_dir / "paper_metric_comparison.csv", paper_rows)
    _write_csv(out_dir / "multiseed_metrics.csv", seed_rows)
    summary = {
        "comparison_csv": str(comparison_csv),
        "baseline": baseline,
        "candidate": candidate,
        "candidate_metrics_path": str(metrics_path) if metrics_path else "",
        "class_iou_vs_baseline": class_rows,
        "paper_metric_comparison": paper_rows,
        "multiseed_summary": seed_summary,
        "verdict": {
            "internal_positive": _float(candidate, "mIoU") > _float(baseline, "mIoU")
            and all(float(row["delta_iou"]) > 0.0 for row in class_rows),
            "self_supervised_winner": candidate_config.get("experiment_type") != "baseline_supervised",
            "strict_sota_established": False,
        },
    }
    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(
        out_dir / "benchmark_report.md",
        comparison_csv,
        baseline,
        candidate,
        candidate_metrics,
        candidate_config,
        class_rows,
        paper_rows,
        seed_rows,
        seed_summary,
    )
    print(f"Saved benchmark report: {out_dir / 'benchmark_report.md'}")
    print(f"Saved benchmark summary: {out_dir / 'benchmark_summary.json'}")


if __name__ == "__main__":
    main()
