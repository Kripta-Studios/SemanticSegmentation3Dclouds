from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.classes import CLASS_NAMES, IGNORE_INDEX


OVERALL_METRICS = ("macro_iou", "macro_f1", "weighted_f1", "mcc", "kappa")
CLASS_COLUMNS = {
    0: "terrain",
    1: "low_vegetation",
    2: "medium_vegetation",
    3: "high_vegetation",
    4: "building",
    5: "water",
}

BASELINE_DELTA_METRICS = ("OA", "macro_F1", "mIoU")


def load_metrics(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Metrics file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def value(metrics: dict[str, Any], key: str) -> float:
    raw = metrics.get(key, 0.0)
    return float(raw) if raw is not None else 0.0


def class_value(metrics: dict[str, Any], metric_name: str, class_id: int) -> float:
    data = metrics.get(metric_name, {})
    return float(data.get(str(class_id), data.get(class_id, 0.0)))


def pct(x: float) -> str:
    return f"{100.0 * x:6.2f}"


def signed_pct(x: float) -> str:
    return f"{100.0 * x:+6.2f}"


def print_overall_table(baseline: dict[str, Any], candidate: dict[str, Any], baseline_name: str, candidate_name: str) -> None:
    print("\n=== Final test comparison ===")
    print(f"{'metric':<14} {baseline_name:>14} {candidate_name:>14} {'delta pp':>10}")
    print("-" * 56)
    for metric in OVERALL_METRICS:
        b = value(baseline, metric)
        c = value(candidate, metric)
        print(f"{metric:<14} {pct(b):>14} {pct(c):>14} {signed_pct(c - b):>10}")


def print_class_table(baseline: dict[str, Any], candidate: dict[str, Any], baseline_name: str, candidate_name: str) -> None:
    print("\n=== Per-class F1 / IoU ===")
    print(
        f"{'class':<20} "
        f"{baseline_name + '_f1':>14} {candidate_name + '_f1':>14} {'d_f1 pp':>10} "
        f"{baseline_name + '_iou':>14} {candidate_name + '_iou':>14} {'d_iou pp':>10}"
    )
    print("-" * 102)
    for class_id, name in CLASS_NAMES.items():
        if class_id == IGNORE_INDEX:
            continue
        b_f1 = class_value(baseline, "class_f1", class_id)
        c_f1 = class_value(candidate, "class_f1", class_id)
        b_iou = class_value(baseline, "class_iou", class_id)
        c_iou = class_value(candidate, "class_iou", class_id)
        print(
            f"{name:<20} "
            f"{pct(b_f1):>14} {pct(c_f1):>14} {signed_pct(c_f1 - b_f1):>10} "
            f"{pct(b_iou):>14} {pct(c_iou):>14} {signed_pct(c_iou - b_iou):>10}"
        )


def build_summary(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    baseline_path: Path,
    candidate_path: Path,
    baseline_name: str,
    candidate_name: str,
    primary_metric: str,
) -> dict[str, Any]:
    overall = {}
    for metric in OVERALL_METRICS:
        b = value(baseline, metric)
        c = value(candidate, metric)
        overall[metric] = {
            baseline_name: b,
            candidate_name: c,
            "delta": c - b,
        }

    classes = {}
    for class_id, name in CLASS_NAMES.items():
        if class_id == IGNORE_INDEX:
            continue
        b_f1 = class_value(baseline, "class_f1", class_id)
        c_f1 = class_value(candidate, "class_f1", class_id)
        b_iou = class_value(baseline, "class_iou", class_id)
        c_iou = class_value(candidate, "class_iou", class_id)
        classes[name] = {
            "class_id": class_id,
            "f1": {baseline_name: b_f1, candidate_name: c_f1, "delta": c_f1 - b_f1},
            "iou": {baseline_name: b_iou, candidate_name: c_iou, "delta": c_iou - b_iou},
        }

    delta = overall.get(primary_metric, {}).get("delta", 0.0)
    if delta > 0:
        winner = candidate_name
    elif delta < 0:
        winner = baseline_name
    else:
        winner = "tie"

    return {
        "baseline_metrics": str(baseline_path),
        "candidate_metrics": str(candidate_path),
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "primary_metric": primary_metric,
        "winner": winner,
        "overall": overall,
        "classes": classes,
    }


def write_csv(summary: dict[str, Any], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    baseline_name = summary["baseline_name"]
    candidate_name = summary["candidate_name"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scope", "metric", "class", baseline_name, candidate_name, "delta"])
        for metric, values in summary["overall"].items():
            writer.writerow(["overall", metric, "", values[baseline_name], values[candidate_name], values["delta"]])
        for class_name, class_values in summary["classes"].items():
            for metric in ("f1", "iou"):
                values = class_values[metric]
                writer.writerow(["class", metric, class_name, values[baseline_name], values[candidate_name], values["delta"]])


def read_run_config(exp_dir: Path) -> dict[str, Any]:
    path = exp_dir / "run_config.json"
    if not path.exists():
        path = exp_dir / "config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_external_feature_config(config: dict[str, Any]) -> dict[str, Any]:
    embedded = config.get("external_feature_config")
    if isinstance(embedded, dict) and embedded:
        return embedded
    feature_dir = config.get("external_feature_dir")
    if not feature_dir:
        return {}
    path = Path(feature_dir) / "feature_config.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def experiment_rows(experiments_root: Path) -> list[dict[str, Any]]:
    rows = []
    for exp_dir in sorted(path for path in experiments_root.iterdir() if path.is_dir()):
        metrics_path = exp_dir / "metrics.json"
        if not metrics_path.exists():
            metrics_path = exp_dir / "test_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = load_metrics(metrics_path)
        config = read_run_config(exp_dir)
        external_config = read_external_feature_config(config)
        row = {
            "model_name": config.get("model_name", exp_dir.name),
            "experiment_type": config.get("experiment_type", ""),
            "encoder_mode": "frozen" if config.get("frozen_encoder") else ("semi_frozen" if float(config.get("encoder_lr_scale", 1.0)) < 1.0 else "trainable"),
            "probe_type": config.get("probe_type", ""),
            "checkpoint": config.get("checkpoint", ""),
            "OA": value(metrics, "OA"),
            "AA": value(metrics, "AA"),
            "macro_F1": value(metrics, "macro_f1"),
            "mIoU": value(metrics, "macro_iou"),
            "trainable_params": int(config.get("trainable_params", 0)),
            "frozen_encoder": bool(config.get("frozen_encoder", False)),
            "encoder_lr_scale": float(config.get("encoder_lr_scale", 1.0)),
            "fusion_mode": config.get("fusion_mode", ""),
            "external_feature_backend": config.get("external_feature_backend", external_config.get("backend_used", "")),
            "external_feature_model": config.get("external_feature_model", external_config.get("model", "")),
            "used_real_dino": bool(config.get("used_real_dino", external_config.get("used_real_dino", False))),
            "metrics_path": str(metrics_path),
        }
        for cls, name in CLASS_COLUMNS.items():
            row[f"{name}_IoU"] = class_value(metrics, "class_iou", cls)
        rows.append(row)
    return rows


def find_baseline_row(rows: list[dict[str, Any]], reference_model: str | None = None) -> dict[str, Any] | None:
    if reference_model:
        reference = reference_model.lower()
        for row in rows:
            if str(row.get("model_name", "")).lower() == reference:
                return row
        for row in rows:
            metrics_path = str(row.get("metrics_path", "")).lower()
            if reference in metrics_path:
                return row
    for row in rows:
        if str(row.get("model_name", "")).lower() == "baseline":
            return row
    for row in rows:
        if str(row.get("experiment_type", "")).lower() == "baseline_supervised":
            return row
    baseline_candidates = [
        row
        for row in rows
        if "baseline" in str(row.get("model_name", "")).lower()
        or "baseline" in str(row.get("metrics_path", "")).lower()
    ]
    if baseline_candidates:
        return max(baseline_candidates, key=lambda item: float(item.get("mIoU", 0.0)))
    return None


def verdict_against_baseline(row: dict[str, Any], baseline: dict[str, Any]) -> tuple[str, str]:
    if row is baseline:
        return "BASELINE", "Reference supervised baseline."
    d_miou = float(row.get("delta_mIoU_vs_baseline", 0.0))
    d_f1 = float(row.get("delta_macro_F1_vs_baseline", 0.0))
    d_oa = float(row.get("delta_OA_vs_baseline", 0.0))
    improved = int(row.get("improved_class_iou_count", 0))
    worsened = int(row.get("worsened_class_iou_count", 0))
    if d_miou >= 0.005 and d_f1 >= 0.0 and improved >= worsened:
        return (
            "POSITIVE",
            "Improves mIoU by at least 0.5 pp, does not lose macro-F1, and class IoU gains are not outnumbered.",
        )
    if d_miou > 0.0 or d_f1 > 0.0 or d_oa > 0.0 or improved > worsened:
        return (
            "MIXED",
            "Some metric improves, but the gain is not strong or it hurts relevant classes.",
        )
    return (
        "NEGATIVE",
        "Does not improve OA, macro-F1, or mIoU enough against the supervised baseline.",
    )


def annotate_against_baseline(rows: list[dict[str, Any]], reference_model: str | None = None) -> list[dict[str, Any]]:
    baseline = find_baseline_row(rows, reference_model)
    if baseline is None:
        for row in rows:
            row["verdict_vs_baseline"] = "UNKNOWN"
            row["verdict_reason"] = "No baseline row found."
        return rows
    for row in rows:
        for metric in BASELINE_DELTA_METRICS:
            row[f"delta_{metric}_vs_baseline"] = float(row.get(metric, 0.0)) - float(baseline.get(metric, 0.0))
        improved = 0
        worsened = 0
        for name in CLASS_COLUMNS.values():
            delta = float(row.get(f"{name}_IoU", 0.0)) - float(baseline.get(f"{name}_IoU", 0.0))
            row[f"delta_{name}_IoU_vs_baseline"] = delta
            if delta > 0.002:
                improved += 1
            elif delta < -0.002:
                worsened += 1
        row["improved_class_iou_count"] = improved
        row["worsened_class_iou_count"] = worsened
        verdict, reason = verdict_against_baseline(row, baseline)
        row["verdict_vs_baseline"] = verdict
        row["verdict_reason"] = reason
    return rows


def write_experiment_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model_name",
        "experiment_type",
        "encoder_mode",
        "probe_type",
        "checkpoint",
        "OA",
        "AA",
        "macro_F1",
        "mIoU",
        "delta_OA_vs_baseline",
        "delta_macro_F1_vs_baseline",
        "delta_mIoU_vs_baseline",
        "terrain_IoU",
        "low_vegetation_IoU",
        "medium_vegetation_IoU",
        "high_vegetation_IoU",
        "building_IoU",
        "water_IoU",
        "delta_terrain_IoU_vs_baseline",
        "delta_low_vegetation_IoU_vs_baseline",
        "delta_medium_vegetation_IoU_vs_baseline",
        "delta_high_vegetation_IoU_vs_baseline",
        "delta_building_IoU_vs_baseline",
        "delta_water_IoU_vs_baseline",
        "improved_class_iou_count",
        "worsened_class_iou_count",
        "verdict_vs_baseline",
        "verdict_reason",
        "trainable_params",
        "frozen_encoder",
        "encoder_lr_scale",
        "fusion_mode",
        "external_feature_backend",
        "external_feature_model",
        "used_real_dino",
        "metrics_path",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def geo_jepa_interpretation(rows: list[dict[str, Any]]) -> list[str]:
    by_name = {row["model_name"]: row for row in rows}
    lines = []
    baseline = by_name.get("baseline")
    frozen_linear = by_name.get("jepa_frozen_linear")
    frozen_mlp = by_name.get("jepa_frozen_mlp")
    semifrozen = by_name.get("jepa_semifrozen")
    full = by_name.get("jepa_full_finetune")
    if baseline and frozen_linear:
        if frozen_linear["macro_F1"] > baseline["macro_F1"] or frozen_linear["mIoU"] > baseline["mIoU"]:
            lines.append("JEPA frozen linear supera al baseline en macro-F1 o mIoU: el embedding es linealmente reutilizable.")
        else:
            lines.append("JEPA frozen linear no supera al baseline: no queda demostrada reutilizacion lineal del embedding.")
    if baseline and frozen_mlp:
        delta = max(frozen_mlp["macro_F1"] - baseline["macro_F1"], frozen_mlp["mIoU"] - baseline["mIoU"])
        if delta >= -0.02:
            lines.append("JEPA frozen MLP es competitivo con el baseline.")
        else:
            lines.append("JEPA frozen MLP queda por debajo del baseline.")
    if frozen_mlp and semifrozen:
        if semifrozen["macro_F1"] > frozen_mlp["macro_F1"] or semifrozen["mIoU"] > frozen_mlp["mIoU"]:
            lines.append("JEPA semi-frozen mejora a frozen: adaptar ligeramente el encoder ayuda.")
        else:
            lines.append("JEPA semi-frozen no mejora claramente a frozen.")
    if rows and full:
        best = max(rows, key=lambda row: row["mIoU"])
        if best["model_name"] == "jepa_full_finetune":
            lines.append("JEPA full fine-tune es el mejor por mIoU.")
        else:
            lines.append(f"JEPA full fine-tune no es el mejor por mIoU; gana {best['model_name']}.")
    if not lines:
        lines.append("No hay suficientes experimentos para emitir una conclusion Geo-JEPA.")
    return lines


def baseline_verdict_lines(rows: list[dict[str, Any]], reference_model: str | None = None) -> list[str]:
    baseline = find_baseline_row(rows, reference_model)
    if baseline is None:
        return ["No baseline row found; automatic positive/negative verdict is unavailable."]
    lines = [
        (
            f"Baseline: {baseline['model_name']} "
            f"(OA={baseline['OA']:.4f}, macro-F1={baseline['macro_F1']:.4f}, mIoU={baseline['mIoU']:.4f})."
        )
    ]
    candidates = [row for row in rows if row is not baseline]
    if not candidates:
        lines.append("Only baseline metrics were found.")
        return lines
    for row in sorted(candidates, key=lambda item: item.get("delta_mIoU_vs_baseline", -999), reverse=True):
        lines.append(
            f"{row['model_name']}: {row.get('verdict_vs_baseline', 'UNKNOWN')} "
            f"(dOA={row.get('delta_OA_vs_baseline', 0.0):+.4f}, "
            f"dMacroF1={row.get('delta_macro_F1_vs_baseline', 0.0):+.4f}, "
            f"dMIoU={row.get('delta_mIoU_vs_baseline', 0.0):+.4f}, "
            f"class IoU +/{row.get('improved_class_iou_count', 0)} "
            f"-/{row.get('worsened_class_iou_count', 0)}). "
            f"{row.get('verdict_reason', '')}"
        )
    best = max(rows, key=lambda item: item.get("mIoU", 0.0))
    if best is baseline:
        lines.append("Final verdict: no candidate beats the baseline by mIoU.")
    else:
        lines.append(
            f"Final verdict: {best['model_name']} is best by mIoU "
            f"({best['mIoU']:.4f}, delta {best.get('delta_mIoU_vs_baseline', 0.0):+.4f})."
        )
    return lines


def write_experiment_markdown(rows: list[dict[str, Any]], out_md: Path, reference_model: str | None = None) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda row: row["mIoU"], reverse=True)
    headers = [
        "model_name",
        "encoder_mode",
        "probe_type",
        "external_feature_backend",
        "OA",
        "macro_F1",
        "mIoU",
        "delta_mIoU_vs_baseline",
        "verdict_vs_baseline",
        "high_vegetation_IoU",
        "trainable_params",
    ]
    lines = ["# Geo-JEPA Pilot Comparison", "", "|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in sorted_rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("|" + "|".join(values) + "|")
    lines.extend(["", "## Interpretacion", ""])
    lines.extend(f"- {line}" for line in geo_jepa_interpretation(rows))
    lines.extend(["", "## Verdict vs Baseline", ""])
    lines.extend(f"- {line}" for line in baseline_verdict_lines(rows, reference_model))
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and JEPA segmentation metrics.")
    parser.add_argument("--baseline", default="outputs/baseline/test_metrics.json")
    parser.add_argument("--candidate", default="outputs/tw_jepa_finetune/test_metrics.json")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="tw_jepa")
    parser.add_argument("--primary-metric", default="macro_iou", choices=OVERALL_METRICS)
    parser.add_argument("--out-json", default="outputs/comparison/test_comparison.json")
    parser.add_argument("--out-csv", default="outputs/comparison/test_comparison.csv")
    parser.add_argument("--experiments-root", action="append")
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--reference-model", default=None, help="Model name or path fragment to use as the baseline row in --experiments-root mode.")
    args = parser.parse_args()

    if args.experiments_root:
        roots = [Path(item) for item in args.experiments_root]
        rows = []
        for root in roots:
            rows.extend(experiment_rows(root))
        if not rows:
            raise SystemExit(f"No experiment metrics found in {roots}")
        rows = annotate_against_baseline(rows, args.reference_model)
        out_csv = Path(args.out_csv)
        out_md = Path(args.out_md) if args.out_md else out_csv.with_suffix(".md")
        write_experiment_csv(rows, out_csv)
        write_experiment_markdown(rows, out_md, args.reference_model)
        print(f"Saved experiment comparison CSV: {out_csv}")
        print(f"Saved experiment comparison MD:  {out_md}")
        for line in geo_jepa_interpretation(rows):
            print(f"- {line}")
        return

    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    baseline = load_metrics(baseline_path)
    candidate = load_metrics(candidate_path)

    print_overall_table(baseline, candidate, args.baseline_name, args.candidate_name)
    print_class_table(baseline, candidate, args.baseline_name, args.candidate_name)

    summary = build_summary(
        baseline,
        candidate,
        baseline_path,
        candidate_path,
        args.baseline_name,
        args.candidate_name,
        args.primary_metric,
    )
    primary_delta = summary["overall"][args.primary_metric]["delta"]
    print(
        f"\nWinner by {args.primary_metric}: {summary['winner']} "
        f"({signed_pct(primary_delta)} pp vs {args.baseline_name})"
    )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(summary, Path(args.out_csv))
    print(f"Saved comparison JSON: {out_json}")
    print(f"Saved comparison CSV:  {args.out_csv}")


if __name__ == "__main__":
    main()
