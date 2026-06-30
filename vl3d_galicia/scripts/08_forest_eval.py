from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.forest_metrics import (
    anomaly_rows,
    forest_classification_metrics,
    forest_confusions,
    forest_mvp_checklist,
    grid_rows,
    per_tile_rows,
)
from src.eval.geo_inference import block_files, load_segmentation_model, predict_block


def collect_records(args) -> list[dict]:
    baseline_model, baseline_cfg = load_segmentation_model(args.baseline_checkpoint, args.data, split=args.split, run_config=args.baseline_run_config)
    jepa_model, jepa_cfg = load_segmentation_model(args.jepa_checkpoint, args.data, split=args.split, run_config=args.jepa_run_config)
    files = block_files(args.data, args.split, max_blocks=args.max_blocks)
    records = []
    for path in files:
        data = torch.load(path, weights_only=False, map_location="cpu")
        baseline_pred = predict_block(baseline_model, data, baseline_cfg["use_tw_input"], baseline_cfg["device"])
        jepa_pred = predict_block(jepa_model, data, jepa_cfg["use_tw_input"], jepa_cfg["device"])
        coords = data.get("global_coords", data["coords"]).numpy()
        records.append(
            {
                "file_path": str(path),
                "tile_id": data.get("tile_id", path.stem),
                "coords": coords,
                "labels": data["labels"].numpy(),
                "baseline_pred": baseline_pred,
                "jepa_pred": jepa_pred,
            }
        )
    return records


def plot_grid_metric(df: pd.DataFrame, value: str, out: Path, cmap: str = "viridis") -> None:
    if df.empty:
        return
    plt.figure(figsize=(8, 7))
    plt.scatter(df["x"], df["y"], c=df[value], s=8, cmap=cmap)
    plt.axis("equal")
    plt.colorbar(label=value)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180)
    plt.close()


def write_report(out: Path, summary: dict, checklist: dict, paths: dict) -> None:
    lines = [
        "# Forest-JEPA MVP Report",
        "",
        "## 1. Objetivo",
        "Evaluar si Geo-JEPA produce predicciones utiles para tareas forestales basadas en LiDAR/PNOA.",
        "",
        "## 2. Datos usados",
        f"- Data root: `{summary['data_root']}`",
        f"- Split: `{summary['split']}`",
        f"- Blocks evaluados: {summary['blocks']}",
        f"- Grid size: {summary['grid_size']}",
        "",
        "## 3. Definicion de tarea forestal",
        "- Vegetacion agregada: low + medium + high vegetation.",
        "- Forest core: medium + high vegetation.",
        "- High vegetation.",
        "- Canopy height proxy y canopy cover proxy por celda.",
        "- Canopy gaps y anomalias iniciales basadas en reglas.",
        "- Biomass proxy exploratorio: canopy_height_proxy * medium_high_canopy_cover_proxy.",
        "",
        "## 4. Modelos comparados",
        f"- Baseline: `{summary['baseline_checkpoint']}`",
        f"- JEPA: `{summary['jepa_checkpoint']}`",
        "",
        "## 5. Metricas forestales globales",
        f"- CSV: `{paths['classification_metrics']}`",
        f"- Confusiones forestales: `{paths['confusions']}`",
        "",
        "## 6. Metricas por tile",
        f"- CSV: `{paths['by_tile']}`",
        "",
        "## 7. Metricas por celda",
        f"- CSV: `{paths['grid']}`",
        f"- Anomalias: `{paths['anomalies']}`",
        "",
        "## 8. Mapas",
        f"- Canopy gaps: `{paths.get('canopy_gaps_map', '')}`",
        "",
        "## 9. Interpretacion",
        "Comparar los F1/IoU de vegetation, forest_core y high_vegetation entre baseline y JEPA. Revisar tiles con mayor error absoluto y celdas marcadas como anomalas.",
        "",
        "## 10. Limitaciones",
        "- Canopy height es un proxy, no una medicion forestal certificada.",
        "- Biomasa no se estima cientificamente; cualquier biomass_proxy es exploratorio.",
        "- Especies no se clasifican.",
        "- Change detection no se evalua porque el pipeline actual no tiene dataset multitemporal pareado.",
        "- La interfaz de cambio T1/T2 queda preparada en codigo, pero no se reporta sin datos multifecha.",
        "",
        "## 11. Forest-JEPA MVP checklist",
        "",
    ]
    for key, value in checklist["checks"].items():
        lines.append(f"- {key}: {'Si' if value else 'No'}")
    lines.extend(["", f"Resultado: {checklist['passed']}/{checklist['total']} - **{checklist['verdict']}**", ""])
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Forest-JEPA MVP forest metrics from baseline and JEPA predictions.")
    parser.add_argument("--data", default="data/processed/galicia_blocks_pilot_tw")
    parser.add_argument("--split", default="test")
    parser.add_argument("--baseline-checkpoint", default="outputs/pilot/baseline/best_model.pt")
    parser.add_argument("--jepa-checkpoint", default="outputs/pilot/jepa_full_finetune/best_model.pt")
    parser.add_argument("--baseline-run-config", default="outputs/pilot/baseline/run_config.json")
    parser.add_argument("--jepa-run-config", default="outputs/pilot/jepa_full_finetune/run_config.json")
    parser.add_argument("--out", default="outputs/forest_jepa")
    parser.add_argument("--grid-size", type=float, default=2.0)
    parser.add_argument("--gap-threshold", type=float, default=0.2)
    parser.add_argument("--min-points-per-cell", type=int, default=10)
    parser.add_argument("--max-blocks", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = collect_records(args)
    if not records:
        raise SystemExit(f"No blocks found in {Path(args.data) / args.split}")

    target = np.concatenate([record["labels"] for record in records])
    baseline = np.concatenate([record["baseline_pred"] for record in records])
    jepa = np.concatenate([record["jepa_pred"] for record in records])

    classification_rows = []
    classification_rows.extend(forest_classification_metrics(target, baseline, "baseline"))
    classification_rows.extend(forest_classification_metrics(target, jepa, "jepa"))
    classification_path = out / "forest_classification_metrics.csv"
    pd.DataFrame(classification_rows).to_csv(classification_path, index=False)

    confusion_rows = []
    confusion_rows.extend(forest_confusions(target, baseline, "baseline"))
    confusion_rows.extend(forest_confusions(target, jepa, "jepa"))
    confusions_path = out / "forest_confusions.csv"
    pd.DataFrame(confusion_rows).to_csv(confusions_path, index=False)

    by_tile = per_tile_rows(records, "baseline", "baseline_pred") + per_tile_rows(records, "jepa", "jepa_pred")
    by_tile_path = out / "forest_metrics_by_tile.csv"
    pd.DataFrame(by_tile).to_csv(by_tile_path, index=False)

    grid = grid_rows(records, grid_size=args.grid_size, gap_threshold=args.gap_threshold, min_points_per_cell=args.min_points_per_cell)
    grid_path = out / "forest_grid_metrics.csv"
    grid_df = pd.DataFrame(grid)
    grid_df.to_csv(grid_path, index=False)

    anomalies = anomaly_rows(grid, gap_threshold=args.gap_threshold, sparse_threshold=args.min_points_per_cell)
    anomalies_path = out / "forest_anomalies.csv"
    pd.DataFrame(anomalies).to_csv(anomalies_path, index=False)

    maps_dir = out / "maps"
    canopy_gaps_map = maps_dir / "canopy_gaps.png"
    if not grid_df.empty:
        plot_grid_metric(grid_df, "is_canopy_gap", canopy_gaps_map, cmap="coolwarm")

    metrics_path = out / "forest_metrics.json"
    summary = {
        "data_root": args.data,
        "split": args.split,
        "blocks": len(records),
        "grid_size": args.grid_size,
        "gap_threshold": args.gap_threshold,
        "baseline_checkpoint": args.baseline_checkpoint,
        "jepa_checkpoint": args.jepa_checkpoint,
        "classification_metrics": str(classification_path),
        "confusions": str(confusions_path),
        "by_tile": str(by_tile_path),
        "grid": str(grid_path),
        "anomalies": str(anomalies_path),
    }
    paths = {
        "classification_metrics": str(classification_path),
        "confusions": str(confusions_path),
        "by_tile": str(by_tile_path),
        "grid": str(grid_path),
        "anomalies": str(anomalies_path),
        "canopy_gaps_map": str(canopy_gaps_map) if canopy_gaps_map.exists() else "",
        "maps_dir": str(maps_dir) if maps_dir.exists() else "",
        "report": str(out / "forest_report.md"),
    }
    checklist = forest_mvp_checklist(paths, has_comparison=True)
    summary["checklist"] = checklist
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out / "forest_report.md", summary, checklist, paths)
    print(json.dumps({"forest_metrics": str(metrics_path), "verdict": checklist["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
