from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric(df: pd.DataFrame, column: str, out: Path, title: str, cmap: str = "viridis") -> None:
    if df.empty or column not in df:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 7))
    plt.scatter(df["x"], df["y"], c=df[column], s=8, cmap=cmap)
    plt.title(title)
    plt.axis("equal")
    plt.axis("off")
    plt.colorbar(label=column)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Forest-JEPA MVP maps from grid metrics.")
    parser.add_argument("--grid-csv", default="outputs/forest_jepa/forest_grid_metrics.csv")
    parser.add_argument("--anomalies-csv", default="outputs/forest_jepa/forest_anomalies.csv")
    parser.add_argument("--out", default="outputs/forest_jepa/maps")
    args = parser.parse_args()

    grid = pd.read_csv(args.grid_csv)
    out = Path(args.out)
    maps = {
        "vegetation_gt.png": ("vegetation_ratio_gt", "Vegetation GT", "Greens"),
        "vegetation_baseline.png": ("vegetation_ratio_baseline", "Vegetation baseline", "Greens"),
        "vegetation_jepa.png": ("vegetation_ratio_jepa", "Vegetation JEPA", "Greens"),
        "high_vegetation_gt.png": ("high_vegetation_ratio_gt", "High vegetation GT", "Greens"),
        "high_vegetation_baseline.png": ("high_vegetation_ratio_baseline", "High vegetation baseline", "Greens"),
        "high_vegetation_jepa.png": ("high_vegetation_ratio_jepa", "High vegetation JEPA", "Greens"),
        "canopy_height_proxy.png": ("canopy_height_proxy", "Canopy height proxy", "viridis"),
        "canopy_cover_proxy.png": ("medium_high_canopy_cover_proxy", "Canopy cover proxy", "Greens"),
        "canopy_gaps.png": ("is_canopy_gap", "Canopy gaps", "coolwarm"),
        "forest_error_baseline.png": ("baseline_error_rate", "Forest error baseline", "Reds"),
        "forest_error_jepa.png": ("jepa_error_rate", "Forest error JEPA", "Reds"),
        "forest_error_difference.png": ("error_difference_baseline_minus_jepa", "Error difference baseline - JEPA", "coolwarm"),
    }
    for filename, (column, title, cmap) in maps.items():
        plot_metric(grid, column, out / filename, title, cmap)

    anomalies_path = Path(args.anomalies_csv)
    if anomalies_path.exists():
        anomalies = pd.read_csv(anomalies_path)
        if not anomalies.empty:
            plt.figure(figsize=(8, 7))
            plt.scatter(grid["x"], grid["y"], c="lightgray", s=4)
            plt.scatter(anomalies["x"], anomalies["y"], c=anomalies["score"], s=16, cmap="magma")
            plt.title("Forest anomalies")
            plt.axis("equal")
            plt.axis("off")
            plt.colorbar(label="score")
            plt.tight_layout()
            plt.savefig(out / "forest_anomalies.png", dpi=180)
            plt.close()
    print(f"Saved forest maps to {out}")


if __name__ == "__main__":
    main()
