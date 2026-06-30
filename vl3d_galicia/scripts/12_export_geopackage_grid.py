from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def bbox_wkt(x: float, y: float, size: float) -> str:
    h = size / 2.0
    x0, x1 = x - h, x + h
    y0, y1 = y - h, y + h
    return f"POLYGON(({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Geo-JEPA/Forest-JEPA grid metrics to GIS-friendly CSV and optional GPKG.")
    parser.add_argument("--grid-csv", default="outputs/forest_jepa/forest_grid_metrics.csv")
    parser.add_argument("--out-dir", default="outputs/pilot_demo/geopackage")
    parser.add_argument("--grid-size", type=float, default=2.0)
    parser.add_argument("--crs", default="EPSG:25829")
    args = parser.parse_args()

    grid = pd.read_csv(args.grid_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid["geometry_wkt"] = [bbox_wkt(x, y, args.grid_size) for x, y in zip(grid["x"], grid["y"])]
    csv_path = out_dir / "geo_jepa_predictions_grid.csv"
    grid.to_csv(csv_path, index=False)

    gpkg_path = out_dir / "geo_jepa_predictions.gpkg"
    try:
        import geopandas as gpd
        from shapely import wkt

        gdf = gpd.GeoDataFrame(grid.drop(columns=["geometry_wkt"]), geometry=grid["geometry_wkt"].map(wkt.loads), crs=args.crs)
        gdf.to_file(gpkg_path, layer="forest_jepa_grid", driver="GPKG")
        print(f"Saved {csv_path} and {gpkg_path}")
    except Exception as exc:
        print(f"Saved {csv_path}. GeoPackage skipped: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
