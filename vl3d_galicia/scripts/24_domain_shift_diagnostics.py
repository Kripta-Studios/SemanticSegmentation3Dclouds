from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.classes import CLASS_NAMES, IGNORE_INDEX


BASE_FEATURE_NAMES = ["intensity", "red", "green", "blue", "nir"]


def parse_feature_names(example: dict, use_tw: bool, geom_feature_dir: Path | None, geom_key: str) -> list[str]:
    names = ["local_x", "local_y", "height_above_block_min", *BASE_FEATURE_NAMES]
    if use_tw and "tw_features" in example:
        names.extend([f"tw_{i:02d}" for i in range(int(example["tw_features"].shape[1]))])
    if geom_feature_dir is not None:
        geom = load_external(example["_path"], geom_feature_dir, geom_key)
        if geom is not None:
            names.extend([f"geom_{i:02d}" for i in range(int(geom.shape[1]))])
    return names


def list_files(root: Path, split: str) -> list[Path]:
    split_dir = root / split
    if not split_dir.exists():
        return []
    return sorted(split_dir.glob("*.pt"))


def sample_files(files: list[Path], max_blocks: int, seed: int) -> list[Path]:
    if max_blocks <= 0 or len(files) <= max_blocks:
        return files
    rng = random.Random(seed)
    return sorted(rng.sample(files, max_blocks))


def load_external(source_path: Path, feature_dir: Path, key: str) -> torch.Tensor | None:
    candidates = [
        feature_dir / source_path.parent.name / source_path.name,
        feature_dir / source_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            payload = torch.load(candidate, map_location="cpu", weights_only=False)
            return payload[key].float()
    return None


def feature_matrix(data: dict, path: Path, use_tw: bool, geom_feature_dir: Path | None, geom_key: str) -> np.ndarray:
    cols = [data["coords"].float(), data["features_original"].float() if "features_original" in data else data["features"].float()]
    if use_tw and "tw_features" in data:
        cols.append(data["tw_features"].float())
    if geom_feature_dir is not None:
        ext = load_external(path, geom_feature_dir, geom_key)
        if ext is not None:
            cols.append(ext.float())
    return torch.cat(cols, dim=1).numpy().astype(np.float64, copy=False)


def weighted_mean(values: list[float], weights: list[int]) -> float:
    denom = sum(weights)
    if denom <= 0:
        return 0.0
    return float(sum(v * w for v, w in zip(values, weights)) / denom)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p = p / max(float(p.sum()), 1e-12)
    q = q / max(float(q.sum()), 1e-12)
    m = 0.5 * (p + q)
    mask_p = p > 0
    mask_q = q > 0
    kl_pm = float(np.sum(p[mask_p] * np.log2(p[mask_p] / m[mask_p])))
    kl_qm = float(np.sum(q[mask_q] * np.log2(q[mask_q] / m[mask_q])))
    return 0.5 * (kl_pm + kl_qm)


def histogram(values: list[float], bins: int = 50, lo: float | None = None, hi: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.zeros(bins), np.linspace(0.0, 1.0, bins + 1)
    if lo is None:
        lo = float(np.nanpercentile(arr, 0.5))
    if hi is None:
        hi = float(np.nanpercentile(arr, 99.5))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr) + 1e-6)
    counts, edges = np.histogram(arr, bins=bins, range=(lo, hi))
    return counts.astype(np.float64), edges


def collect_stats(
    name: str,
    root: Path,
    split: str,
    max_blocks: int,
    max_points_per_block: int,
    seed: int,
    use_tw: bool,
    geom_feature_dir: Path | None,
    geom_key: str,
) -> dict:
    files = sample_files(list_files(root, split), max_blocks=max_blocks, seed=seed)
    if not files:
        raise SystemExit(f"No block files found for {name}: {root / split}")
    rng = np.random.default_rng(seed)
    class_counts = Counter()
    block_point_counts = []
    reliable_ratios = []
    missing_nir_ratios = []
    z_ranges = []
    z_p95s = []
    z_means = []
    density_points_m2 = []
    feature_values: dict[str, list[float]] = defaultdict(list)
    class_feature_values: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    first = torch.load(files[0], map_location="cpu", weights_only=False)
    first["_path"] = files[0]
    feature_names = parse_feature_names(first, use_tw=use_tw, geom_feature_dir=geom_feature_dir, geom_key=geom_key)

    for path in files:
        data = torch.load(path, map_location="cpu", weights_only=False)
        labels = data["labels"].long().numpy()
        reliable = data.get("reliable_mask", torch.from_numpy(labels != IGNORE_INDEX)).bool().numpy()
        valid_labels = labels[reliable & (labels != IGNORE_INDEX)]
        class_counts.update({int(k): int(v) for k, v in zip(*np.unique(valid_labels, return_counts=True))})
        n = int(labels.shape[0])
        block_point_counts.append(n)
        reliable_ratios.append(float(valid_labels.size / max(n, 1)))
        if "missing_nir_mask" in data:
            missing_nir_ratios.append(float(data["missing_nir_mask"].float().mean()))
        coords = data["coords"].float().numpy()
        z = coords[:, 2].astype(np.float64)
        z_ranges.append(float(np.nanmax(z) - np.nanmin(z)))
        z_p95s.append(float(np.nanpercentile(z, 95)))
        z_means.append(float(np.nanmean(z)))
        bbox = data.get("bbox", None)
        if bbox and len(bbox) == 4:
            area = max(float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])), 1.0)
            density_points_m2.append(n / area)
        x = feature_matrix(data, path, use_tw=use_tw, geom_feature_dir=geom_feature_dir, geom_key=geom_key)
        take = min(max_points_per_block, x.shape[0])
        if take < x.shape[0]:
            idx = rng.choice(x.shape[0], take, replace=False)
        else:
            idx = np.arange(x.shape[0])
        x_sample = x[idx]
        labels_sample = labels[idx]
        reliable_sample = reliable[idx] & (labels_sample != IGNORE_INDEX)
        for j, fname in enumerate(feature_names[: x_sample.shape[1]]):
            vals = x_sample[:, j]
            vals = vals[np.isfinite(vals)]
            feature_values[fname].extend(vals.tolist())
            for cls in range(IGNORE_INDEX):
                cls_vals = x_sample[(labels_sample == cls) & reliable_sample, j]
                if cls_vals.size:
                    cls_vals = cls_vals[np.isfinite(cls_vals)]
                    class_feature_values[fname][cls].extend(cls_vals.tolist())

    total_reliable = sum(class_counts.values())
    return {
        "name": name,
        "root": str(root),
        "split": split,
        "files_total": len(list_files(root, split)),
        "files_sampled": len(files),
        "points_sampled_blocks": int(sum(block_point_counts)),
        "class_counts": {str(k): int(class_counts.get(k, 0)) for k in range(IGNORE_INDEX)},
        "class_pct": {str(k): float(class_counts.get(k, 0) / max(total_reliable, 1)) for k in range(IGNORE_INDEX)},
        "block_point_counts": block_point_counts,
        "reliable_ratios": reliable_ratios,
        "missing_nir_ratios": missing_nir_ratios,
        "z_ranges": z_ranges,
        "z_p95s": z_p95s,
        "z_means": z_means,
        "density_points_m2": density_points_m2,
        "feature_names": feature_names,
        "feature_values": feature_values,
        "class_feature_values": class_feature_values,
    }


def summary_rows(stats: dict) -> list[dict]:
    rows = []
    for key in ["block_point_counts", "reliable_ratios", "missing_nir_ratios", "z_ranges", "z_p95s", "z_means", "density_points_m2"]:
        arr = np.asarray(stats.get(key, []), dtype=np.float64)
        if arr.size == 0:
            continue
        rows.append(
            {
                "dataset": stats["name"],
                "metric": key,
                "mean": float(np.nanmean(arr)),
                "p05": float(np.nanpercentile(arr, 5)),
                "p50": float(np.nanpercentile(arr, 50)),
                "p95": float(np.nanpercentile(arr, 95)),
            }
        )
    return rows


def feature_shift_rows(a: dict, b: dict, top_k: int = 80) -> list[dict]:
    rows = []
    for fname in a["feature_names"]:
        va = a["feature_values"].get(fname, [])
        vb = b["feature_values"].get(fname, [])
        if len(va) < 10 or len(vb) < 10:
            continue
        combined = np.asarray(random.sample(va, min(len(va), 20000)) + random.sample(vb, min(len(vb), 20000)), dtype=np.float64)
        lo = float(np.nanpercentile(combined, 0.5))
        hi = float(np.nanpercentile(combined, 99.5))
        ha, _ = histogram(va, lo=lo, hi=hi)
        hb, _ = histogram(vb, lo=lo, hi=hi)
        ma, mb = float(np.nanmean(va)), float(np.nanmean(vb))
        sa, sb = float(np.nanstd(va)), float(np.nanstd(vb))
        denom = max((sa + sb) * 0.5, 1e-9)
        rows.append(
            {
                "feature": fname,
                "mean_a": ma,
                "mean_b": mb,
                "std_a": sa,
                "std_b": sb,
                "abs_mean_delta": abs(mb - ma),
                "standardized_mean_delta": abs(mb - ma) / denom,
                "js_divergence": js_divergence(ha, hb),
                "n_a": len(va),
                "n_b": len(vb),
            }
        )
    return sorted(rows, key=lambda r: (r["js_divergence"], r["standardized_mean_delta"]), reverse=True)[:top_k]


def class_feature_shift_rows(a: dict, b: dict, features: Iterable[str]) -> list[dict]:
    rows = []
    for fname in features:
        for cls in range(IGNORE_INDEX):
            va = a["class_feature_values"].get(fname, {}).get(cls, [])
            vb = b["class_feature_values"].get(fname, {}).get(cls, [])
            if len(va) < 50 or len(vb) < 50:
                continue
            ma, mb = float(np.nanmean(va)), float(np.nanmean(vb))
            sa, sb = float(np.nanstd(va)), float(np.nanstd(vb))
            rows.append(
                {
                    "class_id": cls,
                    "class_name": CLASS_NAMES[cls],
                    "feature": fname,
                    "mean_a": ma,
                    "mean_b": mb,
                    "std_a": sa,
                    "std_b": sb,
                    "standardized_mean_delta": abs(mb - ma) / max((sa + sb) * 0.5, 1e-9),
                    "n_a": len(va),
                    "n_b": len(vb),
                }
            )
    return sorted(rows, key=lambda r: r["standardized_mean_delta"], reverse=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compact_stats(stats: dict) -> dict:
    return {
        key: value
        for key, value in stats.items()
        if key not in {"feature_values", "class_feature_values"}
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Galicia vs external holdout domain shift.")
    parser.add_argument("--a-name", default="galicia_test")
    parser.add_argument("--a-root", default="data/processed/galicia_blocks_medium_tw")
    parser.add_argument("--a-split", default="test")
    parser.add_argument("--a-geom-feature-dir", default="data/processed/galicia_blocks_medium_geom_context")
    parser.add_argument("--b-name", default="cat32_external")
    parser.add_argument("--b-root", default="data/processed/pnoa_varias_ccaa_holdout_cat32_tw")
    parser.add_argument("--b-split", default="test")
    parser.add_argument("--b-geom-feature-dir", default="data/processed/pnoa_varias_ccaa_holdout_cat32_geom_context")
    parser.add_argument("--geom-key", default="geom_features")
    parser.add_argument("--no-tw", action="store_true")
    parser.add_argument("--no-geom", action="store_true")
    parser.add_argument("--max-blocks", type=int, default=3000)
    parser.add_argument("--max-points-per-block", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="reports/domain_shift_cat32")
    args = parser.parse_args()

    out = Path(args.out)
    use_tw = not args.no_tw
    a_geom = None if args.no_geom else Path(args.a_geom_feature_dir)
    b_geom = None if args.no_geom else Path(args.b_geom_feature_dir)
    a = collect_stats(
        args.a_name,
        Path(args.a_root),
        args.a_split,
        args.max_blocks,
        args.max_points_per_block,
        args.seed,
        use_tw,
        a_geom,
        args.geom_key,
    )
    b = collect_stats(
        args.b_name,
        Path(args.b_root),
        args.b_split,
        args.max_blocks,
        args.max_points_per_block,
        args.seed + 1,
        use_tw,
        b_geom,
        args.geom_key,
    )
    class_rows = []
    for cls in range(IGNORE_INDEX):
        class_rows.append(
            {
                "class_id": cls,
                "class_name": CLASS_NAMES[cls],
                f"{args.a_name}_count": a["class_counts"][str(cls)],
                f"{args.a_name}_pct": a["class_pct"][str(cls)],
                f"{args.b_name}_count": b["class_counts"][str(cls)],
                f"{args.b_name}_pct": b["class_pct"][str(cls)],
                "pct_delta_b_minus_a": b["class_pct"][str(cls)] - a["class_pct"][str(cls)],
            }
        )
    global_rows = summary_rows(a) + summary_rows(b)
    shift_rows = feature_shift_rows(a, b)
    class_shift = class_feature_shift_rows(a, b, ["height_above_block_min", *BASE_FEATURE_NAMES, "tw_00", "tw_01", "tw_02", "geom_00", "geom_01", "geom_02"])

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "class_distribution.csv", class_rows)
    write_csv(out / "global_block_stats.csv", global_rows)
    write_csv(out / "feature_shift_top.csv", shift_rows)
    write_csv(out / "class_feature_shift_top.csv", class_shift[:120])
    (out / "summary.json").write_text(json.dumps({"a": compact_stats(a), "b": compact_stats(b)}, indent=2), encoding="utf-8")

    lines = [
        "# Domain Shift Diagnostics",
        "",
        f"A: `{args.a_name}` = `{args.a_root}/{args.a_split}`",
        f"B: `{args.b_name}` = `{args.b_root}/{args.b_split}`",
        "",
        "## Class Distribution",
        "",
        "|class|A pct|B pct|delta B-A|",
        "|---|---:|---:|---:|",
    ]
    for row in class_rows:
        lines.append(f"|{row['class_name']}|{row[f'{args.a_name}_pct']:.4f}|{row[f'{args.b_name}_pct']:.4f}|{row['pct_delta_b_minus_a']:+.4f}|")
    lines.extend(["", "## Largest Feature Shifts", "", "|feature|JS|std mean delta|mean A|mean B|", "|---|---:|---:|---:|---:|"])
    for row in shift_rows[:20]:
        lines.append(f"|{row['feature']}|{row['js_divergence']:.4f}|{row['standardized_mean_delta']:.3f}|{row['mean_a']:.4f}|{row['mean_b']:.4f}|")
    lines.extend(["", "## Largest Class-Conditional Shifts", "", "|class|feature|std mean delta|mean A|mean B|", "|---|---|---:|---:|---:|"])
    for row in class_shift[:20]:
        lines.append(f"|{row['class_name']}|{row['feature']}|{row['standardized_mean_delta']:.3f}|{row['mean_a']:.4f}|{row['mean_b']:.4f}|")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `class_distribution.csv`",
            "- `global_block_stats.csv`",
            "- `feature_shift_top.csv`",
            "- `class_feature_shift_top.csv`",
            "- `summary.json`",
            "",
        ]
    )
    (out / "domain_shift_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved diagnostics to {out}")


if __name__ == "__main__":
    main()
