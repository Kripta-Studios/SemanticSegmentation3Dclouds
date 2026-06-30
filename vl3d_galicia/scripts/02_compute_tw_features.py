from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.taubin_weingarten import TW_FEATURE_NAMES, compute_taubin_weingarten_features
from src.utils.progress import eta_line


def torch_save_atomic(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def block_files(root: str, split: str) -> list[str]:
    return sorted(glob.glob(os.path.join(root, split, "*.pt")))


def fit_stats(files: list[str], args) -> dict:
    samples = []
    selected = files if args.fit_stats_blocks == 0 else files[: args.fit_stats_blocks]
    rng = np.random.default_rng(args.seed)
    start_time = time.perf_counter()
    for i, path in enumerate(tqdm(selected, desc="fit TW stats", unit="block"), 1):
        data = torch.load(path, weights_only=False, map_location="cpu")
        tw = compute_taubin_weingarten_features(
            data["coords"].numpy(),
            neighbor_mode=args.neighbor_mode,
            k_neighbors=args.k_neighbors,
            radius=args.radius,
            min_neighbors=args.min_neighbors,
            eigenthreshold=args.eigenthreshold,
            tikhonov=args.tikhonov,
        )
        valid = tw[:, -1] > 0
        values = tw[valid, :-1]
        if values.size == 0:
            continue
        take = min(args.sample_points_per_block, values.shape[0])
        idx = rng.choice(values.shape[0], take, replace=False)
        samples.append(values[idx])
        if i % 100 == 0 or i == len(selected):
            print(eta_line("fit TW stats", start_time, i, len(selected)))
    if not samples:
        dim = len(TW_FEATURE_NAMES) - 1
        return {"p01": [0.0] * dim, "p99": [1.0] * dim, "mean": [0.0] * dim, "std": [1.0] * dim}
    arr = np.concatenate(samples, axis=0)
    p01 = np.percentile(arr, 1, axis=0)
    p99 = np.percentile(arr, 99, axis=0)
    clipped = np.clip(arr, p01, p99)
    mean = clipped.mean(axis=0)
    std = clipped.std(axis=0)
    std[std < 1e-6] = 1.0
    return {"p01": p01.tolist(), "p99": p99.tolist(), "mean": mean.tolist(), "std": std.tolist()}


def enrich_file(path: str, out_path: Path, stats: dict, args) -> dict:
    if not args.no_skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        return {"points": 0, "valid_points": 0, "written_blocks": 0, "skipped_blocks": 1}
    data = torch.load(path, weights_only=False, map_location="cpu")
    tw = compute_taubin_weingarten_features(
        data["coords"].numpy(),
        neighbor_mode=args.neighbor_mode,
        k_neighbors=args.k_neighbors,
        radius=args.radius,
        min_neighbors=args.min_neighbors,
        eigenthreshold=args.eigenthreshold,
        tikhonov=args.tikhonov,
    )
    p01 = np.asarray(stats["p01"], dtype=np.float32)
    p99 = np.asarray(stats["p99"], dtype=np.float32)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    tw_values = tw[:, :-1]
    valid = tw[:, -1:] > 0
    tw_norm = (np.clip(tw_values, p01, p99) - mean) / std
    tw_norm[~valid[:, 0]] = 0.0
    tw_final = np.concatenate([tw_norm, valid.astype(np.float32)], axis=1).astype(np.float32)
    base = data.get("features_original", data["features"]).float()
    data["features"] = base
    data.pop("features_original", None)
    data["tw_features"] = torch.from_numpy(tw_final)
    if args.store_combined:
        data["features_with_tw"] = torch.cat([base, data["tw_features"]], dim=1)
    else:
        data.pop("features_with_tw", None)
    data["tw_valid_mask"] = torch.from_numpy(valid[:, 0])
    torch_save_atomic(data, out_path)
    return {"points": int(tw.shape[0]), "valid_points": int(valid.sum()), "written_blocks": 1, "skipped_blocks": 0}


def enrich_worker(payload: dict) -> dict:
    args = SimpleNamespace(**payload["args"])
    return enrich_file(payload["path"], Path(payload["out_path"]), payload["stats"], args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Taubin-Weingarten descriptors for block tensors.")
    parser.add_argument("--input", default="data/processed/galicia_blocks")
    parser.add_argument("--out", default="data/processed/galicia_blocks_tw")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--neighbor-mode", choices=["knn", "radius"], default="knn")
    parser.add_argument("--k-neighbors", type=int, default=32)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--min-neighbors", type=int, default=10)
    parser.add_argument("--eigenthreshold", type=float, default=1e-5)
    parser.add_argument("--tikhonov", type=float, default=1e-6)
    parser.add_argument("--fit-stats-blocks", type=int, default=500)
    parser.add_argument("--stats-json", default="", help="Reuse TW normalization stats fitted on training data.")
    parser.add_argument("--sample-points-per-block", type=int, default=256)
    parser.add_argument("--max-blocks", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--store-combined", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    train_files = block_files(args.input, "train")
    if args.max_blocks > 0:
        train_files = train_files[: args.max_blocks]
    Path(args.reports).mkdir(parents=True, exist_ok=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    stats_path = Path(args.reports) / "tw_normalization.json"
    if args.stats_json:
        stats = json.loads(Path(args.stats_json).read_text(encoding="utf-8"))
        stats["reused_from"] = str(Path(args.stats_json).resolve())
        print(f"Reusing TW normalization stats from {args.stats_json}")
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    elif stats_path.exists() and not args.no_skip_existing:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        print(f"Reusing TW normalization stats from {stats_path}")
    else:
        stats = fit_stats(train_files, args)
        stats["feature_names"] = TW_FEATURE_NAMES[:-1]
        stats["tw_method"] = "taubin_weingarten_implicit_quadric"
        stats["fit_input"] = str(Path(args.input).resolve())
        stats["fit_stats_blocks"] = args.fit_stats_blocks
        stats["sample_points_per_block"] = args.sample_points_per_block
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    report = {"splits": {}, "stats": stats}
    worker_args = vars(args).copy()
    for split in args.splits:
        files = block_files(args.input, split)
        if args.max_blocks > 0:
            files = files[: args.max_blocks]
        total = 0
        valid = 0
        written = 0
        skipped = 0
        split_start = time.perf_counter()
        if args.num_workers <= 1:
            for i, path in enumerate(tqdm(files, desc=f"TW {split}", unit="block"), 1):
                out_path = Path(args.out) / split / Path(path).name
                item = enrich_file(path, out_path, stats, args)
                total += item["points"]
                valid += item["valid_points"]
                written += item["written_blocks"]
                skipped += item["skipped_blocks"]
                if i % 100 == 0 or i == len(files):
                    print(eta_line(f"TW {split}", split_start, i, len(files)))
        else:
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [
                    executor.submit(
                        enrich_worker,
                        {
                            "path": path,
                            "out_path": str(Path(args.out) / split / Path(path).name),
                            "stats": stats,
                            "args": worker_args,
                        },
                    )
                    for path in files
                ]
                for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc=f"TW {split}", unit="block"), 1):
                    item = future.result()
                    total += item["points"]
                    valid += item["valid_points"]
                    written += item["written_blocks"]
                    skipped += item["skipped_blocks"]
                    if i % 100 == 0 or i == len(files):
                        print(eta_line(f"TW {split}", split_start, i, len(files)))
        report["splits"][split] = {
            "blocks": len(files),
            "written_blocks": written,
            "skipped_blocks": skipped,
            "points": total,
            "valid_points": valid,
        }
    (Path(args.reports) / "tw_compute_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (Path(args.out) / "_tw_complete.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["splits"], indent=2))


if __name__ == "__main__":
    main()
