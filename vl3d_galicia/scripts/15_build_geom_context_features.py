from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.geom_context import GeomContextConfig, build_geom_context_features, geom_context_schema
from src.training.segmentation_trainer import torch_save_atomic
from src.utils.progress import eta_line


def iter_block_paths(data_root: Path, splits: list[str], max_blocks_per_split: int = 0) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for split in splits:
        files = sorted((data_root / split).glob("*.pt"))
        if max_blocks_per_split > 0:
            files = files[:max_blocks_per_split]
        paths.extend((split, path) for path in files)
    return paths


def build_one(payload: dict) -> dict:
    split = payload["split"]
    path = Path(payload["path"])
    out_path = Path(payload["out_path"])
    force = bool(payload["force"])
    cell_sizes = tuple(float(v) for v in payload["cell_sizes"])
    include_tw_summary = bool(payload["include_tw_summary"])
    include_metric_height = bool(payload["include_metric_height"])
    feature_key = payload["feature_key"]
    split_hash = payload["split_hash"]
    block = torch.load(path, weights_only=False, map_location="cpu")
    config = GeomContextConfig(
        cell_sizes=cell_sizes,
        include_tw_summary=include_tw_summary,
        include_metric_height=include_metric_height,
    )
    tw_count = int(block["tw_features"].shape[1]) if "tw_features" in block else 0
    schema = geom_context_schema(config, tw_feature_count=tw_count)
    if out_path.exists() and not force:
        cached = torch.load(out_path, weights_only=False, map_location="cpu")
        if cached.get("feature_schema_sha256") == schema.sha256 and cached.get("split_hash") == split_hash:
            return {
                "split": split,
                "path": str(path),
                "out": str(out_path),
                "written": False,
                "points": 0,
                "feature_dim": schema.dimension,
                "feature_schema_sha256": schema.sha256,
            }
        raise ValueError(f"Existing geometric cache is incompatible and was preserved: {out_path}")
    features, names = build_geom_context_features(
        block,
        config,
    )
    torch_save_atomic(
        {
            feature_key: features.float(),
            "feature_names": names,
            "feature_dim": int(features.shape[1]),
            "point_count": int(features.shape[0]),
            "source_block": str(path),
            "cell_sizes": list(cell_sizes),
            "include_tw_summary": include_tw_summary,
            "include_metric_height": include_metric_height,
            "feature_schema": schema.as_dict(),
            "feature_schema_version": schema.version,
            "feature_schema_sha256": schema.sha256,
            "uses_labels": False,
            "tile_id": block.get("tile_id"),
            "split": block.get("split"),
            "split_hash": split_hash,
            "git_commit": payload["git_commit"],
        },
        out_path,
    )
    return {
        "split": split,
        "path": str(path),
        "out": str(out_path),
        "written": True,
        "points": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "feature_schema_sha256": schema.sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build non-supervised local geometric/spectral context features for Galicia blocks.")
    parser.add_argument("--data", default="data/processed/galicia_blocks_medium_tw")
    parser.add_argument("--out", default="data/processed/galicia_blocks_medium_geom_context")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--feature-key", default="geom_features")
    parser.add_argument("--cell-sizes", default="2.5,5.0,10.0")
    parser.add_argument("--no-tw-summary", action="store_true")
    parser.add_argument("--no-metric-height", action="store_true")
    parser.add_argument("--max-blocks-per-split", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data)
    out_root = Path(args.out)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    cell_sizes = [float(item.strip()) for item in args.cell_sizes.split(",") if item.strip()]
    paths = iter_block_paths(data_root, splits, max_blocks_per_split=args.max_blocks_per_split)
    input_manifest = {}
    for name in ("_dataset_manifest.json", "_prepare_complete.json"):
        candidate = data_root / name
        if candidate.exists():
            input_manifest = json.loads(candidate.read_text(encoding="utf-8"))
            break
    input_run = input_manifest.get("run", input_manifest)
    split_hash = input_run.get("split_hash") or input_manifest.get("split_hash")
    if not split_hash:
        raise ValueError("Geometric context features require a validated split_hash")
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "data_root": str(data_root),
        "out_root": str(out_root),
        "splits": splits,
        "feature_key": args.feature_key,
        "cell_sizes": cell_sizes,
        "include_tw_summary": not args.no_tw_summary,
        "include_metric_height": not args.no_metric_height,
        "max_blocks_per_split": int(args.max_blocks_per_split),
        "num_workers": int(args.num_workers),
        "uses_labels": False,
        "split_hash": split_hash,
        "git_commit": git_commit,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_root / "feature_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    jobs = []
    for split, path in paths:
        out_path = out_root / split / path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        jobs.append(
            {
                "split": split,
                "path": str(path),
                "out_path": str(out_path),
                "force": bool(args.force),
                "cell_sizes": cell_sizes,
                "include_tw_summary": not args.no_tw_summary,
                "include_metric_height": not args.no_metric_height,
                "feature_key": args.feature_key,
                "split_hash": split_hash,
                "git_commit": git_commit,
            }
        )

    start = time.perf_counter()
    written = 0
    skipped = 0
    total_points = 0
    feature_dim = 0
    schema_hashes: set[str] = set()
    if args.num_workers <= 1:
        iterator = enumerate((build_one(job) for job in jobs), 1)
        for done, info in iterator:
            written += int(bool(info["written"]))
            skipped += int(not info["written"])
            total_points += int(info["points"])
            feature_dim = max(feature_dim, int(info["feature_dim"]))
            schema_hashes.add(str(info["feature_schema_sha256"]))
            if done % 100 == 0 or done == len(jobs):
                print(eta_line("geom context feature cache", start, done, len(jobs)))
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(build_one, job) for job in jobs]
            for done, future in enumerate(as_completed(futures), 1):
                info = future.result()
                written += int(bool(info["written"]))
                skipped += int(not info["written"])
                total_points += int(info["points"])
                feature_dim = max(feature_dim, int(info["feature_dim"]))
                schema_hashes.add(str(info["feature_schema_sha256"]))
                if done % 100 == 0 or done == len(jobs):
                    print(eta_line("geom context feature cache", start, done, len(jobs)))

    manifest.update(
        {
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "blocks_total": len(jobs),
            "blocks_written": written,
            "blocks_skipped": skipped,
            "points_written": total_points,
            "feature_dim": feature_dim,
            "feature_schema_sha256": next(iter(schema_hashes)) if len(schema_hashes) == 1 else None,
            "feature_schema_hashes": sorted(schema_hashes),
        }
    )
    manifest["cache_hash"] = hashlib.sha256(
        json.dumps(
            {
                "split_hash": split_hash,
                "feature_schema_sha256": manifest.get("feature_schema_sha256"),
                "cell_sizes": cell_sizes,
                "include_tw_summary": not args.no_tw_summary,
                "include_metric_height": not args.no_metric_height,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    (out_root / "feature_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
