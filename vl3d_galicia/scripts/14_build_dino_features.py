from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.raster_dino import DinoDenseExtractor, make_multichannel_raster, raster_to_image
from src.training.segmentation_trainer import torch_save_atomic
from src.utils.progress import eta_line


def iter_block_paths(
    data_root: Path,
    splits: list[str],
    max_blocks_per_split: int = 0,
    max_train_blocks: int = 0,
    max_val_blocks: int = 0,
    max_test_blocks: int = 0,
):
    split_limits = {
        "train": int(max_train_blocks),
        "val": int(max_val_blocks),
        "test": int(max_test_blocks),
    }
    for split in splits:
        split_dir = data_root / split
        files = sorted(split_dir.glob("*.pt"))
        limit = split_limits.get(split, 0)
        if limit <= 0:
            limit = int(max_blocks_per_split)
        if limit > 0:
            files = files[:limit]
        for path in files:
            yield split, path


def build_one(
    path: Path,
    out_path: Path,
    extractor: DinoDenseExtractor,
    grid_size: int,
    image_mode: str,
    tw_channels: int,
    out_dim: int,
    projection_seed: int,
    include_stat_features: bool,
) -> dict:
    block = torch.load(path, weights_only=False, map_location="cpu")
    rasterized = make_multichannel_raster(block, grid_size=grid_size, tw_channels=tw_channels)
    rasterized = type(rasterized)(
        image=raster_to_image(rasterized.raster, rasterized.channel_names, mode=image_mode),
        raster=rasterized.raster,
        point_cells=rasterized.point_cells,
        channel_names=rasterized.channel_names,
    )
    features = extractor.point_features(
        rasterized,
        out_dim=out_dim,
        projection_seed=projection_seed,
        include_stat_features=include_stat_features,
    )
    payload = {
        "dino_features": features.cpu().float(),
        "source_block": str(path),
        "feature_dim": int(features.shape[1]),
        "point_count": int(features.shape[0]),
        "grid_size": int(grid_size),
        "image_mode": image_mode,
        "tw_channels": int(tw_channels),
        "raster_channels": rasterized.channel_names,
        "dino_backend_requested": extractor.backend,
        "dino_backend": extractor.real_backend,
        "dino_model": extractor.model_name,
        "used_real_dino": bool(extractor.uses_real_dino),
        "include_stat_features": bool(include_stat_features),
    }
    torch_save_atomic(payload, out_path)
    return {
        "path": str(path),
        "out": str(out_path),
        "points": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DINO/DINO-like dense raster features for Galicia point blocks.")
    parser.add_argument("--data", default="data/processed/galicia_blocks_medium_tw")
    parser.add_argument("--out", default="data/processed/galicia_blocks_medium_dino")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--backend", choices=["auto", "hf", "timm", "torchhub", "dinov2", "stat"], default="stat")
    parser.add_argument("--model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--repo-dir", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--normalize", choices=["imagenet", "sat493m"], default="imagenet")
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--image-mode", choices=["rgb", "cir", "height", "nir_height_density", "rgb_nir_height"], default="rgb_nir_height")
    parser.add_argument("--tw-channels", type=int, default=8)
    parser.add_argument("--out-dim", type=int, default=64)
    parser.add_argument("--projection-seed", type=int, default=13)
    parser.add_argument("--max-blocks-per-split", type=int, default=0)
    parser.add_argument("--max-train-blocks", type=int, default=0)
    parser.add_argument("--max-val-blocks", type=int, default=0)
    parser.add_argument("--max-test-blocks", type=int, default=0)
    parser.add_argument("--no-stat-features", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data)
    out_root = Path(args.out)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    extractor = DinoDenseExtractor(
        backend=args.backend,
        model_name=args.model,
        repo_dir=args.repo_dir,
        weights=args.weights,
        device=args.device,
        normalize=args.normalize,
    )
    manifest = {
        "data_root": str(data_root),
        "out_root": str(out_root),
        "splits": splits,
        "backend_requested": args.backend,
        "backend_used": extractor.real_backend,
        "model": args.model,
        "repo_dir": args.repo_dir,
        "weights": args.weights,
        "normalize": args.normalize,
        "grid_size": args.grid_size,
        "image_mode": args.image_mode,
        "tw_channels": args.tw_channels,
        "out_dim": args.out_dim,
        "projection_seed": args.projection_seed,
        "include_stat_features": not args.no_stat_features,
        "used_real_dino": bool(extractor.uses_real_dino),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "feature_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    paths = list(
        iter_block_paths(
            data_root,
            splits,
            max_blocks_per_split=args.max_blocks_per_split,
            max_train_blocks=args.max_train_blocks,
            max_val_blocks=args.max_val_blocks,
            max_test_blocks=args.max_test_blocks,
        )
    )
    start = time.perf_counter()
    done = 0
    written = 0
    skipped = 0
    total_points = 0
    for split, path in tqdm(paths, desc="DINO feature blocks"):
        out_path = out_root / split / path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        done += 1
        if out_path.exists() and not args.force:
            skipped += 1
            if done % 50 == 0 or done == len(paths):
                print(eta_line("DINO feature cache", start, done, len(paths)))
            continue
        info = build_one(
            path,
            out_path,
            extractor,
            grid_size=args.grid_size,
            image_mode=args.image_mode,
            tw_channels=args.tw_channels,
            out_dim=args.out_dim,
            projection_seed=args.projection_seed,
            include_stat_features=not args.no_stat_features,
        )
        written += 1
        total_points += int(info["points"])
        if done % 25 == 0 or done == len(paths):
            print(eta_line("DINO feature cache", start, done, len(paths)))

    manifest.update(
        {
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "blocks_total": len(paths),
            "blocks_written": written,
            "blocks_skipped": skipped,
            "points_written": total_points,
        }
    )
    (out_root / "feature_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
