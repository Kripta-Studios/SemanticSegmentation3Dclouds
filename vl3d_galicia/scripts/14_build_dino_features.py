from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.raster_dino import DinoDenseExtractor, make_multichannel_raster, raster_to_image
from src.data.pnoa import PNOA_FEATURE_SCHEMA
from src.training.segmentation_trainer import torch_save_atomic
from src.utils.progress import eta_line


DINO_CACHE_SCHEMA_VERSION = "dino-raster-cache-v2-named-pnoa"
EXTERNAL_FEATURE_SCHEMA_VERSION = "dino-external-feature-v1"


def file_sha256(path: str | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def external_feature_schema(
    extractor: DinoDenseExtractor,
    grid_size: int,
    image_mode: str,
    tw_channels: int,
    out_dim: int,
    projection_seed: int,
    include_stat_features: bool,
) -> dict:
    requested_backbone = extractor.requested_model_name if extractor.backend != "stat" else "stat_features"
    actual_backbone = extractor.model_name if extractor.uses_real_dino else "stat_features"
    weights_sha256 = file_sha256(extractor.weights)
    schema = {
        "version": EXTERNAL_FEATURE_SCHEMA_VERSION,
        "cache_schema_version": DINO_CACHE_SCHEMA_VERSION,
        "feature_key": "dino_features",
        "dtype": "float32",
        "feature_dim": int(out_dim),
        "grid_size": int(grid_size),
        "image_mode": image_mode,
        "tw_channels": int(tw_channels),
        "projection_seed": int(projection_seed),
        "include_stat_features": bool(include_stat_features),
        "normalization": extractor.normalize,
        "requested_backend": extractor.backend,
        "actual_backend": extractor.real_backend,
        "requested_backbone": requested_backbone,
        "actual_backbone": actual_backbone,
        "used_real_dino": bool(extractor.uses_real_dino),
        "weights_sha256": weights_sha256,
        "backbone_identity_matches": requested_backbone == actual_backbone,
        "input_feature_schema_sha256": PNOA_FEATURE_SCHEMA.sha256,
    }
    encoded = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    promotion_eligible = bool(
        schema["used_real_dino"] and schema["backbone_identity_matches"] and schema["weights_sha256"]
    )
    return {**schema, "promotion_eligible": promotion_eligible, "sha256": hashlib.sha256(encoded).hexdigest()}


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
    schema = external_feature_schema(
        extractor,
        grid_size=grid_size,
        image_mode=image_mode,
        tw_channels=tw_channels,
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
        "requested_backbone": schema["requested_backbone"],
        "actual_backbone": schema["actual_backbone"],
        "artifact_kind": "dino_features" if extractor.uses_real_dino else "stat_raster_baseline",
        "promotion_eligible": bool(schema["promotion_eligible"]),
        "include_stat_features": bool(include_stat_features),
        "cache_schema_version": DINO_CACHE_SCHEMA_VERSION,
        "input_feature_schema": PNOA_FEATURE_SCHEMA.as_dict(),
        "input_feature_schema_version": PNOA_FEATURE_SCHEMA.version,
        "input_feature_schema_sha256": PNOA_FEATURE_SCHEMA.sha256,
        "feature_schema": schema,
        "feature_schema_sha256": schema["sha256"],
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
    parser.add_argument("--backend", choices=["auto", "hf", "timm", "torchhub", "dinov2", "stat"], default="auto")
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
    parser.add_argument("--feature-schema-version", default=PNOA_FEATURE_SCHEMA.version)
    args = parser.parse_args()

    if args.feature_schema_version != PNOA_FEATURE_SCHEMA.version:
        raise ValueError(
            f"Requested feature schema {args.feature_schema_version!r}, but this code implements "
            f"{PNOA_FEATURE_SCHEMA.version!r}"
        )

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
    schema = external_feature_schema(
        extractor,
        grid_size=args.grid_size,
        image_mode=args.image_mode,
        tw_channels=args.tw_channels,
        out_dim=args.out_dim,
        projection_seed=args.projection_seed,
        include_stat_features=not args.no_stat_features,
    )
    previous_manifest = None
    config_path = out_root / "feature_config.json"
    if config_path.exists():
        previous_manifest = json.loads(config_path.read_text(encoding="utf-8"))
    cache_compatible = bool(
        previous_manifest
        and previous_manifest.get("cache_schema_version") == DINO_CACHE_SCHEMA_VERSION
        and previous_manifest.get("input_feature_schema_sha256") == PNOA_FEATURE_SCHEMA.sha256
        and previous_manifest.get("feature_schema_sha256") == schema["sha256"]
        and previous_manifest.get("image_mode") == args.image_mode
        and int(previous_manifest.get("grid_size", -1)) == int(args.grid_size)
        and previous_manifest.get("backend_requested") == args.backend
        and previous_manifest.get("model") == args.model
        and bool(previous_manifest.get("normalize", False)) == bool(args.normalize)
        and int(previous_manifest.get("tw_channels", -1)) == int(args.tw_channels)
        and int(previous_manifest.get("out_dim", -1)) == int(args.out_dim)
        and int(previous_manifest.get("projection_seed", -1)) == int(args.projection_seed)
        and bool(previous_manifest.get("include_stat_features", False)) == bool(not args.no_stat_features)
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
        "requested_backbone": schema["requested_backbone"],
        "actual_backbone": schema["actual_backbone"],
        "artifact_kind": "dino_features" if extractor.uses_real_dino else "stat_raster_baseline",
        "promotion_eligible": bool(schema["promotion_eligible"]),
        "promotion_blocker": (
            None
            if schema["promotion_eligible"]
            else "requires real DINO, identical requested/actual backbone, and SHA256 of explicit weights"
        ),
        "cache_schema_version": DINO_CACHE_SCHEMA_VERSION,
        "input_feature_schema": PNOA_FEATURE_SCHEMA.as_dict(),
        "input_feature_schema_version": PNOA_FEATURE_SCHEMA.version,
        "input_feature_schema_sha256": PNOA_FEATURE_SCHEMA.sha256,
        "feature_schema": schema,
        "feature_schema_sha256": schema["sha256"],
        "previous_cache_compatible": cache_compatible,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

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
        if out_path.exists() and not args.force and cache_compatible:
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
    config_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
