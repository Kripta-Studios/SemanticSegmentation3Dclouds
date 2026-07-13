from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.geographic_split import OFFICIAL_SPLITS, load_split_manifest


def selected_files(files: list[Path], limit: int, seed: int) -> list[Path]:
    files = sorted(files)
    if limit <= 0 or len(files) <= limit:
        return files
    rng = random.Random(int(seed))
    indices = list(range(len(files)))
    rng.shuffle(indices)
    return sorted(files[index] for index in indices[:limit])


def tile_id_from_block(path: Path) -> str:
    return path.stem.rsplit("_block_", 1)[0]


def aggregate_hash(files_by_split: dict[str, list[Path]]) -> str:
    payload = {
        split: [{"name": path.name, "size": path.stat().st_size} for path in paths]
        for split, paths in sorted(files_by_split.items())
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a label-blind, seed-independent block subset for all ablations.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--max-train-blocks", type=int, default=3000)
    parser.add_argument("--max-val-blocks", type=int, default=500)
    parser.add_argument("--max-test-blocks", type=int, default=1000)
    parser.add_argument("--selection-seed", type=int, default=20260714)
    parser.add_argument("--copy", action="store_true", help="Copy instead of creating same-volume hard links.")
    args = parser.parse_args()

    source_root = Path(args.input)
    out_root = Path(args.out)
    split_manifest = load_split_manifest(args.split_manifest)
    expected_split_hash = split_manifest["split_hash"]
    prepare_path = source_root / "_prepare_complete.json"
    if not prepare_path.exists():
        raise ValueError(f"Prepared block root has no provenance summary: {prepare_path}")
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    actual_split_hash = prepare.get("run", {}).get("split_hash")
    if actual_split_hash != expected_split_hash:
        raise ValueError(f"Prepared block split hash mismatch: expected {expected_split_hash}, got {actual_split_hash}")

    limits = {
        "train": args.max_train_blocks,
        "val": args.max_val_blocks,
        "test": args.max_test_blocks,
    }
    files_by_split = {
        split: selected_files(
            sorted((source_root / split).glob("*.pt")),
            limits[split],
            args.selection_seed,
        )
        for split in OFFICIAL_SPLITS
    }
    if any(not paths for paths in files_by_split.values()):
        raise ValueError("Every official split must contribute at least one block")
    block_hash = aggregate_hash(files_by_split)
    manifest_path = out_root / "_dataset_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("selected_blocks_hash") != block_hash:
            raise ValueError(f"Existing subset at {out_root} is incompatible and was preserved")
    out_root.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    for split, paths in files_by_split.items():
        split_out = out_root / split
        split_out.mkdir(parents=True, exist_ok=True)
        for source in paths:
            destination = split_out / source.name
            if destination.exists():
                if destination.stat().st_size != source.stat().st_size:
                    raise ValueError(f"Existing subset block differs and was preserved: {destination}")
                skipped += 1
                continue
            if args.copy:
                shutil.copy2(source, destination)
            else:
                os.link(source, destination)
            written += 1
    manifest = {
        "schema_version": "vl3d-experiment-block-subset-v1",
        "source_root": str(source_root),
        "out_root": str(out_root),
        "split_manifest": args.split_manifest,
        "split_hash": expected_split_hash,
        "selection_policy": "uniform_random_label_blind",
        "selection_seed": args.selection_seed,
        "model_seed_independent": True,
        "selected_blocks_hash": block_hash,
        "materialization": "copy" if args.copy else "hardlink",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "written": written,
        "skipped": skipped,
        "splits": {
            split: {
                "blocks": len(paths),
                "tiles": len({tile_id_from_block(path) for path in paths}),
                "files": [path.name for path in paths],
            }
            for split, paths in files_by_split.items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "split_hash": expected_split_hash,
                "selected_blocks_hash": block_hash,
                "written": written,
                "skipped": skipped,
                "counts": {split: len(paths) for split, paths in files_by_split.items()},
                "tiles": {
                    split: len({tile_id_from_block(path) for path in paths})
                    for split, paths in files_by_split.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
