from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.classes import IGNORE_INDEX
from src.utils.progress import eta_line


def count_file(path: str) -> tuple[str, list[int]]:
    torch.set_num_threads(1)
    labels = torch.load(path, weights_only=False, map_location="cpu")["labels"].long()
    labels = labels[(labels >= 0) & (labels <= IGNORE_INDEX)]
    counts = torch.bincount(labels, minlength=IGNORE_INDEX + 1)[: IGNORE_INDEX + 1]
    return Path(path).name, [int(v) for v in counts.tolist()]


def write_cache(split_dir: Path, workers: int, force: bool) -> dict:
    cache_path = split_dir / "class_counts_cache.csv"
    files = sorted(split_dir.glob("*.pt"))
    if cache_path.exists() and not force:
        return {"split_dir": str(split_dir), "files": len(files), "cache": str(cache_path), "skipped": True}
    rows: list[tuple[str, list[int]]] = []
    started = time.perf_counter()
    done = 0

    def report() -> None:
        nonlocal done
        done += 1
        if done == len(files) or done % 500 == 0:
            print(eta_line(f"label-count cache {split_dir.name}", started, done, len(files)), flush=True)

    if workers <= 1:
        for path in files:
            rows.append(count_file(str(path)))
            report()
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(count_file, str(path)) for path in files]
            for future in as_completed(futures):
                rows.append(future.result())
                report()
    rows.sort(key=lambda item: item[0])
    with cache_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", *[f"c{i}" for i in range(IGNORE_INDEX + 1)]])
        for name, counts in rows:
            writer.writerow([name, *counts])
    return {"split_dir": str(split_dir), "files": len(files), "cache": str(cache_path), "skipped": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache per-block label counts for fast balanced block selection and sampling.")
    parser.add_argument("--data", required=True, help="Processed dataset root containing train/val/test split directories.")
    parser.add_argument("--splits", default="train,val", help="Comma-separated split names to cache.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(args.data)
    summaries = []
    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(split_dir)
        summaries.append(write_cache(split_dir, workers=max(int(args.num_workers), 1), force=bool(args.force)))
    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()
