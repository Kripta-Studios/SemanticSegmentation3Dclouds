from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SMOKE_DIRS = [
    "data/processed/dino_smoke",
    "data/processed/dinov2_smoke",
    "data/processed/dinov2_quick",
    "data/processed/geom_context_smoke",
    "outputs/dino_smoke",
    "outputs/dinov2_quick",
    "outputs/dinov2_gated_smoke",
    "outputs/geom_context_smoke",
    "outputs/test_pipeline_jepa",
    "outputs/test_pipeline_jepa2",
]

NEGATIVE_DINO_FEATURE_DIRS = [
    "data/processed/galicia_blocks_medium_dinov2s14",
]

PYTEST_DIRS = [
    ".pytest_cache",
    ".pytest_basetemp_dino",
    ".pytest_basetemp_gated",
    ".tmp_pytest",
    "pytest-cache-files-22a_xvwf",
    "pytest-cache-files-jur30jye",
    "pytest-cache-files-lxkx30jr",
    "pytest-cache-files-nraz_l62",
    "pytest-cache-files-qjtdmmgy",
    "pytest-cache-files-zb0q9lax",
]


def dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def remove_path(path: Path, dry_run: bool) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if dry_run:
        return True, "dry_run"
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True, "removed"
    except Exception as exc:
        return False, f"failed: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove temporary experiment artifacts without touching canonical data.")
    parser.add_argument("--profile", choices=["smoke", "negative-dino", "pytest", "all"], default="smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Required for destructive deletion.")
    args = parser.parse_args()

    targets: list[str] = []
    if args.profile in {"smoke", "all"}:
        targets.extend(SMOKE_DIRS)
    if args.profile in {"negative-dino", "all"}:
        targets.extend(NEGATIVE_DINO_FEATURE_DIRS)
    if args.profile in {"pytest", "all"}:
        targets.extend(PYTEST_DIRS)

    if not args.dry_run and not args.yes:
        raise SystemExit("Refusing to delete without --yes. Re-run with --dry-run to inspect.")

    print("Cleanup targets:")
    reclaimed = 0
    for target in targets:
        path = Path(target)
        size = dir_size(path)
        ok, status = remove_path(path, dry_run=args.dry_run)
        if ok and status in {"removed", "dry_run"}:
            reclaimed += size
        print(f"- {target}: {status}, size={size / (1024 ** 3):.3f} GB")
    action = "Would reclaim" if args.dry_run else "Reclaimed"
    print(f"{action}: {reclaimed / (1024 ** 3):.3f} GB")


if __name__ == "__main__":
    main()
