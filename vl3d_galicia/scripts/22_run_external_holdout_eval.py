from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], dry_run: bool) -> None:
    print("\n> " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def find_model_dirs(roots: list[str]) -> list[Path]:
    out = []
    for root in roots:
        root_path = Path(root)
        if root_path.joinpath("best_model.pt").exists():
            out.append(root_path)
            continue
        if root_path.exists():
            out.extend(sorted(path for path in root_path.iterdir() if path.is_dir() and path.joinpath("best_model.pt").exists()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate completed models on an external no-training holdout.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model-root", action="append", default=["outputs/local_context_medium", "outputs/local_context_multiseed"])
    parser.add_argument("--data", default="data/processed/pnoa_varias_ccaa_holdout_tw")
    parser.add_argument("--external-feature-dir", default="data/processed/pnoa_varias_ccaa_holdout_geom_context")
    parser.add_argument("--out-root", default="outputs/external_holdout_eval")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-blocks", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    model_dirs = find_model_dirs(args.model_root)
    if not model_dirs:
        raise SystemExit(f"No completed model dirs found in {args.model_root}")
    for model_dir in model_dirs:
        cfg_path = model_dir / "run_config.json"
        if not cfg_path.exists():
            cfg_path = model_dir / "config.json"
        text = cfg_path.read_text(encoding="utf-8") if cfg_path.exists() else ""
        uses_external = "external_feature_dir" in text and "geom_features" in text
        cmd = [
            args.python,
            "scripts/20_evaluate_segmentation_model.py",
            "--model-dir",
            str(model_dir),
            "--data",
            args.data,
            "--out",
            str(out_root / model_dir.name),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--max-blocks",
            str(args.max_blocks),
        ]
        if "use_tw_input" in text and "true" in text.lower():
            cmd.append("--use-tw-input")
        if uses_external:
            cmd.extend(["--external-feature-dir", args.external_feature_dir, "--external-feature-key", "geom_features"])
        run(cmd, dry_run=args.dry_run)

    run(
        [
            args.python,
            "scripts/07_compare_results.py",
            "--experiments-root",
            args.out_root,
            "--out-csv",
            str(out_root / "comparison" / "external_comparison.csv"),
            "--out-md",
            str(out_root / "comparison" / "external_comparison.md"),
        ],
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
