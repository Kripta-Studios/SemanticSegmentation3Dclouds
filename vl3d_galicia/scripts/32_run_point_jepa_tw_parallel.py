import subprocess
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def run(command: list[str]) -> None:
    print("\n> " + " ".join(command), flush=True)
    subprocess.run(command, check=True)

def run_pipeline_for_seed(seed: int, args: argparse.Namespace, out_root: Path) -> None:
    print(f"=== Starting pipeline for seed {seed} ===")
    python = sys.executable

    # Pre-train Phase
    pretrain_out = out_root / f"pretrain_seed{seed}"
    cmd_pretrain = [
        python, "scripts/04b_pretrain_point_jepa_3d.py",
        "--out", str(pretrain_out),
        "--epochs", str(args.epochs_pretrain),
        "--batch-size", str(args.batch_size_pretrain),
        "--seed", str(seed),
        "--use-tw-input"  # Injected TW features for pre-training
    ]
    run(cmd_pretrain)

    ckpt = str(pretrain_out / "best_jepa.pt")
    
    # Fine-tune Phase 1: Point-JEPA + TW Base
    finetune_out = out_root / f"PointJEPA_TW_base_seed{seed}"
    cmd_finetune = [
        python, "scripts/05b_finetune_point_jepa_3d.py",
        "--checkpoint", ckpt,
        "--out", str(finetune_out),
        "--epochs", str(args.epochs_finetune),
        "--batch-size", str(args.batch_size_finetune),
        "--seed", str(seed),
        "--balanced-sampler",
        "--class-weight-mode", "inverse_sqrt",
        "--loss-type", "focal",
        "--focal-gamma", "1.5",
        "--use-tw-input"  # Injected TW features for fine-tuning
    ]
    run(cmd_finetune)
    
    # Fine-tune Phase 2: Point-JEPA + TW + DINOv3 Gated Fusion (M4)
    finetune_dino_out = out_root / f"M4_PointJEPA_TW_DINOv3_seed{seed}"
    cmd_finetune_dino = [
        python, "scripts/05b_finetune_point_jepa_3d.py",
        "--checkpoint", ckpt,
        "--out", str(finetune_dino_out),
        "--epochs", str(args.epochs_finetune),
        "--batch-size", str(args.batch_size_finetune),
        "--seed", str(seed),
        "--balanced-sampler",
        "--class-weight-mode", "inverse_sqrt",
        "--loss-type", "focal",
        "--focal-gamma", "1.5",
        "--use-tw-input",  # Injected TW features for fine-tuning
        "--external-feature-dir", args.dinov3_dir,
        "--external-feature-key", "dino_features"
    ]
    run(cmd_finetune_dino)
    print(f"=== Completed pipeline for seed {seed} ===")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs-pretrain", type=int, default=15)
    parser.add_argument("--epochs-finetune", type=int, default=18)
    parser.add_argument("--batch-size-pretrain", type=int, default=8)  # Reduced from 16 to 8 to avoid OOM
    parser.add_argument("--batch-size-finetune", type=int, default=4)  # Reduced from 8 to 4 to avoid OOM
    parser.add_argument("--dinov3-dir", type=str, default="data/processed/galicia_experiment_v1_dinov3_top")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    seeds = [7, 13, 21]
    out_root = Path("outputs/point_jepa_ablations")
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Starting parallel execution with {args.workers} workers...")
    
    # Execute pipelines in parallel across seeds
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_pipeline_for_seed, seed, args, out_root): seed for seed in seeds}
        
        for future in as_completed(futures):
            seed = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Pipeline for seed {seed} generated an exception: {e}")

if __name__ == "__main__":
    main()
