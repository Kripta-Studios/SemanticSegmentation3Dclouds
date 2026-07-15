import subprocess
import sys
import argparse
from pathlib import Path

def run(command: list[str]) -> None:
    print("\n> " + " ".join(command), flush=True)
    subprocess.run(command, check=True)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs-pretrain", type=int, default=15)
    parser.add_argument("--epochs-finetune", type=int, default=18)
    parser.add_argument("--dinov3-dir", type=str, default="data/processed/galicia_experiment_v1_dinov3_top")
    args = parser.parse_args()

    seeds = [7, 13, 21]
    python = sys.executable
    out_root = Path("outputs/point_jepa_ablations")
    out_root.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        print(f"=== Running pipeline for seed {seed} ===")
        pretrain_out = out_root / f"pretrain_seed{seed}"
        cmd_pretrain = [
            python, "scripts/04b_pretrain_point_jepa_3d.py",
            "--out", str(pretrain_out),
            "--epochs", str(args.epochs_pretrain),
            "--batch-size", "16",
            "--seed", str(seed)
        ]
        run(cmd_pretrain)

        ckpt = str(pretrain_out / "best_jepa.pt")
        
        # Geometry-only Point-JEPA
        finetune_out = out_root / f"PointJEPA_base_seed{seed}"
        cmd_finetune = [
            python, "scripts/05b_finetune_point_jepa_3d.py",
            "--checkpoint", ckpt,
            "--out", str(finetune_out),
            "--epochs", str(args.epochs_finetune),
            "--batch-size", "8",
            "--seed", str(seed),
            "--balanced-sampler",
            "--class-weight-mode", "inverse_sqrt",
            "--loss-type", "focal",
            "--focal-gamma", "1.5"
        ]
        run(cmd_finetune)
        
        # Point-JEPA + DINOv3 Gated Fusion (M4)
        finetune_dino_out = out_root / f"M4_PointJEPA_DINOv3_seed{seed}"
        cmd_finetune_dino = [
            python, "scripts/05b_finetune_point_jepa_3d.py",
            "--checkpoint", ckpt,
            "--out", str(finetune_dino_out),
            "--epochs", str(args.epochs_finetune),
            "--batch-size", "8",
            "--seed", str(seed),
            "--balanced-sampler",
            "--class-weight-mode", "inverse_sqrt",
            "--loss-type", "focal",
            "--focal-gamma", "1.5",
            "--external-feature-dir", args.dinov3_dir,
            "--external-feature-key", "dino_features"
        ]
        run(cmd_finetune_dino)

if __name__ == "__main__":
    main()
