from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_common import infer_channel_layout, load_config, save_json, set_global_seed
from src.data.segmentation_dataset import SegmentationBlockDataset, segmentation_collate_fn
from src.models.point_jepa_3d import PointJEPA3D
from src.utils.progress import eta_line

def torch_save_atomic(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)

def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain true 3D patch-based Point-JEPA.")
    parser.add_argument("--config")
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--use-tw-input", action="store_true")
    parser.add_argument("--max-blocks", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-patches", type=int, default=256)
    parser.add_argument("--points-per-patch", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    
    cfg = load_config(args.config) if args.config else {}
    data_root = args.data or cfg.get("data", {}).get("blocks_dir", "data/processed/galicia_blocks")
    train_dir = str(Path(data_root) / "train")
    out = Path(args.out or cfg.get("output", {}).get("dir", "outputs/point_jepa_3d_pretrain"))
    out.mkdir(parents=True, exist_ok=True)
    
    epochs = args.epochs or cfg.get("training", {}).get("epochs", 50)
    batch_size = args.batch_size or cfg.get("training", {}).get("batch_size", 16)
    workers = args.num_workers if args.num_workers is not None else cfg.get("training", {}).get("num_workers", 4)
    # Reduced learning rate from 5e-4 to 1e-4 to prevent exploding gradients
    lr = args.lr or cfg.get("training", {}).get("learning_rate", 1e-4)
    use_tw = args.use_tw_input or cfg.get("data", {}).get("use_tw_input", False)
    max_blocks = args.max_blocks or cfg.get("data", {}).get("max_blocks", 0)
    generator = set_global_seed(args.seed)

    best_path = out / "best_jepa.pt"
    last_path = out / "last_jepa.pt"
    complete_path = out / "pretrain_complete.json"
    
    if complete_path.exists() and best_path.exists() and not args.no_resume:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if int(complete.get("epochs_completed", 0)) >= epochs:
            print(f"PointJEPA3D pretraining already complete at {complete_path}; skipping training.")
            return

    channel_layout = infer_channel_layout(train_dir, use_tw_input=use_tw)
    in_channels = channel_layout["in_channels"]
    
    dataset = SegmentationBlockDataset(train_dir, max_blocks=max_blocks, use_tw_input=use_tw)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=segmentation_collate_fn, pin_memory=True, drop_last=True, generator=generator)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointJEPA3D(
        in_channels=in_channels,
        embed_dim=256,
        num_patches=args.num_patches,
        points_per_patch=args.points_per_patch
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    
    global_step = 0
    rows = []
    best = float("inf")
    epochs_without_improvement = 0
    completed_epoch = 0
    start_epoch = 1
    
    if last_path.exists() and not args.no_resume:
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        global_step = int(state.get("global_step", 0))
        rows = list(state.get("rows", []))
        best = float(state.get("best_loss", best))
        epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
        completed_epoch = int(state.get("completed_epoch", state.get("epoch", 0)))
        start_epoch = int(state.get("epoch", 0)) + 1
        print(f"Resuming PointJEPA3D pretraining from epoch {start_epoch}/{epochs}")

    run_config = {
        "in_channels": in_channels,
        "use_tw_input": use_tw,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": workers,
        "seed": args.seed,
        "num_patches": args.num_patches,
        "points_per_patch": args.points_per_patch,
    }
    
    pretrain_start = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        completed_epoch = epoch
        model.train()
        total_loss = 0.0
        steps = 0
        
        for batch in tqdm(loader, desc=f"PointJEPA3D {epoch}/{epochs}"):
            x = batch["features"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            # Use float32 context or float16 context. Switched to float16 to prevent bfloat16 instability with huge values
            with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
                pred_tgt, true_tgt, _ = model(x, mask)
                loss = F.smooth_l1_loss(pred_tgt, true_tgt)
                if torch.isnan(loss):
                    print("Warning: NaN loss detected! Zeroing gradients.")
                    loss = loss.new_tensor(0.0, requires_grad=True)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Stricter gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
            
            model.update_target_encoder()
            
            total_loss += float(loss.detach().cpu()) if not torch.isnan(loss) else 0.0
            steps += 1
            global_step += 1
            
        avg_loss = total_loss / max(steps, 1)
        row = {"epoch": epoch, "loss": avg_loss}
        rows.append(row)
        print(row)
        print(eta_line("PointJEPA3D pretraining", pretrain_start, epoch - start_epoch + 1, max(epochs - start_epoch + 1, 0)))
        
        if avg_loss < best:
            best = avg_loss
            epochs_without_improvement = 0
            torch_save_atomic({"model": model.state_dict(), "config": run_config}, best_path)
        else:
            epochs_without_improvement += 1
            print(f"Loss did not improve. Patience: {epochs_without_improvement}/{args.patience}")
            
        torch_save_atomic({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "global_step": global_step,
            "best_loss": best,
            "epochs_without_improvement": epochs_without_improvement,
            "rows": rows,
            "config": run_config,
            "completed_epoch": completed_epoch,
        }, last_path)
        
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping triggered! No improvement for {args.patience} epochs.")
            break
        
    if rows:
        with (out / "pretrain_log.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
            
    save_json(out / "pretrain_complete.json", {
        "best_loss": best,
        "epochs_requested": epochs,
        "epochs_completed": completed_epoch,
        "global_steps": global_step,
        "best_jepa": str(best_path),
        "checkpoint": str(last_path),
    })

if __name__ == "__main__":
    main()
