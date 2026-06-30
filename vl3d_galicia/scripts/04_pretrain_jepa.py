from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_common import infer_in_channels, load_config, save_json, set_global_seed
from src.data.jepa_dataset import GeoPointJepaDataset, JepaBlockMasker, jepa_collate_fn
from src.models.jepa import GeoPointJEPA, LeJEPALoss
from src.models.tw_jepa import TWAuxLoss, TW_JEPA
from src.utils.progress import eta_line


def torch_save_atomic(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain GeoPoint LeJEPA/TW-JEPA without labels.")
    parser.add_argument("--config")
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--sigreg-weight", type=float)
    parser.add_argument("--sigreg-slices", type=int)
    parser.add_argument("--use-tw-input", action="store_true")
    parser.add_argument("--tw-target", action="store_true")
    parser.add_argument("--max-blocks", type=int)
    parser.add_argument("--mask-type", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_root = args.data or cfg.get("data", {}).get("blocks_dir", "data/processed/galicia_blocks_tw")
    train_dir = str(Path(data_root) / "train")
    out = Path(args.out or cfg.get("output", {}).get("dir", "outputs/tw_jepa_pretrain"))
    out.mkdir(parents=True, exist_ok=True)
    epochs = args.epochs or cfg.get("training", {}).get("epochs", 80)
    batch_size = args.batch_size or cfg.get("training", {}).get("batch_size", 48)
    workers = args.num_workers if args.num_workers is not None else cfg.get("training", {}).get("num_workers", 4)
    lr = args.lr or cfg.get("training", {}).get("learning_rate", 3e-4)
    sigreg_weight = args.sigreg_weight if args.sigreg_weight is not None else cfg.get("training", {}).get("sigreg_weight", 0.1)
    sigreg_slices = args.sigreg_slices or cfg.get("training", {}).get("sigreg_slices", 256)
    use_tw = args.use_tw_input or cfg.get("data", {}).get("use_tw_input", False)
    tw_target = args.tw_target or cfg.get("training", {}).get("tw_target", False)
    max_blocks = args.max_blocks or cfg.get("data", {}).get("max_blocks", 0)
    mask_type = args.mask_type or cfg.get("masking", {}).get("type", "spatial_block")
    generator = set_global_seed(args.seed)

    best_path = out / "best_jepa.pt"
    last_path = out / "last_jepa.pt"
    complete_path = out / "pretrain_complete.json"
    if complete_path.exists() and best_path.exists() and not args.no_resume:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if int(complete.get("epochs_completed", 0)) >= epochs or bool(complete.get("early_stopped", False)):
            print(f"JEPA pretraining already complete at {complete_path}; skipping training.")
            return

    in_channels = infer_in_channels(train_dir, use_tw_input=use_tw)
    masker = JepaBlockMasker(mask_type=mask_type, seed=args.seed)
    dataset = GeoPointJepaDataset(train_dir, max_blocks=max_blocks, masker=masker, use_tw_input=use_tw, return_tw_target=tw_target)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=jepa_collate_fn, pin_memory=True, drop_last=True, generator=generator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = torch.load(dataset.files[0], weights_only=False, map_location="cpu")
    tw_dim = int(sample["tw_features"].shape[1] - 1) if tw_target else 0
    model = TW_JEPA(in_channels=in_channels, tw_dim=tw_dim).to(device) if tw_target else GeoPointJEPA(in_channels=in_channels).to(device)
    loss_fn = LeJEPALoss(sigreg_weight=sigreg_weight, num_slices=sigreg_slices)
    tw_loss_fn = TWAuxLoss(tw_dim=tw_dim) if tw_target else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    global_step = 0
    rows = []
    best = float("inf")
    epochs_without_improvement = 0
    early_stopped = False
    completed_epoch = 0
    start_epoch = 1
    if last_path.exists() and not args.no_resume:
        state = torch.load(last_path, weights_only=False, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        global_step = int(state.get("global_step", 0))
        rows = list(state.get("rows", []))
        best = float(state.get("best_loss", best))
        epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
        early_stopped = bool(state.get("early_stopped", False))
        completed_epoch = int(state.get("completed_epoch", state.get("epoch", 0)))
        start_epoch = int(state.get("epoch", 0)) + 1
        print(f"Resuming JEPA pretraining from {last_path} at epoch {start_epoch}/{epochs}")

    run_config = {
        "in_channels": in_channels,
        "use_tw_input": use_tw,
        "tw_target": tw_target,
        "sigreg_weight": sigreg_weight,
        "sigreg_slices": sigreg_slices,
        "mask_type": mask_type,
        "target_mode": "shared",
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": workers,
        "seed": args.seed,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
    }
    pretrain_start = time.perf_counter()
    total_epochs_this_run = max(epochs - start_epoch + 1, 0)
    for epoch in range(start_epoch, epochs + 1):
        completed_epoch = epoch
        model.train()
        totals = {"loss": 0.0, "loss_pred": 0.0, "loss_sigreg": 0.0, "tw_aux_loss": 0.0}
        steps = 0
        for batch in tqdm(loader, desc=f"JEPA {epoch}/{epochs}"):
            x_ctx = batch["x_context"].to(device, non_blocking=True)
            x_tgt = batch["x_target"].to(device, non_blocking=True)
            mask_ctx = batch["mask_context"].to(device, non_blocking=True)
            mask_tgt = batch["mask_target"].to(device, non_blocking=True)
            coords_tgt = batch["coords_target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                if tw_target:
                    pred, target, pred_tw, _, embeddings = model(x_ctx, mask_ctx, x_tgt, mask_tgt, coords_tgt)
                else:
                    pred, target, _, embeddings = model(x_ctx, mask_ctx, x_tgt, mask_tgt, coords_tgt)
                loss, metrics = loss_fn(pred, target, embeddings, global_step=global_step)
                if tw_target:
                    tw_loss, tw_metrics = tw_loss_fn(
                        pred_tw,
                        batch["tw_target"].to(device, non_blocking=True),
                        batch["tw_valid_target"].to(device, non_blocking=True),
                        mask_tgt,
                    )
                    loss = loss + 0.1 * tw_loss
                    metrics.update(tw_metrics)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if hasattr(model, "update_target_encoder"):
                model.update_target_encoder()
            totals["loss"] += float(loss.detach().cpu())
            for key in ["loss_pred", "loss_sigreg", "tw_aux_loss"]:
                totals[key] += float(metrics.get(key, 0.0))
            steps += 1
            global_step += 1
        row = {"epoch": epoch, **{k: v / max(steps, 1) for k, v in totals.items()}}
        rows.append(row)
        print(row)
        print(eta_line("JEPA pretraining", pretrain_start, epoch - start_epoch + 1, total_epochs_this_run))
        improved = row["loss"] < best - args.early_stopping_min_delta
        if improved:
            best = row["loss"]
            torch_save_atomic({"model": model.state_dict(), "config": run_config}, best_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        torch_save_atomic(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "global_step": global_step,
                "best_loss": best,
                "rows": rows,
                "config": run_config,
                "epochs_without_improvement": epochs_without_improvement,
                "early_stopped": early_stopped,
                "completed_epoch": completed_epoch,
            },
            last_path,
        )
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            early_stopped = True
            print(
                "Early stopping: "
                f"pretrain loss did not improve by {args.early_stopping_min_delta:g} "
                f"for {epochs_without_improvement} epochs."
            )
            torch_save_atomic(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "global_step": global_step,
                    "best_loss": best,
                    "rows": rows,
                    "config": run_config,
                    "epochs_without_improvement": epochs_without_improvement,
                    "early_stopped": early_stopped,
                    "completed_epoch": completed_epoch,
                },
                last_path,
            )
            break
    if rows:
        with (out / "pretrain_log.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    save_json(
        out / "pretrain_summary.json",
        {
            "best_loss": best,
            "epochs_requested": epochs,
            "epochs_completed": completed_epoch,
            "early_stopped": early_stopped,
            "global_steps": global_step,
        },
    )
    save_json(
        out / "pretrain_complete.json",
        {
            "best_loss": best,
            "epochs_requested": epochs,
            "epochs_completed": completed_epoch,
            "early_stopped": early_stopped,
            "global_steps": global_step,
            "best_jepa": str(best_path),
            "checkpoint": str(last_path),
        },
    )


if __name__ == "__main__":
    main()
