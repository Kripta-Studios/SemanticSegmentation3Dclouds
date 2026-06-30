from __future__ import annotations

import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.data.classes import IGNORE_INDEX
from src.eval.segmentation_metrics import compute_segmentation_metrics
from src.utils.progress import eta_line


def torch_save_atomic(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


class SegmentationTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: str = "cuda",
        lr: float = 1e-3,
        ignore_index: int = IGNORE_INDEX,
        epochs: int = 15,
        class_weights: torch.Tensor | None = None,
        weight_decay: float = 1e-4,
        amp: bool = True,
        optimizer_param_groups: list[dict] | None = None,
        early_stopping_patience: int = 0,
        early_stopping_min_delta: float = 0.0,
        loss_type: str = "ce",
        focal_gamma: float = 2.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.ignore_index = ignore_index
        self.epochs = epochs
        self.amp = amp and self.device.type == "cuda"
        self.class_weights = class_weights.to(self.device) if class_weights is not None else None
        if loss_type not in {"ce", "focal"}:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        self.loss_type = loss_type
        self.focal_gamma = float(focal_gamma)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=self.ignore_index)
        if optimizer_param_groups is None:
            trainable_params = [param for param in self.model.parameters() if param.requires_grad]
            if not trainable_params:
                raise ValueError("No trainable parameters available for optimizer")
            optimizer_param_groups = [{"params": trainable_params, "lr": lr}]
        self.optimizer = torch.optim.AdamW(optimizer_param_groups, lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(epochs, 1))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.best_macro_f1 = -1.0
        self.best_metrics: dict = {}
        self.best_state_dict: dict | None = None
        self.history = {"train_loss": [], "val_macro_f1": [], "val_weighted_f1": []}
        self.early_stopping_patience = max(int(early_stopping_patience), 0)
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.epochs_without_improvement = 0
        self.early_stopped = False
        self.completed_epoch = 0

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits_flat = logits.reshape(-1, logits.shape[-1])
        labels_flat = labels.reshape(-1)
        if self.loss_type == "ce":
            return self.criterion(logits_flat, labels_flat)
        valid = labels_flat != self.ignore_index
        if not torch.any(valid):
            return logits_flat.sum() * 0.0
        logits_valid = logits_flat[valid]
        labels_valid = labels_flat[valid]
        log_probs = F.log_softmax(logits_valid, dim=-1)
        log_pt = log_probs.gather(1, labels_valid.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt).clamp_min(1e-6) ** self.focal_gamma) * log_pt
        if self.class_weights is not None:
            loss = loss * self.class_weights[labels_valid]
        return loss.mean()

    def checkpoint_state(self, epoch: int) -> dict:
        return {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_macro_f1": self.best_macro_f1,
            "best_metrics": self.best_metrics,
            "best_state_dict": self.best_state_dict,
            "history": self.history,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "epochs_without_improvement": self.epochs_without_improvement,
            "early_stopped": self.early_stopped,
            "completed_epoch": epoch,
            "loss_type": self.loss_type,
            "focal_gamma": self.focal_gamma,
        }

    def save_training_checkpoint(self, path: str | Path, epoch: int) -> None:
        torch_save_atomic(self.checkpoint_state(epoch), path)

    def load_training_checkpoint(self, path: str | Path) -> int:
        state = torch.load(path, weights_only=False, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])
        self.best_macro_f1 = float(state.get("best_macro_f1", -1.0))
        self.best_metrics = state.get("best_metrics", {})
        self.best_state_dict = state.get("best_state_dict")
        self.history = state.get("history", self.history)
        self.epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
        self.early_stopped = bool(state.get("early_stopped", False))
        self.completed_epoch = int(state.get("completed_epoch", state.get("epoch", 0)))
        return int(state.get("epoch", 0)) + 1

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total = 0.0
        steps = 0
        for batch in tqdm(self.train_loader, desc=f"Train {epoch}/{self.epochs}"):
            features = batch["features"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.amp):
                logits = self.model(features, mask)
                loss = self.compute_loss(logits, labels)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += float(loss.detach().cpu())
            steps += 1
        self.scheduler.step()
        return total / max(steps, 1)

    @torch.no_grad()
    def evaluate(self, loader=None, return_predictions: bool = False):
        loader = loader or self.val_loader
        self.model.eval()
        preds_all = []
        labels_all = []
        for batch in tqdm(loader, desc="Eval"):
            features = batch["features"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            logits = self.model(features, mask)
            preds = logits.argmax(dim=-1)
            preds_all.append(preds.reshape(-1).cpu())
            labels_all.append(labels.reshape(-1).cpu())
        preds_cat = torch.cat(preds_all)
        labels_cat = torch.cat(labels_all)
        metrics = compute_segmentation_metrics(
            preds_cat,
            labels_cat,
            num_classes=7,
            ignore_index=self.ignore_index,
        )
        if return_predictions:
            return metrics, preds_cat.numpy(), labels_cat.numpy()
        return metrics

    def fit(
        self,
        start_epoch: int = 1,
        checkpoint_path: str | Path | None = None,
        best_model_path: str | Path | None = None,
        progress_label: str = "segmentation training",
    ) -> dict:
        fit_start = time.perf_counter()
        total_epochs_this_run = max(self.epochs - start_epoch + 1, 0)
        for epoch in range(start_epoch, self.epochs + 1):
            self.completed_epoch = epoch
            train_loss = self.train_epoch(epoch)
            val = self.evaluate()
            self.history["train_loss"].append(train_loss)
            self.history["val_macro_f1"].append(val["macro_f1"])
            self.history["val_weighted_f1"].append(val["weighted_f1"])
            print(
                f"Epoch {epoch:03d} | loss={train_loss:.4f} "
                f"| macro_f1={val['macro_f1']:.4f} | weighted_f1={val['weighted_f1']:.4f}"
            )
            print(eta_line(progress_label, fit_start, epoch - start_epoch + 1, total_epochs_this_run))
            improved = val["macro_f1"] > self.best_macro_f1 + self.early_stopping_min_delta
            if improved:
                self.best_macro_f1 = val["macro_f1"]
                self.best_metrics = val
                self.best_state_dict = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                if best_model_path is not None:
                    torch_save_atomic(self.best_state_dict, best_model_path)
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            if checkpoint_path is not None:
                self.save_training_checkpoint(checkpoint_path, epoch)
            if self.early_stopping_patience > 0 and self.epochs_without_improvement >= self.early_stopping_patience:
                self.early_stopped = True
                print(
                    "Early stopping: "
                    f"val_macro_f1 did not improve by {self.early_stopping_min_delta:g} "
                    f"for {self.epochs_without_improvement} epochs."
                )
                if checkpoint_path is not None:
                    self.save_training_checkpoint(checkpoint_path, epoch)
                break
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        return self.best_metrics

    def save_reports(
        self,
        output_dir: str | Path,
        test_metrics: dict | None = None,
        config: dict | None = None,
        test_predictions=None,
        test_labels=None,
    ) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if config is not None:
            (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        pd.DataFrame(self.history).to_csv(output / "train_log.csv", index=False)
        (output / "val_metrics.json").write_text(json.dumps(self.best_metrics, indent=2), encoding="utf-8")
        if self.best_state_dict is not None:
            torch_save_atomic(self.best_state_dict, output / "best_model.pt")
        final_metrics = test_metrics if test_metrics is not None else self.best_metrics
        (output / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
        self._save_tabular_metrics(output, final_metrics)
        if test_metrics is not None:
            (output / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
            cm = np.array(test_metrics["confusion_matrix"])
            np.savetxt(output / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=False, cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(output / "confusion_matrix.png", dpi=160)
            plt.close()
        if test_predictions is not None:
            np.save(output / "test_predictions.npy", np.asarray(test_predictions))
        if test_labels is not None:
            np.save(output / "test_labels.npy", np.asarray(test_labels))
        plt.figure(figsize=(8, 4))
        plt.plot(self.history["train_loss"], label="train_loss")
        plt.plot(self.history["val_macro_f1"], label="val_macro_f1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / "training_curve.png", dpi=160)
        plt.close()

    def _save_tabular_metrics(self, output: Path, metrics: dict) -> None:
        rows = []
        class_names = {
            0: "ground",
            1: "low_vegetation",
            2: "medium_vegetation",
            3: "high_vegetation",
            4: "building",
            5: "water",
        }
        for cls, name in class_names.items():
            rows.append(
                {
                    "class_id": cls,
                    "class_name": name,
                    "precision": float(metrics.get("class_precision", {}).get(str(cls), metrics.get("class_precision", {}).get(cls, 0.0))),
                    "recall": float(metrics.get("class_recall", {}).get(str(cls), metrics.get("class_recall", {}).get(cls, 0.0))),
                    "f1": float(metrics.get("class_f1", {}).get(str(cls), metrics.get("class_f1", {}).get(cls, 0.0))),
                    "iou": float(metrics.get("class_iou", {}).get(str(cls), metrics.get("class_iou", {}).get(cls, 0.0))),
                    "support": int(metrics.get("class_support", {}).get(str(cls), metrics.get("class_support", {}).get(cls, 0))),
                }
            )
        pd.DataFrame(rows).to_csv(output / "per_class_metrics.csv", index=False)
