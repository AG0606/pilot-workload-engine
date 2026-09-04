from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .metrics import compute_classification_metrics


class WorkloadTrainer:
    """Trainer orchestrating multi-modal forward pass, backpropagation, and metric tracking."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        device_type = "cuda" if device.type == "cuda" else "cpu"
        self.scaler = torch.amp.GradScaler(device_type, enabled=(device.type == "cuda"))

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Runs a full training epoch across batches."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        device_type = "cuda" if self.device.type == "cuda" else "cpu"

        for batch in dataloader:
            batch_dev = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            targets = batch_dev["label"]

            self.optimizer.zero_grad()
            with torch.amp.autocast(device_type, enabled=(self.device.type == "cuda")):
                logits = self.model(batch=batch_dev)
                loss = self.criterion(logits, targets)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        return {"train_loss": avg_loss}

    def evaluate(self, dataloader: DataLoader) -> Dict[str, Union[float, np.ndarray]]:
        """Runs validation pass and computes full multi-class metrics."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)
        all_targets: List[np.ndarray] = []
        all_probs: List[np.ndarray] = []

        with torch.no_grad():
            for batch in dataloader:
                batch_dev = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                targets = batch_dev["label"]

                device_type = "cuda" if self.device.type == "cuda" else "cpu"
                with torch.amp.autocast(device_type, enabled=(self.device.type == "cuda")):
                    logits = self.model(batch=batch_dev)
                    loss = self.criterion(logits, targets)

                total_loss += loss.item()
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
                all_targets.append(targets.cpu().numpy())

        avg_loss = total_loss / max(num_batches, 1)
        if all_targets:
            y_true = np.concatenate(all_targets, axis=0)
            y_prob = np.concatenate(all_probs, axis=0)
            metrics = compute_classification_metrics(y_true, y_prob)
        else:
            metrics = {"accuracy": 0.0, "balanced_accuracy": 0.0, "macro_f1": 0.0, "kaggle_log_loss": 0.0}

        metrics["val_loss"] = avg_loss
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 5,
        checkpoint_dir: Optional[Path] = None,
    ) -> Dict[str, List[float]]:
        """Executes full training schedule with checkpointing on minimum validation log loss."""
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_log_loss": [],
            "val_balanced_acc": [],
            "val_macro_f1": [],
        }

        best_val_loss = float("inf")
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            train_res = self.train_epoch(train_loader)
            val_res = self.evaluate(val_loader)

            val_log_loss = float(val_res["kaggle_log_loss"])
            val_loss = float(val_res["val_loss"])
            val_bal_acc = float(val_res["balanced_accuracy"])
            val_f1 = float(val_res["macro_f1"])

            history["train_loss"].append(train_res["train_loss"])
            history["val_loss"].append(val_loss)
            history["val_log_loss"].append(val_log_loss)
            history["val_balanced_acc"].append(val_bal_acc)
            history["val_macro_f1"].append(val_f1)

            if checkpoint_dir is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt_path = checkpoint_dir / "best_workload_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_log_loss": val_log_loss,
                    },
                    best_ckpt_path,
                )

        return history
