import argparse
from pathlib import Path
import sys
from typing import Tuple

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.datasets.workload_dataset import PilotWorkloadDataset
from src.models.workload_classifier import MultiModalWorkloadClassifier
from src.training.losses import FocalLoss, compute_class_weights
from src.training.trainer import WorkloadTrainer


def create_synthetic_dataset(num_windows: int = 200) -> PilotWorkloadDataset:
    """Generates synthetic dataset for model training verification."""
    windows = {
        "eeg": np.random.randn(num_windows, 17, 80).astype(np.float32),
        "cardio": np.random.randn(num_windows, 3, 80).astype(np.float32),
        "ocular": np.random.randn(num_windows, 4, 80).astype(np.float32),
        "context": np.random.randn(num_windows, 8, 80).astype(np.float32),
        # Skewed class distribution: ~70% baseline (0), ~15% CA (1), ~10% DA (2), ~5% SS (3)
        "label": np.random.choice([0, 1, 2, 3], size=num_windows, p=[0.70, 0.15, 0.10, 0.05]).astype(np.int64),
    }
    return PilotWorkloadDataset.from_window_dict(windows)


def run_training(
    data_path: Path | None,
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    synthetic: bool = False,
) -> None:
    """Executes multi-modal model training with FocalLoss and checkpointing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    if synthetic or data_path is None or not data_path.exists():
        print("Initializing synthetic dataset for model training...")
        dataset = create_synthetic_dataset(num_windows=300)
    else:
        print(f"Loading preprocessed dataset from {data_path}...")
        dataset = PilotWorkloadDataset.load(data_path)

    total_len = len(dataset)
    val_len = max(int(total_len * 0.2), 1)
    train_len = total_len - val_len

    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Calculate class balancing weights
    train_labels = dataset.labels[train_set.indices]
    class_weights = compute_class_weights(train_labels, num_classes=4).to(device)

    model = MultiModalWorkloadClassifier(
        num_classes=4,
        eeg_channels=17,
        cardio_channels=3,
        ocular_channels=4,
        context_channels=8,
        temporal_samples=80,
        embed_dim=128,
        fused_dim=256,
        dropout_rate=0.2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = FocalLoss(gamma=2.0, alpha=class_weights)

    trainer = WorkloadTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        checkpoint_dir=output_dir,
    )

    for ep in range(epochs):
        tr_loss = history["train_loss"][ep]
        vl_loss = history["val_loss"][ep]
        vl_ll = history["val_log_loss"][ep]
        vl_acc = history["val_balanced_acc"][ep]
        vl_f1 = history["val_macro_f1"][ep]
        print(
            f"Epoch {ep+1:2d}/{epochs:2d} | Train Loss: {tr_loss:.4f} | "
            f"Val Loss: {vl_loss:.4f} | Log Loss: {vl_ll:.4f} | "
            f"Bal Acc: {vl_acc:.4f} | Macro F1: {vl_f1:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-modal pilot workload classifier.")
    parser.add_argument("--data-path", type=Path, default=Path("output/pilot_workload_dataset.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--synthetic", action="store_true", default=False)

    args = parser.parse_args()
    run_training(
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        synthetic=args.synthetic,
    )


if __name__ == "__main__":
    main()
