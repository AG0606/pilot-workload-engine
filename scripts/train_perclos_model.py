import argparse
from pathlib import Path
import sys
from typing import Tuple

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.models.ocular_vision import OcularVigilanceNet


def generate_synthetic_ocular_image_dataset(
    num_samples: int = 1000,
    img_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates synthetic eye crop dataset with open vs closed morphology for pipeline training."""
    images = np.zeros((num_samples, 1, img_size, img_size), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.int64)

    y_grid, x_grid = np.ogrid[:img_size, :img_size]
    center_y, center_x = img_size // 2, img_size // 2

    for i in range(num_samples):
        # 0: Open Eye (round pupil + iris), 1: Closed Eye (horizontal slit line)
        is_closed = (i % 2 == 1)
        labels[i] = 1 if is_closed else 0

        # Background eye socket texture
        bg = np.random.normal(0.7, 0.05, (img_size, img_size)).astype(np.float32)

        if not is_closed:
            # Open eye: elliptical sclera + dark circular pupil
            sclera_mask = ((x_grid - center_x) ** 2) / 20.0**2 + ((y_grid - center_y) ** 2) / 12.0**2 <= 1.0
            pupil_mask = ((x_grid - center_x) ** 2) / 6.0**2 + ((y_grid - center_y) ** 2) / 6.0**2 <= 1.0

            bg[sclera_mask] = 0.9
            bg[pupil_mask] = 0.15
        else:
            # Closed eye: horizontal eyelid crease line
            slit_mask = np.abs(y_grid - center_y) <= 1.5
            slit_length = np.abs(x_grid - center_x) <= 18.0
            bg[slit_mask & slit_length] = 0.2

        # Add camera sensor noise
        bg += np.random.normal(0.0, 0.03, (img_size, img_size)).astype(np.float32)
        images[i, 0] = np.clip(bg, 0.0, 1.0)

    return torch.from_numpy(images), torch.from_numpy(labels)


def train_ocular_model(
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> None:
    """Trains OcularVigilanceNet and saves the checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training OcularVigilanceNet on device: {device}")

    images, labels = generate_synthetic_ocular_image_dataset(num_samples=1200)
    dataset = TensorDataset(images, labels)

    train_len = int(len(dataset) * 0.8)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = OcularVigilanceNet(in_channels=1, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                preds = torch.argmax(model(x_b), dim=-1)
                correct += (preds == y_b).sum().item()
                total += len(y_b)

        val_acc = correct / total
        print(f"Epoch {epoch+1:2d}/{epochs:2d} | Train Loss: {total_loss/len(train_loader):.4f} | Val Accuracy: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_ocular_vigilance_model.pt")

    print(f"Best OcularVigilanceNet model checkpoint saved to {output_dir / 'best_ocular_vigilance_model.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OcularVigilanceNet for PERCLOS calculation.")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()
    train_ocular_model(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
