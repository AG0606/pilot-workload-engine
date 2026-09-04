from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(
    labels: Union[np.ndarray, torch.Tensor],
    num_classes: int = 4,
    smoothing: float = 0.05,
) -> torch.Tensor:
    """Computes smoothed inverse class frequency weights for addressing severe class imbalance."""
    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)

    counts = np.bincount(labels_np, minlength=num_classes).astype(np.float32)
    total_samples = float(np.sum(counts))

    # Add smoothing to avoid division by zero on rare classes
    smoothed_counts = counts + (total_samples * smoothing)
    weights = total_samples / (num_classes * smoothed_counts)
    # Normalize weights so mean is 1.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Multi-class Focal Loss down-weighting well-classified easy samples."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes focal loss between logits [B, C] and targets [B]."""
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Gather probability corresponding to true class
        targets = targets.view(-1, 1)
        log_pt = log_probs.gather(1, targets).squeeze(-1)
        pt = probs.gather(1, targets).squeeze(-1)

        focal_weight = torch.pow(1.0 - pt, self.gamma)

        if self.alpha is not None:
            alpha_weight = self.alpha.gather(0, targets.squeeze(-1))
            focal_loss = -alpha_weight * focal_weight * log_pt
        else:
            focal_loss = -focal_weight * log_pt

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
