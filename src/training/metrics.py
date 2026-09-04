from typing import Dict, Union
import numpy as np
import torch


def compute_kaggle_log_loss(
    y_true: Union[np.ndarray, torch.Tensor],
    y_prob: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-15,
) -> float:
    """Computes Kaggle multi-class log loss with boundary probability clipping."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().numpy()

    # Clip probabilities to avoid extreme penalization from log(0)
    probs_clipped = np.clip(y_prob, eps, 1.0 - eps)
    # Re-normalize rows to ensure probability sum of 1
    probs_norm = probs_clipped / probs_clipped.sum(axis=1, keepdims=True)

    n_samples = len(y_true)
    true_class_probs = probs_norm[np.arange(n_samples), y_true]
    log_loss_val = -float(np.mean(np.log(true_class_probs)))
    return log_loss_val


def compute_classification_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_prob: Union[np.ndarray, torch.Tensor],
    num_classes: int = 4,
) -> Dict[str, Union[float, np.ndarray]]:
    """Calculates accuracy, balanced accuracy, macro F1, per-class sensitivity, and log loss."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().numpy()

    y_pred = np.argmax(y_prob, axis=-1)
    acc = float(np.mean(y_pred == y_true))

    # Confusion matrix
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            conf_mat[t, p] += 1

    per_class_recalls = []
    per_class_precisions = []
    per_class_f1s = []

    for c in range(num_classes):
        tp = conf_mat[c, c]
        fn = np.sum(conf_mat[c, :]) - tp
        fp = np.sum(conf_mat[:, c]) - tp

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_recalls.append(recall)
        per_class_precisions.append(precision)
        per_class_f1s.append(f1)

    balanced_acc = float(np.mean(per_class_recalls))
    macro_f1 = float(np.mean(per_class_f1s))
    log_loss_val = compute_kaggle_log_loss(y_true, y_prob)

    metrics: Dict[str, Union[float, np.ndarray]] = {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "kaggle_log_loss": log_loss_val,
        "confusion_matrix": conf_mat,
    }

    state_names = ["baseline", "channelized_attention", "diverted_attention", "startle"]
    for c, name in enumerate(state_names):
        metrics[f"recall_{name}"] = float(per_class_recalls[c])
        metrics[f"f1_{name}"] = float(per_class_f1s[c])

    return metrics
