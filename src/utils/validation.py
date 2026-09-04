from typing import Dict, Optional, Union
import numpy as np
import torch


def validate_timestamps(time_vec: np.ndarray, max_allowed_gap_sec: float = 1.0) -> bool:
    """Verifies that timestamps are monotonically increasing, finite, and within gap tolerance."""
    if len(time_vec) < 2:
        return True

    if not np.all(np.isfinite(time_vec)):
        raise ValueError("Timestamps contain NaN or Inf values.")

    diffs = np.diff(time_vec)
    if np.any(diffs <= 0):
        raise ValueError("Timestamps are not strictly monotonically increasing.")

    max_gap = np.max(diffs)
    if max_gap > max_allowed_gap_sec:
        raise ValueError(
            f"Detected timestamp gap of {max_gap:.4f}s exceeding threshold {max_allowed_gap_sec:.4f}s."
        )

    return True


def validate_sliding_windows(
    windows: Dict[str, np.ndarray],
    expected_samples: int = 80,
    expected_modalities: Optional[Dict[str, int]] = None,
) -> bool:
    """Verifies shape, sample count, and absence of NaNs across extracted window dictionary."""
    if not windows:
        raise ValueError("Empty window dictionary provided.")

    num_windows = None
    for mod_name, arr in windows.items():
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Modality {mod_name} must be a numpy.ndarray, got {type(arr)}")

        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Modality {mod_name} contains NaN or Inf values.")

        if num_windows is None:
            num_windows = len(arr)
        elif len(arr) != num_windows:
            raise ValueError(
                f"Window count mismatch: {mod_name} has {len(arr)} windows, expected {num_windows}."
            )

        if mod_name != "label":
            if arr.ndim != 3:
                raise ValueError(
                    f"Modality {mod_name} array must have shape [windows, channels, samples], got ndim={arr.ndim}"
                )
            if arr.shape[2] != expected_samples:
                raise ValueError(
                    f"Modality {mod_name} temporal length {arr.shape[2]} does not match expected {expected_samples}."
                )
            if expected_modalities and mod_name in expected_modalities:
                expected_channels = expected_modalities[mod_name]
                if arr.shape[1] != expected_channels:
                    raise ValueError(
                        f"Modality {mod_name} channel count {arr.shape[1]} does not match expected {expected_channels}."
                    )

    return True


def validate_dataset_batch(batch: Dict[str, torch.Tensor], num_classes: int = 4) -> bool:
    """Verifies PyTorch batch dictionary types, shapes, and label integrity."""
    required_keys = {"eeg", "cardio", "ocular", "context", "label"}
    missing = required_keys - set(batch.keys())
    if missing:
        raise KeyError(f"Batch dictionary is missing keys: {missing}")

    batch_size = len(batch["label"])
    if batch["label"].dtype not in (torch.int32, torch.int64):
        raise TypeError(f"Labels must be integer type, got {batch['label'].dtype}")

    if torch.any(batch["label"] < 0) or torch.any(batch["label"] >= num_classes):
        raise ValueError(f"Batch labels contain values outside valid range [0, {num_classes - 1}]")

    for key in ("eeg", "cardio", "ocular", "context"):
        tensor = batch[key]
        if tensor.dtype != torch.float32:
            raise TypeError(f"Modality {key} tensor must be float32, got {tensor.dtype}")
        if len(tensor) != batch_size:
            raise ValueError(f"Batch size mismatch for {key}: expected {batch_size}, got {len(tensor)}")
        if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
            raise ValueError(f"Modality {key} contains NaN or Inf in batch.")

    return True
