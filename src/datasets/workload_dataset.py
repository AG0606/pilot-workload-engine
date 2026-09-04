from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PilotWorkloadDataset(Dataset):
    """PyTorch Dataset yielding synchronized multimodal sliding-window tensors for cognitive workload."""

    def __init__(
        self,
        eeg: Union[np.ndarray, torch.Tensor],
        cardio: Union[np.ndarray, torch.Tensor],
        ocular: Union[np.ndarray, torch.Tensor],
        context: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
    ) -> None:
        self.eeg = self._to_tensor(eeg, dtype=torch.float32)
        self.cardio = self._to_tensor(cardio, dtype=torch.float32)
        self.ocular = self._to_tensor(ocular, dtype=torch.float32)
        self.context = self._to_tensor(context, dtype=torch.float32)
        self.labels = self._to_tensor(labels, dtype=torch.int64).squeeze()

        num_samples = len(self.labels)
        for name, tensor in [
            ("eeg", self.eeg),
            ("cardio", self.cardio),
            ("ocular", self.ocular),
            ("context", self.context),
        ]:
            if len(tensor) != num_samples:
                raise ValueError(
                    f"Sample count mismatch: {name} has length {len(tensor)}, expected {num_samples}."
                )

        self.num_samples = num_samples

    @staticmethod
    def _to_tensor(data: Union[np.ndarray, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data.to(dtype=dtype)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(dtype=dtype)
        else:
            raise TypeError(f"Unsupported data type for tensor conversion: {type(data)}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "eeg": self.eeg[idx],
            "cardio": self.cardio[idx],
            "ocular": self.ocular[idx],
            "context": self.context[idx],
            "label": self.labels[idx],
        }

    @classmethod
    def from_window_dict(cls, windows: Dict[str, np.ndarray]) -> "PilotWorkloadDataset":
        """Instantiates dataset directly from the output of SlidingWindowExtractor."""
        required_keys = {"eeg", "cardio", "ocular", "context", "label"}
        missing_keys = required_keys - set(windows.keys())
        if missing_keys:
            raise KeyError(f"Window dictionary missing required modalities: {missing_keys}")

        return cls(
            eeg=windows["eeg"],
            cardio=windows["cardio"],
            ocular=windows["ocular"],
            context=windows["context"],
            labels=windows["label"],
        )

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes dataset tensors to disk."""
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "eeg": self.eeg,
                "cardio": self.cardio,
                "ocular": self.ocular,
                "context": self.context,
                "labels": self.labels,
            },
            target,
        )

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "PilotWorkloadDataset":
        """Loads serialized dataset from disk."""
        data = torch.load(filepath, map_location="cpu")
        return cls(
            eeg=data["eeg"],
            cardio=data["cardio"],
            ocular=data["ocular"],
            context=data["context"],
            labels=data["labels"],
        )


def create_workload_dataloader(
    dataset: PilotWorkloadDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    """Factory helper to create standard PyTorch DataLoader for multimodal workload batches."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )
