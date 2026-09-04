from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd


class BaseModalityLoader(ABC):
    """Abstract base loader defining interface for heterogeneous cockpit modalities."""

    def __init__(
        self,
        time_column: str = "time",
        channels: Optional[List[str]] = None,
        sampling_rate_hz: Optional[float] = None,
    ) -> None:
        self.time_column = time_column
        self.channels = channels or []
        self.sampling_rate_hz = sampling_rate_hz

    @abstractmethod
    def load(self, source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Load raw modality data into a standardized pandas DataFrame."""
        pass

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Ensure the loaded DataFrame contains required time column and channels."""
        if self.time_column not in df.columns:
            raise KeyError(f"Missing mandatory time column: {self.time_column}")
        missing_channels = [ch for ch in self.channels if ch not in df.columns]
        if missing_channels:
            raise KeyError(f"Missing channels in dataset: {missing_channels}")
        return True

    def get_time_vector(self, df: pd.DataFrame) -> np.ndarray:
        """Extract continuous time vector in seconds as float64."""
        self.validate_schema(df)
        time_series = df[self.time_column].to_numpy(dtype=np.float64)
        return time_series

    def estimate_sampling_rate(self, df: pd.DataFrame) -> float:
        """Infer sampling rate from median inter-sample time intervals."""
        time_vec = self.get_time_vector(df)
        if len(time_vec) < 2:
            raise ValueError("Insufficient time points to estimate sampling rate.")
        dt = np.median(np.diff(time_vec))
        if dt <= 0:
            raise ValueError("Non-positive time delta encountered.")
        return float(1.0 / dt)
