from pathlib import Path
from typing import Dict, Generator, List, Optional, Union
import numpy as np
import pandas as pd
from .base_loader import BaseModalityLoader

KAGGLE_EEG_CHANNELS = [
    "eeg_fp1",
    "eeg_f7",
    "eeg_f8",
    "eeg_t3",
    "eeg_t4",
    "eeg_t5",
    "eeg_t6",
    "eeg_o1",
    "eeg_o2",
    "eeg_fp2",
    "eeg_fz",
    "eeg_c3",
    "eeg_c4",
    "eeg_cz",
    "eeg_p3",
    "eeg_pz",
    "eeg_p4",
]

KAGGLE_CARDIO_CHANNELS = ["ecg", "r", "gsr"]

LABEL_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}


class KaggleAviationLoader(BaseModalityLoader):
    """Loader for Kaggle Reducing Commercial Aviation Fatalities dataset."""

    def __init__(
        self,
        time_column: str = "time",
        eeg_channels: Optional[List[str]] = None,
        cardio_channels: Optional[List[str]] = None,
        sampling_rate_hz: float = 256.0,
    ) -> None:
        self.eeg_channels = eeg_channels or KAGGLE_EEG_CHANNELS
        self.cardio_channels = cardio_channels or KAGGLE_CARDIO_CHANNELS
        all_channels = self.eeg_channels + self.cardio_channels
        super().__init__(
            time_column=time_column,
            channels=all_channels,
            sampling_rate_hz=sampling_rate_hz,
        )

    def load(
        self,
        source: Union[str, Path, pd.DataFrame],
        crew_filter: Optional[int] = None,
        experiment_filter: Optional[str] = None,
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        """Loads and filters raw Kaggle dataset into a memory-efficient DataFrame."""
        if isinstance(source, (str, Path)):
            df = pd.read_csv(source, nrows=max_rows)
        elif isinstance(source, pd.DataFrame):
            df = source.copy()
            if max_rows is not None:
                df = df.iloc[:max_rows]
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        if crew_filter is not None and "crew" in df.columns:
            df = df[df["crew"] == crew_filter]

        if experiment_filter is not None and "experiment" in df.columns:
            df = df[df["experiment"] == experiment_filter]

        self.validate_schema(df)
        return df

    def stream_chunks(
        self,
        filepath: Union[str, Path],
        chunksize: int = 100_000,
        crew_filter: Optional[int] = None,
    ) -> Generator[pd.DataFrame, None, None]:
        """Streams large CSV in chunks without exhausting system RAM."""
        for chunk in pd.read_csv(filepath, chunksize=chunksize):
            if crew_filter is not None and "crew" in chunk.columns:
                chunk = chunk[chunk["crew"] == crew_filter]
            if not chunk.empty:
                yield chunk

    def load_stratified_sample(
        self,
        filepath: Union[str, Path],
        samples_per_experiment: int = 50_000,
        chunksize: int = 100_000,
    ) -> pd.DataFrame:
        """Collects samples across distinct experimental regimes (CA, DA, SS) for balanced multi-class training."""
        collected: Dict[str, List[pd.DataFrame]] = {"CA": [], "DA": [], "SS": []}
        counts = {"CA": 0, "DA": 0, "SS": 0}

        for chunk in pd.read_csv(filepath, chunksize=chunksize):
            if "experiment" not in chunk.columns:
                return chunk.iloc[: samples_per_experiment * 3]

            for exp in ("CA", "DA", "SS"):
                if counts[exp] < samples_per_experiment:
                    sub = chunk[chunk["experiment"] == exp]
                    if not sub.empty:
                        needed = samples_per_experiment - counts[exp]
                        take = sub.iloc[:needed]
                        collected[exp].append(take)
                        counts[exp] += len(take)

            if all(counts[exp] >= samples_per_experiment for exp in counts):
                break

        frames = []
        for exp in ("CA", "DA", "SS"):
            if collected[exp]:
                frames.extend(collected[exp])

        if not frames:
            raise ValueError("No matching experiment frames collected.")

        balanced_df = pd.concat(frames, ignore_index=True)
        self.validate_schema(balanced_df)
        return balanced_df

    def extract_arrays(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """Separates loaded DataFrame into modal numpy arrays and session metadata."""
        self.validate_schema(df)

        time_vec = df[self.time_column].to_numpy(dtype=np.float64)
        eeg_arr = df[self.eeg_channels].to_numpy(dtype=np.float32)
        cardio_arr = df[self.cardio_channels].to_numpy(dtype=np.float32)

        # Encode target event if present
        if "event" in df.columns:
            raw_labels = df["event"].astype(str).values
            labels = np.array([LABEL_MAP.get(lbl, 0) for lbl in raw_labels], dtype=np.int64)
        else:
            labels = np.zeros(len(df), dtype=np.int64)

        # Unique session key compounding crew, experiment, and seat
        if {"crew", "experiment", "seat"}.issubset(df.columns):
            session_ids = (
                df["crew"].astype(str)
                + "_"
                + df["experiment"].astype(str)
                + "_"
                + df["seat"].astype(str)
            ).to_numpy()
        else:
            session_ids = np.zeros(len(df), dtype=object)

        return {
            "time": time_vec,
            "eeg": eeg_arr,
            "cardio": cardio_arr,
            "label": labels,
            "session_id": session_ids,
        }
