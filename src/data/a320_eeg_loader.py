from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from .base_loader import BaseModalityLoader

# Standard 10-20 EEG electrode subset used in A320 flight deck study (Sensors MDPI 2024)
A320_EEG_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"
]


class A320FlightDeckEEGLoader(BaseModalityLoader):
    """Loader and schema adapter for the A320 Flight-Deck EEG Mental Workload Dataset (Sensors MDPI 2024).

    Dataset DOI: 10.5565/ddd.uab.cat/259591
    Supports:
      - Parquet & CSV session recordings
      - NASA-TLX workload alignment
      - Flight deck emergency vs baseline phase labeling
    """

    def __init__(
        self,
        time_column: str = "time",
        eeg_channels: Optional[List[str]] = None,
        sampling_rate_hz: float = 250.0,
    ) -> None:
        self.eeg_channels = eeg_channels or A320_EEG_CHANNELS
        super().__init__(
            time_column=time_column,
            channels=self.eeg_channels,
            sampling_rate_hz=sampling_rate_hz,
        )

    def load(
        self,
        source: Union[str, Path, pd.DataFrame],
        pilot_id: Optional[str] = None,
        experiment_mode: Optional[str] = None,
    ) -> pd.DataFrame:
        """Loads A320 EEG session recording into a standardized DataFrame."""
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix.lower() == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
        elif isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            raise TypeError(f"Unsupported source format: {type(source)}")

        if pilot_id and "pilot_id" in df.columns:
            df = df[df["pilot_id"] == pilot_id]

        if experiment_mode and "experiment" in df.columns:
            df = df[df["experiment"] == experiment_mode]

        self.validate_schema(df)
        return df

    def generate_synthetic_a320_flight(
        self,
        duration_sec: float = 120.0,
        include_tlx_scores: bool = True,
    ) -> pd.DataFrame:
        """Generates realistic synthetic A320 flight simulator session matching the MDPI Sensors 2024 schema."""
        n_samples = int(duration_sec * self.sampling_rate_hz)
        time_vec = np.linspace(0, duration_sec, n_samples, endpoint=False)

        # Base nominal alpha rhythms around 10 Hz + theta oscillations around 6 Hz
        t = time_vec
        eeg_signals = {}
        for ch in self.eeg_channels:
            # Baseline EEG: 10 Hz alpha + pink noise
            alpha = 15.0 * np.sin(2 * np.pi * 10.0 * t + np.random.uniform(0, 2*np.pi))
            theta = 10.0 * np.sin(2 * np.pi * 6.0 * t + np.random.uniform(0, 2*np.pi))
            noise = np.random.randn(n_samples) * 5.0

            # During emergency phase (second half), theta elevates and alpha suppresses (cognitive overload)
            overload_mask = (t >= duration_sec / 2.0).astype(np.float32)
            sig = alpha * (1.0 - 0.5 * overload_mask) + theta * (1.0 + 1.2 * overload_mask) + noise
            eeg_signals[ch] = sig.astype(np.float32)

        data = {"time": time_vec, **eeg_signals}
        data["pilot_id"] = "Pilot_01"
        data["experiment"] = "A320_SIM_WIND_SHEAR"
        data["flight_phase"] = np.where(time_vec < duration_sec / 2.0, "CRUISE", "EMERGENCY_DESCENT")
        data["mental_workload_level"] = np.where(time_vec < duration_sec / 2.0, 0, 2)  # 0: Low, 2: High

        if include_tlx_scores:
            # NASA-TLX 0-100 score
            data["nasa_tlx_score"] = np.where(time_vec < duration_sec / 2.0, 28.5, 84.0)

        return pd.DataFrame(data)
