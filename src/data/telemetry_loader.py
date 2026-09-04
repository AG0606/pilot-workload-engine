from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from .base_loader import BaseModalityLoader

DEFAULT_TELEMETRY_CHANNELS = [
    "pitch",
    "roll",
    "yaw_rate",
    "vertical_g",
    "lateral_g",
    "indicated_airspeed",
    "barometric_altitude",
    "autopilot_mode",
]


class TelemetryLoader(BaseModalityLoader):
    """Loader for flight kinematics, QAR parameters, and autopilot status."""

    def __init__(
        self,
        time_column: str = "time",
        channels: Optional[List[str]] = None,
        sampling_rate_hz: float = 10.0,
    ) -> None:
        selected_channels = channels or DEFAULT_TELEMETRY_CHANNELS
        super().__init__(
            time_column=time_column,
            channels=selected_channels,
            sampling_rate_hz=sampling_rate_hz,
        )

    def load(self, source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Loads aircraft telemetry into a validated DataFrame."""
        if isinstance(source, (str, Path)):
            df = pd.read_csv(source)
        elif isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        self.validate_schema(df)
        return df

    def extract_arrays(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extracts kinematics array and corresponding flight time vector."""
        self.validate_schema(df)
        time_vec = df[self.time_column].to_numpy(dtype=np.float64)
        context_arr = df[self.channels].to_numpy(dtype=np.float32)

        session_id = (
            df["flight_id"].to_numpy()
            if "flight_id" in df.columns
            else np.zeros(len(df), dtype=object)
        )

        return {
            "time": time_vec,
            "context": context_arr,
            "session_id": session_id,
        }
