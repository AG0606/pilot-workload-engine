from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from .base_loader import BaseModalityLoader

DEFAULT_OCULAR_CHANNELS = [
    "pupil_diameter_left",
    "pupil_diameter_right",
    "blink_detected",
    "fixation_duration",
]


class CogPilotLoader(BaseModalityLoader):
    """Loader for CogPilot ocular metrics and gaze dynamics."""

    def __init__(
        self,
        time_column: str = "time",
        channels: Optional[List[str]] = None,
        sampling_rate_hz: float = 60.0,
    ) -> None:
        selected_channels = channels or DEFAULT_OCULAR_CHANNELS
        super().__init__(
            time_column=time_column,
            channels=selected_channels,
            sampling_rate_hz=sampling_rate_hz,
        )

    def load(self, source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Loads eye tracking and ocular telemetry."""
        if isinstance(source, (str, Path)):
            df = pd.read_csv(source)
        elif isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        self.validate_schema(df)
        return df

    def extract_arrays(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extracts ocular channel arrays and timestamps."""
        self.validate_schema(df)
        time_vec = df[self.time_column].to_numpy(dtype=np.float64)
        ocular_arr = df[self.channels].to_numpy(dtype=np.float32)

        session_id = (
            df["session_id"].to_numpy()
            if "session_id" in df.columns
            else np.zeros(len(df), dtype=object)
        )

        return {
            "time": time_vec,
            "ocular": ocular_arr,
            "session_id": session_id,
        }
