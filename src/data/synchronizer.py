from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class MultiModalSynchronizer:
    """Synchronizes heterogeneous time-series modalities onto a standardized reference grid."""

    def __init__(self, target_fs_hz: float = 20.0) -> None:
        if target_fs_hz <= 0:
            raise ValueError("Target sampling rate must be positive.")
        self.target_fs_hz = float(target_fs_hz)
        self.dt = 1.0 / self.target_fs_hz

    def align_modality(
        self,
        time_vec: np.ndarray,
        data: np.ndarray,
        reference_time: np.ndarray,
        is_categorical: bool = False,
    ) -> np.ndarray:
        """Interpolates multi-channel array onto uniform reference timestamps."""
        if len(time_vec) < 2:
            raise ValueError("Insufficient points for interpolation.")

        # Non-numeric or categorical streams use nearest neighbor indexing
        if not np.issubdtype(data.dtype, np.number) or is_categorical:
            idx = np.clip(np.searchsorted(time_vec, reference_time), 0, len(time_vec) - 1)
            return data[idx]

        if data.ndim == 1:
            data = data[:, np.newaxis]

        interpolator = interp1d(
            time_vec,
            data,
            kind="linear",
            axis=0,
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )
        resampled = interpolator(reference_time)
        return resampled

    def synchronize_streams(
        self,
        streams: Dict[str, Tuple[np.ndarray, np.ndarray]],
        categorical_streams: Optional[List[str]] = None,
        common_time_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Aligns multiple (time, data) streams to a shared reference grid."""
        if not streams:
            raise ValueError("No streams provided for synchronization.")
        categorical_set = set(categorical_streams or [])

        if common_time_range is not None:
            t_start, t_end = common_time_range
        else:
            t_start = max(t[0] for t, _ in streams.values())
            t_end = min(t[-1] for t, _ in streams.values())

        if t_end <= t_start:
            raise ValueError(f"Invalid overlapping time range: [{t_start}, {t_end}]")

        ref_time = np.arange(t_start, t_end, self.dt)
        if len(ref_time) == 0:
            raise ValueError("Reference time grid generated zero points.")

        synchronized: Dict[str, np.ndarray] = {}
        for name, (time_vec, data) in streams.items():
            is_cat = name in categorical_set
            synchronized[name] = self.align_modality(
                time_vec=time_vec,
                data=data,
                reference_time=ref_time,
                is_categorical=is_cat,
            )

        return ref_time, synchronized


class SlidingWindowExtractor:
    """Segments synchronized continuous streams into sliding window blocks with boundary integrity."""

    def __init__(
        self,
        window_size_sec: float = 4.0,
        stride_sec: float = 1.0,
        target_fs_hz: float = 20.0,
        max_timestamp_gap_sec: float = 0.15,
    ) -> None:
        self.window_size_sec = window_size_sec
        self.stride_sec = stride_sec
        self.target_fs_hz = target_fs_hz
        self.max_timestamp_gap_sec = max_timestamp_gap_sec

        self.window_samples = int(round(window_size_sec * target_fs_hz))
        self.stride_samples = int(round(stride_sec * target_fs_hz))

        if self.window_samples <= 0 or self.stride_samples <= 0:
            raise ValueError("Window and stride samples must be positive integers.")

    def extract_windows(
        self,
        data_streams: Dict[str, np.ndarray],
        reference_time: np.ndarray,
        session_ids: Optional[np.ndarray] = None,
        label_stream_name: str = "label",
    ) -> Dict[str, np.ndarray]:
        """Extracts strictly continuous sliding windows, discarding boundary-crossing windows."""
        total_samples = len(reference_time)
        if total_samples < self.window_samples:
            return {k: np.empty((0, 0, 0)) for k in data_streams}

        valid_window_indices: List[Tuple[int, int]] = []
        expected_window_duration = (self.window_samples - 1) / self.target_fs_hz

        start_idx = 0
        while start_idx + self.window_samples <= total_samples:
            end_idx = start_idx + self.window_samples
            win_time = reference_time[start_idx:end_idx]

            # Enforce max sampling gap constraint
            time_diffs = np.diff(win_time)
            if np.any(time_diffs > self.max_timestamp_gap_sec) or np.any(time_diffs <= 0):
                start_idx += self.stride_samples
                continue

            # Enforce total window duration tolerance
            actual_duration = win_time[-1] - win_time[0]
            if abs(actual_duration - expected_window_duration) > (self.max_timestamp_gap_sec):
                start_idx += self.stride_samples
                continue

            # Enforce single session boundary constraint
            if session_ids is not None:
                win_sessions = session_ids[start_idx:end_idx]
                if win_sessions[0] != win_sessions[-1] or not np.all(win_sessions == win_sessions[0]):
                    start_idx += self.stride_samples
                    continue

            valid_window_indices.append((start_idx, end_idx))
            start_idx += self.stride_samples

        if not valid_window_indices:
            result = {}
            for name, arr in data_streams.items():
                if name == label_stream_name:
                    result[name] = np.empty((0,), dtype=np.int64)
                else:
                    ch_dim = arr.shape[1] if arr.ndim > 1 else 1
                    result[name] = np.empty((0, ch_dim, self.window_samples), dtype=np.float32)
            return result

        result: Dict[str, np.ndarray] = {}
        num_windows = len(valid_window_indices)

        for name, arr in data_streams.items():
            if name == label_stream_name:
                # Mode or terminal timestamp label representation
                labels = np.zeros(num_windows, dtype=np.int64)
                for w_idx, (s_i, e_i) in enumerate(valid_window_indices):
                    win_labels = arr[s_i:e_i].squeeze()
                    # Assign label of final timestamp in window for predictive causal integrity
                    labels[w_idx] = int(win_labels[-1])
                result[name] = labels
            else:
                arr_2d = arr if arr.ndim > 1 else arr[:, np.newaxis]
                num_channels = arr_2d.shape[1]
                # Format: [num_windows, num_channels, window_samples]
                windows_tensor = np.zeros((num_windows, num_channels, self.window_samples), dtype=np.float32)
                for w_idx, (s_i, e_i) in enumerate(valid_window_indices):
                    # Transpose slice from [time, channels] to [channels, time]
                    windows_tensor[w_idx] = arr_2d[s_i:e_i].T
                result[name] = windows_tensor

        return result
