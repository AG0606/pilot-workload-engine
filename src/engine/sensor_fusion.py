from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class CognitiveLoadLevel(str, Enum):
    NORMAL = "NORMAL"
    ELEVATED_LOAD = "ELEVATED_LOAD"
    CRITICAL_OVERLOAD = "CRITICAL_OVERLOAD"


@dataclass
class CognitiveSnapshotRecord:
    flight_id: str
    timestamp: float
    raw_eeg_metric: float
    raw_hrv_metric: float
    raw_gsr_metric: float
    raw_composite_z: float
    filtered_cli: float
    state_uncertainty_p: float
    load_level: str
    alarm_triggered: bool


class KalmanCognitiveSensorFusion:
    """Kalman Filter & Weighted Multi-Sensor Fusion Engine for Cognitive Load Index (CLI).

    Implements Section 5 of the e-Pilot DBMS Architecture:
    - Fuses heterogeneous asynchronous streams (EEG spectral exhaustion, HRV vagal suppression, GSR arousal).
    - Tracks latent cognitive state with 1D Linear Kalman Filter to remove sensor noise and motion artifacts.
    - Emits normalized Cognitive Load Index (CLI in [0.0, 1.0]) and structured database records.
    """

    def __init__(
        self,
        process_noise_q: float = 1e-3,
        measurement_noise_r: float = 0.05,
        initial_state: float = 0.15,
        initial_covariance_p: float = 0.10,
        weights: Optional[Dict[str, float]] = None,
        elevated_threshold: float = 0.40,
        critical_threshold: float = 0.75,
    ) -> None:
        self.q = float(process_noise_q)
        self.r = float(measurement_noise_r)
        self.x = float(initial_state)
        self.p = float(initial_covariance_p)

        self.weights = weights or {"eeg": 0.45, "hrv": 0.30, "gsr": 0.25}
        # Normalize weights to sum to 1.0
        total_w = sum(self.weights.values())
        self.weights = {k: v / total_w for k, v in self.weights.items()}

        self.elevated_threshold = elevated_threshold
        self.critical_threshold = critical_threshold

    def compute_composite_measurement(
        self,
        eeg_engagement: float,  # Normalized 0.0 - 1.0
        hrv_stress: float,       # Normalized 0.0 - 1.0 (higher = more stress / lower RMSSD)
        gsr_conductance: float,  # Normalized 0.0 - 1.0
    ) -> float:
        """Combines normalized modality indicators into a single scalar observation z_t."""
        eeg_val = np.clip(float(eeg_engagement), 0.0, 1.0)
        hrv_val = np.clip(float(hrv_stress), 0.0, 1.0)
        gsr_val = np.clip(float(gsr_conductance), 0.0, 1.0)

        z = (
            self.weights["eeg"] * eeg_val
            + self.weights["hrv"] * hrv_val
            + self.weights["gsr"] * gsr_val
        )
        return float(np.clip(z, 0.0, 1.0))

    def update(
        self,
        eeg_metric: float,
        hrv_metric: float,
        gsr_metric: float,
    ) -> Tuple[float, float, CognitiveLoadLevel, float]:
        """Performs Kalman predict-update cycle with multi-sensor observation.

        Returns:
            (filtered_cli, uncertainty_p, load_level, raw_z)
        """
        # 1. Prediction step
        x_pred = self.x
        p_pred = self.p + self.q

        # 2. Measurement calculation
        z = self.compute_composite_measurement(eeg_metric, hrv_metric, gsr_metric)

        # 3. Innovation and Kalman Gain
        y_residual = z - x_pred
        s_innovation = p_pred + self.r
        k_gain = p_pred / s_innovation

        # 4. State update
        self.x = float(np.clip(x_pred + k_gain * y_residual, 0.0, 1.0))
        self.p = float((1.0 - k_gain) * p_pred)

        # 5. Cognitive Load Level categorization
        if self.x >= self.critical_threshold:
            level = CognitiveLoadLevel.CRITICAL_OVERLOAD
        elif self.x >= self.elevated_threshold:
            level = CognitiveLoadLevel.ELEVATED_LOAD
        else:
            level = CognitiveLoadLevel.NORMAL

        return self.x, self.p, level, z

    def filter_trajectory(
        self,
        eeg_series: np.ndarray,
        hrv_series: np.ndarray,
        gsr_series: np.ndarray,
        flight_id: str = "FLIGHT_001",
        start_time_sec: float = 0.0,
        dt_sec: float = 1.0,
    ) -> List[CognitiveSnapshotRecord]:
        """Filters an entire flight recording trajectory, emitting structured database records."""
        n_steps = len(eeg_series)
        records: List[CognitiveSnapshotRecord] = []

        for i in range(n_steps):
            t = start_time_sec + i * dt_sec
            eeg_v = float(eeg_series[i])
            hrv_v = float(hrv_series[i])
            gsr_v = float(gsr_series[i])

            cli, p_cov, level, z = self.update(eeg_v, hrv_v, gsr_v)
            alarm = level != CognitiveLoadLevel.NORMAL

            rec = CognitiveSnapshotRecord(
                flight_id=flight_id,
                timestamp=round(t, 2),
                raw_eeg_metric=round(eeg_v, 4),
                raw_hrv_metric=round(hrv_v, 4),
                raw_gsr_metric=round(gsr_v, 4),
                raw_composite_z=round(z, 4),
                filtered_cli=round(cli, 4),
                state_uncertainty_p=round(p_cov, 6),
                load_level=level.value,
                alarm_triggered=alarm,
            )
            records.append(rec)

        return records
