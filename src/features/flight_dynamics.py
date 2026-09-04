from typing import Dict, List, Optional
import numpy as np

DEFAULT_CONTEXT_CHANNELS = [
    "pitch",
    "roll",
    "yaw_rate",
    "vertical_g",
    "lateral_g",
    "indicated_airspeed",
    "barometric_altitude",
    "autopilot_mode",
]


def compute_flight_dynamics_features(
    context_window: np.ndarray,
    channel_names: Optional[List[str]] = None,
    fs_hz: float = 20.0,
) -> Dict[str, float]:
    """Computes kinematic deltas, turbulence intensity, and G-transition metrics over a flight window."""
    if channel_names is None:
        channel_names = DEFAULT_CONTEXT_CHANNELS

    if context_window.ndim != 2:
        raise ValueError(f"Expected 2D array [channels, time], got shape {context_window.shape}")

    channel_map = {name: i for i, name in enumerate(channel_names) if i < context_window.shape[0]}
    features: Dict[str, float] = {}

    dt = 1.0 / fs_hz if fs_hz > 0 else 1.0

    # Vertical G-load dynamics and turbulence estimation
    if "vertical_g" in channel_map:
        vert_g = context_window[channel_map["vertical_g"]]
        # Standard deviation around 1.0G trim equilibrium
        features["turbulence_intensity_rms"] = float(np.sqrt(np.mean((vert_g - 1.0) ** 2)))
        features["vertical_g_std"] = float(np.std(vert_g))
        features["vertical_g_max"] = float(np.max(vert_g))
        features["vertical_g_min"] = float(np.min(vert_g))
        features["vertical_g_peak_to_peak"] = float(np.ptp(vert_g))

        # First derivative of vertical load factor (jerk proxy)
        if len(vert_g) > 1:
            jerk = np.diff(vert_g) / dt
            features["max_vertical_jerk"] = float(np.max(np.abs(jerk)))
        else:
            features["max_vertical_jerk"] = 0.0

    # Lateral load factor transitions
    if "lateral_g" in channel_map:
        lat_g = context_window[channel_map["lateral_g"]]
        features["lateral_g_std"] = float(np.std(lat_g))
        features["lateral_g_max_abs"] = float(np.max(np.abs(lat_g)))

    # Attitude angular rate deltas
    if "pitch" in channel_map:
        pitch = context_window[channel_map["pitch"]]
        features["pitch_mean"] = float(np.mean(pitch))
        features["pitch_delta"] = float(pitch[-1] - pitch[0]) if len(pitch) > 1 else 0.0
        if len(pitch) > 1:
            pitch_rate = np.diff(pitch) / dt
            features["pitch_rate_rms"] = float(np.sqrt(np.mean(pitch_rate**2)))
        else:
            features["pitch_rate_rms"] = 0.0

    if "roll" in channel_map:
        roll = context_window[channel_map["roll"]]
        features["roll_mean"] = float(np.mean(roll))
        features["roll_delta"] = float(roll[-1] - roll[0]) if len(roll) > 1 else 0.0
        if len(roll) > 1:
            roll_rate = np.diff(roll) / dt
            features["roll_rate_rms"] = float(np.sqrt(np.mean(roll_rate**2)))
        else:
            features["roll_rate_rms"] = 0.0

    if "yaw_rate" in channel_map:
        yaw_rate = context_window[channel_map["yaw_rate"]]
        features["yaw_rate_mean"] = float(np.mean(yaw_rate))
        features["yaw_rate_max_abs"] = float(np.max(np.abs(yaw_rate)))

    # Airspeed and barometric altitude rate of change
    if "indicated_airspeed" in channel_map:
        ias = context_window[channel_map["indicated_airspeed"]]
        features["airspeed_mean"] = float(np.mean(ias))
        features["airspeed_rate"] = float((ias[-1] - ias[0]) / (len(ias) * dt)) if len(ias) > 1 else 0.0

    if "barometric_altitude" in channel_map:
        alt = context_window[channel_map["barometric_altitude"]]
        features["altitude_mean"] = float(np.mean(alt))
        features["vertical_speed"] = float((alt[-1] - alt[0]) / (len(alt) * dt)) if len(alt) > 1 else 0.0

    # Autopilot status
    if "autopilot_mode" in channel_map:
        ap = context_window[channel_map["autopilot_mode"]]
        features["autopilot_active_ratio"] = float(np.mean(ap > 0.5))

    return features
