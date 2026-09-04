from typing import Dict, Optional, Tuple
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, welch


def detect_r_peaks(
    ecg_signal: np.ndarray,
    fs: float = 256.0,
    min_rr_sec: float = 0.35,
) -> np.ndarray:
    """Identifies ventricular R-wave peak indices from raw ECG lead signals."""
    if len(ecg_signal) < 3:
        return np.empty(0, dtype=np.int64)

    # Normalize signal amplitude
    sig_norm = (ecg_signal - np.mean(ecg_signal)) / (np.std(ecg_signal) + 1e-8)
    min_distance = int(round(min_rr_sec * fs))
    threshold = float(np.percentile(sig_norm, 85))

    peaks, _ = find_peaks(
        sig_norm,
        distance=max(min_distance, 1),
        height=max(threshold, 0.5),
    )
    return peaks


def compute_time_domain_hrv(
    rr_intervals_ms: np.ndarray,
) -> Dict[str, float]:
    """Extracts standard time-domain heart rate variability metrics."""
    if len(rr_intervals_ms) < 2:
        return {
            "mean_hr_bpm": 0.0,
            "mean_nn_ms": 0.0,
            "sdnn_ms": 0.0,
            "rmssd_ms": 0.0,
            "pnn50": 0.0,
        }

    diff_nn = np.diff(rr_intervals_ms)
    sdnn = float(np.std(rr_intervals_ms, ddof=1)) if len(rr_intervals_ms) > 1 else 0.0
    rmssd = float(np.sqrt(np.mean(diff_nn**2)))
    pnn50 = float(np.mean(np.abs(diff_nn) > 50.0) * 100.0)
    mean_nn = float(np.mean(rr_intervals_ms))
    mean_hr = float(60000.0 / mean_nn) if mean_nn > 0 else 0.0

    return {
        "mean_hr_bpm": mean_hr,
        "mean_nn_ms": mean_nn,
        "sdnn_ms": sdnn,
        "rmssd_ms": rmssd,
        "pnn50": pnn50,
    }


def compute_frequency_domain_hrv(
    rr_intervals_ms: np.ndarray,
    interp_fs_hz: float = 4.0,
    lf_band: Tuple[float, float] = (0.04, 0.15),
    hf_band: Tuple[float, float] = (0.15, 0.40),
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Interpolates irregularly sampled RR intervals and computes Welch LF and HF spectral powers."""
    if len(rr_intervals_ms) < 4:
        return {
            "lf_power": 0.0,
            "hf_power": 0.0,
            "lf_hf_ratio": 0.0,
        }

    # Derive continuous cumulative time axis
    time_series = np.cumsum(rr_intervals_ms) / 1000.0
    time_series -= time_series[0]
    total_time = time_series[-1]

    if total_time <= 1.0:
        return {
            "lf_power": 0.0,
            "hf_power": 0.0,
            "lf_hf_ratio": 0.0,
        }

    # Interpolate RR tachogram to uniform 4 Hz grid
    t_uniform = np.arange(0, total_time, 1.0 / interp_fs_hz)
    if len(t_uniform) < 4:
        return {
            "lf_power": 0.0,
            "hf_power": 0.0,
            "lf_hf_ratio": 0.0,
        }

    interp_fn = interp1d(time_series, rr_intervals_ms, kind="linear", fill_value="extrapolate")
    rr_uniform = interp_fn(t_uniform)

    # Detrend interpolated tachogram
    rr_detrended = rr_uniform - np.mean(rr_uniform)

    freqs, psd = welch(
        rr_detrended,
        fs=interp_fs_hz,
        nperseg=min(len(rr_detrended), 256),
    )

    lf_mask = (freqs >= lf_band[0]) & (freqs < lf_band[1])
    hf_mask = (freqs >= hf_band[0]) & (freqs <= hf_band[1])

    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    lf_power = float(np.sum(psd[lf_mask]) * df) if np.any(lf_mask) else 0.0
    hf_power = float(np.sum(psd[hf_mask]) * df) if np.any(hf_mask) else 0.0
    lf_hf_ratio = float(lf_power / (hf_power + eps))

    return {
        "lf_power": lf_power,
        "hf_power": hf_power,
        "lf_hf_ratio": lf_hf_ratio,
    }


def compute_hrv_features(
    ecg_or_rr: np.ndarray,
    fs: float = 256.0,
    is_rr_ms: bool = False,
) -> Dict[str, float]:
    """End-to-end HRV feature extraction pipeline handling raw ECG or pre-calculated RR intervals."""
    if is_rr_ms:
        rr_ms = ecg_or_rr
    else:
        peaks = detect_r_peaks(ecg_signal=ecg_or_rr, fs=fs)
        if len(peaks) < 2:
            rr_ms = np.empty(0, dtype=np.float64)
        else:
            rr_ms = (np.diff(peaks) / fs) * 1000.0

    time_metrics = compute_time_domain_hrv(rr_ms)
    freq_metrics = compute_frequency_domain_hrv(rr_ms)

    time_metrics.update(freq_metrics)
    return time_metrics
