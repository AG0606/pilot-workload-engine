from typing import Dict, Optional, Tuple
import numpy as np
from scipy.integrate import simpson
from scipy.signal import welch

DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def compute_welch_psd(
    signal: np.ndarray,
    fs: float = 256.0,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    axis: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates Welch Power Spectral Density along the specified temporal axis."""
    n_samples = signal.shape[axis]
    if nperseg is None:
        nperseg = min(n_samples, int(fs * 2.0)) if n_samples >= int(fs * 2.0) else n_samples
    if noverlap is None:
        noverlap = nperseg // 2

    freqs, psd = welch(
        signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        axis=axis,
    )
    return freqs, psd


def compute_band_powers(
    signal: np.ndarray,
    fs: float = 256.0,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    eps: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """Computes integrated absolute band powers and theta/beta & alpha/beta spectral ratios."""
    if bands is None:
        bands = DEFAULT_BANDS

    freqs, psd = compute_welch_psd(signal=signal, fs=fs, axis=-1)

    band_powers: Dict[str, np.ndarray] = {}
    for band_name, (low_f, high_f) in bands.items():
        # Mask valid frequency range for numerical integration
        idx_band = np.logical_and(freqs >= low_f, freqs <= high_f)
        if not np.any(idx_band):
            # Fallback for narrow frequency grids
            power_arr = np.zeros(psd.shape[:-1], dtype=np.float32)
        else:
            freq_sub = freqs[idx_band]
            psd_sub = np.take(psd, np.where(idx_band)[0], axis=-1)
            if len(freq_sub) == 1:
                power_arr = psd_sub.squeeze(axis=-1)
            else:
                power_arr = simpson(psd_sub, x=freq_sub, axis=-1)
        band_powers[band_name] = power_arr.astype(np.float32)

    # Workload and cognitive arousal spectral ratios
    theta = band_powers["theta"]
    beta = band_powers["beta"]
    alpha = band_powers["alpha"]

    band_powers["theta_beta_ratio"] = (theta / (beta + eps)).astype(np.float32)
    band_powers["alpha_beta_ratio"] = (alpha / (beta + eps)).astype(np.float32)

    return band_powers
