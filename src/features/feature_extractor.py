from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from .spectral_eeg import compute_band_powers
from .flight_dynamics import compute_flight_dynamics_features


class MultiModalFeatureExtractor:
    """Extracts standardized tabular feature vectors from synchronized multi-modal windows.

    Transforms continuous sliding window tensors into statistical, spectral, and autonomic
    features for classical machine learning models (Random Forest, XGBoost, LightGBM, MLP).
    """

    def __init__(
        self,
        fs_hz: float = 20.0,
        eeg_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        self.fs_hz = float(fs_hz)
        # For 20 Hz sampled signals, Nyquist is 10 Hz, so spectral bands are adjusted up to 10 Hz
        self.eeg_bands = eeg_bands or {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 9.9),
        }
        self.feature_names: List[str] = self._init_feature_names()

    def _init_feature_names(self) -> List[str]:
        names: List[str] = []
        # EEG statistical features across 17 channels
        names.extend(["eeg_global_mean", "eeg_global_std", "eeg_global_ptp", "eeg_global_energy"])
        # Frontal and Parietal regional power
        names.extend(["eeg_frontal_mean", "eeg_frontal_std", "eeg_parietal_mean", "eeg_parietal_std"])
        # Spectral band powers and ratios
        for band in self.eeg_bands.keys():
            names.append(f"eeg_{band}_power_mean")
            names.append(f"eeg_{band}_power_std")
        names.extend(["eeg_tbr", "eeg_abr", "eeg_engagement_index"])

        # Cardio & autonomic features (ECG, Respiration, GSR)
        names.extend([
            "ecg_mean", "ecg_std", "ecg_ptp", "ecg_rms", "ecg_estimated_hr",
            "resp_mean", "resp_std", "resp_ptp", "resp_rate_proxy",
            "gsr_mean", "gsr_std", "gsr_slope", "gsr_max", "gsr_min",
        ])

        # Ocular features (Pupil, Blinks, Fixation)
        names.extend([
            "pupil_diam_mean", "pupil_diam_std", "pupil_asymmetry",
            "blink_ratio", "fixation_duration_mean", "fixation_duration_std",
        ])

        # Context & Flight Kinematics
        names.extend([
            "pitch_mean", "pitch_std",
            "roll_mean", "roll_std",
            "yaw_rate_mean", "yaw_rate_std",
            "vertical_g_mean", "vertical_g_std", "vertical_g_max", "vertical_g_min",
            "turbulence_rms", "vertical_jerk_max", "vertical_jerk_rms",
            "lateral_g_mean", "lateral_g_std",
            "airspeed_mean", "airspeed_delta",
            "altitude_mean", "altitude_delta",
            "autopilot_active_ratio",
        ])
        return names

    def extract_single_window(
        self,
        eeg: np.ndarray,      # [17, T]
        cardio: np.ndarray,   # [3, T]
        ocular: np.ndarray,   # [4, T]
        context: np.ndarray,  # [8, T]
    ) -> np.ndarray:
        """Extracts a 1D vector of engineered features from a single window."""
        feats: List[float] = []

        # ---------------- EEG Features ----------------
        feats.append(float(np.mean(eeg)))
        feats.append(float(np.std(eeg)))
        feats.append(float(np.ptp(eeg)))
        feats.append(float(np.mean(eeg**2)))

        # Frontal channels: 0 (fp1), 9 (fp2), 10 (fz)
        frontal_idx = [0, min(9, eeg.shape[0]-1), min(10, eeg.shape[0]-1)]
        frontal_eeg = eeg[frontal_idx, :]
        feats.append(float(np.mean(frontal_eeg)))
        feats.append(float(np.std(frontal_eeg)))

        # Parietal channels: 14 (p3), 15 (pz), 16 (p4)
        parietal_idx = [min(i, eeg.shape[0]-1) for i in [14, 15, 16]]
        parietal_eeg = eeg[parietal_idx, :]
        feats.append(float(np.mean(parietal_eeg)))
        feats.append(float(np.std(parietal_eeg)))

        # Spectral band powers (via Welch PSD)
        band_powers = compute_band_powers(signal=eeg, fs=self.fs_hz, bands=self.eeg_bands)
        theta_power = band_powers.get("theta", np.zeros(eeg.shape[0]))
        alpha_power = band_powers.get("alpha", np.zeros(eeg.shape[0]))
        delta_power = band_powers.get("delta", np.zeros(eeg.shape[0]))

        for band in self.eeg_bands.keys():
            bp = band_powers.get(band, np.zeros(eeg.shape[0]))
            feats.append(float(np.mean(bp)))
            feats.append(float(np.std(bp)))

        # Ratios
        m_theta = float(np.mean(theta_power))
        m_alpha = float(np.mean(alpha_power))
        m_delta = float(np.mean(delta_power))

        tbr = m_theta / (m_alpha + 1e-6)
        abr = m_alpha / (m_theta + 1e-6)
        engagement = m_alpha / (m_theta + m_delta + 1e-6)

        feats.extend([tbr, abr, engagement])

        # ---------------- Cardio Features ----------------
        ecg = cardio[0, :]
        resp = cardio[1, :]
        gsr = cardio[2, :]

        feats.append(float(np.mean(ecg)))
        feats.append(float(np.std(ecg)))
        feats.append(float(np.ptp(ecg)))
        feats.append(float(np.sqrt(np.mean(ecg**2))))
        # Peak count proxy for HR
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(ecg, distance=max(int(self.fs_hz * 0.4), 1))
        duration_sec = eeg.shape[-1] / self.fs_hz
        est_hr = (len(peaks) / max(duration_sec, 0.1)) * 60.0
        feats.append(float(est_hr))

        feats.append(float(np.mean(resp)))
        feats.append(float(np.std(resp)))
        feats.append(float(np.ptp(resp)))
        resp_peaks, _ = find_peaks(resp, distance=max(int(self.fs_hz * 1.5), 1))
        feats.append(float((len(resp_peaks) / max(duration_sec, 0.1)) * 60.0))

        feats.append(float(np.mean(gsr)))
        feats.append(float(np.std(gsr)))
        slope = (gsr[-1] - gsr[0]) / max(duration_sec, 0.1) if len(gsr) > 1 else 0.0
        feats.append(float(slope))
        feats.append(float(np.max(gsr)))
        feats.append(float(np.min(gsr)))

        # ---------------- Ocular Features ----------------
        p_left = ocular[0, :]
        p_right = ocular[1, :]
        blinks = ocular[2, :]
        fix_dur = ocular[3, :]

        pupil_mean = float(np.mean((p_left + p_right) / 2.0))
        pupil_std = float(np.std((p_left + p_right) / 2.0))
        pupil_asym = float(np.mean(np.abs(p_left - p_right)))
        blink_ratio = float(np.mean(blinks > 0.5))
        fix_mean = float(np.mean(fix_dur))
        fix_std = float(np.std(fix_dur))

        feats.extend([pupil_mean, pupil_std, pupil_asym, blink_ratio, fix_mean, fix_std])

        # ---------------- Context / Kinematics Features ----------------
        pitch = context[0, :]
        roll = context[1, :]
        yaw_rate = context[2, :]
        vertical_g = context[3, :]
        lateral_g = context[4, :]
        airspeed = context[5, :]
        altitude = context[6, :]
        ap_mode = context[7, :]

        feats.append(float(np.mean(pitch)))
        feats.append(float(np.std(pitch)))
        feats.append(float(np.mean(roll)))
        feats.append(float(np.std(roll)))
        feats.append(float(np.mean(yaw_rate)))
        feats.append(float(np.std(yaw_rate)))

        feats.append(float(np.mean(vertical_g)))
        feats.append(float(np.std(vertical_g)))
        feats.append(float(np.max(vertical_g)))
        feats.append(float(np.min(vertical_g)))

        turb_rms = float(np.sqrt(np.mean((vertical_g - 1.0) ** 2)))
        dt = 1.0 / self.fs_hz
        jerk = np.diff(vertical_g) / dt if len(vertical_g) > 1 else np.zeros(1, dtype=np.float32)
        feats.append(turb_rms)
        feats.append(float(np.max(np.abs(jerk))))
        feats.append(float(np.sqrt(np.mean(jerk**2))))

        feats.append(float(np.mean(lateral_g)))
        feats.append(float(np.std(lateral_g)))

        feats.append(float(np.mean(airspeed)))
        feats.append(float(airspeed[-1] - airspeed[0]))
        feats.append(float(np.mean(altitude)))
        feats.append(float(altitude[-1] - altitude[0]))
        feats.append(float(np.mean(ap_mode > 0.5)))

        arr = np.nan_to_num(np.array(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def extract_batch(
        self,
        eeg: Union[np.ndarray, torch.Tensor],      # [B, 17, T]
        cardio: Union[np.ndarray, torch.Tensor],   # [B, 3, T]
        ocular: Union[np.ndarray, torch.Tensor],   # [B, 4, T]
        context: Union[np.ndarray, torch.Tensor],  # [B, 8, T]
    ) -> np.ndarray:
        """Extracts tabular 2D array [B, num_features] across all sliding windows."""
        if isinstance(eeg, torch.Tensor):
            eeg = eeg.detach().cpu().numpy()
        if isinstance(cardio, torch.Tensor):
            cardio = cardio.detach().cpu().numpy()
        if isinstance(ocular, torch.Tensor):
            ocular = ocular.detach().cpu().numpy()
        if isinstance(context, torch.Tensor):
            context = context.detach().cpu().numpy()

        n_windows = eeg.shape[0]
        feature_matrix = np.zeros((n_windows, len(self.feature_names)), dtype=np.float32)

        for i in range(n_windows):
            feature_matrix[i] = self.extract_single_window(
                eeg=eeg[i],
                cardio=cardio[i],
                ocular=ocular[i],
                context=context[i],
            )

        return feature_matrix
