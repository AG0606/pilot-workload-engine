import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.datasets.workload_dataset import PilotWorkloadDataset, create_workload_dataloader
from src.features.flight_dynamics import compute_flight_dynamics_features
from src.features.hrv_features import compute_hrv_features
from src.features.spectral_eeg import compute_band_powers
from src.utils.validation import validate_dataset_batch, validate_sliding_windows


class TestDataLoaderAndFeatures(unittest.TestCase):
    """Unit tests for PyTorch dataset scaffolding, feature extraction, and batch validation."""

    def setUp(self) -> None:
        self.num_windows = 32
        self.samples_per_window = 80

        # Synthetic window dictionary
        self.windows = {
            "eeg": np.random.randn(self.num_windows, 17, self.samples_per_window).astype(np.float32),
            "cardio": np.random.randn(self.num_windows, 3, self.samples_per_window).astype(np.float32),
            "ocular": np.random.randn(self.num_windows, 4, self.samples_per_window).astype(np.float32),
            "context": np.random.randn(self.num_windows, 8, self.samples_per_window).astype(np.float32),
            "label": np.random.randint(0, 4, size=(self.num_windows,), dtype=np.int64),
        }

    def test_window_validation(self) -> None:
        """Verifies sliding window structure validator."""
        valid = validate_sliding_windows(
            self.windows,
            expected_samples=80,
            expected_modalities={"eeg": 17, "cardio": 3, "ocular": 4, "context": 8},
        )
        self.assertTrue(valid)

    def test_dataset_and_dataloader(self) -> None:
        """Verifies PyTorch Dataset and DataLoader tensor batch generation."""
        dataset = PilotWorkloadDataset.from_window_dict(self.windows)
        self.assertEqual(len(dataset), self.num_windows)

        batch_size = 8
        loader = create_workload_dataloader(dataset, batch_size=batch_size, shuffle=True)

        batch_count = 0
        for batch in loader:
            batch_count += 1
            self.assertTrue(validate_dataset_batch(batch, num_classes=4))
            self.assertEqual(batch["eeg"].shape, (batch_size, 17, 80))
            self.assertEqual(batch["cardio"].shape, (batch_size, 3, 80))
            self.assertEqual(batch["ocular"].shape, (batch_size, 4, 80))
            self.assertEqual(batch["context"].shape, (batch_size, 8, 80))
            self.assertEqual(batch["label"].shape, (batch_size,))
            self.assertEqual(batch["eeg"].dtype, torch.float32)
            self.assertEqual(batch["label"].dtype, torch.int64)

        self.assertEqual(batch_count, self.num_windows // batch_size)

    def test_dataset_serialization(self) -> None:
        """Verifies torch serialization and deserialization of pre-extracted datasets."""
        dataset = PilotWorkloadDataset.from_window_dict(self.windows)
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_dataset.pt"
            dataset.save(save_path)
            self.assertTrue(save_path.exists())

            loaded = PilotWorkloadDataset.load(save_path)
            self.assertEqual(len(loaded), len(dataset))
            self.assertTrue(torch.allclose(dataset.eeg, loaded.eeg))
            self.assertTrue(torch.equal(dataset.labels, loaded.labels))

    def test_spectral_eeg_features(self) -> None:
        """Verifies Welch PSD band extraction and cognitive workload spectral ratios."""
        fs = 256.0
        duration = 4.0
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        # Construct synthetic alpha (10 Hz) and beta (20 Hz) signals
        sig = (np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t))[np.newaxis, :]

        band_powers = compute_band_powers(sig, fs=fs)
        for band in ["delta", "theta", "alpha", "beta", "gamma", "theta_beta_ratio", "alpha_beta_ratio"]:
            self.assertIn(band, band_powers)
            self.assertFalse(np.isnan(band_powers[band]).any())
            self.assertGreaterEqual(band_powers[band].item(), 0.0)

    def test_hrv_features(self) -> None:
        """Verifies time-domain and frequency-domain HRV metric calculation."""
        fs = 256.0
        # Synthetic ECG with regular R-peaks every 0.8s (75 BPM)
        ecg = np.zeros(int(fs * 10.0), dtype=np.float64)
        peak_indices = np.arange(int(0.5 * fs), len(ecg), int(0.8 * fs))
        for p in peak_indices:
            ecg[p : min(p + 3, len(ecg))] = 5.0

        hrv = compute_hrv_features(ecg, fs=fs, is_rr_ms=False)
        for key in ["mean_hr_bpm", "sdnn_ms", "rmssd_ms", "lf_power", "hf_power", "lf_hf_ratio"]:
            self.assertIn(key, hrv)
            self.assertFalse(np.isnan(hrv[key]))

    def test_flight_dynamics_features(self) -> None:
        """Verifies kinematic delta and turbulence calculation."""
        # 8 channels x 80 samples
        context_win = np.zeros((8, 80), dtype=np.float32)
        # Vertical G channel (index 3) with oscillation around 1.0G
        context_win[3, :] = 1.0 + 0.2 * np.sin(np.linspace(0, 4 * np.pi, 80))
        # Pitch channel (index 0) with linear pitch ramp
        context_win[0, :] = np.linspace(2.0, 6.0, 80)

        dyn_features = compute_flight_dynamics_features(context_win, fs_hz=20.0)
        self.assertIn("turbulence_intensity_rms", dyn_features)
        self.assertIn("max_vertical_jerk", dyn_features)
        self.assertIn("pitch_delta", dyn_features)
        self.assertAlmostEqual(dyn_features["pitch_delta"], 4.0, places=2)
        self.assertGreater(dyn_features["turbulence_intensity_rms"], 0.0)


if __name__ == "__main__":
    unittest.main()
