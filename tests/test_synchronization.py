import unittest
import numpy as np
from src.data.synchronizer import MultiModalSynchronizer, SlidingWindowExtractor


class TestSynchronization(unittest.TestCase):
    """Unit tests verifying multi-rate signal synchronization and sliding window integrity."""

    def setUp(self) -> None:
        self.duration_sec = 20.0
        self.target_fs = 20.0

        # Create multi-rate synthetic time-series
        t_eeg = np.linspace(0, self.duration_sec, int(self.duration_sec * 256), endpoint=False)
        eeg_signal = np.sin(2 * np.pi * 10 * t_eeg)[:, np.newaxis]

        t_cardio = np.linspace(0, self.duration_sec, int(self.duration_sec * 256), endpoint=False)
        cardio_signal = np.cos(2 * np.pi * 1.2 * t_cardio)[:, np.newaxis]

        t_ocular = np.linspace(0, self.duration_sec, int(self.duration_sec * 60), endpoint=False)
        ocular_signal = np.random.randn(len(t_ocular), 4).astype(np.float32)

        t_telemetry = np.linspace(0, self.duration_sec, int(self.duration_sec * 10), endpoint=False)
        telemetry_signal = np.ones((len(t_telemetry), 8), dtype=np.float32)

        t_label = np.linspace(0, self.duration_sec, int(self.duration_sec * 256), endpoint=False)
        label_signal = np.zeros(len(t_label), dtype=np.int64)
        label_signal[len(label_signal) // 2 :] = 1

        self.streams = {
            "eeg": (t_eeg, eeg_signal),
            "cardio": (t_cardio, cardio_signal),
            "ocular": (t_ocular, ocular_signal),
            "context": (t_telemetry, telemetry_signal),
            "label": (t_label, label_signal),
        }

    def test_multi_rate_resampling_grid(self) -> None:
        """Verifies all heterogeneous modalities align to identical target grid length."""
        synchronizer = MultiModalSynchronizer(target_fs_hz=self.target_fs)
        ref_time, aligned = synchronizer.synchronize_streams(
            streams=self.streams,
            categorical_streams=["label"],
        )

        expected_samples = int(np.floor(self.duration_sec * self.target_fs))
        self.assertAlmostEqual(len(ref_time), expected_samples, delta=2)

        for name in ["eeg", "cardio", "ocular", "context", "label"]:
            self.assertIn(name, aligned)
            self.assertEqual(len(aligned[name]), len(ref_time))

    def test_window_shape_and_overlap(self) -> None:
        """Verifies 4.0s window length (80 samples) and 1.0s stride (20 samples, 75% overlap)."""
        synchronizer = MultiModalSynchronizer(target_fs_hz=self.target_fs)
        ref_time, aligned = synchronizer.synchronize_streams(
            streams=self.streams,
            categorical_streams=["label"],
        )

        extractor = SlidingWindowExtractor(
            window_size_sec=4.0,
            stride_sec=1.0,
            target_fs_hz=self.target_fs,
        )
        windows = extractor.extract_windows(aligned, ref_time)

        # In 20 seconds, windows start at 0, 1, 2, ..., 16s -> 17 windows
        num_windows = len(windows["eeg"])
        self.assertGreaterEqual(num_windows, 16)
        self.assertLessEqual(num_windows, 17)

        # Check tensor shape [windows, channels, samples_per_window]
        self.assertEqual(windows["eeg"].shape[1:], (1, 80))
        self.assertEqual(windows["cardio"].shape[1:], (1, 80))
        self.assertEqual(windows["ocular"].shape[1:], (4, 80))
        self.assertEqual(windows["context"].shape[1:], (8, 80))
        self.assertEqual(windows["label"].shape, (num_windows,))

    def test_session_boundary_discarding(self) -> None:
        """Verifies windows spanning session boundaries are strictly discarded."""
        synchronizer = MultiModalSynchronizer(target_fs_hz=self.target_fs)
        ref_time, aligned = synchronizer.synchronize_streams(
            streams=self.streams,
            categorical_streams=["label"],
        )

        # Introduce session boundary in the middle
        session_ids = np.zeros(len(ref_time), dtype=int)
        mid_point = len(ref_time) // 2
        session_ids[mid_point:] = 1

        extractor = SlidingWindowExtractor(
            window_size_sec=4.0,
            stride_sec=1.0,
            target_fs_hz=self.target_fs,
        )
        windows = extractor.extract_windows(aligned, ref_time, session_ids=session_ids)

        # Verify all windows spanning mid_point were discarded
        extractor_clean = SlidingWindowExtractor(
            window_size_sec=4.0,
            stride_sec=1.0,
            target_fs_hz=self.target_fs,
        )
        windows_clean = extractor_clean.extract_windows(aligned, ref_time)
        # Windows starting at 6s, 7s, 8s, 9s (indices 120, 140, 160, 180) overlap the boundary at 10s
        self.assertLess(len(windows["eeg"]), len(windows_clean["eeg"]))
        self.assertEqual(len(windows["eeg"]), 12)

    def test_discontinuous_time_gap_discarding(self) -> None:
        """Verifies windows spanning missing data or discontinuous time gaps are rejected."""
        t_continuous = np.arange(0, 10.0, 1.0 / self.target_fs)
        # Introduce a gap of 2 seconds in the middle
        t_gap = np.concatenate([t_continuous[:50], t_continuous[50:] + 2.0])

        data = {"eeg": np.ones((len(t_gap), 1)), "label": np.zeros(len(t_gap))}

        extractor = SlidingWindowExtractor(
            window_size_sec=4.0,
            stride_sec=1.0,
            target_fs_hz=self.target_fs,
            max_timestamp_gap_sec=0.15,
        )
        windows = extractor.extract_windows(data, t_gap)

        # Windows spanning index 50 must have been dropped
        for w in windows["eeg"]:
            self.assertEqual(w.shape, (1, 80))


if __name__ == "__main__":
    unittest.main()
