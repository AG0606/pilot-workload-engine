import unittest
import numpy as np

from src.engine.sensor_fusion import CognitiveLoadLevel, KalmanCognitiveSensorFusion


class TestSensorFusion(unittest.TestCase):
    """Unit tests for the Kalman Filter and Weighted Sensor Fusion Engine."""

    def setUp(self) -> None:
        self.fusion = KalmanCognitiveSensorFusion(
            process_noise_q=1e-3,
            measurement_noise_r=0.05,
            initial_state=0.20,
            initial_covariance_p=0.10,
        )

    def test_composite_measurement_bounds(self) -> None:
        """Verifies multi-sensor composite observation stays strictly within [0, 1]."""
        z_min = self.fusion.compute_composite_measurement(0.0, 0.0, 0.0)
        self.assertAlmostEqual(z_min, 0.0, places=4)

        z_max = self.fusion.compute_composite_measurement(1.0, 1.0, 1.0)
        self.assertAlmostEqual(z_max, 1.0, places=4)

        z_mid = self.fusion.compute_composite_measurement(0.5, 0.5, 0.5)
        self.assertAlmostEqual(z_mid, 0.5, places=4)

    def test_kalman_noise_filtering_convergence(self) -> None:
        """Verifies Kalman filter reduces noise variance around a constant baseline."""
        np.random.seed(42)
        true_state = 0.30
        measurements = np.clip(true_state + np.random.randn(50) * 0.15, 0.0, 1.0)

        filtered_states = []
        for m in measurements:
            cli, p_cov, _, _ = self.fusion.update(m, m, m)
            filtered_states.append(cli)

        raw_var = np.var(measurements)
        filtered_var = np.var(filtered_states[20:])  # After convergence
        self.assertLess(filtered_var, raw_var)
        self.assertLess(self.fusion.p, 0.10)  # Error covariance decreased

    def test_acute_emergency_escalation(self) -> None:
        """Verifies rapid escalation to CRITICAL_OVERLOAD under acute multi-sensor stress spikes."""
        # Initial nominal state
        cli, _, level, _ = self.fusion.update(0.1, 0.1, 0.1)
        self.assertEqual(level, CognitiveLoadLevel.NORMAL)

        # Ingest intense emergency stress inputs
        for _ in range(5):
            cli, _, level, _ = self.fusion.update(0.95, 0.90, 0.95)

        self.assertGreaterEqual(cli, 0.75)
        self.assertEqual(level, CognitiveLoadLevel.CRITICAL_OVERLOAD)

    def test_database_snapshot_serialization(self) -> None:
        """Verifies trajectory generation emits valid schema-compliant database records."""
        eeg_series = np.array([0.1, 0.2, 0.8, 0.9])
        hrv_series = np.array([0.1, 0.2, 0.7, 0.85])
        gsr_series = np.array([0.2, 0.3, 0.8, 0.9])

        records = self.fusion.filter_trajectory(
            eeg_series=eeg_series,
            hrv_series=hrv_series,
            gsr_series=gsr_series,
            flight_id="AF-320-TEST",
        )

        self.assertEqual(len(records), 4)
        rec = records[-1]
        self.assertEqual(rec.flight_id, "AF-320-TEST")
        self.assertTrue(rec.alarm_triggered)
        self.assertIn("LOAD", rec.load_level)


if __name__ == "__main__":
    unittest.main()
