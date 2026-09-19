import unittest
import numpy as np

from src.features.flight_phase_segmenter import FlightPhase, FlightPhaseSegmenter, FlightTelemetrySample


class TestFlightPhaseSegmenter(unittest.TestCase):
    """Unit tests for the FlightPhaseSegmenter rule-based and clustering state engine."""

    def setUp(self) -> None:
        self.segmenter = FlightPhaseSegmenter(gmm_components=3, random_state=42)

    def test_taxi_ground_detection(self) -> None:
        sample = FlightTelemetrySample(
            altitude_ft=50.0,
            vertical_speed_fpm=0.0,
            indicated_airspeed_kts=25.0,
            pitch_deg=0.0,
            roll_deg=0.0,
            vertical_g=1.0,
        )
        phase, is_emerg = self.segmenter.classify_rule_based(sample)
        self.assertEqual(phase, FlightPhase.TAXI_GROUND)
        self.assertFalse(is_emerg)

    def test_climb_phase_detection(self) -> None:
        sample = FlightTelemetrySample(
            altitude_ft=5000.0,
            vertical_speed_fpm=1800.0,
            indicated_airspeed_kts=250.0,
            pitch_deg=8.0,
            roll_deg=0.0,
            vertical_g=1.05,
        )
        phase, is_emerg = self.segmenter.classify_rule_based(sample)
        self.assertEqual(phase, FlightPhase.CLIMB)
        self.assertFalse(is_emerg)

    def test_cruise_phase_detection(self) -> None:
        sample = FlightTelemetrySample(
            altitude_ft=35000.0,
            vertical_speed_fpm=50.0,
            indicated_airspeed_kts=460.0,
            pitch_deg=2.0,
            roll_deg=0.0,
            vertical_g=1.0,
        )
        phase, is_emerg = self.segmenter.classify_rule_based(sample)
        self.assertEqual(phase, FlightPhase.CRUISE)
        self.assertFalse(is_emerg)

    def test_descent_phase_detection(self) -> None:
        sample = FlightTelemetrySample(
            altitude_ft=12000.0,
            vertical_speed_fpm=-1500.0,
            indicated_airspeed_kts=280.0,
            pitch_deg=-3.0,
            roll_deg=0.0,
            vertical_g=0.98,
        )
        phase, is_emerg = self.segmenter.classify_rule_based(sample)
        self.assertEqual(phase, FlightPhase.DESCENT)
        self.assertFalse(is_emerg)

    def test_emergency_upset_overrides(self) -> None:
        # Extreme steep bank angle > 45 deg
        sample_bank = FlightTelemetrySample(
            altitude_ft=20000.0,
            vertical_speed_fpm=0.0,
            indicated_airspeed_kts=300.0,
            pitch_deg=0.0,
            roll_deg=58.0,
            vertical_g=1.8,
        )
        phase, is_emerg = self.segmenter.classify_rule_based(sample_bank)
        self.assertEqual(phase, FlightPhase.EMERGENCY_UPSET)
        self.assertTrue(is_emerg)

        # Extreme G-pull > 2.5G
        sample_g = FlightTelemetrySample(
            altitude_ft=15000.0,
            vertical_speed_fpm=2000.0,
            indicated_airspeed_kts=350.0,
            pitch_deg=10.0,
            roll_deg=5.0,
            vertical_g=3.2,
        )
        phase, is_emerg = self.segmenter.classify_rule_based(sample_g)
        self.assertEqual(phase, FlightPhase.EMERGENCY_UPSET)
        self.assertTrue(is_emerg)

    def test_gmm_clustering(self) -> None:
        np.random.seed(42)
        kinematics = np.random.randn(100, 6)
        clusters = self.segmenter.predict_cluster(kinematics)
        self.assertEqual(len(clusters), 100)
        self.assertTrue(np.all(clusters >= 0) and np.all(clusters < 3))


if __name__ == "__main__":
    unittest.main()
