import unittest
import numpy as np
from src.engine.decision_engine import (
    AuditoryAlert,
    CockpitDecisionEngine,
    DeclutterLevel,
    TaskAllocation,
)


class TestDecisionEngine(unittest.TestCase):
    """Unit tests verifying cockpit display decluttering, alerts, and hysteresis transitions."""

    def setUp(self) -> None:
        self.engine = CockpitDecisionEngine(
            ca_threshold=0.40,
            da_threshold=0.40,
            ss_threshold=0.30,
            hysteresis_window_count=2,
        )

    def test_baseline_nominal(self) -> None:
        """Verifies nominal cognitive workload maintains full display symbology."""
        # Baseline = 0.85, CA = 0.05, DA = 0.05, SS = 0.05
        probs = np.array([0.85, 0.05, 0.05, 0.05])
        # Two consecutive windows to satisfy hysteresis
        _ = self.engine.process_window(probs)
        decision = self.engine.process_window(probs)

        self.assertEqual(decision.cognitive_state, "Baseline")
        self.assertEqual(decision.declutter_level, DeclutterLevel.LEVEL_0_FULL)
        self.assertEqual(decision.auditory_alert, AuditoryAlert.NONE)
        self.assertEqual(decision.task_allocation, TaskAllocation.PILOT_FLYING)
        self.assertTrue(decision.active_display_elements["weather_radar"])
        self.assertTrue(decision.active_display_elements["tcas_traffic"])

    def test_channelized_attention_transition(self) -> None:
        """Verifies channelized attention declutters secondary gauges and triggers cross-check chime."""
        probs = np.array([0.20, 0.65, 0.10, 0.05])
        # First window: candidate updated, hysteresis not yet met
        d1 = self.engine.process_window(probs)
        self.assertEqual(d1.cognitive_state, "Baseline")

        # Second window: hysteresis satisfied
        d2 = self.engine.process_window(probs)
        self.assertEqual(d2.cognitive_state, "Channelized_Attention")
        self.assertEqual(d2.declutter_level, DeclutterLevel.LEVEL_1_MODERATE)
        self.assertEqual(d2.auditory_alert, AuditoryAlert.CROSS_CHECK_CHIME)
        self.assertEqual(d2.task_allocation, TaskAllocation.SHARED_COCKPIT)
        self.assertFalse(d2.active_display_elements["weather_radar"])

    def test_startle_critical_override(self) -> None:
        """Verifies startle state triggers Level 2 emergency declutter and autopilot assist."""
        probs = np.array([0.10, 0.10, 0.10, 0.70])
        _ = self.engine.process_window(probs)
        decision = self.engine.process_window(probs)

        self.assertEqual(decision.cognitive_state, "Startle")
        self.assertEqual(decision.declutter_level, DeclutterLevel.LEVEL_2_ESSENTIALS)
        self.assertEqual(decision.auditory_alert, AuditoryAlert.ATTITUDE_RECOVERY_WARNING)
        self.assertEqual(decision.task_allocation, TaskAllocation.AUTOPILOT_AUTO_ASSIST)
        self.assertFalse(decision.active_display_elements["nav_flight_plan"])
        self.assertTrue(decision.active_display_elements["pfd_attitude"])

    def test_flight_telemetry_g_force_escalation(self) -> None:
        """Verifies extreme G-load transition triggers startle override regardless of probability."""
        nominal_probs = np.array([0.70, 0.10, 0.10, 0.10])
        # Severe G load (e.g. 2.5G pull-up maneuver)
        decision = self.engine.process_window(nominal_probs, flight_telemetry={"vertical_g": 2.5})
        self.assertEqual(decision.cognitive_state, "Startle")
        self.assertEqual(decision.declutter_level, DeclutterLevel.LEVEL_2_ESSENTIALS)


if __name__ == "__main__":
    unittest.main()
