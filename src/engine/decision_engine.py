from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import numpy as np


class DeclutterLevel(str, Enum):
    LEVEL_0_FULL = "LEVEL_0_FULL"
    LEVEL_1_MODERATE = "LEVEL_1_MODERATE"
    LEVEL_2_ESSENTIALS = "LEVEL_2_ESSENTIALS"


class AuditoryAlert(str, Enum):
    NONE = "NONE"
    CROSS_CHECK_CHIME = "CROSS_CHECK_CHIME"
    HEADS_UP_CHIME = "HEADS_UP_CHIME"
    ATTITUDE_RECOVERY_WARNING = "ATTITUDE_RECOVERY_WARNING"


class TaskAllocation(str, Enum):
    PILOT_FLYING = "PILOT_FLYING"
    SHARED_COCKPIT = "SHARED_COCKPIT"
    AUTOPILOT_AUTO_ASSIST = "AUTOPILOT_AUTO_ASSIST"


@dataclass
class CockpitActionDecision:
    """Structured decision emitted by the multi-agent decision engine for avionics integration."""

    cognitive_state: str
    confidence: float
    probabilities: Dict[str, float]
    declutter_level: DeclutterLevel
    auditory_alert: AuditoryAlert
    task_allocation: TaskAllocation
    active_display_elements: Dict[str, bool]
    recommendations: List[str] = field(default_factory=list)


class CockpitDecisionEngine:
    """Multi-agent cockpit decision engine automating display decluttering and task allocation."""

    CLASS_NAMES = ["Baseline", "Channelized_Attention", "Diverted_Attention", "Startle"]

    def __init__(
        self,
        ca_threshold: float = 0.40,
        da_threshold: float = 0.40,
        ss_threshold: float = 0.30,
        hysteresis_window_count: int = 2,
    ) -> None:
        self.ca_threshold = ca_threshold
        self.da_threshold = da_threshold
        self.ss_threshold = ss_threshold
        self.hysteresis_window_count = hysteresis_window_count

        self.current_state: str = "Baseline"
        self.state_consecutive_counts: Dict[str, int] = {c: 0 for c in self.CLASS_NAMES}

    def process_window(
        self,
        probabilities: np.ndarray,
        flight_telemetry: Optional[Dict[str, float]] = None,
    ) -> CockpitActionDecision:
        """Evaluates single-window probability distribution and context to trigger cockpit actions."""
        if len(probabilities) != 4:
            raise ValueError(f"Expected 4 class probabilities, got {len(probabilities)}")

        prob_dict = {name: float(probabilities[i]) for i, name in enumerate(self.CLASS_NAMES)}

        p_baseline = prob_dict["Baseline"]
        p_ca = prob_dict["Channelized_Attention"]
        p_da = prob_dict["Diverted_Attention"]
        p_ss = prob_dict["Startle"]

        # Determine raw candidate state using safety-critical priority order
        if p_ss >= self.ss_threshold:
            candidate = "Startle"
        elif p_ca >= self.ca_threshold:
            candidate = "Channelized_Attention"
        elif p_da >= self.da_threshold:
            candidate = "Diverted_Attention"
        else:
            candidate = "Baseline"

        # Update hysteresis buffer to prevent display flicker
        for state_name in self.CLASS_NAMES:
            if state_name == candidate:
                self.state_consecutive_counts[state_name] += 1
            else:
                self.state_consecutive_counts[state_name] = 0

        # State transition requires consecutive window agreement
        if self.state_consecutive_counts[candidate] >= self.hysteresis_window_count:
            self.current_state = candidate

        # Telemetry overrides (e.g. high G-load or sudden altitude descent escalates startle)
        vert_g = flight_telemetry.get("vertical_g", 1.0) if flight_telemetry else 1.0
        if abs(vert_g - 1.0) > 0.8:
            # Extreme load transition forces attitude recovery advisory
            self.current_state = "Startle"

        # Map current stabilized state to cockpit avionics actions
        declutter, audio, allocation, display_elements, recs = self._generate_cockpit_actions(
            state=self.current_state,
            prob_dict=prob_dict,
        )

        return CockpitActionDecision(
            cognitive_state=self.current_state,
            confidence=prob_dict[self.current_state],
            probabilities=prob_dict,
            declutter_level=declutter,
            auditory_alert=audio,
            task_allocation=allocation,
            active_display_elements=display_elements,
            recommendations=recs,
        )

    def _generate_cockpit_actions(
        self,
        state: str,
        prob_dict: Dict[str, float],
    ) -> tuple[DeclutterLevel, AuditoryAlert, TaskAllocation, Dict[str, bool], List[str]]:
        """Determines display decluttering level, auditory chimes, and autopilot handoffs."""
        elements = {
            "pfd_attitude": True,
            "pfd_airspeed": True,
            "pfd_altitude": True,
            "hsi_heading": True,
            "nav_flight_plan": True,
            "weather_radar": True,
            "tcas_traffic": True,
            "engine_secondary_gauges": True,
            "comm_frequencies": True,
        }
        recs: List[str] = []

        if state == "Startle":
            declutter = DeclutterLevel.LEVEL_2_ESSENTIALS
            audio = AuditoryAlert.ATTITUDE_RECOVERY_WARNING
            allocation = TaskAllocation.AUTOPILOT_AUTO_ASSIST
            # Strip display to basic six-pack attitude and airspeed
            elements["nav_flight_plan"] = False
            elements["weather_radar"] = False
            elements["tcas_traffic"] = False
            elements["engine_secondary_gauges"] = False
            elements["comm_frequencies"] = False
            recs.append("Emergency upset recovery advisory engaged.")
            recs.append("Autopilot control-assist advisory activated.")

        elif state == "Channelized_Attention":
            declutter = DeclutterLevel.LEVEL_1_MODERATE
            audio = AuditoryAlert.CROSS_CHECK_CHIME
            allocation = TaskAllocation.SHARED_COCKPIT
            elements["weather_radar"] = False
            elements["engine_secondary_gauges"] = False
            recs.append("Tunnel vision detected: Cross-check instruments prompt issued.")
            recs.append("PFD primary attitude tape highlighted.")

        elif state == "Diverted_Attention":
            declutter = DeclutterLevel.LEVEL_1_MODERATE
            audio = AuditoryAlert.HEADS_UP_CHIME
            allocation = TaskAllocation.SHARED_COCKPIT
            elements["weather_radar"] = False
            elements["tcas_traffic"] = False
            recs.append("Diverted attention detected: Secondary MFD displays decluttered.")
            recs.append("Non-critical COM audio dimmed.")

        else:  # Baseline
            declutter = DeclutterLevel.LEVEL_0_FULL
            audio = AuditoryAlert.NONE
            allocation = TaskAllocation.PILOT_FLYING
            recs.append("Cockpit operating under nominal cognitive workload.")

        return declutter, audio, allocation, elements, recs
