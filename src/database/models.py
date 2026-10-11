from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class FlightStatus(str, Enum):
    SCHEDULED = "SCHEDULED"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    DIVERTED = "DIVERTED"
    CANCELLED = "CANCELLED"


class CrewRole(str, Enum):
    PILOT_IN_COMMAND = "PILOT_IN_COMMAND"
    FIRST_OFFICER = "FIRST_OFFICER"
    RELIEF_PILOT = "RELIEF_PILOT"
    OBSERVER = "OBSERVER"


class SeatPosition(str, Enum):
    LEFT_SEAT = "LEFT_SEAT"
    RIGHT_SEAT = "RIGHT_SEAT"
    OBSERVER_SEAT = "OBSERVER_SEAT"


class DeviceType(str, Enum):
    EEG_HEADSET = "EEG_HEADSET"
    PPG_GSR_WRISTBAND = "PPG_GSR_WRISTBAND"
    EYE_TRACKER = "EYE_TRACKER"
    QAR_TELEMETRY = "QAR_TELEMETRY"
    COCKPIT_CAMERA = "COCKPIT_CAMERA"


class CognitiveStateName(str, Enum):
    BASELINE = "BASELINE"
    CHANNELIZED_ATTENTION = "CHANNELIZED_ATTENTION"
    DIVERTED_ATTENTION = "DIVERTED_ATTENTION"
    STARTLE = "STARTLE"


class SeverityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    CRITICAL = "CRITICAL"


class AlertCategory(str, Enum):
    TUNNEL_VISION = "TUNNEL_VISION"
    DISTRACTION = "DISTRACTION"
    ACUTE_STARTLE = "ACUTE_STARTLE"
    MICROSLEEP = "MICROSLEEP"
    HIGH_G_UPSET = "HIGH_G_UPSET"


class DeclutterLevel(str, Enum):
    LEVEL_0_FULL = "LEVEL_0_FULL"
    LEVEL_1_MODERATE = "LEVEL_1_MODERATE"
    LEVEL_2_ESSENTIALS = "LEVEL_2_ESSENTIALS"


class AdvisoryAudio(str, Enum):
    NONE = "NONE"
    CROSS_CHECK_CHIME = "CROSS_CHECK_CHIME"
    HEADS_UP_CHIME = "HEADS_UP_CHIME"
    ATTITUDE_RECOVERY_WARNING = "ATTITUDE_RECOVERY_WARNING"


class TaskAllocation(str, Enum):
    PILOT_FLYING = "PILOT_FLYING"
    SHARED_COCKPIT = "SHARED_COCKPIT"
    AUTOPILOT_AUTO_ASSIST = "AUTOPILOT_AUTO_ASSIST"


@dataclass
class Aircraft:
    tail_number: str
    model: str
    manufacturer: str
    max_gross_weight_kg: float
    seating_capacity: int
    certification_status: str = "AIRWORTHY"
    created_at: Optional[str] = None


@dataclass
class Pilot:
    pilot_id: str
    license_number: str
    first_name: str
    last_name: str
    total_flight_hours: float
    medical_cert_expiry: str  # YYYY-MM-DD
    created_at: Optional[str] = None


@dataclass
class Flight:
    flight_id: str
    flight_number: str
    aircraft_tail_number: str
    origin_icao: str
    destination_icao: str
    scheduled_departure: str
    scheduled_arrival: str
    actual_departure: Optional[str] = None
    actual_arrival: Optional[str] = None
    flight_status: str = "SCHEDULED"
    created_at: Optional[str] = None


@dataclass
class CrewAssignment:
    assignment_id: str
    flight_id: str
    pilot_id: str
    crew_role: str
    seat_position: str


@dataclass
class SensorDevice:
    device_id: str
    device_serial: str
    device_type: str
    hardware_model: str
    firmware_version: str
    sampling_rate_hz: float
    calibration_date: str  # YYYY-MM-DD
    is_active: int = 1


@dataclass
class SensorReading:
    reading_id: str
    flight_id: str
    device_id: str
    pilot_id: str
    timestamp: float
    channel_name: str
    raw_value: float
    unit: str
    quality_index: float = 1.0

    def to_db_tuple(self) -> Tuple[Any, ...]:
        return (
            self.reading_id,
            self.flight_id,
            self.device_id,
            self.pilot_id,
            self.timestamp,
            self.channel_name,
            self.raw_value,
            self.unit,
            self.quality_index,
        )


@dataclass
class FlightPhaseRecord:
    phase_id: str
    flight_id: str
    phase_name: str
    start_timestamp: float
    end_timestamp: Optional[float] = None
    is_emergency_simulated: int = 0


@dataclass
class EmergencyEventRecord:
    event_id: str
    flight_id: str
    phase_id: str
    event_type: str
    onset_timestamp: float
    severity_level: str
    trigger_cause: str
    resolution_timestamp: Optional[float] = None


@dataclass
class CognitiveStateSnapshotRecord:
    snapshot_id: str
    flight_id: str
    pilot_id: str
    timestamp: float
    predicted_state: str
    confidence_score: float
    cognitive_load_index: float
    raw_eeg_metric: float
    raw_hrv_metric: float
    raw_gsr_metric: float
    perclos_score: float
    flight_phase_id: Optional[str] = None
    is_alert_condition: int = 0
    created_at: Optional[str] = None

    def to_db_tuple(self) -> Tuple[Any, ...]:
        return (
            self.snapshot_id,
            self.flight_id,
            self.pilot_id,
            self.timestamp,
            self.flight_phase_id,
            self.predicted_state,
            self.confidence_score,
            self.cognitive_load_index,
            self.raw_eeg_metric,
            self.raw_hrv_metric,
            self.raw_gsr_metric,
            self.perclos_score,
            self.is_alert_condition,
        )


@dataclass
class AlertRecord:
    alert_id: str
    snapshot_id: str
    flight_id: str
    pilot_id: str
    timestamp: float
    severity_level: str
    alert_category: str
    declutter_level: str
    advisory_audio: str
    task_allocation: str
    acknowledged: int = 0
    acknowledged_at: Optional[str] = None

    def to_db_tuple(self) -> Tuple[Any, ...]:
        return (
            self.alert_id,
            self.snapshot_id,
            self.flight_id,
            self.pilot_id,
            self.timestamp,
            self.severity_level,
            self.alert_category,
            self.declutter_level,
            self.advisory_audio,
            self.task_allocation,
            self.acknowledged,
            self.acknowledged_at,
        )


@dataclass
class AlertAuditLogRecord:
    audit_id: str
    alert_id: str
    action_taken: str
    operator_notes: Optional[str] = None
    logged_timestamp: Optional[str] = None

    def to_db_tuple(self) -> Tuple[Any, ...]:
        return (
            self.audit_id,
            self.alert_id,
            self.action_taken,
            self.operator_notes,
        )
