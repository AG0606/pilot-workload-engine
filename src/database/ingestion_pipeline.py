import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.database.db_manager import DatabaseManager
from src.database.models import (
    Aircraft,
    AlertRecord,
    CognitiveStateSnapshotRecord,
    CrewAssignment,
    Flight,
    FlightPhaseRecord,
    Pilot,
    SensorDevice,
    SensorReading,
)
from src.database.nosql_store import HybridNoSQLTimeSeriesStore
from src.database.transaction_manager import TransactionManager
from src.engine.decision_engine import CockpitDecisionEngine, DeclutterLevel
from src.engine.sensor_fusion import CognitiveLoadLevel, KalmanCognitiveSensorFusion


class AvionicsDatabaseIngestionPipeline:
    """High-throughput ingestion & real-time cognitive analytics pipeline for e-Pilot.

    Integrates:
    - Relational DBMS ingestion with composite B+ Tree indexing.
    - Kalman filter multi-sensor fusion (EEG + HRV + GSR).
    - Multi-Agent Cockpit Decision Engine with display decluttering taxonomy.
    - ACID transaction coordinator for safety-critical alert logging.
    - Secondary NoSQL time-series store for polyglot persistence.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        nosql_store: Optional[HybridNoSQLTimeSeriesStore] = None,
        enable_decision_engine: bool = True,
    ) -> None:
        self.db = db_manager
        self.nosql = nosql_store or HybridNoSQLTimeSeriesStore()
        self.tx_mgr = TransactionManager(self.db)
        self.fusion = KalmanCognitiveSensorFusion()
        self.decision_engine = CockpitDecisionEngine() if enable_decision_engine else None

    def seed_master_records(self, flight_id: str = "AF-320-EXP1") -> Tuple[str, str, str]:
        """Seeds base Aircraft, Pilot, Flight, Crew, and Devices into relational DB."""
        # 1. Aircraft
        aircraft = Aircraft(
            tail_number="F-WWBA",
            model="Airbus A320-200",
            manufacturer="Airbus Industrie",
            max_gross_weight_kg=78000.0,
            seating_capacity=180,
            certification_status="AIRWORTHY",
        )
        self.db.execute_update(
            """INSERT OR IGNORE INTO aircraft 
               (tail_number, model, manufacturer, max_gross_weight_kg, seating_capacity, certification_status)
               VALUES (?, ?, ?, ?, ?, ?);""",
            (
                aircraft.tail_number,
                aircraft.model,
                aircraft.manufacturer,
                aircraft.max_gross_weight_kg,
                aircraft.seating_capacity,
                aircraft.certification_status,
            ),
        )

        # 2. Pilots (Captain & First Officer)
        pic = Pilot(
            pilot_id="PLT-001",
            license_number="ATPL-849201",
            first_name="Jean",
            last_name="Dupont",
            total_flight_hours=8450.0,
            medical_cert_expiry="2027-12-31",
        )
        fo = Pilot(
            pilot_id="PLT-002",
            license_number="CPL-992144",
            first_name="Marc",
            last_name="Laurent",
            total_flight_hours=2100.0,
            medical_cert_expiry="2028-06-30",
        )
        for p in [pic, fo]:
            self.db.execute_update(
                """INSERT OR IGNORE INTO pilot 
                   (pilot_id, license_number, first_name, last_name, total_flight_hours, medical_cert_expiry)
                   VALUES (?, ?, ?, ?, ?, ?);""",
                (
                    p.pilot_id,
                    p.license_number,
                    p.first_name,
                    p.last_name,
                    p.total_flight_hours,
                    p.medical_cert_expiry,
                ),
            )

        # 3. Flight
        flight = Flight(
            flight_id=flight_id,
            flight_number="AF1024",
            aircraft_tail_number="F-WWBA",
            origin_icao="LFPG",
            destination_icao="EGLL",
            scheduled_departure="2026-10-12 08:00:00",
            scheduled_arrival="2026-10-12 09:20:00",
            actual_departure="2026-10-12 08:05:00",
            flight_status="ACTIVE",
        )
        self.db.execute_update(
            """INSERT OR IGNORE INTO flight 
               (flight_id, flight_number, aircraft_tail_number, origin_icao, destination_icao,
                scheduled_departure, actual_departure, scheduled_arrival, flight_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);""",
            (
                flight.flight_id,
                flight.flight_number,
                flight.aircraft_tail_number,
                flight.origin_icao,
                flight.destination_icao,
                flight.scheduled_departure,
                flight.actual_departure,
                flight.scheduled_arrival,
                flight.flight_status,
            ),
        )

        # 4. Crew Assignments
        self.db.execute_update(
            """INSERT OR IGNORE INTO crew_assignment 
               (assignment_id, flight_id, pilot_id, crew_role, seat_position)
               VALUES (?, ?, ?, ?, ?);""",
            ("CRW-01", flight_id, pic.pilot_id, "PILOT_IN_COMMAND", "LEFT_SEAT"),
        )
        self.db.execute_update(
            """INSERT OR IGNORE INTO crew_assignment 
               (assignment_id, flight_id, pilot_id, crew_role, seat_position)
               VALUES (?, ?, ?, ?, ?);""",
            ("CRW-02", flight_id, fo.pilot_id, "FIRST_OFFICER", "RIGHT_SEAT"),
        )

        # 5. Sensor Devices
        devices = [
            SensorDevice("DEV-EEG-01", "SN-EEG-994", "EEG_HEADSET", "OpenBCI Cyton", "v3.1.2", 256.0, "2026-01-15"),
            SensorDevice("DEV-BIO-01", "SN-BIO-108", "PPG_GSR_WRISTBAND", "Empatica E4", "v2.0.4", 256.0, "2026-02-10"),
            SensorDevice("DEV-EYE-01", "SN-EYE-552", "EYE_TRACKER", "Tobii Pro Spark", "v1.8.0", 60.0, "2026-03-01"),
            SensorDevice("DEV-QAR-01", "SN-QAR-771", "QAR_TELEMETRY", "Teledyne QAR", "v4.5.1", 10.0, "2025-11-20"),
        ]
        for d in devices:
            self.db.execute_update(
                """INSERT OR IGNORE INTO sensor_device 
                   (device_id, device_serial, device_type, hardware_model, firmware_version, sampling_rate_hz, calibration_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?);""",
                (
                    d.device_id,
                    d.device_serial,
                    d.device_type,
                    d.hardware_model,
                    d.firmware_version,
                    d.sampling_rate_hz,
                    d.calibration_date,
                ),
            )

        # 6. Flight Phases (Normal Cruise & Simulated Emergency Upset)
        phase_cruise = FlightPhaseRecord("PH-CRUISE-01", flight_id, "CRUISE", 0.0, 1800.0, 0)
        phase_emergency = FlightPhaseRecord("PH-EMERG-01", flight_id, "EMERGENCY_UPSET", 1800.0, 2400.0, 1)
        for ph in [phase_cruise, phase_emergency]:
            self.db.execute_update(
                """INSERT OR IGNORE INTO flight_phase 
                   (phase_id, flight_id, phase_name, start_timestamp, end_timestamp, is_emergency_simulated)
                   VALUES (?, ?, ?, ?, ?, ?);""",
                (
                    ph.phase_id,
                    ph.flight_id,
                    ph.phase_name,
                    ph.start_timestamp,
                    ph.end_timestamp,
                    ph.is_emergency_simulated,
                ),
            )

        return flight_id, pic.pilot_id, fo.pilot_id

    def process_and_ingest_window(
        self,
        flight_id: str,
        pilot_id: str,
        timestamp: float,
        flight_phase_id: str,
        eeg_engagement: float,
        hrv_stress: float,
        gsr_conductance: float,
        perclos_score: float,
        predicted_state: str,
        confidence_score: float,
        raw_waveform_dict: Optional[Dict[str, np.ndarray]] = None,
    ) -> CognitiveStateSnapshotRecord:
        """Processes a single 4.0s window:
        1. Updates Kalman Filter to compute filtered Cognitive Load Index (CLI).
        2. Evaluates Cockpit Decision Engine state machine.
        3. If alert threshold crossed, executes atomic ACID transaction.
        4. Persists raw waveforms to NoSQL document store if provided.
        """
        # 1. Kalman state estimation
        cli, p_cov, load_level, raw_z = self.fusion.update(
            eeg_metric=eeg_engagement,
            hrv_metric=hrv_stress,
            gsr_metric=gsr_conductance,
        )

        is_alert = (
            load_level == CognitiveLoadLevel.CRITICAL_OVERLOAD
            or predicted_state in ("CHANNELIZED_ATTENTION", "STARTLE")
            or perclos_score >= 0.40
        )

        snapshot_id = f"SNP-{flight_id}-{int(timestamp * 100)}"
        snapshot = CognitiveStateSnapshotRecord(
            snapshot_id=snapshot_id,
            flight_id=flight_id,
            pilot_id=pilot_id,
            timestamp=timestamp,
            flight_phase_id=flight_phase_id,
            predicted_state=predicted_state,
            confidence_score=confidence_score,
            cognitive_load_index=cli,
            raw_eeg_metric=eeg_engagement,
            raw_hrv_metric=hrv_stress,
            raw_gsr_metric=gsr_conductance,
            perclos_score=perclos_score,
            is_alert_condition=1 if is_alert else 0,
        )

        # 2. Decision Engine & Alert Synthesis
        alert_record = None
        if is_alert:
            alert_id = f"ALT-{flight_id}-{int(timestamp * 100)}"
            if predicted_state == "STARTLE" or load_level == CognitiveLoadLevel.CRITICAL_OVERLOAD:
                severity = "CRITICAL"
                category = "ACUTE_STARTLE" if predicted_state == "STARTLE" else "HIGH_G_UPSET"
                declutter = "LEVEL_2_ESSENTIALS"
                audio = "ATTITUDE_RECOVERY_WARNING"
                allocation = "AUTOPILOT_AUTO_ASSIST"
            elif predicted_state == "CHANNELIZED_ATTENTION":
                severity = "MEDIUM"
                category = "TUNNEL_VISION"
                declutter = "LEVEL_1_MODERATE"
                audio = "CROSS_CHECK_CHIME"
                allocation = "SHARED_COCKPIT"
            elif perclos_score >= 0.40:
                severity = "CRITICAL"
                category = "MICROSLEEP"
                declutter = "LEVEL_1_MODERATE"
                audio = "HEADS_UP_CHIME"
                allocation = "SHARED_COCKPIT"
            else:
                severity = "MEDIUM"
                category = "DISTRACTION"
                declutter = "LEVEL_1_MODERATE"
                audio = "HEADS_UP_CHIME"
                allocation = "SHARED_COCKPIT"

            alert_record = AlertRecord(
                alert_id=alert_id,
                snapshot_id=snapshot_id,
                flight_id=flight_id,
                pilot_id=pilot_id,
                timestamp=timestamp,
                severity_level=severity,
                alert_category=category,
                declutter_level=declutter,
                advisory_audio=audio,
                task_allocation=allocation,
            )

        # 3. Persist atomically
        self.tx_mgr.save_cognitive_snapshot_transactionally(snapshot, alert_record)

        # 4. Optional NoSQL Waveform Storage
        if raw_waveform_dict and self.nosql:
            self.nosql.store_raw_waveform_document(
                flight_id=flight_id,
                pilot_id=pilot_id,
                start_time=timestamp,
                end_time=timestamp + 4.0,
                sampling_rate_hz=20.0,
                channels=raw_waveform_dict,
                metadata={"load_level": str(load_level), "predicted_state": predicted_state},
            )

        return snapshot
