import time
from typing import Any, Dict, List, Optional, Tuple
from src.database.db_manager import DatabaseManager
from src.database.models import (
    AlertAuditLogRecord,
    AlertRecord,
    CognitiveStateSnapshotRecord,
    SensorReading,
)


class TransactionManager:
    """Coordinates ACID-compliant transactional operations for flight-deck safety.

    Directly addresses BCSE302L / BCSE302P Module 5 (Transaction Management)
    and Section 6.4 of e-Pilot DBMS architecture:
    - Atomicity: The Alert write, CognitiveStateSnapshot flag update, and AuditLog entry commit as one atomic unit.
    - Consistency: Preserves referential integrity and CHECK constraints across all three entities.
    - Isolation: Serialized / IMMEDIATE transaction locking prevents dirty reads or phantom writes during stress regimes.
    - Durability: Write-Ahead Logging (WAL) ensures committed records persist across crashes.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager

    def dispatch_alert_transactionally(
        self,
        alert: AlertRecord,
        audit_action: str = "AUTOMATED_PFD_DECLUTTER_AND_CHIME_TRIGGERED",
        audit_notes: Optional[str] = None,
    ) -> bool:
        """Executes atomic 3-step ACID transaction for safety-critical alert dispatch:
        1. Insert Alert record.
        2. Update CognitiveStateSnapshot is_alert_condition = 1.
        3. Insert immutable AlertAuditLog entry.
        """
        insert_alert_sql = """
            INSERT INTO alert (
                alert_id, snapshot_id, flight_id, pilot_id, timestamp,
                severity_level, alert_category, declutter_level, advisory_audio,
                task_allocation, acknowledged, acknowledged_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """

        update_snapshot_sql = """
            UPDATE cognitive_state_snapshot
            SET is_alert_condition = 1
            WHERE snapshot_id = ?;
        """

        insert_audit_sql = """
            INSERT INTO alert_audit_log (
                audit_id, alert_id, action_taken, operator_notes
            ) VALUES (?, ?, ?, ?);
        """

        audit_id = f"AUD-{alert.alert_id}"
        notes = audit_notes or f"Auto-generated alert for state {alert.alert_category} at {alert.timestamp:.2f}s"

        with self.db.transaction() as cur:
            # Step 1: Insert Alert
            cur.execute(insert_alert_sql, alert.to_db_tuple())

            # Step 2: Flag Snapshot as alert condition
            cur.execute(update_snapshot_sql, (alert.snapshot_id,))

            # Step 3: Append to Audit Log
            cur.execute(insert_audit_sql, (audit_id, alert.alert_id, audit_action, notes))

        return True

    def acknowledge_alert_transactionally(
        self,
        alert_id: str,
        acknowledged_at: str,
        pilot_id: str,
        action_note: str = "MANUAL_PILOT_ACKNOWLEDGEMENT",
    ) -> bool:
        """Atomically acknowledges a cockpit alert and records operator action."""
        update_alert_sql = """
            UPDATE alert
            SET acknowledged = 1, acknowledged_at = ?
            WHERE alert_id = ?;
        """

        insert_audit_sql = """
            INSERT INTO alert_audit_log (
                audit_id, alert_id, action_taken, operator_notes
            ) VALUES (?, ?, ?, ?);
        """

        audit_id = f"ACK-{alert_id}-{int(time.time() * 1000)}"
        notes = f"Acknowledged by pilot {pilot_id} at {acknowledged_at}"

        with self.db.transaction() as cur:
            cur.execute(update_alert_sql, (acknowledged_at, alert_id))
            cur.execute(insert_audit_sql, (audit_id, alert_id, action_note, notes))

        return True

    def ingest_sensor_batch_transactionally(
        self,
        readings: List[SensorReading],
    ) -> int:
        """Atomically writes an entire high-frequency sensor reading batch."""
        if not readings:
            return 0

        insert_sql = """
            INSERT INTO sensor_reading (
                reading_id, flight_id, device_id, pilot_id, timestamp,
                channel_name, raw_value, unit, quality_index
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        tuples = [r.to_db_tuple() for r in readings]
        with self.db.transaction() as cur:
            cur.executemany(insert_sql, tuples)
            return len(tuples)

    def save_cognitive_snapshot_transactionally(
        self,
        snapshot: CognitiveStateSnapshotRecord,
        associated_alert: Optional[AlertRecord] = None,
    ) -> None:
        """Atomically persists a cognitive snapshot and optional alert."""
        insert_snapshot_sql = """
            INSERT INTO cognitive_state_snapshot (
                snapshot_id, flight_id, pilot_id, timestamp, flight_phase_id,
                predicted_state, confidence_score, cognitive_load_index,
                raw_eeg_metric, raw_hrv_metric, raw_gsr_metric, perclos_score,
                is_alert_condition
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """

        with self.db.transaction() as cur:
            cur.execute(insert_snapshot_sql, snapshot.to_db_tuple())
            if associated_alert is not None:
                # If alert provided, execute alert write + audit entry
                insert_alert_sql = """
                    INSERT INTO alert (
                        alert_id, snapshot_id, flight_id, pilot_id, timestamp,
                        severity_level, alert_category, declutter_level, advisory_audio,
                        task_allocation, acknowledged, acknowledged_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """
                cur.execute(insert_alert_sql, associated_alert.to_db_tuple())

                audit_id = f"AUD-{associated_alert.alert_id}"
                insert_audit_sql = """
                    INSERT INTO alert_audit_log (
                        audit_id, alert_id, action_taken, operator_notes
                    ) VALUES (?, ?, ?, ?);
                """
                cur.execute(
                    insert_audit_sql,
                    (
                        audit_id,
                        associated_alert.alert_id,
                        "AUTOMATED_PFD_DECLUTTER_AND_CHIME_TRIGGERED",
                        f"Alert trigger for {associated_alert.alert_category}",
                    ),
                )
