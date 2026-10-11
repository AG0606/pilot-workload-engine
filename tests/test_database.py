import os
import shutil
import tempfile
import unittest
import numpy as np

from src.database.analytics import AvionicsAnalyticsEngine
from src.database.db_manager import DatabaseManager
from src.database.ingestion_pipeline import AvionicsDatabaseIngestionPipeline
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


class TestDatabaseSystems(unittest.TestCase):
    """Unit tests for e-Pilot Database Systems & IoT Storage Architecture."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_epilot_db_")
        self.db_path = os.path.join(self.temp_dir, "test_cockpit.db")
        self.nosql_path = os.path.join(self.temp_dir, "test_nosql")
        self.db = DatabaseManager(db_path=self.db_path, enable_wal=True)
        self.nosql = HybridNoSQLTimeSeriesStore(storage_dir=self.nosql_path)
        self.pipeline = AvionicsDatabaseIngestionPipeline(self.db, nosql_store=self.nosql)
        self.tx_mgr = TransactionManager(self.db)

    def tearDown(self) -> None:
        self.db.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_schema_creation_and_master_seeding(self) -> None:
        """Verifies all tables are created and master records seed without errors."""
        flight_id, pic_id, fo_id = self.pipeline.seed_master_records()

        # Check tables populated
        flight_count = self.db.execute_scalar("SELECT COUNT(*) FROM flight WHERE flight_id = ?;", (flight_id,))
        self.assertEqual(flight_count, 1)

        pilot_count = self.db.execute_scalar("SELECT COUNT(*) FROM pilot;")
        self.assertGreaterEqual(pilot_count, 2)

        device_count = self.db.execute_scalar("SELECT COUNT(*) FROM sensor_device;")
        self.assertGreaterEqual(device_count, 4)

        phase_count = self.db.execute_scalar("SELECT COUNT(*) FROM flight_phase WHERE flight_id = ?;", (flight_id,))
        self.assertGreaterEqual(phase_count, 2)

    def test_foreign_key_integrity_enforcement(self) -> None:
        """Confirms SQLite enforces foreign key constraints against orphan records."""
        # Attempting to insert a sensor reading with a non-existent flight_id must raise an IntegrityError
        invalid_reading = SensorReading(
            reading_id="RD-ORPHAN",
            flight_id="NON_EXISTENT_FLIGHT_999",
            device_id="DEV-EEG-01",
            pilot_id="PLT-001",
            timestamp=12.5,
            channel_name="eeg_fp1",
            raw_value=4.2,
            unit="uV",
        )
        with self.assertRaises(Exception):
            self.tx_mgr.ingest_sensor_batch_transactionally([invalid_reading])

    def test_batch_sensor_insertion_and_composite_index(self) -> None:
        """Verifies high-throughput batch sensor insertion and B+ tree range query."""
        flight_id, pic_id, _ = self.pipeline.seed_master_records()

        readings = [
            SensorReading(
                reading_id=f"RD-TEST-{i}",
                flight_id=flight_id,
                device_id="DEV-EEG-01",
                pilot_id=pic_id,
                timestamp=i * 0.05,
                channel_name="eeg_cz",
                raw_value=float(np.sin(i * 0.1)),
                unit="uV",
            )
            for i in range(100)
        ]
        inserted = self.tx_mgr.ingest_sensor_batch_transactionally(readings)
        self.assertEqual(inserted, 100)

        # Query range using composite index (flight_id, timestamp)
        query = "SELECT COUNT(*) FROM sensor_reading WHERE flight_id = ? AND timestamp BETWEEN ? AND ?;"
        count = self.db.execute_scalar(query, (flight_id, 1.0, 3.0))
        self.assertGreater(count, 0)

    def test_acid_alert_transaction_atomicity(self) -> None:
        """Verifies all 3 steps of alert generation commit atomically."""
        flight_id, pic_id, _ = self.pipeline.seed_master_records()

        snapshot = CognitiveStateSnapshotRecord(
            snapshot_id="SNP-ACID-01",
            flight_id=flight_id,
            pilot_id=pic_id,
            timestamp=100.0,
            predicted_state="STARTLE",
            confidence_score=0.95,
            cognitive_load_index=0.88,
            raw_eeg_metric=0.85,
            raw_hrv_metric=0.90,
            raw_gsr_metric=0.82,
            perclos_score=0.15,
            flight_phase_id="PH-EMERG-01",
            is_alert_condition=0,
        )
        self.db.execute_update(
            """INSERT INTO cognitive_state_snapshot (
                snapshot_id, flight_id, pilot_id, timestamp, flight_phase_id,
                predicted_state, confidence_score, cognitive_load_index,
                raw_eeg_metric, raw_hrv_metric, raw_gsr_metric, perclos_score, is_alert_condition
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0);""",
            snapshot.to_db_tuple()[:12],
        )

        alert = AlertRecord(
            alert_id="ALT-ACID-01",
            snapshot_id="SNP-ACID-01",
            flight_id=flight_id,
            pilot_id=pic_id,
            timestamp=100.0,
            severity_level="CRITICAL",
            alert_category="ACUTE_STARTLE",
            declutter_level="LEVEL_2_ESSENTIALS",
            advisory_audio="ATTITUDE_RECOVERY_WARNING",
            task_allocation="AUTOPILOT_AUTO_ASSIST",
        )

        success = self.tx_mgr.dispatch_alert_transactionally(alert)
        self.assertTrue(success)

        # 1. Alert row exists
        alert_exists = self.db.execute_scalar("SELECT COUNT(*) FROM alert WHERE alert_id = ?;", ("ALT-ACID-01",))
        self.assertEqual(alert_exists, 1)

        # 2. Snapshot updated to is_alert_condition = 1
        is_flagged = self.db.execute_scalar(
            "SELECT is_alert_condition FROM cognitive_state_snapshot WHERE snapshot_id = ?;",
            ("SNP-ACID-01",),
        )
        self.assertEqual(is_flagged, 1)

        # 3. Audit log contains audit entry
        audit_exists = self.db.execute_scalar(
            "SELECT COUNT(*) FROM alert_audit_log WHERE alert_id = ?;",
            ("ALT-ACID-01",),
        )
        self.assertEqual(audit_exists, 1)

    def test_acid_transaction_rollback_on_failure(self) -> None:
        """Verifies that an error in any transactional step rolls back previous writes."""
        flight_id, pic_id, _ = self.pipeline.seed_master_records()

        # Non-existent snapshot_id should violate foreign key constraint when inserting Alert
        invalid_alert = AlertRecord(
            alert_id="ALT-FAIL-01",
            snapshot_id="SNP-DOES-NOT-EXIST",
            flight_id=flight_id,
            pilot_id=pic_id,
            timestamp=50.0,
            severity_level="LOW",
            alert_category="DISTRACTION",
            declutter_level="LEVEL_1_MODERATE",
            advisory_audio="HEADS_UP_CHIME",
            task_allocation="SHARED_COCKPIT",
        )

        with self.assertRaises(Exception):
            self.tx_mgr.dispatch_alert_transactionally(invalid_alert)

        # Verify nothing was committed to alert table
        count = self.db.execute_scalar("SELECT COUNT(*) FROM alert WHERE alert_id = ?;", ("ALT-FAIL-01",))
        self.assertEqual(count, 0)

    def test_ingestion_pipeline_end_to_end(self) -> None:
        """Verifies end-to-end window processing, Kalman fusion, and alert persistence."""
        flight_id, pic_id, _ = self.pipeline.seed_master_records()

        snapshot = self.pipeline.process_and_ingest_window(
            flight_id=flight_id,
            pilot_id=pic_id,
            timestamp=25.0,
            flight_phase_id="PH-EMERG-01",
            eeg_engagement=0.92,
            hrv_stress=0.88,
            gsr_conductance=0.85,
            perclos_score=0.18,
            predicted_state="STARTLE",
            confidence_score=0.94,
            raw_waveform_dict={"eeg_f7": np.ones(80)},
        )

        self.assertIsNotNone(snapshot.snapshot_id)
        self.assertEqual(snapshot.is_alert_condition, 1)

        # Confirm persisted in DB
        persisted_cli = self.db.execute_scalar(
            "SELECT cognitive_load_index FROM cognitive_state_snapshot WHERE snapshot_id = ?;",
            (snapshot.snapshot_id,),
        )
        self.assertAlmostEqual(persisted_cli, snapshot.cognitive_load_index, places=3)

    def test_analytical_sql_queries(self) -> None:
        """Verifies analytical OLAP engine executes Section 6.5 queries."""
        flight_id, pic_id, fo_id = self.pipeline.seed_master_records()

        # Seed nominal windows
        for i in range(5):
            self.pipeline.process_and_ingest_window(
                flight_id=flight_id,
                pilot_id=pic_id,
                timestamp=i * 1.0,
                flight_phase_id="PH-CRUISE-01",
                eeg_engagement=0.20,
                hrv_stress=0.20,
                gsr_conductance=0.20,
                perclos_score=0.10,
                predicted_state="BASELINE",
                confidence_score=0.95,
            )

        # Seed emergency windows
        for i in range(5):
            self.pipeline.process_and_ingest_window(
                flight_id=flight_id,
                pilot_id=fo_id,
                timestamp=1800.0 + i * 1.0,
                flight_phase_id="PH-EMERG-01",
                eeg_engagement=0.90,
                hrv_stress=0.85,
                gsr_conductance=0.90,
                perclos_score=0.35,
                predicted_state="STARTLE",
                confidence_score=0.92,
            )

        analytics = AvionicsAnalyticsEngine(self.db)
        df_contrast = analytics.compare_normal_vs_emergency()
        self.assertGreaterEqual(len(df_contrast), 2)
        self.assertIn("mean_cognitive_load_index", df_contrast.columns)

        df_trends = analytics.get_rolling_cognitive_trends()
        self.assertGreaterEqual(len(df_trends), 10)
        self.assertIn("rolling_avg_cli_10_windows", df_trends.columns)

    def test_nosql_document_store(self) -> None:
        """Verifies NoSQL waveform document persistence and retrieval."""
        doc_id = self.nosql.store_raw_waveform_document(
            flight_id="AF-320-TEST",
            pilot_id="PLT-001",
            start_time=10.0,
            end_time=14.0,
            sampling_rate_hz=20.0,
            channels={"eeg_cz": [1.0, 2.0, 3.0], "ecg": [0.5, 0.6, 0.7]},
        )
        self.assertTrue(doc_id.startswith("DOC-AF-320-TEST"))

        retrieved = self.nosql.query_waveform_by_window("AF-320-TEST", 12.0)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["pilot_id"], "PLT-001")
        self.assertEqual(len(retrieved["channels"]["eeg_cz"]), 3)


if __name__ == "__main__":
    unittest.main()
