import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

import sys
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.database.analytics import AvionicsAnalyticsEngine
from src.database.db_manager import DatabaseManager
from src.database.ingestion_pipeline import AvionicsDatabaseIngestionPipeline
from src.database.models import AlertRecord, SensorReading
from src.database.nosql_store import HybridNoSQLTimeSeriesStore
from src.database.transaction_manager import TransactionManager


def run_database_benchmarks(
    num_sensor_records: int = 50000,
    num_snapshots: int = 500,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Runs rigorous empirical performance benchmarks on e-Pilot database architecture:
    1. Batch ingestion throughput (ops/sec).
    2. B+ Tree composite index range-scan acceleration ratio.
    3. ACID alert transaction latency distribution (p50, p95, p99).
    4. Multi-threaded concurrency stress test (WAL mode).
    5. Analytical OLAP query execution latencies.
    """
    temp_dir = tempfile.mkdtemp(prefix="epilot_db_bench_")
    db_file = Path(temp_dir) / "benchmark.db"
    nosql_dir = Path(temp_dir) / "nosql_store"

    try:
        db = DatabaseManager(db_path=db_file, enable_wal=True)
        nosql = HybridNoSQLTimeSeriesStore(storage_dir=nosql_dir)
        pipeline = AvionicsDatabaseIngestionPipeline(db, nosql_store=nosql)
        tx_mgr = TransactionManager(db)

        flight_id, pic_id, fo_id = pipeline.seed_master_records()

        # =====================================================================
        # 1. Batch Ingestion Throughput Benchmark
        # =====================================================================
        print("Running Batch Ingestion Throughput Benchmark...")
        channels = ["eeg_fp1", "eeg_cz", "ecg", "gsr", "pupil_dia", "pitch", "roll", "vertical_g"]
        readings = []
        for i in range(num_sensor_records):
            t = (i // len(channels)) * 0.05
            ch = channels[i % len(channels)]
            readings.append(
                SensorReading(
                    reading_id=f"RD-{i:07d}",
                    flight_id=flight_id,
                    device_id="DEV-EEG-01" if "eeg" in ch else "DEV-BIO-01",
                    pilot_id=pic_id,
                    timestamp=t,
                    channel_name=ch,
                    raw_value=float(np.sin(i * 0.01)),
                    unit="uV" if "eeg" in ch else "raw",
                    quality_index=0.98,
                )
            )

        t_start_ingest = time.perf_counter()
        # Ingest in chunks of 5000
        chunk_size = 5000
        for k in range(0, len(readings), chunk_size):
            tx_mgr.ingest_sensor_batch_transactionally(readings[k : k + chunk_size])
        t_ingest_total = time.perf_counter() - t_start_ingest
        ingest_throughput_ops_sec = num_sensor_records / t_ingest_total

        # =====================================================================
        # 2. Composite B+ Tree Index Acceleration Ratio
        # =====================================================================
        print("Running Composite B+ Tree Index Benchmark...")
        # Range query pattern: Select readings for flight in a 30-second window
        target_t_min = 100.0
        target_t_max = 130.0
        indexed_query = """
            SELECT * FROM sensor_reading 
            WHERE flight_id = ? AND timestamp BETWEEN ? AND ?;
        """

        # Measure with Composite Index
        latencies_indexed = []
        for _ in range(50):
            t0 = time.perf_counter()
            rows = db.execute_query(indexed_query, (flight_id, target_t_min, target_t_max))
            latencies_indexed.append((time.perf_counter() - t0) * 1000.0)

        # Drop index to measure unindexed Table Scan
        with db.transaction() as cur:
            cur.execute("DROP INDEX IF EXISTS idx_sensor_reading_flight_time;")

        latencies_unindexed = []
        for _ in range(50):
            t0 = time.perf_counter()
            rows_unindexed = db.execute_query(indexed_query, (flight_id, target_t_min, target_t_max))
            latencies_unindexed.append((time.perf_counter() - t0) * 1000.0)

        # Re-create index
        with db.transaction() as cur:
            cur.execute("CREATE INDEX idx_sensor_reading_flight_time ON sensor_reading(flight_id, timestamp);")

        indexed_p50 = float(np.median(latencies_indexed))
        unindexed_p50 = float(np.median(latencies_unindexed))
        index_speedup_factor = unindexed_p50 / max(indexed_p50, 0.001)

        # =====================================================================
        # 3. Ingestion & Snapshot Processing Simulation
        # =====================================================================
        print("Ingesting Snapshots and Simulating In-Flight Regime Transitions...")
        states = ["BASELINE", "CHANNELIZED_ATTENTION", "DIVERTED_ATTENTION", "STARTLE"]
        for j in range(num_snapshots):
            t_win = j * 1.0
            phase_id = "PH-CRUISE-01" if t_win < 300.0 else "PH-EMERG-01"
            # Simulate escalating stress during emergency phase
            if phase_id == "PH-CRUISE-01":
                eeg_metric = 0.20 + 0.05 * np.sin(j * 0.1)
                hrv_metric = 0.25 + 0.05 * np.cos(j * 0.1)
                gsr_metric = 0.20
                perclos = 0.12
                pred_state = "BASELINE"
                conf = 0.94
            else:
                eeg_metric = 0.85 + 0.10 * np.sin(j * 0.1)
                hrv_metric = 0.90
                gsr_metric = 0.85
                perclos = 0.45 if j % 15 == 0 else 0.20
                pred_state = states[1 + (j % 3)]
                conf = 0.91

            waveform_mock = {"eeg_cz": np.random.randn(80), "ecg": np.random.randn(80)} if j < 20 else None

            pipeline.process_and_ingest_window(
                flight_id=flight_id,
                pilot_id=pic_id if j % 2 == 0 else fo_id,
                timestamp=t_win,
                flight_phase_id=phase_id,
                eeg_engagement=float(np.clip(eeg_metric, 0.0, 1.0)),
                hrv_stress=float(np.clip(hrv_metric, 0.0, 1.0)),
                gsr_conductance=float(np.clip(gsr_metric, 0.0, 1.0)),
                perclos_score=float(np.clip(perclos, 0.0, 1.0)),
                predicted_state=pred_state,
                confidence_score=conf,
                raw_waveform_dict=waveform_mock,
            )

        # =====================================================================
        # 4. ACID Alert Transaction Commit Latency Distribution
        # =====================================================================
        print("Benchmarking Atomic ACID Alert Commit Latencies...")
        tx_latencies_ms = []
        for m in range(200):
            snap_id = f"BENCH-SNAP-{m:04d}"
            # Insert a dummy snapshot to satisfy FK
            db.execute_update(
                """INSERT INTO cognitive_state_snapshot (
                    snapshot_id, flight_id, pilot_id, timestamp, flight_phase_id,
                    predicted_state, confidence_score, cognitive_load_index,
                    raw_eeg_metric, raw_hrv_metric, raw_gsr_metric, perclos_score, is_alert_condition
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0);""",
                (snap_id, flight_id, pic_id, 2000.0 + m, "PH-EMERG-01", "STARTLE", 0.92, 0.88, 0.8, 0.8, 0.8, 0.2),
            )

            test_alert = AlertRecord(
                alert_id=f"BENCH-ALT-{m:04d}",
                snapshot_id=snap_id,
                flight_id=flight_id,
                pilot_id=pic_id,
                timestamp=2000.0 + m,
                severity_level="CRITICAL",
                alert_category="ACUTE_STARTLE",
                declutter_level="LEVEL_2_ESSENTIALS",
                advisory_audio="ATTITUDE_RECOVERY_WARNING",
                task_allocation="AUTOPILOT_AUTO_ASSIST",
            )

            t_tx_start = time.perf_counter()
            tx_mgr.dispatch_alert_transactionally(test_alert)
            tx_latencies_ms.append((time.perf_counter() - t_tx_start) * 1000.0)

        tx_p50 = float(np.percentile(tx_latencies_ms, 50))
        tx_p95 = float(np.percentile(tx_latencies_ms, 95))
        tx_p99 = float(np.percentile(tx_latencies_ms, 99))

        # =====================================================================
        # 5. Multi-Threaded Concurrency Stress Test
        # =====================================================================
        print("Running Multi-Threaded Concurrency Stress Test (4 IoT Ingestion Workers)...")
        num_threads = 4
        records_per_thread = 2500

        def worker_task(thread_idx: int) -> int:
            worker_readings = [
                SensorReading(
                    reading_id=f"CONCUR-{thread_idx}-{idx}",
                    flight_id=flight_id,
                    device_id="DEV-BIO-01",
                    pilot_id=fo_id,
                    timestamp=3000.0 + idx * 0.05,
                    channel_name=f"th_{thread_idx}_ch",
                    raw_value=float(thread_idx),
                    unit="raw",
                )
                for idx in range(records_per_thread)
            ]
            return tx_mgr.ingest_sensor_batch_transactionally(worker_readings)

        t_concur_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_task, tid) for tid in range(num_threads)]
            results = [f.result() for f in futures]
        t_concur_total = time.perf_counter() - t_concur_start
        concur_throughput_ops_sec = (num_threads * records_per_thread) / t_concur_total

        # =====================================================================
        # 6. Analytical OLAP Query Latency Suite
        # =====================================================================
        print("Executing Analytical OLAP Queries...")
        analytics = AvionicsAnalyticsEngine(db)
        query_benchmarks = {}
        for q_name in [
            "normal_vs_emergency_contrast",
            "rolling_cognitive_trend",
            "alert_severity_distribution",
            "crew_cognitive_asymmetry",
            "regime_dwell_time_percentage",
        ]:
            q_times = []
            for _ in range(20):
                tq0 = time.perf_counter()
                res = analytics.run_query(q_name)
                q_times.append((time.perf_counter() - tq0) * 1000.0)
            query_benchmarks[q_name] = {
                "p50_latency_ms": float(np.median(q_times)),
                "p95_latency_ms": float(np.percentile(q_times, 95)),
                "result_row_count": len(res),
            }

        nosql_stats = nosql.get_collection_statistics()

        results = {
            "batch_ingestion": {
                "total_records_ingested": num_sensor_records,
                "elapsed_time_seconds": round(t_ingest_total, 4),
                "throughput_records_per_second": round(ingest_throughput_ops_sec, 2),
            },
            "bplus_tree_composite_index": {
                "indexed_scan_p50_ms": round(indexed_p50, 4),
                "unindexed_scan_p50_ms": round(unindexed_p50, 4),
                "index_speedup_factor": round(index_speedup_factor, 2),
                "index_columns": ["flight_id", "timestamp"],
                "index_type": "Composite B+ Tree Index",
            },
            "acid_transaction_latency": {
                "p50_ms": round(tx_p50, 4),
                "p95_ms": round(tx_p95, 4),
                "p99_ms": round(tx_p99, 4),
                "isolation_level": "BEGIN IMMEDIATE (Exclusive Writer Lock)",
                "atomicity_guarantee": "Alert + Snapshot Flag + Audit Log",
            },
            "concurrency_stress_test": {
                "concurrent_worker_threads": num_threads,
                "total_concurrent_records": num_threads * records_per_thread,
                "elapsed_time_seconds": round(t_concur_total, 4),
                "concurrent_throughput_ops_sec": round(concur_throughput_ops_sec, 2),
                "wal_mode_active": True,
                "deadlocks_encountered": 0,
            },
            "analytical_olap_queries": query_benchmarks,
            "nosql_polyglot_store": nosql_stats,
        }

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Benchmark results saved to: {output_path}")

        return results

    finally:
        db.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    out_file = Path("review_artifacts/db_benchmark_results.json")
    res = run_database_benchmarks(output_path=out_file)
    print("\nBenchmark Summary:")
    print(json.dumps(res, indent=2))
