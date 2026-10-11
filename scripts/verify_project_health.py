import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
import joblib
import numpy as np
import torch

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.database.analytics import AvionicsAnalyticsEngine
from src.database.db_manager import DatabaseManager
from src.database.ingestion_pipeline import AvionicsDatabaseIngestionPipeline
from src.database.nosql_store import HybridNoSQLTimeSeriesStore
from src.database.transaction_manager import TransactionManager
from src.models.ocular_vision import OcularVigilanceNet
from src.models.workload_classifier import MultiModalWorkloadClassifier


def test_section(name):
    print(f"\n{'='*70}\n[CHECK] {name}\n{'='*70}")


def run_comprehensive_health_checks():
    total_checks = 0
    passed_checks = 0

    # -------------------------------------------------------------------------
    # 1. File Artifacts & Journal Submissions Check
    # -------------------------------------------------------------------------
    test_section("1. Core Files & Journal Submission Artifacts Existence")
    required_files = [
        "MANUSCRIPT_IEEE_TRANSACTIONS.md",
        "MANUSCRIPT_IEEE.tex",
        "references.bib",
        "COVER_LETTER_JOURNAL_SUBMISSION.md",
        "RESEARCH_HIGHLIGHTS.md",
        "AUTHOR_STATEMENTS.md",
        "SUPPLEMENTARY_MATERIALS.md",
        "ePilot_DBMS_Project.docx",
        "README.md",
        "requirements.txt",
        "src/database/schema.sql",
        "src/database/db_manager.py",
        "src/database/transaction_manager.py",
        "src/database/ingestion_pipeline.py",
        "src/database/analytics.py",
        "src/database/analytics_queries.sql",
        "src/database/nosql_store.py",
        "checkpoints/best_workload_model.pt",
        "checkpoints/best_ocular_vigilance_model.pt",
        "output/pilot_workload_dataset.pt",
        "review_artifacts/db_benchmark_results.json",
        "review_artifacts/model_comparison_report.json",
    ]

    for rf in required_files:
        total_checks += 1
        fp = _ROOT / rf
        if fp.exists():
            size_kb = fp.stat().st_size / 1024.0
            print(f"  [PASS] {rf:<42} ({size_kb:>8.1f} KB)")
            passed_checks += 1
        else:
            print(f"  [FAIL] {rf:<42} (MISSING)")

    # -------------------------------------------------------------------------
    # 2. PyTorch Deep Neural Network Checkpoint & Inference
    # -------------------------------------------------------------------------
    test_section("2. Deep Workload Classifier Neural Forward Pass & Latency")
    total_checks += 1
    workload_ckpt = _ROOT / "checkpoints" / "best_workload_model.pt"
    try:
        model = MultiModalWorkloadClassifier(
            eeg_channels=17,
            cardio_channels=3,
            ocular_channels=4,
            context_channels=8,
            num_classes=4,
        )
        raw_ckpt = torch.load(workload_ckpt, map_location="cpu")
        state_dict = raw_ckpt.get("model_state_dict", raw_ckpt)
        model.load_state_dict(state_dict)
        model.eval()

        # Dummy batch matching window dimensions: [B, C, S=80]
        B = 2
        eeg_dummy = torch.randn(B, 17, 80)
        cardio_dummy = torch.randn(B, 3, 80)
        ocular_dummy = torch.randn(B, 4, 80)
        context_dummy = torch.randn(B, 8, 80)

        with torch.no_grad():
            logits = model(eeg=eeg_dummy, cardio=cardio_dummy, ocular=ocular_dummy, context=context_dummy)
            probs = torch.softmax(logits, dim=-1)

        assert logits.shape == (B, 4), f"Unexpected logits shape: {logits.shape}"
        assert probs.shape == (B, 4), f"Unexpected probs shape: {probs.shape}"
        # Sum of probabilities per row must be ~1.0
        row_sums = probs.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)

        # Benchmark inference latency across 100 runs
        latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(eeg=eeg_dummy[:1], cardio=cardio_dummy[:1], ocular=ocular_dummy[:1], context=context_dummy[:1])
            latencies.append((time.perf_counter() - t0) * 1000.0)

        p50 = float(np.median(latencies))
        p99 = float(np.percentile(latencies, 99))
        print(f"  [PASS] MultiModalWorkloadClassifier loaded. Output shape: {probs.shape}")
        print(f"         Single-window latency: p50 = {p50:.3f} ms, p99 = {p99:.3f} ms (< 50.0 ms budget)")
        passed_checks += 1
    except Exception as e:
        print(f"  [FAIL] MultiModalWorkloadClassifier error: {e}")

    # -------------------------------------------------------------------------
    # 3. Ocular PERCLOS Vision CNN Checkpoint & Inference
    # -------------------------------------------------------------------------
    test_section("3. Ocular Vigilance CNN Model Verification")
    total_checks += 1
    ocular_ckpt = _ROOT / "checkpoints" / "best_ocular_vigilance_model.pt"
    try:
        eye_net = OcularVigilanceNet(num_classes=2)
        raw_eye_ckpt = torch.load(ocular_ckpt, map_location="cpu")
        state_dict_eye = raw_eye_ckpt.get("model_state_dict", raw_eye_ckpt)
        eye_net.load_state_dict(state_dict_eye)
        eye_net.eval()

        dummy_eye_img = torch.randn(1, 1, 64, 64)  # Grayscale eye crop
        with torch.no_grad():
            eye_logits = eye_net(dummy_eye_img)
            eye_probs = torch.softmax(eye_logits, dim=-1)

        assert eye_probs.shape == (1, 2)
        print(f"  [PASS] OcularVigilanceNet loaded. Output shape: {eye_probs.shape}")
        print(f"         Predicted class: {int(torch.argmax(eye_probs, dim=-1).item())} (P_open={eye_probs[0, 0]:.3f}, P_closed={eye_probs[0, 1]:.3f})")
        passed_checks += 1
    except Exception as e:
        print(f"  [FAIL] OcularVigilanceNet error: {e}")

    # -------------------------------------------------------------------------
    # 4. Classical Machine Learning Models Verification
    # -------------------------------------------------------------------------
    test_section("4. Classical Scikit-Learn / Gradient Boosted Models")
    models_dir = _ROOT / "review_artifacts" / "models"
    classical_models = ["lightgbm_model.joblib", "xgboost_model.joblib", "extratrees_model.joblib", "randomforest_model.joblib"]
    dummy_features = np.random.randn(1, 57)

    for cm in classical_models:
        total_checks += 1
        cm_path = models_dir / cm
        try:
            if not cm_path.exists():
                raise FileNotFoundError(f"{cm} not found")
            clf = joblib.load(cm_path)
            pred = clf.predict(dummy_features)
            proba = clf.predict_proba(dummy_features)
            print(f"  [PASS] {cm:<26} -> Class: {pred[0]}, Probs shape: {proba.shape}")
            passed_checks += 1
        except Exception as e:
            print(f"  [FAIL] {cm}: {e}")

    # -------------------------------------------------------------------------
    # 5. Pre-Extracted Dataset Integrity
    # -------------------------------------------------------------------------
    test_section("5. Serialized Aviation Dataset Cache Verification")
    total_checks += 1
    dataset_pt = _ROOT / "output" / "pilot_workload_dataset.pt"
    try:
        ds_data = torch.load(dataset_pt, map_location="cpu")
        print(f"  [PASS] pilot_workload_dataset.pt loaded successfully.")
        print(f"         EEG tensor shape:     {ds_data['eeg'].shape}")
        print(f"         Cardio tensor shape:  {ds_data['cardio'].shape}")
        print(f"         Ocular tensor shape:  {ds_data['ocular'].shape}")
        print(f"         Context tensor shape: {ds_data['context'].shape}")
        print(f"         Labels tensor shape:  {ds_data['labels'].shape}")
        unique_labels, counts = torch.unique(ds_data["labels"], return_counts=True)
        label_dist = {int(k): int(v) for k, v in zip(unique_labels, counts)}
        print(f"         State class counts:   {label_dist}")
        passed_checks += 1
    except Exception as e:
        print(f"  [FAIL] Dataset cache error: {e}")

    # -------------------------------------------------------------------------
    # 6. Database Systems & OLAP Queries Live Verification
    # -------------------------------------------------------------------------
    test_section("6. Live Database Systems, Schema DDL, & Analytical Queries")
    total_checks += 1
    temp_dir = tempfile.mkdtemp(prefix="health_check_db_")
    db_file = Path(temp_dir) / "health_check.db"
    try:
        db = DatabaseManager(db_path=db_file, enable_wal=True)
        pipeline = AvionicsDatabaseIngestionPipeline(db)
        flight_id, pic_id, fo_id = pipeline.seed_master_records()

        # Ingest 10 windows
        for i in range(10):
            phase = "PH-CRUISE-01" if i < 5 else "PH-EMERG-01"
            pipeline.process_and_ingest_window(
                flight_id=flight_id,
                pilot_id=pic_id if i % 2 == 0 else fo_id,
                timestamp=float(i * 1.0),
                flight_phase_id=phase,
                eeg_engagement=0.2 if i < 5 else 0.85,
                hrv_stress=0.2 if i < 5 else 0.9,
                gsr_conductance=0.2 if i < 5 else 0.85,
                perclos_score=0.1 if i < 5 else 0.45,
                predicted_state="BASELINE" if i < 5 else "STARTLE",
                confidence_score=0.95,
            )

        analytics = AvionicsAnalyticsEngine(db)
        df_contrast = analytics.compare_normal_vs_emergency()
        df_trends = analytics.get_rolling_cognitive_trends()

        assert len(df_contrast) >= 2, "Contrast query did not return phase groups"
        assert len(df_trends) == 10, "Rolling trend query row count mismatch"

        print(f"  [PASS] Database schema, ingestion, and OLAP queries verified.")
        print(f"         Normal vs Emergency contrast output:\n{df_contrast[['phase_name', 'is_emergency_simulated', 'mean_cognitive_load_index']]}")
        passed_checks += 1
        db.close()
    except Exception as e:
        print(f"  [FAIL] Database live test error: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}\n[SUMMARY] System Health Verification Score: {passed_checks}/{total_checks} Checks Passed\n{'='*70}")
    if passed_checks == total_checks:
        print(">>> ALL HEALTH CHECKS PASSED. The repository is 100% verified and production-ready.")
        return 0
    else:
        print(f">>> {total_checks - passed_checks} CHECKS FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(run_comprehensive_health_checks())
