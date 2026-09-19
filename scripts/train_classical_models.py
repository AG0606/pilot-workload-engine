import argparse
import json
from pathlib import Path
import sys

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.datasets.workload_dataset import PilotWorkloadDataset
from src.features.feature_extractor import MultiModalFeatureExtractor
from src.models.classical_baselines import ClassicalWorkloadBaselines


def run_classical_benchmarks(
    data_path: Path,
    output_dir: Path,
    test_size: float = 0.20,
    random_state: int = 42,
) -> None:
    """Trains and benchmarks all classical ML models on extracted features."""
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading preprocessed dataset from {data_path}...")
    dataset = PilotWorkloadDataset.load(data_path)
    print(f"Loaded {len(dataset):,} sliding windows.")

    print("Extracting multi-modal spectral, autonomic, and kinematic tabular features...")
    extractor = MultiModalFeatureExtractor(fs_hz=20.0)
    X = extractor.extract_batch(
        eeg=dataset.eeg,
        cardio=dataset.cardio,
        ocular=dataset.ocular,
        context=dataset.context,
    )
    y = dataset.labels.numpy()
    print(f"Extracted feature matrix shape: {X.shape} ({len(extractor.feature_names)} features per window)")

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"Train samples: {len(X_train):,} | Test samples: {len(X_test):,}")
    print(f"Train class distribution: {np.bincount(y_train, minlength=4)}")
    print(f"Test class distribution:  {np.bincount(y_test, minlength=4)}")

    # Feature Scaling using RobustScaler (handles physiological outliers gracefully)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save feature scaler and feature names
    joblib.dump(scaler, models_dir / "feature_scaler.joblib")
    with open(output_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(extractor.feature_names, f, indent=2)

    baselines = ClassicalWorkloadBaselines(random_state=random_state)
    results = {}

    print("\n" + "=" * 80)
    print(f"{'Model':<15} | {'Bal Acc (%)':<12} | {'Macro F1':<10} | {'Log Loss':<10} | {'p50 Latency':<12} | {'Fit Time':<10}")
    print("-" * 80)

    for model_name in list(baselines.models.keys()):
        model, fit_time_ms = baselines.train_model(model_name, X_train_scaled, y_train)
        metrics = baselines.evaluate_model(model_name, X_test_scaled, y_test)
        metrics["fit_time_ms"] = fit_time_ms
        results[model_name] = metrics

        # Save trained model artifact
        joblib.dump(model, models_dir / f"{model_name.lower()}_model.joblib")

        print(
            f"{model_name:<15} | "
            f"{metrics['balanced_accuracy']*100:>10.2f}% | "
            f"{metrics['macro_f1']:>10.4f} | "
            f"{metrics['log_loss']:>10.4f} | "
            f"{metrics['p50_latency_ms']:>8.3f} ms | "
            f"{fit_time_ms:>7.1f} ms"
        )

    print("=" * 80 + "\n")

    # Feature Importance analysis from best tree model (RandomForest)
    top_features = baselines.get_feature_importances("RandomForest", extractor.feature_names, top_k=10)
    print("Top 10 Most Discriminative Physiological & Kinematic Features (Random Forest):")
    for rank, (feat, imp) in enumerate(top_features, 1):
        print(f"  {rank:2d}. {feat:<28} : {imp*100:>5.2f}%")

    # Serialize results to review_artifacts
    clean_results = {}
    for m_name, m_data in results.items():
        clean_results[m_name] = {
            "balanced_accuracy": m_data["balanced_accuracy"],
            "macro_f1": m_data["macro_f1"],
            "weighted_f1": m_data["weighted_f1"],
            "log_loss": m_data["log_loss"],
            "p50_latency_ms": m_data["p50_latency_ms"],
            "p99_latency_ms": m_data["p99_latency_ms"],
            "fit_time_ms": m_data["fit_time_ms"],
            "classification_report": m_data["report"],
        }

    with open(output_dir / "classical_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2)

    print(f"\nAll classical models and evaluation metrics saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate classical ML workload models.")
    parser.add_argument("--data-path", type=Path, default=Path("output/pilot_workload_dataset.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("review_artifacts"))
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    run_classical_benchmarks(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
