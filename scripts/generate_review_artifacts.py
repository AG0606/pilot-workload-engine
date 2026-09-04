import json
from pathlib import Path
import sys

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.datasets.workload_dataset import PilotWorkloadDataset
from src.engine.decision_engine import CockpitDecisionEngine
from src.models.workload_classifier import MultiModalWorkloadClassifier
from src.training.losses import FocalLoss, compute_class_weights
from src.training.metrics import compute_classification_metrics, compute_kaggle_log_loss
from src.training.trainer import WorkloadTrainer


def generate_review_artifacts(output_dir: Path = Path("review_artifacts")) -> None:
    """Generates evaluation metrics, confusion matrix, and decluttering timeline for Review 1."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate synthetic 4-class multi-modal dataset
    num_windows = 400
    np.random.seed(42)
    torch.manual_seed(42)

    # Simulating distinct feature distributions per state
    eeg = np.random.randn(num_windows, 17, 80).astype(np.float32)
    cardio = np.random.randn(num_windows, 3, 80).astype(np.float32)
    ocular = np.random.randn(num_windows, 4, 80).astype(np.float32)
    context = np.random.randn(num_windows, 8, 80).astype(np.float32)

    # Class distribution: 50% Baseline (0), 20% CA (1), 20% DA (2), 10% SS (3)
    labels = np.random.choice([0, 1, 2, 3], size=num_windows, p=[0.50, 0.20, 0.20, 0.10]).astype(np.int64)

    # Correlate physiological features with labels
    for i in range(num_windows):
        c = labels[i]
        if c == 1:  # CA: high theta, beta
            eeg[i, :4] += 1.5
        elif c == 2:  # DA: erratic ocular gaze
            ocular[i] += 1.8
        elif c == 3:  # SS: high tachycardia, G-load spike
            cardio[i, 0] += 2.5
            context[i, 3] += 2.0  # Vertical G spike

    dataset = PilotWorkloadDataset.from_window_dict({
        "eeg": eeg,
        "cardio": cardio,
        "ocular": ocular,
        "context": context,
        "label": labels,
    })

    val_len = int(num_windows * 0.25)
    train_len = num_windows - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    train_labels = dataset.labels[train_set.indices]
    class_weights = compute_class_weights(train_labels, num_classes=4).to(device)

    model = MultiModalWorkloadClassifier(
        num_classes=4,
        eeg_channels=17,
        cardio_channels=3,
        ocular_channels=4,
        context_channels=8,
        temporal_samples=80,
        embed_dim=128,
        fused_dim=256,
        dropout_rate=0.2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = FocalLoss(gamma=2.0, alpha=class_weights)
    trainer = WorkloadTrainer(model=model, optimizer=optimizer, criterion=criterion, device=device)

    _ = trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=6)
    eval_metrics = trainer.evaluate(val_loader)

    # Save metrics JSON
    clean_metrics = {}
    for k, v in eval_metrics.items():
        if isinstance(v, np.ndarray):
            clean_metrics[k] = v.tolist()
        else:
            clean_metrics[k] = float(v)

    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(clean_metrics, f, indent=2)

    # Format Confusion Matrix ASCII table
    conf_mat = eval_metrics["confusion_matrix"]
    class_names = ["Baseline", "CA", "DA", "Startle"]
    header_title = "True \\ Pred"
    conf_str = "================ CONFUSION MATRIX ================\n"
    conf_str += f"{header_title:<12} | " + " | ".join(f"{name:<10}" for name in class_names) + "\n"
    conf_str += "-" * 60 + "\n"
    for r_idx, true_name in enumerate(class_names):
        row_vals = " | ".join(f"{conf_mat[r_idx, c]:<10d}" for c in range(4))
        conf_str += f"{true_name:<12} | {row_vals}\n"
    conf_str += "==================================================\n"

    conf_path = output_dir / "confusion_matrix.txt"
    with open(conf_path, "w", encoding="utf-8") as f:
        f.write(conf_str)

    # Simulate Cockpit Decision Engine on validation sequence
    engine = CockpitDecisionEngine()
    timeline_lines = [
        "================ COCKPIT DECISION ENGINE ACTION TIMELINE ================",
        f"{'Window':<7} | {'State':<22} | {'Confidence':<10} | {'Declutter Level':<20} | {'Audio Alert':<26} | {'Task Allocation'}",
        "-" * 125,
    ]

    model.eval()
    val_sample_batch = next(iter(val_loader))
    probs = model.predict_proba(val_sample_batch).cpu().numpy()

    timeline_data = []
    for w_idx in range(min(len(probs), 20)):
        p_vec = probs[w_idx]
        decision = engine.process_window(p_vec)
        line = (
            f"W_{w_idx:03d}   | {decision.cognitive_state:<22} | {decision.confidence*100:5.1f}%     | "
            f"{decision.declutter_level.value:<20} | {decision.auditory_alert.value:<26} | {decision.task_allocation.value}"
        )
        timeline_lines.append(line)
        timeline_data.append({
            "window_index": w_idx,
            "state": decision.cognitive_state,
            "confidence": decision.confidence,
            "declutter_level": decision.declutter_level.value,
            "auditory_alert": decision.auditory_alert.value,
            "task_allocation": decision.task_allocation.value,
            "active_displays": decision.active_display_elements,
            "recommendations": decision.recommendations,
        })

    timeline_lines.append("=" * 125)
    timeline_txt_path = output_dir / "cockpit_declutter_timeline.txt"
    with open(timeline_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(timeline_lines))

    timeline_json_path = output_dir / "cockpit_declutter_timeline.json"
    with open(timeline_json_path, "w", encoding="utf-8") as f:
        json.dump(timeline_data, f, indent=2)

    print("Review 1 artifacts successfully generated in review_artifacts/:")
    print(f"  - {metrics_path}")
    print(f"  - {conf_path}")
    print(f"  - {timeline_txt_path}")
    print(f"  - {timeline_json_path}")
    print("\n" + conf_str)


if __name__ == "__main__":
    generate_review_artifacts()
