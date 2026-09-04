import argparse
from pathlib import Path
import sys
from typing import Dict, Tuple

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml

from src.data.kaggle_aviation_loader import KaggleAviationLoader
from src.data.synchronizer import MultiModalSynchronizer, SlidingWindowExtractor
from src.datasets.workload_dataset import PilotWorkloadDataset, create_workload_dataloader
from src.utils.validation import validate_dataset_batch, validate_sliding_windows


def generate_synthetic_cockpit_stream(
    duration_sec: float = 60.0,
    target_fs_hz: float = 20.0,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Generates synthetic multi-rate flight and physiological streams for pipeline dry-runs."""
    # Raw EEG (256 Hz, 17 channels)
    n_eeg = int(duration_sec * 256)
    t_eeg = np.linspace(0, duration_sec, n_eeg, endpoint=False)
    eeg_data = np.random.randn(n_eeg, 17).astype(np.float32)

    # Cardio & autonomic (256 Hz: ecg, r, gsr)
    n_cardio = int(duration_sec * 256)
    t_cardio = np.linspace(0, duration_sec, n_cardio, endpoint=False)
    cardio_data = np.random.randn(n_cardio, 3).astype(np.float32)

    # Ocular (60 Hz: 4 channels)
    n_ocular = int(duration_sec * 60)
    t_ocular = np.linspace(0, duration_sec, n_ocular, endpoint=False)
    ocular_data = np.random.randn(n_ocular, 4).astype(np.float32)

    # Context / flight telemetry (10 Hz: 8 channels)
    n_telemetry = int(duration_sec * 10)
    t_telemetry = np.linspace(0, duration_sec, n_telemetry, endpoint=False)
    telemetry_data = np.random.randn(n_telemetry, 8).astype(np.float32)
    # Set vertical G (index 3) around 1.0G
    telemetry_data[:, 3] += 1.0

    # Labels (256 Hz, classes 0 to 3)
    t_label = t_eeg
    label_data = np.zeros(n_eeg, dtype=np.int64)
    # Transition across 4 states
    q = n_eeg // 4
    label_data[q : 2 * q] = 1
    label_data[2 * q : 3 * q] = 2
    label_data[3 * q :] = 3

    streams = {
        "eeg": (t_eeg, eeg_data),
        "cardio": (t_cardio, cardio_data),
        "ocular": (t_ocular, ocular_data),
        "context": (t_telemetry, telemetry_data),
        "label": (t_label, label_data),
    }

    # Session markers with a single session for synthetic duration
    ref_time_len = int(np.floor(duration_sec * target_fs_hz))
    session_ids = np.zeros(ref_time_len, dtype=int)

    return streams, session_ids


def run_pipeline(
    config_path: Path,
    raw_csv: Path | None = None,
    output_dir: Path = Path("output"),
    synthetic: bool = False,
    crew_filter: int | None = None,
    experiment_filter: str | None = None,
    max_rows: int | None = None,
) -> None:
    """Executes ingestion, time synchronization, windowing, and PyTorch dataset serialization."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    target_fs = float(config["sampling_rates"]["target_reference_hz"])
    win_size_sec = float(config["windowing"]["window_size_sec"])
    stride_sec = float(config["windowing"]["stride_sec"])
    max_gap_sec = float(config["windowing"]["max_timestamp_gap_sec"])

    output_dir.mkdir(parents=True, exist_ok=True)
    synchronizer = MultiModalSynchronizer(target_fs_hz=target_fs)

    if synthetic or raw_csv is None:
        streams, session_ids = generate_synthetic_cockpit_stream(
            duration_sec=60.0,
            target_fs_hz=target_fs,
        )
        ref_time, aligned = synchronizer.synchronize_streams(
            streams=streams,
            categorical_streams=["label"],
        )
    else:
        loader = KaggleAviationLoader(
            time_column="time",
            eeg_channels=config["channels"]["eeg"],
            cardio_channels=config["channels"]["cardio"],
        )
        df = loader.load(
            raw_csv,
            crew_filter=crew_filter,
            experiment_filter=experiment_filter,
            max_rows=max_rows,
        )
        extracted = loader.extract_arrays(df)

        t_vec = extracted["time"]
        # Generate companion dummy ocular and flight telemetry for single-source datasets
        n_samples = len(t_vec)
        ocular_dummy = np.zeros((n_samples, len(config["channels"]["ocular"])), dtype=np.float32)
        telemetry_dummy = np.zeros((n_samples, len(config["channels"]["context"])), dtype=np.float32)
        # Standard 1.0G equilibrium
        telemetry_dummy[:, 3] = 1.0

        streams = {
            "eeg": (t_vec, extracted["eeg"]),
            "cardio": (t_vec, extracted["cardio"]),
            "ocular": (t_vec, ocular_dummy),
            "context": (t_vec, telemetry_dummy),
            "label": (t_vec, extracted["label"]),
        }
        ref_time, aligned = synchronizer.synchronize_streams(
            streams=streams,
            categorical_streams=["label"],
        )
        # Resample session IDs via nearest neighbor
        session_ids = synchronizer.align_modality(
            time_vec=t_vec,
            data=extracted["session_id"],
            reference_time=ref_time,
            is_categorical=True,
        ).squeeze()

    extractor = SlidingWindowExtractor(
        window_size_sec=win_size_sec,
        stride_sec=stride_sec,
        target_fs_hz=target_fs,
        max_timestamp_gap_sec=max_gap_sec,
    )
    windows = extractor.extract_windows(
        data_streams=aligned,
        reference_time=ref_time,
        session_ids=session_ids,
    )

    validate_sliding_windows(
        windows=windows,
        expected_samples=int(round(win_size_sec * target_fs)),
    )

    dataset = PilotWorkloadDataset.from_window_dict(windows)
    dataloader = create_workload_dataloader(dataset, batch_size=16, shuffle=False)

    for batch in dataloader:
        validate_dataset_batch(batch, num_classes=4)
        break

    save_path = output_dir / "pilot_workload_dataset.pt"
    dataset.save(save_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process multimodal pilot workload streams.")
    parser.add_argument("--config", type=Path, default=Path("config/data_config.yaml"))
    parser.add_argument("--raw-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--synthetic", action="store_true", default=False)
    parser.add_argument("--crew", type=int, default=None)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--max-rows", type=int, default=None)

    args = parser.parse_args()
    run_pipeline(
        config_path=args.config,
        raw_csv=args.raw_csv,
        output_dir=args.output_dir,
        synthetic=args.synthetic,
        crew_filter=args.crew,
        experiment_filter=args.experiment,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
