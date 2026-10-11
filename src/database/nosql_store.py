import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np


class HybridNoSQLTimeSeriesStore:
    """Secondary Time-Series & Document Store for Raw High-Frequency Avionics Waveforms.

    Fulfills BCSE302L / BCSE302P Module 7 (NoSQL Systems & Polyglot Persistence):
    - Stores multi-channel raw time-series arrays (256 Hz EEG, ECG, 60 Hz Ocular) as JSON documents.
    - Prevents relational table row explosion while keeping relational DB as the authoritative system of record.
    - Demonstrates CAP theorem trade-offs: Eventual Consistency for high-throughput raw telemetry documents
      vs. Strict ACID Consistency for alerts and state transitions in the relational DBMS.
    """

    def __init__(self, storage_dir: Union[str, Path] = "nosql_telemetry_store") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._doc_count = 0

    def store_raw_waveform_document(
        self,
        flight_id: str,
        pilot_id: str,
        start_time: float,
        end_time: float,
        sampling_rate_hz: float,
        channels: Dict[str, Union[List[float], np.ndarray]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Stores a raw sensor window as a BSON/JSON document."""
        doc_id = f"DOC-{flight_id}-{int(start_time * 1000)}"
        file_path = self.storage_dir / f"{doc_id}.json"

        # Serialize numpy arrays safely
        serialized_channels = {}
        for ch_name, data in channels.items():
            if isinstance(data, np.ndarray):
                serialized_channels[ch_name] = data.tolist()
            else:
                serialized_channels[ch_name] = list(data)

        doc = {
            "_id": doc_id,
            "flight_id": flight_id,
            "pilot_id": pilot_id,
            "time_window": {
                "start": float(start_time),
                "end": float(end_time),
                "duration_seconds": float(end_time - start_time),
            },
            "sampling_rate_hz": float(sampling_rate_hz),
            "channel_count": len(channels),
            "channels": serialized_channels,
            "metadata": metadata or {},
            "created_at_epoch": time.time(),
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(doc, f)

        self._doc_count += 1
        return doc_id

    def query_waveform_by_window(
        self,
        flight_id: str,
        target_timestamp: float,
    ) -> Optional[Dict[str, Any]]:
        """Finds document covering target timestamp for a specific flight."""
        for file_path in self.storage_dir.glob(f"DOC-{flight_id}-*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                    win = doc.get("time_window", {})
                    if win.get("start", 0) <= target_timestamp <= win.get("end", 0):
                        return doc
            except Exception:
                continue
        return None

    def get_collection_statistics(self) -> Dict[str, Any]:
        """Returns document collection metrics for evaluation reports."""
        all_files = list(self.storage_dir.glob("*.json"))
        total_bytes = sum(f.stat().st_size for f in all_files) if all_files else 0
        return {
            "document_count": len(all_files),
            "total_size_kb": total_bytes / 1024.0,
            "storage_path": str(self.storage_dir),
            "format": "JSON/BSON Document Collection",
        }
