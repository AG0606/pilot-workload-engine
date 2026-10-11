"""e-Pilot Database Systems & IoT Persistence Architecture.

Modules:
    - db_manager: SQLite/PostgreSQL connection lifecycle, pool, and PRAGMA management.
    - models: Strongly-typed dataclass representations of all 11 BCNF entities.
    - transaction_manager: ACID transaction coordinator for safety-critical alerts.
    - ingestion_pipeline: High-throughput streaming ingestion from sensors to DB.
    - analytics: Analytical queries (OLAP, window functions, emergency vs normal).
    - nosql_store: Secondary document/time-series store adapter for raw high-frequency waveforms.
"""

from src.database.db_manager import DatabaseManager
from src.database.models import (
    Aircraft,
    Pilot,
    Flight,
    CrewAssignment,
    SensorDevice,
    SensorReading,
    FlightPhaseRecord,
    EmergencyEventRecord,
    CognitiveStateSnapshotRecord,
    AlertRecord,
    AlertAuditLogRecord,
)
from src.database.transaction_manager import TransactionManager
from src.database.ingestion_pipeline import AvionicsDatabaseIngestionPipeline
from src.database.analytics import AvionicsAnalyticsEngine
from src.database.nosql_store import HybridNoSQLTimeSeriesStore

__all__ = [
    "DatabaseManager",
    "Aircraft",
    "Pilot",
    "Flight",
    "CrewAssignment",
    "SensorDevice",
    "SensorReading",
    "FlightPhaseRecord",
    "EmergencyEventRecord",
    "CognitiveStateSnapshotRecord",
    "AlertRecord",
    "AlertAuditLogRecord",
    "TransactionManager",
    "AvionicsDatabaseIngestionPipeline",
    "AvionicsAnalyticsEngine",
    "HybridNoSQLTimeSeriesStore",
]
