import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union


class DatabaseManager:
    """Manages SQLite / PostgreSQL database connections, schema lifecycle, and PRAGMA tunings.

    Features:
    - Thread-local connection pool supporting concurrent IoT ingestion threads.
    - Write-Ahead Logging (WAL) mode for simultaneous reads and writes without lock stalls.
    - Automatic DDL schema initialization from schema.sql.
    - Full Foreign Key enforcement and transaction rollback handling.
    """

    def __init__(
        self,
        db_path: Union[str, Path] = "pilot_workload.db",
        schema_path: Optional[Union[str, Path]] = None,
        timeout: float = 30.0,
        enable_wal: bool = True,
    ) -> None:
        self.db_path = str(db_path)
        self.timeout = timeout
        self.enable_wal = enable_wal
        self._local = threading.local()
        self._lock = threading.Lock()

        if schema_path is None:
            # Default to schema.sql in same directory
            schema_path = Path(__file__).parent / "schema.sql"
        self.schema_path = Path(schema_path)

        # Initialize schema once upon startup
        self.initialize_schema()

    def get_connection(self) -> sqlite3.Connection:
        """Retrieves or creates a thread-local SQLite connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            conn = sqlite3.connect(
                self.db_path,
                timeout=self.timeout,
                check_same_thread=False,
                isolation_level=None,  # Autocommit mode by default; explicit transactions used
            )
            conn.row_factory = sqlite3.Row

            # Configure performance & integrity PRAGMAs
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON;")
            if self.enable_wal and self.db_path != ":memory:":
                cursor.execute("PRAGMA journal_mode = WAL;")
                cursor.execute("PRAGMA synchronous = NORMAL;")
            cursor.execute("PRAGMA cache_size = -64000;")  # 64 MB memory cache
            cursor.execute("PRAGMA temp_store = MEMORY;")
            cursor.close()

            self._local.connection = conn

        return self._local.connection

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Provides an explicit ACID transaction context with atomic commit / rollback."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("BEGIN IMMEDIATE;")
        try:
            yield cursor
            cursor.execute("COMMIT;")
        except Exception:
            cursor.execute("ROLLBACK;")
            raise
        finally:
            cursor.close()

    @contextmanager
    def cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """Provides a standard cursor without explicit transaction wrapping."""
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            yield cur
        finally:
            cur.close()

    def initialize_schema(self, force: bool = False) -> None:
        """Executes schema.sql DDL to create tables, constraints, and B+ tree indexes."""
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found at {self.schema_path}")

        schema_ddl = self.schema_path.read_text(encoding="utf-8")
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=self.timeout)
            conn.execute("PRAGMA foreign_keys = ON;")
            if force:
                tables = [
                    "alert_audit_log",
                    "alert",
                    "cognitive_state_snapshot",
                    "emergency_event",
                    "flight_phase",
                    "sensor_reading",
                    "sensor_device",
                    "crew_assignment",
                    "flight",
                    "pilot",
                    "aircraft",
                ]
                for tbl in tables:
                    conn.execute(f"DROP TABLE IF EXISTS {tbl};")
            conn.executescript(schema_ddl)
            conn.commit()
            conn.close()

    def execute_query(
        self, query: str, params: Union[Tuple[Any, ...], Dict[str, Any]] = ()
    ) -> List[sqlite3.Row]:
        """Executes a SELECT query and returns all matching rows."""
        with self.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def execute_scalar(
        self, query: str, params: Union[Tuple[Any, ...], Dict[str, Any]] = ()
    ) -> Any:
        """Executes a query and returns the single scalar value in the first row."""
        with self.cursor() as cur:
            cur.execute(query, params)
            row = cur.fetchone()
            return row[0] if row else None

    def execute_update(
        self, query: str, params: Union[Tuple[Any, ...], Dict[str, Any]] = ()
    ) -> int:
        """Executes an INSERT, UPDATE, or DELETE query and returns the rowcount."""
        with self.cursor() as cur:
            cur.execute(query, params)
            return cur.rowcount

    def execute_batch(
        self, query: str, batch_params: Iterable[Tuple[Any, ...]]
    ) -> int:
        """Executes a batch parameterized INSERT / UPDATE using executemany."""
        with self.transaction() as cur:
            cur.executemany(query, batch_params)
            return cur.rowcount

    def close(self) -> None:
        """Closes thread-local connection if open."""
        if hasattr(self._local, "connection") and self._local.connection is not None:
            try:
                self._local.connection.close()
            except Exception:
                pass
            self._local.connection = None
