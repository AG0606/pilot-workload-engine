from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from src.database.db_manager import DatabaseManager


class AvionicsAnalyticsEngine:
    """Executes longitudinal and operational analytics queries on e-Pilot database."""

    def __init__(self, db_manager: DatabaseManager, sql_path: Optional[Path] = None) -> None:
        self.db = db_manager
        if sql_path is None:
            sql_path = Path(__file__).parent / "analytics_queries.sql"
        self.sql_path = Path(sql_path)
        self.queries = self._parse_queries()

    def _parse_queries(self) -> Dict[str, str]:
        """Parses named queries from analytics_queries.sql."""
        text = self.sql_path.read_text(encoding="utf-8")
        parsed = {}
        current_name = None
        current_sql = []

        for line in text.splitlines():
            if line.strip().startswith("-- QUERY_NAME:"):
                if current_name and current_sql:
                    parsed[current_name] = "\n".join(current_sql).strip()
                current_name = line.split(":", 1)[1].strip()
                current_sql = []
            elif current_name:
                # Exclude comment-only lines before query
                if not line.strip().startswith("--"):
                    current_sql.append(line)

        if current_name and current_sql:
            parsed[current_name] = "\n".join(current_sql).strip()

        return parsed

    def run_query(self, query_name: str) -> List[Dict[str, Any]]:
        """Executes a named query and returns list of dictionaries."""
        if query_name not in self.queries:
            raise KeyError(f"Query '{query_name}' not found. Available: {list(self.queries.keys())}")

        sql = self.queries[query_name]
        with self.db.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    def run_query_df(self, query_name: str) -> pd.DataFrame:
        """Executes a named query and returns pandas DataFrame."""
        records = self.run_query(query_name)
        return pd.DataFrame(records)

    def compare_normal_vs_emergency(self) -> pd.DataFrame:
        """Executes Section 6.5 Normal vs. Simulated Emergency contrast."""
        return self.run_query_df("normal_vs_emergency_contrast")

    def get_rolling_cognitive_trends(self) -> pd.DataFrame:
        """Executes Section 6.5 Window Function rolling moving averages."""
        return self.run_query_df("rolling_cognitive_trend")

    def get_alert_distribution(self) -> pd.DataFrame:
        """Executes Section 6.4/6.5 alert distribution analysis."""
        return self.run_query_df("alert_severity_distribution")

    def get_crew_asymmetry(self) -> pd.DataFrame:
        """Executes dual-pilot workload asymmetry analysis."""
        return self.run_query_df("crew_cognitive_asymmetry")

    def get_dwell_time_percentage(self) -> pd.DataFrame:
        """Executes flight cognitive state dwell time percentages."""
        return self.run_query_df("regime_dwell_time_percentage")
