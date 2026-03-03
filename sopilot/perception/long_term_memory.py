"""Long-Term World Model — persistent SQLite-backed fact store.

Facts: (fact_type, location, time_slot, entity_type, metric_name) tuples.
Values updated via EMA (α=0.2) with observation count tracking.

Fact types:
  hourly_activity  — entity/event counts by hour (0-23)
  severity_pattern — how often each severity level occurs by hour
  entity_frequency — how often each entity type appears

DB path: configurable (default "data/long_term_memory.db").
"""

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class FactRecord:
    id: int
    fact_type: str
    location: str
    time_slot: int    # 0-23 for hourly, -1 = all-day
    entity_type: str
    metric_name: str
    metric_value: float
    confidence: float
    observations: int
    first_seen: str   # ISO timestamp
    last_seen: str    # ISO timestamp


class LongTermMemoryStore:
    """SQLite-backed long-term world model."""
    EMA_ALPHA = 0.2

    def __init__(self, db_path: str | Path = "data/long_term_memory.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ltm_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact_type TEXT NOT NULL,
                    location TEXT NOT NULL DEFAULT '',
                    time_slot INTEGER NOT NULL DEFAULT -1,
                    entity_type TEXT NOT NULL DEFAULT '',
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    observations INTEGER NOT NULL DEFAULT 1,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    UNIQUE(fact_type, location, time_slot, entity_type, metric_name)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ltm_time_slot ON ltm_facts(time_slot)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ltm_fact_type ON ltm_facts(fact_type)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint=100")
        return conn

    def close(self) -> None:
        """Checkpoint WAL and release any file locks. Call before deleting the DB file."""
        with self._lock:
            try:
                conn = sqlite3.connect(str(self._db_path), timeout=5)
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.close()
            except Exception:
                pass

    def upsert_fact(
        self,
        fact_type: str,
        metric_name: str,
        new_value: float,
        location: str = "",
        time_slot: int = -1,
        entity_type: str = "",
    ) -> None:
        """Insert or update fact with EMA blending and confidence growth."""
        now_str = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._lock:
            with self._connect() as conn:
                row = conn.execute("""
                    SELECT id, metric_value, confidence FROM ltm_facts
                    WHERE fact_type=? AND location=? AND time_slot=? AND entity_type=? AND metric_name=?
                """, (fact_type, location, time_slot, entity_type, metric_name)).fetchone()
                if row:
                    new_ema = self.EMA_ALPHA * new_value + (1 - self.EMA_ALPHA) * row["metric_value"]
                    new_conf = min(1.0, row["confidence"] + (1 - row["confidence"]) * 0.05)
                    conn.execute("""
                        UPDATE ltm_facts SET metric_value=?, confidence=?, observations=observations+1, last_seen=?
                        WHERE id=?
                    """, (new_ema, new_conf, now_str, row["id"]))
                else:
                    conn.execute("""
                        INSERT INTO ltm_facts
                        (fact_type, location, time_slot, entity_type, metric_name, metric_value, confidence, observations, first_seen, last_seen)
                        VALUES (?, ?, ?, ?, ?, ?, 0.5, 1, ?, ?)
                    """, (fact_type, location, time_slot, entity_type, metric_name, new_value, now_str, now_str))

    def record_episode_facts(self, episode: dict[str, Any]) -> None:
        """Extract and store facts from a closed episode dict.
        Expected keys: start_time, end_time, event_count, severity, event_type_counts, entity_ids, duration_seconds."""
        if not episode.get("end_time"):
            return  # Only closed episodes
        start_ts = episode.get("start_time", time.time())
        hour = int(time.localtime(start_ts).tm_hour)
        event_count = float(episode.get("event_count", 0))
        entity_count = float(len(episode.get("entity_ids", [])))
        severity = (episode.get("severity") or "info").lower()
        duration = float(episode.get("duration_seconds", 0.0))

        self.upsert_fact("hourly_activity", "entity_count", entity_count, time_slot=hour)
        self.upsert_fact("hourly_activity", "event_count", event_count, time_slot=hour)
        self.upsert_fact("hourly_activity", "episode_duration_seconds", duration, time_slot=hour)

        sev_score = {"critical": 1.0, "warning": 0.5, "info": 0.1}.get(severity, 0.0)
        self.upsert_fact("severity_pattern", "severity_score", sev_score, time_slot=hour)

        for event_type, count in (episode.get("event_type_counts") or {}).items():
            self.upsert_fact("entity_frequency", "count", float(count), entity_type=event_type)

    def get_hourly_pattern(self, hour: int) -> list[FactRecord]:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM ltm_facts WHERE time_slot=? ORDER BY observations DESC",
                    (hour,)
                ).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def get_location_facts(self, location: str) -> list[FactRecord]:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM ltm_facts WHERE location=? ORDER BY observations DESC",
                    (location,)
                ).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def get_expected_activity(self, hour: int, entity_type: str = "") -> float:
        with self._lock:
            with self._connect() as conn:
                if entity_type:
                    row = conn.execute("""
                        SELECT metric_value FROM ltm_facts
                        WHERE fact_type='hourly_activity' AND time_slot=? AND entity_type=? AND metric_name='entity_count'
                    """, (hour, entity_type)).fetchone()
                else:
                    row = conn.execute("""
                        SELECT metric_value FROM ltm_facts
                        WHERE fact_type='hourly_activity' AND time_slot=? AND metric_name='entity_count'
                        ORDER BY observations DESC LIMIT 1
                    """, (hour,)).fetchone()
        return float(row["metric_value"]) if row else 0.0

    def generate_summary_ja(self, max_facts: int = 5) -> str:
        with self._lock:
            with self._connect() as conn:
                total = conn.execute("SELECT COUNT(*) as c FROM ltm_facts").fetchone()["c"]
                if total == 0:
                    return "長期記憶にデータなし。"
                busy_row = conn.execute("""
                    SELECT time_slot, metric_value FROM ltm_facts
                    WHERE fact_type='hourly_activity' AND metric_name='entity_count'
                    ORDER BY metric_value DESC LIMIT 1
                """).fetchone()
                top_events = conn.execute("""
                    SELECT entity_type, metric_value FROM ltm_facts
                    WHERE fact_type='entity_frequency'
                    ORDER BY metric_value DESC LIMIT 3
                """).fetchall()
        parts = [f"長期記憶: {total}件の事実を蓄積。"]
        if busy_row:
            parts.append(f"最も活動が多い時間帯: {busy_row['time_slot']}時頃 (平均{busy_row['metric_value']:.1f}エンティティ)。")
        if top_events:
            types_str = "、".join(r["entity_type"] for r in top_events if r["entity_type"])
            if types_str:
                parts.append(f"頻出イベント種別: {types_str}。")
        return "".join(parts)

    def get_state_dict(self) -> dict:
        with self._lock:
            with self._connect() as conn:
                total = conn.execute("SELECT COUNT(*) as c FROM ltm_facts").fetchone()["c"]
                by_type = conn.execute(
                    "SELECT fact_type, COUNT(*) as c FROM ltm_facts GROUP BY fact_type"
                ).fetchall()
        return {
            "total_facts": total,
            "by_type": {r["fact_type"]: r["c"] for r in by_type},
            "db_path": str(self._db_path),
        }

    def clear(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM ltm_facts")

    def _row_to_fact(self, row: sqlite3.Row) -> FactRecord:
        return FactRecord(
            id=row["id"], fact_type=row["fact_type"], location=row["location"],
            time_slot=row["time_slot"], entity_type=row["entity_type"],
            metric_name=row["metric_name"], metric_value=row["metric_value"],
            confidence=row["confidence"], observations=row["observations"],
            first_seen=row["first_seen"], last_seen=row["last_seen"],
        )
