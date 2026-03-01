"""SQLite repository for VigilPilot sessions and violation events."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


class VigilRepository:
    """Thin data-access layer for vigil_sessions and vigil_events tables."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Sessions ───────────────────────────────────────────────────────────

    def create_session(
        self,
        name: str,
        rules: list[str],
        sample_fps: float,
        severity_threshold: str,
    ) -> int:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO vigil_sessions
                  (name, rules_json, sample_fps, severity_threshold, status,
                   total_frames_analyzed, violation_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'idle', 0, 0, ?, ?)
                """,
                (name, json.dumps(rules, ensure_ascii=False), sample_fps, severity_threshold, now, now),
            )
            return int(cur.lastrowid)  # type: ignore[arg-type]

    def get_session(self, session_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM vigil_sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["rules"] = json.loads(d.pop("rules_json"))
        return d

    def list_sessions(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM vigil_sessions ORDER BY created_at DESC"
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["rules"] = json.loads(d.pop("rules_json"))
            result.append(d)
        return result

    def update_session_status(
        self,
        session_id: int,
        status: str,
        video_filename: str | None = None,
        total_frames_analyzed: int | None = None,
        violation_count: int | None = None,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            sets = ["status = ?", "updated_at = ?"]
            params: list = [status, now]
            if video_filename is not None:
                sets.append("video_filename = ?")
                params.append(video_filename)
            if total_frames_analyzed is not None:
                sets.append("total_frames_analyzed = ?")
                params.append(total_frames_analyzed)
            if violation_count is not None:
                sets.append("violation_count = ?")
                params.append(violation_count)
            params.append(session_id)
            conn.execute(
                f"UPDATE vigil_sessions SET {', '.join(sets)} WHERE id = ?",  # noqa: S608
                params,
            )

    def delete_session(self, session_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM vigil_sessions WHERE id = ?", (session_id,)
            )
            conn.execute(
                "DELETE FROM vigil_events WHERE session_id = ?", (session_id,)
            )
            return cur.rowcount > 0

    # ── Events ─────────────────────────────────────────────────────────────

    def create_event(
        self,
        session_id: int,
        timestamp_sec: float,
        frame_number: int,
        violations: list[dict],
        frame_path: str | None = None,
    ) -> int:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO vigil_events
                  (session_id, timestamp_sec, frame_number, violations_json,
                   frame_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    timestamp_sec,
                    frame_number,
                    json.dumps(violations, ensure_ascii=False),
                    frame_path,
                    now,
                ),
            )
            return int(cur.lastrowid)  # type: ignore[arg-type]

    def list_events(self, session_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM vigil_events WHERE session_id = ? ORDER BY timestamp_sec",
                (session_id,),
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["violations"] = json.loads(d.pop("violations_json"))
            result.append(d)
        return result

    def get_event(self, event_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM vigil_events WHERE id = ?", (event_id,)
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["violations"] = json.loads(d.pop("violations_json"))
        return d
