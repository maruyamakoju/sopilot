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
            d.setdefault("acknowledged_at", None)
            d.setdefault("acknowledged_by", None)
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
        d.setdefault("acknowledged_at", None)
        d.setdefault("acknowledged_by", None)
        return d

    def acknowledge_event(self, event_id: int, acknowledged_by: str) -> bool:
        """Mark a violation event as acknowledged. Returns True if found and updated."""
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE vigil_events SET acknowledged_at = ?, acknowledged_by = ? WHERE id = ?",
                (now, acknowledged_by, event_id),
            )
            return cur.rowcount > 0

    def list_events_since(self, session_id: int, after_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM vigil_events WHERE session_id = ? AND id > ? ORDER BY id ASC",
                (session_id, after_id),
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["violations"] = json.loads(d.pop("violations_json"))
            result.append(d)
        return result

    # ── Webhooks ───────────────────────────────────────────────────────────────

    def set_webhook(self, session_id: int, url: str, min_severity: str = "warning") -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE vigil_sessions SET webhook_url = ?, webhook_min_severity = ?, updated_at = ? WHERE id = ?",
                (url, min_severity, now, session_id),
            )

    def clear_webhook(self, session_id: int) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE vigil_sessions SET webhook_url = NULL, updated_at = ? WHERE id = ?",
                (now, session_id),
            )

    def get_webhook(self, session_id: int) -> tuple[str, str] | None:
        row = self.get_session(session_id)
        if row is None:
            return None
        url = row.get("webhook_url")
        if not url:
            return None
        min_sev = row.get("webhook_min_severity") or "warning"
        return (url, min_sev)

    # ── Analytics ──────────────────────────────────────────────────────────────

    def get_analytics(self, days: int = 30) -> dict:
        """Return aggregated analytics for the last *days* days."""
        with self._connect() as conn:
            # Total sessions
            total_sessions: int = conn.execute(
                "SELECT COUNT(*) FROM vigil_sessions"
            ).fetchone()[0]

            # Total events
            total_events: int = conn.execute(
                "SELECT COUNT(*) FROM vigil_events"
            ).fetchone()[0]

            # Events grouped by severity (across all violations_json entries)
            # violations_json is a JSON array; each element has a "severity" key.
            # We use json_each to expand the array per event row.
            sev_rows = conn.execute(
                """
                SELECT json_extract(v.value, '$.severity') AS severity,
                       COUNT(*) AS cnt
                FROM vigil_events e,
                     json_each(e.violations_json) v
                GROUP BY severity
                """
            ).fetchall()
            events_by_severity: dict[str, int] = {"info": 0, "warning": 0, "critical": 0}
            for row in sev_rows:
                sev = row[0] or "warning"
                if sev in events_by_severity:
                    events_by_severity[sev] = row[1]

            # Events per session (join vigil_sessions for name)
            session_rows = conn.execute(
                """
                SELECT
                    e.session_id,
                    COALESCE(s.name, 'Unknown') AS session_name,
                    COUNT(DISTINCT e.id) AS total,
                    SUM(CASE WHEN json_extract(v.value,'$.severity')='critical' THEN 1 ELSE 0 END) AS critical,
                    SUM(CASE WHEN json_extract(v.value,'$.severity')='warning'  THEN 1 ELSE 0 END) AS warning,
                    SUM(CASE WHEN json_extract(v.value,'$.severity')='info'     THEN 1 ELSE 0 END) AS info
                FROM vigil_events e
                LEFT JOIN vigil_sessions s ON s.id = e.session_id,
                     json_each(e.violations_json) v
                GROUP BY e.session_id
                ORDER BY total DESC
                """
            ).fetchall()
            events_by_session = [
                {
                    "session_id": r[0],
                    "session_name": r[1],
                    "total": r[2],
                    "critical": r[3] or 0,
                    "warning": r[4] or 0,
                    "info": r[5] or 0,
                }
                for r in session_rows
            ]

            # Events per rule (top 20 by count)
            rule_rows = conn.execute(
                """
                SELECT json_extract(v.value, '$.rule') AS rule,
                       COUNT(*) AS cnt
                FROM vigil_events e,
                     json_each(e.violations_json) v
                WHERE json_extract(v.value, '$.rule') IS NOT NULL
                GROUP BY rule
                ORDER BY cnt DESC
                LIMIT 20
                """
            ).fetchall()
            events_by_rule = [{"rule": r[0], "count": r[1]} for r in rule_rows]

            # Events per day (last `days` days, grouped by date, split by severity)
            day_rows = conn.execute(
                """
                SELECT
                    strftime('%Y-%m-%d', e.created_at) AS day,
                    COUNT(DISTINCT e.id) AS total,
                    SUM(CASE WHEN json_extract(v.value,'$.severity')='critical' THEN 1 ELSE 0 END) AS critical,
                    SUM(CASE WHEN json_extract(v.value,'$.severity')='warning'  THEN 1 ELSE 0 END) AS warning,
                    SUM(CASE WHEN json_extract(v.value,'$.severity')='info'     THEN 1 ELSE 0 END) AS info
                FROM vigil_events e,
                     json_each(e.violations_json) v
                WHERE e.created_at >= datetime('now', ? || ' days')
                GROUP BY day
                ORDER BY day ASC
                """,
                (f"-{days}",),
            ).fetchall()
            events_per_day = [
                {
                    "date": r[0],
                    "total": r[1],
                    "critical": r[2] or 0,
                    "warning": r[3] or 0,
                    "info": r[4] or 0,
                }
                for r in day_rows
            ]

            # Events per hour (0-23)
            hour_rows = conn.execute(
                """
                SELECT CAST(strftime('%H', e.created_at) AS INTEGER) AS hour,
                       COUNT(*) AS cnt
                FROM vigil_events e
                GROUP BY hour
                ORDER BY cnt DESC
                """
            ).fetchall()
            top_violation_hours = [{"hour": r[0], "count": r[1]} for r in hour_rows]

        return {
            "total_sessions": total_sessions,
            "total_events": total_events,
            "events_by_severity": events_by_severity,
            "events_by_session": events_by_session,
            "events_by_rule": events_by_rule,
            "events_per_day": events_per_day,
            "top_violation_hours": top_violation_hours,
        }
