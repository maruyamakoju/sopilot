"""SQLite repository for VigilPilot camera group management.

Camera groups allow operators to:
1. Organise sessions (cameras) into logical groups (e.g. "Floor 1", "Entrance")
2. Apply a shared anomaly normality profile to all sessions in a group
3. Get a cross-camera overview of active violations per group
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


class CameraGroupRepository:
    """Data-access layer for the camera_groups table and related queries."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── CRUD ───────────────────────────────────────────────────────────────

    def create(
        self,
        name: str,
        description: str = "",
        anomaly_profile_name: str = "",
        location: str = "",
    ) -> dict:
        """Create a new camera group and return the created row."""
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO camera_groups
                  (name, description, anomaly_profile_name, location, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (name, description, anomaly_profile_name, location, now, now),
            )
            group_id = int(cur.lastrowid)  # type: ignore[arg-type]
            row = conn.execute(
                "SELECT * FROM camera_groups WHERE id = ?", (group_id,)
            ).fetchone()
            return dict(row)

    def get(self, group_id: int) -> dict | None:
        """Return a group by ID, or None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM camera_groups WHERE id = ?", (group_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_all(self) -> list[dict]:
        """Return all groups ordered by name."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM camera_groups ORDER BY name"
            ).fetchall()
        return [dict(r) for r in rows]

    def update(
        self,
        group_id: int,
        name: str | None = None,
        description: str | None = None,
        anomaly_profile_name: str | None = None,
        location: str | None = None,
    ) -> dict | None:
        """Update one or more fields on a group. Returns updated row or None."""
        now = datetime.now(UTC).isoformat()
        sets = ["updated_at = ?"]
        params: list = [now]
        if name is not None:
            sets.append("name = ?")
            params.append(name)
        if description is not None:
            sets.append("description = ?")
            params.append(description)
        if anomaly_profile_name is not None:
            sets.append("anomaly_profile_name = ?")
            params.append(anomaly_profile_name)
        if location is not None:
            sets.append("location = ?")
            params.append(location)
        params.append(group_id)

        with self._connect() as conn:
            conn.execute(
                f"UPDATE camera_groups SET {', '.join(sets)} WHERE id = ?",  # noqa: S608
                params,
            )
            row = conn.execute(
                "SELECT * FROM camera_groups WHERE id = ?", (group_id,)
            ).fetchone()
        return dict(row) if row else None

    def delete(self, group_id: int) -> bool:
        """Delete a group. Sessions are unlinked (group_id set to NULL)."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE vigil_sessions SET camera_group_id = NULL WHERE camera_group_id = ?",
                (group_id,),
            )
            cur = conn.execute(
                "DELETE FROM camera_groups WHERE id = ?", (group_id,)
            )
            return cur.rowcount > 0

    # ── Session membership ─────────────────────────────────────────────────

    def add_session(self, group_id: int, session_id: int) -> bool:
        """Add a vigil session to a group. Returns True on success."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE vigil_sessions SET camera_group_id = ? WHERE id = ?",
                (group_id, session_id),
            )
            return cur.rowcount > 0

    def remove_session(self, session_id: int) -> bool:
        """Remove a vigil session from its group."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE vigil_sessions SET camera_group_id = NULL WHERE id = ?",
                (session_id,),
            )
            return cur.rowcount > 0

    def list_sessions_in_group(self, group_id: int) -> list[dict]:
        """Return all vigil sessions belonging to a group."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT vs.*, cg.name as group_name
                FROM vigil_sessions vs
                JOIN camera_groups cg ON vs.camera_group_id = cg.id
                WHERE vs.camera_group_id = ?
                ORDER BY vs.created_at DESC
                """,
                (group_id,),
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["rules"] = json.loads(d.pop("rules_json", "[]"))
            result.append(d)
        return result

    # ── Cross-camera overview ──────────────────────────────────────────────

    def get_group_overview(self, group_id: int) -> dict:
        """Return aggregated violation stats for all sessions in the group."""
        with self._connect() as conn:
            group_row = conn.execute(
                "SELECT * FROM camera_groups WHERE id = ?", (group_id,)
            ).fetchone()
            if group_row is None:
                return {}

            sessions = conn.execute(
                """
                SELECT id, name, status, violation_count,
                       total_frames_analyzed, updated_at
                FROM vigil_sessions
                WHERE camera_group_id = ?
                ORDER BY name
                """,
                (group_id,),
            ).fetchall()

            total_sessions = len(sessions)
            active_sessions = sum(
                1 for s in sessions if s["status"] in ("running", "analyzing")
            )
            total_violations = sum(s["violation_count"] for s in sessions)
            total_frames = sum(s["total_frames_analyzed"] for s in sessions)

            # Recent events (last 10 across all sessions in group)
            session_ids = [s["id"] for s in sessions]
            recent_events: list[dict] = []
            if session_ids:
                placeholders = ",".join("?" * len(session_ids))
                event_rows = conn.execute(
                    f"""
                    SELECT ve.*, vs.name as session_name
                    FROM vigil_events ve
                    JOIN vigil_sessions vs ON ve.session_id = vs.id
                    WHERE ve.session_id IN ({placeholders})
                    ORDER BY ve.id DESC LIMIT 10
                    """,  # noqa: S608
                    session_ids,
                ).fetchall()
                for row in event_rows:
                    d = dict(row)
                    d["violations"] = json.loads(d.pop("violations_json", "[]"))
                    recent_events.append(d)

        return {
            "group": dict(group_row),
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "idle_sessions": total_sessions - active_sessions,
            "total_violations": total_violations,
            "total_frames_analyzed": total_frames,
            "sessions": [dict(s) for s in sessions],
            "recent_events": recent_events,
        }
