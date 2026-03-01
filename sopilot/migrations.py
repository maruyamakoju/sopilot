"""Versioned database migration framework for SOPilot.

Migrations are applied in order by version number.  Each migration is
wrapped in its own transaction so that a failure leaves the database in
a known state.  The ``schema_version`` table tracks which migrations
have already been applied, making ``run_migrations`` safe to call on
every application start.

This module intentionally uses ``sqlite3`` directly (instead of the
``Database`` helper class) to avoid circular imports -- ``Database``
calls ``run_migrations`` at the end of ``_init_schema()``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class Migration:
    """A single schema migration."""

    version: int
    description: str
    sql: str


# ------------------------------------------------------------------
# Migration registry -- append new migrations to the end of this list
# ------------------------------------------------------------------

MIGRATIONS: list[Migration] = [
    Migration(
        version=1,
        description="Initial schema: videos, clips, score_jobs, task_profiles, score_reviews",
        sql="""
        -- This migration is a no-op for existing databases since
        -- Database._init_schema() already creates these tables.
        -- It exists to establish the version baseline.
        SELECT 1;
        """,
    ),
    Migration(
        version=2,
        description="Add original_filename to videos table",
        sql="""
        -- Also a baseline migration; _ensure_column() handles this.
        SELECT 1;
        """,
    ),
    Migration(
        version=3,
        description="Add index on score_jobs(gold_video_id) for faster joins",
        sql="CREATE INDEX IF NOT EXISTS idx_score_jobs_gold ON score_jobs(gold_video_id);",
    ),
    Migration(
        version=4,
        description="Add index on score_jobs(trainee_video_id) for faster joins",
        sql="CREATE INDEX IF NOT EXISTS idx_score_jobs_trainee ON score_jobs(trainee_video_id);",
    ),
    Migration(
        version=5,
        description="VigilPilot: add vigil_sessions and vigil_events tables",
        sql="""
        CREATE TABLE IF NOT EXISTS vigil_sessions (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            name                  TEXT    NOT NULL,
            rules_json            TEXT    NOT NULL DEFAULT '[]',
            sample_fps            REAL    NOT NULL DEFAULT 1.0,
            severity_threshold    TEXT    NOT NULL DEFAULT 'warning',
            status                TEXT    NOT NULL DEFAULT 'idle',
            video_filename        TEXT,
            total_frames_analyzed INTEGER NOT NULL DEFAULT 0,
            violation_count       INTEGER NOT NULL DEFAULT 0,
            created_at            TEXT    NOT NULL,
            updated_at            TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS vigil_events (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id     INTEGER NOT NULL REFERENCES vigil_sessions(id) ON DELETE CASCADE,
            timestamp_sec  REAL    NOT NULL,
            frame_number   INTEGER NOT NULL,
            violations_json TEXT   NOT NULL DEFAULT '[]',
            frame_path     TEXT,
            created_at     TEXT    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_vigil_events_session
            ON vigil_events(session_id, timestamp_sec);
        """,
    ),
]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def _ensure_schema_version_table(conn: sqlite3.Connection) -> None:
    """Create the ``schema_version`` table if it does not exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL,
            description TEXT
        );
        """
    )
    conn.commit()


def _current_version(conn: sqlite3.Connection) -> int:
    """Return the highest migration version that has been applied, or 0."""
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    if row is None or row[0] is None:
        return 0
    return int(row[0])


def run_migrations(db_path: str) -> int:
    """Apply all pending migrations to the database at *db_path*.

    Parameters
    ----------
    db_path:
        File-system path to the SQLite database (as used by
        ``sqlite3.connect``).

    Returns
    -------
    int
        The number of migrations that were applied during this call.
    """
    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema_version_table(conn)
        current = _current_version(conn)

        applied = 0
        for migration in sorted(MIGRATIONS, key=lambda m: m.version):
            if migration.version <= current:
                continue

            # Each migration runs inside its own transaction.
            conn.execute("BEGIN")
            try:
                conn.executescript(migration.sql)
                conn.execute(
                    "INSERT INTO schema_version (version, applied_at, description) VALUES (?, ?, ?)",
                    (
                        migration.version,
                        datetime.now(UTC).isoformat(),
                        migration.description,
                    ),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

            applied += 1

        return applied
    finally:
        conn.close()
