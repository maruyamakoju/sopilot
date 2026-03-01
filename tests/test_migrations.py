"""Tests for sopilot.migrations — versioned schema migration framework."""

import sqlite3
from pathlib import Path

from sopilot.migrations import MIGRATIONS, run_migrations

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path: Path) -> str:
    """Return a path string for a fresh SQLite database inside *tmp_path*."""
    return str(tmp_path / "test_migrations.db")


def _bootstrap_tables(db_path: str) -> None:
    """Create the core SOPilot tables so that migrations referencing them
    (e.g. CREATE INDEX on score_jobs) will succeed on an otherwise empty
    database.  This mirrors what ``Database._init_schema()`` does before
    ``run_migrations`` is called.
    """
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            site_id TEXT,
            camera_id TEXT,
            operator_id_hash TEXT,
            recorded_at TEXT,
            is_gold INTEGER NOT NULL DEFAULT 0,
            file_path TEXT,
            status TEXT NOT NULL DEFAULT 'processing',
            clip_count INTEGER NOT NULL DEFAULT 0,
            step_boundaries_json TEXT NOT NULL DEFAULT '[]',
            embedding_model TEXT NOT NULL DEFAULT 'color-motion-v1',
            created_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT '',
            error TEXT,
            original_filename TEXT
        );

        CREATE TABLE IF NOT EXISTS clips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL,
            clip_index INTEGER NOT NULL,
            start_sec REAL NOT NULL,
            end_sec REAL NOT NULL,
            embedding_json TEXT NOT NULL,
            quality_flag TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(video_id) REFERENCES videos(id),
            UNIQUE(video_id, clip_index)
        );

        CREATE TABLE IF NOT EXISTS score_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gold_video_id INTEGER NOT NULL,
            trainee_video_id INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            score_json TEXT,
            weights_json TEXT,
            created_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT '',
            started_at TEXT,
            finished_at TEXT,
            error TEXT
        );

        CREATE TABLE IF NOT EXISTS task_profiles (
            task_id TEXT PRIMARY KEY,
            task_name TEXT NOT NULL,
            pass_score REAL NOT NULL,
            retrain_score REAL NOT NULL,
            default_weights_json TEXT NOT NULL,
            deviation_policy_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS score_reviews (
            job_id INTEGER PRIMARY KEY,
            verdict TEXT NOT NULL,
            note TEXT,
            created_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT '',
            FOREIGN KEY(job_id) REFERENCES score_jobs(id)
        );
        """
    )
    conn.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunMigrationsFreshDB:
    """Running migrations on a fresh database with bootstrapped tables."""

    def test_creates_schema_version_table(self, tmp_path):
        db_path = _make_db(tmp_path)
        _bootstrap_tables(db_path)
        run_migrations(db_path)

        conn = sqlite3.connect(db_path)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "schema_version" in tables

    def test_applies_all_migrations(self, tmp_path):
        db_path = _make_db(tmp_path)
        _bootstrap_tables(db_path)
        applied = run_migrations(db_path)
        assert applied == len(MIGRATIONS)

    def test_records_every_version(self, tmp_path):
        db_path = _make_db(tmp_path)
        _bootstrap_tables(db_path)
        run_migrations(db_path)

        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT version FROM schema_version ORDER BY version"
        ).fetchall()
        conn.close()
        recorded_versions = [row[0] for row in rows]
        expected_versions = sorted(m.version for m in MIGRATIONS)
        assert recorded_versions == expected_versions


class TestIdempotency:
    """Running migrations multiple times must be safe."""

    def test_second_run_applies_zero(self, tmp_path):
        db_path = _make_db(tmp_path)
        _bootstrap_tables(db_path)
        first = run_migrations(db_path)
        second = run_migrations(db_path)
        assert first == len(MIGRATIONS)
        assert second == 0

    def test_schema_version_rows_unchanged_on_rerun(self, tmp_path):
        db_path = _make_db(tmp_path)
        _bootstrap_tables(db_path)
        run_migrations(db_path)

        conn = sqlite3.connect(db_path)
        rows_before = conn.execute("SELECT * FROM schema_version").fetchall()
        conn.close()

        run_migrations(db_path)

        conn = sqlite3.connect(db_path)
        rows_after = conn.execute("SELECT * FROM schema_version").fetchall()
        conn.close()

        assert rows_before == rows_after


class TestVersionTracking:
    """Version tracking correctly identifies current version and applies
    only newer migrations."""

    def test_current_version_after_full_run(self, tmp_path):
        db_path = _make_db(tmp_path)
        _bootstrap_tables(db_path)
        run_migrations(db_path)

        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        conn.close()
        assert row[0] == max(m.version for m in MIGRATIONS)

    def test_partial_run_then_full(self, tmp_path):
        """Simulate a database that already has version 2 applied, then
        run the full migration list -- only versions 3+ should apply."""
        db_path = _make_db(tmp_path)
        _bootstrap_tables(db_path)

        # Manually insert versions 1 and 2 into schema_version
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL,
                description TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO schema_version (version, applied_at, description) VALUES (1, '2026-01-01T00:00:00+00:00', 'seed')"
        )
        conn.execute(
            "INSERT INTO schema_version (version, applied_at, description) VALUES (2, '2026-01-01T00:00:00+00:00', 'seed')"
        )
        conn.commit()
        conn.close()

        applied = run_migrations(db_path)

        # Only migrations with version > 2 should be applied
        expected = len([m for m in MIGRATIONS if m.version > 2])
        assert applied == expected

    def test_applied_at_is_populated(self, tmp_path):
        db_path = _make_db(tmp_path)
        _bootstrap_tables(db_path)
        run_migrations(db_path)

        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT applied_at FROM schema_version").fetchall()
        conn.close()
        for row in rows:
            # Every applied_at value should be a non-empty ISO timestamp string
            assert row[0] is not None
            assert len(row[0]) > 0

    def test_description_is_stored(self, tmp_path):
        db_path = _make_db(tmp_path)
        _bootstrap_tables(db_path)
        run_migrations(db_path)

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT version, description FROM schema_version ORDER BY version"
        ).fetchall()
        conn.close()

        desc_map = {row["version"]: row["description"] for row in rows}
        for migration in MIGRATIONS:
            assert desc_map[migration.version] == migration.description


class TestMigrationEffects:
    """Verify that non-no-op migrations actually change the schema."""

    def test_index_on_gold_video_id_created(self, tmp_path):
        db_path = _make_db(tmp_path)
        _bootstrap_tables(db_path)
        run_migrations(db_path)

        conn = sqlite3.connect(db_path)
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        conn.close()
        assert "idx_score_jobs_gold" in indexes

    def test_index_on_trainee_video_id_created(self, tmp_path):
        db_path = _make_db(tmp_path)
        _bootstrap_tables(db_path)
        run_migrations(db_path)

        conn = sqlite3.connect(db_path)
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        conn.close()
        assert "idx_score_jobs_trainee" in indexes
