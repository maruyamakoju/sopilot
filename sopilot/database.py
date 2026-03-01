"""Database facade -- backward-compatible entry point.

The ``Database`` class composes focused repository modules while
preserving its original public API.  All existing imports of
``from sopilot.database import Database`` continue to work unchanged.

Internally, operations are delegated to:
- ``VideoRepository`` -- video + clip CRUD
- ``ScoreRepository`` -- score job lifecycle + reviews
- ``TaskProfileRepository`` -- task profiles + SOP step definitions
- ``AdminRepository`` -- backup, vacuum, statistics
- ``ScoreAnalyticsRepository`` -- read-only analytics queries (unchanged)

Connection management (WAL init, schema creation, migrations) remains
in this module since it is infrastructure that repositories consume.
"""

import logging
import os
import sqlite3
import threading
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from sopilot.analytics_repository import ScoreAnalyticsRepository
from sopilot.migrations import run_migrations
from sopilot.repositories import (
    AdminRepository,
    ScoreRepository,
    TaskProfileRepository,
    VideoRepository,
)
from sopilot.types import (
    ClipRow,
    JoinedClipRow,
    ScoreJobInputRow,
    ScoreJobRow,
    ScoreReviewRow,
    TaskProfileRow,
    VideoListRow,
    VideoRow,
)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    """Backward-compatible facade composing repository modules.

    Every public method that existed before the refactor is preserved as
    a thin delegation to the appropriate repository, so callers such as
    ``sopilot.services`` and the test suite require zero changes.
    """

    # Class-level weak-ref registry of live instances for bulk cleanup (e.g. tests)
    _instances: weakref.WeakSet["Database"] = weakref.WeakSet()
    _instances_lock = threading.Lock()

    def __init__(self, path: Path, *, _pool_size: int | None = None) -> None:
        self.path = str(path)
        self._pool: list[sqlite3.Connection] = []
        self._pool_lock = threading.Lock()
        if _pool_size is not None:
            self._pool_size = _pool_size
        else:
            self._pool_size = int(os.environ.get("SOPILOT_DB_POOL_SIZE", "5"))
        with Database._instances_lock:
            Database._instances.add(self)

        # -- Infrastructure --------------------------------------------------
        self._enable_wal()
        self._init_schema()

        # -- Compose repositories ---------------------------------------------
        self._videos = VideoRepository(self.connect)
        self._scores = ScoreRepository(self.connect)
        self._task_profiles = TaskProfileRepository(self.connect)
        self._admin = AdminRepository(self.connect, self.path)
        self.analytics = ScoreAnalyticsRepository(self.connect)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _enable_wal(self) -> None:
        """Enable WAL mode for better concurrent read/write performance."""
        conn = sqlite3.connect(self.path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        finally:
            conn.close()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn: sqlite3.Connection | None = None
        # Try to reuse a pooled connection
        with self._pool_lock:
            if self._pool:
                conn = self._pool.pop()
        if conn is None:
            conn = sqlite3.connect(
                self.path, timeout=10.0,
                check_same_thread=(self._pool_size == 0),
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            # Return to pool if there is room; otherwise close
            returned = False
            with self._pool_lock:
                if len(self._pool) < self._pool_size:
                    self._pool.append(conn)
                    returned = True
            if not returned:
                conn.close()

    def close(self) -> None:
        """Close all pooled connections."""
        with self._pool_lock:
            while self._pool:
                try:
                    self._pool.pop().close()
                except Exception:
                    pass
        with Database._instances_lock:
            Database._instances.discard(self)
        logger.info("database connection pool closed")

    @classmethod
    def close_all(cls) -> None:
        """Close pooled connections on every tracked Database instance."""
        with cls._instances_lock:
            instances = list(cls._instances)
        for db in instances:
            db.close()

    def __del__(self) -> None:
        """Best-effort cleanup of pooled connections on garbage collection."""
        try:
            with self._pool_lock:
                while self._pool:
                    try:
                        self._pool.pop().close()
                    except Exception:
                        pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self.connect() as conn:
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
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT
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
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
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
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS score_reviews (
                    job_id INTEGER PRIMARY KEY,
                    verdict TEXT NOT NULL,
                    note TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES score_jobs(id)
                );

                CREATE INDEX IF NOT EXISTS idx_videos_task ON videos(task_id);
                CREATE INDEX IF NOT EXISTS idx_videos_created_at ON videos(created_at);
                CREATE INDEX IF NOT EXISTS idx_videos_operator ON videos(operator_id_hash);
                CREATE INDEX IF NOT EXISTS idx_videos_site ON videos(site_id);
                CREATE INDEX IF NOT EXISTS idx_clips_video ON clips(video_id);
                CREATE INDEX IF NOT EXISTS idx_score_jobs_status ON score_jobs(status);
                CREATE INDEX IF NOT EXISTS idx_score_jobs_gold ON score_jobs(gold_video_id);
                CREATE INDEX IF NOT EXISTS idx_score_jobs_trainee ON score_jobs(trainee_video_id);
                CREATE INDEX IF NOT EXISTS idx_score_jobs_created_at ON score_jobs(created_at);
                CREATE INDEX IF NOT EXISTS idx_score_jobs_status_created ON score_jobs(status, created_at);
                """
            )
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sop_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    name_ja TEXT NOT NULL DEFAULT '',
                    name_en TEXT NOT NULL DEFAULT '',
                    expected_duration_sec REAL,
                    min_duration_sec REAL,
                    max_duration_sec REAL,
                    is_critical INTEGER NOT NULL DEFAULT 0,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(task_id, step_index)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sop_steps_task ON sop_steps(task_id)"
            )
            self._ensure_column(conn, "score_jobs", "weights_json", "TEXT")
            self._ensure_column(conn, "score_jobs", "started_at", "TEXT")
            self._ensure_column(conn, "score_jobs", "finished_at", "TEXT")
            self._ensure_column(conn, "videos", "original_filename", "TEXT")

        run_migrations(self.path)

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, spec: str) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {row["name"] for row in rows}
        if column in existing:
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {spec}")

    # ==================================================================
    # Delegated Video operations
    # ==================================================================

    def insert_video(
        self,
        task_id: str,
        site_id: str | None,
        camera_id: str | None,
        operator_id_hash: str | None,
        recorded_at: str | None,
        is_gold: bool,
        original_filename: str | None = None,
    ) -> int:
        return self._videos.insert_video(
            task_id, site_id, camera_id, operator_id_hash, recorded_at, is_gold, original_filename
        )

    def finalize_video(
        self,
        video_id: int,
        file_path: str,
        step_boundaries: list[int],
        clips: list[dict[str, Any]],
        embedding_model: str,
    ) -> None:
        self._videos.finalize_video(video_id, file_path, step_boundaries, clips, embedding_model)

    def fail_video(self, video_id: int, error: str) -> None:
        self._videos.fail_video(video_id, error)

    def get_video(self, video_id: int) -> VideoRow | None:
        return self._videos.get_video(video_id)

    def list_videos(
        self,
        *,
        task_id: str | None = None,
        site_id: str | None = None,
        is_gold: bool | None = None,
        limit: int = 200,
    ) -> list[VideoListRow]:
        return self._videos.list_videos(task_id=task_id, site_id=site_id, is_gold=is_gold, limit=limit)

    def delete_video(self, video_id: int, *, force: bool = False) -> bool:
        return self._videos.delete_video(video_id, force=force)

    def update_video_metadata(
        self,
        video_id: int,
        *,
        site_id: str | None = None,
        camera_id: str | None = None,
        operator_id_hash: str | None = None,
        recorded_at: str | None = None,
    ) -> bool:
        return self._videos.update_video_metadata(
            video_id,
            site_id=site_id,
            camera_id=camera_id,
            operator_id_hash=operator_id_hash,
            recorded_at=recorded_at,
        )

    def count_videos(
        self,
        *,
        task_id: str | None = None,
        is_gold: bool | None = None,
        status: str | None = None,
    ) -> int:
        return self._videos.count_videos(task_id=task_id, is_gold=is_gold, status=status)

    def count_videos_by_site(self, *, task_id: str | None = None) -> dict[str, int]:
        return self._videos.count_videos_by_site(task_id=task_id)

    # -- Clips --

    def get_gold_version(self, video_id: int, task_id: str) -> int | None:
        return self._videos.get_gold_version(video_id, task_id)

    def get_video_clips(self, video_id: int) -> list[ClipRow]:
        return self._videos.get_video_clips(video_id)

    def get_clip(self, video_id: int, clip_index: int) -> JoinedClipRow | None:
        return self._videos.get_clip(video_id, clip_index)

    def iter_clips(
        self,
        task_id: str | None = None,
        exclude_video_id: int | None = None,
    ) -> list[JoinedClipRow]:
        return self._videos.iter_clips(task_id, exclude_video_id)

    # ==================================================================
    # Delegated Score operations
    # ==================================================================

    def create_score_job(
        self,
        gold_video_id: int,
        trainee_video_id: int,
        weights: dict[str, float] | None = None,
    ) -> int:
        return self._scores.create_score_job(gold_video_id, trainee_video_id, weights)

    def list_pending_score_job_ids(self) -> list[int]:
        return self._scores.list_pending_score_job_ids()

    def claim_score_job(self, job_id: int) -> bool:
        return self._scores.claim_score_job(job_id)

    def get_score_job_input(self, job_id: int) -> ScoreJobInputRow | None:
        return self._scores.get_score_job_input(job_id)

    def complete_score_job(self, job_id: int, score_payload: dict[str, Any]) -> None:
        self._scores.complete_score_job(job_id, score_payload)

    def reset_score_job_for_retry(self, job_id: int) -> None:
        self._scores.reset_score_job_for_retry(job_id)

    def cancel_score_job(self, job_id: int) -> bool:
        return self._scores.cancel_score_job(job_id)

    def fail_score_job(self, job_id: int, error: str) -> None:
        self._scores.fail_score_job(job_id, error)

    def get_score_job(self, job_id: int) -> ScoreJobRow | None:
        return self._scores.get_score_job(job_id)

    def list_completed_score_jobs(self, task_id: str | None = None) -> list[dict[str, Any]]:
        return self._scores.list_completed_score_jobs(task_id)

    def list_score_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return score jobs with lightweight score summary (no full score_json)."""
        return self._scores.list_score_jobs(status=status, limit=limit, offset=offset)

    def update_score_json(self, job_id: int, score_payload: dict[str, Any]) -> None:
        self._scores.update_score_json(job_id, score_payload)

    def count_score_jobs(self, *, status: str | None = None) -> int:
        return self._scores.count_score_jobs(status=status)

    def get_operator_scores_chronological(
        self,
        operator_id: str,
        *,
        task_id: str | None = None,
        limit: int = 50,
    ) -> list[float]:
        """Return chronological list of completed scores for an operator (oldest first)."""
        return self._scores.get_operator_scores_chronological(
            operator_id, task_id=task_id, limit=limit
        )

    # -- Score reviews --

    def upsert_score_review(self, job_id: int, verdict: str, note: str | None) -> ScoreReviewRow:
        return self._scores.upsert_score_review(job_id, verdict, note)

    def get_score_review(self, job_id: int) -> ScoreReviewRow | None:
        return self._scores.get_score_review(job_id)

    # ==================================================================
    # Delegated Score analytics
    # ==================================================================

    def get_analytics(self, *, task_id: str | None = None, days: int | None = None) -> dict:
        """Return aggregate scoring analytics for dashboard visualization."""
        return self.analytics.get_analytics(task_id=task_id, days=days)

    def get_operator_trend(
        self,
        operator_id: str,
        *,
        task_id: str | None = None,
        limit: int = 20,
    ) -> dict:
        """Return per-job score history and trend statistics for a single operator."""
        return self.analytics.get_operator_trend(operator_id, task_id=task_id, limit=limit)

    def get_step_performance(
        self,
        *,
        task_id: str | None = None,
        days: int | None = None,
    ) -> dict:
        """Return per-step deviation frequency statistics across all completed jobs."""
        return self.analytics.get_step_performance(task_id=task_id, days=days)

    def get_compliance_overview(
        self,
        *,
        task_id: str | None = None,
        days: int | None = None,
    ) -> dict:
        """Return compliance rate overview, site breakdown, and operator rankings."""
        return self.analytics.get_compliance_overview(task_id=task_id, days=days)

    # ==================================================================
    # Delegated Task profile / SOP step operations
    # ==================================================================

    def upsert_task_profile(
        self,
        *,
        task_id: str,
        task_name: str,
        pass_score: float,
        retrain_score: float,
        default_weights: dict[str, float],
        deviation_policy: dict[str, str],
    ) -> None:
        self._task_profiles.upsert_task_profile(
            task_id=task_id,
            task_name=task_name,
            pass_score=pass_score,
            retrain_score=retrain_score,
            default_weights=default_weights,
            deviation_policy=deviation_policy,
        )

    def get_task_profile(self, task_id: str) -> TaskProfileRow | None:
        return self._task_profiles.get_task_profile(task_id)

    def upsert_sop_steps(self, task_id: str, steps: list[dict]) -> int:
        """Insert or replace step definitions for a task. Returns count upserted."""
        return self._task_profiles.upsert_sop_steps(task_id, steps)

    def get_sop_steps(self, task_id: str) -> list[dict]:
        """Return all step definitions for a task, ordered by step_index."""
        return self._task_profiles.get_sop_steps(task_id)

    def delete_sop_steps(self, task_id: str) -> int:
        """Delete all step definitions for a task. Returns count deleted."""
        return self._task_profiles.delete_sop_steps(task_id)

    # ==================================================================
    # Delegated Admin operations
    # ==================================================================

    def backup(self, dest_path: str) -> None:
        """Create a hot backup of the database using SQLite's backup API."""
        self._admin.backup(dest_path)

    def vacuum(self) -> None:
        """Reclaim disk space and rebuild indices."""
        self._admin.vacuum()

    def get_stats(self) -> dict[str, Any]:
        """Return table row counts and database file size for monitoring."""
        return self._admin.get_stats()
