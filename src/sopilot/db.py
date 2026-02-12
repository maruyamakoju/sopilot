from __future__ import annotations

import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class VideoCreateInput:
    task_id: str
    role: str
    file_path: str
    embedding_model: str
    site_id: str | None = None
    camera_id: str | None = None
    operator_id_hash: str | None = None


_KNOWN_TABLES = frozenset({"videos", "clips", "ingest_jobs", "score_jobs", "training_jobs"})


class Database:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    _DDL_RE = re.compile(r"^[A-Z]+(\s+DEFAULT\s+'[^']*')?$", re.IGNORECASE)

    def _ensure_column(self, table: str, column: str, ddl: str) -> None:
        if table not in _KNOWN_TABLES:
            raise ValueError(f"unknown table: {table}")
        if not column.isidentifier():
            raise ValueError(f"invalid column name: {column}")
        if not self._DDL_RE.match(ddl):
            raise ValueError(f"invalid column ddl: {ddl}")
        cur = self._conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        cols = {row["name"] for row in cur.fetchall()}
        if column not in cols:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    site_id TEXT,
                    camera_id TEXT,
                    operator_id_hash TEXT,
                    file_path TEXT NOT NULL,
                    raw_embedding_path TEXT,
                    embedding_path TEXT,
                    clip_meta_path TEXT,
                    embedding_model TEXT DEFAULT '',
                    num_clips INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._ensure_column("videos", "raw_embedding_path", "TEXT")
            self._ensure_column("videos", "embedding_model", "TEXT DEFAULT ''")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS clips (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER NOT NULL,
                    task_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    clip_idx INTEGER NOT NULL,
                    start_sec REAL NOT NULL,
                    end_sec REAL NOT NULL,
                    quality_flags TEXT,
                    FOREIGN KEY(video_id) REFERENCES videos(id)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ingest_jobs (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    requested_by TEXT,
                    file_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    site_id TEXT,
                    camera_id TEXT,
                    operator_id_hash TEXT,
                    status TEXT NOT NULL,
                    video_id INTEGER,
                    num_clips INTEGER,
                    source_fps REAL,
                    sampled_fps REAL,
                    embedding_model TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    queued_at TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    FOREIGN KEY(video_id) REFERENCES videos(id)
                )
                """
            )
            self._ensure_column("ingest_jobs", "queued_at", "TEXT")
            self._ensure_column("ingest_jobs", "started_at", "TEXT")
            self._ensure_column("ingest_jobs", "finished_at", "TEXT")
            self._ensure_column("ingest_jobs", "requested_by", "TEXT")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS score_jobs (
                    id TEXT PRIMARY KEY,
                    gold_video_id INTEGER NOT NULL,
                    trainee_video_id INTEGER NOT NULL,
                    requested_by TEXT,
                    status TEXT NOT NULL,
                    score REAL,
                    result_path TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    queued_at TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    FOREIGN KEY(gold_video_id) REFERENCES videos(id),
                    FOREIGN KEY(trainee_video_id) REFERENCES videos(id)
                )
                """
            )
            self._ensure_column("score_jobs", "queued_at", "TEXT")
            self._ensure_column("score_jobs", "started_at", "TEXT")
            self._ensure_column("score_jobs", "finished_at", "TEXT")
            self._ensure_column("score_jobs", "requested_by", "TEXT")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS training_jobs (
                    id TEXT PRIMARY KEY,
                    trigger TEXT NOT NULL,
                    requested_by TEXT,
                    status TEXT NOT NULL,
                    summary_path TEXT,
                    metrics_json TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    queued_at TEXT,
                    started_at TEXT,
                    finished_at TEXT
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_videos_task ON videos(task_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_videos_created ON videos(created_at)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_clips_video ON clips(video_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_clips_task ON clips(task_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ingest_jobs_status ON ingest_jobs(status)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_score_jobs_status ON score_jobs(status)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status)")
            self._ensure_column("training_jobs", "requested_by", "TEXT")
            self._conn.commit()

    def create_video(self, payload: VideoCreateInput) -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO videos (
                    task_id, role, site_id, camera_id, operator_id_hash,
                    file_path, raw_embedding_path, embedding_path, clip_meta_path,
                    embedding_model, num_clips, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, '', '', '', ?, 0, ?)
                """,
                (
                    payload.task_id,
                    payload.role,
                    payload.site_id,
                    payload.camera_id,
                    payload.operator_id_hash,
                    payload.file_path,
                    payload.embedding_model,
                    self._now(),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def finalize_video(
        self,
        video_id: int,
        raw_embedding_path: str,
        embedding_path: str,
        clip_meta_path: str,
        embedding_model: str,
        num_clips: int,
    ) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                UPDATE videos
                SET raw_embedding_path = ?,
                    embedding_path = ?,
                    clip_meta_path = ?,
                    embedding_model = ?,
                    num_clips = ?
                WHERE id = ?
                """,
                (
                    raw_embedding_path,
                    embedding_path,
                    clip_meta_path,
                    embedding_model,
                    num_clips,
                    video_id,
                ),
            )
            self._conn.commit()

    def update_video_embedding(
        self,
        video_id: int,
        embedding_path: str,
        embedding_model: str,
    ) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                UPDATE videos
                SET embedding_path = ?, embedding_model = ?
                WHERE id = ?
                """,
                (embedding_path, embedding_model, video_id),
            )
            self._conn.commit()

    def add_clips(self, video_id: int, task_id: str, role: str, rows: list[dict]) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.executemany(
                """
                INSERT INTO clips (
                    video_id, task_id, role, clip_idx, start_sec, end_sec, quality_flags
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        video_id,
                        task_id,
                        role,
                        int(row["clip_idx"]),
                        float(row["start_sec"]),
                        float(row["end_sec"]),
                        row.get("quality_flags", ""),
                    )
                    for row in rows
                ],
            )
            self._conn.commit()

    def get_video(self, video_id: int) -> dict | None:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def create_ingest_job(
        self,
        *,
        task_id: str,
        role: str,
        requested_by: str | None,
        file_name: str,
        file_path: str,
        site_id: str | None,
        camera_id: str | None,
        operator_id_hash: str | None,
    ) -> str:
        job_id = str(uuid.uuid4())
        now = self._now()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO ingest_jobs (
                    id, task_id, role, requested_by, file_name, file_path, site_id, camera_id,
                    operator_id_hash, status, video_id, num_clips, source_fps, sampled_fps,
                    embedding_model, error_message, created_at, updated_at, queued_at, started_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', NULL, NULL, NULL, NULL, NULL, NULL, ?, ?, ?, NULL, NULL)
                """,
                (
                    job_id,
                    task_id,
                    role,
                    requested_by,
                    file_name,
                    file_path,
                    site_id,
                    camera_id,
                    operator_id_hash,
                    now,
                    now,
                    now,
                ),
            )
            self._conn.commit()
        return job_id

    def _mark_job_running(self, table: str, job_id: str) -> None:
        if table not in _KNOWN_TABLES:
            raise ValueError(f"unknown table: {table}")
        now = self._now()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"UPDATE {table} SET status = 'running', started_at = ?, updated_at = ?, error_message = NULL WHERE id = ?",
                (now, now, job_id),
            )
            self._conn.commit()

    def _fail_job(self, table: str, job_id: str, message: str) -> None:
        if table not in _KNOWN_TABLES:
            raise ValueError(f"unknown table: {table}")
        now = self._now()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"UPDATE {table} SET status = 'failed', error_message = ?, updated_at = ?, finished_at = ? WHERE id = ?",
                (message, now, now, job_id),
            )
            self._conn.commit()

    def _get_job(self, table: str, job_id: str) -> dict | None:
        if table not in _KNOWN_TABLES:
            raise ValueError(f"unknown table: {table}")
        cur = self._conn.cursor()
        cur.execute(f"SELECT * FROM {table} WHERE id = ?", (job_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def mark_ingest_job_running(self, job_id: str) -> None:
        self._mark_job_running("ingest_jobs", job_id)

    def complete_ingest_job(
        self,
        *,
        job_id: str,
        video_id: int,
        num_clips: int,
        source_fps: float,
        sampled_fps: float,
        embedding_model: str,
    ) -> None:
        now = self._now()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                UPDATE ingest_jobs
                SET status = 'completed',
                    video_id = ?,
                    num_clips = ?,
                    source_fps = ?,
                    sampled_fps = ?,
                    embedding_model = ?,
                    updated_at = ?,
                    finished_at = ?
                WHERE id = ?
                """,
                (
                    int(video_id),
                    int(num_clips),
                    float(source_fps),
                    float(sampled_fps),
                    embedding_model,
                    now,
                    now,
                    job_id,
                ),
            )
            self._conn.commit()

    def fail_ingest_job(self, job_id: str, message: str) -> None:
        self._fail_job("ingest_jobs", job_id, message)

    def get_ingest_job(self, job_id: str) -> dict | None:
        return self._get_job("ingest_jobs", job_id)

    def list_videos(self, task_id: str | None = None, limit: int = 50) -> list[dict]:
        cur = self._conn.cursor()
        lim = max(1, min(int(limit), 500))
        if task_id:
            cur.execute(
                """
                SELECT id, task_id, role, site_id, camera_id, num_clips, created_at, embedding_model
                FROM videos
                WHERE task_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (task_id, lim),
            )
        else:
            cur.execute(
                """
                SELECT id, task_id, role, site_id, camera_id, num_clips, created_at, embedding_model
                FROM videos
                ORDER BY id DESC
                LIMIT ?
                """,
                (lim,),
            )
        return [dict(r) for r in cur.fetchall()]

    def list_videos_with_artifacts(self) -> list[dict]:
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT id, task_id, role, embedding_path, raw_embedding_path, clip_meta_path
            FROM videos
            WHERE embedding_path IS NOT NULL AND embedding_path != ''
            """
        )
        return [dict(r) for r in cur.fetchall()]

    def list_task_videos_with_artifacts(self, task_id: str) -> list[dict]:
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT id, task_id, role, embedding_path, raw_embedding_path, clip_meta_path
            FROM videos
            WHERE task_id = ?
              AND embedding_path IS NOT NULL
              AND embedding_path != ''
              AND clip_meta_path IS NOT NULL
              AND clip_meta_path != ''
            ORDER BY id ASC
            """,
            (task_id,),
        )
        return [dict(r) for r in cur.fetchall()]

    def list_videos_for_training(self, since_created_at: str | None = None) -> list[dict]:
        cur = self._conn.cursor()
        if since_created_at:
            cur.execute(
                """
                SELECT id, task_id, role, file_path, raw_embedding_path, embedding_path, created_at
                FROM videos
                WHERE embedding_path IS NOT NULL
                  AND embedding_path != ''
                  AND created_at > ?
                ORDER BY created_at ASC
                """,
                (since_created_at,),
            )
        else:
            cur.execute(
                """
                SELECT id, task_id, role, file_path, raw_embedding_path, embedding_path, created_at
                FROM videos
                WHERE embedding_path IS NOT NULL
                  AND embedding_path != ''
                ORDER BY created_at ASC
                """
            )
        return [dict(r) for r in cur.fetchall()]

    def count_videos_since(self, since_created_at: str | None) -> int:
        cur = self._conn.cursor()
        if since_created_at:
            cur.execute("SELECT COUNT(*) AS c FROM videos WHERE created_at > ?", (since_created_at,))
        else:
            cur.execute("SELECT COUNT(*) AS c FROM videos")
        row = cur.fetchone()
        return int(row["c"]) if row else 0

    def create_score_job(
        self,
        gold_video_id: int,
        trainee_video_id: int,
        requested_by: str | None,
    ) -> str:
        job_id = str(uuid.uuid4())
        now = self._now()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO score_jobs (
                    id, gold_video_id, trainee_video_id, requested_by, status, score, result_path,
                    error_message, created_at, updated_at, queued_at, started_at, finished_at
                ) VALUES (?, ?, ?, ?, 'queued', NULL, NULL, NULL, ?, ?, ?, NULL, NULL)
                """,
                (job_id, gold_video_id, trainee_video_id, requested_by, now, now, now),
            )
            self._conn.commit()
        return job_id

    def mark_score_job_running(self, job_id: str) -> None:
        self._mark_job_running("score_jobs", job_id)

    def complete_score_job(self, job_id: str, score: float, result_path: str) -> None:
        now = self._now()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                UPDATE score_jobs
                SET status = 'completed',
                    score = ?,
                    result_path = ?,
                    updated_at = ?,
                    finished_at = ?
                WHERE id = ?
                """,
                (float(score), result_path, now, now, job_id),
            )
            self._conn.commit()

    def fail_score_job(self, job_id: str, message: str) -> None:
        self._fail_job("score_jobs", job_id, message)

    def get_score_job(self, job_id: str) -> dict | None:
        return self._get_job("score_jobs", job_id)

    def create_training_job(self, trigger: str, requested_by: str | None) -> str:
        job_id = str(uuid.uuid4())
        now = self._now()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO training_jobs (
                    id, trigger, requested_by, status, summary_path, metrics_json, error_message,
                    created_at, updated_at, queued_at, started_at, finished_at
                ) VALUES (?, ?, ?, 'queued', NULL, NULL, NULL, ?, ?, ?, NULL, NULL)
                """,
                (job_id, trigger, requested_by, now, now, now),
            )
            self._conn.commit()
        return job_id

    def mark_training_job_running(self, job_id: str) -> None:
        self._mark_job_running("training_jobs", job_id)

    def complete_training_job(
        self,
        job_id: str,
        status: str,
        summary_path: str,
        metrics_json: str,
    ) -> None:
        now = self._now()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                UPDATE training_jobs
                SET status = ?, summary_path = ?, metrics_json = ?, updated_at = ?, finished_at = ?
                WHERE id = ?
                """,
                (status, summary_path, metrics_json, now, now, job_id),
            )
            self._conn.commit()

    def fail_training_job(self, job_id: str, message: str) -> None:
        self._fail_job("training_jobs", job_id, message)

    def get_training_job(self, job_id: str) -> dict | None:
        return self._get_job("training_jobs", job_id)

    def has_active_training_job(self) -> bool:
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT 1 AS ok
            FROM training_jobs
            WHERE status IN ('queued', 'running')
            LIMIT 1
            """
        )
        row = cur.fetchone()
        return row is not None

    def latest_training_finished_at(self) -> str | None:
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT finished_at
            FROM training_jobs
            WHERE status = 'completed'
            ORDER BY finished_at DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        if not row:
            return None
        return row["finished_at"]

    def delete_video(self, video_id: int) -> dict | None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
            row = cur.fetchone()
            if not row:
                return None
            payload = dict(row)
            cur.execute("DELETE FROM clips WHERE video_id = ?", (video_id,))
            cur.execute("DELETE FROM videos WHERE id = ?", (video_id,))
            self._conn.commit()
            return payload

    def list_audit_trail(self, limit: int = 100) -> list[dict]:
        lim = max(1, min(int(limit), 1000))
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM (
                SELECT
                    id AS job_id,
                    'ingest' AS job_type,
                    requested_by,
                    task_id,
                    CAST(video_id AS TEXT) AS subject,
                    status,
                    embedding_model AS model_name,
                    NULL AS score,
                    error_message,
                    queued_at,
                    started_at,
                    finished_at,
                    created_at
                FROM ingest_jobs
                UNION ALL
                SELECT
                    id AS job_id,
                    'score' AS job_type,
                    requested_by,
                    NULL AS task_id,
                    CAST(gold_video_id AS TEXT) || '->' || CAST(trainee_video_id AS TEXT) AS subject,
                    status,
                    NULL AS model_name,
                    score,
                    error_message,
                    queued_at,
                    started_at,
                    finished_at,
                    created_at
                FROM score_jobs
                UNION ALL
                SELECT
                    id AS job_id,
                    'training' AS job_type,
                    requested_by,
                    NULL AS task_id,
                    trigger AS subject,
                    status,
                    NULL AS model_name,
                    NULL AS score,
                    error_message,
                    queued_at,
                    started_at,
                    finished_at,
                    created_at
                FROM training_jobs
            )
            ORDER BY COALESCE(finished_at, started_at, queued_at, created_at) DESC
            LIMIT ?
            """,
            (lim,),
        )
        return [dict(r) for r in cur.fetchall()]

    def job_status_counts(self) -> dict[str, dict[str, int]]:
        def _counts(table: str) -> dict[str, int]:
            if table not in _KNOWN_TABLES:
                raise ValueError(f"unknown table: {table}")
            cur = self._conn.cursor()
            cur.execute(
                f"""
                SELECT status, COUNT(*) AS c
                FROM {table}
                GROUP BY status
                """
            )
            rows = cur.fetchall()
            out: dict[str, int] = {}
            total = 0
            for row in rows:
                status = str(row["status"])
                count = int(row["c"])
                out[status] = count
                total += count
            out["total"] = total
            return out

        return {
            "ingest": _counts("ingest_jobs"),
            "score": _counts("score_jobs"),
            "training": _counts("training_jobs"),
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
