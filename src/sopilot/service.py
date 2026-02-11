from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path

import numpy as np

from .audit_service import AuditService
from .config import Settings
from .db import Database
from .embedding_manager import EmbeddingManager
from .embeddings import build_embedder
from .ingest_service import IngestService
from .queueing import InlineQueueManager, QueueManager, RqQueueManager
from .scoring_service import ScoringService
from .training_service import TrainingService
from .vector_index import NpyVectorIndex


logger = logging.getLogger(__name__)


class SopilotService:
    def __init__(self, settings: Settings, db: Database, runtime_mode: str = "api") -> None:
        self.settings = settings
        self.db = db
        self.runtime_mode = runtime_mode.strip().lower()
        self.embedder = build_embedder(settings)
        self.index = NpyVectorIndex(settings.index_dir)

        # Shared embedding / adapter manager
        self._embedding_mgr = EmbeddingManager(settings, self.embedder)

        # Queue
        self._queue: QueueManager | None = self._build_queue_manager()

        # Domain services
        self._ingest = IngestService(settings, db, self.index, self._embedding_mgr, self._queue)
        self._scoring = ScoringService(settings, db, self.index, self._embedding_mgr, self._queue)
        self._training = TrainingService(settings, db, self.index, self._embedding_mgr, self._queue)
        self._audit = AuditService(settings, db)

        # Nightly scheduler
        if self.runtime_mode == "api":
            self._training.start_nightly_scheduler_if_enabled(
                enqueue_fn=self._training.enqueue_training,
            )

    def _build_queue_manager(self) -> QueueManager | None:
        if self.runtime_mode not in {"api", "watcher"}:
            return None

        backend = self.settings.queue_backend.strip().lower()
        if backend == "inline":
            return InlineQueueManager(
                ingest_handler=self.run_ingest_job,
                score_handler=self.run_score_job,
                training_handler=self.run_training_job,
            )
        if backend == "rq":
            return RqQueueManager(
                redis_url=self.settings.redis_url,
                queue_prefix=self.settings.rq_queue_prefix,
                timeout_sec=self.settings.rq_job_timeout_sec,
                result_ttl_sec=self.settings.rq_result_ttl_sec,
                failure_ttl_sec=self.settings.rq_failure_ttl_sec,
                retry_max=self.settings.rq_retry_max,
            )
        raise ValueError(f"unsupported queue backend: {self.settings.queue_backend}")

    def shutdown(self) -> None:
        self._training.stop_scheduler()
        if self._queue is not None:
            try:
                self._queue.close()
            except Exception:
                logger.exception("failed to close queue manager")
        try:
            self.db.close()
        except Exception:
            logger.exception("Failed to close database cleanly")

    # -- Adapter (delegation to EmbeddingManager) --

    def _load_current_adapter(self) -> tuple[np.ndarray, np.ndarray] | None:
        return self._embedding_mgr.load_adapter()

    def _apply_feature_adapter(self, embeddings: np.ndarray) -> np.ndarray:
        return self._embedding_mgr.apply_adapter(embeddings)

    def _active_embedding_model_name(self) -> str:
        return self._embedding_mgr.active_model_name()

    def _embed_batch(self, clips: list) -> tuple[np.ndarray, np.ndarray]:
        return self._embedding_mgr.embed_batch(clips)

    # -- Ingest --

    def enqueue_ingest(self, **kwargs) -> dict:
        return self._ingest.enqueue_ingest(**kwargs)

    def enqueue_ingest_from_path(self, **kwargs) -> dict:
        return self._ingest.enqueue_ingest_from_path(**kwargs)

    def run_ingest_job(self, job_id: str) -> None:
        self._ingest.run_ingest_job(job_id)

    def get_ingest_job(self, ingest_job_id: str) -> dict | None:
        return self._ingest.get_ingest_job(ingest_job_id)

    def _rebuild_task_index(self, task_id: str, preferred_dim: int | None = None) -> int:
        return self._ingest.rebuild_task_index(task_id, preferred_dim)

    # -- Video CRUD --

    def get_video_info(self, video_id: int) -> dict | None:
        row = self.db.get_video(video_id)
        if row is None:
            return None
        return {
            "video_id": int(row["id"]),
            "task_id": row["task_id"],
            "role": row["role"],
            "site_id": row.get("site_id"),
            "camera_id": row.get("camera_id"),
            "num_clips": int(row.get("num_clips", 0) or 0),
            "embedding_model": row.get("embedding_model") or "",
            "created_at": row.get("created_at"),
        }

    def list_videos(self, task_id: str | None, limit: int) -> list[dict]:
        rows = self.db.list_videos(task_id=task_id, limit=limit)
        return [
            {
                "video_id": int(row["id"]),
                "task_id": row["task_id"],
                "role": row["role"],
                "site_id": row.get("site_id"),
                "camera_id": row.get("camera_id"),
                "num_clips": int(row.get("num_clips", 0) or 0),
                "embedding_model": row.get("embedding_model") or "",
                "created_at": row.get("created_at"),
            }
            for row in rows
        ]

    def get_video_file_path(self, video_id: int) -> Path:
        row = self.db.get_video(video_id)
        if row is None:
            raise ValueError(f"video not found: {video_id}")
        path = Path(row["file_path"])
        if not path.exists():
            raise ValueError(f"video file missing on disk: {path}")
        return path

    def delete_video(self, video_id: int) -> dict:
        row = self.db.delete_video(video_id)
        if row is None:
            raise ValueError(f"video not found: {video_id}")

        removed_files: list[str] = []
        paths = {
            str(row.get("file_path") or ""),
            str(row.get("raw_embedding_path") or ""),
            str(row.get("embedding_path") or ""),
            str(row.get("clip_meta_path") or ""),
        }
        for raw in paths:
            if not raw:
                continue
            path = Path(raw)
            if not path.exists():
                continue
            path.unlink(missing_ok=True)
            removed_files.append(str(path))

        task_id = str(row["task_id"])
        reindexed_clips = self._rebuild_task_index(task_id)
        return {
            "video_id": int(video_id),
            "task_id": task_id,
            "removed_files": removed_files,
            "reindexed_clips": int(reindexed_clips),
        }

    # -- Scoring --

    def enqueue_score(self, gold_video_id: int, trainee_video_id: int,
                      requested_by: str | None = None) -> dict:
        return self._scoring.enqueue_score(gold_video_id, trainee_video_id, requested_by)

    def run_score_job(self, job_id: str) -> None:
        self._scoring.run_score_job(job_id)

    def get_score(self, score_job_id: str) -> dict | None:
        return self._scoring.get_score(score_job_id)

    def build_score_pdf(self, score_job_id: str) -> Path:
        return self._scoring.build_score_pdf(score_job_id)

    def search(self, task_id: str, video_id: int, clip_idx: int, k: int) -> dict:
        return self._scoring.search(task_id, video_id, clip_idx, k)

    # -- Training --

    def enqueue_training(self, trigger: str = "manual", requested_by: str | None = None) -> dict:
        return self._training.enqueue_training(trigger, requested_by)

    def run_training_job(self, job_id: str) -> None:
        self._training.run_training_job(job_id)

    def get_training_job(self, training_job_id: str) -> dict | None:
        return self._training.get_training_job(training_job_id)

    def get_nightly_status(self) -> dict:
        return self._training.get_nightly_status()

    def _run_builtin_feature_adaptation(self, job_id: str, videos: list[dict]) -> dict:
        return self._training._run_builtin_feature_adaptation(job_id, videos)

    def _compute_next_nightly_run(self, now_local: datetime) -> datetime:
        return self._training._compute_next_nightly_run(now_local)

    # -- Audit --

    def get_audit_trail(self, limit: int = 100) -> list[dict]:
        return self._audit.get_audit_trail(limit)

    def export_signed_audit_trail(self, *, limit: int = 500) -> dict:
        return self._audit.export_signed_audit_trail(limit=limit)

    def get_audit_export_path(self, export_id: str) -> Path:
        return self._audit.get_audit_export_path(export_id)

    # -- Ops --

    def get_queue_metrics(self) -> dict:
        if self._queue is not None:
            queue_payload = self._queue.metrics()
        else:
            queue_payload = {
                "backend": "none",
                "redis_ok": None,
                "error": "queue manager unavailable in this runtime mode",
                "queues": [],
            }
        return {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "runtime_mode": self.runtime_mode,
            "queue": queue_payload,
            "jobs": self.db.job_status_counts(),
        }
