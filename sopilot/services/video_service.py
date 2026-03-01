"""Video ingest, detail retrieval, listing, and dataset summary."""

import json
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from sopilot.api.cache import timed_cache
from sopilot.config import Settings
from sopilot.constants import DATASET_TARGET_GOLD, DATASET_TARGET_TRAINEE
from sopilot.core.segmentation import detect_step_boundaries
from sopilot.core.video_quality import VideoQualityChecker
from sopilot.database import Database
from sopilot.exceptions import InvalidStateError, NotFoundError, ServiceError, ValidationError
from sopilot.services.storage import FileStorage
from sopilot.services.video_processor import VideoProcessor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoLoadResult:
    video: dict[str, Any]
    clips: list[dict[str, Any]]
    embeddings: np.ndarray
    boundaries: list[int]

    def __iter__(self) -> Iterator[Any]:
        """Backward-compatible tuple unpacking support."""
        return iter((self.video, self.clips, self.embeddings, self.boundaries))


class VideoService:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        storage: FileStorage,
        video_processor: VideoProcessor,
    ) -> None:
        self.settings = settings
        self.database = database
        self.storage = storage
        self.video_processor = video_processor

    def assert_task_allowed(self, task_id: str) -> None:
        if self.settings.enforce_primary_task and task_id != self.settings.primary_task_id:
            raise InvalidStateError(
                f"PoC mode allows only task_id='{self.settings.primary_task_id}', got '{task_id}'"
            )

    def ingest_video(
        self,
        *,
        original_filename: str,
        file_obj: BinaryIO,
        task_id: str,
        site_id: str | None,
        camera_id: str | None,
        operator_id_hash: str | None,
        recorded_at: str | None,
        is_gold: bool,
        enforce_quality: bool = False,
    ) -> dict:
        self.assert_task_allowed(task_id)
        started = time.perf_counter()
        video_id = self.database.insert_video(
            task_id=task_id,
            site_id=site_id,
            camera_id=camera_id,
            operator_id_hash=operator_id_hash,
            recorded_at=recorded_at,
            is_gold=is_gold,
            original_filename=original_filename,
        )
        file_path = self.storage.save_upload(video_id, original_filename, file_obj)
        try:
            process_start = time.perf_counter()
            clips = self.video_processor.process(file_path)
            process_ms = int((time.perf_counter() - process_start) * 1000)
            embeddings = np.asarray([clip.embedding for clip in clips], dtype=np.float32)
            boundaries = detect_step_boundaries(
                embeddings,
                min_gap=self.settings.min_boundary_gap,
                z_threshold=self.settings.boundary_z_threshold,
            )
            clip_payload = [
                {
                    "clip_index": clip.clip_index,
                    "start_sec": clip.start_sec,
                    "end_sec": clip.end_sec,
                    "embedding": clip.embedding,
                    "quality_flag": clip.quality_flag,
                }
                for clip in clips
            ]
            # Run quality check (informational by default; blocks ingest when enforce_quality=True)
            quality_report = None
            try:
                quality_checker = VideoQualityChecker()
                quality_report = quality_checker.check(file_path)
                if not quality_report.overall_pass:
                    logger.warning(
                        "video quality check failed video_id=%s checks=%s",
                        video_id,
                        [c.name for c in quality_report.checks if not c.passed],
                    )
            except ValidationError:
                raise
            except Exception as qc_exc:
                logger.warning(
                    "video quality check error video_id=%s err=%s", video_id, qc_exc,
                )

            # Gold Builder quality gate: block ingest if requested and quality fails
            if enforce_quality and quality_report and not quality_report.overall_pass:
                raise ValidationError(
                    "Gold動画が品質基準を満たしていません",
                    error_code="QUALITY_GATE_FAILED",
                    context={"quality": quality_report.to_dict()},
                )

            self.database.finalize_video(
                video_id=video_id,
                file_path=file_path,
                step_boundaries=boundaries,
                clips=clip_payload,
                embedding_model=self.video_processor.embedder.name,
            )
            gold_version = None
            if is_gold:
                gold_version = self.database.get_gold_version(video_id, task_id)
            total_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "ingest completed video_id=%s task_id=%s is_gold=%s clips=%s process_ms=%s total_ms=%s",
                video_id,
                task_id,
                is_gold,
                len(clips),
                process_ms,
                total_ms,
            )
            return {
                "video_id": video_id,
                "task_id": task_id,
                "is_gold": is_gold,
                "status": "ready",
                "clip_count": len(clips),
                "step_boundaries": boundaries,
                "original_filename": original_filename,
                "quality": quality_report.to_dict() if quality_report else None,
                "gold_version": gold_version,
            }
        except ValidationError:
            # Quality gate failure: mark failed and clean up without wrapping in ServiceError
            self.database.fail_video(video_id, "Quality gate failed")
            try:
                p = Path(file_path)
                if p.exists():
                    p.unlink()
                    logger.info("cleaned up quality-failed upload file_path=%s video_id=%s", file_path, video_id)
            except Exception as cleanup_exc:
                logger.warning(
                    "failed to clean up quality-failed file file_path=%s video_id=%s err=%s",
                    file_path, video_id, cleanup_exc,
                )
            raise
        except Exception as exc:
            self.database.fail_video(video_id, str(exc))
            # Clean up the uploaded file to prevent disk space leaks
            try:
                p = Path(file_path)
                if p.exists():
                    p.unlink()
                    logger.info("cleaned up failed upload file_path=%s video_id=%s", file_path, video_id)
            except Exception as cleanup_exc:
                logger.warning(
                    "failed to clean up uploaded file file_path=%s video_id=%s err=%s",
                    file_path, video_id, cleanup_exc,
                )
            logger.exception("ingest failed video_id=%s err=%s", video_id, exc)
            raise ServiceError(f"Failed to ingest video {video_id}: {exc}") from exc

    def delete_video(self, video_id: int, *, force: bool = False) -> dict:
        """Delete a video and its associated clips and file.

        When ``force=True``, also deletes score jobs and reviews referencing this video.
        """
        video = self.database.get_video(video_id)
        if video is None:
            raise NotFoundError(f"Video {video_id} not found")
        try:
            deleted = self.database.delete_video(video_id, force=force)
        except ValueError as exc:
            raise InvalidStateError(str(exc)) from exc
        if not deleted:
            raise NotFoundError(f"Video {video_id} not found")
        # Clean up file on disk
        file_path = video.get("file_path")
        if file_path:
            p = Path(file_path)
            if p.exists():
                p.unlink()
        logger.info("video deleted video_id=%s task_id=%s force=%s", video_id, video.get("task_id"), force)
        return {"video_id": video_id, "deleted": True}

    def update_video_metadata(
        self,
        video_id: int,
        *,
        site_id: str | None = None,
        camera_id: str | None = None,
        operator_id_hash: str | None = None,
        recorded_at: str | None = None,
    ) -> dict:
        """Update mutable metadata fields on a video."""
        if not self.database.update_video_metadata(
            video_id,
            site_id=site_id,
            camera_id=camera_id,
            operator_id_hash=operator_id_hash,
            recorded_at=recorded_at,
        ):
            raise NotFoundError(f"Video {video_id} not found")
        logger.info("video metadata updated video_id=%s", video_id)
        return self.get_video_detail(video_id)

    def get_video_detail(self, video_id: int) -> dict:
        video = self.database.get_video(video_id)
        if video is None:
            raise NotFoundError(f"Video {video_id} not found")
        gold_version = None
        if video["is_gold"]:
            gold_version = self.database.get_gold_version(video_id, video["task_id"])
        return {
            "video_id": int(video["id"]),
            "task_id": video["task_id"],
            "is_gold": bool(video["is_gold"]),
            "status": video["status"],
            "clip_count": int(video["clip_count"]),
            "site_id": video["site_id"],
            "camera_id": video["camera_id"],
            "operator_id_hash": video["operator_id_hash"],
            "recorded_at": video["recorded_at"],
            "embedding_model": video["embedding_model"],
            "step_boundaries": json.loads(video["step_boundaries_json"]),
            "error": video["error"],
            "original_filename": video.get("original_filename"),
            "created_at": video["created_at"],
            "updated_at": video["updated_at"],
            "gold_version": gold_version,
        }

    def get_video_stream_path(self, video_id: int) -> Path:
        video = self.database.get_video(video_id)
        if video is None:
            raise NotFoundError(f"Video {video_id} not found")
        file_path = video.get("file_path")
        if not file_path:
            raise InvalidStateError(f"Video {video_id} has no file path")
        path = Path(file_path)
        if not path.exists():
            raise NotFoundError(f"Stored file for video {video_id} does not exist")
        return path

    def list_videos(
        self,
        *,
        site_id: str | None,
        is_gold: bool | None,
        limit: int,
    ) -> list[dict]:
        task_filter = self.settings.primary_task_id if self.settings.enforce_primary_task else None
        rows = self.database.list_videos(task_id=task_filter, site_id=site_id, is_gold=is_gold, limit=limit)
        return [
            {
                "video_id": int(row["id"]),
                "task_id": row["task_id"],
                "is_gold": bool(row["is_gold"]),
                "status": row["status"],
                "site_id": row["site_id"],
                "camera_id": row["camera_id"],
                "operator_id_hash": row["operator_id_hash"],
                "recorded_at": row["recorded_at"],
                "created_at": row["created_at"],
                "clip_count": int(row["clip_count"]),
                "original_filename": row.get("original_filename"),
                "gold_version": int(row["gold_version"]) if row.get("gold_version") is not None else None,  # type: ignore[arg-type]
            }
            for row in rows
        ]

    def count_videos(
        self,
        *,
        site_id: str | None = None,
        is_gold: bool | None = None,
    ) -> int:
        """Count videos matching the given filters (respects primary_task enforcement)."""
        task_filter = self.settings.primary_task_id if self.settings.enforce_primary_task else None
        return self.database.count_videos(task_id=task_filter, is_gold=is_gold)

    @timed_cache(ttl_seconds=10.0)
    def get_dataset_summary(self) -> dict:
        task_filter = self.settings.primary_task_id if self.settings.enforce_primary_task else None
        gold_total = self.database.count_videos(task_id=task_filter, is_gold=True)
        trainee_total = self.database.count_videos(task_id=task_filter, is_gold=False)
        ready_total = self.database.count_videos(task_id=task_filter, status="ready")
        by_site = self.database.count_videos_by_site(task_id=task_filter)
        total = gold_total + trainee_total
        return {
            "task_id": self.settings.primary_task_id,
            "total_videos": total,
            "gold_videos": gold_total,
            "trainee_videos": trainee_total,
            "ready_videos": ready_total,
            "by_site": by_site,
            "target": {"gold": DATASET_TARGET_GOLD, "trainee": DATASET_TARGET_TRAINEE},
            "progress": {
                "gold_pct": round(100.0 * gold_total / float(DATASET_TARGET_GOLD), 1) if total else 0.0,
                "trainee_pct": round(100.0 * trainee_total / float(DATASET_TARGET_TRAINEE), 1) if total else 0.0,
            },
        }

    def load_video_data(self, video_id: int) -> VideoLoadResult:
        video = self.database.get_video(video_id)
        if video is None:
            raise NotFoundError(f"Video {video_id} not found")
        if video["status"] != "ready":
            raise InvalidStateError(f"Video {video_id} status is '{video['status']}'")

        clips = self.database.get_video_clips(video_id)
        if not clips:
            raise InvalidStateError(f"Video {video_id} has no extracted clips")

        embeddings = np.asarray([clip["embedding"] for clip in clips], dtype=np.float32)
        boundaries = json.loads(video["step_boundaries_json"])
        return VideoLoadResult(
            video=dict(video),
            clips=[dict(c) for c in clips],
            embeddings=embeddings,
            boundaries=boundaries,
        )
