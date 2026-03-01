"""Thin facade that delegates to focused sub-services.

Backward compatibility: SOPilotService, ServiceError, NotFoundError,
and InvalidStateError are all importable from this module.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

from sopilot.config import Settings

logger = logging.getLogger(__name__)
from sopilot.database import Database
from sopilot.exceptions import InvalidStateError, NotFoundError, ServiceError
from sopilot.schemas import ScoreWeights
from sopilot.services.recommendation_service import RecommendationService
from sopilot.services.scoring_service import ScoringService
from sopilot.services.search_service import SearchService
from sopilot.services.step_definition_service import StepDefinitionService
from sopilot.services.storage import FileStorage
from sopilot.services.task_profile_service import TaskProfileService
from sopilot.services.video_processor import VideoProcessor
from sopilot.services.video_service import VideoLoadResult, VideoService

# Re-export for backward compatibility
__all__ = [
    "SOPilotService",
    "ServiceError",
    "NotFoundError",
    "InvalidStateError",
]


class SOPilotService:
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

        # Sub-services
        self.task_profile_service = TaskProfileService(settings, database)
        self.video_service = VideoService(settings, database, storage, video_processor)
        self.search_service = SearchService(database)
        self.scoring_service = ScoringService(
            settings, database, self.video_service, self.task_profile_service.get_task_profile_for
        )
        self.recommendation_service = RecommendationService(database)
        self.step_definition_service = StepDefinitionService(database)

    # ---- Task profile delegation ----

    def _ensure_primary_task_profile(self) -> None:
        self.task_profile_service._ensure_primary_task_profile()

    def get_task_profile(self) -> dict:
        return self.task_profile_service.get_task_profile()

    def update_task_profile(
        self,
        *,
        task_name: str | None,
        pass_score: float | None,
        retrain_score: float | None,
        default_weights: dict[str, float] | None,
        deviation_policy: dict[str, str] | None,
    ) -> dict:
        return self.task_profile_service.update_task_profile(
            task_name=task_name,
            pass_score=pass_score,
            retrain_score=retrain_score,
            default_weights=default_weights,
            deviation_policy=deviation_policy,
        )

    def get_task_profile_for(self, task_id: str) -> dict:
        return self.task_profile_service.get_task_profile_for(task_id)

    # ---- Video delegation ----

    def list_videos(self, *, site_id: str | None, is_gold: bool | None, limit: int) -> list[dict]:
        return self.video_service.list_videos(site_id=site_id, is_gold=is_gold, limit=limit)

    def count_videos(self, *, site_id: str | None = None, is_gold: bool | None = None) -> int:
        return self.video_service.count_videos(site_id=site_id, is_gold=is_gold)

    def get_dataset_summary(self) -> dict:
        return self.video_service.get_dataset_summary()

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
        return self.video_service.ingest_video(
            original_filename=original_filename,
            file_obj=file_obj,
            task_id=task_id,
            site_id=site_id,
            camera_id=camera_id,
            operator_id_hash=operator_id_hash,
            recorded_at=recorded_at,
            is_gold=is_gold,
            enforce_quality=enforce_quality,
        )

    def delete_video(self, video_id: int, *, force: bool = False) -> dict:
        logger.info("audit: video_deleted video_id=%s force=%s", video_id, force)
        return self.video_service.delete_video(video_id, force=force)

    def update_video_metadata(
        self,
        video_id: int,
        *,
        site_id: str | None = None,
        camera_id: str | None = None,
        operator_id_hash: str | None = None,
        recorded_at: str | None = None,
    ) -> dict:
        return self.video_service.update_video_metadata(
            video_id,
            site_id=site_id,
            camera_id=camera_id,
            operator_id_hash=operator_id_hash,
            recorded_at=recorded_at,
        )

    def get_video_detail(self, video_id: int) -> dict:
        return self.video_service.get_video_detail(video_id)

    def get_video_stream_path(self, video_id: int) -> Path:
        return self.video_service.get_video_stream_path(video_id)

    # ---- Scoring delegation ----

    def queue_score_job(
        self,
        *,
        gold_video_id: int,
        trainee_video_id: int,
        weights: ScoreWeights | None = None,
    ) -> dict:
        logger.info(
            "audit: score_job_created gold_video_id=%s trainee_video_id=%s",
            gold_video_id, trainee_video_id,
        )
        return self.scoring_service.queue_score_job(
            gold_video_id=gold_video_id,
            trainee_video_id=trainee_video_id,
            weights=weights,
        )

    def run_score_job(self, job_id: int) -> None:
        self.scoring_service.run_score_job(job_id)
        # Log completion with score and decision from the stored result
        try:
            job = self.scoring_service.get_score_job(job_id)
            result = job.get("result") or {}
            logger.info(
                "audit: score_job_completed job_id=%s score=%s decision=%s",
                job_id,
                result.get("score"),
                (result.get("summary") or {}).get("decision"),
            )
        except Exception:
            pass  # Don't let audit logging break the flow

    def requeue_pending_jobs(self, enqueue_fn: Callable[[int], None]) -> None:
        return self.scoring_service.requeue_pending_jobs(enqueue_fn)

    def rerun_score_job(self, job_id: int) -> dict:
        return self.scoring_service.rerun_score_job(job_id)

    def cancel_score_job(self, job_id: int) -> dict:
        return self.scoring_service.cancel_score_job(job_id)

    def get_score_job(self, job_id: int) -> dict:
        return self.scoring_service.get_score_job(job_id)

    def update_score_review(self, *, job_id: int, verdict: str, note: str | None) -> dict:
        logger.info("audit: score_review_submitted job_id=%s verdict=%s", job_id, verdict)
        return self.scoring_service.update_score_review(job_id=job_id, verdict=verdict, note=note)

    def export_score_job(self, job_id: int) -> dict:
        return self.scoring_service.export_score_job(job_id)

    def get_score_report(self, job_id: int) -> str:
        return self.scoring_service.get_score_report(job_id)

    def get_score_report_pdf(self, job_id: int) -> bytes:
        return self.scoring_service.get_score_report_pdf(job_id)

    def get_score_timeline(self, job_id: int) -> dict:
        return self.scoring_service.get_score_timeline(job_id)

    def rescore_decisions(
        self,
        *,
        task_id: str | None = None,
        dry_run: bool = False,
    ) -> dict:
        return self.scoring_service.rescore_decisions(task_id=task_id, dry_run=dry_run)

    def list_score_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        return self.scoring_service.list_score_jobs(status=status, limit=limit, offset=offset)

    def get_analytics(self, *, task_id: str | None = None, days: int | None = None) -> dict:
        return self.database.get_analytics(task_id=task_id, days=days)

    def get_operator_trend(
        self, operator_id: str, *, task_id: str | None = None
    ) -> dict:
        return self.database.get_operator_trend(operator_id, task_id=task_id)

    def get_step_performance(
        self, *, task_id: str | None = None, days: int | None = None
    ) -> dict:
        return self.database.get_step_performance(task_id=task_id, days=days)

    def get_compliance_overview(
        self, *, task_id: str | None = None, days: int | None = None
    ) -> dict:
        return self.database.get_compliance_overview(task_id=task_id, days=days)

    def get_recommendations(
        self, operator_id: str, *, task_id: str | None = None
    ) -> dict:
        return self.recommendation_service.get_recommendations(
            operator_id, task_id=task_id
        )

    # ---- Learning curve analysis ----

    def get_operator_learning_curve(
        self,
        operator_id: str,
        *,
        task_id: str | None = None,
    ) -> dict:
        """Analyze operator score trajectory and project certification pathway."""
        from sopilot.core.learning_curve import analyze_learning_curve
        scores = self.database.get_operator_scores_chronological(
            operator_id, task_id=task_id
        )
        pass_threshold = self.settings.default_pass_score
        result = analyze_learning_curve(
            operator_id, scores, pass_threshold=pass_threshold
        )
        return {
            "operator_id": result.operator_id,
            "job_count": result.job_count,
            "avg_score": result.avg_score,
            "latest_score": result.latest_score,
            "trend_slope": result.trend_slope,
            "trajectory": result.trajectory,
            "is_certified": result.is_certified,
            "pass_threshold": result.pass_threshold,
            "evaluations_to_certification": result.evaluations_to_certification,
            "confidence": result.confidence,
            "model_type": result.model_type,
            "projected_scores": result.projected_scores,
            "scores": result.scores,
        }

    # ---- Step definition delegation ----

    def get_sop_steps(self, task_id: str) -> dict:
        return self.step_definition_service.get_steps(task_id)

    def upsert_sop_steps(self, task_id: str, steps: list[dict]) -> dict:
        return self.step_definition_service.upsert_steps(task_id, steps)

    def delete_sop_steps(self, task_id: str) -> dict:
        return self.step_definition_service.delete_steps(task_id)

    def get_score_uncertainty(self, job_id: int) -> dict:
        return self.scoring_service.get_score_uncertainty(job_id)

    def compute_soft_dtw(
        self,
        *,
        gold_video_id: int,
        trainee_video_id: int,
        gamma: float = 1.0,
    ) -> dict:
        return self.scoring_service.compute_soft_dtw(
            gold_video_id=gold_video_id,
            trainee_video_id=trainee_video_id,
            gamma=gamma,
        )

    # ---- Ensemble scoring delegation ----

    def score_ensemble(
        self,
        *,
        gold_video_ids: list[int],
        trainee_video_id: int,
        weights_payload: dict | None = None,
    ) -> dict:
        return self.scoring_service.score_ensemble(
            gold_video_ids=gold_video_ids,
            trainee_video_id=trainee_video_id,
            weights_payload=weights_payload,
        )

    # ---- Search delegation ----

    def search(
        self,
        *,
        query_video_id: int,
        query_clip_index: int,
        k: int,
        task_id: str | None,
    ) -> dict:
        return self.search_service.search(
            query_video_id=query_video_id,
            query_clip_index=query_clip_index,
            k=k,
            task_id=task_id,
        )

    # ---- Internal helpers kept for backward compat ----

    def _assert_task_allowed(self, task_id: str) -> None:
        self.video_service.assert_task_allowed(task_id)

    def _load_video_data(self, video_id: int) -> VideoLoadResult:
        return self.video_service.load_video_data(video_id)
