"""Background Task Processing

Asynchronous video processing using Python's concurrent.futures.
Production systems should use Celery + Redis/RabbitMQ.
"""

import logging
import time
import traceback
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from insurance_mvp.api.database import AssessmentRepository, AuditLogRepository, ClaimRepository, DatabaseManager
from insurance_mvp.api.models import ClaimStatus, EventType

logger = logging.getLogger(__name__)


class BackgroundWorker:
    """
    Background task worker using ThreadPoolExecutor.

    Production: Replace with Celery for distributed task queue.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        max_workers: int = 4,
        process_function: Callable | None = None,
    ):
        """
        Initialize background worker.

        Args:
            db_manager: Database manager for task persistence
            max_workers: Maximum concurrent workers
            process_function: Function to process claims (for testing)
        """
        self.db_manager = db_manager
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks: dict[str, Future] = {}
        self.process_function = process_function or self._default_process_function
        self._shutdown = False

        logger.info(f"Background worker initialized with {max_workers} workers")

    def submit_claim(self, claim_id: str) -> bool:
        """
        Submit claim for background processing.

        Args:
            claim_id: Claim ID to process

        Returns:
            True if submitted, False if already processing
        """
        if claim_id in self.active_tasks:
            logger.warning(f"Claim {claim_id} already processing")
            return False

        # Update status to queued
        with self.db_manager.get_session() as session:
            repo = ClaimRepository(session)
            repo.update_status(claim_id, ClaimStatus.QUEUED)

        # Submit task
        future = self.executor.submit(self._process_claim_wrapper, claim_id)
        self.active_tasks[claim_id] = future

        # Cleanup on completion
        future.add_done_callback(lambda f: self._cleanup_task(claim_id))

        logger.info(f"Claim {claim_id} submitted for processing")
        return True

    def _process_claim_wrapper(self, claim_id: str):
        """
        Wrapper for claim processing with error handling and status updates.
        """
        start_time = time.time()
        logger.info(f"Starting processing for claim {claim_id}")

        try:
            # Update status to processing
            with self.db_manager.get_session() as session:
                repo = ClaimRepository(session)
                claim = repo.update_status(claim_id, ClaimStatus.PROCESSING, progress_percent=0.0)

                if not claim:
                    logger.error(f"Claim {claim_id} not found")
                    return

            # Run actual processing
            assessment_result = self.process_function(claim_id, self._update_progress)

            # Store assessment
            with self.db_manager.get_session() as session:
                claim_repo = ClaimRepository(session)
                assessment_repo = AssessmentRepository(session)
                audit_repo = AuditLogRepository(session)

                # Create assessment
                assessment = assessment_repo.create_from_dict(claim_id, assessment_result)

                # Update claim status
                claim_repo.update_status(claim_id, ClaimStatus.ASSESSED, progress_percent=100.0)

                # Audit log
                audit_repo.create(
                    claim_id=claim_id,
                    event_type=EventType.AI_ASSESSMENT,
                    actor_type="AI",
                    actor_id=assessment_result.get("model_version", "unknown"),
                    explanation=f"AI assessment completed: {assessment_result.get('severity')}",
                    after_state={
                        "severity": assessment_result.get("severity"),
                        "confidence": assessment_result.get("confidence"),
                        "review_priority": assessment_result.get("review_priority"),
                    },
                )

            processing_time = time.time() - start_time
            logger.info(f"Claim {claim_id} processed successfully in {processing_time:.2f}s")

        except Exception as e:
            logger.error(f"Error processing claim {claim_id}: {str(e)}")
            logger.error(traceback.format_exc())

            # Update status to failed
            with self.db_manager.get_session() as session:
                repo = ClaimRepository(session)
                repo.update_status(
                    claim_id,
                    ClaimStatus.FAILED,
                    error_message=str(e),
                )

                # Audit log
                audit_repo = AuditLogRepository(session)
                audit_repo.create(
                    claim_id=claim_id,
                    event_type=EventType.STATUS_UPDATE,
                    actor_type="SYSTEM",
                    actor_id="background_worker",
                    explanation=f"Processing failed: {str(e)}",
                )

    def _update_progress(self, claim_id: str, progress_percent: float):
        """Update processing progress"""
        try:
            with self.db_manager.get_session() as session:
                repo = ClaimRepository(session)
                repo.update_status(claim_id, ClaimStatus.PROCESSING, progress_percent=progress_percent)
        except Exception as e:
            logger.error(f"Error updating progress for {claim_id}: {e}")

    def _cleanup_task(self, claim_id: str):
        """Remove task from active tasks"""
        if claim_id in self.active_tasks:
            del self.active_tasks[claim_id]
            logger.debug(f"Cleaned up task for claim {claim_id}")

    def get_status(self, claim_id: str) -> str | None:
        """
        Get processing status for claim.

        Returns:
            "processing", "completed", or None if not found
        """
        if claim_id in self.active_tasks:
            future = self.active_tasks[claim_id]
            if future.done():
                return "completed"
            return "processing"
        return None

    def is_active(self) -> bool:
        """Check if worker is active"""
        return not self._shutdown and len(self.active_tasks) > 0

    def shutdown(self, wait: bool = True):
        """Shutdown worker"""
        self._shutdown = True
        logger.info("Shutting down background worker...")
        self.executor.shutdown(wait=wait)
        logger.info("Background worker shutdown complete")

    def _default_process_function(self, claim_id: str, update_progress: Callable) -> dict[str, Any]:
        """
        Default mock processing function for testing.

        Production: Replace with actual pipeline integration.
        """
        logger.warning("Using default mock processing function!")

        # Simulate processing with progress updates
        update_progress(claim_id, 10.0)
        time.sleep(0.5)

        update_progress(claim_id, 30.0)
        time.sleep(0.5)

        update_progress(claim_id, 60.0)
        time.sleep(0.5)

        update_progress(claim_id, 90.0)
        time.sleep(0.5)

        # Return mock assessment
        return {
            "severity": "MEDIUM",
            "confidence": 0.85,
            "prediction_set": ["MEDIUM", "HIGH"],
            "review_priority": "STANDARD",
            "fault_assessment": {
                "fault_ratio": 40.0,
                "reasoning": "Mock assessment - partial fault detected",
                "applicable_rules": ["Following distance", "Defensive driving"],
                "scenario_type": "rear_end",
            },
            "fraud_risk": {
                "risk_score": 0.15,
                "indicators": [],
                "reasoning": "No fraud indicators detected",
            },
            "hazards": [
                {
                    "type": "near_miss",
                    "actors": ["car"],
                    "spatial_relation": "front",
                    "timestamp_sec": 5.2,
                }
            ],
            "evidence": [
                {
                    "timestamp_sec": 5.2,
                    "description": "Vehicle ahead braked suddenly",
                    "frame_path": None,
                }
            ],
            "causal_reasoning": "Mock processing - sudden braking event detected",
            "recommended_action": "REVIEW",
            "processing_time_sec": 2.0,
            "model_version": "mock-v1.0",
        }


# Pipeline Integration Helper


class PipelineProcessor:
    """
    Integration with the insurance claim processing pipeline.

    Bridges the background worker with InsurancePipeline, running the full
    5-stage pipeline (Mining → VLM → Ranking → Conformal → Review Priority).
    """

    def __init__(self, pipeline_config: dict[str, Any] | None = None):
        """
        Initialize pipeline processor.

        Args:
            pipeline_config: Configuration overrides (database_url, backend, etc.)
        """
        from insurance_mvp.config import CosmosBackend, PipelineConfig
        from insurance_mvp.pipeline import InsurancePipeline

        self.config = pipeline_config or {}

        # Build PipelineConfig with optional overrides
        pcfg = PipelineConfig()
        if self.config.get("backend") == "real":
            pcfg.cosmos.backend = CosmosBackend.QWEN25VL
        pcfg.continue_on_error = True

        self._pipeline = InsurancePipeline(pcfg)
        self._db_url = self.config.get("database_url", "sqlite:///./insurance.db")
        logger.info("Pipeline processor initialized with real InsurancePipeline")

    def process_claim(self, claim_id: str, update_progress: Callable) -> dict[str, Any]:
        """
        Process claim through the full 5-stage pipeline.

        Args:
            claim_id: Claim ID
            update_progress: Callback to update progress (claim_id, progress_percent)

        Returns:
            Assessment result dictionary
        """
        from insurance_mvp.api.database import ClaimRepository, DatabaseManager

        db_manager = DatabaseManager(self._db_url)

        with db_manager.get_session() as session:
            repo = ClaimRepository(session)
            claim = repo.get_by_id(claim_id)
            if not claim:
                raise ValueError(f"Claim {claim_id} not found")
            video_path = claim.video_path

        logger.info(f"Processing video via real pipeline: {video_path}")
        update_progress(claim_id, 10.0)

        # Run through the real 5-stage pipeline
        result = self._pipeline.process_video(video_path, video_id=claim_id)

        if not result.success:
            raise RuntimeError(f"Pipeline failed: {result.error_message}")

        update_progress(claim_id, 90.0)

        # Convert the first (highest-severity) assessment to dict format
        if result.assessments:
            return self._assessment_to_dict(result.assessments[0], result)

        # No danger clips found — return safe assessment
        return {
            "severity": "NONE",
            "confidence": 0.95,
            "prediction_set": ["NONE"],
            "review_priority": "LOW_PRIORITY",
            "fault_assessment": {
                "fault_ratio": 0.0,
                "reasoning": "No hazardous events detected in video",
                "applicable_rules": [],
                "scenario_type": "normal_driving",
                "traffic_signal": None,
                "right_of_way": None,
            },
            "fraud_risk": {
                "risk_score": 0.0,
                "indicators": [],
                "reasoning": "No anomalies detected",
            },
            "hazards": [],
            "evidence": [],
            "causal_reasoning": "Normal driving — no incidents detected by mining pipeline",
            "recommended_action": "APPROVE",
            "processing_time_sec": result.processing_time_sec,
            "model_version": "insurance-pipeline-v1.0",
        }

    @staticmethod
    def _assessment_to_dict(assessment, result) -> dict[str, Any]:
        """Convert a ClaimAssessment + VideoResult into the dict format expected by BackgroundWorker."""
        fault = assessment.fault_assessment
        fraud = assessment.fraud_risk

        return {
            "severity": assessment.severity,
            "confidence": assessment.confidence,
            "prediction_set": list(assessment.prediction_set),
            "review_priority": assessment.review_priority,
            "fault_assessment": {
                "fault_ratio": fault.fault_ratio,
                "reasoning": fault.reasoning,
                "applicable_rules": fault.applicable_rules,
                "scenario_type": fault.scenario_type,
                "traffic_signal": getattr(fault, "traffic_signal", None),
                "right_of_way": getattr(fault, "right_of_way", None),
            },
            "fraud_risk": {
                "risk_score": fraud.risk_score,
                "indicators": fraud.indicators,
                "reasoning": fraud.reasoning,
            },
            "hazards": [
                h.model_dump() if hasattr(h, "model_dump") else h
                for h in assessment.hazards
            ],
            "evidence": [
                e.model_dump() if hasattr(e, "model_dump") else e
                for e in assessment.evidence
            ],
            "causal_reasoning": assessment.causal_reasoning,
            "recommended_action": assessment.recommended_action,
            "processing_time_sec": result.processing_time_sec,
            "model_version": "insurance-pipeline-v1.0",
        }


# Global worker instance (singleton)
_global_worker: BackgroundWorker | None = None


def initialize_worker(
    db_manager: DatabaseManager,
    max_workers: int = 4,
    use_pipeline: bool = False,
) -> BackgroundWorker:
    """
    Initialize global background worker.

    Args:
        db_manager: Database manager
        max_workers: Maximum concurrent workers
        use_pipeline: Use actual pipeline (vs mock)

    Returns:
        BackgroundWorker instance
    """
    global _global_worker

    if _global_worker is not None:
        logger.warning("Background worker already initialized")
        return _global_worker

    # Choose processing function
    if use_pipeline:
        pipeline = PipelineProcessor()
        process_func = pipeline.process_claim
    else:
        process_func = None  # Use default mock

    _global_worker = BackgroundWorker(
        db_manager=db_manager,
        max_workers=max_workers,
        process_function=process_func,
    )

    return _global_worker


def get_worker() -> BackgroundWorker:
    """Get global worker instance"""
    if _global_worker is None:
        raise RuntimeError("Background worker not initialized. Call initialize_worker() first.")
    return _global_worker


def shutdown_worker(wait: bool = True):
    """Shutdown global worker"""
    global _global_worker
    if _global_worker:
        _global_worker.shutdown(wait=wait)
        _global_worker = None
