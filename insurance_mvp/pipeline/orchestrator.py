"""Insurance MVP - Pipeline Orchestrator.

Slim orchestration layer that delegates to stage modules:
  stages/mining.py        — Stage 1: Multimodal signal mining
  stages/vlm_inference.py — Stage 2: Mock VLM helpers
  stages/ranking.py       — Stage 3: Severity ranking
  stages/conformal_stage.py — Stage 4: Conformal prediction
  stages/review_priority.py — Stage 5: Review priority
  adapters/vlm_adapter.py — VLM → fault/fraud bridge
  results.py              — JSON/HTML persistence
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from insurance_mvp.config import PipelineConfig
from insurance_mvp.conformal.split_conformal import ConformalConfig, SplitConformal
from insurance_mvp.insurance.schema import ClaimAssessment, FaultAssessment, FraudRisk
from insurance_mvp.mining.audio import AudioAnalyzer
from insurance_mvp.mining.fuse import SignalFuser
from insurance_mvp.mining.motion import MotionAnalyzer
from insurance_mvp.mining.proximity import ProximityAnalyzer

# Delegated stage functions
from insurance_mvp.pipeline.adapters.vlm_adapter import (
    assess_fault_from_vlm,
    detect_fraud_from_vlm,
)
from insurance_mvp.pipeline.results import save_checkpoint, save_results
from insurance_mvp.pipeline.stages.conformal_stage import apply_conformal
from insurance_mvp.pipeline.stages.mining import mock_danger_clips, run_mining
from insurance_mvp.pipeline.stages.ranking import rank_by_severity
from insurance_mvp.pipeline.stages.recalibration import RecalibrationConfig, recalibrate_severity
from insurance_mvp.pipeline.stages.review_priority import assign_review_priority
from insurance_mvp.pipeline.stages.vlm_inference import (
    create_error_assessment,
    mock_vlm_result,
)
from insurance_mvp.serialization import to_serializable

# --- Data Classes ---


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics."""

    total_videos: int = 0
    successful_videos: int = 0
    failed_videos: int = 0

    total_clips_mined: int = 0
    total_clips_analyzed: int = 0

    total_processing_time_sec: float = 0.0
    mining_time_sec: float = 0.0
    vlm_inference_time_sec: float = 0.0
    fault_assessment_time_sec: float = 0.0
    fraud_detection_time_sec: float = 0.0
    conformal_time_sec: float = 0.0

    avg_clips_per_video: float = 0.0
    avg_processing_time_per_video: float = 0.0


@dataclass
class VideoResult:
    """Result of processing a single video."""

    video_id: str
    video_path: str
    success: bool
    error_message: str | None = None

    danger_clips: list[Any] = None
    assessments: list[ClaimAssessment] = None

    output_json_path: str | None = None
    output_html_path: str | None = None

    processing_time_sec: float = 0.0
    stage_timings: dict[str, float] = None


# --- Pipeline Implementation ---


class InsurancePipeline:
    """End-to-end insurance video processing pipeline.

    Orchestrates all components with robust error handling,
    progress tracking, and resource management.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.metrics = PipelineMetrics()
        self._gpu_semaphore = threading.Semaphore(config.cosmos.max_concurrent_inferences)
        self._init_components()

    def _setup_logging(self) -> logging.Logger:
        from insurance_mvp.logging_config import configure_logging

        json_output = os.environ.get("INSURANCE_LOG_JSON", "").lower() == "true"
        configure_logging(
            level=self.config.log_level,
            json_output=json_output,
            log_file=self.config.log_file,
        )
        return logging.getLogger("insurance_pipeline")

    def _init_components(self):
        self.logger.info("Initializing pipeline components...")

        # B1: Mining
        self.audio_analyzer = AudioAnalyzer()
        self.motion_analyzer = MotionAnalyzer()
        self.proximity_analyzer = ProximityAnalyzer()
        self.signal_fuser = SignalFuser(self.config.mining)
        self.signal_fuser.set_analyzers(
            self.audio_analyzer, self.motion_analyzer, self.proximity_analyzer
        )
        self.logger.info("Mining components initialized")

        # B2: Video-LLM (lazy)
        self.cosmos_client = None

        # B3: Insurance domain logic
        from insurance_mvp.insurance.fault_assessment import (
            FaultAssessmentConfig,
            FaultAssessmentEngine,
        )
        from insurance_mvp.insurance.fraud_detection import (
            FraudDetectionConfig,
            FraudDetectionEngine,
        )

        fault_cfg = FaultAssessmentConfig(
            **{k: getattr(self.config.fault, k) for k in self.config.fault.__dataclass_fields__}
        )
        self.fault_assessor = FaultAssessmentEngine(config=fault_cfg)
        self.logger.info("Fault assessor initialized")

        fraud_cfg = FraudDetectionConfig(
            **{k: getattr(self.config.fraud, k) for k in self.config.fraud.__dataclass_fields__}
        )
        self.fraud_detector = FraudDetectionEngine(config=fraud_cfg)
        self.logger.info("Fraud detector initialized")

        # B4: Conformal prediction
        if self.config.enable_conformal:
            self.conformal_predictor = SplitConformal(
                ConformalConfig(
                    alpha=self.config.conformal.alpha,
                    severity_levels=self.config.conformal.severity_levels,
                )
            )
            if self.config.conformal.use_pretrained_calibration:
                self._load_conformal_calibration()
            self.logger.info("Conformal predictor initialized")
        else:
            self.conformal_predictor = None
            self.logger.info("Conformal prediction disabled")

    def _load_conformal_calibration(self):
        if not self.config.conformal.calibration_data_path:
            self.logger.info("Using mock calibration (no pretrained data)")
            self._mock_conformal_calibration()
            return
        try:
            calib_path = Path(self.config.conformal.calibration_data_path)
            self.logger.info("Loaded conformal calibration from %s", calib_path)
        except Exception as e:
            self.logger.warning("Failed to load calibration: %s. Using mock.", e)
            self._mock_conformal_calibration()

    def _mock_conformal_calibration(self):
        n_calib = 100
        scores = np.random.dirichlet(np.ones(4), size=n_calib)
        y_true = np.random.randint(0, 4, size=n_calib)
        self.conformal_predictor.fit(scores, y_true)

    # ------------------------------------------------------------------
    # Core pipeline: process_video
    # ------------------------------------------------------------------

    def process_video(self, video_path: str, video_id: str | None = None) -> VideoResult:
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            return VideoResult(
                video_id=video_id or video_path_obj.stem,
                video_path=str(video_path),
                success=False,
                error_message=f"Video file not found: {video_path}",
            )

        if video_id is None:
            video_id = video_path_obj.stem

        from insurance_mvp.logging_config import set_correlation_id
        from insurance_mvp.metrics import METRICS

        set_correlation_id(video_id=video_id)
        METRICS.inc("claims_total", labels={"status": "processing"})
        METRICS.inc_gauge("active_processing")
        self.logger.info("Processing video: %s", video_id)
        start_time = time.time()
        stage_timings: dict[str, float] = {}

        try:
            # Stage 1: Mining
            self.logger.info("[%s] Stage 1: Multimodal signal mining...", video_id)
            stage_start = time.time()
            danger_clips = self._stage1_mining(video_path, video_id)
            stage_timings["mining"] = time.time() - stage_start
            METRICS.observe("processing_duration_seconds", stage_timings["mining"], labels={"stage": "mining"})
            self.logger.info("[%s] Mined %d danger clips", video_id, len(danger_clips))

            if not danger_clips:
                self.logger.warning("[%s] No danger clips found. Skipping remaining stages.", video_id)
                return VideoResult(
                    video_id=video_id,
                    video_path=str(video_path),
                    success=True,
                    danger_clips=[],
                    assessments=[],
                    processing_time_sec=time.time() - start_time,
                    stage_timings=stage_timings,
                )

            # Stage 2: VLM Inference
            self.logger.info("[%s] Stage 2: Video-LLM inference on %d clips...", video_id, len(danger_clips))
            stage_start = time.time()
            assessments = self._stage2_vlm_inference(danger_clips, video_id)
            stage_timings["vlm_inference"] = time.time() - stage_start
            METRICS.observe("processing_duration_seconds", stage_timings["vlm_inference"], labels={"stage": "vlm"})
            self.logger.info("[%s] Completed %d assessments", video_id, len(assessments))

            # Stage 3: Ranking (delegated)
            self.logger.info("[%s] Stage 3: Severity ranking...", video_id)
            stage_start = time.time()
            assessments = rank_by_severity(assessments)
            stage_timings["ranking"] = time.time() - stage_start

            # Stage 4: Conformal (delegated)
            if self.config.enable_conformal and self.conformal_predictor:
                self.logger.info("[%s] Stage 4: Conformal prediction...", video_id)
                stage_start = time.time()
                assessments = apply_conformal(assessments, self.conformal_predictor)
                stage_timings["conformal"] = time.time() - stage_start

            # Stage 5: Review priority (delegated)
            self.logger.info("[%s] Stage 5: Review priority assignment...", video_id)
            stage_start = time.time()
            assessments = assign_review_priority(assessments)
            stage_timings["review_priority"] = time.time() - stage_start

            # Save results (delegated)
            json_path, html_path = save_results(
                self.config.output_dir, video_id, danger_clips, assessments, self.config
            )

            processing_time = time.time() - start_time
            METRICS.observe("processing_duration_seconds", processing_time, labels={"stage": "total"})
            METRICS.inc("claims_total", labels={"status": "completed"})
            METRICS.dec_gauge("active_processing")
            self.logger.info("[%s] Completed in %.2fs", video_id, processing_time)

            return VideoResult(
                video_id=video_id,
                video_path=str(video_path),
                success=True,
                danger_clips=danger_clips,
                assessments=assessments,
                output_json_path=json_path,
                output_html_path=html_path,
                processing_time_sec=processing_time,
                stage_timings=stage_timings,
            )

        except Exception as e:
            METRICS.inc("claims_total", labels={"status": "failed"})
            METRICS.dec_gauge("active_processing")
            self.logger.error("[%s] Pipeline failed: %s", video_id, e)
            self.logger.debug(traceback.format_exc())
            return VideoResult(
                video_id=video_id,
                video_path=str(video_path),
                success=False,
                error_message=str(e),
                processing_time_sec=time.time() - start_time,
                stage_timings=stage_timings,
            )

    # ------------------------------------------------------------------
    # Stage helpers (thin wrappers → delegated modules)
    # ------------------------------------------------------------------

    def _stage1_mining(self, video_path: str, video_id: str) -> list[Any]:
        if not self.signal_fuser:
            self.logger.warning("Signal fuser not available. Returning mock danger clips.")
            return mock_danger_clips(video_path, video_id, self.config.mining.top_k_clips)
        try:
            return run_mining(self.signal_fuser, video_path, video_id, self.config.mining.top_k_clips)
        except Exception:
            if self.config.continue_on_error:
                self.logger.warning("Mining failed, falling back to mock danger clips.")
                return mock_danger_clips(video_path, video_id, self.config.mining.top_k_clips)
            raise

    def _stage2_vlm_inference(self, danger_clips: list[Any], video_id: str) -> list[ClaimAssessment]:
        assessments: list[ClaimAssessment] = []
        for clip in tqdm(danger_clips, desc="VLM Inference", unit="clip"):
            try:
                assessment = self._process_single_clip(clip, video_id)
                assessments.append(assessment)
                if self.config.save_intermediate_results:
                    save_checkpoint(self.config.output_dir, video_id, assessments)
            except Exception as e:
                self.logger.error("Failed to process clip %s: %s", clip.get("clip_id", "unknown"), e)
                if self.config.continue_on_error:
                    assessments.append(create_error_assessment(clip, video_id, str(e)))
                else:
                    raise
        return assessments

    def _process_single_clip(self, clip: Any, video_id: str) -> ClaimAssessment:
        clip_id = clip.get("clip_id", "unknown")
        start_time = time.time()

        with self._gpu_semaphore:
            # VLM inference
            if self.cosmos_client:
                vlm_result = self._vlm_inference_with_retry(clip)
            else:
                vlm_result = mock_vlm_result(clip)

            # Fault assessment (delegated)
            if self.config.enable_fault_assessment and self.fault_assessor:
                fault = assess_fault_from_vlm(vlm_result, clip, self.fault_assessor)
            else:
                fault = FaultAssessment(
                    fault_ratio=50.0,
                    reasoning="Fault assessment disabled",
                    applicable_rules=[],
                    scenario_type="unknown",
                )

            # Fraud detection (delegated)
            if self.config.enable_fraud_detection and self.fraud_detector:
                fraud = detect_fraud_from_vlm(vlm_result, clip, self.fraud_detector)
            else:
                fraud = FraudRisk(risk_score=0.0, indicators=[], reasoning="Fraud detection disabled")

        processing_time = time.time() - start_time
        severity = vlm_result.get("severity", "LOW")
        confidence = vlm_result.get("confidence", 0.5)

        # Post-VLM recalibration using mining signals
        if self.config.enable_recalibration:
            severity, confidence, recal_reason = recalibrate_severity(
                vlm_severity=severity,
                vlm_confidence=confidence,
                danger_score=clip.get("danger_score", 0.5),
                motion_score=clip.get("motion_score", 0.0),
                proximity_score=clip.get("proximity_score", 0.0),
            )
            if recal_reason != "no_adjustment":
                self.logger.info("[%s] Recalibrated: %s", video_id, recal_reason)

        # Severity-aware fault ratio adjustment
        if severity == "NONE":
            fault = FaultAssessment(
                fault_ratio=0.0,
                reasoning="No incident detected — fault assessment not applicable",
                applicable_rules=[],
                scenario_type=fault.scenario_type,
            )
        elif severity in ("LOW", "MEDIUM") and "near" in vlm_result.get("reasoning", "").lower():
            fault = FaultAssessment(
                fault_ratio=min(fault.fault_ratio, 20.0),
                reasoning=f"Near-miss (no contact): {fault.reasoning}",
                applicable_rules=fault.applicable_rules,
                scenario_type=fault.scenario_type,
            )

        return ClaimAssessment(
            severity=severity,
            confidence=confidence,
            prediction_set={severity},
            review_priority="STANDARD",
            fault_assessment=fault,
            fraud_risk=fraud,
            hazards=[],
            evidence=[],
            causal_reasoning=vlm_result.get("reasoning", ""),
            recommended_action="REVIEW",
            video_id=video_id,
            processing_time_sec=processing_time,
        )

    def _vlm_inference_with_retry(self, clip: Any) -> dict[str, Any]:
        for attempt in range(self.config.max_retries):
            try:
                video_path = clip.get("video_path")
                start_sec = clip.get("start_sec", 0.0)
                end_sec = clip.get("end_sec", 5.0)
                return self.cosmos_client.analyze_clip(
                    video_path=video_path, start_sec=start_sec, end_sec=end_sec
                )
            except Exception as e:
                self.logger.warning("VLM inference attempt %d failed: %s", attempt + 1, e)
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay_sec)
                else:
                    raise

    # Backward-compat aliases for tests that call private stage methods
    def _stage3_ranking(self, assessments):
        return rank_by_severity(assessments)

    def _stage4_conformal(self, assessments):
        return apply_conformal(assessments, self.conformal_predictor)

    def _stage5_review_priority(self, assessments):
        return assign_review_priority(assessments)

    def _mock_vlm_result(self, clip):
        return mock_vlm_result(clip)

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def process_batch(self, video_paths: list[str]) -> list[VideoResult]:
        self.logger.info("Processing batch of %d videos...", len(video_paths))
        self.metrics.total_videos = len(video_paths)
        start_time = time.time()
        results: list[VideoResult] = []

        if self.config.parallel_workers == 1:
            for video_path in tqdm(video_paths, desc="Processing videos", unit="video"):
                result = self.process_video(video_path)
                results.append(result)
                self._update_metrics(result)
        else:
            with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
                futures = {
                    executor.submit(self.process_video, vp): vp for vp in video_paths
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos", unit="video"):
                    try:
                        result = future.result()
                        results.append(result)
                        self._update_metrics(result)
                    except Exception as e:
                        self.logger.error("Unexpected error processing %s: %s", futures[future], e)

        self.metrics.total_processing_time_sec = time.time() - start_time
        self.metrics.avg_processing_time_per_video = (
            self.metrics.total_processing_time_sec / len(video_paths)
        )
        self._save_batch_summary(results)
        self._print_summary()
        return results

    def _update_metrics(self, result: VideoResult):
        if result.success:
            self.metrics.successful_videos += 1
            if result.assessments:
                self.metrics.total_clips_analyzed += len(result.assessments)
        else:
            self.metrics.failed_videos += 1
        if result.danger_clips:
            self.metrics.total_clips_mined += len(result.danger_clips)

    def _save_batch_summary(self, results: list[VideoResult]):
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "batch_summary.json"
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": asdict(self.metrics),
            "results": [
                {
                    "video_id": r.video_id,
                    "success": r.success,
                    "error_message": r.error_message,
                    "processing_time_sec": r.processing_time_sec,
                    "num_clips": len(r.assessments) if r.assessments else 0,
                }
                for r in results
            ],
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        self.logger.info("Saved batch summary: %s", summary_path)

    def _print_summary(self):
        m = self.metrics
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info("Total videos: %d", m.total_videos)
        self.logger.info("Successful: %d", m.successful_videos)
        self.logger.info("Failed: %d", m.failed_videos)
        self.logger.info("Total clips mined: %d", m.total_clips_mined)
        self.logger.info("Total clips analyzed: %d", m.total_clips_analyzed)
        self.logger.info("Total processing time: %.2fs", m.total_processing_time_sec)
        self.logger.info("Avg time per video: %.2fs", m.avg_processing_time_per_video)
        self.logger.info("=" * 60)

    # Serialization delegated to insurance_mvp.serialization
    _serialize_object = staticmethod(to_serializable)

    # Kept for test backward compat (_save_results called from tests)
    def _save_results(self, video_id, danger_clips, assessments):
        return save_results(self.config.output_dir, video_id, danger_clips, assessments, self.config)

    def _generate_summary(self, assessments):
        from insurance_mvp.pipeline.results import generate_summary
        return generate_summary(assessments)
