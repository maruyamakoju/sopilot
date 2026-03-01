"""Scoring pipeline: queue jobs, run alignments, apply task policies."""

import json
import logging
import threading
import time
import urllib.request
from collections.abc import Callable
from typing import Any

from sopilot.api.cache import timed_cache
from sopilot.config import Settings
from sopilot.core.dtw import dtw_align
from sopilot.core.report import build_report_html
from sopilot.core.report_pdf import build_report_pdf
from sopilot.core.score_pipeline import (
    apply_task_policy,
    attach_timecodes,
    build_score_timeline,
)
from sopilot.core.scoring import ScoreWeights as CoreScoreWeights
from sopilot.core.scoring import (
    compute_score_confidence,
    compute_step_contributions,
    compute_time_compliance_per_step,
    score_alignment,
)

try:
    from sopilot.core.uncertainty import compute_score_uncertainty as _compute_score_uncertainty
    _UNCERTAINTY_AVAILABLE = True
except Exception:  # pragma: no cover
    _UNCERTAINTY_AVAILABLE = False
    _compute_score_uncertainty = None  # type: ignore[assignment]

from sopilot.database import Database
from sopilot.exceptions import InvalidStateError, NotFoundError
from sopilot.logging_config import log_context
from sopilot.schemas import ScoreWeights
from sopilot.services.video_service import VideoService

logger = logging.getLogger(__name__)


class ScoringService:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        video_service: VideoService,
        get_task_profile_for: Callable[[str], dict],
    ) -> None:
        self.settings = settings
        self.database = database
        self.video_service = video_service
        self._get_task_profile_for = get_task_profile_for

    def queue_score_job(
        self,
        *,
        gold_video_id: int,
        trainee_video_id: int,
        weights: ScoreWeights | None = None,
    ) -> dict:
        task_id = self._validate_score_pair(gold_video_id, trainee_video_id)
        profile = self._get_task_profile_for(task_id)
        if weights is None:
            weight_payload = ScoreWeights(**profile["default_weights"]).model_dump()
        else:
            weight_payload = weights.model_dump()
        job_id = self.database.create_score_job(
            gold_video_id=gold_video_id,
            trainee_video_id=trainee_video_id,
            weights=weight_payload,
        )
        logger.info("score job queued job_id=%s gold=%s trainee=%s", job_id, gold_video_id, trainee_video_id)
        return self.get_score_job(job_id)

    def run_score_job(self, job_id: int) -> None:
        if not self.database.claim_score_job(job_id):
            return
        started = time.perf_counter()
        with log_context(job_id=job_id):
            logger.info("score job started")
            try:
                input_data = self.database.get_score_job_input(job_id)
                if input_data is None:
                    raise NotFoundError(f"Score job {job_id} not found")
                result = self._score_pair(
                    gold_video_id=int(input_data["gold_video_id"]),
                    trainee_video_id=int(input_data["trainee_video_id"]),
                    weights_payload=input_data.get("weights"),
                )
                self.database.complete_score_job(job_id, result)
                total_ms = int((time.perf_counter() - started) * 1000)
                logger.info(
                    "score job completed score=%s",
                    result.get("score"),
                    extra={"score": result.get("score"), "duration_ms": total_ms},
                )
                self._fire_webhook(job_id, result)
            except Exception as exc:
                self.database.fail_score_job(job_id, str(exc))
                logger.exception("score job failed: %s", exc)
                raise  # re-raise so ScoreJobQueue can apply retry logic

    def requeue_pending_jobs(self, enqueue_fn: Callable[[int], None]) -> None:
        for job_id in self.database.list_pending_score_job_ids():
            enqueue_fn(job_id)

    def get_score_job(self, job_id: int) -> dict:
        from sopilot.core.score_result import enrich_score_result

        job = self.database.get_score_job(job_id)
        if not job:
            raise NotFoundError(f"Score job {job_id} not found")
        review = self.database.get_score_review(job_id)
        result = job["score"]
        if result is not None:
            enrich_score_result(result)
        return {
            "job_id": int(job["id"]),
            "status": job["status"],
            "result": result,
            "weights": job["weights"],
            "review": review,
            "error": job["error"],
            "created_at": job.get("created_at"),
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
        }

    def rerun_score_job(self, job_id: int) -> dict:
        """Create a new score job with the same inputs as an existing one."""
        job = self.database.get_score_job(job_id)
        if job is None:
            raise NotFoundError(f"Score job {job_id} not found")
        weights_payload = job.get("weights")
        weights = ScoreWeights(**weights_payload) if weights_payload else None
        return self.queue_score_job(
            gold_video_id=int(job["gold_video_id"]),
            trainee_video_id=int(job["trainee_video_id"]),
            weights=weights,
        )

    def cancel_score_job(self, job_id: int) -> dict:
        """Cancel a queued or running score job."""
        job = self.database.get_score_job(job_id)
        if job is None:
            raise NotFoundError(f"Score job {job_id} not found")
        if job["status"] not in ("queued", "running"):
            raise InvalidStateError(f"Cannot cancel job in '{job['status']}' status")
        if not self.database.cancel_score_job(job_id):
            raise InvalidStateError(f"Job {job_id} could not be cancelled")
        logger.info("score job cancelled job_id=%s", job_id)
        return self.get_score_job(job_id)

    def update_score_review(self, *, job_id: int, verdict: str, note: str | None) -> dict[str, Any]:
        job = self.database.get_score_job(job_id)
        if job is None:
            raise NotFoundError(f"Score job {job_id} not found")
        if job["status"] != "completed":
            raise InvalidStateError("Only completed jobs can be reviewed")
        review = self.database.upsert_score_review(job_id=job_id, verdict=verdict, note=note)
        logger.info("score review updated job_id=%s verdict=%s", job_id, verdict)
        return dict(review)

    def export_score_job(self, job_id: int) -> dict:
        payload = self.get_score_job(job_id)
        if payload["result"] is None:
            raise InvalidStateError("No score result to export")
        return payload

    @timed_cache(ttl_seconds=5.0)
    def list_score_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        items = self.database.list_score_jobs(status=status, limit=limit, offset=offset)
        total = self.database.count_score_jobs(status=status)
        return {"items": items, "total": total}

    # ------------------------------------------------------------------
    # Internal scoring methods
    # ------------------------------------------------------------------

    def _validate_score_pair(self, gold_video_id: int, trainee_video_id: int) -> str:
        gold_video = self.database.get_video(gold_video_id)
        trainee_video = self.database.get_video(trainee_video_id)

        if gold_video is None:
            raise NotFoundError(f"Gold video {gold_video_id} not found")
        if trainee_video is None:
            raise NotFoundError(f"Trainee video {trainee_video_id} not found")
        if gold_video["status"] != "ready":
            raise InvalidStateError(f"Gold video {gold_video_id} status is '{gold_video['status']}'")
        if trainee_video["status"] != "ready":
            raise InvalidStateError(f"Trainee video {trainee_video_id} status is '{trainee_video['status']}'")
        if not gold_video["is_gold"]:
            raise InvalidStateError(f"Video {gold_video_id} is not registered as gold")
        if gold_video["task_id"] != trainee_video["task_id"]:
            raise InvalidStateError("Gold and trainee videos must share the same task_id")
        self.video_service.assert_task_allowed(gold_video["task_id"])
        return str(gold_video["task_id"])

    def _score_pair(
        self,
        *,
        gold_video_id: int,
        trainee_video_id: int,
        weights_payload: dict | None,
    ) -> dict:
        gold_video, gold_clips, gold_embeddings, gold_boundaries = self.video_service.load_video_data(gold_video_id)
        trainee_video, trainee_clips, trainee_embeddings, trainee_boundaries = self.video_service.load_video_data(
            trainee_video_id
        )
        if not gold_video["is_gold"]:
            raise InvalidStateError(f"Video {gold_video_id} is not registered as gold")
        if gold_video["task_id"] != trainee_video["task_id"]:
            raise InvalidStateError("Gold and trainee videos must share the same task_id")

        task_id = str(gold_video["task_id"])
        profile = self._get_task_profile_for(task_id)
        alignment = dtw_align(gold_embeddings, trainee_embeddings)
        core_weights = self._core_weights_from_payload(weights_payload)
        result = score_alignment(
            alignment=alignment,
            gold_len=len(gold_embeddings),
            trainee_len=len(trainee_embeddings),
            gold_boundaries=gold_boundaries,
            trainee_boundaries=trainee_boundaries,
            weights=core_weights,
            deviation_threshold=self.settings.deviation_threshold,
        )
        result["gold_video_id"] = gold_video_id
        result["trainee_video_id"] = trainee_video_id
        result["task_id"] = task_id
        result["deviations"] = attach_timecodes(
            deviations=result.get("deviations", []),
            gold_clips=gold_clips,
            trainee_clips=trainee_clips,
        )
        result = apply_task_policy(
            result,
            profile=profile,
            efficiency_over_time_threshold=self.settings.efficiency_over_time_threshold,
            default_pass_score=self.settings.default_pass_score,
            default_retrain_score=self.settings.default_retrain_score,
        )

        # ------------------------------------------------------------------
        # Augment result with per-step score contributions (explanatory only)
        # ------------------------------------------------------------------
        boundaries_list: list[int] = result.get("boundaries", {}).get("gold", [])
        gold_emb_len: int = len(gold_embeddings)

        # Fetch step definitions (may be empty if not configured yet)
        step_defs: list[dict] = self.database.get_sop_steps(task_id)

        step_contributions = compute_step_contributions(
            result.get("deviations", []),
            boundaries_list,
            gold_emb_len,
            core_weights,
            step_definitions=step_defs if step_defs else None,
        )
        result["step_contributions"] = step_contributions

        # Per-step time compliance (only when step defs are configured)
        if step_defs and boundaries_list:
            time_compliance = compute_time_compliance_per_step(
                boundaries_list,
                step_defs,
                clip_seconds=self.settings.clip_seconds,
            )
            result["time_compliance"] = time_compliance

        # ------------------------------------------------------------------
        # Augment result with heuristic score confidence interval
        # ------------------------------------------------------------------
        confidence = compute_score_confidence(
            result.get("score"),
            result.get("deviations", []),
            result.get("metrics", {}),
            gold_len=gold_emb_len,
            trainee_len=len(trainee_embeddings),
        )
        if confidence is not None:
            result["confidence"] = confidence

        # ------------------------------------------------------------------
        # Augment result with statistical uncertainty decomposition
        # ------------------------------------------------------------------
        if _UNCERTAINTY_AVAILABLE and _compute_score_uncertainty is not None:
            try:
                score = result.get("score", 0.0) or 0.0
                metrics = result.get("metrics", {})
                uncertainty = _compute_score_uncertainty(
                    base_score=float(score),
                    dtw_cost=float(metrics.get("dtw_normalized_cost", 0.1)),
                    n_clips_gold=gold_emb_len,
                    n_clips_trainee=len(trainee_embeddings),
                    clip_embeddings_gold=gold_embeddings,
                    clip_embeddings_trainee=trainee_embeddings,
                )
                result["uncertainty"] = {
                    "epistemic": uncertainty.epistemic,
                    "aleatoric": uncertainty.aleatoric,
                    "total": uncertainty.total,
                    "ci_lower": uncertainty.bootstrap_ci.lower,
                    "ci_upper": uncertainty.bootstrap_ci.upper,
                    "ci_stability": uncertainty.bootstrap_ci.stability,
                    "note": uncertainty.note,
                }
            except Exception:
                pass  # Never fail scoring due to uncertainty computation

        return result

    def score_ensemble(
        self,
        *,
        gold_video_ids: list[int],
        trainee_video_id: int,
        weights_payload: dict | None = None,
    ) -> dict:
        """Score a trainee video against multiple gold videos and return ensemble result.

        Args:
            gold_video_ids: List of gold video IDs (1-10 supported).
            trainee_video_id: Trainee video ID to evaluate.
            weights_payload: Optional custom score weights.

        Returns:
            Dict with ensemble consensus score + per-gold breakdown.
        """
        from sopilot.core.ensemble import aggregate_ensemble

        if not gold_video_ids:
            raise InvalidStateError("gold_video_ids must not be empty")
        if len(gold_video_ids) > 10:
            raise InvalidStateError("Maximum 10 gold videos per ensemble scoring request")
        if len(set(gold_video_ids)) != len(gold_video_ids):
            raise InvalidStateError("gold_video_ids must not contain duplicates")

        # Score against each gold video
        individual_results = []
        errors = []
        for gold_id in gold_video_ids:
            try:
                result = self._score_pair(
                    gold_video_id=gold_id,
                    trainee_video_id=trainee_video_id,
                    weights_payload=weights_payload,
                )
                individual_results.append(result)
            except (NotFoundError, InvalidStateError) as e:
                errors.append({"gold_video_id": gold_id, "error": str(e)})

        if not individual_results:
            raise InvalidStateError(
                "All gold video scoring attempts failed: " + "; ".join(str(e["error"]) for e in errors)
            )

        ensemble = aggregate_ensemble(individual_results)

        return {
            "trainee_video_id": trainee_video_id,
            "gold_video_ids": gold_video_ids,
            "consensus_score": ensemble.consensus_score,
            "mean_score": ensemble.mean_score,
            "min_score": ensemble.min_score,
            "max_score": ensemble.max_score,
            "std_score": ensemble.std_score,
            "agreement": ensemble.agreement,
            "gold_count": ensemble.gold_count,
            "individual_scores": ensemble.individual_scores,
            "best_gold_video_id": ensemble.best_gold_video_id,
            "best_result": ensemble.best_result,
            "errors": errors,
            # Use the best-matching gold's decision for overall pass/fail
            "decision": ensemble.best_result.get("summary", {}).get("decision", "unknown"),
            "decision_source": "best_gold",
        }

    def get_score_uncertainty(self, job_id: int) -> dict:
        """Compute and return uncertainty decomposition for a completed score job.

        Retrieves score, dtw_cost, and clip counts from the stored result and
        runs compute_score_uncertainty.  Returns a plain dict with the
        uncertainty breakdown plus a note field.

        Raises NotFoundError if the job does not exist.
        Raises InvalidStateError if the job is not completed.
        """
        job = self.database.get_score_job(job_id)
        if job is None:
            raise NotFoundError(f"Score job {job_id} not found")
        if job["status"] != "completed":
            raise InvalidStateError(f"Score job {job_id} is not completed (status: {job['status']})")

        result_data = job.get("score") or {}
        score_val = float(result_data.get("score", 0.0) or 0.0)
        metrics = result_data.get("metrics") or {}
        dtw_cost = float(metrics.get("dtw_normalized_cost", 0.1))

        # Infer clip counts from stored input fields if available; otherwise 0
        input_data = self.database.get_score_job_input(job_id)
        n_gold = 0
        n_trainee = 0
        if input_data is not None:
            try:
                _, gold_clips, _, _ = self.video_service.load_video_data(int(input_data["gold_video_id"]))  # type: ignore[call-overload]
                n_gold = len(gold_clips)
            except Exception:
                pass
            try:
                _, trainee_clips, _, _ = self.video_service.load_video_data(int(input_data["trainee_video_id"]))  # type: ignore[call-overload]
                n_trainee = len(trainee_clips)
            except Exception:
                pass

        if _UNCERTAINTY_AVAILABLE and _compute_score_uncertainty is not None:
            try:
                uncertainty = _compute_score_uncertainty(
                    base_score=score_val,
                    dtw_cost=dtw_cost,
                    n_clips_gold=n_gold,
                    n_clips_trainee=n_trainee,
                )
                return {
                    "job_id": job_id,
                    "score": round(score_val, 2),
                    "uncertainty": {
                        "epistemic": uncertainty.epistemic,
                        "aleatoric": uncertainty.aleatoric,
                        "total": uncertainty.total,
                        "ci_lower": uncertainty.bootstrap_ci.lower,
                        "ci_upper": uncertainty.bootstrap_ci.upper,
                        "ci_stability": uncertainty.bootstrap_ci.stability,
                        "note": uncertainty.note,
                    },
                }
            except Exception as exc:
                logger.warning("uncertainty computation failed job_id=%s: %s", job_id, exc)

        # Fallback when uncertainty module unavailable
        return {
            "job_id": job_id,
            "score": round(score_val, 2),
            "uncertainty": None,
            "note": "Uncertainty module not available.",
        }

    def compute_soft_dtw(
        self,
        *,
        gold_video_id: int,
        trainee_video_id: int,
        gamma: float = 1.0,
    ) -> dict:
        """Run Soft-DTW on the stored embeddings of two videos.

        Args:
            gold_video_id:     Gold video ID.
            trainee_video_id:  Trainee video ID.
            gamma:             Soft-DTW smoothing temperature (> 0).

        Returns:
            Dict with soft_dtw_distance, normalized_cost, gamma, and
            alignment_path_length.

        Raises NotFoundError / InvalidStateError if either video is missing or
        not in 'ready' status.  Raises RuntimeError if the soft_dtw module is
        not available.
        """
        try:
            from sopilot.core.soft_dtw import soft_dtw as _soft_dtw_fn
        except ImportError as exc:
            raise RuntimeError("sopilot.core.soft_dtw is not available") from exc

        _, _, gold_embeddings, _ = self.video_service.load_video_data(gold_video_id)
        _, _, trainee_embeddings, _ = self.video_service.load_video_data(trainee_video_id)

        sdtw_result = _soft_dtw_fn(gold_embeddings, trainee_embeddings, gamma=gamma)
        return {
            "soft_dtw_distance": round(float(sdtw_result.distance), 6),
            "normalized_cost": round(float(sdtw_result.normalized_cost), 6),
            "gamma": sdtw_result.gamma,
            "alignment_path_length": len(sdtw_result.alignment_path),
        }

    def _apply_task_policy(self, result: dict, profile: dict) -> dict:
        return apply_task_policy(
            result,
            profile=profile,
            efficiency_over_time_threshold=self.settings.efficiency_over_time_threshold,
            default_pass_score=self.settings.default_pass_score,
            default_retrain_score=self.settings.default_retrain_score,
        )

    def _fire_webhook(self, job_id: int, result: dict) -> None:
        """Best-effort POST to configured webhook URL on score completion.

        Runs in a background thread with retry (3 attempts, exponential backoff)
        so that webhook failures never block scoring.
        """
        url = self.settings.webhook_url
        if not url:
            return
        payload = {
            "event": "score.completed",
            "job_id": job_id,
            "score": result.get("score"),
            "decision": (result.get("summary") or {}).get("decision"),
            "task_id": result.get("task_id"),
            "gold_video_id": result.get("gold_video_id"),
            "trainee_video_id": result.get("trainee_video_id"),
        }

        def _send_with_retry() -> None:
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                try:
                    data = json.dumps(payload).encode("utf-8")
                    req = urllib.request.Request(
                        url,
                        data=data,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        logger.info("webhook sent job_id=%s status=%s attempt=%s", job_id, resp.status, attempt)
                    return  # success
                except Exception as exc:
                    if attempt < max_attempts:
                        backoff = 2 ** (attempt - 1)  # 1s, 2s
                        logger.warning(
                            "webhook retry job_id=%s attempt=%s/%s backoff=%ss err=%s",
                            job_id, attempt, max_attempts, backoff, exc,
                        )
                        time.sleep(backoff)
                    else:
                        logger.warning(
                            "webhook failed after %s attempts job_id=%s url=%s err=%s",
                            max_attempts, job_id, url, exc,
                        )

        thread = threading.Thread(target=_send_with_retry, daemon=True)
        thread.start()

    def _core_weights_from_payload(self, payload: dict[str, Any] | None) -> CoreScoreWeights:
        if not payload:
            return CoreScoreWeights()
        return ScoreWeights(**payload).to_core_weights()  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Audit report generation
    # ------------------------------------------------------------------

    def get_score_report(self, job_id: int) -> str:
        """Generate a printable HTML audit report for a completed score job."""
        job = self.get_score_job(job_id)
        return build_report_html(job_id, job)

    def get_score_report_pdf(self, job_id: int) -> bytes:
        """Generate a PDF audit report for a completed score job."""
        job = self.get_score_job(job_id)
        return build_report_pdf(job_id, job)

    def get_score_timeline(self, job_id: int) -> dict:
        """Build a UI-friendly timeline from a completed score job.

        Returns a list of steps, each with gold/trainee time ranges,
        alignment quality, and deviation markers.
        """
        job = self.get_score_job(job_id)
        result = job.get("result")
        if result is None:
            raise InvalidStateError(f"Score job {job_id} has no result yet")
        return build_score_timeline(result, job_id)

    # ------------------------------------------------------------------
    # Batch re-decision
    # ------------------------------------------------------------------

    def rescore_decisions(
        self,
        *,
        task_id: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Re-apply make_decision() with current task profile thresholds to all stored results.

        Preserves the original score and deviations (with their existing severity
        assignments); only the summary block is updated.  Safe to re-run
        multiple times — idempotent when thresholds haven't changed.

        Args:
            task_id:  Restrict to jobs whose gold video belongs to this task.
                      None processes all tasks.
            dry_run:  If True, compute changes but do not write to the database.

        Returns:
            Dict with total, changed, breakdown (old->new transition counts),
            and dry_run flag.
        """
        from sopilot.core.score_pipeline import make_decision

        jobs = self.database.list_completed_score_jobs(task_id=task_id)
        total = len(jobs)
        changed = 0
        breakdown: dict[str, int] = {}

        for job in jobs:
            score_data = job.get("score") or {}
            raw_score = score_data.get("score")
            if raw_score is None:
                continue
            score_f = float(raw_score)

            job_task_id = job.get("task_id") or self.settings.primary_task_id
            profile = self._get_task_profile_for(job_task_id)
            pass_score = float(profile.get("pass_score") or self.settings.default_pass_score)
            retrain_score = float(profile.get("retrain_score") or self.settings.default_retrain_score)

            deviations = score_data.get("deviations") or []
            old_summary = score_data.get("summary") or {}
            old_decision = old_summary.get("decision")

            new_summary = make_decision(
                score=score_f,
                deviations=deviations,
                pass_score=pass_score,
                retrain_score=retrain_score,
            )
            new_decision = new_summary["decision"]

            if new_decision != old_decision:
                changed += 1
                key = f"{old_decision or 'unknown'} -> {new_decision}"
                breakdown[key] = breakdown.get(key, 0) + 1
                if not dry_run:
                    updated = dict(score_data)
                    updated["summary"] = new_summary
                    self.database.update_score_json(int(job["id"]), updated)

        return {
            "total_jobs_processed": total,
            "decisions_changed": changed,
            "breakdown": breakdown,
            "dry_run": dry_run,
        }
