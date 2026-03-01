import asyncio
import json as json_mod
import os
import shutil
import time
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

from sopilot import __version__
from sopilot.api.error_handling import service_errors
from sopilot.schemas import (
    BatchScoreRequest,
    ScoreJobListItem,
    ScoreJobListResponse,
    ScoreJobResponse,
    ScoreRequest,
    ScoreReviewRequest,
    ScoreReviewResponse,
    SearchResponse,
    TaskProfileResponse,
    TaskProfileUpdateRequest,
    VideoDetailResponse,
    VideoIngestResponse,
    VideoListItem,
    VideoListResponse,
    VideoUpdateRequest,
)
from sopilot.services.score_queue import ScoreJobQueue
from sopilot.services.sopilot_service import SOPilotService

_VALID_SCORE_STATUSES = frozenset({"queued", "running", "completed", "failed", "cancelled"})


def _service(request: Request) -> SOPilotService:
    svc: SOPilotService = request.app.state.sopilot_service
    return svc


def _queue(request: Request) -> ScoreJobQueue:
    q: ScoreJobQueue = request.app.state.score_queue
    return q


def build_router() -> APIRouter:
    router = APIRouter()

    @router.get("/readiness", tags=["Health"], summary="Readiness probe for container orchestration")
    def readiness(request: Request) -> JSONResponse:
        """Lightweight readiness check for Kubernetes/Docker.

        Returns 200 if the service is ready to accept traffic (embedder loaded,
        database reachable). Returns 503 if not ready.
        """
        ready = True
        details: dict[str, str] = {}

        # DB reachable
        try:
            service = _service(request)
            with service.database.connect() as conn:
                conn.execute("SELECT 1")
            details["database"] = "ok"
        except Exception:
            details["database"] = "unavailable"
            ready = False

        # Embedder loaded
        embedder = getattr(request.app.state, "embedder", None)
        if embedder and hasattr(embedder, "name"):
            stats = getattr(embedder, "get_stats", lambda: None)()
            if stats and stats.get("permanently_failed"):
                details["embedder"] = "permanently_failed"
                ready = False
            else:
                details["embedder"] = "ok"
        else:
            details["embedder"] = "unavailable"
            ready = False

        if not ready:
            return JSONResponse(
                {"ready": False, "details": details},
                status_code=503,
            )
        return JSONResponse({"ready": True, "details": details}, status_code=200)

    @router.get("/health", tags=["Health"], summary="Run health checks")
    def health(request: Request) -> dict:
        """Check database, disk, and embedder health and return overall status."""
        checks: dict[str, dict] = {}
        overall = "healthy"

        # ── DB check ──────────────────────────────────────────────────
        try:
            service = _service(request)
            t0 = time.perf_counter()
            with service.database.connect() as conn:
                conn.execute("SELECT 1")
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            checks["database"] = {"status": "up", "latency_ms": latency_ms}
        except Exception:
            checks["database"] = {"status": "down"}
            overall = "unhealthy"

        # ── Disk check ────────────────────────────────────────────────
        try:
            settings = request.app.state.settings
            data_dir = settings.data_dir
            writable = os.access(str(data_dir), os.W_OK) if data_dir.exists() else False
            usage = shutil.disk_usage(str(data_dir))
            free_gb = round(usage.free / (1024 ** 3), 1)
            total_gb = round(usage.total / (1024 ** 3), 1)
            checks["disk"] = {
                "status": "up" if writable else "down",
                "free_gb": free_gb,
                "total_gb": total_gb,
                "writable": writable,
            }
            if not writable:
                overall = "unhealthy"
        except Exception:
            checks["disk"] = {"status": "down", "writable": False}
            overall = "unhealthy"

        # ── Embedder check ────────────────────────────────────────────
        try:
            embedder = getattr(request.app.state, "embedder", None)
            if embedder is None or not hasattr(embedder, "name"):
                raise RuntimeError("embedder not available")
            embed_name = embedder.name
            failed_over = False
            stats = getattr(embedder, "get_stats", lambda: None)()
            if stats:
                failed_over = bool(stats.get("failed_over", False))
                permanently_failed = bool(stats.get("permanently_failed", False))
                if permanently_failed:
                    checks["embedder"] = {
                        "status": "down",
                        "name": embed_name,
                        "failed_over": failed_over,
                    }
                    overall = "unhealthy"
                elif failed_over:
                    checks["embedder"] = {
                        "status": "degraded",
                        "name": embed_name,
                        "failed_over": failed_over,
                    }
                    if overall == "healthy":
                        overall = "degraded"
                else:
                    checks["embedder"] = {
                        "status": "up",
                        "name": embed_name,
                        "failed_over": False,
                    }
            else:
                checks["embedder"] = {
                    "status": "up",
                    "name": embed_name,
                    "failed_over": False,
                }
        except Exception:
            checks["embedder"] = {"status": "down"}
            overall = "unhealthy"

        return {
            "status": overall,
            "checks": checks,
            "version": __version__,
        }

    @router.get("/status", tags=["Health"], summary="Get service status")
    async def get_status(request: Request) -> dict:
        """Return version, embedder info, queue depth, and configuration summary."""
        settings = request.app.state.settings
        queue = _queue(request)
        embedder = getattr(request.app.state, "embedder", None)
        return {
            "version": __version__,
            "primary_task_id": settings.primary_task_id,
            "embedder": getattr(embedder, "name", "unknown"),
            "embedder_stats": getattr(embedder, "get_stats", lambda: None)(),
            "queue_depth": queue.depth(),
            "data_dir": str(settings.data_dir),
        }

    @router.get("/metrics", tags=["Health"], summary="Prometheus metrics endpoint")
    async def get_metrics(request: Request) -> PlainTextResponse:
        """Prometheus/OpenMetrics compatible text exposition.

        Scrape with: ``prometheus.yml`` -> ``scrape_configs`` -> ``targets: ["host:8000"]``
        or ``curl http://localhost:8000/metrics``.
        """
        settings = request.app.state.settings
        queue = _queue(request)
        embedder = getattr(request.app.state, "embedder", None)

        embed_stats: dict = getattr(embedder, "get_stats", lambda: {})() or {}
        queue_depth: int = queue.depth()

        lines: list[str] = []

        def _metric(name: str, value: object, help_text: str, kind: str = "gauge") -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {kind}")
            lines.append(f"{name} {value}")

        # ── queue ──────────────────────────────────────────────────────────
        _metric("sopilot_queue_depth", queue_depth,
                "Number of score jobs currently waiting in the queue")

        # ── embedder ───────────────────────────────────────────────────────
        _metric("sopilot_embed_requests_total", embed_stats.get("total_requests", 0),
                "Total embed() calls received by the embedder", "counter")
        _metric("sopilot_embed_primary_successes_total",
                embed_stats.get("primary_successes", 0),
                "Embed calls served by the primary (V-JEPA2) embedder", "counter")
        _metric("sopilot_embed_fallback_uses_total",
                embed_stats.get("fallback_uses", 0),
                "Embed calls routed to the fallback (color-motion) embedder", "counter")
        _metric("sopilot_embed_failure_capsules_total",
                embed_stats.get("failure_capsules_written", 0),
                "Number of failure capsule records written to disk", "counter")
        _metric("sopilot_embed_failed_over",
                1 if embed_stats.get("failed_over", False) else 0,
                "1 if the embedder is currently running on the fallback path")
        _metric("sopilot_embed_permanently_failed",
                1 if embed_stats.get("permanently_failed", False) else 0,
                "1 if the primary embedder has permanently failed (max retries exceeded)")
        _metric("sopilot_embed_retry_interval_seconds",
                embed_stats.get("current_retry_interval_sec", 0),
                "Current exponential-backoff retry interval in seconds")

        # ── VRAM (CUDA only) ───────────────────────────────────────────────
        try:
            import torch
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                _metric("sopilot_vram_free_bytes", free_bytes,
                        "Free GPU VRAM in bytes")
                _metric("sopilot_vram_total_bytes", total_bytes,
                        "Total GPU VRAM in bytes")
        except Exception:
            pass

        # ── service info ───────────────────────────────────────────────────
        embedder_name = str(getattr(embedder, "name", "unknown")).replace('"', "")
        lines.append("# HELP sopilot_info Static service metadata")
        lines.append("# TYPE sopilot_info gauge")
        lines.append(
            f'sopilot_info{{version="{__version__}",'
            f'task_id="{settings.primary_task_id}",'
            f'embedder="{embedder_name}"}} 1'
        )

        lines.append("")  # trailing newline
        return PlainTextResponse(
            "\n".join(lines),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @router.get("/task-profile", response_model=TaskProfileResponse, tags=["Configuration"], summary="Get task profile")
    @service_errors
    async def get_task_profile(request: Request) -> TaskProfileResponse:
        """Retrieve the current task profile including score thresholds and weights."""
        service = _service(request)
        payload = await run_in_threadpool(service.get_task_profile)
        return TaskProfileResponse(**payload)

    @router.put("/task-profile", response_model=TaskProfileResponse, tags=["Configuration"], summary="Update task profile")
    @service_errors
    async def update_task_profile(request: Request, body: TaskProfileUpdateRequest) -> TaskProfileResponse:
        """Update task profile fields such as thresholds, weights, or deviation policy."""
        service = _service(request)
        payload = await run_in_threadpool(
            service.update_task_profile,
            task_name=body.task_name,
            pass_score=body.pass_score,
            retrain_score=body.retrain_score,
            default_weights=body.default_weights.model_dump() if body.default_weights else None,
            deviation_policy=body.deviation_policy,
        )
        return TaskProfileResponse(**payload)

    @router.get("/dataset/summary", tags=["Configuration"], summary="Get dataset summary")
    @service_errors
    async def dataset_summary(request: Request) -> dict:
        """Return aggregate statistics about ingested videos and score jobs."""
        service = _service(request)
        return await run_in_threadpool(service.get_dataset_summary)

    @router.get("/analytics", tags=["Configuration"], summary="Get scoring analytics")
    @service_errors
    async def get_analytics(
        request: Request,
        task_id: str | None = Query(default=None),
        days: int | None = Query(default=None, ge=1, le=365, description="Filter to last N days"),
    ) -> dict:
        """Return aggregate scoring analytics for dashboard visualization."""
        service = _service(request)
        return await run_in_threadpool(service.get_analytics, task_id=task_id, days=days)

    @router.get("/analytics/compliance", tags=["Analytics"], summary="Get SOP compliance overview")
    @service_errors
    async def get_compliance(
        request: Request,
        task_id: str | None = Query(default=None),
        days: int | None = Query(default=None, ge=1, le=365),
    ) -> dict:
        service = _service(request)
        return await run_in_threadpool(service.get_compliance_overview, task_id=task_id, days=days)

    @router.get("/analytics/steps", tags=["Analytics"], summary="Get per-step difficulty analysis")
    @service_errors
    async def get_step_performance(
        request: Request,
        task_id: str | None = Query(default=None),
        days: int | None = Query(default=None, ge=1, le=365),
    ) -> dict:
        service = _service(request)
        return await run_in_threadpool(service.get_step_performance, task_id=task_id, days=days)

    @router.get("/analytics/operators/{operator_id}/trend", tags=["Analytics"], summary="Get operator score trend")
    @service_errors
    async def get_operator_trend(
        request: Request,
        operator_id: str,
        task_id: str | None = Query(default=None),
        limit: int = Query(default=20, ge=1, le=100),
    ) -> dict:
        service = _service(request)
        return await run_in_threadpool(service.get_operator_trend, operator_id=operator_id, task_id=task_id)

    @router.get("/analytics/recommendations/{operator_id}", tags=["Analytics"], summary="Get training recommendations for operator")
    @service_errors
    async def get_recommendations(
        request: Request,
        operator_id: str,
        task_id: str | None = Query(default=None),
    ) -> dict:
        service = _service(request)
        return await run_in_threadpool(service.get_recommendations, operator_id=operator_id, task_id=task_id)

    @router.get("/analytics/operators/{operator_id}/projection", tags=["Analytics"], summary="Predict operator certification pathway")
    @service_errors
    async def get_operator_projection(
        request: Request,
        operator_id: str,
        task_id: str | None = Query(default=None),
    ) -> dict:
        """Analyze an operator's score trajectory and predict when they'll reach certification.

        Returns trend analysis, confidence level, projected scores for the next 5 evaluations,
        and estimated number of evaluations needed to reach the pass threshold.
        """
        service = _service(request)
        return await run_in_threadpool(
            service.get_operator_learning_curve,
            operator_id=operator_id,
            task_id=task_id,
        )

    @router.get("/tasks/steps", tags=["SOP Steps"], summary="Get SOP step definitions for the primary task")
    @service_errors
    async def get_sop_steps(
        request: Request,
        task_id: str | None = Query(default=None),
    ) -> dict:
        """Return named step definitions with expected durations for a task."""
        service = _service(request)
        tid = task_id or service.settings.primary_task_id
        return await run_in_threadpool(service.get_sop_steps, tid)

    @router.put("/tasks/steps", tags=["SOP Steps"], summary="Upsert SOP step definitions")
    @service_errors
    async def upsert_sop_steps(
        request: Request,
        body: dict,
        task_id: str | None = Query(default=None),
    ) -> dict:
        """Create or update step definitions (name, expected duration, critical flag)."""
        service = _service(request)
        tid = task_id or service.settings.primary_task_id
        steps = body.get("steps", [])
        if not isinstance(steps, list):
            raise HTTPException(status_code=422, detail="'steps' must be a list")
        return await run_in_threadpool(service.upsert_sop_steps, tid, steps)

    @router.delete("/tasks/steps", tags=["SOP Steps"], summary="Delete all SOP step definitions")
    @service_errors
    async def delete_sop_steps(
        request: Request,
        task_id: str | None = Query(default=None),
    ) -> dict:
        """Delete all step definitions for a task."""
        service = _service(request)
        tid = task_id or service.settings.primary_task_id
        return await run_in_threadpool(service.delete_sop_steps, tid)

    @router.post("/score/ensemble", tags=["Scoring"], summary="Score trainee video against multiple gold videos")
    @service_errors
    async def score_ensemble(
        request: Request,
        body: dict,
    ) -> dict:
        """Score a trainee video against multiple gold reference videos.

        Returns a consensus score (median) plus individual per-gold scores and
        an agreement metric indicating how consistent the gold videos are with each other.

        Request body:
        - gold_video_ids: list[int] — 1-10 gold video IDs
        - trainee_video_id: int — trainee video ID
        - weights: optional score weight overrides
        """
        service = _service(request)
        gold_video_ids = body.get("gold_video_ids", [])
        trainee_video_id = body.get("trainee_video_id")
        if not isinstance(gold_video_ids, list) or not gold_video_ids:
            raise HTTPException(status_code=422, detail="gold_video_ids must be a non-empty list")
        if not isinstance(trainee_video_id, int):
            raise HTTPException(status_code=422, detail="trainee_video_id must be an integer")
        weights_payload = body.get("weights")
        return await run_in_threadpool(
            service.score_ensemble,
            gold_video_ids=gold_video_ids,
            trainee_video_id=trainee_video_id,
            weights_payload=weights_payload,
        )

    @router.get("/analytics/report/pdf", tags=["Analytics"], summary="Generate executive analytics PDF report")
    @service_errors
    async def get_analytics_report_pdf(
        request: Request,
        task_id: str | None = Query(default=None),
        days: int | None = Query(default=None, ge=1, le=365, description="Filter to last N days"),
    ) -> Response:
        """Generate a multi-page executive PDF report with compliance KPIs, operator rankings, and trends."""
        from sopilot.core.analytics_report_pdf import generate_analytics_pdf

        service = _service(request)
        analytics = await run_in_threadpool(service.get_analytics, task_id=task_id, days=days)
        pdf_bytes = await run_in_threadpool(generate_analytics_pdf, analytics)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="sopilot_analytics_report.pdf"'},
        )

    @router.post("/videos", response_model=VideoIngestResponse, tags=["Videos"], summary="Upload a trainee video")
    @service_errors
    async def upload_video(
        request: Request,
        file: UploadFile = File(...),  # noqa: B008
        task_id: str = Form(...),
        site_id: str | None = Form(default=None),
        camera_id: str | None = Form(default=None),
        operator_id_hash: str | None = Form(default=None),
        recorded_at: str | None = Form(default=None),
    ) -> VideoIngestResponse:
        """Upload and process a trainee video for SOP scoring."""
        service = _service(request)
        payload = await run_in_threadpool(
            service.ingest_video,
            original_filename=file.filename or "upload",
            file_obj=file.file,
            task_id=task_id,
            site_id=site_id,
            camera_id=camera_id,
            operator_id_hash=operator_id_hash,
            recorded_at=recorded_at,
            is_gold=False,
        )
        return VideoIngestResponse(**payload)

    @router.post("/gold", response_model=VideoIngestResponse, tags=["Videos"], summary="Upload a gold (reference) video")
    @service_errors
    async def upload_gold(
        request: Request,
        file: UploadFile = File(...),  # noqa: B008
        task_id: str = Form(...),
        site_id: str | None = Form(default=None),
        camera_id: str | None = Form(default=None),
        operator_id_hash: str | None = Form(default=None),
        recorded_at: str | None = Form(default=None),
    ) -> VideoIngestResponse:
        """Upload and process a gold (reference) video that defines the correct SOP execution."""
        service = _service(request)
        payload = await run_in_threadpool(
            service.ingest_video,
            original_filename=file.filename or "upload",
            file_obj=file.file,
            task_id=task_id,
            site_id=site_id,
            camera_id=camera_id,
            operator_id_hash=operator_id_hash,
            recorded_at=recorded_at,
            is_gold=True,
        )
        return VideoIngestResponse(**payload)

    @router.get("/videos", response_model=VideoListResponse, tags=["Videos"], summary="List videos")
    @service_errors
    async def list_videos(
        request: Request,
        site_id: str | None = Query(default=None),
        is_gold: bool | None = Query(default=None),
        limit: int = Query(default=200, ge=1, le=2000),
    ) -> VideoListResponse:
        """List ingested videos with optional filters for site and gold status."""
        service = _service(request)
        payload = await run_in_threadpool(
            service.list_videos,
            site_id=site_id,
            is_gold=is_gold,
            limit=limit,
        )
        total = await run_in_threadpool(
            service.count_videos,
            site_id=site_id,
            is_gold=is_gold,
        )
        items = [VideoListItem(**item) for item in payload]
        return VideoListResponse(
            items=items,
            total=total,
            has_more=len(items) >= limit,
        )

    @router.get("/videos/{video_id}", response_model=VideoDetailResponse, tags=["Videos"], summary="Get video details")
    @service_errors
    async def get_video(request: Request, video_id: int) -> VideoDetailResponse:
        """Retrieve full metadata for a single video by its ID."""
        service = _service(request)
        payload = await run_in_threadpool(service.get_video_detail, video_id=video_id)
        return VideoDetailResponse(**payload)

    @router.patch("/videos/{video_id}", response_model=VideoDetailResponse, tags=["Videos"], summary="Update video metadata")
    @service_errors
    async def update_video(request: Request, video_id: int, body: VideoUpdateRequest) -> VideoDetailResponse:
        """Update mutable metadata fields (site_id, camera_id, operator, recorded_at) on an existing video."""
        service = _service(request)
        payload = await run_in_threadpool(
            service.update_video_metadata,
            video_id,
            site_id=body.site_id,
            camera_id=body.camera_id,
            operator_id_hash=body.operator_id_hash,
            recorded_at=body.recorded_at,
        )
        return VideoDetailResponse(**payload)

    @router.delete("/videos/{video_id}", tags=["Videos"], summary="Delete a video")
    @service_errors
    async def delete_video(
        request: Request,
        video_id: int,
        force: bool = Query(default=False, description="Also delete associated score jobs and reviews"),
    ) -> dict:
        """Delete a video and its clips.

        Without ``force=true``, fails if the video is referenced by score jobs.
        With ``force=true``, also deletes associated score jobs and reviews.
        """
        service = _service(request)
        return await run_in_threadpool(service.delete_video, video_id=video_id, force=force)

    @router.get("/videos/{video_id}/stream", tags=["Videos"], summary="Stream video file")
    @service_errors
    async def stream_video(request: Request, video_id: int) -> FileResponse:
        """Stream the raw video file for playback or download."""
        service = _service(request)
        path = await run_in_threadpool(service.get_video_stream_path, video_id=video_id)
        return FileResponse(path=path)

    @router.get("/score", response_model=ScoreJobListResponse, tags=["Scoring"], summary="List score jobs")
    @service_errors
    async def list_score_jobs(
        request: Request,
        status: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> ScoreJobListResponse:
        """List score jobs with optional status filter and pagination."""
        if status is not None and status not in _VALID_SCORE_STATUSES:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid status filter '{status}'. Must be one of: {', '.join(sorted(_VALID_SCORE_STATUSES))}",
            )
        service = _service(request)
        payload = await run_in_threadpool(
            service.list_score_jobs, status=status, limit=limit, offset=offset
        )
        items = [ScoreJobListItem(**item) for item in payload["items"]]
        return ScoreJobListResponse(
            items=items,
            total=payload.get("total", 0),
            has_more=len(items) >= limit,
        )

    @router.post("/score", response_model=ScoreJobResponse, tags=["Scoring"], summary="Submit a score job")
    @service_errors
    async def score_video(request: Request, body: ScoreRequest) -> ScoreJobResponse:
        """Queue a DTW-based scoring job comparing a trainee video against a gold video."""
        service = _service(request)
        queue = _queue(request)
        payload = await run_in_threadpool(
            service.queue_score_job,
            gold_video_id=body.gold_video_id,
            trainee_video_id=body.trainee_video_id,
            weights=body.weights,
        )
        queue.enqueue(payload["job_id"])
        return ScoreJobResponse(**payload)

    @router.post("/score/batch", tags=["Scoring"], summary="Submit batch score jobs")
    @service_errors
    async def batch_score(request: Request, body: BatchScoreRequest) -> dict:
        """Queue multiple score jobs at once. Returns a list of job IDs."""
        service = _service(request)
        queue = _queue(request)
        jobs: list[dict] = []
        for pair in body.pairs:
            payload = await run_in_threadpool(
                service.queue_score_job,
                gold_video_id=pair.gold_video_id,
                trainee_video_id=pair.trainee_video_id,
                weights=pair.weights,
            )
            queue.enqueue(payload["job_id"])
            jobs.append({"job_id": payload["job_id"], "status": payload["status"]})
        return {"jobs": jobs, "count": len(jobs)}

    @router.post("/score/{job_id}/rerun", response_model=ScoreJobResponse, tags=["Scoring"], summary="Re-run a score job")
    @service_errors
    async def rerun_score_job(request: Request, job_id: int) -> ScoreJobResponse:
        """Create a new score job with the same gold/trainee pair and weights as an existing job.

        Useful for re-scoring after updating task profile thresholds or weights.
        """
        service = _service(request)
        queue = _queue(request)
        payload = await run_in_threadpool(service.rerun_score_job, job_id=job_id)
        queue.enqueue(payload["job_id"])
        return ScoreJobResponse(**payload)

    @router.post("/score/{job_id}/cancel", tags=["Scoring"], summary="Cancel a score job")
    @service_errors
    async def cancel_score_job(request: Request, job_id: int) -> dict:
        """Cancel a queued or running score job. Completed/failed jobs cannot be cancelled."""
        service = _service(request)
        return await run_in_threadpool(service.cancel_score_job, job_id=job_id)

    @router.get("/score/{job_id}", response_model=ScoreJobResponse, tags=["Scoring"], summary="Get score job result")
    @service_errors
    async def get_score_job(request: Request, job_id: int) -> ScoreJobResponse:
        """Retrieve the status and result of a specific score job."""
        service = _service(request)
        payload = await run_in_threadpool(service.get_score_job, job_id=job_id)
        return ScoreJobResponse(**payload)

    @router.put("/score/{job_id}/review", response_model=ScoreReviewResponse, tags=["Scoring"], summary="Submit or update a score review")
    @service_errors
    async def put_score_review(
        request: Request,
        job_id: int,
        body: ScoreReviewRequest,
    ) -> ScoreReviewResponse:
        """Submit or update a human reviewer verdict for a completed score job."""
        service = _service(request)
        payload = await run_in_threadpool(
            service.update_score_review,
            job_id=job_id,
            verdict=body.verdict,
            note=body.note,
        )
        return ScoreReviewResponse(**payload)

    @router.get("/score/{job_id}/export", tags=["Scoring"], summary="Export score job data")
    @service_errors
    async def export_score_job(request: Request, job_id: int) -> JSONResponse:
        """Export a score job as a self-contained JSON document for archival or integration."""
        service = _service(request)
        payload = await run_in_threadpool(service.export_score_job, job_id=job_id)
        return JSONResponse(payload)

    @router.get("/score/export/csv", tags=["Scoring"], summary="Export completed scores as CSV")
    @service_errors
    async def export_scores_csv(
        request: Request,
        task_id: str | None = Query(default=None),
    ) -> Response:
        """Download all completed score jobs as a CSV file for analysis in Excel or Google Sheets."""
        import csv
        import io

        service = _service(request)
        rows = await run_in_threadpool(
            service.database.list_completed_score_jobs, task_id=task_id
        )
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "job_id", "task_id", "gold_video_id", "trainee_video_id",
            "operator_id_hash", "site_id",
            "score", "decision", "miss_steps", "swap_steps", "deviation_steps",
            "over_time_ratio", "dtw_normalized_cost", "created_at", "finished_at",
        ])
        for row in rows:
            score_data = row.get("score") or {}
            summary = score_data.get("summary") or {}
            metrics = score_data.get("metrics") or {}
            writer.writerow([
                row["id"],
                row.get("task_id", ""),
                row["gold_video_id"],
                row["trainee_video_id"],
                row.get("operator_id_hash", ""),
                row.get("trainee_site_id", ""),
                score_data.get("score", ""),
                summary.get("decision", ""),
                metrics.get("miss_steps", ""),
                metrics.get("swap_steps", ""),
                metrics.get("deviation_steps", ""),
                metrics.get("over_time_ratio", ""),
                metrics.get("dtw_normalized_cost", ""),
                row.get("created_at", ""),
                row.get("finished_at", ""),
            ])
        csv_content = buf.getvalue()
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="sopilot_scores.csv"'},
        )

    @router.get("/score/{job_id}/timeline", tags=["Scoring"], summary="Get score alignment timeline")
    @service_errors
    async def get_score_timeline(request: Request, job_id: int) -> dict:
        """Return a step-by-step alignment timeline for visualization.

        Each step includes gold/trainee clip ranges, alignment status (ok/missing/deviation/swapped),
        and any deviation details.
        """
        service = _service(request)
        return await run_in_threadpool(service.get_score_timeline, job_id=job_id)

    @router.get("/score/{job_id}/report", tags=["Scoring"], summary="Get score report HTML")
    @service_errors
    async def get_score_report(request: Request, job_id: int) -> HTMLResponse:
        """Generate and return an HTML audit report for a completed score job."""
        service = _service(request)
        html = await run_in_threadpool(service.get_score_report, job_id=job_id)
        return HTMLResponse(content=html)

    @router.get("/score/{job_id}/report/pdf", tags=["Scoring"], summary="Download score report as PDF")
    @service_errors
    async def get_score_report_pdf(request: Request, job_id: int) -> Response:
        """Generate and download a PDF audit report for a completed score job."""
        service = _service(request)
        pdf_bytes = await run_in_threadpool(service.get_score_report_pdf, job_id=job_id)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="sopilot_job{job_id}_report.pdf"'},
        )

    @router.get("/search", response_model=SearchResponse, tags=["Search"], summary="Search similar clips")
    @service_errors
    async def search(
        request: Request,
        video_id: int = Query(..., ge=1),
        clip_index: int = Query(..., ge=0),
        k: int = Query(default=5, ge=1, le=50),
        task_id: str | None = Query(default=None),
    ) -> SearchResponse:
        """Find the k most similar clips to a given query clip using embedding similarity."""
        service = _service(request)
        payload = await run_in_threadpool(
            service.search,
            query_video_id=video_id,
            query_clip_index=clip_index,
            k=k,
            task_id=task_id,
        )
        return SearchResponse(**payload)

    # ── SSE: score job progress stream ───────────────────────────────
    @router.get("/score/{job_id}/stream", tags=["Scoring"], summary="Stream score job progress via SSE")
    async def score_job_stream(
        request: Request,
        job_id: int,
        poll_interval: float = Query(default=1.0, ge=0.5, le=10.0),
    ) -> StreamingResponse:
        """Server-Sent Events stream for score job progress.

        Streams status updates until the job reaches a terminal state
        (completed or failed), then sends a final event and closes.
        """
        service = _service(request)

        async def event_generator() -> Any:
            last_status = None
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await run_in_threadpool(service.get_score_job, job_id=job_id)
                    current_status = payload.get("status")
                    # Only emit when status changes or on first poll
                    if current_status != last_status:
                        last_status = current_status
                        event_data = {
                            "job_id": payload.get("job_id"),
                            "status": current_status,
                        }
                        if current_status == "completed" and payload.get("result"):
                            result = payload["result"]
                            event_data["score"] = result.get("score")
                            decision = (result.get("summary") or {}).get("decision")
                            event_data["decision"] = decision
                        if current_status == "failed":
                            event_data["error"] = payload.get("error")
                        yield f"event: status\ndata: {json_mod.dumps(event_data)}\n\n"
                        # Terminal states: send done event and close
                        if current_status in ("completed", "failed", "cancelled"):
                            yield f"event: done\ndata: {json_mod.dumps({'job_id': job_id, 'final_status': current_status})}\n\n"
                            break
                except Exception as exc:
                    yield f"event: error\ndata: {json_mod.dumps({'error': str(exc)})}\n\n"
                    break
                await asyncio.sleep(poll_interval)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Research / uncertainty endpoints ────────────────────────────

    @router.get("/score/{job_id}/uncertainty", tags=["Scoring"], summary="Get uncertainty breakdown for a score job")
    @service_errors
    async def get_score_uncertainty(request: Request, job_id: int) -> dict:
        """Return the epistemic/aleatoric uncertainty decomposition for a completed score job.

        Computes a bootstrap-inspired confidence interval and separates model
        uncertainty (DTW alignment quality) from data uncertainty (clip count,
        embedding spread).

        Returns::

            {
              "job_id": int,
              "score": float,
              "uncertainty": {
                "epistemic": float,
                "aleatoric": float,
                "total": float,
                "ci_lower": float,
                "ci_upper": float,
                "ci_stability": "high" | "medium" | "low",
                "note": str
              }
            }

        Returns 404 if the job does not exist, 422 if it is not completed.
        """
        service = _service(request)
        return await run_in_threadpool(service.get_score_uncertainty, job_id)

    @router.post("/research/soft-dtw", tags=["Research"], summary="Compute Soft-DTW distance between two videos")
    @service_errors
    async def research_soft_dtw(request: Request, body: dict) -> dict:
        """Compute the differentiable Soft-DTW alignment distance between the stored embeddings of two videos.

        Soft-DTW (Cuturi & Blondel, ICML 2017) is a smooth generalisation of
        hard DTW.  The ``gamma`` parameter controls the smoothing temperature:
        smaller values approach hard DTW, larger values approach Euclidean
        distance.

        Request body:

        - ``gold_video_id`` (int, required) — ID of the gold reference video.
        - ``trainee_video_id`` (int, required) — ID of the trainee video.
        - ``gamma`` (float, optional, default 1.0) — Smoothing temperature (> 0).

        Returns::

            {
              "soft_dtw_distance": float,
              "normalized_cost": float,
              "gamma": float,
              "alignment_path_length": int
            }

        Returns 404 if either video is not found, 422 on validation errors,
        503 if the soft_dtw module is unavailable.
        """
        gold_video_id = body.get("gold_video_id")
        trainee_video_id = body.get("trainee_video_id")
        gamma = float(body.get("gamma", 1.0))

        if not isinstance(gold_video_id, int):
            raise HTTPException(status_code=422, detail="gold_video_id must be an integer")
        if not isinstance(trainee_video_id, int):
            raise HTTPException(status_code=422, detail="trainee_video_id must be an integer")
        if gamma <= 0:
            raise HTTPException(status_code=422, detail="gamma must be > 0")

        service = _service(request)
        try:
            result = await run_in_threadpool(
                service.compute_soft_dtw,
                gold_video_id=gold_video_id,
                trainee_video_id=trainee_video_id,
                gamma=gamma,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return result

    # ── Admin endpoints ─────────────────────────────────────────────
    @router.post("/admin/backup", tags=["Admin"], summary="Create database backup")
    @service_errors
    async def create_backup(request: Request) -> dict:
        """Create a hot backup of the SQLite database.

        Returns the backup file path. Useful for scheduled backups or pre-migration snapshots.
        """
        import time as _time
        service = _service(request)
        settings = request.app.state.settings
        ts = _time.strftime("%Y%m%d_%H%M%S")
        backup_path = str(settings.data_dir / f"backup_{ts}.db")
        await run_in_threadpool(service.database.backup, backup_path)
        return {"backup_path": backup_path, "timestamp": ts}

    @router.post("/admin/optimize", tags=["Admin"], summary="Optimize database")
    @service_errors
    async def optimize_database(request: Request) -> dict:
        """Run VACUUM and ANALYZE on the SQLite database to reclaim space and update statistics."""
        service = _service(request)
        await run_in_threadpool(service.database.vacuum)
        return {"status": "optimized"}

    @router.get("/admin/db-stats", tags=["Admin"], summary="Get database statistics")
    @service_errors
    async def db_stats(request: Request) -> dict:
        """Return row counts for each table and database file size."""
        service = _service(request)
        stats = await run_in_threadpool(service.database.get_stats)
        db_size_bytes = stats.pop("_db_size_bytes", None)
        db_size_human = stats.pop("_db_size_human", None)
        return {
            "tables": stats,
            "db_size_bytes": db_size_bytes,
            "db_size_human": db_size_human,
        }

    return router
