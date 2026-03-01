"""Score job management, results, reviews, reports, and research endpoints."""

import asyncio
import csv
import io
import json as json_mod
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

from sopilot.api.error_handling import service_errors
from sopilot.api.routes._helpers import get_queue, get_service
from sopilot.schemas import (
    BatchScoreRequest,
    EnsembleScoreRequest,
    ScoreJobListItem,
    ScoreJobListResponse,
    ScoreJobResponse,
    ScoreRequest,
    ScoreReviewRequest,
    ScoreReviewResponse,
    SearchResponse,
    SoftDTWRequest,
)

_VALID_SCORE_STATUSES = frozenset({"queued", "running", "completed", "failed", "cancelled"})


def build_scoring_router() -> APIRouter:
    router = APIRouter()

    # ── Score CRUD ─────────────────────────────────────────────────

    @router.get("/score", response_model=ScoreJobListResponse, tags=["Scoring"], summary="List score jobs")
    @service_errors
    async def list_score_jobs(
        request: Request,
        status: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> ScoreJobListResponse:
        if status is not None and status not in _VALID_SCORE_STATUSES:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid status filter '{status}'. Must be one of: {', '.join(sorted(_VALID_SCORE_STATUSES))}",
            )
        service = get_service(request)
        payload = await run_in_threadpool(
            service.list_score_jobs, status=status, limit=limit, offset=offset
        )
        items = [ScoreJobListItem(**item) for item in payload["items"]]
        return ScoreJobListResponse(items=items, total=payload.get("total", 0), has_more=len(items) >= limit)

    @router.post("/score", response_model=ScoreJobResponse, tags=["Scoring"], summary="Submit a score job")
    @service_errors
    async def score_video(request: Request, body: ScoreRequest) -> ScoreJobResponse:
        service = get_service(request)
        queue = get_queue(request)
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
        service = get_service(request)
        queue = get_queue(request)
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

    @router.post("/score/ensemble", tags=["Scoring"], summary="Score trainee video against multiple gold videos")
    @service_errors
    async def score_ensemble(request: Request, body: EnsembleScoreRequest) -> dict:
        service = get_service(request)
        return await run_in_threadpool(
            service.score_ensemble,
            gold_video_ids=body.gold_video_ids,
            trainee_video_id=body.trainee_video_id,
            weights_payload=body.weights.model_dump() if body.weights else None,
        )

    @router.post("/score/{job_id}/rerun", response_model=ScoreJobResponse, tags=["Scoring"], summary="Re-run a score job")
    @service_errors
    async def rerun_score_job(request: Request, job_id: int) -> ScoreJobResponse:
        service = get_service(request)
        queue = get_queue(request)
        payload = await run_in_threadpool(service.rerun_score_job, job_id=job_id)
        queue.enqueue(payload["job_id"])
        return ScoreJobResponse(**payload)

    @router.post("/score/{job_id}/cancel", tags=["Scoring"], summary="Cancel a score job")
    @service_errors
    async def cancel_score_job(request: Request, job_id: int) -> dict:
        service = get_service(request)
        return await run_in_threadpool(service.cancel_score_job, job_id=job_id)

    @router.get("/score/{job_id}", response_model=ScoreJobResponse, tags=["Scoring"], summary="Get score job result")
    @service_errors
    async def get_score_job(request: Request, job_id: int) -> ScoreJobResponse:
        service = get_service(request)
        payload = await run_in_threadpool(service.get_score_job, job_id=job_id)
        return ScoreJobResponse(**payload)

    @router.put("/score/{job_id}/review", response_model=ScoreReviewResponse, tags=["Scoring"], summary="Submit or update a score review")
    @service_errors
    async def put_score_review(request: Request, job_id: int, body: ScoreReviewRequest) -> ScoreReviewResponse:
        service = get_service(request)
        payload = await run_in_threadpool(
            service.update_score_review, job_id=job_id, verdict=body.verdict, note=body.note,
        )
        return ScoreReviewResponse(**payload)

    @router.get("/score/{job_id}/export", tags=["Scoring"], summary="Export score job data")
    @service_errors
    async def export_score_job(request: Request, job_id: int) -> JSONResponse:
        service = get_service(request)
        payload = await run_in_threadpool(service.export_score_job, job_id=job_id)
        return JSONResponse(payload)

    @router.get("/score/export/csv", tags=["Scoring"], summary="Export completed scores as CSV")
    @service_errors
    async def export_scores_csv(
        request: Request,
        task_id: str | None = Query(default=None),
    ) -> Response:
        service = get_service(request)
        rows = await run_in_threadpool(service.database.list_completed_score_jobs, task_id=task_id)
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
                row["id"], row.get("task_id", ""), row["gold_video_id"], row["trainee_video_id"],
                row.get("operator_id_hash", ""), row.get("trainee_site_id", ""),
                score_data.get("score", ""), summary.get("decision", ""),
                metrics.get("miss_steps", ""), metrics.get("swap_steps", ""),
                metrics.get("deviation_steps", ""), metrics.get("over_time_ratio", ""),
                metrics.get("dtw_normalized_cost", ""), row.get("created_at", ""), row.get("finished_at", ""),
            ])
        return Response(
            content=buf.getvalue(), media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="sopilot_scores.csv"'},
        )

    @router.get("/score/{job_id}/timeline", tags=["Scoring"], summary="Get score alignment timeline")
    @service_errors
    async def get_score_timeline(request: Request, job_id: int) -> dict:
        service = get_service(request)
        return await run_in_threadpool(service.get_score_timeline, job_id=job_id)

    @router.get("/score/{job_id}/report", tags=["Scoring"], summary="Get score report HTML")
    @service_errors
    async def get_score_report(request: Request, job_id: int) -> HTMLResponse:
        service = get_service(request)
        html = await run_in_threadpool(service.get_score_report, job_id=job_id)
        return HTMLResponse(content=html)

    @router.get("/score/{job_id}/report/pdf", tags=["Scoring"], summary="Download score report as PDF")
    @service_errors
    async def get_score_report_pdf(request: Request, job_id: int) -> Response:
        service = get_service(request)
        pdf_bytes = await run_in_threadpool(service.get_score_report_pdf, job_id=job_id)
        return Response(
            content=pdf_bytes, media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="sopilot_job{job_id}_report.pdf"'},
        )

    # ── SSE stream ─────────────────────────────────────────────────

    @router.get("/score/{job_id}/stream", tags=["Scoring"], summary="Stream score job progress via SSE")
    async def score_job_stream(
        request: Request, job_id: int,
        poll_interval: float = Query(default=1.0, ge=0.5, le=10.0),
    ) -> StreamingResponse:
        service = get_service(request)

        async def event_generator() -> Any:
            last_status = None
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await run_in_threadpool(service.get_score_job, job_id=job_id)
                    current_status = payload.get("status")
                    if current_status != last_status:
                        last_status = current_status
                        event_data: dict[str, Any] = {"job_id": payload.get("job_id"), "status": current_status}
                        if current_status == "completed" and payload.get("result"):
                            result = payload["result"]
                            event_data["score"] = result.get("score")
                            event_data["decision"] = (result.get("summary") or {}).get("decision")
                        if current_status == "failed":
                            event_data["error"] = payload.get("error")
                        yield f"event: status\ndata: {json_mod.dumps(event_data)}\n\n"
                        if current_status in ("completed", "failed", "cancelled"):
                            yield f"event: done\ndata: {json_mod.dumps({'job_id': job_id, 'final_status': current_status})}\n\n"
                            break
                except Exception as exc:
                    yield f"event: error\ndata: {json_mod.dumps({'error': str(exc)})}\n\n"
                    break
                await asyncio.sleep(poll_interval)

        return StreamingResponse(
            event_generator(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    # ── Research / uncertainty endpoints ───────────────────────────

    @router.get("/score/{job_id}/uncertainty", tags=["Scoring"], summary="Get uncertainty breakdown for a score job")
    @service_errors
    async def get_score_uncertainty(request: Request, job_id: int) -> dict:
        service = get_service(request)
        return await run_in_threadpool(service.get_score_uncertainty, job_id)

    @router.post("/research/soft-dtw", tags=["Research"], summary="Compute Soft-DTW distance between two videos")
    @service_errors
    async def research_soft_dtw(request: Request, body: SoftDTWRequest) -> dict:
        service = get_service(request)
        try:
            result = await run_in_threadpool(
                service.compute_soft_dtw,
                gold_video_id=body.gold_video_id,
                trainee_video_id=body.trainee_video_id,
                gamma=body.gamma,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return result

    # ── Search ─────────────────────────────────────────────────────

    @router.get("/search", response_model=SearchResponse, tags=["Search"], summary="Search similar clips")
    @service_errors
    async def search(
        request: Request,
        video_id: int = Query(..., ge=1),
        clip_index: int = Query(..., ge=0),
        k: int = Query(default=5, ge=1, le=50),
        task_id: str | None = Query(default=None),
    ) -> SearchResponse:
        service = get_service(request)
        payload = await run_in_threadpool(
            service.search, query_video_id=video_id, query_clip_index=clip_index, k=k, task_id=task_id,
        )
        return SearchResponse(**payload)

    return router
