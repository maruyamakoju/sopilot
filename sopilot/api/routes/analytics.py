"""Analytics, SOP steps, configuration, and reporting endpoints."""

from fastapi import APIRouter, Query, Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from sopilot.api.error_handling import service_errors
from sopilot.api.routes._helpers import get_service
from sopilot.schemas import (
    SOPStepsUpsertRequest,
    TaskProfileResponse,
    TaskProfileUpdateRequest,
)


def build_analytics_router() -> APIRouter:
    router = APIRouter()

    # ── Configuration ──────────────────────────────────────────────

    @router.get("/task-profile", response_model=TaskProfileResponse, tags=["Configuration"], summary="Get task profile")
    @service_errors
    async def get_task_profile(request: Request) -> TaskProfileResponse:
        service = get_service(request)
        payload = await run_in_threadpool(service.get_task_profile)
        return TaskProfileResponse(**payload)

    @router.put("/task-profile", response_model=TaskProfileResponse, tags=["Configuration"], summary="Update task profile")
    @service_errors
    async def update_task_profile(request: Request, body: TaskProfileUpdateRequest) -> TaskProfileResponse:
        service = get_service(request)
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
        service = get_service(request)
        return await run_in_threadpool(service.get_dataset_summary)

    # ── Analytics ──────────────────────────────────────────────────

    @router.get("/analytics", tags=["Configuration"], summary="Get scoring analytics")
    @service_errors
    async def get_analytics(
        request: Request,
        task_id: str | None = Query(default=None),
        days: int | None = Query(default=None, ge=1, le=365),
    ) -> dict:
        service = get_service(request)
        return await run_in_threadpool(service.get_analytics, task_id=task_id, days=days)

    @router.get("/analytics/compliance", tags=["Analytics"], summary="Get SOP compliance overview")
    @service_errors
    async def get_compliance(
        request: Request,
        task_id: str | None = Query(default=None),
        days: int | None = Query(default=None, ge=1, le=365),
    ) -> dict:
        service = get_service(request)
        return await run_in_threadpool(service.get_compliance_overview, task_id=task_id, days=days)

    @router.get("/analytics/steps", tags=["Analytics"], summary="Get per-step difficulty analysis")
    @service_errors
    async def get_step_performance(
        request: Request,
        task_id: str | None = Query(default=None),
        days: int | None = Query(default=None, ge=1, le=365),
    ) -> dict:
        service = get_service(request)
        return await run_in_threadpool(service.get_step_performance, task_id=task_id, days=days)

    @router.get("/analytics/operators/{operator_id}/trend", tags=["Analytics"], summary="Get operator score trend")
    @service_errors
    async def get_operator_trend(
        request: Request, operator_id: str,
        task_id: str | None = Query(default=None),
        limit: int = Query(default=20, ge=1, le=100),
    ) -> dict:
        service = get_service(request)
        return await run_in_threadpool(service.get_operator_trend, operator_id=operator_id, task_id=task_id)

    @router.get("/analytics/recommendations/{operator_id}", tags=["Analytics"], summary="Get training recommendations for operator")
    @service_errors
    async def get_recommendations(
        request: Request, operator_id: str,
        task_id: str | None = Query(default=None),
    ) -> dict:
        service = get_service(request)
        return await run_in_threadpool(service.get_recommendations, operator_id=operator_id, task_id=task_id)

    @router.get("/analytics/operators/{operator_id}/projection", tags=["Analytics"], summary="Predict operator certification pathway")
    @service_errors
    async def get_operator_projection(
        request: Request, operator_id: str,
        task_id: str | None = Query(default=None),
    ) -> dict:
        service = get_service(request)
        return await run_in_threadpool(
            service.get_operator_learning_curve, operator_id=operator_id, task_id=task_id,
        )

    # ── SOP Steps ──────────────────────────────────────────────────

    @router.get("/tasks/steps", tags=["SOP Steps"], summary="Get SOP step definitions for the primary task")
    @service_errors
    async def get_sop_steps(
        request: Request,
        task_id: str | None = Query(default=None),
    ) -> dict:
        service = get_service(request)
        tid = task_id or service.settings.primary_task_id
        return await run_in_threadpool(service.get_sop_steps, tid)

    @router.put("/tasks/steps", tags=["SOP Steps"], summary="Upsert SOP step definitions")
    @service_errors
    async def upsert_sop_steps(
        request: Request,
        body: SOPStepsUpsertRequest,
        task_id: str | None = Query(default=None),
    ) -> dict:
        service = get_service(request)
        tid = task_id or service.settings.primary_task_id
        return await run_in_threadpool(service.upsert_sop_steps, tid, body.steps)

    @router.delete("/tasks/steps", tags=["SOP Steps"], summary="Delete all SOP step definitions")
    @service_errors
    async def delete_sop_steps(
        request: Request,
        task_id: str | None = Query(default=None),
    ) -> dict:
        service = get_service(request)
        tid = task_id or service.settings.primary_task_id
        return await run_in_threadpool(service.delete_sop_steps, tid)

    # ── Analytics PDF report ───────────────────────────────────────

    @router.get("/analytics/report/pdf", tags=["Analytics"], summary="Generate executive analytics PDF report")
    @service_errors
    async def get_analytics_report_pdf(
        request: Request,
        task_id: str | None = Query(default=None),
        days: int | None = Query(default=None, ge=1, le=365),
    ) -> Response:
        from sopilot.core.analytics_report_pdf import generate_analytics_pdf

        service = get_service(request)
        analytics = await run_in_threadpool(service.get_analytics, task_id=task_id, days=days)
        pdf_bytes = await run_in_threadpool(generate_analytics_pdf, analytics)
        return Response(
            content=pdf_bytes, media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="sopilot_analytics_report.pdf"'},
        )

    return router
