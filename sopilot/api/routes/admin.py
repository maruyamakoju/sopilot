"""Database administration and maintenance endpoints."""

import time as _time
from typing import Any

from fastapi import APIRouter, Query, Request
from starlette.concurrency import run_in_threadpool

from sopilot.api.error_handling import service_errors
from sopilot.api.routes._helpers import get_service


def build_admin_router() -> APIRouter:
    router = APIRouter(tags=["Admin"])

    @router.post("/admin/backup", summary="Create database backup")
    @service_errors
    async def create_backup(request: Request) -> dict:
        service = get_service(request)
        settings = request.app.state.settings
        ts = _time.strftime("%Y%m%d_%H%M%S")
        backup_path = str(settings.data_dir / f"backup_{ts}.db")
        await run_in_threadpool(service.database.backup, backup_path)
        return {"backup_path": backup_path, "timestamp": ts}

    @router.post("/admin/optimize", summary="Optimize database")
    @service_errors
    async def optimize_database(request: Request) -> dict:
        service = get_service(request)
        await run_in_threadpool(service.database.vacuum)
        return {"status": "optimized"}

    @router.get("/admin/db-stats", summary="Get database statistics")
    @service_errors
    async def db_stats(request: Request) -> dict:
        service = get_service(request)
        stats = await run_in_threadpool(service.database.get_stats)
        db_size_bytes = stats.pop("_db_size_bytes", None)
        db_size_human = stats.pop("_db_size_human", None)
        return {"tables": stats, "db_size_bytes": db_size_bytes, "db_size_human": db_size_human}

    @router.post(
        "/admin/rescore",
        summary="Re-apply current task thresholds to stored score decisions",
        description=(
            "Re-applies `make_decision()` with the current task profile "
            "(pass_score / retrain_score) to every completed score job. "
            "Preserves original scores and deviation severities — only the "
            "`summary.decision` block is updated. "
            "Use `dry_run=true` to preview changes without writing to the database."
        ),
    )
    @service_errors
    async def rescore_decisions(
        request: Request,
        task_id: str | None = Query(None, description="Restrict to a specific task ID"),
        dry_run: bool = Query(False, description="Preview only — do not write changes"),
    ) -> dict[str, Any]:
        service = get_service(request)
        result = await run_in_threadpool(
            service.rescore_decisions,
            task_id=task_id,
            dry_run=dry_run,
        )
        return result

    return router
