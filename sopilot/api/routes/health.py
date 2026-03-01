"""Health, readiness, status, and metrics endpoints."""

import os
import shutil
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from sopilot import __version__
from sopilot.api.routes._helpers import get_queue, get_service


def build_health_router() -> APIRouter:
    router = APIRouter(tags=["Health"])

    @router.get("/readiness", summary="Readiness probe for container orchestration")
    def readiness(request: Request) -> JSONResponse:
        ready = True
        details: dict[str, str] = {}

        try:
            service = get_service(request)
            with service.database.connect() as conn:
                conn.execute("SELECT 1")
            details["database"] = "ok"
        except Exception:
            details["database"] = "unavailable"
            ready = False

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

        status_code = 200 if ready else 503
        return JSONResponse({"ready": ready, "details": details}, status_code=status_code)

    @router.get("/health", summary="Run health checks")
    def health(request: Request) -> dict:
        checks: dict[str, dict] = {}
        overall = "healthy"

        # DB check
        try:
            service = get_service(request)
            t0 = time.perf_counter()
            with service.database.connect() as conn:
                conn.execute("SELECT 1")
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            checks["database"] = {"status": "up", "latency_ms": latency_ms}
        except Exception:
            checks["database"] = {"status": "down"}
            overall = "unhealthy"

        # Disk check
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

        # Embedder check
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
                    checks["embedder"] = {"status": "down", "name": embed_name, "failed_over": failed_over}
                    overall = "unhealthy"
                elif failed_over:
                    checks["embedder"] = {"status": "degraded", "name": embed_name, "failed_over": failed_over}
                    if overall == "healthy":
                        overall = "degraded"
                else:
                    checks["embedder"] = {"status": "up", "name": embed_name, "failed_over": False}
            else:
                checks["embedder"] = {"status": "up", "name": embed_name, "failed_over": False}
        except Exception:
            checks["embedder"] = {"status": "down"}
            overall = "unhealthy"

        return {"status": overall, "checks": checks, "version": __version__}

    @router.get("/status", summary="Get service status")
    async def get_status(request: Request) -> dict:
        settings = request.app.state.settings
        queue = get_queue(request)
        embedder = getattr(request.app.state, "embedder", None)
        return {
            "version": __version__,
            "primary_task_id": settings.primary_task_id,
            "embedder": getattr(embedder, "name", "unknown"),
            "embedder_stats": getattr(embedder, "get_stats", lambda: None)(),
            "queue_depth": queue.depth(),
            "data_dir": str(settings.data_dir),
        }

    @router.get("/metrics", summary="Prometheus metrics endpoint")
    async def get_metrics(request: Request) -> PlainTextResponse:
        settings = request.app.state.settings
        queue = get_queue(request)
        embedder = getattr(request.app.state, "embedder", None)
        embed_stats: dict = getattr(embedder, "get_stats", lambda: {})() or {}
        queue_depth: int = queue.depth()

        lines: list[str] = []

        def _metric(name: str, value: object, help_text: str, kind: str = "gauge") -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {kind}")
            lines.append(f"{name} {value}")

        _metric("sopilot_queue_depth", queue_depth, "Number of score jobs currently waiting in the queue")
        _metric("sopilot_embed_requests_total", embed_stats.get("total_requests", 0), "Total embed() calls received by the embedder", "counter")
        _metric("sopilot_embed_primary_successes_total", embed_stats.get("primary_successes", 0), "Embed calls served by the primary (V-JEPA2) embedder", "counter")
        _metric("sopilot_embed_fallback_uses_total", embed_stats.get("fallback_uses", 0), "Embed calls routed to the fallback (color-motion) embedder", "counter")
        _metric("sopilot_embed_failure_capsules_total", embed_stats.get("failure_capsules_written", 0), "Number of failure capsule records written to disk", "counter")
        _metric("sopilot_embed_failed_over", 1 if embed_stats.get("failed_over", False) else 0, "1 if the embedder is currently running on the fallback path")
        _metric("sopilot_embed_permanently_failed", 1 if embed_stats.get("permanently_failed", False) else 0, "1 if the primary embedder has permanently failed")
        _metric("sopilot_embed_retry_interval_seconds", embed_stats.get("current_retry_interval_sec", 0), "Current exponential-backoff retry interval in seconds")

        try:
            import torch
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                _metric("sopilot_vram_free_bytes", free_bytes, "Free GPU VRAM in bytes")
                _metric("sopilot_vram_total_bytes", total_bytes, "Total GPU VRAM in bytes")
        except Exception:
            pass

        embedder_name = str(getattr(embedder, "name", "unknown")).replace('"', "")
        lines.append("# HELP sopilot_info Static service metadata")
        lines.append("# TYPE sopilot_info gauge")
        lines.append(f'sopilot_info{{version="{__version__}",task_id="{settings.primary_task_id}",embedder="{embedder_name}"}} 1')
        lines.append("")

        return PlainTextResponse("\n".join(lines), media_type="text/plain; version=0.0.4; charset=utf-8")

    return router
