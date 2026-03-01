"""Domain-specific API routers for SOPilot.

Each sub-module defines a focused set of endpoints. The ``build_router()``
factory in this package composes them into a single APIRouter that can be
mounted on the FastAPI application.
"""

from __future__ import annotations

from fastapi import APIRouter

from sopilot.api.routes.admin import build_admin_router
from sopilot.api.routes.analytics import build_analytics_router
from sopilot.api.routes.health import build_health_router
from sopilot.api.routes.scoring import build_scoring_router
from sopilot.api.routes.videos import build_video_router


def build_router() -> APIRouter:
    """Compose all domain routers into a single APIRouter."""
    router = APIRouter()
    router.include_router(build_health_router())
    router.include_router(build_video_router())
    router.include_router(build_scoring_router())
    router.include_router(build_analytics_router())
    router.include_router(build_admin_router())
    return router
