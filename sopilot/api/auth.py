"""API key authentication middleware.

When ``SOPILOT_API_KEY`` is set, every request to a protected endpoint must
include a matching ``X-API-Key`` header.  If the env var is empty or unset,
authentication is disabled (dev / PoC mode).

Exempt paths (always accessible without a key):
  ``/``, ``/health``, ``/metrics``, ``/status``, ``/docs``, ``/redoc``, ``/openapi.json``
"""

import logging

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from sopilot.api import PUBLIC_PATHS

logger = logging.getLogger(__name__)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests that lack a valid ``X-API-Key`` header."""

    def __init__(self, app: ASGIApp, *, api_key: str | None) -> None:
        super().__init__(app)
        self._api_key: str | None = api_key if api_key else None

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if self._api_key is None:
            return await call_next(request)

        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        provided = request.headers.get("X-API-Key", "")
        if provided != self._api_key:
            logger.warning("auth rejected path=%s remote=%s", request.url.path, request.client.host if request.client else "unknown")
            return JSONResponse(
                {"detail": "Invalid or missing API key"},
                status_code=401,
            )

        return await call_next(request)
