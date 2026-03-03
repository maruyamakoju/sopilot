"""Optional API key authentication middleware for FastAPI.

Enabled only when an API key is configured (e.g., via API_KEY env var).
When no key is configured, the middleware passes all requests through
(development/local mode).

Authentication methods (checked in order):
  1. X-API-Key request header
  2. ?api_key= query parameter

Responses:
  401 Unauthorized — key missing
  403 Forbidden    — key present but invalid
  Pass-through     — excluded paths, or key valid, or no key configured
"""
from __future__ import annotations

import logging
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# Paths that never require authentication
DEFAULT_EXCLUDED_PATHS: frozenset[str] = frozenset({
    "/",
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
})


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validates X-API-Key header or ?api_key= query param.

    If api_key is empty/None, all requests pass through (disabled mode).
    """

    def __init__(
        self,
        app,
        api_key: str,
        excluded_paths: frozenset[str] | set[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._api_key = api_key or ""
        self._excluded = frozenset(excluded_paths or DEFAULT_EXCLUDED_PATHS)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Disabled mode
        if not self._api_key:
            return await call_next(request)

        # Excluded paths bypass auth
        if request.url.path in self._excluded:
            return await call_next(request)

        # Extract key from header or query param
        provided = (
            request.headers.get("X-API-Key")
            or request.query_params.get("api_key")
        )

        if provided is None:
            logger.debug("APIKeyMiddleware: missing key for %s", request.url.path)
            return JSONResponse(
                {"detail": "API key required. Provide X-API-Key header or ?api_key= query parameter."},
                status_code=401,
            )

        if provided != self._api_key:
            logger.warning("APIKeyMiddleware: invalid key attempt for %s", request.url.path)
            return JSONResponse(
                {"detail": "Invalid API key."},
                status_code=403,
            )

        return await call_next(request)


def build_api_key_middleware(api_key: str | None) -> type | None:
    """Return APIKeyMiddleware class configured for given api_key, or None if no key.

    Usage with FastAPI:
        mw = build_api_key_middleware(os.getenv("API_KEY"))
        if mw:
            app.add_middleware(mw)

    Returns a partially-applied class (using a closure) so that FastAPI's
    add_middleware can call it with just `app`.
    """
    if not api_key:
        return None

    # Create a subclass with the key baked in
    class _ConfiguredAPIKeyMiddleware(APIKeyMiddleware):
        def __init__(self, app_inner):
            super().__init__(app_inner, api_key=api_key)

    return _ConfiguredAPIKeyMiddleware
