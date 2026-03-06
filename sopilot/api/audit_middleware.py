"""Audit logging middleware.

Captures method, path, status_code, duration_ms, and client identity for
every mutating request (POST/PUT/PATCH/DELETE).  Submits asynchronously
to a background queue — never blocks the request path.
"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

_AUDIT_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


class AuditMiddleware(BaseHTTPMiddleware):
    """Captures mutating requests and queues them for async audit logging."""

    def __init__(self, app: object, audit_worker: object) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._worker = audit_worker

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method not in _AUDIT_METHODS:
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000.0

        # Resolve identity from request.state (set by ApiKeyMiddleware or JWT decode)
        client_id: str | None = getattr(request.state, "auth_client_id", None)
        role: str | None = getattr(request.state, "auth_role", None)
        request_id: str | None = getattr(request.state, "request_id", None)

        # Best-effort IP extraction
        forwarded_for = request.headers.get("X-Forwarded-For")
        ip_address = (
            forwarded_for.split(",")[0].strip() if forwarded_for
            else (request.client.host if request.client else None)
        )

        self._worker.enqueue(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
            client_id=client_id,
            role=role,
            ip_address=ip_address,
            request_id=request_id,
        )
        return response
