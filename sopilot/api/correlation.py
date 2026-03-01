"""Request correlation-ID and security headers middleware.

Assigns a unique ``X-Request-ID`` header to every request so that all log
lines emitted during that request can be traced together.  If the caller
already provides the header (e.g. an API gateway), the existing value is
reused instead of generating a new one.

The ID is also injected into the structured-logging context via
:func:`sopilot.logging_config.log_context`.

Security headers (X-Content-Type-Options, X-Frame-Options, etc.) are
added to every response for defense-in-depth.
"""

import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from sopilot.logging_config import log_context

_HEADER = "X-Request-ID"

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(_HEADER) or uuid.uuid4().hex
        with log_context(request_id=request_id):
            response = await call_next(request)
        response.headers[_HEADER] = request_id
        for key, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(key, value)
        return response
