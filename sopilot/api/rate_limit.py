"""In-memory rate limiting middleware using a sliding window counter.

Configure via environment variables:
  ``SOPILOT_RATE_LIMIT_RPM``   – max requests per minute per client IP (default: 120, 0=disabled)
  ``SOPILOT_RATE_LIMIT_BURST`` – max burst size within 1-second window (default: 20)

Exempt paths (same as auth middleware) are never rate-limited.
"""

import logging
import time
from collections import defaultdict, deque
from threading import Lock

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from sopilot.api import PUBLIC_PATHS

logger = logging.getLogger(__name__)


class _SlidingWindow:
    """Per-client sliding window counter.

    Tracks request timestamps within the window and prunes expired entries
    on each call.  Thread-safe via a single lock per instance.
    """

    __slots__ = ("_window_sec", "_max_requests", "_clients", "_lock")

    def __init__(self, window_sec: float, max_requests: int) -> None:
        self._window_sec = window_sec
        self._max_requests = max_requests
        self._clients: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, client_id: str) -> tuple[bool, int]:
        """Return (allowed, remaining) for the given client."""
        now = time.monotonic()
        cutoff = now - self._window_sec

        with self._lock:
            timestamps = self._clients[client_id]
            # Prune expired entries — O(1) per removal with deque
            while timestamps and timestamps[0] < cutoff:
                timestamps.popleft()

            remaining = max(0, self._max_requests - len(timestamps))
            if len(timestamps) >= self._max_requests:
                return False, 0

            timestamps.append(now)
            return True, remaining - 1

    def cleanup(self) -> None:
        """Remove clients with no recent requests (call periodically)."""
        now = time.monotonic()
        cutoff = now - self._window_sec

        with self._lock:
            empty_keys = [
                k for k, v in self._clients.items()
                if not v or v[-1] < cutoff
            ]
            for k in empty_keys:
                del self._clients[k]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate-limits requests by client IP using a sliding window."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        requests_per_minute: int = 120,
        burst: int = 20,
    ) -> None:
        super().__init__(app)
        self._enabled = requests_per_minute > 0
        # Per-minute window
        self._minute_window = _SlidingWindow(60.0, requests_per_minute)
        # Per-second burst window
        self._burst_window = _SlidingWindow(1.0, burst)
        self._rpm = requests_per_minute
        self._cleanup_counter = 0

    def _client_ip(self, request: Request) -> str:
        """Extract client IP, respecting X-Forwarded-For if present."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self._enabled:
            return await call_next(request)

        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        client = self._client_ip(request)

        # Check burst limit first
        burst_ok, _ = self._burst_window.allow(client)
        if not burst_ok:
            logger.warning("rate_limit burst exceeded client=%s path=%s", client, request.url.path)
            return JSONResponse(
                {"detail": "Rate limit exceeded. Please slow down."},
                status_code=429,
                headers={"Retry-After": "1"},
            )

        # Check per-minute limit
        minute_ok, remaining = self._minute_window.allow(client)
        if not minute_ok:
            logger.warning("rate_limit rpm exceeded client=%s path=%s", client, request.url.path)
            return JSONResponse(
                {"detail": "Rate limit exceeded. Try again later."},
                status_code=429,
                headers={"Retry-After": "60"},
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self._rpm)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        # Periodic cleanup every ~500 requests
        self._cleanup_counter += 1
        if self._cleanup_counter >= 500:
            self._cleanup_counter = 0
            self._minute_window.cleanup()
            self._burst_window.cleanup()

        return response
