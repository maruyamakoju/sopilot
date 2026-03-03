"""Sliding-window rate limiter middleware for FastAPI.

Per-IP rate limiting using a sliding deque window.
Returns 429 Too Many Requests with Retry-After header on breach.

IP resolution order:
  1. X-Forwarded-For header (first IP, for reverse proxy setups)
  2. request.client.host
  3. "unknown" fallback
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

DEFAULT_EXCLUDED_PATHS: frozenset[str] = frozenset({
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
})


class SlidingWindowRateLimiter:
    """Thread-safe per-key sliding window rate limiter.

    Uses a deque of timestamps per key. Old timestamps (outside window) are
    pruned on each check. Keys are evicted by LRU when max_tracked_keys is exceeded.
    """

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: float = 60.0,
        max_tracked_keys: int = 10_000,
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_tracked_keys = max_tracked_keys
        self._lock = threading.RLock()
        self._windows: dict[str, deque[float]] = {}
        self._total_allowed = 0
        self._total_denied = 0

    def is_allowed(self, key: str, _now: float | None = None) -> tuple[bool, float]:
        """Check if key is within rate limit.

        Returns (allowed: bool, retry_after_seconds: float).
        retry_after_seconds is 0.0 when allowed.
        """
        now = _now if _now is not None else time.time()
        cutoff = now - self.window_seconds

        with self._lock:
            if key not in self._windows:
                # Evict oldest key if at capacity
                if len(self._windows) >= self.max_tracked_keys:
                    # Simple eviction: remove first key (FIFO)
                    oldest_key = next(iter(self._windows))
                    del self._windows[oldest_key]
                self._windows[key] = deque()

            window = self._windows[key]
            # Prune expired timestamps
            while window and window[0] <= cutoff:
                window.popleft()

            if len(window) >= self.max_requests:
                # Denied: retry after oldest request expires (window may be empty
                # when max_requests=0, so fall back to full window_seconds)
                retry_after = (
                    window[0] + self.window_seconds - now
                    if window
                    else self.window_seconds
                )
                self._total_denied += 1
                return False, max(0.0, retry_after)

            window.append(now)
            self._total_allowed += 1
            return True, 0.0

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "total_keys": len(self._windows),
                "total_allowed": self._total_allowed,
                "total_denied": self._total_denied,
                "max_requests": self.max_requests,
                "window_seconds": self.window_seconds,
            }

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is not None:
                self._windows.pop(key, None)
            else:
                self._windows.clear()
                self._total_allowed = 0
                self._total_denied = 0


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-client-IP sliding window rate limiter."""

    def __init__(
        self,
        app,
        max_requests: int = 100,
        window_seconds: float = 60.0,
        excluded_paths: frozenset[str] | set[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._limiter = SlidingWindowRateLimiter(
            max_requests=max_requests,
            window_seconds=window_seconds,
        )
        self._excluded = frozenset(excluded_paths or DEFAULT_EXCLUDED_PATHS)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self._excluded:
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        allowed, retry_after = self._limiter.is_allowed(client_ip)

        if not allowed:
            logger.warning("RateLimitMiddleware: rate limited %s", client_ip)
            return JSONResponse(
                {"detail": f"Rate limit exceeded. Retry after {retry_after:.1f}s."},
                status_code=429,
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

        return await call_next(request)

    def get_stats(self) -> dict:
        return self._limiter.get_stats()

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        client = getattr(request.client, "host", None)
        return client or "unknown"


def build_rate_limiter(
    max_requests: int = 100,
    window_seconds: float = 60.0,
) -> type:
    """Return RateLimitMiddleware class configured with given limits.

    Usage:
        app.add_middleware(build_rate_limiter(max_requests=60, window_seconds=60))
    """
    class _ConfiguredRateLimitMiddleware(RateLimitMiddleware):
        def __init__(self, app_inner):
            super().__init__(app_inner, max_requests=max_requests, window_seconds=window_seconds)

    return _ConfiguredRateLimitMiddleware
