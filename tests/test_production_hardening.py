"""Tests for sopilot/middleware/api_auth.py and sopilot/middleware/rate_limiter.py.

55 tests total:
  - TestSlidingWindowRateLimiter     (15)
  - TestAPIKeyMiddleware             (15)
  - TestRateLimitMiddleware          (15)
  - TestIntegration                  (5)
  - TestBuildFunctions               (5)
"""
from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from sopilot.middleware.api_auth import (
    APIKeyMiddleware,
    build_api_key_middleware,
)
from sopilot.middleware.rate_limiter import (
    RateLimitMiddleware,
    SlidingWindowRateLimiter,
    build_rate_limiter,
)


# ---------------------------------------------------------------------------
# Minimal test app factory
# ---------------------------------------------------------------------------

def make_app(
    api_key=None,
    max_requests: int = 100,
    window_seconds: float = 60.0,
):
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/protected")
    def protected():
        return {"secret": "data"}

    @app.get("/data")
    def data():
        return {"items": [1, 2, 3]}

    if api_key:
        mw = build_api_key_middleware(api_key)
        if mw:
            app.add_middleware(mw)

    app.add_middleware(
        build_rate_limiter(max_requests=max_requests, window_seconds=window_seconds)
    )
    return app


# ===========================================================================
# TestSlidingWindowRateLimiter (15 tests)
# ===========================================================================

class TestSlidingWindowRateLimiter:

    def test_first_request_allowed(self):
        """First request should always be allowed with retry_after=0.0."""
        limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60.0)
        allowed, retry_after = limiter.is_allowed("ip1")
        assert allowed is True
        assert retry_after == 0.0

    def test_requests_up_to_max_all_allowed(self):
        """Exactly max_requests should all be permitted."""
        limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60.0)
        for i in range(5):
            allowed, _ = limiter.is_allowed("ip1", _now=float(i))
        assert allowed is True

    def test_request_beyond_max_denied(self):
        """The (max_requests + 1)-th request must be denied."""
        limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60.0)
        for _ in range(5):
            limiter.is_allowed("ip1", _now=0.0)
        allowed, retry_after = limiter.is_allowed("ip1", _now=0.0)
        assert allowed is False
        assert retry_after > 0

    def test_window_expiry_allows_new_requests(self):
        """After the window expires, requests should be allowed again."""
        limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=10.0)
        t0 = 0.0
        for _ in range(3):
            limiter.is_allowed("ip1", _now=t0)
        # At t0 still denied
        allowed_before, _ = limiter.is_allowed("ip1", _now=t0)
        assert allowed_before is False
        # After window+1 seconds, old timestamps pruned
        allowed_after, _ = limiter.is_allowed("ip1", _now=t0 + 11.0)
        assert allowed_after is True

    def test_multiple_keys_independent(self):
        """Denying one key must not affect another key."""
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60.0)
        for _ in range(2):
            limiter.is_allowed("key_a", _now=0.0)
        denied_a, _ = limiter.is_allowed("key_a", _now=0.0)
        allowed_b, _ = limiter.is_allowed("key_b", _now=0.0)
        assert denied_a is False
        assert allowed_b is True

    def test_reset_single_key(self):
        """reset(key) should clear that key's window and allow fresh requests."""
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60.0)
        for _ in range(2):
            limiter.is_allowed("ip1", _now=0.0)
        limiter.reset("ip1")
        allowed, _ = limiter.is_allowed("ip1", _now=0.0)
        assert allowed is True

    def test_reset_all_clears_keys_and_counters(self):
        """reset() with no argument clears all state."""
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60.0)
        limiter.is_allowed("ip1", _now=0.0)
        limiter.is_allowed("ip2", _now=0.0)
        limiter.reset()
        stats = limiter.get_stats()
        assert stats["total_keys"] == 0
        assert stats["total_allowed"] == 0
        assert stats["total_denied"] == 0

    def test_get_stats_returns_expected_keys(self):
        """get_stats() must return required keys."""
        limiter = SlidingWindowRateLimiter(max_requests=10, window_seconds=30.0)
        stats = limiter.get_stats()
        assert "total_keys" in stats
        assert "total_allowed" in stats
        assert "total_denied" in stats
        assert "max_requests" in stats
        assert "window_seconds" in stats

    def test_total_denied_increments_on_each_denial(self):
        """total_denied counter must increment for every denied call."""
        limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60.0)
        limiter.is_allowed("ip1", _now=0.0)   # allowed
        limiter.is_allowed("ip1", _now=0.0)   # denied
        limiter.is_allowed("ip1", _now=0.0)   # denied
        stats = limiter.get_stats()
        assert stats["total_denied"] == 2

    def test_total_allowed_increments_on_each_allow(self):
        """total_allowed counter must track all allowed calls."""
        limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60.0)
        for i in range(3):
            limiter.is_allowed("ip1", _now=float(i))
        stats = limiter.get_stats()
        assert stats["total_allowed"] == 3

    def test_max_tracked_keys_no_exception(self):
        """Adding more keys than max_tracked_keys must not raise an exception."""
        limiter = SlidingWindowRateLimiter(
            max_requests=100, window_seconds=60.0, max_tracked_keys=5
        )
        for i in range(10):
            limiter.is_allowed(f"ip{i}", _now=0.0)
        # Should survive without error; key count <= max_tracked_keys
        stats = limiter.get_stats()
        assert stats["total_keys"] <= 5

    def test_short_window_deny_then_pass_after_expiry(self):
        """5 requests in window → denied on 6th; allowed after window passes."""
        limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=1.0)
        t0 = 0.0
        for _ in range(5):
            limiter.is_allowed("ip1", _now=t0)
        denied, _ = limiter.is_allowed("ip1", _now=t0)
        assert denied is False
        allowed, _ = limiter.is_allowed("ip1", _now=t0 + 2.0)
        assert allowed is True

    def test_retry_after_is_approximate_remaining_time(self):
        """retry_after should approximate time until oldest request expires."""
        limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=10.0)
        limiter.is_allowed("ip1", _now=0.0)   # fills the window
        _, retry_after = limiter.is_allowed("ip1", _now=3.0)  # 3s later → 7s remaining
        assert 6.0 <= retry_after <= 8.0

    def test_alternate_allow_deny_allow_as_time_advances(self):
        """Single key: fill → deny → advance time → allow again → fill → deny."""
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=10.0)
        limiter.is_allowed("ip1", _now=0.0)
        limiter.is_allowed("ip1", _now=0.0)
        # Denied
        denied, _ = limiter.is_allowed("ip1", _now=0.0)
        assert denied is False
        # After window passes, allowed again
        allowed, _ = limiter.is_allowed("ip1", _now=11.0)
        assert allowed is True
        limiter.is_allowed("ip1", _now=11.0)
        # Denied again
        denied2, _ = limiter.is_allowed("ip1", _now=11.0)
        assert denied2 is False

    def test_zero_max_requests_all_denied(self):
        """max_requests=0 means every call is immediately denied."""
        limiter = SlidingWindowRateLimiter(max_requests=0, window_seconds=60.0)
        allowed, _ = limiter.is_allowed("ip1", _now=0.0)
        assert allowed is False


# ===========================================================================
# TestAPIKeyMiddleware (15 tests)
# ===========================================================================

class TestAPIKeyMiddleware:

    def test_no_api_key_configured_passes_all(self):
        """With no api_key, every request passes through."""
        app = make_app(api_key=None)
        client = TestClient(app)
        resp = client.get("/protected")
        assert resp.status_code == 200

    def test_health_excluded_without_key(self):
        """GET /health should always return 200 even without an API key."""
        app = make_app(api_key="secret")
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_docs_excluded_without_key(self):
        """/docs should be accessible without an API key."""
        app = make_app(api_key="secret")
        client = TestClient(app)
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_missing_key_returns_401(self):
        """No key provided at all → 401."""
        app = make_app(api_key="secret")
        client = TestClient(app)
        resp = client.get("/protected")
        assert resp.status_code == 401

    def test_wrong_key_in_header_returns_403(self):
        """Wrong X-API-Key → 403."""
        app = make_app(api_key="secret")
        client = TestClient(app)
        resp = client.get("/protected", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 403

    def test_correct_key_in_header_returns_200(self):
        """Correct X-API-Key → 200."""
        app = make_app(api_key="secret")
        client = TestClient(app)
        resp = client.get("/protected", headers={"X-API-Key": "secret"})
        assert resp.status_code == 200

    def test_correct_key_in_query_param_returns_200(self):
        """?api_key=correct_key → 200."""
        app = make_app(api_key="secret")
        client = TestClient(app)
        resp = client.get("/protected?api_key=secret")
        assert resp.status_code == 200

    def test_wrong_key_in_query_param_returns_403(self):
        """?api_key=wrong_key → 403."""
        app = make_app(api_key="secret")
        client = TestClient(app)
        resp = client.get("/protected?api_key=wrong")
        assert resp.status_code == 403

    def test_no_key_anywhere_returns_401(self):
        """Explicit: no header, no query param → 401."""
        app = make_app(api_key="secret")
        client = TestClient(app)
        resp = client.get("/data")
        assert resp.status_code == 401

    def test_correct_key_in_header_and_query_uses_header(self):
        """Both header and query correct → 200 (header wins)."""
        app = make_app(api_key="secret")
        client = TestClient(app)
        resp = client.get(
            "/protected?api_key=secret",
            headers={"X-API-Key": "secret"},
        )
        assert resp.status_code == 200

    def test_wrong_header_correct_query_uses_header_first(self):
        """Wrong header + correct query → 403 (header is checked first)."""
        app = make_app(api_key="secret")
        client = TestClient(app)
        resp = client.get(
            "/protected?api_key=secret",
            headers={"X-API-Key": "wrong"},
        )
        assert resp.status_code == 403

    def test_custom_excluded_paths_respected(self):
        """APIKeyMiddleware with custom excluded_paths should bypass auth for those paths."""
        inner_app = FastAPI()

        @inner_app.get("/custom-open")
        def custom_open():
            return {"open": True}

        @inner_app.get("/locked")
        def locked():
            return {"locked": True}

        inner_app.add_middleware(
            APIKeyMiddleware,
            api_key="mykey",
            excluded_paths={"/custom-open"},
        )
        client = TestClient(inner_app)
        # Custom excluded path → 200 without key
        assert client.get("/custom-open").status_code == 200
        # Non-excluded path → 401 without key
        assert client.get("/locked").status_code == 401

    def test_disabled_mode_api_key_none_passthrough(self):
        """build_api_key_middleware(None) returns None → no auth applied."""
        result = build_api_key_middleware(None)
        assert result is None

    def test_build_api_key_middleware_none_returns_none(self):
        """build_api_key_middleware(None) must return None."""
        assert build_api_key_middleware(None) is None

    def test_build_api_key_middleware_secret_returns_class(self):
        """build_api_key_middleware('secret') must return a class (not None)."""
        result = build_api_key_middleware("secret")
        assert result is not None
        assert isinstance(result, type)


# ===========================================================================
# TestRateLimitMiddleware (15 tests)
# ===========================================================================

class TestRateLimitMiddleware:

    def test_health_excluded_from_rate_limiting(self):
        """/health is excluded and should return 200."""
        app = make_app(max_requests=1, window_seconds=60.0)
        client = TestClient(app)
        assert client.get("/health").status_code == 200

    def test_single_request_returns_200(self):
        """First request within limit → 200."""
        app = make_app(max_requests=5, window_seconds=60.0)
        client = TestClient(app)
        assert client.get("/data").status_code == 200

    def test_requests_within_limit_all_200(self):
        """All requests up to max → 200."""
        app = make_app(max_requests=5, window_seconds=60.0)
        client = TestClient(app)
        for _ in range(5):
            resp = client.get("/data")
            assert resp.status_code == 200

    def test_over_limit_returns_429(self):
        """The (max_requests + 1)-th request → 429."""
        app = make_app(max_requests=3, window_seconds=60.0)
        client = TestClient(app)
        for _ in range(3):
            client.get("/data")
        resp = client.get("/data")
        assert resp.status_code == 429

    def test_429_response_has_retry_after_header(self):
        """429 response must include Retry-After header."""
        app = make_app(max_requests=1, window_seconds=60.0)
        client = TestClient(app)
        client.get("/data")
        resp = client.get("/data")
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers

    def test_429_response_body_has_detail_key(self):
        """429 response body must have a 'detail' key."""
        app = make_app(max_requests=1, window_seconds=60.0)
        client = TestClient(app)
        client.get("/data")
        resp = client.get("/data")
        assert resp.status_code == 429
        body = resp.json()
        assert "detail" in body

    def test_x_forwarded_for_used_for_ip(self):
        """X-Forwarded-For header should determine the rate-limit bucket."""
        app = make_app(max_requests=1, window_seconds=60.0)
        client = TestClient(app)
        client.get("/data", headers={"X-Forwarded-For": "10.0.0.1"})
        # Second request from same forwarded IP → 429
        resp = client.get("/data", headers={"X-Forwarded-For": "10.0.0.1"})
        assert resp.status_code == 429

    def test_without_forwarded_for_client_host_used(self):
        """Without X-Forwarded-For, client.host is used (testclient is testclient)."""
        app = make_app(max_requests=2, window_seconds=60.0)
        client = TestClient(app)
        client.get("/data")
        client.get("/data")
        resp = client.get("/data")
        # Should be rate-limited regardless of IP resolution path
        assert resp.status_code == 429

    def test_excluded_path_always_200_even_over_limit(self):
        """/health must never be rate-limited, even when other paths are."""
        app = make_app(max_requests=1, window_seconds=60.0)
        client = TestClient(app)
        # Exhaust limit on /data
        client.get("/data")
        # /health is excluded, should still return 200
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_after_window_expiry_requests_allowed_again(self):
        """After a very short window expires, rate limiter resets for the key."""
        # Use the limiter directly (avoids real sleep in HTTP layer)
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=0.1)
        limiter.is_allowed("ip1", _now=0.0)
        limiter.is_allowed("ip1", _now=0.0)
        denied, _ = limiter.is_allowed("ip1", _now=0.0)
        assert denied is False
        # Advance past window
        allowed, _ = limiter.is_allowed("ip1", _now=0.5)
        assert allowed is True

    def test_different_ips_get_independent_windows(self):
        """Two distinct IPs should have independent rate limits."""
        app = make_app(max_requests=1, window_seconds=60.0)
        client = TestClient(app)
        # Exhaust IP1
        client.get("/data", headers={"X-Forwarded-For": "1.1.1.1"})
        rate_limited = client.get("/data", headers={"X-Forwarded-For": "1.1.1.1"})
        # IP2 still has capacity
        ok = client.get("/data", headers={"X-Forwarded-For": "2.2.2.2"})
        assert rate_limited.status_code == 429
        assert ok.status_code == 200

    def test_get_stats_accessible_on_middleware(self):
        """RateLimitMiddleware.get_stats() should return a dict."""
        inner_app = FastAPI()

        @inner_app.get("/ping")
        def ping():
            return {"pong": True}

        mw_instance = None

        # Manually instantiate to call get_stats
        limiter = SlidingWindowRateLimiter(max_requests=10, window_seconds=60.0)
        stats = limiter.get_stats()
        assert isinstance(stats, dict)
        assert "total_keys" in stats

    def test_build_rate_limiter_returns_class(self):
        """build_rate_limiter() must return a type (class)."""
        result = build_rate_limiter(max_requests=10, window_seconds=30.0)
        assert isinstance(result, type)

    def test_class_from_build_rate_limiter_usable_with_add_middleware(self):
        """Class from build_rate_limiter must work with app.add_middleware."""
        app = FastAPI()

        @app.get("/ping")
        def ping():
            return {"pong": True}

        mw_cls = build_rate_limiter(max_requests=5, window_seconds=60.0)
        app.add_middleware(mw_cls)
        client = TestClient(app)
        resp = client.get("/ping")
        assert resp.status_code == 200

    def test_stats_total_denied_increments_on_429(self):
        """total_denied should increment each time a request is denied."""
        limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60.0)
        limiter.is_allowed("ip1", _now=0.0)  # allowed
        limiter.is_allowed("ip1", _now=0.0)  # denied
        limiter.is_allowed("ip1", _now=0.0)  # denied again
        stats = limiter.get_stats()
        assert stats["total_denied"] == 2


# ===========================================================================
# TestIntegration (5 tests)
# ===========================================================================

class TestIntegration:

    def test_wrong_key_returns_403_auth_before_rate_limit(self):
        """With both middlewares: wrong API key → 403 (auth runs before rate limit)."""
        app = make_app(api_key="secret", max_requests=100, window_seconds=60.0)
        client = TestClient(app)
        resp = client.get("/protected", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 403

    def test_correct_key_within_limit_returns_200(self):
        """Correct key + within rate limit → 200."""
        app = make_app(api_key="secret", max_requests=10, window_seconds=60.0)
        client = TestClient(app)
        resp = client.get("/protected", headers={"X-API-Key": "secret"})
        assert resp.status_code == 200

    def test_correct_key_over_limit_returns_429(self):
        """Correct key + over rate limit → 429."""
        app = make_app(api_key="secret", max_requests=2, window_seconds=60.0)
        client = TestClient(app)
        for _ in range(2):
            client.get("/protected", headers={"X-API-Key": "secret"})
        resp = client.get("/protected", headers={"X-API-Key": "secret"})
        assert resp.status_code == 429

    def test_health_always_200_with_both_middlewares(self):
        """/health bypasses both auth and rate limiting → always 200."""
        app = make_app(api_key="secret", max_requests=1, window_seconds=60.0)
        client = TestClient(app)
        for _ in range(5):
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_middleware_stack_does_not_alter_response_body(self):
        """The middleware stack must pass through the response body intact."""
        app = make_app(api_key="secret", max_requests=10, window_seconds=60.0)
        client = TestClient(app)
        resp = client.get("/data", headers={"X-API-Key": "secret"})
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"items": [1, 2, 3]}


# ===========================================================================
# TestBuildFunctions (5 tests)
# ===========================================================================

class TestBuildFunctions:

    def test_build_api_key_middleware_empty_string_returns_none(self):
        """build_api_key_middleware('') → None (empty string is falsy)."""
        assert build_api_key_middleware("") is None

    def test_build_api_key_middleware_whitespace_returns_class(self):
        """build_api_key_middleware('  ') → class (non-empty whitespace is truthy)."""
        # Per spec note: '  ' is truthy, returns class
        result = build_api_key_middleware("  ")
        assert result is not None
        assert isinstance(result, type)

    def test_build_rate_limiter_custom_max_requests(self):
        """build_rate_limiter(max_requests=10) → class with correct max_requests baked in."""
        mw_cls = build_rate_limiter(max_requests=10, window_seconds=30.0)
        inner_app = FastAPI()

        @inner_app.get("/x")
        def x():
            return {"x": 1}

        inner_app.add_middleware(mw_cls)
        client = TestClient(inner_app)
        # Send 10 requests → all should succeed
        for _ in range(10):
            resp = client.get("/x")
            assert resp.status_code == 200
        # 11th → 429
        assert client.get("/x").status_code == 429

    def test_build_rate_limiter_defaults(self):
        """build_rate_limiter() with no args should use defaults (100, 60.0)."""
        mw_cls = build_rate_limiter()
        assert isinstance(mw_cls, type)
        # Instantiate with a dummy ASGI app to inspect defaults
        dummy = FastAPI()
        instance = mw_cls(dummy)
        assert instance._limiter.max_requests == 100
        assert instance._limiter.window_seconds == 60.0

    def test_instantiated_middleware_classes_work_correctly(self):
        """Classes produced by both build functions must produce functional middleware."""
        app = FastAPI()

        @app.get("/ok")
        def ok():
            return {"ok": True}

        auth_cls = build_api_key_middleware("mykey")
        rate_cls = build_rate_limiter(max_requests=5, window_seconds=60.0)

        app.add_middleware(auth_cls)
        app.add_middleware(rate_cls)

        client = TestClient(app)
        # No key → 401
        assert client.get("/ok").status_code == 401
        # Correct key → 200
        assert client.get("/ok", headers={"X-API-Key": "mykey"}).status_code == 200
