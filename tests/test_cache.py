"""Tests for sopilot.api.cache module."""

import time

from sopilot.api.cache import timed_cache


class TestTimedCache:
    def test_returns_cached_value(self):
        call_count = 0

        @timed_cache(ttl_seconds=10.0)
        def expensive(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        assert expensive(5) == 10
        assert expensive(5) == 10
        assert call_count == 1

    def test_different_args_not_cached(self):
        call_count = 0

        @timed_cache(ttl_seconds=10.0)
        def fn(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x

        fn(1)
        fn(2)
        assert call_count == 2

    def test_expires_after_ttl(self):
        call_count = 0

        @timed_cache(ttl_seconds=0.05)
        def fn() -> int:
            nonlocal call_count
            call_count += 1
            return 42

        fn()
        assert call_count == 1
        time.sleep(0.1)
        fn()
        assert call_count == 2

    def test_cache_clear(self):
        call_count = 0

        @timed_cache(ttl_seconds=10.0)
        def fn() -> int:
            nonlocal call_count
            call_count += 1
            return 1

        fn()
        assert call_count == 1
        fn.cache_clear()
        fn()
        assert call_count == 2

    def test_kwargs_cached(self):
        call_count = 0

        @timed_cache(ttl_seconds=10.0)
        def fn(a: int, b: int = 0) -> int:
            nonlocal call_count
            call_count += 1
            return a + b

        fn(1, b=2)
        fn(1, b=2)
        assert call_count == 1
        fn(1, b=3)
        assert call_count == 2
