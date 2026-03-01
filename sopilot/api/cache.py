"""Simple in-memory TTL cache for expensive endpoint responses.

Usage::

    @timed_cache(ttl_seconds=10)
    def get_dataset_summary(self) -> dict:
        ...

The cache is keyed on all positional and keyword arguments. Thread-safe.
"""

import functools
import time
from collections.abc import Callable
from threading import Lock
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class _CacheEntry:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, expires_at: float) -> None:
        self.value = value
        self.expires_at = expires_at


def timed_cache(ttl_seconds: float = 10.0) -> Callable[[F], F]:
    """Decorator that caches return values for ``ttl_seconds``.

    - Cache is keyed on ``(args, frozenset(kwargs.items()))``.
    - Thread-safe via a per-function lock.
    - Call ``func.cache_clear()`` to manually invalidate.
    """
    def decorator(fn: F) -> F:
        cache: dict[tuple[Any, ...], _CacheEntry] = {}
        lock = Lock()

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = (args, tuple(sorted(kwargs.items())))
            now = time.monotonic()

            with lock:
                entry = cache.get(key)
                if entry is not None and entry.expires_at > now:
                    return entry.value

            # Compute outside the lock to avoid blocking other calls
            result = fn(*args, **kwargs)

            with lock:
                cache[key] = _CacheEntry(result, now + ttl_seconds)
                # Prune expired entries (keep cache bounded)
                expired = [k for k, v in cache.items() if v.expires_at <= now]
                for k in expired:
                    del cache[k]

            return result

        def cache_clear() -> None:
            with lock:
                cache.clear()

        wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator
