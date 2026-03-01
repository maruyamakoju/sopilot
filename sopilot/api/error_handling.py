"""Decorator for mapping service exceptions to structured HTTP error responses.

Every error response follows a consistent JSON envelope::

    {
        "error": {
            "code": "VIDEO_NOT_FOUND",
            "message": "Video abc123 does not exist",
            "details": {"video_id": "abc123"}
        }
    }

Transient errors include a ``Retry-After`` header so clients can
implement automatic back-off.
"""

import functools
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import HTTPException, Request
from starlette.responses import JSONResponse

from sopilot.exceptions import (
    AlgorithmError,
    ConfigurationError,
    FileTooLargeError,
    InvalidStateError,
    NotFoundError,
    ServiceError,
    TransientError,
    ValidationError,
)

logger = logging.getLogger(__name__)


def _build_error_body(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the canonical ``{"error": {...}}`` envelope."""
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
        }
    }


def _extract_request_meta(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, str | None]:
    """Best-effort extraction of request metadata from endpoint arguments.

    FastAPI injects ``Request`` either positionally or as a keyword arg.
    We also check for the correlation-ID header set by CorrelationIDMiddleware.
    """
    request: Request | None = None
    for a in args:
        if isinstance(a, Request):
            request = a
            break
    if request is None:
        for v in kwargs.values():
            if isinstance(v, Request):
                request = v
                break

    if request is not None:
        return {
            "request_id": request.headers.get("X-Request-ID"),
            "endpoint": f"{request.method} {request.url.path}",
        }
    return {"request_id": None, "endpoint": None}


def service_errors(fn: Callable[..., Coroutine[Any, Any, Any]]) -> Callable[..., Coroutine[Any, Any, Any]]:
    """Maps ServiceError subclasses to structured JSON error responses.

    Exception mapping:
    - ``NotFoundError``      -> 404
    - ``InvalidStateError``  -> 409
    - ``FileTooLargeError``  -> 413
    - ``ValidationError``    -> 422
    - ``TransientError``     -> 503  (with ``Retry-After`` header)
    - ``ConfigurationError`` -> 500
    - ``AlgorithmError``     -> 500
    - ``ServiceError`` (base)-> 400
    - Any other exception    -> 500 (catch-all)
    """

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        try:
            return await fn(*args, **kwargs)
        except NotFoundError as exc:
            _log_service_error(exc, 404, t0, args, kwargs)
            return JSONResponse(
                status_code=404,
                content=_build_error_body(exc.error_code, str(exc), exc.context),
            )
        except InvalidStateError as exc:
            _log_service_error(exc, 409, t0, args, kwargs)
            return JSONResponse(
                status_code=409,
                content=_build_error_body(exc.error_code, str(exc), exc.context),
            )
        except FileTooLargeError as exc:
            _log_service_error(exc, 413, t0, args, kwargs)
            return JSONResponse(
                status_code=413,
                content=_build_error_body(exc.error_code, str(exc), exc.context),
            )
        except ValidationError as exc:
            _log_service_error(exc, 422, t0, args, kwargs)
            return JSONResponse(
                status_code=422,
                content=_build_error_body(exc.error_code, str(exc), exc.context),
            )
        except TransientError as exc:
            _log_service_error(exc, 503, t0, args, kwargs)
            return JSONResponse(
                status_code=503,
                headers={"Retry-After": str(exc.retry_after)},
                content=_build_error_body(exc.error_code, str(exc), exc.context),
            )
        except ConfigurationError as exc:
            _log_service_error(exc, 500, t0, args, kwargs, level=logging.ERROR)
            return JSONResponse(
                status_code=500,
                content=_build_error_body(exc.error_code, str(exc), exc.context),
            )
        except AlgorithmError as exc:
            _log_service_error(exc, 500, t0, args, kwargs, level=logging.ERROR)
            return JSONResponse(
                status_code=500,
                content=_build_error_body(exc.error_code, str(exc), exc.context),
            )
        except ServiceError as exc:
            _log_service_error(exc, 400, t0, args, kwargs)
            return JSONResponse(
                status_code=400,
                content=_build_error_body(exc.error_code, str(exc), exc.context),
            )
        except HTTPException:
            # Let FastAPI's own HTTPException propagate untouched so that
            # endpoints can still use `raise HTTPException(...)` directly.
            raise
        except Exception:
            # Catch-all for unexpected exceptions — never leak internals.
            duration_ms = (time.monotonic() - t0) * 1000
            meta = _extract_request_meta(args, kwargs)
            logger.exception(
                "unhandled exception in %s | request_id=%s endpoint=%s duration_ms=%.1f",
                fn.__qualname__,
                meta.get("request_id"),
                meta.get("endpoint"),
                duration_ms,
            )
            return JSONResponse(
                status_code=500,
                content=_build_error_body(
                    "INTERNAL_ERROR",
                    "An unexpected error occurred. Check logs for details.",
                    {},
                ),
            )

    # Fix: Clear copied string annotations so FastAPI resolves types
    # via __wrapped__ using the original function's globals context.
    wrapper.__annotations__ = {}
    return wrapper


def _log_service_error(
    exc: ServiceError,
    status: int,
    t0: float,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    level: int = logging.WARNING,
) -> None:
    """Emit a structured log record for a handled service exception."""
    duration_ms = (time.monotonic() - t0) * 1000
    meta = _extract_request_meta(args, kwargs)
    logger.log(
        level,
        "%s -> %d | code=%s request_id=%s endpoint=%s duration_ms=%.1f context=%s",
        type(exc).__name__,
        status,
        exc.error_code,
        meta.get("request_id"),
        meta.get("endpoint"),
        duration_ms,
        exc.context,
        exc_info=(level >= logging.ERROR),
    )
