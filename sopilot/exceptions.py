"""Shared exception hierarchy for SOPilot services.

All service-layer exceptions inherit from ``ServiceError`` which carries:

- ``error_code``: a machine-readable uppercase string (e.g. ``"VIDEO_NOT_FOUND"``)
- ``context``: an optional dict of structured metadata for diagnostics
"""

from __future__ import annotations

from typing import Any


class ServiceError(RuntimeError):
    """Base class for all SOPilot service exceptions.

    Parameters
    ----------
    message:
        Human-readable error description.
    error_code:
        Machine-readable code such as ``"VIDEO_NOT_FOUND"`` or
        ``"SCORE_JOB_CONFLICT"``.  Defaults to ``"SERVICE_ERROR"``.
    context:
        Optional dict of structured metadata (video_id, job_id, etc.)
        that will be included in error responses and log records.
    """

    def __init__(
        self,
        message: str = "",
        *,
        error_code: str = "SERVICE_ERROR",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code: str = error_code
        self.context: dict[str, Any] = context or {}


class NotFoundError(ServiceError):
    """Raised when a requested resource does not exist."""

    def __init__(
        self,
        message: str = "",
        *,
        error_code: str = "NOT_FOUND",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, error_code=error_code, context=context)


class InvalidStateError(ServiceError):
    """Raised when an operation is invalid for the current resource state."""

    def __init__(
        self,
        message: str = "",
        *,
        error_code: str = "INVALID_STATE",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, error_code=error_code, context=context)


class FileTooLargeError(ServiceError):
    """Raised when an uploaded file exceeds the configured size limit."""

    def __init__(
        self,
        message: str = "",
        *,
        error_code: str = "FILE_TOO_LARGE",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, error_code=error_code, context=context)


class TransientError(ServiceError):
    """Raised for retry-able failures (DB busy, embedder timeout, etc.).

    The ``retry_after`` field suggests how many seconds the client should
    wait before retrying.
    """

    def __init__(
        self,
        message: str = "",
        *,
        error_code: str = "TRANSIENT_ERROR",
        context: dict[str, Any] | None = None,
        retry_after: int = 5,
    ) -> None:
        super().__init__(message, error_code=error_code, context=context)
        self.retry_after: int = retry_after


class ValidationError(ServiceError):
    """Raised when user input fails validation rules."""

    def __init__(
        self,
        message: str = "",
        *,
        error_code: str = "VALIDATION_ERROR",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, error_code=error_code, context=context)


class ConfigurationError(ServiceError):
    """Raised when the service is misconfigured or a required resource is missing."""

    def __init__(
        self,
        message: str = "",
        *,
        error_code: str = "CONFIGURATION_ERROR",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, error_code=error_code, context=context)


class AlgorithmError(ServiceError):
    """Raised when an internal algorithm fails (DTW divergence, scoring NaN, etc.)."""

    def __init__(
        self,
        message: str = "",
        *,
        error_code: str = "ALGORITHM_ERROR",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, error_code=error_code, context=context)
