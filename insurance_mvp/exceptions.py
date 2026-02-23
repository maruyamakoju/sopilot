"""Standardized exception hierarchy for Insurance MVP.

All custom exceptions inherit from ``InsuranceMVPError`` so callers
can catch the whole family with a single ``except InsuranceMVPError``.
"""

from __future__ import annotations


class InsuranceMVPError(Exception):
    """Base exception for all Insurance MVP errors."""


class PipelineStageError(InsuranceMVPError):
    """A named pipeline stage failed.

    Attributes:
        stage: Human-readable stage name (e.g. "mining", "vlm_inference").
        detail: Short explanation of what went wrong.
        cause: The original exception, if any.
    """

    def __init__(self, stage: str, detail: str, cause: Exception | None = None):
        self.stage = stage
        self.detail = detail
        self.cause = cause
        super().__init__(f"Stage '{stage}' failed: {detail}")


class VLMInferenceError(InsuranceMVPError):
    """Video-LLM inference failed after all retries."""

    def __init__(self, message: str, attempts: int = 0, cause: Exception | None = None):
        self.attempts = attempts
        self.cause = cause
        super().__init__(message)


class FrameExtractionError(InsuranceMVPError):
    """Frame extraction from video failed (timeout, corrupt, etc.)."""


class ConfigurationError(InsuranceMVPError):
    """Invalid or missing configuration."""


class DependencyMissingError(InsuranceMVPError):
    """An optional dependency is not installed.

    Attributes:
        package: The missing package name (e.g. "torch", "qwen-vl-utils").
        feature: The feature that requires this package.
    """

    def __init__(self, package: str, feature: str = ""):
        self.package = package
        self.feature = feature
        msg = f"Required package '{package}' is not installed"
        if feature:
            msg += f" (needed for {feature})"
        msg += f". Install with: pip install {package}"
        super().__init__(msg)
