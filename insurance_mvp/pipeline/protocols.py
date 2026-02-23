"""Pipeline Protocol Definitions for Type-Safe Stage Composition.

This module defines the structural contracts (typing.Protocol) and value types
(dataclasses) that govern the insurance video-analysis pipeline.  Every stage
-- mining, VLM inference, severity prediction, calibration, evaluation -- is
expressed as a Protocol so that concrete backends can be swapped at runtime
without subclassing.  The TypedPipeline orchestrator chains registered stages
with automatic timing, error capture, and stage-level introspection.

Design principles:
    1. Structural subtyping only -- no ABC inheritance required.
    2. Value objects are frozen dataclasses for immutability and hashability.
    3. Generic PipelineStageResult preserves per-stage type provenance.
    4. Zero imports from insurance_mvp -- this file is self-contained.

References:
    - PEP 544  (typing.Protocol)
    - PEP 557  (dataclasses)
    - PEP 604  (X | Y union syntax)
    - PEP 613  (TypeAlias)
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

# ---------------------------------------------------------------------------
# Foundational types
# ---------------------------------------------------------------------------

SeverityLevel: TypeAlias = Literal["NONE", "LOW", "MEDIUM", "HIGH"]
"""Closed enumeration of incident severity levels.

NONE   -- no incident detected (normal driving).
LOW    -- minor event, no contact (hard braking, swerve).
MEDIUM -- moderate event, possible contact (near miss, minor bump).
HIGH   -- severe event, confirmed contact or dangerous hazard.
"""

# ---------------------------------------------------------------------------
# Value objects (frozen dataclasses)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MiningResult:
    """Output of a single danger-clip extraction from a video file.

    Each field corresponds to a signal channel in the multimodal mining
    stage (motion, proximity, audio) fused into a composite danger score.

    Attributes:
        clip_id:         Unique identifier for this clip within the video.
        start_sec:       Start timestamp in seconds.
        end_sec:         End timestamp in seconds.
        danger_score:    Fused danger score in [0, 1].
        motion_score:    Optical-flow / acceleration magnitude in [0, 1].
        proximity_score: Object-proximity signal in [0, 1].
        video_path:      Filesystem path to the source video.
    """

    clip_id: str
    start_sec: float
    end_sec: float
    danger_score: float
    motion_score: float
    proximity_score: float
    video_path: str

    def __post_init__(self) -> None:
        if self.end_sec <= self.start_sec:
            raise ValueError(
                f"end_sec ({self.end_sec}) must be strictly greater "
                f"than start_sec ({self.start_sec})"
            )

    @property
    def duration_sec(self) -> float:
        """Clip duration in seconds."""
        return self.end_sec - self.start_sec


@dataclass(frozen=True, slots=True)
class VLMResult:
    """Output of a Video-Language Model assessment on a single clip.

    Attributes:
        severity:    Predicted severity level.
        confidence:  Model confidence in [0, 1].
        reasoning:   Free-text causal explanation.
        hazards:     List of detected hazard descriptions.
        evidence:    List of temporal evidence descriptions.
    """

    severity: SeverityLevel
    confidence: float
    reasoning: str
    hazards: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SeverityPrediction:
    """Output of conformal-calibrated severity prediction.

    Combines a point prediction with a conformal prediction set and
    per-class calibrated probabilities for uncertainty quantification.

    Attributes:
        severity:          Point-predicted severity level.
        confidence:        Calibrated confidence in [0, 1].
        prediction_set:    Conformal prediction set (coverage-guaranteed).
        calibrated_probs:  Mapping from severity level to calibrated
                           probability.
        adjustment_reason: Human-readable rationale for any post-hoc
                           recalibration applied (e.g. mining-signal
                           override).  Empty string if no adjustment.
    """

    severity: SeverityLevel
    confidence: float
    prediction_set: frozenset[str]
    calibrated_probs: dict[str, float]
    adjustment_reason: str = ""

    @property
    def is_uncertain(self) -> bool:
        """True when the prediction set contains more than one level."""
        return len(self.prediction_set) > 1


# ---------------------------------------------------------------------------
# Protocol definitions (structural subtyping)
# ---------------------------------------------------------------------------


@runtime_checkable
class MiningBackend(Protocol):
    """Extracts the top-k most dangerous clips from a video file.

    Implementors: SignalFuser, MockMiningBackend, etc.
    """

    def extract_clips(
        self,
        video_path: str,
        top_k: int = 5,
    ) -> list[MiningResult]: ...


@runtime_checkable
class VLMBackend(Protocol):
    """Assesses a video clip's severity using a Video-Language Model.

    Implementors: CosmosClient, Qwen25VLBackend, MockVLMBackend, etc.
    """

    def assess(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
    ) -> VLMResult: ...


@runtime_checkable
class SeverityPredictor(Protocol):
    """Produces a calibrated severity prediction from VLM + mining signals.

    Implementors: RecalibrationPredictor, ConformalPredictor, etc.
    """

    def predict(
        self,
        vlm_result: VLMResult,
        mining_result: MiningResult,
    ) -> SeverityPrediction: ...


@runtime_checkable
class CalibrationBackend(Protocol):
    """Fits and applies probability calibration (e.g. isotonic, Platt).

    ``fit`` ingests held-out calibration data.
    ``calibrate`` maps raw softmax probabilities to calibrated ones.

    Implementors: SplitConformal, IsotonicCalibrator, etc.
    """

    def calibrate(
        self,
        raw_probs: dict[str, float],
    ) -> dict[str, float]: ...

    def fit(
        self,
        scores: list[list[float]],
        labels: list[int],
    ) -> None: ...


@runtime_checkable
class EvaluationBackend(Protocol):
    """Computes evaluation metrics from ground-truth and predicted labels.

    Implementors: SklearnEvaluator, InsuranceAccuracyReport, etc.
    """

    def evaluate(
        self,
        y_true: list[str],
        y_pred: list[str],
    ) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Pipeline stage result (generic, typed)
# ---------------------------------------------------------------------------

_In = TypeVar("_In")
_Out = TypeVar("_Out")


@dataclass(frozen=True, slots=True)
class PipelineStageResult(Generic[_In, _Out]):
    """Captures the outcome of a single pipeline stage execution.

    Generic over the input type ``_In`` and output type ``_Out`` so that
    downstream consumers can recover the concrete types via
    ``stage_result.output``.

    Attributes:
        stage_name:  Human-readable name of the stage.
        input_type:  Qualified class name of the input payload.
        output_type: Qualified class name of the output payload.
        output:      The actual output value produced by the stage.
        timing_sec:  Wall-clock execution time in seconds.
        metadata:    Arbitrary key-value metadata (e.g. clip count,
                     model version, error message).
        success:     Whether the stage completed without error.
    """

    stage_name: str
    input_type: str
    output_type: str
    output: _Out
    timing_sec: float
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True


# ---------------------------------------------------------------------------
# Typed pipeline orchestrator
# ---------------------------------------------------------------------------

# A stage callable accepts any input and returns any output.
_StageCallable: TypeAlias = Callable[[Any], Any]


class StageExecutionError(Exception):
    """Raised when a pipeline stage fails during execution."""

    def __init__(self, stage_name: str, cause: Exception) -> None:
        self.stage_name = stage_name
        self.cause = cause
        super().__init__(f"Stage '{stage_name}' failed: {cause}")


def _qualified_name(obj: object) -> str:
    """Return the fully-qualified type name of *obj*."""
    cls = type(obj)
    module = cls.__module__
    qualname = cls.__qualname__
    if module and module != "builtins":
        return f"{module}.{qualname}"
    return qualname


class TypedPipeline:
    """A typed, introspectable, linear pipeline of named stages.

    Stages are registered via :meth:`register_stage` and executed in
    registration order by :meth:`run`.  Each stage receives the output
    of the previous stage as its sole argument (the first stage receives
    the external input).

    The pipeline records a :class:`PipelineStageResult` for every stage,
    providing wall-clock timing, type provenance, success/failure status,
    and optional metadata.

    Example::

        pipe = TypedPipeline()
        pipe.register_stage("mining",   mining_backend.extract_clips)
        pipe.register_stage("vlm",      vlm_backend.assess)
        pipe.register_stage("predict",  severity_predictor.predict)

        result = pipe.run(video_path)
        for r in result.stage_results:
            print(f"{r.stage_name}: {r.timing_sec:.3f}s  ok={r.success}")
    """

    def __init__(self, *, continue_on_error: bool = False) -> None:
        """Initialise a new pipeline.

        Args:
            continue_on_error: When True, a failing stage records the
                error in :attr:`PipelineStageResult.metadata` and passes
                ``None`` as input to the next stage.  When False (the
                default), a :class:`StageExecutionError` is raised
                immediately.
        """
        self._stages: OrderedDict[str, _StageCallable] = OrderedDict()
        self._continue_on_error = continue_on_error

    # -- Registration -------------------------------------------------------

    def register_stage(
        self,
        name: str,
        fn: _StageCallable,
    ) -> TypedPipeline:
        """Register a named stage.

        Args:
            name: Unique, human-readable stage name.
            fn:   Callable that accepts one argument (the output of the
                  previous stage) and returns the stage output.

        Returns:
            ``self``, for fluent chaining.

        Raises:
            ValueError: If *name* is already registered.
            TypeError:  If *fn* is not callable.
        """
        if name in self._stages:
            raise ValueError(f"Stage '{name}' is already registered")
        if not callable(fn):
            raise TypeError(f"Stage function must be callable, got {type(fn).__name__}")
        self._stages[name] = fn
        return self

    # -- Inspection ---------------------------------------------------------

    @property
    def stage_names(self) -> list[str]:
        """Ordered list of registered stage names."""
        return list(self._stages.keys())

    @property
    def stage_count(self) -> int:
        """Number of registered stages."""
        return len(self._stages)

    def has_stage(self, name: str) -> bool:
        """Return True if a stage with *name* is registered."""
        return name in self._stages

    # -- Execution ----------------------------------------------------------

    @dataclass(frozen=True, slots=True)
    class PipelineRunResult:
        """Aggregate result of a full pipeline run.

        Attributes:
            output:        Final output of the last stage (or None on error).
            stage_results: Per-stage result objects in execution order.
            total_sec:     Wall-clock time for the entire run.
            success:       True iff every stage succeeded.
        """

        output: Any
        stage_results: list[PipelineStageResult]
        total_sec: float
        success: bool

    def run(self, initial_input: Any) -> PipelineRunResult:
        """Execute all registered stages sequentially.

        The first stage receives *initial_input*; each subsequent stage
        receives the output of the previous one.

        Args:
            initial_input: Payload fed into the first stage.

        Returns:
            A :class:`PipelineRunResult` containing per-stage results,
            the final output, and aggregate timing.

        Raises:
            StageExecutionError: If a stage raises and
                ``continue_on_error`` is False.
            RuntimeError: If no stages have been registered.
        """
        if not self._stages:
            raise RuntimeError("Cannot run an empty pipeline -- register at least one stage")

        stage_results: list[PipelineStageResult] = []
        current_input: Any = initial_input
        all_ok = True
        run_start = time.perf_counter()

        for name, fn in self._stages.items():
            stage_start = time.perf_counter()
            try:
                output = fn(current_input)
                elapsed = time.perf_counter() - stage_start
                result: PipelineStageResult = PipelineStageResult(
                    stage_name=name,
                    input_type=_qualified_name(current_input),
                    output_type=_qualified_name(output),
                    output=output,
                    timing_sec=elapsed,
                    success=True,
                )
                stage_results.append(result)
                current_input = output

            except Exception as exc:
                elapsed = time.perf_counter() - stage_start
                all_ok = False
                error_result: PipelineStageResult = PipelineStageResult(
                    stage_name=name,
                    input_type=_qualified_name(current_input),
                    output_type="NoneType",
                    output=None,
                    timing_sec=elapsed,
                    metadata={
                        "error": str(exc),
                        "error_type": type(exc).__qualname__,
                    },
                    success=False,
                )
                stage_results.append(error_result)

                if not self._continue_on_error:
                    total_sec = time.perf_counter() - run_start
                    return TypedPipeline.PipelineRunResult(
                        output=None,
                        stage_results=stage_results,
                        total_sec=total_sec,
                        success=False,
                    )
                # Propagate None to next stage in lenient mode.
                current_input = None

        total_sec = time.perf_counter() - run_start
        return TypedPipeline.PipelineRunResult(
            output=current_input,
            stage_results=stage_results,
            total_sec=total_sec,
            success=all_ok,
        )

    # -- Representation -----------------------------------------------------

    def __repr__(self) -> str:
        stages = " -> ".join(self._stages.keys()) or "(empty)"
        return f"TypedPipeline({stages})"

    def __len__(self) -> int:
        return len(self._stages)
