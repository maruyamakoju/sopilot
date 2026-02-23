"""Tests for insurance_mvp.pipeline.protocols module.

Covers:
- MiningResult, VLMResult, SeverityPrediction: dataclass creation, field access
- TypedPipeline: register_stage, run with simple functions, timing, error handling
- Protocol classes: structural matching via isinstance checks
"""

import time

import pytest

from insurance_mvp.pipeline.protocols import (
    CalibrationBackend,
    EvaluationBackend,
    MiningBackend,
    MiningResult,
    PipelineStageResult,
    SeverityPrediction,
    SeverityPredictor,
    StageExecutionError,
    TypedPipeline,
    VLMBackend,
    VLMResult,
)


# ---------------------------------------------------------------------------
# MiningResult
# ---------------------------------------------------------------------------


class TestMiningResult:
    """Tests for the MiningResult frozen dataclass."""

    def test_basic_creation(self):
        """Create a MiningResult with valid fields."""
        mr = MiningResult(
            clip_id="c1",
            start_sec=1.0,
            end_sec=5.0,
            danger_score=0.8,
            motion_score=0.7,
            proximity_score=0.6,
            video_path="/video.mp4",
        )
        assert mr.clip_id == "c1"
        assert mr.start_sec == 1.0
        assert mr.end_sec == 5.0
        assert mr.danger_score == 0.8
        assert mr.motion_score == 0.7
        assert mr.proximity_score == 0.6
        assert mr.video_path == "/video.mp4"

    def test_duration_sec_property(self):
        """duration_sec computes end - start."""
        mr = MiningResult(
            clip_id="c2", start_sec=2.5, end_sec=7.5,
            danger_score=0.5, motion_score=0.5, proximity_score=0.5,
            video_path="v.mp4",
        )
        assert mr.duration_sec == pytest.approx(5.0)

    def test_end_before_start_raises(self):
        """end_sec <= start_sec raises ValueError."""
        with pytest.raises(ValueError, match="must be strictly greater"):
            MiningResult(
                clip_id="bad", start_sec=10.0, end_sec=5.0,
                danger_score=0.5, motion_score=0.5, proximity_score=0.5,
                video_path="v.mp4",
            )

    def test_end_equals_start_raises(self):
        """end_sec == start_sec raises ValueError (zero-length clip)."""
        with pytest.raises(ValueError, match="must be strictly greater"):
            MiningResult(
                clip_id="bad", start_sec=3.0, end_sec=3.0,
                danger_score=0.5, motion_score=0.5, proximity_score=0.5,
                video_path="v.mp4",
            )

    def test_frozen(self):
        """MiningResult is frozen -- attribute assignment raises."""
        mr = MiningResult(
            clip_id="c3", start_sec=0.0, end_sec=1.0,
            danger_score=0.5, motion_score=0.5, proximity_score=0.5,
            video_path="v.mp4",
        )
        with pytest.raises(AttributeError):
            mr.clip_id = "changed"


# ---------------------------------------------------------------------------
# VLMResult
# ---------------------------------------------------------------------------


class TestVLMResult:
    """Tests for the VLMResult frozen dataclass."""

    def test_basic_creation(self):
        """Create a VLMResult with required fields only."""
        vr = VLMResult(severity="HIGH", confidence=0.9, reasoning="collision detected")
        assert vr.severity == "HIGH"
        assert vr.confidence == 0.9
        assert vr.reasoning == "collision detected"
        assert vr.hazards == []
        assert vr.evidence == []

    def test_with_hazards_and_evidence(self):
        """VLMResult carries hazard and evidence lists."""
        vr = VLMResult(
            severity="MEDIUM",
            confidence=0.7,
            reasoning="near miss",
            hazards=["pedestrian", "cyclist"],
            evidence=["frame 45", "frame 60"],
        )
        assert len(vr.hazards) == 2
        assert "pedestrian" in vr.hazards
        assert len(vr.evidence) == 2

    def test_frozen(self):
        """VLMResult is frozen."""
        vr = VLMResult(severity="LOW", confidence=0.5, reasoning="minor")
        with pytest.raises(AttributeError):
            vr.severity = "HIGH"


# ---------------------------------------------------------------------------
# SeverityPrediction
# ---------------------------------------------------------------------------


class TestSeverityPrediction:
    """Tests for the SeverityPrediction frozen dataclass."""

    def test_basic_creation(self):
        """Create a SeverityPrediction with all fields."""
        sp = SeverityPrediction(
            severity="HIGH",
            confidence=0.85,
            prediction_set=frozenset({"HIGH", "MEDIUM"}),
            calibrated_probs={"NONE": 0.05, "LOW": 0.05, "MEDIUM": 0.15, "HIGH": 0.75},
        )
        assert sp.severity == "HIGH"
        assert sp.confidence == 0.85
        assert "HIGH" in sp.prediction_set
        assert sp.adjustment_reason == ""

    def test_is_uncertain_true(self):
        """is_uncertain is True when prediction_set has > 1 element."""
        sp = SeverityPrediction(
            severity="MEDIUM",
            confidence=0.6,
            prediction_set=frozenset({"LOW", "MEDIUM"}),
            calibrated_probs={"NONE": 0.1, "LOW": 0.3, "MEDIUM": 0.4, "HIGH": 0.2},
        )
        assert sp.is_uncertain is True

    def test_is_uncertain_false(self):
        """is_uncertain is False when prediction_set has exactly 1 element."""
        sp = SeverityPrediction(
            severity="HIGH",
            confidence=0.95,
            prediction_set=frozenset({"HIGH"}),
            calibrated_probs={"NONE": 0.01, "LOW": 0.01, "MEDIUM": 0.03, "HIGH": 0.95},
        )
        assert sp.is_uncertain is False

    def test_adjustment_reason(self):
        """adjustment_reason carries the rationale string."""
        sp = SeverityPrediction(
            severity="HIGH",
            confidence=0.9,
            prediction_set=frozenset({"HIGH"}),
            calibrated_probs={"HIGH": 0.9},
            adjustment_reason="mining danger_score override",
        )
        assert sp.adjustment_reason == "mining danger_score override"


# ---------------------------------------------------------------------------
# Protocol structural matching
# ---------------------------------------------------------------------------


class TestProtocolStructuralMatching:
    """Verify that Protocol classes support runtime isinstance checks."""

    def test_mining_backend_structural_match(self):
        """A class with extract_clips(video_path, top_k) matches MiningBackend."""

        class FakeMiner:
            def extract_clips(self, video_path: str, top_k: int = 5) -> list:
                return []

        assert isinstance(FakeMiner(), MiningBackend)

    def test_vlm_backend_structural_match(self):
        """A class with assess(video_path, start_sec, end_sec) matches VLMBackend."""

        class FakeVLM:
            def assess(self, video_path: str, start_sec: float, end_sec: float):
                return VLMResult(severity="NONE", confidence=0.5, reasoning="ok")

        assert isinstance(FakeVLM(), VLMBackend)

    def test_severity_predictor_structural_match(self):
        """A class with predict(vlm_result, mining_result) matches SeverityPredictor."""

        class FakePredictor:
            def predict(self, vlm_result, mining_result):
                return None

        assert isinstance(FakePredictor(), SeverityPredictor)

    def test_calibration_backend_structural_match(self):
        """A class with calibrate() and fit() matches CalibrationBackend."""

        class FakeCalibrator:
            def calibrate(self, raw_probs):
                return raw_probs

            def fit(self, scores, labels):
                pass

        assert isinstance(FakeCalibrator(), CalibrationBackend)

    def test_evaluation_backend_structural_match(self):
        """A class with evaluate(y_true, y_pred) matches EvaluationBackend."""

        class FakeEvaluator:
            def evaluate(self, y_true, y_pred):
                return {"accuracy": 1.0}

        assert isinstance(FakeEvaluator(), EvaluationBackend)

    def test_non_matching_class_fails(self):
        """A class without the required method does NOT match the protocol."""

        class NotAMiner:
            def some_other_method(self):
                pass

        assert not isinstance(NotAMiner(), MiningBackend)


# ---------------------------------------------------------------------------
# TypedPipeline
# ---------------------------------------------------------------------------


class TestTypedPipeline:
    """Tests for the TypedPipeline orchestrator."""

    def test_empty_pipeline_raises(self):
        """Running an empty pipeline raises RuntimeError."""
        pipe = TypedPipeline()
        with pytest.raises(RuntimeError, match="Cannot run an empty pipeline"):
            pipe.run("input")

    def test_register_stage_returns_self(self):
        """register_stage returns self for fluent chaining."""
        pipe = TypedPipeline()
        result = pipe.register_stage("a", lambda x: x)
        assert result is pipe

    def test_register_duplicate_stage_raises(self):
        """Registering a stage with the same name twice raises ValueError."""
        pipe = TypedPipeline()
        pipe.register_stage("dup", lambda x: x)
        with pytest.raises(ValueError, match="already registered"):
            pipe.register_stage("dup", lambda x: x)

    def test_register_non_callable_raises(self):
        """Registering a non-callable raises TypeError."""
        pipe = TypedPipeline()
        with pytest.raises(TypeError, match="callable"):
            pipe.register_stage("bad", 42)

    def test_single_stage_run(self):
        """A single-stage pipeline returns the stage output."""
        pipe = TypedPipeline()
        pipe.register_stage("double", lambda x: x * 2)
        result = pipe.run(5)
        assert result.output == 10
        assert result.success is True
        assert len(result.stage_results) == 1
        assert result.stage_results[0].stage_name == "double"
        assert result.stage_results[0].success is True

    def test_multi_stage_chaining(self):
        """Multiple stages chain: each receives the previous stage's output."""
        pipe = TypedPipeline()
        pipe.register_stage("add1", lambda x: x + 1)
        pipe.register_stage("mul3", lambda x: x * 3)
        pipe.register_stage("str", lambda x: f"result={x}")
        result = pipe.run(2)
        # (2 + 1) * 3 = 9 -> "result=9"
        assert result.output == "result=9"
        assert result.success is True
        assert len(result.stage_results) == 3

    def test_timing_recorded(self):
        """Each stage records non-negative timing_sec."""
        pipe = TypedPipeline()
        pipe.register_stage("sleep", lambda x: (time.sleep(0.05), x)[1])
        result = pipe.run("test")
        assert result.total_sec >= 0.02
        assert result.stage_results[0].timing_sec >= 0.02

    def test_stage_names_and_count(self):
        """stage_names and stage_count reflect registered stages."""
        pipe = TypedPipeline()
        pipe.register_stage("a", lambda x: x)
        pipe.register_stage("b", lambda x: x)
        pipe.register_stage("c", lambda x: x)
        assert pipe.stage_names == ["a", "b", "c"]
        assert pipe.stage_count == 3

    def test_has_stage(self):
        """has_stage returns True for registered names, False otherwise."""
        pipe = TypedPipeline()
        pipe.register_stage("mining", lambda x: x)
        assert pipe.has_stage("mining")
        assert not pipe.has_stage("vlm")

    def test_len(self):
        """len(pipeline) returns stage count."""
        pipe = TypedPipeline()
        assert len(pipe) == 0
        pipe.register_stage("x", lambda x: x)
        assert len(pipe) == 1

    def test_repr(self):
        """repr shows stage names."""
        pipe = TypedPipeline()
        pipe.register_stage("mine", lambda x: x)
        pipe.register_stage("predict", lambda x: x)
        r = repr(pipe)
        assert "mine" in r
        assert "predict" in r

    def test_error_stops_pipeline(self):
        """By default, an error in a stage stops the pipeline."""
        pipe = TypedPipeline(continue_on_error=False)
        pipe.register_stage("fail", lambda x: 1 / 0)
        pipe.register_stage("unreachable", lambda x: x)
        result = pipe.run(1)
        assert result.success is False
        assert len(result.stage_results) == 1
        assert result.stage_results[0].success is False
        assert "error" in result.stage_results[0].metadata
        assert result.output is None

    def test_continue_on_error(self):
        """With continue_on_error=True, subsequent stages receive None."""
        pipe = TypedPipeline(continue_on_error=True)
        pipe.register_stage("fail", lambda x: 1 / 0)
        pipe.register_stage("recover", lambda x: "recovered" if x is None else x)
        result = pipe.run(1)
        assert result.success is False
        assert len(result.stage_results) == 2
        assert result.stage_results[0].success is False
        assert result.stage_results[1].success is True
        assert result.output == "recovered"

    def test_stage_result_type_provenance(self):
        """PipelineStageResult captures input/output type names."""
        pipe = TypedPipeline()
        pipe.register_stage("to_list", lambda x: [x, x])
        result = pipe.run("hello")
        sr = result.stage_results[0]
        assert "str" in sr.input_type
        assert "list" in sr.output_type

    def test_fluent_chaining(self):
        """Stages can be registered via fluent chaining."""
        pipe = (
            TypedPipeline()
            .register_stage("a", lambda x: x + 1)
            .register_stage("b", lambda x: x * 2)
        )
        result = pipe.run(3)
        assert result.output == 8


# ---------------------------------------------------------------------------
# StageExecutionError
# ---------------------------------------------------------------------------


class TestStageExecutionError:
    """Tests for the StageExecutionError exception."""

    def test_creation(self):
        """StageExecutionError stores stage_name and cause."""
        cause = ValueError("bad input")
        err = StageExecutionError("mining", cause)
        assert err.stage_name == "mining"
        assert err.cause is cause
        assert "mining" in str(err)
        assert "bad input" in str(err)


# ---------------------------------------------------------------------------
# PipelineStageResult
# ---------------------------------------------------------------------------


class TestPipelineStageResult:
    """Tests for the PipelineStageResult generic dataclass."""

    def test_creation(self):
        """Basic PipelineStageResult creation."""
        psr = PipelineStageResult(
            stage_name="test",
            input_type="str",
            output_type="int",
            output=42,
            timing_sec=0.01,
        )
        assert psr.stage_name == "test"
        assert psr.output == 42
        assert psr.success is True
        assert psr.metadata == {}

    def test_frozen(self):
        """PipelineStageResult is frozen."""
        psr = PipelineStageResult(
            stage_name="x", input_type="str", output_type="str",
            output="out", timing_sec=0.0,
        )
        with pytest.raises(AttributeError):
            psr.stage_name = "changed"
