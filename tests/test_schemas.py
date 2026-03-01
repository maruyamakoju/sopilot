"""Tests for sopilot.schemas — Pydantic model validation."""
import pytest
from pydantic import ValidationError

from sopilot.constants import DEFAULT_WEIGHTS
from sopilot.schemas import (
    ScoreRequest,
    ScoreReviewRequest,
    ScoreWeights,
    VideoIngestResponse,
)

# ---------------------------------------------------------------------------
# ScoreWeights
# ---------------------------------------------------------------------------

class TestScoreWeights:
    def test_default_values_from_constants(self):
        w = ScoreWeights()
        assert w.w_miss == DEFAULT_WEIGHTS["w_miss"]
        assert w.w_swap == DEFAULT_WEIGHTS["w_swap"]
        assert w.w_dev == DEFAULT_WEIGHTS["w_dev"]
        assert w.w_time == DEFAULT_WEIGHTS["w_time"]

    def test_custom_values(self):
        w = ScoreWeights(w_miss=0.1, w_swap=0.2, w_dev=0.3, w_time=0.4)
        assert w.w_miss == 0.1
        assert w.w_swap == 0.2
        assert w.w_dev == 0.3
        assert w.w_time == 0.4

    def test_zero_weights_accepted(self):
        w = ScoreWeights(w_miss=0.0, w_swap=0.0, w_dev=0.0, w_time=0.0)
        assert w.w_miss == 0.0

    def test_negative_w_miss_rejected(self):
        with pytest.raises(ValidationError):
            ScoreWeights(w_miss=-0.1)

    def test_negative_w_swap_rejected(self):
        with pytest.raises(ValidationError):
            ScoreWeights(w_swap=-0.01)

    def test_negative_w_dev_rejected(self):
        with pytest.raises(ValidationError):
            ScoreWeights(w_dev=-1.0)

    def test_negative_w_time_rejected(self):
        with pytest.raises(ValidationError):
            ScoreWeights(w_time=-0.5)

    def test_to_core_weights_conversion(self):
        w = ScoreWeights(w_miss=0.5, w_swap=0.2, w_dev=0.2, w_time=0.1)
        core = w.to_core_weights()
        assert core.w_miss == 0.5
        assert core.w_swap == 0.2
        assert core.w_dev == 0.2
        assert core.w_time == 0.1

    def test_to_core_weights_type(self):
        from sopilot.core.scoring import ScoreWeights as CoreScoreWeights
        w = ScoreWeights()
        core = w.to_core_weights()
        assert isinstance(core, CoreScoreWeights)

    def test_to_core_weights_preserves_defaults(self):
        w = ScoreWeights()
        core = w.to_core_weights()
        assert core.w_miss == DEFAULT_WEIGHTS["w_miss"]
        assert core.w_swap == DEFAULT_WEIGHTS["w_swap"]


# ---------------------------------------------------------------------------
# ScoreReviewRequest
# ---------------------------------------------------------------------------

class TestScoreReviewRequest:
    def test_valid_verdict_pass(self):
        r = ScoreReviewRequest(verdict="pass")
        assert r.verdict == "pass"

    def test_valid_verdict_fail(self):
        r = ScoreReviewRequest(verdict="fail")
        assert r.verdict == "fail"

    def test_valid_verdict_needs_review(self):
        r = ScoreReviewRequest(verdict="needs_review")
        assert r.verdict == "needs_review"

    def test_valid_verdict_retrain(self):
        r = ScoreReviewRequest(verdict="retrain")
        assert r.verdict == "retrain"

    def test_invalid_verdict_rejected(self):
        with pytest.raises(ValidationError):
            ScoreReviewRequest(verdict="approved")

    def test_empty_verdict_rejected(self):
        with pytest.raises(ValidationError):
            ScoreReviewRequest(verdict="")

    def test_note_optional(self):
        r = ScoreReviewRequest(verdict="pass")
        assert r.note is None

    def test_note_accepted(self):
        r = ScoreReviewRequest(verdict="pass", note="Looks correct.")
        assert r.note == "Looks correct."

    def test_note_max_length(self):
        long_note = "x" * 5000
        r = ScoreReviewRequest(verdict="fail", note=long_note)
        assert len(r.note) == 5000

    def test_note_exceeds_max_length(self):
        too_long = "x" * 5001
        with pytest.raises(ValidationError):
            ScoreReviewRequest(verdict="fail", note=too_long)


# ---------------------------------------------------------------------------
# ScoreRequest
# ---------------------------------------------------------------------------

class TestScoreRequest:
    def test_required_fields(self):
        r = ScoreRequest(gold_video_id=1, trainee_video_id=2)
        assert r.gold_video_id == 1
        assert r.trainee_video_id == 2

    def test_missing_gold_video_id_rejected(self):
        with pytest.raises(ValidationError):
            ScoreRequest(trainee_video_id=2)

    def test_missing_trainee_video_id_rejected(self):
        with pytest.raises(ValidationError):
            ScoreRequest(gold_video_id=1)

    def test_weights_optional(self):
        r = ScoreRequest(gold_video_id=1, trainee_video_id=2)
        assert r.weights is None

    def test_weights_accepted(self):
        w = ScoreWeights(w_miss=0.5, w_swap=0.2, w_dev=0.2, w_time=0.1)
        r = ScoreRequest(gold_video_id=1, trainee_video_id=2, weights=w)
        assert r.weights is not None
        assert r.weights.w_miss == 0.5


# ---------------------------------------------------------------------------
# VideoIngestResponse
# ---------------------------------------------------------------------------

class TestVideoIngestResponse:
    def test_model_fields(self):
        resp = VideoIngestResponse(
            video_id=1,
            task_id="task-1",
            is_gold=True,
            status="ready",
            clip_count=5,
            step_boundaries=[2, 4],
            original_filename="gold.mp4",
        )
        assert resp.video_id == 1
        assert resp.task_id == "task-1"
        assert resp.is_gold is True
        assert resp.status == "ready"
        assert resp.clip_count == 5
        assert resp.step_boundaries == [2, 4]
        assert resp.original_filename == "gold.mp4"

    def test_original_filename_optional(self):
        resp = VideoIngestResponse(
            video_id=1,
            task_id="task-1",
            is_gold=False,
            status="processing",
            clip_count=0,
            step_boundaries=[],
        )
        assert resp.original_filename is None

    def test_missing_required_field_rejected(self):
        with pytest.raises(ValidationError):
            VideoIngestResponse(
                video_id=1,
                task_id="task-1",
                # missing is_gold, status, clip_count, step_boundaries
            )

    def test_serialization_roundtrip(self):
        resp = VideoIngestResponse(
            video_id=42,
            task_id="t",
            is_gold=True,
            status="ready",
            clip_count=3,
            step_boundaries=[1],
        )
        data = resp.model_dump()
        assert data["video_id"] == 42
        assert data["step_boundaries"] == [1]
        # Reconstruct from dict
        resp2 = VideoIngestResponse(**data)
        assert resp2 == resp
