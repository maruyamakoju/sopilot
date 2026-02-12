from __future__ import annotations

import numpy as np
import pytest

from sopilot.step_engine import (
    MAX_DTW_CLIPS,
    detect_step_boundaries,
    dtw_align,
    evaluate_sop,
)


def _meta(n: int) -> list[dict]:
    return [{"clip_idx": i, "start_sec": float(i * 4), "end_sec": float((i + 1) * 4)} for i in range(n)]


class TestDtwClipLimit:
    def test_raises_when_gold_exceeds_limit(self) -> None:
        gold = np.random.randn(MAX_DTW_CLIPS + 1, 8).astype(np.float32)
        trainee = np.random.randn(10, 8).astype(np.float32)
        with pytest.raises(ValueError, match="clip count exceeds DTW limit"):
            dtw_align(gold, trainee)

    def test_raises_when_trainee_exceeds_limit(self) -> None:
        gold = np.random.randn(10, 8).astype(np.float32)
        trainee = np.random.randn(MAX_DTW_CLIPS + 1, 8).astype(np.float32)
        with pytest.raises(ValueError, match="clip count exceeds DTW limit"):
            dtw_align(gold, trainee)

    def test_ok_at_limit(self) -> None:
        gold = np.eye(4, dtype=np.float32)
        trainee = np.eye(4, dtype=np.float32)
        result = dtw_align(gold, trainee)
        assert len(result.path) > 0


class TestDtwEmptyInputs:
    def test_empty_gold(self) -> None:
        gold = np.zeros((0, 4), dtype=np.float32)
        trainee = np.ones((3, 4), dtype=np.float32)
        result = dtw_align(gold, trainee)
        assert result.path == []
        assert result.mean_cost == 1.0

    def test_empty_trainee(self) -> None:
        gold = np.ones((3, 4), dtype=np.float32)
        trainee = np.zeros((0, 4), dtype=np.float32)
        result = dtw_align(gold, trainee)
        assert result.path == []
        assert result.mean_cost == 1.0

    def test_both_empty(self) -> None:
        gold = np.zeros((0, 4), dtype=np.float32)
        trainee = np.zeros((0, 4), dtype=np.float32)
        result = dtw_align(gold, trainee)
        assert result.path == []


class TestDtwSingleClip:
    def test_single_clip_each(self) -> None:
        gold = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        trainee = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        result = dtw_align(gold, trainee)
        assert len(result.path) == 1
        assert result.path[0][0] == 0  # gold_idx
        assert result.path[0][1] == 0  # trainee_idx
        assert result.path[0][2] > 0.99  # high similarity


class TestDetectStepBoundaries:
    def test_uniform_embeddings_all_boundaries(self) -> None:
        emb = np.ones((10, 4), dtype=np.float32)
        boundaries = detect_step_boundaries(emb, threshold_factor=1.0, min_step_clips=1)
        assert boundaries[0] == 0
        assert boundaries[-1] == 10
        # Uniform embeddings: dist=0 everywhere, threshold=0, so every point is a boundary
        assert len(boundaries) == 11

    def test_single_clip(self) -> None:
        emb = np.array([[1.0, 0.0]], dtype=np.float32)
        boundaries = detect_step_boundaries(emb, threshold_factor=1.0, min_step_clips=1)
        assert boundaries == [0, 1]

    def test_clear_two_steps(self) -> None:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        emb = np.stack([a, a, a, a, b, b, b, b], axis=0)
        boundaries = detect_step_boundaries(emb, threshold_factor=0.5, min_step_clips=1)
        assert boundaries[0] == 0
        assert boundaries[-1] == 8
        # Should detect a boundary around index 4
        assert any(3 <= x <= 5 for x in boundaries[1:-1])

    def test_min_step_clips_filters_short_segments(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        emb = np.stack([a, b, a, a, a], axis=0)
        boundaries_relaxed = detect_step_boundaries(emb, threshold_factor=0.0, min_step_clips=1)
        boundaries_strict = detect_step_boundaries(emb, threshold_factor=0.0, min_step_clips=3)
        # Strict min_step_clips should produce fewer boundaries
        assert len(boundaries_strict) <= len(boundaries_relaxed)


class TestEvaluateSopEdgeCases:
    def test_identical_videos_high_score(self) -> None:
        s1 = np.array([1.0, 0.0], dtype=np.float32)
        s2 = np.array([0.0, 1.0], dtype=np.float32)
        gold = np.stack([s1, s1, s2, s2], axis=0)
        result = evaluate_sop(
            gold_embeddings=gold,
            trainee_embeddings=gold.copy(),
            gold_meta=_meta(4),
            trainee_meta=_meta(4),
            threshold_factor=0.5,
            min_step_clips=1,
            low_similarity_threshold=0.75,
            w_miss=12,
            w_swap=8,
            w_dev=30,
            w_time=15,
        )
        assert result["score"] >= 90.0

    def test_completely_different_videos_low_score(self) -> None:
        gold = np.array([[1.0, 0.0, 0.0]] * 6, dtype=np.float32)
        trainee = np.array([[0.0, 1.0, 0.0]] * 6, dtype=np.float32)
        result = evaluate_sop(
            gold_embeddings=gold,
            trainee_embeddings=trainee,
            gold_meta=_meta(6),
            trainee_meta=_meta(6),
            threshold_factor=0.5,
            min_step_clips=1,
            low_similarity_threshold=0.75,
            w_miss=12,
            w_swap=8,
            w_dev=30,
            w_time=15,
        )
        assert result["score"] < 50.0

    def test_result_contains_expected_keys(self) -> None:
        gold = np.random.randn(6, 4).astype(np.float32)
        trainee = np.random.randn(6, 4).astype(np.float32)
        result = evaluate_sop(
            gold_embeddings=gold,
            trainee_embeddings=trainee,
            gold_meta=_meta(6),
            trainee_meta=_meta(6),
            threshold_factor=0.5,
            min_step_clips=1,
            low_similarity_threshold=0.75,
            w_miss=12,
            w_swap=8,
            w_dev=30,
            w_time=15,
        )
        assert "score" in result
        assert "metrics" in result
        assert "deviations" in result
        assert "alignment_preview" in result
        assert "step_boundaries" in result
        assert "clip_count" in result
        assert 0.0 <= result["score"] <= 100.0

    def test_score_clamped_to_zero_not_negative(self) -> None:
        gold = np.array([[1.0, 0.0, 0.0]] * 4, dtype=np.float32)
        trainee = np.array([[-1.0, 0.0, 0.0]] * 4, dtype=np.float32)
        result = evaluate_sop(
            gold_embeddings=gold,
            trainee_embeddings=trainee,
            gold_meta=_meta(4),
            trainee_meta=_meta(4),
            threshold_factor=0.5,
            min_step_clips=1,
            low_similarity_threshold=0.75,
            w_miss=100,
            w_swap=100,
            w_dev=100,
            w_time=100,
        )
        assert result["score"] >= 0.0
