from __future__ import annotations

import numpy as np

from sopilot.step_engine import detect_step_boundaries, evaluate_sop


def _meta(n: int) -> list[dict]:
    return [{"clip_idx": i, "start_sec": float(i * 4), "end_sec": float((i + 1) * 4)} for i in range(n)]


def test_detect_step_boundaries_basic_transition() -> None:
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    emb = np.stack([a, a, a, b, b, b], axis=0)

    boundaries = detect_step_boundaries(embeddings=emb, threshold_factor=0.5, min_step_clips=1)
    assert boundaries[0] == 0
    assert boundaries[-1] == emb.shape[0]
    assert any(2 <= x <= 4 for x in boundaries[1:-1])


def test_evaluate_sop_high_score_when_aligned() -> None:
    s1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    s2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    s3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    gold = np.stack([s1, s1, s2, s2, s3, s3], axis=0)
    trainee = np.stack([s1, s1, s1, s2, s2, s3, s3], axis=0)

    result = evaluate_sop(
        gold_embeddings=gold,
        trainee_embeddings=trainee,
        gold_meta=_meta(gold.shape[0]),
        trainee_meta=_meta(trainee.shape[0]),
        threshold_factor=0.5,
        min_step_clips=1,
        low_similarity_threshold=0.75,
        w_miss=12,
        w_swap=8,
        w_dev=30,
        w_time=15,
    )

    assert result["score"] >= 85.0
    assert result["metrics"]["miss"] == 0


def test_evaluate_sop_lower_score_for_missing_and_swapped_steps() -> None:
    s1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    s2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    s3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    gold = np.stack([s1, s1, s2, s2, s3, s3], axis=0)
    trainee_bad = np.stack([s1, s1, s3, s3], axis=0)

    result_bad = evaluate_sop(
        gold_embeddings=gold,
        trainee_embeddings=trainee_bad,
        gold_meta=_meta(gold.shape[0]),
        trainee_meta=_meta(trainee_bad.shape[0]),
        threshold_factor=0.5,
        min_step_clips=1,
        low_similarity_threshold=0.75,
        w_miss=12,
        w_swap=8,
        w_dev=30,
        w_time=15,
    )

    assert result_bad["score"] < 85.0
    assert result_bad["metrics"]["miss"] >= 1


def test_evaluate_sop_penalizes_local_deviation_even_when_order_is_preserved() -> None:
    s1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    s2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    s3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    gold = np.stack([s1, s1, s2, s2, s3, s3], axis=0)
    trainee = np.stack([s1, s1, s2, np.array([0.7, 0.3, 0.0], dtype=np.float32), s3, s3], axis=0)

    baseline = evaluate_sop(
        gold_embeddings=gold,
        trainee_embeddings=gold,
        gold_meta=_meta(gold.shape[0]),
        trainee_meta=_meta(gold.shape[0]),
        threshold_factor=0.5,
        min_step_clips=1,
        low_similarity_threshold=0.75,
        w_miss=12,
        w_swap=8,
        w_dev=30,
        w_time=15,
    )

    result = evaluate_sop(
        gold_embeddings=gold,
        trainee_embeddings=trainee,
        gold_meta=_meta(gold.shape[0]),
        trainee_meta=_meta(trainee.shape[0]),
        threshold_factor=0.5,
        min_step_clips=1,
        low_similarity_threshold=0.75,
        w_miss=12,
        w_swap=8,
        w_dev=30,
        w_time=15,
    )

    assert result["score"] < 95.0
    assert result["score"] < baseline["score"]
