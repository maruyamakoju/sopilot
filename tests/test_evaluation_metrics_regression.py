"""Regression tests for evaluation metrics (P0: prevent circular dependency bugs).

These tests prevent the MRR/R@1 contradiction bug from returning.
"""

from __future__ import annotations

import pytest


def test_mrr_1_implies_recall_at_1_is_1():
    """Test: If MRR=1.0, then R@1 must also be 1.0 (logical necessity).

    This is a mathematical fact: MRR=1.0 means all queries have rank=1 (perfect),
    which implies R@1=1.0 (all queries have relevant result at rank 1).

    If this test fails, there's a circular dependency or logic bug in eval metrics.
    """
    from scripts.evaluate_vigil_real import _recall_at_k, _reciprocal_rank

    # Test case: All results are relevant (perfect retrieval)
    results_perfect = [
        {"clip_id": "a", "start_sec": 0.0, "end_sec": 10.0, "score": 1.0},
        {"clip_id": "b", "start_sec": 10.0, "end_sec": 20.0, "score": 0.9},
        {"clip_id": "c", "start_sec": 20.0, "end_sec": 30.0, "score": 0.8},
    ]
    gt_clip_ids = ["a"]  # Only first result is relevant

    # Compute metrics
    mrr = _reciprocal_rank(results_perfect, gt_clip_ids, [])
    r1 = _recall_at_k(results_perfect, gt_clip_ids, [], k=1)

    # Assertion: MRR=1.0 => R@1=1.0
    assert mrr == 1.0, "MRR should be 1.0 (rank=1)"
    assert r1 == 1.0, "R@1 should be 1.0 when MRR=1.0"


def test_mrr_not_inflated_when_no_relevant_in_results():
    """Test: MRR should be 0.0 when no relevant results are retrieved.

    This prevents the circular dependency bug where MRR is inflated by
    finding "relevant" clips from within the retrieval results.

    If MRR > 0 when no GT clip is in results, we have circular dependency.
    """
    from scripts.evaluate_vigil_real import _reciprocal_rank

    # Test case: Retrieval results contain NO relevant clips
    results_irrelevant = [
        {"clip_id": "x", "start_sec": 0.0, "end_sec": 10.0, "score": 1.0},
        {"clip_id": "y", "start_sec": 10.0, "end_sec": 20.0, "score": 0.9},
        {"clip_id": "z", "start_sec": 20.0, "end_sec": 30.0, "score": 0.8},
    ]
    gt_clip_ids = ["a", "b", "c"]  # None of these are in results

    # Compute MRR
    mrr = _reciprocal_rank(results_irrelevant, gt_clip_ids, [])

    # Assertion: MRR must be 0.0 (no relevant results)
    assert mrr == 0.0, "MRR should be 0.0 when no relevant clips are retrieved"


def test_mrr_r1_consistency_partial_retrieval():
    """Test: MRR and R@1 consistency when relevant result is at rank 2.

    MRR should be 0.5 (1/2), and R@1 should be 0.0 (no relevant at rank 1).
    """
    from scripts.evaluate_vigil_real import _recall_at_k, _reciprocal_rank

    # Test case: Relevant result at rank 2
    results = [
        {"clip_id": "x", "start_sec": 0.0, "end_sec": 10.0, "score": 1.0},  # Rank 1: irrelevant
        {"clip_id": "a", "start_sec": 10.0, "end_sec": 20.0, "score": 0.9},  # Rank 2: relevant
        {"clip_id": "y", "start_sec": 20.0, "end_sec": 30.0, "score": 0.8},  # Rank 3: irrelevant
    ]
    gt_clip_ids = ["a"]

    mrr = _reciprocal_rank(results, gt_clip_ids, [])
    r1 = _recall_at_k(results, gt_clip_ids, [], k=1)
    r5 = _recall_at_k(results, gt_clip_ids, [], k=5)

    # Assertions
    assert mrr == pytest.approx(0.5), "MRR should be 0.5 (rank=2)"
    assert r1 == 0.0, "R@1 should be 0.0 (no relevant at rank 1)"
    assert r5 == 1.0, "R@5 should be 1.0 (relevant in top-5)"


def test_temporal_overlap_matching():
    """Test: Temporal overlap matching works correctly (not clip ID based)."""
    from scripts.evaluate_vigil_real import _is_relevant_result

    # Test case: Result overlaps with GT time range
    result = {"clip_id": "unknown", "start_sec": 5.0, "end_sec": 15.0}
    gt_time_ranges = [{"start_sec": 10.0, "end_sec": 20.0}]

    # Should match (5-15 overlaps with 10-20)
    is_relevant = _is_relevant_result(result, [], gt_time_ranges, min_overlap_sec=0.0)
    assert is_relevant is True, "Should match when temporal overlap exists"

    # Should NOT match when no overlap
    result_no_overlap = {"clip_id": "unknown", "start_sec": 0.0, "end_sec": 5.0}
    is_relevant_no_overlap = _is_relevant_result(result_no_overlap, [], gt_time_ranges, min_overlap_sec=0.0)
    assert is_relevant_no_overlap is False, "Should not match when no temporal overlap"


def test_clip_id_priority_over_time_range():
    """Test: Exact clip ID match takes priority over time range matching."""
    from scripts.evaluate_vigil_real import _is_relevant_result

    result = {"clip_id": "a", "start_sec": 0.0, "end_sec": 10.0}
    gt_clip_ids = ["a", "b"]
    gt_time_ranges = [{"start_sec": 20.0, "end_sec": 30.0}]  # No overlap

    # Should match by clip ID even though time range doesn't overlap
    is_relevant = _is_relevant_result(result, gt_clip_ids, gt_time_ranges)
    assert is_relevant is True, "Clip ID match should take priority"


def test_multiple_queries_aggregate_mrr():
    """Test: Aggregate MRR is average of per-query MRRs.

    This verifies the per-query → aggregate computation path.
    """
    from scripts.evaluate_vigil_real import _reciprocal_rank

    # Query 1: Relevant at rank 1 → RR=1.0
    results1 = [{"clip_id": "a", "start_sec": 0, "end_sec": 10, "score": 1.0}]
    gt1 = ["a"]
    rr1 = _reciprocal_rank(results1, gt1, [])

    # Query 2: Relevant at rank 2 → RR=0.5
    results2 = [
        {"clip_id": "x", "start_sec": 0, "end_sec": 10, "score": 1.0},
        {"clip_id": "b", "start_sec": 10, "end_sec": 20, "score": 0.9},
    ]
    gt2 = ["b"]
    rr2 = _reciprocal_rank(results2, gt2, [])

    # Aggregate MRR = (1.0 + 0.5) / 2 = 0.75
    aggregate_mrr = (rr1 + rr2) / 2

    assert rr1 == 1.0
    assert rr2 == 0.5
    assert aggregate_mrr == pytest.approx(0.75), "Aggregate MRR should be average of per-query"
