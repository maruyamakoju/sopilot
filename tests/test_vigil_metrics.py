"""Tests for VIGIL-RAG evaluation metrics."""

from __future__ import annotations

import pytest

from sopilot.evaluation.vigil_metrics import (
    _compute_ap,
    _compute_dcg,
    event_detection_metrics,
    evidence_recall_at_k,
    mrr,
    ndcg_at_k,
)
from sopilot.temporal import temporal_iou


class TestEvidenceRecallAtK:
    """Tests for Evidence Recall@K."""

    def test_perfect_recall(self):
        """Test perfect recall (all ground truth retrieved)."""
        retrieved = [["clip-1", "clip-2", "clip-3"]]
        ground_truth = [["clip-1", "clip-2", "clip-3"]]

        result = evidence_recall_at_k(retrieved, ground_truth, k_values=[1, 3])

        assert result.recall_at_1 == pytest.approx(1 / 3)  # Only 1 of 3 in top-1
        assert result.recall_at_3 == pytest.approx(1.0)  # All 3 in top-3
        assert result.num_queries == 1

    def test_partial_recall(self):
        """Test partial recall."""
        retrieved = [["clip-1", "clip-2", "clip-4"]]
        ground_truth = [["clip-1", "clip-2", "clip-3"]]  # clip-3 missing

        result = evidence_recall_at_k(retrieved, ground_truth, k_values=[3])

        assert result.recall_at_3 == pytest.approx(2 / 3)  # 2 out of 3

    def test_no_recall(self):
        """Test zero recall (no relevant clips retrieved)."""
        retrieved = [["clip-4", "clip-5", "clip-6"]]
        ground_truth = [["clip-1", "clip-2", "clip-3"]]

        result = evidence_recall_at_k(retrieved, ground_truth, k_values=[3])

        assert result.recall_at_3 == pytest.approx(0.0)

    def test_multiple_queries(self):
        """Test recall averaged over multiple queries."""
        retrieved = [
            ["clip-1", "clip-2", "clip-X"],
            ["clip-3", "clip-4", "clip-Y"],
        ]
        ground_truth = [
            ["clip-1", "clip-2"],  # 100% recall at k=3
            ["clip-3", "clip-5"],  # 50% recall at k=3 (clip-5 missing)
        ]

        result = evidence_recall_at_k(retrieved, ground_truth, k_values=[3])

        # Average: (1.0 + 0.5) / 2 = 0.75
        assert result.recall_at_3 == pytest.approx(0.75)
        assert result.num_queries == 2

    def test_empty_ground_truth(self):
        """Test with empty ground truth (skip query)."""
        retrieved = [["clip-1", "clip-2"], ["clip-3"]]
        ground_truth = [[], ["clip-3"]]  # First query has no ground truth

        result = evidence_recall_at_k(retrieved, ground_truth, k_values=[1])

        # Only second query counts: 1/1 = 1.0, but divided by 2 queries
        assert result.recall_at_1 == pytest.approx(0.5)

    def test_length_mismatch(self):
        """Test error on length mismatch."""
        retrieved = [["clip-1"]]
        ground_truth = [["clip-1"], ["clip-2"]]  # Mismatch

        with pytest.raises(ValueError, match="Mismatch"):
            evidence_recall_at_k(retrieved, ground_truth)


class TestEventDetectionMetrics:
    """Tests for event detection metrics."""

    def test_perfect_detection(self):
        """Test perfect detection (100% P/R/F1)."""
        predictions = [
            {"start_sec": 0.0, "end_sec": 5.0, "event_type": "intrusion", "confidence": 0.9},
        ]
        ground_truth = [
            {"start_sec": 0.0, "end_sec": 5.0, "event_type": "intrusion"},
        ]

        result = event_detection_metrics(predictions, ground_truth)

        assert result.precision == pytest.approx(1.0)
        assert result.recall == pytest.approx(1.0)
        assert result.f1_score == pytest.approx(1.0)
        assert result.ap == pytest.approx(1.0)
        assert result.fah == pytest.approx(0.0)  # No false alarms

    def test_false_alarm(self):
        """Test false alarm detection."""
        predictions = [
            {"start_sec": 10.0, "end_sec": 15.0, "event_type": "intrusion", "confidence": 0.8},
        ]
        ground_truth = []  # No ground truth

        result = event_detection_metrics(predictions, ground_truth, video_duration_hours=1.0)

        assert result.precision == pytest.approx(0.0)
        assert result.recall == pytest.approx(0.0)
        assert result.f1_score == pytest.approx(0.0)
        assert result.fah == pytest.approx(1.0)  # 1 false alarm in 1 hour

    def test_partial_overlap(self):
        """Test detection with partial temporal overlap."""
        predictions = [
            {"start_sec": 2.0, "end_sec": 7.0, "event_type": "intrusion", "confidence": 0.9},
        ]
        ground_truth = [
            {"start_sec": 5.0, "end_sec": 10.0, "event_type": "intrusion"},
        ]

        # IoU = overlap / union = 2.0 / 8.0 = 0.25 < 0.5 threshold
        result = event_detection_metrics(predictions, ground_truth, iou_threshold=0.5)

        # Should be false positive (IoU too low)
        assert result.precision == pytest.approx(0.0)
        assert result.recall == pytest.approx(0.0)

        # Lower threshold
        result_low_thresh = event_detection_metrics(predictions, ground_truth, iou_threshold=0.2)
        assert result_low_thresh.precision == pytest.approx(1.0)

    def test_wrong_event_type(self):
        """Test detection with wrong event type."""
        predictions = [
            {"start_sec": 0.0, "end_sec": 5.0, "event_type": "ppe_violation", "confidence": 0.9},
        ]
        ground_truth = [
            {"start_sec": 0.0, "end_sec": 5.0, "event_type": "intrusion"},
        ]

        result = event_detection_metrics(predictions, ground_truth)

        # Same time range but wrong type
        assert result.precision == pytest.approx(0.0)
        assert result.recall == pytest.approx(0.0)


class TestNDCGAtK:
    """Tests for NDCG@K."""

    def test_perfect_ranking(self):
        """Test perfect ranking (ideal order)."""
        retrieved = [["clip-1", "clip-2", "clip-3"]]
        relevance = [{"clip-1": 1.0, "clip-2": 0.8, "clip-3": 0.5}]

        ndcg = ndcg_at_k(retrieved, relevance, k=3)

        assert ndcg == pytest.approx(1.0)  # DCG == IDCG

    def test_reversed_ranking(self):
        """Test worst ranking (reversed order)."""
        retrieved = [["clip-3", "clip-2", "clip-1"]]
        relevance = [{"clip-1": 1.0, "clip-2": 0.8, "clip-3": 0.5}]

        ndcg = ndcg_at_k(retrieved, relevance, k=3)

        # NDCG < 1.0 (suboptimal order)
        assert ndcg < 1.0

    def test_partial_relevance(self):
        """Test with some irrelevant items."""
        retrieved = [["clip-1", "clip-4", "clip-2"]]  # clip-4 is irrelevant
        relevance = [{"clip-1": 1.0, "clip-2": 0.8, "clip-3": 0.5}]  # clip-4 not in relevance

        ndcg = ndcg_at_k(retrieved, relevance, k=3)

        # Should penalize irrelevant item in position 2
        assert 0.0 < ndcg < 1.0

    def test_empty_input(self):
        """Test with empty input."""
        assert ndcg_at_k([], [], k=5) == 0.0


class TestMRR:
    """Tests for Mean Reciprocal Rank."""

    def test_first_rank(self):
        """Test MRR when first item is relevant."""
        retrieved = [["clip-1", "clip-2", "clip-3"]]
        relevant = [{"clip-1"}]

        score = mrr(retrieved, relevant)

        assert score == pytest.approx(1.0)  # 1 / 1

    def test_third_rank(self):
        """Test MRR when third item is relevant."""
        retrieved = [["clip-1", "clip-2", "clip-3"]]
        relevant = [{"clip-3"}]

        score = mrr(retrieved, relevant)

        assert score == pytest.approx(1 / 3)

    def test_no_relevant(self):
        """Test MRR when no relevant item retrieved."""
        retrieved = [["clip-1", "clip-2", "clip-3"]]
        relevant = [{"clip-4"}]

        score = mrr(retrieved, relevant)

        assert score == pytest.approx(0.0)

    def test_multiple_queries(self):
        """Test MRR averaged over multiple queries."""
        retrieved = [
            ["clip-1", "clip-2"],
            ["clip-3", "clip-4", "clip-5"],
        ]
        relevant = [
            {"clip-1"},  # Rank 1: RR = 1.0
            {"clip-5"},  # Rank 3: RR = 1/3
        ]

        score = mrr(retrieved, relevant)

        # Average: (1.0 + 1/3) / 2 = 2/3
        assert score == pytest.approx(2 / 3)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_temporal_iou_perfect_overlap(self):
        """Test IoU with perfect overlap."""
        iou = temporal_iou(0.0, 10.0, 0.0, 10.0)
        assert iou == pytest.approx(1.0)

    def test_temporal_iou_partial_overlap(self):
        """Test IoU with partial overlap."""
        iou = temporal_iou(0.0, 10.0, 5.0, 15.0)
        # Intersection: 5.0, Union: 15.0
        assert iou == pytest.approx(5 / 15)

    def test_temporal_iou_no_overlap(self):
        """Test IoU with no overlap."""
        iou = temporal_iou(0.0, 5.0, 10.0, 15.0)
        assert iou == pytest.approx(0.0)

    def test_dcg_basic(self):
        """Test DCG computation."""
        relevances = [1.0, 0.8, 0.5]
        dcg = _compute_dcg(relevances)
        # DCG = (2^1 - 1)/log2(2) + (2^0.8 - 1)/log2(3) + (2^0.5 - 1)/log2(4)
        assert dcg > 0.0

    def test_dcg_empty(self):
        """Test DCG with empty input."""
        assert _compute_dcg([]) == 0.0

    def test_ap_perfect(self):
        """Test AP with perfect precision."""
        import numpy as np

        precisions = np.array([1.0, 1.0, 1.0])
        recalls = np.array([0.33, 0.66, 1.0])
        ap = _compute_ap(precisions, recalls)
        # Area under perfect P-R curve
        assert ap == pytest.approx(1.0)

    def test_ap_zero(self):
        """Test AP with zero precision."""
        import numpy as np

        precisions = np.array([0.0, 0.0, 0.0])
        recalls = np.array([0.0, 0.0, 0.0])
        ap = _compute_ap(precisions, recalls)
        assert ap == pytest.approx(0.0)
