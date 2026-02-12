"""Evaluation metrics for VIGIL-RAG.

This module implements evaluation metrics for:
1. RAG Quality: Evidence Recall@K, Answer Faithfulness, Context Precision
2. Event Detection: mAP, F1, Precision, Recall, FAH (False Alarms per Hour)
3. Retrieval: NDCG@K, MRR

Metrics are designed according to VIGIL-RAG evaluation framework (see EVALUATION.md).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvidenceRecallResult:
    """Result of Evidence Recall@K evaluation."""

    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    num_queries: int


@dataclass
class EventDetectionResult:
    """Result of event detection evaluation."""

    precision: float
    recall: float
    f1_score: float
    ap: float  # Average Precision
    fah: float  # False Alarms per Hour


@dataclass
class RetrievalResult:
    """Result of retrieval evaluation."""

    ndcg_at_5: float
    ndcg_at_10: float
    mrr: float  # Mean Reciprocal Rank
    num_queries: int


def evidence_recall_at_k(
    retrieved_clip_ids: list[list[str]],
    ground_truth_clip_ids: list[list[str]],
    k_values: list[int] | None = None,
) -> EvidenceRecallResult:
    """Compute Evidence Recall@K for RAG evaluation.

    Evidence Recall@K = (# relevant clips in top-K) / (# total relevant clips)

    Args:
        retrieved_clip_ids: List of retrieved clip ID lists (one per query)
        ground_truth_clip_ids: List of ground truth clip ID lists (one per query)
        k_values: K values to evaluate (default: [1, 3, 5, 10])

    Returns:
        EvidenceRecallResult with recall@K for each K

    Raises:
        ValueError: If input lists have different lengths
    """
    if len(retrieved_clip_ids) != len(ground_truth_clip_ids):
        raise ValueError(
            f"Mismatch: {len(retrieved_clip_ids)} retrieved vs {len(ground_truth_clip_ids)} ground truth"
        )

    if k_values is None:
        k_values = [1, 3, 5, 10]

    num_queries = len(retrieved_clip_ids)
    recall_sums = {k: 0.0 for k in k_values}

    for retrieved, ground_truth in zip(retrieved_clip_ids, ground_truth_clip_ids):
        if not ground_truth:
            # No ground truth clips, skip this query
            continue

        ground_truth_set = set(ground_truth)
        for k in k_values:
            retrieved_at_k = set(retrieved[:k])
            num_relevant_at_k = len(retrieved_at_k & ground_truth_set)
            recall_sums[k] += num_relevant_at_k / len(ground_truth_set)

    # Average over queries
    recall_at_k = {k: recall_sums[k] / num_queries if num_queries > 0 else 0.0 for k in k_values}

    return EvidenceRecallResult(
        recall_at_1=recall_at_k.get(1, 0.0),
        recall_at_3=recall_at_k.get(3, 0.0),
        recall_at_5=recall_at_k.get(5, 0.0),
        recall_at_10=recall_at_k.get(10, 0.0),
        num_queries=num_queries,
    )


def event_detection_metrics(
    predictions: list[dict],
    ground_truth: list[dict],
    *,
    iou_threshold: float = 0.5,
    video_duration_hours: float = 1.0,
) -> EventDetectionResult:
    """Compute event detection metrics (Precision, Recall, F1, AP, FAH).

    Each detection is a dict with:
    - 'start_sec': float
    - 'end_sec': float
    - 'event_type': str
    - 'confidence': float (for predictions only)

    Args:
        predictions: List of predicted events
        ground_truth: List of ground truth events
        iou_threshold: IoU threshold for matching (default 0.5)
        video_duration_hours: Video duration in hours (for FAH calculation)

    Returns:
        EventDetectionResult with P/R/F1/AP/FAH
    """
    if not ground_truth:
        # No ground truth, all predictions are false alarms
        fah = len(predictions) / video_duration_hours if video_duration_hours > 0 else 0.0
        return EventDetectionResult(
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            ap=0.0,
            fah=fah,
        )

    # Sort predictions by confidence (descending)
    sorted_predictions = sorted(predictions, key=lambda x: x.get("confidence", 0.0), reverse=True)

    # Track which ground truth events have been matched
    gt_matched = [False] * len(ground_truth)

    tp_at_k = []  # True positives at each prediction k
    fp_at_k = []  # False positives at each prediction k

    for pred in sorted_predictions:
        # Find best matching ground truth
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truth):
            if gt_matched[gt_idx]:
                continue  # Already matched

            # Check event type
            if pred.get("event_type") != gt.get("event_type"):
                continue

            # Compute IoU
            iou = _compute_temporal_iou(
                pred["start_sec"],
                pred["end_sec"],
                gt["start_sec"],
                gt["end_sec"],
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Check if match is good enough
        if best_iou >= iou_threshold:
            gt_matched[best_gt_idx] = True
            tp_at_k.append(1)
            fp_at_k.append(0)
        else:
            tp_at_k.append(0)
            fp_at_k.append(1)

    # Compute cumulative TP/FP
    cum_tp = np.cumsum(tp_at_k) if tp_at_k else np.array([0])
    cum_fp = np.cumsum(fp_at_k) if fp_at_k else np.array([0])

    # Precision at each k
    precisions = cum_tp / (cum_tp + cum_fp + 1e-9)

    # Recall at each k
    num_gt = len(ground_truth)
    recalls = cum_tp / num_gt

    # Average Precision (area under P-R curve)
    ap = _compute_ap(precisions, recalls)

    # Final precision/recall/F1
    if len(cum_tp) > 0:
        final_tp = cum_tp[-1]
        final_fp = cum_fp[-1]
        precision = final_tp / (final_tp + final_fp) if (final_tp + final_fp) > 0 else 0.0
        recall = final_tp / num_gt
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        precision = 0.0
        recall = 0.0
        f1_score = 0.0

    # False Alarms per Hour
    fah = final_fp / video_duration_hours if len(cum_fp) > 0 and video_duration_hours > 0 else 0.0

    return EventDetectionResult(
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1_score),
        ap=float(ap),
        fah=float(fah),
    )


def ndcg_at_k(
    retrieved_clip_ids: list[list[str]],
    relevance_scores: list[dict[str, float]],
    k: int = 5,
) -> float:
    """Compute Normalized Discounted Cumulative Gain @K.

    NDCG@K measures ranking quality with graded relevance.

    Args:
        retrieved_clip_ids: List of retrieved clip ID lists (one per query)
        relevance_scores: List of dicts mapping clip_id -> relevance score (0-1)
        k: K value for NDCG@K

    Returns:
        NDCG@K averaged over all queries
    """
    if not retrieved_clip_ids or not relevance_scores:
        return 0.0

    ndcg_scores = []

    for retrieved, relevance in zip(retrieved_clip_ids, relevance_scores):
        # Get relevance scores for retrieved clips
        retrieved_relevances = [relevance.get(clip_id, 0.0) for clip_id in retrieved[:k]]

        # Compute DCG@K
        dcg = _compute_dcg(retrieved_relevances)

        # Compute ideal DCG@K (sorted by relevance descending)
        ideal_relevances = sorted(relevance.values(), reverse=True)[:k]
        idcg = _compute_dcg(ideal_relevances)

        # NDCG = DCG / IDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def mrr(
    retrieved_clip_ids: list[list[str]],
    relevant_clip_ids: list[set[str]],
) -> float:
    """Compute Mean Reciprocal Rank.

    MRR = average of (1 / rank of first relevant item)

    Args:
        retrieved_clip_ids: List of retrieved clip ID lists (one per query)
        relevant_clip_ids: List of relevant clip ID sets (one per query)

    Returns:
        MRR score
    """
    if not retrieved_clip_ids or not relevant_clip_ids:
        return 0.0

    rr_scores = []

    for retrieved, relevant in zip(retrieved_clip_ids, relevant_clip_ids):
        # Find rank of first relevant item (1-indexed)
        for rank, clip_id in enumerate(retrieved, start=1):
            if clip_id in relevant:
                rr_scores.append(1.0 / rank)
                break
        else:
            # No relevant item found
            rr_scores.append(0.0)

    return float(np.mean(rr_scores)) if rr_scores else 0.0


# ===== Helper Functions =====


def _compute_temporal_iou(
    start1: float,
    end1: float,
    start2: float,
    end2: float,
) -> float:
    """Compute temporal Intersection over Union.

    Args:
        start1, end1: First interval
        start2, end2: Second interval

    Returns:
        IoU in [0, 1]
    """
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0.0, intersection_end - intersection_start)

    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union = union_end - union_start

    return intersection / union if union > 0 else 0.0


def _compute_dcg(relevances: list[float]) -> float:
    """Compute Discounted Cumulative Gain.

    DCG = sum_{i=1}^{k} (2^{rel_i} - 1) / log2(i + 1)

    Args:
        relevances: List of relevance scores

    Returns:
        DCG score
    """
    if not relevances:
        return 0.0

    dcg = 0.0
    for i, rel in enumerate(relevances, start=1):
        dcg += (2**rel - 1) / np.log2(i + 1)

    return dcg


def _compute_ap(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """Compute Average Precision (area under P-R curve).

    Args:
        precisions: Precision at each prediction
        recalls: Recall at each prediction

    Returns:
        Average Precision
    """
    if len(precisions) == 0 or len(recalls) == 0:
        return 0.0

    # Add sentinel values
    precisions = np.concatenate([[0.0], precisions, [0.0]])
    recalls = np.concatenate([[0.0], recalls, [1.0]])

    # Compute area using trapezoidal rule
    ap = 0.0
    for i in range(len(recalls) - 1):
        delta_recall = recalls[i + 1] - recalls[i]
        ap += precisions[i + 1] * delta_recall

    return float(ap)
