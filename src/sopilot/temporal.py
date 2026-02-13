"""Shared temporal utilities for VIGIL-RAG and SOPilot.

Centralizes temporal IoU computation and temporal-overlap deduplication
used by rag_service, event_detection_service, and vigil_metrics.
"""

from __future__ import annotations


def temporal_iou(s1: float, e1: float, s2: float, e2: float) -> float:
    """Compute temporal Intersection over Union between two intervals.

    Args:
        s1, e1: Start and end of the first interval.
        s2, e2: Start and end of the second interval.

    Returns:
        IoU in [0, 1].
    """
    inter_start = max(s1, s2)
    inter_end = min(e1, e2)
    intersection = max(0.0, inter_end - inter_start)
    union = (e1 - s1) + (e2 - s2) - intersection
    if union <= 0:
        return 0.0
    return intersection / union
