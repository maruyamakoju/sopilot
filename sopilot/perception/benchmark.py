"""Perception Engine benchmark evaluation framework.

Provides evaluators for:
1. Object detection (IoU-based precision/recall)
2. Multi-object tracking (MOTA, IDF1 simplified)
3. Anomaly detection (wraps AnomalyDetectorEnsemble)

All evaluators use pure Python + stdlib (no sklearn/torch required).
They accept simple dict-based ground-truth/prediction formats for
easy integration with annotation tools.

Ground truth format for detection:
    List[{"frame_id": int, "detections": List[{"label": str, "bbox": [x,y,w,h]}]}]

Ground truth format for tracking (MOT-style):
    List[{"frame_id": int, "tracks": List[{"track_id": int, "bbox": [x,y,w,h]}]}]

All bboxes in [x_min, y_min, width, height] pixel format.
"""
from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DetectionMetrics:
    """Aggregate detection metrics across all frames and classes."""

    iou_threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float

    @classmethod
    def from_counts(
        cls,
        tp: int,
        fp: int,
        fn: int,
        iou_threshold: float = 0.5,
    ) -> "DetectionMetrics":
        """Construct from raw TP/FP/FN counts."""
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return cls(
            iou_threshold=iou_threshold,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=p,
            recall=r,
            f1=f1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "iou_threshold": self.iou_threshold,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class TrackingMetrics:
    """MOT-style tracking metrics (simplified MOTA + IDF1)."""

    mota: float          # Multi-Object Tracking Accuracy
    motp: float          # Multi-Object Tracking Precision (avg IoU of matched pairs)
    idf1: float          # ID F1 Score
    mostly_tracked: int  # GT tracks covered >= 80% of their lifetime
    mostly_lost: int     # GT tracks covered < 20% of their lifetime
    id_switches: int     # Track ID changes on the same GT object
    total_gt_tracks: int
    total_pred_tracks: int
    frames_evaluated: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mota": round(self.mota, 4),
            "motp": round(self.motp, 4),
            "idf1": round(self.idf1, 4),
            "mostly_tracked": self.mostly_tracked,
            "mostly_lost": self.mostly_lost,
            "id_switches": self.id_switches,
            "total_gt_tracks": self.total_gt_tracks,
            "total_pred_tracks": self.total_pred_tracks,
            "frames_evaluated": self.frames_evaluated,
        }


@dataclass
class AnomalyBenchmarkResult:
    """Results from the synthetic anomaly detection benchmark."""

    threshold_used: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    false_alarm_rate: float         # FP / (FP + TN)
    detection_latency_frames: float  # avg frames from anomaly start to detection
    n_normal: int
    n_anomaly: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold_used": self.threshold_used,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "false_alarm_rate": round(self.false_alarm_rate, 4),
            "detection_latency_frames": round(self.detection_latency_frames, 2),
            "n_normal": self.n_normal,
            "n_anomaly": self.n_anomaly,
        }


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------


def _iou_bbox(a: list, b: list) -> float:
    """Compute IoU between two bboxes in [x, y, w, h] format.

    Args:
        a: [x_min, y_min, width, height]
        b: [x_min, y_min, width, height]

    Returns:
        IoU value in [0, 1].
    """
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# DetectionEvaluator
# ---------------------------------------------------------------------------


class DetectionEvaluator:
    """Evaluate object detection performance against ground truth annotations.

    Handles multi-class, multi-frame evaluation with greedy IoU matching.

    Args:
        iou_threshold: Minimum IoU for a predicted box to count as a TP.
            Defaults to 0.5 (PASCAL VOC standard).
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        ground_truth: list[dict],
        predictions: list[dict],
    ) -> DetectionMetrics:
        """Compute TP/FP/FN across all frames using greedy IoU matching.

        Each predicted bbox is matched to at most one GT bbox (greedy,
        highest IoU first).  Unmatched predictions are FP; unmatched GT
        boxes are FN.

        Args:
            ground_truth: List of frame dicts:
                [{"frame_id": int, "detections": [{"label": str, "bbox": [x,y,w,h]}]}]
            predictions: Same structure.

        Returns:
            DetectionMetrics aggregated over all frames.
        """
        # Index predictions by frame_id for fast lookup
        pred_by_frame: dict[int, list[dict]] = {}
        for frame in predictions:
            fid = frame.get("frame_id", 0)
            pred_by_frame[fid] = list(frame.get("detections", []))

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for gt_frame in ground_truth:
            fid = gt_frame.get("frame_id", 0)
            gt_dets = list(gt_frame.get("detections", []))
            pred_dets = list(pred_by_frame.get(fid, []))

            tp, fp, fn = self._match_frame(gt_dets, pred_dets)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        # Count FP from prediction frames with no GT
        gt_frame_ids = {f.get("frame_id", 0) for f in ground_truth}
        for frame in predictions:
            fid = frame.get("frame_id", 0)
            if fid not in gt_frame_ids:
                total_fp += len(frame.get("detections", []))

        return DetectionMetrics.from_counts(total_tp, total_fp, total_fn, self.iou_threshold)

    def _match_frame(
        self,
        gt_dets: list[dict],
        pred_dets: list[dict],
    ) -> tuple[int, int, int]:
        """Greedily match predictions to GT in one frame.

        Returns:
            (tp, fp, fn)
        """
        if not gt_dets and not pred_dets:
            return 0, 0, 0
        if not gt_dets:
            return 0, len(pred_dets), 0
        if not pred_dets:
            return 0, 0, len(gt_dets)

        # Compute IoU matrix: rows = GT, cols = pred
        iou_matrix: list[list[float]] = []
        for gt in gt_dets:
            row = [_iou_bbox(gt["bbox"], p["bbox"]) for p in pred_dets]
            iou_matrix.append(row)

        matched_gt: set[int] = set()
        matched_pred: set[int] = set()

        # Collect all (iou, gt_idx, pred_idx) pairs above threshold, sort descending
        candidates: list[tuple[float, int, int]] = []
        for gi, row in enumerate(iou_matrix):
            for pi, iou in enumerate(row):
                if iou >= self.iou_threshold:
                    candidates.append((iou, gi, pi))
        candidates.sort(key=lambda x: x[0], reverse=True)

        for iou, gi, pi in candidates:
            if gi in matched_gt or pi in matched_pred:
                continue
            matched_gt.add(gi)
            matched_pred.add(pi)

        tp = len(matched_gt)
        fp = len(pred_dets) - len(matched_pred)
        fn = len(gt_dets) - len(matched_gt)
        return tp, fp, fn


# ---------------------------------------------------------------------------
# MOTEvaluator
# ---------------------------------------------------------------------------


class MOTEvaluator:
    """Simplified MOT evaluation.

    Metrics computed:
        MOTA  = 1 - (FP + FN + IDSW) / GT_total
        MOTP  = sum(IoU of matched pairs) / total_matches
        IDF1  = 2 * IDTP / (2 * IDTP + IDFP + IDFN)

    Where IDTP counts detections where the GT object is assigned to the
    correct pred track_id (longest-match assignment).

    Args:
        iou_threshold: IoU required to consider a prediction matched to a GT track.
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        ground_truth: list[dict],
        predictions: list[dict],
    ) -> TrackingMetrics:
        """Evaluate multi-object tracking performance.

        Args:
            ground_truth: List of frame dicts:
                [{"frame_id": int, "tracks": [{"track_id": int, "bbox": [x,y,w,h]}]}]
            predictions: Same structure.

        Returns:
            TrackingMetrics with MOTA, MOTP, IDF1, and auxiliary counts.
        """
        if not ground_truth and not predictions:
            return TrackingMetrics(
                mota=0.0, motp=0.0, idf1=0.0,
                mostly_tracked=0, mostly_lost=0, id_switches=0,
                total_gt_tracks=0, total_pred_tracks=0, frames_evaluated=0,
            )

        # Index predictions by frame_id
        pred_by_frame: dict[int, list[dict]] = {}
        for frame in predictions:
            fid = frame.get("frame_id", 0)
            pred_by_frame[fid] = list(frame.get("tracks", []))

        # Collect all unique track IDs
        all_gt_track_ids: set[int] = set()
        all_pred_track_ids: set[int] = set()
        for f in ground_truth:
            for t in f.get("tracks", []):
                all_gt_track_ids.add(t["track_id"])
        for f in predictions:
            for t in f.get("tracks", []):
                all_pred_track_ids.add(t["track_id"])

        # Per-GT-track statistics for MT/ML and IDF1
        # gt_track_id -> {pred_track_id: matched_count}
        gt_track_pred_counts: dict[int, dict[int, int]] = {
            gt_id: {} for gt_id in all_gt_track_ids
        }
        # Total GT detections per GT track
        gt_track_total: dict[int, int] = {gt_id: 0 for gt_id in all_gt_track_ids}

        total_fp = 0
        total_fn = 0
        total_idsw = 0
        total_matches = 0
        total_iou_sum = 0.0
        total_gt_dets = 0

        # Per-frame matching: gt_track_id -> last assigned pred_track_id
        last_pred_for_gt: dict[int, int] = {}

        frames_evaluated = 0

        for gt_frame in ground_truth:
            fid = gt_frame.get("frame_id", 0)
            gt_tracks = list(gt_frame.get("tracks", []))
            pred_tracks = list(pred_by_frame.get(fid, []))
            frames_evaluated += 1

            total_gt_dets += len(gt_tracks)
            for t in gt_tracks:
                gt_track_total[t["track_id"]] = gt_track_total.get(t["track_id"], 0) + 1

            tp, fp, fn, idsw, iou_sum, assignments = self._match_frame(
                gt_tracks, pred_tracks, last_pred_for_gt
            )
            total_fp += fp
            total_fn += fn
            total_idsw += idsw
            total_matches += tp
            total_iou_sum += iou_sum

            # Update per-GT-track pred assignment counts for IDF1
            for gt_id, pred_id in assignments.items():
                if gt_id not in gt_track_pred_counts:
                    gt_track_pred_counts[gt_id] = {}
                cnt = gt_track_pred_counts[gt_id]
                cnt[pred_id] = cnt.get(pred_id, 0) + 1

            last_pred_for_gt = {
                gt_id: pred_id for gt_id, pred_id in assignments.items()
            }

        # MOTA
        if total_gt_dets == 0:
            mota = 0.0
        else:
            mota = 1.0 - (total_fp + total_fn + total_idsw) / total_gt_dets

        # MOTP
        motp = total_iou_sum / max(1, total_matches)

        # IDF1: longest-match assignment
        # For each GT track, find the pred track with most co-detections
        idtp = 0
        for gt_id, pred_counts in gt_track_pred_counts.items():
            if pred_counts:
                best_pred_count = max(pred_counts.values())
                idtp += best_pred_count

        # IDFP: pred detections that are not correctly IDd
        # IDFN: GT detections not covered by the best-match pred
        total_pred_dets = sum(
            len(f.get("tracks", [])) for f in predictions
        )
        idfp = total_pred_dets - idtp
        idfn = total_gt_dets - idtp
        idf1_denom = 2 * idtp + idfp + idfn
        idf1 = (2 * idtp / idf1_denom) if idf1_denom > 0 else 0.0

        # Mostly tracked / mostly lost
        mostly_tracked = 0
        mostly_lost = 0
        for gt_id in all_gt_track_ids:
            total_for_track = gt_track_total.get(gt_id, 0)
            if total_for_track == 0:
                continue
            # Count how many frames this GT track was matched to any pred
            matched_count = sum(
                gt_track_pred_counts.get(gt_id, {}).values()
            )
            coverage = matched_count / total_for_track
            if coverage >= 0.8:
                mostly_tracked += 1
            elif coverage < 0.2:
                mostly_lost += 1

        return TrackingMetrics(
            mota=mota,
            motp=motp,
            idf1=idf1,
            mostly_tracked=mostly_tracked,
            mostly_lost=mostly_lost,
            id_switches=total_idsw,
            total_gt_tracks=len(all_gt_track_ids),
            total_pred_tracks=len(all_pred_track_ids),
            frames_evaluated=frames_evaluated,
        )

    def _match_frame(
        self,
        gt_tracks: list[dict],
        pred_tracks: list[dict],
        last_pred_for_gt: dict[int, int],
    ) -> tuple[int, int, int, int, float, dict[int, int]]:
        """Match predicted tracks to GT tracks in one frame.

        Returns:
            (tp, fp, fn, id_switches, iou_sum, assignments)
            where assignments: {gt_track_id -> pred_track_id}
        """
        if not gt_tracks and not pred_tracks:
            return 0, 0, 0, 0, 0.0, {}
        if not gt_tracks:
            return 0, len(pred_tracks), 0, 0, 0.0, {}
        if not pred_tracks:
            return 0, 0, len(gt_tracks), 0, 0.0, {}

        # Build IoU matrix
        iou_matrix: list[list[float]] = []
        for gt in gt_tracks:
            row = [_iou_bbox(gt["bbox"], p["bbox"]) for p in pred_tracks]
            iou_matrix.append(row)

        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        assignments: dict[int, int] = {}  # gt_track_id -> pred_track_id
        iou_sum = 0.0

        # Sort candidates by IoU descending for greedy matching
        candidates: list[tuple[float, int, int]] = []
        for gi, row in enumerate(iou_matrix):
            for pi, iou in enumerate(row):
                if iou >= self.iou_threshold:
                    candidates.append((iou, gi, pi))
        candidates.sort(key=lambda x: x[0], reverse=True)

        for iou, gi, pi in candidates:
            if gi in matched_gt or pi in matched_pred:
                continue
            matched_gt.add(gi)
            matched_pred.add(pi)
            iou_sum += iou
            gt_id = gt_tracks[gi]["track_id"]
            pred_id = pred_tracks[pi]["track_id"]
            assignments[gt_id] = pred_id

        tp = len(matched_gt)
        fp = len(pred_tracks) - len(matched_pred)
        fn = len(gt_tracks) - len(matched_gt)

        # Count ID switches
        idsw = 0
        for gt_id, pred_id in assignments.items():
            prev_pred_id = last_pred_for_gt.get(gt_id)
            if prev_pred_id is not None and prev_pred_id != pred_id:
                idsw += 1

        return tp, fp, fn, idsw, iou_sum, assignments


# ---------------------------------------------------------------------------
# AnomalyBenchmarkEvaluator
# ---------------------------------------------------------------------------


class AnomalyBenchmarkEvaluator:
    """Run a synthetic benchmark for the anomaly detection ensemble.

    Generates synthetic normal/anomaly world states using the
    AnomalyDetectorEnsemble and evaluates detection performance.

    The benchmark:
    1. Warm up the ensemble with 50 normal frames so it learns a baseline.
    2. Feed n_normal normal frames and n_anomaly anomaly frames in shuffled order.
    3. Compare predicted anomaly events (any event in a frame) vs ground truth label.
    4. Compute precision/recall/F1/accuracy/false-alarm-rate.

    Normal frames: 5 entities with slow speed (~0.01-0.05 units/frame).
    Anomaly frames: 5 entities with spike speed (0.3-0.8 units/frame).
    """

    def run_synthetic_benchmark(
        self,
        n_normal: int = 100,
        n_anomaly: int = 20,
        threshold: float = 2.5,
        seed: int = 42,
    ) -> AnomalyBenchmarkResult:
        """Run the synthetic benchmark.

        Args:
            n_normal:  Number of normal frames to evaluate (after warmup).
            n_anomaly: Number of anomaly frames to evaluate.
            threshold: Sigma threshold passed to AnomalyDetectorEnsemble config.
            seed:      Random seed for reproducibility.

        Returns:
            AnomalyBenchmarkResult with classification metrics.
        """
        try:
            from sopilot.perception.anomaly import AnomalyDetectorEnsemble
            from sopilot.perception.types import (
                BBox,
                EntityEvent,
                EntityEventType,
                PerceptionConfig,
                SceneEntity,
                SceneGraph,
                Track,
                TrackState,
                WorldState,
            )
        except ImportError as exc:
            raise RuntimeError(
                "sopilot.perception modules are required for AnomalyBenchmarkEvaluator"
            ) from exc

        rng = random.Random(seed)

        # Build PerceptionConfig with the requested threshold and short warmup
        config = PerceptionConfig(
            anomaly_enabled=True,
            anomaly_warmup_frames=50,
            anomaly_sigma_threshold=threshold,
            anomaly_cooldown_seconds=0.0,  # no cooldown during eval
            anomaly_ema_alpha=0.05,
            anomaly_spatial_grid_size=10,
        )

        ensemble = AnomalyDetectorEnsemble(config=config)

        def _make_world_state(frame_idx: int, speeds: list[float], ts: float) -> WorldState:
            """Build a minimal WorldState for the ensemble."""
            entities: list[SceneEntity] = []
            tracks: dict[int, Track] = {}
            for i, speed in enumerate(speeds):
                eid = i + 1
                # Vary positions slightly so the spatial detector sees normal occupancy
                x1 = rng.uniform(0.1, 0.7)
                y1 = rng.uniform(0.1, 0.7)
                x2 = min(x1 + 0.1, 1.0)
                y2 = min(y1 + 0.2, 1.0)
                bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
                entities.append(
                    SceneEntity(
                        entity_id=eid,
                        label="person",
                        bbox=bbox,
                        confidence=0.9,
                    )
                )
                tracks[eid] = Track(
                    track_id=eid,
                    label="person",
                    state=TrackState.ACTIVE,
                    bbox=bbox,
                    velocity=(speed, 0.0),
                    confidence=0.9,
                )

            sg = SceneGraph(
                timestamp=ts,
                frame_number=frame_idx,
                entities=entities,
                relations=[],
                frame_shape=(480, 640),
            )
            return WorldState(
                timestamp=ts,
                frame_number=frame_idx,
                scene_graph=sg,
                active_tracks=tracks,
                events=[],
                zone_occupancy={},
                entity_count=len(entities),
                person_count=len(entities),
            )

        def _normal_speeds() -> list[float]:
            return [rng.uniform(0.01, 0.05) for _ in range(5)]

        def _anomaly_speeds() -> list[float]:
            return [rng.uniform(0.3, 0.8) for _ in range(5)]

        # --- Phase 1: Warmup with 50 purely normal frames ---
        WARMUP = 50
        for wi in range(WARMUP):
            ws = _make_world_state(wi, _normal_speeds(), float(wi))
            ensemble.observe(ws)

        # --- Phase 2: Build evaluation dataset ---
        # Labels: 0 = normal, 1 = anomaly
        labels: list[int] = [0] * n_normal + [1] * n_anomaly
        frame_speeds: list[list[float]] = (
            [_normal_speeds() for _ in range(n_normal)]
            + [_anomaly_speeds() for _ in range(n_anomaly)]
        )

        # Shuffle together so anomaly frames are interleaved
        combined = list(zip(labels, frame_speeds))
        rng.shuffle(combined)
        labels, frame_speeds = zip(*combined) if combined else ([], [])
        labels = list(labels)
        frame_speeds = list(frame_speeds)

        # --- Phase 3: Evaluate ---
        # Track when each anomaly frame first appeared to compute latency
        # We consider consecutive anomaly frames as one "anomaly event"
        predictions: list[int] = []  # 0 or 1 per frame
        latencies: list[float] = []

        anomaly_event_start: int | None = None

        for fi, (label, speeds) in enumerate(zip(labels, frame_speeds)):
            ts = float(WARMUP + fi)
            ws = _make_world_state(WARMUP + fi, speeds, ts)

            # observe updates statistics
            ensemble.observe(ws)
            # check_anomalies uses the updated model
            events = ensemble.check_anomalies(ws)
            detected = 1 if events else 0
            predictions.append(detected)

            # Latency tracking
            if label == 1 and anomaly_event_start is None:
                anomaly_event_start = fi
            if label == 0:
                anomaly_event_start = None
            if detected == 1 and label == 1 and anomaly_event_start is not None:
                latency = fi - anomaly_event_start
                latencies.append(float(latency))
                anomaly_event_start = None  # reset for next event

        # --- Phase 4: Compute classification metrics ---
        tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
        tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        logger.info(
            "AnomalyBenchmark: TP=%d FP=%d FN=%d TN=%d "
            "P=%.3f R=%.3f F1=%.3f latency=%.1f frames",
            tp, fp, fn, tn, precision, recall, f1, avg_latency,
        )

        return AnomalyBenchmarkResult(
            threshold_used=threshold,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            false_alarm_rate=far,
            detection_latency_frames=avg_latency,
            n_normal=n_normal,
            n_anomaly=n_anomaly,
        )
