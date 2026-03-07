"""Active Learning Loop Closure — operator labels → OnlineLearner update.

Phase 45: Closes the self-learning loop by connecting the ReviewQueue
(frames flagged for human review) to the OnlineLearner, so operator
labels immediately improve the model without retraining.

Architecture::

    PerceptionEngine
        → ReviewQueue._archive[frame_id] = {frame, features, score}
        → ActiveLearningLoop.get_pending_frames() → UI labeling panel
        → Operator submits label via POST /active-learning/label
        → ActiveLearningLoop.submit_label(frame_id, label, ...)
        → OnlineLearner.observe(features, label)    ← closes the loop
        → ReviewQueue.remove(frame_id)              ← clears the queue

Label flow::

    ReviewQueue    →  pending_frames (frame_id, score, snapshot)
    Operator UI    →  label = "helmet_violation" | "normal" | ...
    submit_label() →  OnlineLearner.observe(feature_vector, label)
                   →  LabelSubmission persisted to JSON
    get_stats()    →  labels_submitted, categories_learned, ...

Usage::

    loop = ActiveLearningLoop(
        review_queue=engine._review_queue,
        online_learner=engine._online_learner,
    )
    pending = loop.get_pending_frames(limit=10)
    for frame_info in pending:
        loop.submit_label(
            frame_id=frame_info["frame_id"],
            label="helmet_violation",
            feature_vector=frame_info["feature_vector"],
            confidence=0.95,
            operator_id="operator_01",
        )
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_MAX_LABELS = 1000   # max label submissions to persist


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class LabelSubmission:
    """A single operator label for a queued frame."""

    submission_id: str
    frame_id: str
    label: str
    confidence: float
    operator_id: str
    submitted_at: float
    feature_dim: int = 0
    previous_prediction: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "frame_id": self.frame_id,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "operator_id": self.operator_id,
            "submitted_at": self.submitted_at,
            "feature_dim": self.feature_dim,
            "previous_prediction": self.previous_prediction,
        }


# ---------------------------------------------------------------------------
# ActiveLearningLoop
# ---------------------------------------------------------------------------


class ActiveLearningLoop:
    """Bridges ReviewQueue → Operator labels → OnlineLearner updates.

    Parameters
    ----------
    review_queue:
        The PerceptionEngine's :class:`ReviewQueue` instance.
        If None, operates in standalone mode (labels still accepted).
    online_learner:
        The PerceptionEngine's :class:`OnlineLearner` instance.
        If None, labels are stored but not propagated.
    persist_path:
        Optional JSON path for label submission persistence.
    min_confidence:
        Minimum operator confidence to forward to OnlineLearner.
    """

    def __init__(
        self,
        review_queue: Any = None,
        online_learner: Any = None,
        persist_path: Path | None = None,
        min_confidence: float = 0.5,
    ) -> None:
        self._queue = review_queue
        self._learner = online_learner
        self._persist_path = persist_path
        self._min_confidence = min_confidence

        self._lock = threading.Lock()
        self._labels: dict[str, LabelSubmission] = {}    # frame_id → latest label
        self._submissions: list[LabelSubmission] = []    # chronological log
        self._label_counts: dict[str, int] = defaultdict(int)
        self._learner_updates = 0
        self._created_at = time.time()

        if persist_path:
            self._load()

        logger.info(
            "ActiveLearningLoop created (queue=%s, learner=%s, min_conf=%.2f)",
            "yes" if review_queue else "no",
            "yes" if online_learner else "no",
            min_confidence,
        )

    # ── Submit a label ─────────────────────────────────────────────────

    def submit_label(
        self,
        frame_id: str,
        label: str,
        feature_vector: np.ndarray | list[float] | None = None,
        confidence: float = 1.0,
        operator_id: str = "operator",
        previous_prediction: str = "",
    ) -> dict[str, Any]:
        """Submit an operator label for a queued frame.

        If confidence ≥ min_confidence and a feature vector is provided,
        the label is forwarded to the OnlineLearner immediately.

        Parameters
        ----------
        frame_id:
            UUID of the frame from ReviewQueue / get_pending_frames().
        label:
            Human-assigned label string (e.g., "helmet_violation", "normal").
        feature_vector:
            DINOv2 or other feature vector for OnlineLearner. If None,
            attempts to retrieve from ReviewQueue archive.
        confidence:
            Operator confidence [0, 1].
        operator_id:
            Operator identifier for audit trail.
        previous_prediction:
            Model's previous prediction (for disagreement tracking).

        Returns
        -------
        dict
            Keys: ``submission_id``, ``learner_updated``, ``label``, ``frame_id``.
        """
        # Try to get feature vector from queue archive if not provided
        feat = _to_numpy(feature_vector)
        if feat is None and self._queue is not None:
            feat = _get_queue_features(self._queue, frame_id)

        submission = LabelSubmission(
            submission_id=str(uuid.uuid4()),
            frame_id=frame_id,
            label=label,
            confidence=float(confidence),
            operator_id=operator_id,
            submitted_at=time.time(),
            feature_dim=len(feat) if feat is not None else 0,
            previous_prediction=previous_prediction,
        )

        learner_updated = False
        if (
            feat is not None
            and confidence >= self._min_confidence
            and self._learner is not None
        ):
            try:
                self._learner.observe(feat, label)
                learner_updated = True
            except Exception:
                logger.debug("ActiveLearningLoop: learner.observe failed", exc_info=True)

        with self._lock:
            self._labels[frame_id] = submission
            self._submissions.append(submission)
            # Rolling window
            if len(self._submissions) > _MAX_LABELS:
                old = self._submissions.pop(0)
                self._labels.pop(old.frame_id, None)
            self._label_counts[label] += 1
            if learner_updated:
                self._learner_updates += 1

        if self._persist_path:
            self._save()

        logger.info(
            "ActiveLearningLoop: label='%s' for frame %s (conf=%.2f, learner=%s)",
            label,
            frame_id[:8],
            confidence,
            "updated" if learner_updated else "skipped",
        )

        return {
            "submission_id": submission.submission_id,
            "frame_id": frame_id,
            "label": label,
            "learner_updated": learner_updated,
            "confidence": confidence,
        }

    # ── Query pending frames ───────────────────────────────────────────

    def get_pending_frames(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return frames awaiting operator labels from the ReviewQueue.

        Returns
        -------
        list[dict]
            Each dict: ``frame_id``, ``score``, ``timestamp``,
            ``already_labeled``, ``feature_vector`` (base64 or None).
        """
        if self._queue is None:
            return []

        items = _get_queue_items(self._queue, limit)
        with self._lock:
            labeled_ids = set(self._labels.keys())

        result = []
        for item in items:
            fid = item.get("frame_id", "")
            result.append({
                "frame_id": fid,
                "score": item.get("score", 0.0),
                "timestamp": item.get("timestamp", 0.0),
                "already_labeled": fid in labeled_ids,
                "existing_label": self._labels[fid].label if fid in labeled_ids else None,
            })
        return result

    # ── Stats ──────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "labels_submitted": len(self._submissions),
                "unique_frames_labeled": len(self._labels),
                "learner_updates": self._learner_updates,
                "label_distribution": dict(self._label_counts),
                "unique_labels": len(self._label_counts),
                "min_confidence": self._min_confidence,
                "queue_available": self._queue is not None,
                "learner_available": self._learner is not None,
                "uptime_seconds": round(time.time() - self._created_at, 1),
            }

    def get_recent_labels(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            recent = self._submissions[-limit:]
        return [s.to_dict() for s in reversed(recent)]

    # ── Persistence ────────────────────────────────────────────────────

    def _save(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {
                "submissions": [s.to_dict() for s in self._submissions[-_MAX_LABELS:]],
                "label_counts": dict(self._label_counts),
                "learner_updates": self._learner_updates,
            }
        try:
            with open(self._persist_path, "w", encoding="utf-8") as fp:
                json.dump(data, fp, ensure_ascii=False)
        except Exception:
            logger.debug("ActiveLearningLoop: save failed", exc_info=True)

    def _load(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            with self._lock:
                self._learner_updates = data.get("learner_updates", 0)
                for lc_key, count in data.get("label_counts", {}).items():
                    self._label_counts[lc_key] = count
                for s in data.get("submissions", []):
                    sub = LabelSubmission(
                        submission_id=s["submission_id"],
                        frame_id=s["frame_id"],
                        label=s["label"],
                        confidence=s["confidence"],
                        operator_id=s["operator_id"],
                        submitted_at=s["submitted_at"],
                        feature_dim=s.get("feature_dim", 0),
                        previous_prediction=s.get("previous_prediction", ""),
                    )
                    self._submissions.append(sub)
                    self._labels[sub.frame_id] = sub
            logger.info(
                "ActiveLearningLoop: loaded %d submissions from %s",
                len(self._submissions),
                self._persist_path,
            )
        except Exception:
            logger.debug("ActiveLearningLoop: load failed", exc_info=True)


# ---------------------------------------------------------------------------
# Helpers for ReviewQueue integration
# ---------------------------------------------------------------------------


def _to_numpy(v: Any) -> np.ndarray | None:
    if v is None:
        return None
    try:
        arr = np.asarray(v, dtype=np.float32)
        return arr if arr.ndim == 1 and len(arr) > 0 else None
    except Exception:
        return None


def _get_queue_features(queue: Any, frame_id: str) -> np.ndarray | None:
    """Attempt to retrieve feature vector from ReviewQueue archive."""
    try:
        archive = getattr(queue, "_archive", {})
        entry = archive.get(frame_id, {})
        feat = entry.get("features") or entry.get("feature_vector")
        return _to_numpy(feat)
    except Exception:
        return None


def _get_queue_items(queue: Any, limit: int) -> list[dict[str, Any]]:
    """Extract pending items from ReviewQueue."""
    items: list[dict[str, Any]] = []
    try:
        # ReviewQueue stores items in _queue (list or deque)
        raw = getattr(queue, "_queue", None) or []
        for entry in list(raw)[:limit]:
            if isinstance(entry, dict):
                items.append(entry)
            else:
                # May be an object with attributes
                items.append({
                    "frame_id": getattr(entry, "frame_id", str(uuid.uuid4())),
                    "score": getattr(entry, "score", 0.0),
                    "timestamp": getattr(entry, "timestamp", time.time()),
                })
    except Exception:
        logger.debug("ActiveLearningLoop: cannot read queue items", exc_info=True)
    return items
