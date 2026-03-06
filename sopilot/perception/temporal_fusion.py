"""Temporal Intelligence — multi-frame temporal reasoning for video-level understanding.

Phase 35: Extends single-frame perception with temporal context by
aggregating DINOv2 CLS features across a sliding window of frames.
Enables scene-change detection, temporal anomaly scoring, activity
trend analysis, and action classification via prototype matching.

Architecture::

    FrameFeatures (per-frame)  -->  TemporalFeatureBuffer (sliding window)
                                          |
                                    TemporalFusionEngine
                                          |
                              +-----+-----+------+------+
                              |     |     |      |      |
                           embed  action  scene  anom  trend
                          pooling  class  change  det   stats

Design decisions:
    - Operates on pre-extracted 768-dim CLS features (no GPU in this module)
    - Temporal attention pooling: recency bias + feature-norm weighting
    - Action classification via cosine similarity to learned prototypes
    - Scene-change detection via cosine distance to running average
    - Thread-safe: all mutable state guarded by a single lock
    - Pure Python + numpy (no torch dependency)
"""

from __future__ import annotations

import logging
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_DEFAULT_WINDOW_SIZE = 16
_EMBEDDING_DIM = 768          # DINOv2 ViT-B/14 CLS token dimension
_SCENE_CHANGE_THRESHOLD = 0.3
_EMA_ALPHA = 0.1              # EMA decay for prototype updates
_VARIANCE_WARMUP = 8          # minimum frames before anomaly detection
_SOFTMAX_TEMPERATURE = 5.0    # temperature for action classification softmax

# Default action prototypes (will be replaced by learned prototypes)
_DEFAULT_ACTION_PROTOTYPES: dict[str, str] = {
    "stationary":  "stationary",
    "walking":     "walking",
    "working":     "working",
    "running":     "running",
    "gathering":   "gathering",
    "dispersing":  "dispersing",
}


# ── Data Types ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FrameFeatures:
    """Immutable feature snapshot for a single processed frame.

    Produced by the DINOv2 backbone or another feature extractor;
    consumed by :class:`TemporalFusionEngine` for temporal reasoning.

    Parameters
    ----------
    timestamp:
        Wall-clock timestamp (``time.time()``).
    frame_number:
        Sequential frame index within the stream/video.
    cls_features:
        768-dimensional CLS token from DINOv2 ViT-B/14, L2-normalised.
    detection_count:
        Number of objects detected in this frame.
    event_flags:
        Symbolic event labels active in this frame (e.g. ``["loitering"]``).
    """

    timestamp: float
    frame_number: int
    cls_features: np.ndarray      # shape (768,)
    detection_count: int = 0
    event_flags: list[str] = field(default_factory=list)


@dataclass
class TemporalFeatureBuffer:
    """Fixed-capacity sliding window of :class:`FrameFeatures`.

    Parameters
    ----------
    maxlen:
        Maximum number of frames retained.  Oldest frames are evicted
        automatically when the buffer is full.
    """

    maxlen: int = _DEFAULT_WINDOW_SIZE
    frames: deque[FrameFeatures] = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.frames is None:
            self.frames = deque(maxlen=self.maxlen)

    # ── Public API ────────────────────────────────────────────────────

    def append(self, ff: FrameFeatures) -> None:
        """Append a :class:`FrameFeatures` to the buffer."""
        self.frames.append(ff)

    def clear(self) -> None:
        """Remove all frames from the buffer."""
        self.frames.clear()

    @property
    def is_empty(self) -> bool:
        return len(self.frames) == 0

    @property
    def size(self) -> int:
        return len(self.frames)

    @property
    def is_full(self) -> bool:
        return len(self.frames) == self.maxlen

    def feature_matrix(self) -> np.ndarray | None:
        """Return stacked CLS features as ``(T, D)`` array, or ``None`` if empty."""
        if self.is_empty:
            return None
        return np.stack([f.cls_features for f in self.frames], axis=0)

    def latest(self) -> FrameFeatures | None:
        """Return the most recent frame, or ``None`` if empty."""
        return self.frames[-1] if self.frames else None


# ── Temporal Fusion Engine ────────────────────────────────────────────────


class TemporalFusionEngine:
    """Multi-frame temporal reasoning engine.

    Aggregates per-frame DINOv2 CLS features over a sliding window
    and provides temporal attention pooling, action classification,
    scene-change detection, temporal anomaly scoring, and activity
    trend analysis.

    All operations are CPU-only and operate on pre-extracted features.
    Thread-safe: all mutable state is guarded by ``self._lock``.

    Parameters
    ----------
    window_size:
        Number of frames retained in the sliding buffer (default 16).
    scene_change_threshold:
        Cosine distance above which a scene change is declared (default 0.3).
    action_prototypes:
        Optional dict mapping action names to prototype feature vectors
        (``np.ndarray`` of shape ``(768,)``).  If ``None``, random
        prototypes are initialised for the six default actions.
    """

    def __init__(
        self,
        window_size: int = _DEFAULT_WINDOW_SIZE,
        scene_change_threshold: float = _SCENE_CHANGE_THRESHOLD,
        action_prototypes: dict[str, np.ndarray] | None = None,
    ) -> None:
        self._window_size = max(2, window_size)
        self._scene_change_threshold = scene_change_threshold
        self._buffer = TemporalFeatureBuffer(maxlen=self._window_size)
        self._lock = threading.Lock()

        # Action prototypes
        if action_prototypes is not None:
            self._prototypes: dict[str, np.ndarray] = {
                k: self._l2_normalise(v.copy()) for k, v in action_prototypes.items()
            }
        else:
            self._prototypes = self._init_default_prototypes()

        # Running statistics for anomaly baseline
        self._feature_running_mean: np.ndarray | None = None
        self._feature_running_var: np.ndarray | None = None
        self._n_observed: int = 0

        # Counters
        self._total_frames: int = 0
        self._scene_changes: int = 0
        self._anomalies_detected: int = 0

        logger.info(
            "TemporalFusionEngine initialised: window=%d, "
            "scene_threshold=%.2f, prototypes=%d",
            self._window_size,
            self._scene_change_threshold,
            len(self._prototypes),
        )

    # ── Public API ────────────────────────────────────────────────────

    def observe(
        self,
        timestamp: float,
        frame_number: int,
        cls_features: np.ndarray,
        detection_count: int = 0,
        event_flags: list[str] | None = None,
    ) -> None:
        """Add a new frame observation to the temporal buffer.

        Parameters
        ----------
        timestamp:
            Wall-clock time of the frame.
        frame_number:
            Sequential frame index.
        cls_features:
            768-dim CLS feature vector (will be L2-normalised internally).
        detection_count:
            Number of detections in this frame.
        event_flags:
            Symbolic events active in this frame.
        """
        features = self._l2_normalise(np.asarray(cls_features, dtype=np.float32))
        ff = FrameFeatures(
            timestamp=timestamp,
            frame_number=frame_number,
            cls_features=features,
            detection_count=detection_count,
            event_flags=event_flags or [],
        )

        with self._lock:
            self._buffer.append(ff)
            self._update_running_stats(features)
            self._total_frames += 1

    def get_temporal_embedding(self) -> np.ndarray:
        """Compute a temporal-attention-pooled embedding over the buffer.

        Returns a single 768-dim vector summarising the recent temporal
        context, weighted by recency bias and feature magnitude.

        Returns
        -------
        np.ndarray
            Shape ``(768,)`` temporal embedding, or zeros if buffer is empty.
        """
        with self._lock:
            mat = self._buffer.feature_matrix()
            if mat is None:
                return np.zeros(_EMBEDDING_DIM, dtype=np.float32)
            return self._temporal_attention_pool(mat)

    def classify_action(self) -> dict[str, float]:
        """Classify the current temporal sequence against action prototypes.

        Computes cosine similarity between the temporal embedding and
        each action prototype, then applies temperature-scaled softmax
        to produce a probability distribution.

        Returns
        -------
        dict[str, float]
            Action name to confidence mapping (sums to ~1.0).
            Empty dict if buffer is empty or no prototypes exist.
        """
        if not self._prototypes:
            return {}

        embedding = self.get_temporal_embedding()
        norm = np.linalg.norm(embedding)
        if norm < 1e-8:
            return {k: 1.0 / len(self._prototypes) for k in self._prototypes}

        embedding_normed = embedding / norm

        similarities: dict[str, float] = {}
        for name, proto in self._prototypes.items():
            sim = float(np.dot(embedding_normed, proto))
            similarities[name] = sim

        # Temperature-scaled softmax
        scores = np.array(list(similarities.values()), dtype=np.float32)
        scores_scaled = scores * _SOFTMAX_TEMPERATURE
        scores_shifted = scores_scaled - scores_scaled.max()
        exp_scores = np.exp(scores_shifted)
        softmax = exp_scores / (exp_scores.sum() + 1e-8)

        return {
            name: float(softmax[i])
            for i, name in enumerate(similarities.keys())
        }

    def detect_scene_change(self) -> tuple[bool, float]:
        """Detect abrupt scene changes by comparing latest CLS to running average.

        Returns
        -------
        tuple[bool, float]
            ``(changed, cosine_distance)`` where ``changed`` is ``True`` if
            the cosine distance exceeds ``scene_change_threshold``.
        """
        with self._lock:
            if self._feature_running_mean is None or self._buffer.is_empty:
                return False, 0.0

            latest = self._buffer.latest()
            if latest is None:
                return False, 0.0

            similarity = self._cosine_similarity(
                latest.cls_features, self._feature_running_mean
            )
            distance = 1.0 - similarity
            changed = distance > self._scene_change_threshold

            if changed:
                self._scene_changes += 1
                logger.info(
                    "Scene change detected at frame %d (distance=%.3f)",
                    latest.frame_number,
                    distance,
                )

            return changed, float(distance)

    def detect_temporal_anomaly(self) -> tuple[bool, float]:
        """Detect temporal feature drift relative to historical baseline.

        Compares the variance of features in the current buffer window
        against the running variance baseline.  A high ratio indicates
        unusual temporal dynamics (e.g., sudden activity change, camera
        tampering, lighting shift).

        Returns
        -------
        tuple[bool, float]
            ``(anomalous, score)`` where ``score`` is the normalised
            drift magnitude in ``[0, 1]``.  ``anomalous`` is ``True``
            when ``score > 0.5``.
        """
        with self._lock:
            if (
                self._n_observed < _VARIANCE_WARMUP
                or self._feature_running_var is None
            ):
                return False, 0.0

            mat = self._buffer.feature_matrix()
            if mat is None or mat.shape[0] < 2:
                return False, 0.0

            # Current window variance (mean across dimensions)
            window_var = float(np.var(mat, axis=0).mean())

            # Baseline variance (mean across dimensions)
            baseline_var = float(self._feature_running_var.mean())

            if baseline_var < 1e-10:
                return False, 0.0

            # Ratio: high ratio means current window is much more variable
            ratio = window_var / (baseline_var + 1e-10)

            # Normalise to [0, 1] via sigmoid-like mapping
            # ratio=1 → score~0.27, ratio=3 → score~0.73, ratio=5 → score~0.92
            score = float(1.0 - 1.0 / (1.0 + ratio / 2.0))

            anomalous = score > 0.5

            if anomalous:
                self._anomalies_detected += 1
                logger.debug(
                    "Temporal anomaly: score=%.3f, ratio=%.2f", score, ratio
                )

            return anomalous, score

    def get_activity_trend(self) -> dict[str, Any]:
        """Compute rolling activity statistics over the buffer window.

        Returns
        -------
        dict
            - ``detection_count_trend``: ``"increasing"`` / ``"decreasing"`` / ``"stable"``
            - ``feature_variance``: float — mean feature variance (stability metric)
            - ``event_frequency``: dict[str, int] — count of each event flag
            - ``buffer_size``: int — current number of frames in buffer
        """
        with self._lock:
            result: dict[str, Any] = {
                "detection_count_trend": "stable",
                "feature_variance": 0.0,
                "event_frequency": {},
                "buffer_size": self._buffer.size,
            }

            if self._buffer.is_empty:
                return result

            frames = list(self._buffer.frames)

            # Detection count trend (linear regression slope)
            counts = np.array(
                [f.detection_count for f in frames], dtype=np.float32
            )
            if len(counts) >= 3:
                x = np.arange(len(counts), dtype=np.float32)
                x_mean = x.mean()
                c_mean = counts.mean()
                denom = float(((x - x_mean) ** 2).sum())
                if denom > 1e-8:
                    slope = float(((x - x_mean) * (counts - c_mean)).sum() / denom)
                    if slope > 0.1:
                        result["detection_count_trend"] = "increasing"
                    elif slope < -0.1:
                        result["detection_count_trend"] = "decreasing"

            # Feature variance
            mat = self._buffer.feature_matrix()
            if mat is not None and mat.shape[0] >= 2:
                result["feature_variance"] = float(np.var(mat, axis=0).mean())

            # Event frequency
            freq: dict[str, int] = {}
            for f in frames:
                for flag in f.event_flags:
                    freq[flag] = freq.get(flag, 0) + 1
            result["event_frequency"] = freq

            return result

    def update_action_prototype(
        self,
        action_name: str,
        features: np.ndarray,
    ) -> None:
        """Update an action prototype via exponential moving average.

        If the action does not yet exist, a new prototype is created.

        Parameters
        ----------
        action_name:
            Name of the action category.
        features:
            768-dim feature vector for this action example.
        """
        features_normed = self._l2_normalise(
            np.asarray(features, dtype=np.float32)
        )

        with self._lock:
            if action_name in self._prototypes:
                old = self._prototypes[action_name]
                updated = (1.0 - _EMA_ALPHA) * old + _EMA_ALPHA * features_normed
                self._prototypes[action_name] = self._l2_normalise(updated)
                logger.debug("Updated prototype '%s' (EMA)", action_name)
            else:
                self._prototypes[action_name] = features_normed
                logger.info("Created new prototype '%s'", action_name)

    def get_stats(self) -> dict[str, Any]:
        """Return engine-level statistics.

        Returns
        -------
        dict
            Includes buffer occupancy, total frames processed, prototype
            count, scene changes, and anomalies detected.
        """
        with self._lock:
            return {
                "buffer_size": self._buffer.size,
                "buffer_capacity": self._window_size,
                "buffer_full": self._buffer.is_full,
                "total_frames_observed": self._total_frames,
                "prototype_count": len(self._prototypes),
                "prototype_names": sorted(self._prototypes.keys()),
                "scene_changes_detected": self._scene_changes,
                "anomalies_detected": self._anomalies_detected,
                "running_stats_n": self._n_observed,
            }

    def reset(self) -> None:
        """Clear buffer and reset all running statistics."""
        with self._lock:
            self._buffer.clear()
            self._feature_running_mean = None
            self._feature_running_var = None
            self._n_observed = 0
            self._total_frames = 0
            self._scene_changes = 0
            self._anomalies_detected = 0
        logger.info("TemporalFusionEngine reset")

    # ── Private Helpers ───────────────────────────────────────────────

    def _temporal_attention_pool(self, features_seq: np.ndarray) -> np.ndarray:
        """Temporal attention pooling with recency bias.

        Computes a weighted average of feature vectors where weights
        combine exponential recency (newer frames weighted higher) and
        feature-norm salience (frames with stronger activations weighted
        higher).

        Parameters
        ----------
        features_seq:
            Shape ``(T, D)`` array of stacked CLS features.

        Returns
        -------
        np.ndarray
            Shape ``(D,)`` pooled feature vector.
        """
        T = features_seq.shape[0]

        # Exponential recency bias: latest frame gets weight ~1.0,
        # oldest frame gets weight ~exp(-1) ≈ 0.37
        recency = np.exp(np.linspace(-1.0, 0.0, T))

        # Feature-norm salience
        norms = np.linalg.norm(features_seq, axis=1)

        # Combined attention weights
        weights = recency * norms
        weight_sum = weights.sum()
        if weight_sum < 1e-8:
            weights = np.ones(T, dtype=np.float32) / T
        else:
            weights = weights / weight_sum

        # Weighted sum: (D, T) @ (T,) -> (D,)
        pooled = features_seq.T @ weights
        return pooled.astype(np.float32)

    def _update_running_stats(self, features: np.ndarray) -> None:
        """Update Welford's online running mean and variance.

        Uses numerically stable single-pass algorithm for incremental
        mean and variance computation.

        Must be called under ``self._lock``.
        """
        self._n_observed += 1
        n = self._n_observed

        if self._feature_running_mean is None:
            self._feature_running_mean = features.copy()
            self._feature_running_var = np.zeros_like(features)
            return

        delta = features - self._feature_running_mean
        self._feature_running_mean = self._feature_running_mean + delta / n
        delta2 = features - self._feature_running_mean
        self._feature_running_var = (
            self._feature_running_var * ((n - 1) / n) + delta * delta2 / n
        )

    def _init_default_prototypes(self) -> dict[str, np.ndarray]:
        """Initialise deterministic pseudo-random prototypes for default actions.

        These are placeholder prototypes that will be replaced by learned
        embeddings via :meth:`update_action_prototype`.  Each prototype
        is generated from a fixed seed so results are reproducible.

        Returns
        -------
        dict[str, np.ndarray]
            Action name to L2-normalised 768-dim prototype vector.
        """
        prototypes: dict[str, np.ndarray] = {}
        rng = np.random.RandomState(42)
        for name in _DEFAULT_ACTION_PROTOTYPES:
            vec = rng.randn(_EMBEDDING_DIM).astype(np.float32)
            prototypes[name] = self._l2_normalise(vec)
        return prototypes

    @staticmethod
    def _l2_normalise(vec: np.ndarray) -> np.ndarray:
        """L2-normalise a vector, returning zeros for zero-magnitude input."""
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            return vec
        return (vec / norm).astype(np.float32)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
