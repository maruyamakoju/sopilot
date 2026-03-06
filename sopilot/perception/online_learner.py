"""Phase 36: On-Device Learning — デプロイメント適応型 Few-Shot 分類.

OnlineLearner は PerceptionEngine に対して、再学習なしに新しいカテゴリの学習と
デプロイ環境への適応を可能にする。事前抽出された 768 次元 DINOv2 特徴ベクトルに
基づき、k-NN 分類およびプロトタイプベース分類を提供する。

主要機能:
  - **Exemplar ストア**: オペレーター / 自動 / フィードバックによるサンプル蓄積
  - **k-NN 分類**: 距離重み付き投票による few-shot 分類
  - **プロトタイプ分類**: カテゴリ重心との cos 類似度による高速分類
  - **新規カテゴリ検出**: 既知プロトタイプとの距離に基づく novel detection
  - **フィードバック統合**: オペレーター確認による精度追跡
  - **JSON 永続化**: base64 エンコード float32 配列によるコンパクト保存

Architecture::

    OnlineLearner
      ├── add_exemplar(category, features)       — サンプル追加 + プロトタイプ更新
      ├── remove_exemplar(exemplar_id)            — サンプル削除 + 再計算
      ├── classify(features, k=5)                 — k-NN 距離重み付き投票
      ├── classify_with_prototypes(features)      — プロトタイプベース高速分類
      ├── detect_novel(features)                  — 新規カテゴリ検出
      ├── record_feedback(category, features, correct) — 精度統計更新
      ├── get_categories()                        — カテゴリ一覧 + 統計
      ├── save() / load()                         — JSON 永続化
      └── reset()                                 — 全クリア

Design decisions:
  - GPU 不要: 事前抽出済み 768 次元特徴ベクトルのみ使用
  - スレッドセーフ: RLock でエキゼンプラストアを保護
  - Best-effort 永続化: 例外は swallow + logger.exception (既存パターン踏襲)
  - EMA プロトタイプ更新: add_exemplar 時に即時反映
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FEATURE_DIM = 768       # DINOv2 output dimension
_VERSION = 1             # Persistence format version
_VALID_SOURCES = frozenset({"operator", "auto", "feedback"})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Exemplar:
    """A single labelled feature vector stored in the exemplar bank.

    Frozen to prevent accidental mutation after creation. The feature
    vector is a 768-dim L2-normalised embedding from DINOv2.

    Attributes:
        exemplar_id: Unique identifier (UUID4 hex string).
        category: Semantic category label (e.g. "helmet", "vest").
        features: 768-dim float32 numpy array (L2-normalised).
        confidence: Operator confidence when this exemplar was added [0, 1].
        added_at: Unix timestamp of creation.
        source: Origin — "operator" (manual), "auto" (engine), "feedback".
    """

    exemplar_id: str
    category: str
    features: np.ndarray
    confidence: float
    added_at: float
    source: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Exemplar):
            return NotImplemented
        return self.exemplar_id == other.exemplar_id

    def __hash__(self) -> int:
        return hash(self.exemplar_id)


@dataclass
class CategoryPrototype:
    """Running mean prototype for a category.

    The prototype vector is the L2-normalised centroid of all exemplars
    belonging to this category. It is updated incrementally via EMA when
    new exemplars are added, and fully recomputed when exemplars are removed.

    Attributes:
        category: Category label.
        prototype: 768-dim float32 running mean vector (L2-normalised).
        exemplar_count: Number of exemplars in this category.
        total_queries: Total classification queries that returned this category.
        correct_queries: Queries confirmed correct by operator feedback.
        last_updated: Unix timestamp of last prototype update.
    """

    category: str
    prototype: np.ndarray
    exemplar_count: int = 0
    total_queries: int = 0
    correct_queries: int = 0
    last_updated: float = field(default_factory=time.time)

    @property
    def accuracy(self) -> float:
        """Classification accuracy for this category [0, 1]."""
        if self.total_queries == 0:
            return 0.0
        return self.correct_queries / self.total_queries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _l2_normalise(v: np.ndarray) -> np.ndarray:
    """L2-normalise a vector (or return zero vector if norm is 0)."""
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v
    return v / norm


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two L2-normalised vectors.

    Returns value in [0, 2]. For normalised vectors this simplifies to
    ``1 - dot(a, b)``.
    """
    return float(1.0 - np.dot(a, b))


def _features_to_b64(features: np.ndarray) -> str:
    """Encode float32 array to base64 string for JSON persistence."""
    return base64.b64encode(features.astype(np.float32).tobytes()).decode("ascii")


def _b64_to_features(b64: str) -> np.ndarray:
    """Decode base64 string back to float32 numpy array."""
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# OnlineLearner
# ---------------------------------------------------------------------------


class OnlineLearner:
    """On-device few-shot classifier with exemplar-based learning.

    Enables the perception engine to learn new categories from a handful of
    labelled examples without GPU-based retraining. Operates entirely on
    pre-extracted 768-dim DINOv2 feature vectors.

    Thread-safety:
        All mutable state is protected by a ``threading.RLock``. Multiple
        concurrent classify / add / remove calls are safe.

    Persistence:
        Uses JSON with base64-encoded float32 arrays, consistent with
        SigmaTuner and AnomalyTuner patterns. Persistence is best-effort
        — exceptions are logged but never propagated.

    Args:
        distance_threshold: Maximum cosine distance for a match to be
            accepted. Lower values are stricter. Default 0.65.
        min_exemplars_for_prediction: Minimum exemplars per category
            before classification predictions are emitted. Default 3.
        ema_alpha: Exponential moving average coefficient for prototype
            updates. Higher values weight recent exemplars more. Default 0.1.
        persist_path: Path to JSON file for state persistence. ``None``
            disables persistence.

    Example::

        learner = OnlineLearner(persist_path=Path("data/online_learner.json"))
        eid = learner.add_exemplar("helmet", features_768d)
        result = learner.classify(query_features)
        if result:
            category, confidence = result
            print(f"Predicted: {category} ({confidence:.2f})")
    """

    def __init__(
        self,
        distance_threshold: float = 0.65,
        min_exemplars_for_prediction: int = 3,
        ema_alpha: float = 0.1,
        persist_path: Path | str | None = None,
    ) -> None:
        self._distance_threshold = distance_threshold
        self._min_exemplars = min_exemplars_for_prediction
        self._ema_alpha = ema_alpha
        self._persist_path = Path(persist_path) if persist_path else None

        # Core state
        self._exemplars: list[Exemplar] = []
        self._prototypes: dict[str, CategoryPrototype] = {}
        self._lock = threading.RLock()

        # Counters
        self._total_classifications: int = 0
        self._total_novel_detections: int = 0
        self._created_at: float = time.time()

        # Load persisted state if available
        self.load()

    # ------------------------------------------------------------------
    # Exemplar management
    # ------------------------------------------------------------------

    def add_exemplar(
        self,
        category: str,
        features: np.ndarray,
        confidence: float = 1.0,
        source: str = "operator",
    ) -> str:
        """Add a labelled exemplar and update the category prototype.

        The feature vector is L2-normalised before storage. If this is the
        first exemplar for the category, a new prototype is created. Otherwise
        the existing prototype is updated via exponential moving average.

        Args:
            category: Category label (e.g. "helmet", "fire_extinguisher").
            features: 768-dim feature vector (will be L2-normalised).
            confidence: Operator confidence in this label [0, 1].
            source: Origin of the exemplar ("operator", "auto", "feedback").

        Returns:
            The UUID hex string assigned to this exemplar.

        Raises:
            ValueError: If features has wrong dimensionality or source is
                not one of the valid values.
        """
        features = np.asarray(features, dtype=np.float32).ravel()
        if features.shape[0] != _FEATURE_DIM:
            raise ValueError(
                f"Expected {_FEATURE_DIM}-dim features, got {features.shape[0]}-dim"
            )
        if source not in _VALID_SOURCES:
            raise ValueError(
                f"Invalid source '{source}'; must be one of {sorted(_VALID_SOURCES)}"
            )

        features = _l2_normalise(features)
        confidence = float(max(0.0, min(1.0, confidence)))
        exemplar_id = uuid.uuid4().hex

        exemplar = Exemplar(
            exemplar_id=exemplar_id,
            category=category,
            features=features,
            confidence=confidence,
            added_at=time.time(),
            source=source,
        )

        with self._lock:
            self._exemplars.append(exemplar)
            self._update_prototype(category, features)

        logger.info(
            "OnlineLearner: added exemplar %s for category '%s' (source=%s, n=%d)",
            exemplar_id[:8],
            category,
            source,
            self.get_exemplar_count(),
        )
        self._auto_save()
        return exemplar_id

    def remove_exemplar(self, exemplar_id: str) -> bool:
        """Remove an exemplar by ID and recompute the affected prototype.

        If the removed exemplar was the last in its category, the category
        prototype is deleted entirely.

        Args:
            exemplar_id: The UUID hex string of the exemplar to remove.

        Returns:
            True if the exemplar was found and removed, False otherwise.
        """
        with self._lock:
            idx = None
            for i, ex in enumerate(self._exemplars):
                if ex.exemplar_id == exemplar_id:
                    idx = i
                    break
            if idx is None:
                return False

            removed = self._exemplars.pop(idx)
            category = removed.category
            self._recompute_prototype(category)

        logger.info(
            "OnlineLearner: removed exemplar %s from category '%s'",
            exemplar_id[:8],
            category,
        )
        self._auto_save()
        return True

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(
        self,
        features: np.ndarray,
        k: int = 5,
    ) -> tuple[str, float] | None:
        """Classify a query vector using k-NN weighted voting.

        Computes cosine distance to all exemplars, selects the k nearest,
        and performs a distance-weighted vote. The category with the highest
        total vote weight wins.

        Args:
            features: 768-dim query feature vector.
            k: Number of nearest neighbours to consider.

        Returns:
            ``(predicted_category, confidence)`` if the best match is within
            ``distance_threshold`` and the category has at least
            ``min_exemplars_for_prediction`` exemplars. ``None`` otherwise.
        """
        features = np.asarray(features, dtype=np.float32).ravel()
        if features.shape[0] != _FEATURE_DIM:
            return None

        features = _l2_normalise(features)

        with self._lock:
            if not self._exemplars:
                return None

            result = self._knn_vote(features, k)
            if result is not None:
                self._total_classifications += 1

        return result

    def classify_with_prototypes(
        self,
        features: np.ndarray,
    ) -> tuple[str, float] | None:
        """Fast classification using prototype (centroid) matching only.

        Computes cosine similarity between the query and each category
        prototype, selecting the closest one. This is O(C) where C is the
        number of categories, compared to O(N) for full k-NN.

        Args:
            features: 768-dim query feature vector.

        Returns:
            ``(predicted_category, confidence)`` or ``None`` if no match
            is within the distance threshold.
        """
        features = np.asarray(features, dtype=np.float32).ravel()
        if features.shape[0] != _FEATURE_DIM:
            return None

        features = _l2_normalise(features)

        with self._lock:
            if not self._prototypes:
                return None

            best_category: str | None = None
            best_similarity: float = -1.0

            for cat, proto in self._prototypes.items():
                if proto.exemplar_count < self._min_exemplars:
                    continue
                similarity = float(np.dot(features, proto.prototype))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_category = cat

            if best_category is None:
                return None

            distance = 1.0 - best_similarity
            if distance > self._distance_threshold:
                return None

            self._total_classifications += 1
            # Confidence = similarity clamped to [0, 1]
            confidence = max(0.0, min(1.0, best_similarity))
            return (best_category, round(confidence, 4))

    # ------------------------------------------------------------------
    # Novel category detection
    # ------------------------------------------------------------------

    def detect_novel(
        self,
        features: np.ndarray,
    ) -> tuple[bool, float]:
        """Detect whether a query represents a novel (unseen) category.

        Computes the minimum cosine distance from the query to all existing
        category prototypes. If this distance exceeds the threshold, the
        query is flagged as potentially novel.

        Args:
            features: 768-dim query feature vector.

        Returns:
            ``(is_novel, novelty_score)`` where ``novelty_score`` is the
            minimum distance to any prototype (higher = more novel).
            If there are no prototypes, returns ``(True, 1.0)``.
        """
        features = np.asarray(features, dtype=np.float32).ravel()
        if features.shape[0] != _FEATURE_DIM:
            return (True, 1.0)

        features = _l2_normalise(features)

        with self._lock:
            if not self._prototypes:
                return (True, 1.0)

            min_distance = float("inf")
            for proto in self._prototypes.values():
                dist = _cosine_distance(features, proto.prototype)
                if dist < min_distance:
                    min_distance = dist

        # Clamp novelty score to [0, 1]
        novelty_score = round(max(0.0, min(1.0, min_distance)), 4)
        is_novel = min_distance > self._distance_threshold

        if is_novel:
            self._total_novel_detections += 1
            logger.debug(
                "OnlineLearner: novel detection (score=%.3f, threshold=%.3f)",
                novelty_score,
                self._distance_threshold,
            )

        return (is_novel, novelty_score)

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def record_feedback(
        self,
        category: str,
        features: np.ndarray,
        correct: bool,
    ) -> None:
        """Record operator feedback for a classification result.

        Updates the accuracy statistics (total_queries, correct_queries)
        for the given category. Does not modify prototypes or exemplars.

        Args:
            category: The category that was predicted.
            features: The query features (unused currently, reserved for
                future auto-exemplar addition on confirmation).
            correct: True if the prediction was correct.
        """
        with self._lock:
            proto = self._prototypes.get(category)
            if proto is None:
                logger.warning(
                    "OnlineLearner: feedback for unknown category '%s'", category
                )
                return

            proto.total_queries += 1
            if correct:
                proto.correct_queries += 1

        logger.debug(
            "OnlineLearner: feedback for '%s' correct=%s (accuracy=%.3f)",
            category,
            correct,
            proto.accuracy,
        )
        self._auto_save()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_categories(self) -> list[dict[str, Any]]:
        """List all categories with their statistics.

        Returns:
            List of dicts, each containing:
            - category: str
            - exemplar_count: int
            - total_queries: int
            - correct_queries: int
            - accuracy: float
            - last_updated: float (Unix timestamp)
        """
        with self._lock:
            result = []
            for cat, proto in sorted(self._prototypes.items()):
                result.append({
                    "category": cat,
                    "exemplar_count": proto.exemplar_count,
                    "total_queries": proto.total_queries,
                    "correct_queries": proto.correct_queries,
                    "accuracy": round(proto.accuracy, 4),
                    "last_updated": proto.last_updated,
                })
            return result

    def get_exemplar_count(self) -> int:
        """Return total number of exemplars across all categories."""
        with self._lock:
            return len(self._exemplars)

    def get_stats(self) -> dict[str, Any]:
        """Return overall learner statistics for API/UI.

        Returns:
            Dict with keys: total_exemplars, total_categories,
            total_classifications, total_novel_detections,
            distance_threshold, min_exemplars_for_prediction,
            ema_alpha, categories, uptime_seconds.
        """
        with self._lock:
            categories = self.get_categories()
            total_exemplars = len(self._exemplars)
            total_categories = len(self._prototypes)

        return {
            "total_exemplars": total_exemplars,
            "total_categories": total_categories,
            "total_classifications": self._total_classifications,
            "total_novel_detections": self._total_novel_detections,
            "distance_threshold": self._distance_threshold,
            "min_exemplars_for_prediction": self._min_exemplars,
            "ema_alpha": self._ema_alpha,
            "categories": categories,
            "uptime_seconds": round(time.time() - self._created_at, 1),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist learner state to JSON file (best-effort).

        Format uses base64-encoded float32 arrays for compact storage.
        If ``persist_path`` was not set, this is a no-op.
        """
        if self._persist_path is None:
            return

        with self._lock:
            data = self._serialize()

        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.debug("OnlineLearner: saved state to %s", self._persist_path)
        except Exception:
            logger.exception(
                "OnlineLearner: failed to save state to %s", self._persist_path
            )

    def load(self) -> None:
        """Load learner state from JSON file (best-effort).

        If the file does not exist or is malformed, the learner starts
        with an empty state. Existing in-memory state is replaced.
        """
        if self._persist_path is None or not self._persist_path.exists():
            return
        try:
            raw = self._persist_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            with self._lock:
                self._deserialize(data)
            logger.info(
                "OnlineLearner: loaded %d exemplars, %d categories from %s",
                len(self._exemplars),
                len(self._prototypes),
                self._persist_path,
            )
        except Exception:
            logger.exception(
                "OnlineLearner: failed to load state from %s", self._persist_path
            )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all exemplars, prototypes, and statistics."""
        with self._lock:
            self._exemplars.clear()
            self._prototypes.clear()
            self._total_classifications = 0
            self._total_novel_detections = 0

        self.save()
        logger.info("OnlineLearner: reset all state")

    # ------------------------------------------------------------------
    # Internal: k-NN
    # ------------------------------------------------------------------

    def _knn_vote(
        self,
        features: np.ndarray,
        k: int,
    ) -> tuple[str, float] | None:
        """Compute k-NN distance-weighted vote (called under lock).

        For each exemplar, compute cosine distance. Take the top-k nearest
        and perform a weighted vote where weight = (1 - distance). The
        category with the highest total vote weight wins.

        Args:
            features: L2-normalised 768-dim query vector.
            k: Number of nearest neighbours.

        Returns:
            ``(category, confidence)`` or ``None`` if no valid match.
        """
        if not self._exemplars:
            return None

        # Build feature matrix for vectorised distance computation
        n = len(self._exemplars)
        # Stack all exemplar features into (N, 768) matrix
        exemplar_matrix = np.stack(
            [ex.features for ex in self._exemplars], axis=0
        )  # (N, 768)

        # Cosine similarity = dot product of normalised vectors
        similarities = exemplar_matrix @ features  # (N,)
        distances = 1.0 - similarities  # cosine distance

        # Select top-k nearest (smallest distance)
        effective_k = min(k, n)
        if effective_k >= n:
            # All exemplars fit within k — just sort by distance
            top_k_indices = np.argsort(distances)
        else:
            top_k_indices = np.argpartition(distances, effective_k)[:effective_k]
            top_k_indices = top_k_indices[np.argsort(distances[top_k_indices])]

        # Check if the nearest neighbour is within threshold
        if distances[top_k_indices[0]] > self._distance_threshold:
            return None

        # Distance-weighted vote
        votes: dict[str, float] = {}
        for idx in top_k_indices:
            idx_int = int(idx)
            ex = self._exemplars[idx_int]
            dist = float(distances[idx_int])

            # Skip exemplars beyond the threshold
            if dist > self._distance_threshold:
                continue

            weight = (1.0 - dist) * ex.confidence
            votes[ex.category] = votes.get(ex.category, 0.0) + weight

        if not votes:
            return None

        # Find winning category
        best_category = max(votes, key=lambda c: votes[c])

        # Check minimum exemplar count for this category
        proto = self._prototypes.get(best_category)
        if proto is None or proto.exemplar_count < self._min_exemplars:
            return None

        # Confidence: normalise vote weight to [0, 1]
        total_weight = sum(votes.values())
        if total_weight < 1e-12:
            return None
        confidence = votes[best_category] / total_weight

        return (best_category, round(confidence, 4))

    # ------------------------------------------------------------------
    # Internal: prototype management
    # ------------------------------------------------------------------

    def _update_prototype(self, category: str, new_features: np.ndarray) -> None:
        """Update prototype with EMA when a new exemplar is added (under lock).

        If this is the first exemplar, the prototype is set to the feature
        vector directly. Otherwise EMA is applied::

            prototype = (1 - alpha) * prototype + alpha * new_features
            prototype = L2_normalise(prototype)

        Args:
            category: Category label.
            new_features: L2-normalised 768-dim feature vector.
        """
        now = time.time()

        if category not in self._prototypes:
            self._prototypes[category] = CategoryPrototype(
                category=category,
                prototype=new_features.copy(),
                exemplar_count=1,
                last_updated=now,
            )
        else:
            proto = self._prototypes[category]
            alpha = self._ema_alpha
            updated = (1.0 - alpha) * proto.prototype + alpha * new_features
            proto.prototype = _l2_normalise(updated)
            proto.exemplar_count += 1
            proto.last_updated = now

    def _recompute_prototype(self, category: str) -> None:
        """Fully recompute prototype for a category from remaining exemplars.

        Called after exemplar removal. If no exemplars remain for the
        category, the prototype is deleted.
        """
        cat_exemplars = [ex for ex in self._exemplars if ex.category == category]

        if not cat_exemplars:
            # Remove the category entirely
            self._prototypes.pop(category, None)
            return

        # Compute mean of all exemplar features
        features_stack = np.stack(
            [ex.features for ex in cat_exemplars], axis=0
        )  # (M, 768)
        mean_features = features_stack.mean(axis=0)
        mean_features = _l2_normalise(mean_features)

        if category in self._prototypes:
            proto = self._prototypes[category]
            proto.prototype = mean_features
            proto.exemplar_count = len(cat_exemplars)
            proto.last_updated = time.time()
        else:
            # Should not happen, but handle gracefully
            self._prototypes[category] = CategoryPrototype(
                category=category,
                prototype=mean_features,
                exemplar_count=len(cat_exemplars),
                last_updated=time.time(),
            )

    # ------------------------------------------------------------------
    # Internal: serialization
    # ------------------------------------------------------------------

    def _serialize(self) -> dict[str, Any]:
        """Serialize learner state to a JSON-compatible dict (under lock).

        Exemplar features are base64-encoded float32 arrays for compact
        storage. Prototype vectors are similarly encoded.
        """
        categories: dict[str, Any] = {}
        for cat, proto in self._prototypes.items():
            categories[cat] = {
                "prototype": _features_to_b64(proto.prototype),
                "exemplar_count": proto.exemplar_count,
                "total_queries": proto.total_queries,
                "correct_queries": proto.correct_queries,
                "accuracy": round(proto.accuracy, 4),
                "last_updated": proto.last_updated,
            }

        exemplars: list[dict[str, Any]] = []
        for ex in self._exemplars:
            exemplars.append({
                "id": ex.exemplar_id,
                "category": ex.category,
                "features_b64": _features_to_b64(ex.features),
                "confidence": round(ex.confidence, 4),
                "added_at": ex.added_at,
                "source": ex.source,
            })

        return {
            "version": _VERSION,
            "total_classifications": self._total_classifications,
            "total_novel_detections": self._total_novel_detections,
            "created_at": self._created_at,
            "distance_threshold": self._distance_threshold,
            "min_exemplars_for_prediction": self._min_exemplars,
            "ema_alpha": self._ema_alpha,
            "categories": categories,
            "exemplars": exemplars,
        }

    def _deserialize(self, data: dict[str, Any]) -> None:
        """Deserialize learner state from a JSON dict (under lock).

        Replaces all current in-memory state. Tolerant of missing keys
        to support forward compatibility.
        """
        version = data.get("version", 1)
        if version != _VERSION:
            logger.warning(
                "OnlineLearner: persistence version mismatch (got %d, expected %d)",
                version,
                _VERSION,
            )

        self._total_classifications = int(data.get("total_classifications", 0))
        self._total_novel_detections = int(data.get("total_novel_detections", 0))
        self._created_at = float(data.get("created_at", time.time()))

        # Restore exemplars
        self._exemplars.clear()
        for ex_data in data.get("exemplars", []):
            try:
                features = _b64_to_features(ex_data["features_b64"])
                if features.shape[0] != _FEATURE_DIM:
                    logger.warning(
                        "OnlineLearner: skipping exemplar %s with wrong dim %d",
                        ex_data.get("id", "?"),
                        features.shape[0],
                    )
                    continue
                exemplar = Exemplar(
                    exemplar_id=ex_data["id"],
                    category=ex_data["category"],
                    features=_l2_normalise(features),
                    confidence=float(ex_data.get("confidence", 1.0)),
                    added_at=float(ex_data.get("added_at", 0.0)),
                    source=str(ex_data.get("source", "operator")),
                )
                self._exemplars.append(exemplar)
            except Exception:
                logger.exception(
                    "OnlineLearner: failed to deserialize exemplar %s",
                    ex_data.get("id", "?"),
                )

        # Restore prototypes
        self._prototypes.clear()
        for cat, cat_data in data.get("categories", {}).items():
            try:
                prototype = _b64_to_features(cat_data["prototype"])
                if prototype.shape[0] != _FEATURE_DIM:
                    logger.warning(
                        "OnlineLearner: skipping prototype '%s' with wrong dim %d",
                        cat,
                        prototype.shape[0],
                    )
                    continue
                self._prototypes[cat] = CategoryPrototype(
                    category=cat,
                    prototype=_l2_normalise(prototype),
                    exemplar_count=int(cat_data.get("exemplar_count", 0)),
                    total_queries=int(cat_data.get("total_queries", 0)),
                    correct_queries=int(cat_data.get("correct_queries", 0)),
                    last_updated=float(cat_data.get("last_updated", 0.0)),
                )
            except Exception:
                logger.exception(
                    "OnlineLearner: failed to deserialize prototype '%s'", cat
                )

    # ------------------------------------------------------------------
    # Internal: auto-save helper
    # ------------------------------------------------------------------

    def _auto_save(self) -> None:
        """Best-effort persist after state changes."""
        try:
            self.save()
        except Exception:
            logger.exception("OnlineLearner: auto-save failed")
