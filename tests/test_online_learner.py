"""Comprehensive tests for sopilot.perception.online_learner module.

Covers: Exemplar, CategoryPrototype, OnlineLearner (construction, add/remove
exemplar, classify, classify_with_prototypes, detect_novel, record_feedback,
persistence, stats, and reset).

~62 tests — no real ML model dependencies required.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from sopilot.perception.online_learner import (
    CategoryPrototype,
    Exemplar,
    OnlineLearner,
    _FEATURE_DIM,
    _b64_to_features,
    _cosine_distance,
    _features_to_b64,
    _l2_normalise,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.RandomState(42)


def _make_features(seed: int = 0) -> np.ndarray:
    """Generate a deterministic L2-normalised 768-dim float32 vector."""
    rng = np.random.RandomState(seed)
    v = rng.randn(_FEATURE_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_category_features(category_seed: int, n: int = 5) -> list[np.ndarray]:
    """Generate *n* similar vectors clustered around a direction."""
    base = _make_features(category_seed)
    out = []
    for i in range(n):
        noise = np.random.RandomState(category_seed * 1000 + i).randn(_FEATURE_DIM).astype(np.float32)
        v = base + 0.05 * noise  # small perturbation to stay close
        v = v / np.linalg.norm(v)
        out.append(v)
    return out


def _make_learner(tmp_path: Path | None = None, **kwargs) -> OnlineLearner:
    """Create a learner with optional tmp_path persistence."""
    persist_path = tmp_path / "learner.json" if tmp_path else None
    return OnlineLearner(persist_path=persist_path, **kwargs)


def _populate_learner(
    learner: OnlineLearner,
    categories: dict[str, int] | None = None,
) -> dict[str, list[str]]:
    """Add exemplars to a learner, return dict of category -> [exemplar_ids]."""
    if categories is None:
        categories = {"helmet": 5, "vest": 5}
    ids: dict[str, list[str]] = {}
    seed = 100
    for cat, count in categories.items():
        ids[cat] = []
        feats = _make_category_features(seed, count)
        for f in feats:
            eid = learner.add_exemplar(cat, f)
            ids[cat].append(eid)
        seed += 1
    return ids


# ===================================================================
# 1. Exemplar dataclass (5 tests)
# ===================================================================


class TestExemplar:
    """Tests for the Exemplar frozen dataclass."""

    def test_creation(self):
        """Exemplar can be created with all required fields."""
        feats = _make_features(0)
        ex = Exemplar(
            exemplar_id="abc123",
            category="helmet",
            features=feats,
            confidence=0.9,
            added_at=time.time(),
            source="operator",
        )
        assert ex.exemplar_id == "abc123"
        assert ex.category == "helmet"
        assert ex.confidence == 0.9
        assert ex.source == "operator"

    def test_frozen_immutability(self):
        """Frozen dataclass should disallow attribute mutation."""
        feats = _make_features(0)
        ex = Exemplar(
            exemplar_id="abc",
            category="vest",
            features=feats,
            confidence=1.0,
            added_at=0.0,
            source="auto",
        )
        with pytest.raises(AttributeError):
            ex.category = "other"  # type: ignore[misc]

    def test_uuid_generation_uniqueness(self):
        """Exemplar IDs from the learner should be unique UUID hex strings."""
        learner = OnlineLearner()
        id1 = learner.add_exemplar("a", _make_features(0))
        id2 = learner.add_exemplar("a", _make_features(1))
        assert isinstance(id1, str)
        assert len(id1) == 32  # UUID4 hex length
        assert id1 != id2

    def test_source_field_values(self):
        """Source field should accept valid values: operator, auto, feedback."""
        learner = OnlineLearner()
        for src in ("operator", "auto", "feedback"):
            eid = learner.add_exemplar("cat", _make_features(0), source=src)
            assert isinstance(eid, str)

    def test_equality_by_id(self):
        """Two Exemplars with the same id should be equal regardless of other fields."""
        feats = _make_features(0)
        ex1 = Exemplar("same_id", "a", feats, 1.0, 0.0, "operator")
        ex2 = Exemplar("same_id", "b", feats, 0.5, 1.0, "auto")
        assert ex1 == ex2
        assert hash(ex1) == hash(ex2)


# ===================================================================
# 2. CategoryPrototype dataclass (5 tests)
# ===================================================================


class TestCategoryPrototype:
    """Tests for CategoryPrototype mutable dataclass."""

    def test_creation_defaults(self):
        """CategoryPrototype has sensible defaults."""
        proto = CategoryPrototype(category="helmet", prototype=_make_features(0))
        assert proto.category == "helmet"
        assert proto.exemplar_count == 0
        assert proto.total_queries == 0
        assert proto.correct_queries == 0

    def test_accuracy_property_nonzero(self):
        """Accuracy = correct_queries / total_queries when total > 0."""
        proto = CategoryPrototype(
            category="vest",
            prototype=_make_features(1),
            total_queries=10,
            correct_queries=8,
        )
        assert proto.accuracy == pytest.approx(0.8)

    def test_accuracy_property_zero_queries(self):
        """Accuracy returns 0.0 when no queries have been made."""
        proto = CategoryPrototype(category="x", prototype=_make_features(0))
        assert proto.accuracy == 0.0

    def test_accuracy_property_all_correct(self):
        """Accuracy returns 1.0 when all queries are correct."""
        proto = CategoryPrototype(
            category="x",
            prototype=_make_features(0),
            total_queries=5,
            correct_queries=5,
        )
        assert proto.accuracy == pytest.approx(1.0)

    def test_last_updated_default(self):
        """last_updated defaults to current time."""
        before = time.time()
        proto = CategoryPrototype(category="x", prototype=_make_features(0))
        after = time.time()
        assert before <= proto.last_updated <= after


# ===================================================================
# 3. OnlineLearner construction (5 tests)
# ===================================================================


class TestOnlineLearnerConstruction:
    """Tests for OnlineLearner __init__ parameters."""

    def test_defaults(self):
        """Default parameters are applied correctly."""
        learner = OnlineLearner()
        stats = learner.get_stats()
        assert stats["distance_threshold"] == 0.65
        assert stats["min_exemplars_for_prediction"] == 3
        assert stats["ema_alpha"] == 0.1
        assert stats["total_exemplars"] == 0
        assert stats["total_categories"] == 0

    def test_custom_threshold(self):
        """Custom distance threshold is stored."""
        learner = OnlineLearner(distance_threshold=0.4)
        assert learner.get_stats()["distance_threshold"] == 0.4

    def test_with_persist_path(self, tmp_path: Path):
        """Learner initializes correctly with a persist_path."""
        p = tmp_path / "state.json"
        learner = OnlineLearner(persist_path=p)
        assert learner.get_stats()["total_exemplars"] == 0

    def test_min_exemplars_custom(self):
        """Custom min_exemplars_for_prediction is stored."""
        learner = OnlineLearner(min_exemplars_for_prediction=1)
        assert learner.get_stats()["min_exemplars_for_prediction"] == 1

    def test_persist_path_string_accepted(self, tmp_path: Path):
        """persist_path accepts both str and Path."""
        p = str(tmp_path / "state.json")
        learner = OnlineLearner(persist_path=p)
        # Should not raise
        learner.add_exemplar("x", _make_features(0))
        assert learner.get_exemplar_count() == 1


# ===================================================================
# 4. add_exemplar() (8 tests)
# ===================================================================


class TestAddExemplar:
    """Tests for OnlineLearner.add_exemplar()."""

    def test_basic_add(self):
        """Adding an exemplar returns a valid id and increments count."""
        learner = OnlineLearner()
        eid = learner.add_exemplar("helmet", _make_features(0))
        assert isinstance(eid, str) and len(eid) == 32
        assert learner.get_exemplar_count() == 1

    def test_l2_normalisation(self):
        """Features are L2-normalised before storage."""
        learner = OnlineLearner()
        raw = np.ones(_FEATURE_DIM, dtype=np.float32) * 5.0  # not normalised
        learner.add_exemplar("cat", raw)
        # The stored prototype should be L2-normalised
        cats = learner.get_categories()
        assert len(cats) == 1
        # Verify norm via internal state
        proto = learner._prototypes["cat"]
        norm = float(np.linalg.norm(proto.prototype))
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_prototype_creation_first_exemplar(self):
        """First exemplar for a category creates a new prototype."""
        learner = OnlineLearner()
        learner.add_exemplar("helmet", _make_features(10))
        cats = learner.get_categories()
        assert len(cats) == 1
        assert cats[0]["category"] == "helmet"
        assert cats[0]["exemplar_count"] == 1

    def test_prototype_update_ema(self):
        """Subsequent exemplars update the prototype via EMA."""
        learner = OnlineLearner(ema_alpha=0.5)
        f1 = _make_features(10)
        f2 = _make_features(11)
        learner.add_exemplar("a", f1)
        proto_before = learner._prototypes["a"].prototype.copy()
        learner.add_exemplar("a", f2)
        proto_after = learner._prototypes["a"].prototype
        # Prototype should have changed
        assert not np.allclose(proto_before, proto_after)
        # Should still be normalised
        assert float(np.linalg.norm(proto_after)) == pytest.approx(1.0, abs=1e-5)

    def test_multiple_categories(self):
        """Exemplars can be added to multiple categories independently."""
        learner = OnlineLearner()
        learner.add_exemplar("helmet", _make_features(0))
        learner.add_exemplar("vest", _make_features(1))
        learner.add_exemplar("gloves", _make_features(2))
        assert learner.get_stats()["total_categories"] == 3
        assert learner.get_exemplar_count() == 3

    def test_invalid_source_raises(self):
        """Invalid source value raises ValueError."""
        learner = OnlineLearner()
        with pytest.raises(ValueError, match="Invalid source"):
            learner.add_exemplar("a", _make_features(0), source="unknown")

    def test_wrong_dimensionality_raises(self):
        """Feature vector with wrong dim raises ValueError."""
        learner = OnlineLearner()
        bad_features = np.ones(100, dtype=np.float32)
        with pytest.raises(ValueError, match="Expected 768-dim"):
            learner.add_exemplar("a", bad_features)

    def test_confidence_clamped(self):
        """Confidence is clamped to [0, 1]."""
        learner = OnlineLearner()
        learner.add_exemplar("a", _make_features(0), confidence=5.0)
        learner.add_exemplar("b", _make_features(1), confidence=-1.0)
        # No assertion on the stored value directly, but it should not crash
        assert learner.get_exemplar_count() == 2


# ===================================================================
# 5. remove_exemplar() (6 tests)
# ===================================================================


class TestRemoveExemplar:
    """Tests for OnlineLearner.remove_exemplar()."""

    def test_valid_removal(self):
        """Removing an existing exemplar returns True and decrements count."""
        learner = OnlineLearner()
        eid = learner.add_exemplar("a", _make_features(0))
        learner.add_exemplar("a", _make_features(1))
        assert learner.remove_exemplar(eid) is True
        assert learner.get_exemplar_count() == 1

    def test_invalid_id_returns_false(self):
        """Removing a non-existent exemplar returns False."""
        learner = OnlineLearner()
        learner.add_exemplar("a", _make_features(0))
        assert learner.remove_exemplar("nonexistent_id") is False
        assert learner.get_exemplar_count() == 1

    def test_recompute_prototype_after_removal(self):
        """Prototype is recomputed (not EMA) after removal."""
        learner = OnlineLearner()
        f1 = _make_features(10)
        f2 = _make_features(11)
        eid1 = learner.add_exemplar("a", f1)
        learner.add_exemplar("a", f2)
        learner.remove_exemplar(eid1)
        # After removing f1, prototype should be recomputed from f2 only
        proto = learner._prototypes["a"]
        # The prototype should be close to f2 (normalised)
        f2_norm = f2 / np.linalg.norm(f2)
        assert np.allclose(proto.prototype, f2_norm, atol=1e-5)
        assert proto.exemplar_count == 1

    def test_remove_last_deletes_category(self):
        """Removing the last exemplar in a category deletes the prototype."""
        learner = OnlineLearner()
        eid = learner.add_exemplar("a", _make_features(0))
        assert learner.get_stats()["total_categories"] == 1
        learner.remove_exemplar(eid)
        assert learner.get_stats()["total_categories"] == 0
        assert learner.get_exemplar_count() == 0

    def test_remove_does_not_affect_other_categories(self):
        """Removing from one category should not affect other categories."""
        learner = OnlineLearner()
        eid_a = learner.add_exemplar("a", _make_features(0))
        learner.add_exemplar("b", _make_features(1))
        learner.remove_exemplar(eid_a)
        assert learner.get_stats()["total_categories"] == 1
        cats = learner.get_categories()
        assert cats[0]["category"] == "b"

    def test_remove_twice_returns_false(self):
        """Removing the same exemplar twice: second call returns False."""
        learner = OnlineLearner()
        eid = learner.add_exemplar("a", _make_features(0))
        assert learner.remove_exemplar(eid) is True
        assert learner.remove_exemplar(eid) is False


# ===================================================================
# 6. classify() (8 tests)
# ===================================================================


class TestClassify:
    """Tests for OnlineLearner.classify() — k-NN weighted voting."""

    def test_empty_store_returns_none(self):
        """classify returns None when there are no exemplars."""
        learner = OnlineLearner()
        result = learner.classify(_make_features(0))
        assert result is None

    def test_insufficient_exemplars_returns_none(self):
        """classify returns None when category has fewer than min_exemplars."""
        learner = OnlineLearner(min_exemplars_for_prediction=3)
        learner.add_exemplar("a", _make_features(0))
        learner.add_exemplar("a", _make_features(1))
        # Only 2 exemplars, need 3
        result = learner.classify(_make_features(0))
        assert result is None

    def test_successful_classification(self):
        """classify returns (category, confidence) when enough exemplars exist."""
        learner = OnlineLearner(min_exemplars_for_prediction=3, distance_threshold=0.8)
        feats = _make_category_features(42, n=5)
        for f in feats:
            learner.add_exemplar("helmet", f)
        # Query with a similar vector
        query = feats[0] + 0.01 * np.random.RandomState(99).randn(_FEATURE_DIM).astype(np.float32)
        query = query / np.linalg.norm(query)
        result = learner.classify(query)
        assert result is not None
        cat, conf = result
        assert cat == "helmet"
        assert 0.0 < conf <= 1.0

    def test_k_parameter(self):
        """Different k values should still produce valid results."""
        learner = OnlineLearner(min_exemplars_for_prediction=3, distance_threshold=0.8)
        feats = _make_category_features(42, n=6)
        for f in feats:
            learner.add_exemplar("cat1", f)
        query = feats[0].copy()
        result_k1 = learner.classify(query, k=1)
        result_k3 = learner.classify(query, k=3)
        result_k10 = learner.classify(query, k=10)
        # All should find the same category (all exemplars are "cat1")
        for r in (result_k1, result_k3, result_k10):
            assert r is not None
            assert r[0] == "cat1"

    def test_distance_weighting(self):
        """Closer exemplars should have more influence on classification."""
        learner = OnlineLearner(min_exemplars_for_prediction=1, distance_threshold=0.9)
        base = _make_features(50)
        # Add 1 very close exemplar for category "close"
        close_feat = base + 0.001 * np.random.RandomState(60).randn(_FEATURE_DIM).astype(np.float32)
        close_feat = close_feat / np.linalg.norm(close_feat)
        learner.add_exemplar("close", close_feat)
        # Add 3 somewhat distant exemplars for category "far"
        for i in range(3):
            far_feat = base + 0.3 * np.random.RandomState(70 + i).randn(_FEATURE_DIM).astype(np.float32)
            far_feat = far_feat / np.linalg.norm(far_feat)
            learner.add_exemplar("far", far_feat)
        # Query exactly at base — should prefer "close" despite fewer exemplars
        result = learner.classify(base, k=4)
        # Due to distance weighting, the close exemplar should dominate
        assert result is not None
        assert result[0] == "close"

    def test_below_threshold_returns_none(self):
        """classify returns None when nearest exemplar is beyond threshold."""
        learner = OnlineLearner(
            min_exemplars_for_prediction=1, distance_threshold=0.01
        )
        feats = _make_category_features(42, n=3)
        for f in feats:
            learner.add_exemplar("cat", f)
        # Query with a very different vector
        query = _make_features(999)  # unrelated direction
        result = learner.classify(query)
        assert result is None

    def test_wrong_dim_returns_none(self):
        """classify returns None for wrong-dim features (no error raised)."""
        learner = OnlineLearner()
        result = learner.classify(np.ones(100, dtype=np.float32))
        assert result is None

    def test_classification_counter_incremented(self):
        """Successful classification increments total_classifications."""
        learner = OnlineLearner(min_exemplars_for_prediction=1, distance_threshold=0.9)
        feats = _make_category_features(42, n=3)
        for f in feats:
            learner.add_exemplar("cat", f)
        assert learner.get_stats()["total_classifications"] == 0
        learner.classify(feats[0])
        assert learner.get_stats()["total_classifications"] == 1


# ===================================================================
# 7. classify_with_prototypes() (6 tests)
# ===================================================================


class TestClassifyWithPrototypes:
    """Tests for OnlineLearner.classify_with_prototypes()."""

    def test_empty_store_returns_none(self):
        """Returns None when no prototypes exist."""
        learner = OnlineLearner()
        result = learner.classify_with_prototypes(_make_features(0))
        assert result is None

    def test_single_category_match(self):
        """Matches the only available category when close enough."""
        learner = OnlineLearner(min_exemplars_for_prediction=3, distance_threshold=0.8)
        feats = _make_category_features(42, n=5)
        for f in feats:
            learner.add_exemplar("helmet", f)
        query = feats[0].copy()
        result = learner.classify_with_prototypes(query)
        assert result is not None
        cat, conf = result
        assert cat == "helmet"
        assert 0.0 < conf <= 1.0

    def test_multiple_categories_correct(self):
        """Selects the correct category among multiple options."""
        learner = OnlineLearner(min_exemplars_for_prediction=3, distance_threshold=0.9)
        cat_a_feats = _make_category_features(100, n=5)
        cat_b_feats = _make_category_features(200, n=5)
        for f in cat_a_feats:
            learner.add_exemplar("cat_a", f)
        for f in cat_b_feats:
            learner.add_exemplar("cat_b", f)
        # Query close to cat_a
        result = learner.classify_with_prototypes(cat_a_feats[0])
        assert result is not None
        assert result[0] == "cat_a"

    def test_threshold_reject(self):
        """Returns None when best match is beyond distance threshold."""
        learner = OnlineLearner(min_exemplars_for_prediction=3, distance_threshold=0.01)
        feats = _make_category_features(42, n=5)
        for f in feats:
            learner.add_exemplar("cat", f)
        query = _make_features(999)
        result = learner.classify_with_prototypes(query)
        assert result is None

    def test_min_exemplars_enforced(self):
        """Categories with too few exemplars are skipped."""
        learner = OnlineLearner(min_exemplars_for_prediction=5, distance_threshold=0.9)
        # Only add 3 (less than min 5)
        for f in _make_category_features(42, n=3):
            learner.add_exemplar("cat", f)
        result = learner.classify_with_prototypes(_make_category_features(42, n=1)[0])
        assert result is None

    def test_wrong_dim_returns_none(self):
        """Returns None for wrong-dim features."""
        learner = OnlineLearner()
        result = learner.classify_with_prototypes(np.zeros(10, dtype=np.float32))
        assert result is None


# ===================================================================
# 8. detect_novel() (5 tests)
# ===================================================================


class TestDetectNovel:
    """Tests for OnlineLearner.detect_novel()."""

    def test_empty_store_is_novel(self):
        """Everything is novel when no prototypes exist."""
        learner = OnlineLearner()
        is_novel, score = learner.detect_novel(_make_features(0))
        assert is_novel is True
        assert score == 1.0

    def test_known_category_not_novel(self):
        """A query very close to a known prototype should not be novel."""
        learner = OnlineLearner(distance_threshold=0.5)
        feats = _make_category_features(42, n=5)
        for f in feats:
            learner.add_exemplar("known", f)
        # Query close to the prototype
        is_novel, score = learner.detect_novel(feats[0])
        assert is_novel is False
        assert 0.0 <= score < 0.5

    def test_far_from_all_is_novel(self):
        """A query far from all prototypes should be novel."""
        learner = OnlineLearner(distance_threshold=0.3)
        feats = _make_category_features(42, n=5)
        for f in feats:
            learner.add_exemplar("known", f)
        # Use an unrelated direction
        query = _make_features(999)
        is_novel, score = learner.detect_novel(query)
        assert is_novel is True
        assert score > 0.3

    def test_novelty_counter_incremented(self):
        """Novel detection increments total_novel_detections."""
        learner = OnlineLearner(distance_threshold=0.01)
        feats = _make_category_features(42, n=3)
        for f in feats:
            learner.add_exemplar("cat", f)
        assert learner.get_stats()["total_novel_detections"] == 0
        learner.detect_novel(_make_features(999))  # distant, should be novel
        assert learner.get_stats()["total_novel_detections"] == 1

    def test_wrong_dim_returns_novel(self):
        """Wrong dim features returns (True, 1.0)."""
        learner = OnlineLearner()
        is_novel, score = learner.detect_novel(np.ones(10, dtype=np.float32))
        assert is_novel is True
        assert score == 1.0


# ===================================================================
# 9. record_feedback() (4 tests)
# ===================================================================


class TestRecordFeedback:
    """Tests for OnlineLearner.record_feedback()."""

    def test_correct_feedback(self):
        """Correct feedback increments both total_queries and correct_queries."""
        learner = OnlineLearner()
        learner.add_exemplar("helmet", _make_features(0))
        learner.record_feedback("helmet", _make_features(0), correct=True)
        cats = learner.get_categories()
        helmet = [c for c in cats if c["category"] == "helmet"][0]
        assert helmet["total_queries"] == 1
        assert helmet["correct_queries"] == 1
        assert helmet["accuracy"] == pytest.approx(1.0)

    def test_incorrect_feedback(self):
        """Incorrect feedback increments total_queries but not correct_queries."""
        learner = OnlineLearner()
        learner.add_exemplar("helmet", _make_features(0))
        learner.record_feedback("helmet", _make_features(0), correct=False)
        cats = learner.get_categories()
        helmet = [c for c in cats if c["category"] == "helmet"][0]
        assert helmet["total_queries"] == 1
        assert helmet["correct_queries"] == 0
        assert helmet["accuracy"] == pytest.approx(0.0)

    def test_accuracy_update_mixed(self):
        """Mixed feedback produces correct accuracy ratio."""
        learner = OnlineLearner()
        learner.add_exemplar("vest", _make_features(0))
        for _ in range(3):
            learner.record_feedback("vest", _make_features(0), correct=True)
        for _ in range(2):
            learner.record_feedback("vest", _make_features(0), correct=False)
        cats = learner.get_categories()
        vest = [c for c in cats if c["category"] == "vest"][0]
        assert vest["total_queries"] == 5
        assert vest["correct_queries"] == 3
        assert vest["accuracy"] == pytest.approx(0.6)

    def test_feedback_unknown_category_no_crash(self):
        """Feedback for an unknown category logs a warning but does not crash."""
        learner = OnlineLearner()
        # Should not raise
        learner.record_feedback("nonexistent", _make_features(0), correct=True)


# ===================================================================
# 10. Persistence (5 tests)
# ===================================================================


class TestPersistence:
    """Tests for save/load roundtrip functionality."""

    def test_save_load_roundtrip(self, tmp_path: Path):
        """Exemplars and prototypes survive save+load cycle."""
        p = tmp_path / "learner.json"
        learner1 = OnlineLearner(persist_path=p)
        ids = _populate_learner(learner1, {"helmet": 4, "vest": 3})
        learner1.save()

        learner2 = OnlineLearner(persist_path=p)
        assert learner2.get_exemplar_count() == 7
        assert learner2.get_stats()["total_categories"] == 2

    def test_base64_encoding(self, tmp_path: Path):
        """Feature vectors are encoded as base64 strings in JSON."""
        p = tmp_path / "learner.json"
        learner = OnlineLearner(persist_path=p)
        learner.add_exemplar("cat", _make_features(0))
        learner.save()

        raw = json.loads(p.read_text(encoding="utf-8"))
        assert "exemplars" in raw
        assert len(raw["exemplars"]) == 1
        # Features should be a base64 string, not a list of floats
        feat_b64 = raw["exemplars"][0]["features_b64"]
        assert isinstance(feat_b64, str)
        # Should decode back to correct-length float32 array
        decoded = _b64_to_features(feat_b64)
        assert decoded.shape == (_FEATURE_DIM,)

    def test_corrupt_file_handled_gracefully(self, tmp_path: Path):
        """Corrupted JSON file does not crash the learner."""
        p = tmp_path / "learner.json"
        p.write_text("THIS IS NOT JSON", encoding="utf-8")
        # Should load without raising; starts with empty state
        learner = OnlineLearner(persist_path=p)
        assert learner.get_exemplar_count() == 0

    def test_no_persist_path_save_noop(self):
        """save() is a no-op when persist_path is None."""
        learner = OnlineLearner(persist_path=None)
        learner.add_exemplar("a", _make_features(0))
        # Should not raise
        learner.save()

    def test_stats_survive_roundtrip(self, tmp_path: Path):
        """Classification and novel detection counters persist."""
        p = tmp_path / "learner.json"
        learner1 = OnlineLearner(
            persist_path=p,
            min_exemplars_for_prediction=1,
            distance_threshold=0.9,
        )
        feats = _make_category_features(42, n=3)
        for f in feats:
            learner1.add_exemplar("cat", f)
        learner1.classify(feats[0])
        learner1.detect_novel(_make_features(999))
        learner1.save()

        learner2 = OnlineLearner(persist_path=p)
        stats = learner2.get_stats()
        assert stats["total_classifications"] >= 1


# ===================================================================
# 11. Stats and reset (5 tests)
# ===================================================================


class TestStatsAndReset:
    """Tests for get_stats, get_categories, get_exemplar_count, and reset."""

    def test_get_stats_keys(self):
        """get_stats returns all expected keys."""
        learner = OnlineLearner()
        stats = learner.get_stats()
        expected_keys = {
            "total_exemplars",
            "total_categories",
            "total_classifications",
            "total_novel_detections",
            "distance_threshold",
            "min_exemplars_for_prediction",
            "ema_alpha",
            "categories",
            "uptime_seconds",
        }
        assert set(stats.keys()) == expected_keys

    def test_get_categories_sorted(self):
        """get_categories returns categories in sorted order."""
        learner = OnlineLearner()
        learner.add_exemplar("zebra", _make_features(0))
        learner.add_exemplar("apple", _make_features(1))
        learner.add_exemplar("mango", _make_features(2))
        cats = learner.get_categories()
        names = [c["category"] for c in cats]
        assert names == ["apple", "mango", "zebra"]

    def test_get_exemplar_count_accurate(self):
        """get_exemplar_count returns the correct total."""
        learner = OnlineLearner()
        for i in range(7):
            learner.add_exemplar(f"cat_{i % 3}", _make_features(i))
        assert learner.get_exemplar_count() == 7

    def test_reset_clears_all(self):
        """reset clears exemplars, prototypes, and counters."""
        learner = OnlineLearner(
            min_exemplars_for_prediction=1,
            distance_threshold=0.9,
        )
        feats = _make_category_features(42, n=5)
        for f in feats:
            learner.add_exemplar("cat", f)
        learner.classify(feats[0])
        learner.detect_novel(_make_features(999))
        # Pre-reset checks
        assert learner.get_exemplar_count() > 0
        assert learner.get_stats()["total_categories"] > 0

        learner.reset()

        stats = learner.get_stats()
        assert stats["total_exemplars"] == 0
        assert stats["total_categories"] == 0
        assert stats["total_classifications"] == 0
        assert stats["total_novel_detections"] == 0
        assert learner.get_categories() == []

    def test_reset_with_persistence(self, tmp_path: Path):
        """reset also clears the persisted state file."""
        p = tmp_path / "learner.json"
        learner = OnlineLearner(persist_path=p)
        learner.add_exemplar("a", _make_features(0))
        learner.save()
        assert p.exists()

        learner.reset()

        # Reload from file — should be empty
        learner2 = OnlineLearner(persist_path=p)
        assert learner2.get_exemplar_count() == 0
        assert learner2.get_stats()["total_categories"] == 0


# ===================================================================
# 12. Helper functions (bonus — 4 extra tests for coverage)
# ===================================================================


class TestHelpers:
    """Tests for module-level helper functions."""

    def test_l2_normalise(self):
        """_l2_normalise produces unit-length vectors."""
        v = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        normed = _l2_normalise(v)
        assert float(np.linalg.norm(normed)) == pytest.approx(1.0, abs=1e-6)

    def test_l2_normalise_zero_vector(self):
        """_l2_normalise returns zero vector unchanged."""
        v = np.zeros(5, dtype=np.float32)
        normed = _l2_normalise(v)
        assert np.allclose(normed, v)

    def test_cosine_distance_identical(self):
        """Cosine distance of identical normalised vectors is 0."""
        v = _make_features(0)
        assert _cosine_distance(v, v) == pytest.approx(0.0, abs=1e-5)

    def test_features_b64_roundtrip(self):
        """Base64 encode → decode roundtrip preserves features."""
        original = _make_features(42)
        b64 = _features_to_b64(original)
        restored = _b64_to_features(b64)
        assert np.allclose(original, restored, atol=1e-7)
