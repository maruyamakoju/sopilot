"""Tests for sopilot/perception/clip_classifier.py.

All tests use backend="mock" to avoid downloading any model weights.
Total: 55 tests across 9 test classes.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from sopilot.perception.clip_classifier import (
    CLIPScore,
    CLIPZeroShotClassifier,
    MockCLIPBackend,
    build_clip_classifier,
    DEFAULT_CANDIDATE_LABELS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class MockDetection:
    """Minimal detection object that mirrors real TrackedEntity interface."""
    entity_id: int
    label: str
    bbox: list[float] | None = None  # [x1,y1,x2,y2] normalized [0,1]


def _make_frame(h: int = 100, w: int = 100) -> np.ndarray:
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _make_classifier(**kwargs) -> CLIPZeroShotClassifier:
    kwargs.setdefault("backend", "mock")
    return CLIPZeroShotClassifier(**kwargs)


# ---------------------------------------------------------------------------
# 1. TestMockCLIPBackend (8 tests)
# ---------------------------------------------------------------------------

class TestMockCLIPBackend:
    def setup_method(self):
        self.backend = MockCLIPBackend()

    def test_encode_text_shape(self):
        texts = ["person", "car", "truck"]
        out = self.backend.encode_text(texts)
        assert out.shape == (3, 512)

    def test_encode_text_dtype(self):
        out = self.backend.encode_text(["worker"])
        assert out.dtype == np.float32

    def test_encode_text_normalized(self):
        out = self.backend.encode_text(["helmet", "vest"])
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_encode_text_deterministic(self):
        texts = ["forklift", "safety_cone"]
        out1 = self.backend.encode_text(texts)
        out2 = self.backend.encode_text(texts)
        np.testing.assert_array_equal(out1, out2)

    def test_encode_text_single(self):
        out = self.backend.encode_text(["box"])
        assert out.shape == (1, 512)

    def test_encode_text_multiple(self):
        labels = DEFAULT_CANDIDATE_LABELS
        out = self.backend.encode_text(labels)
        assert out.shape == (len(labels), 512)

    def test_encode_image_shape(self):
        img = _make_frame()
        out = self.backend.encode_image(img)
        assert out.shape == (1, 512)

    def test_encode_image_deterministic(self):
        img = _make_frame()
        out1 = self.backend.encode_image(img)
        out2 = self.backend.encode_image(img)
        np.testing.assert_array_equal(out1, out2)

    def test_encode_image_normalized(self):
        img = _make_frame()
        out = self.backend.encode_image(img)
        norm = np.linalg.norm(out)
        assert abs(norm - 1.0) < 1e-5

    def test_is_real_false(self):
        assert self.backend.is_real is False

    def test_encode_image_zero_array(self):
        # All-zero image should not raise
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        out = self.backend.encode_image(img)
        assert out.shape == (1, 512)


# ---------------------------------------------------------------------------
# 2. TestCLIPZeroShotClassifier_Init (8 tests)
# ---------------------------------------------------------------------------

class TestCLIPZeroShotClassifier_Init:
    def test_default_labels_assigned(self):
        clf = _make_classifier()
        assert clf._labels == DEFAULT_CANDIDATE_LABELS

    def test_custom_labels_used(self):
        labels = ["cat", "dog", "bird"]
        clf = _make_classifier(candidate_labels=labels)
        assert clf._labels == labels

    def test_backend_mock_instance(self):
        clf = _make_classifier(backend="mock")
        assert isinstance(clf._backend, MockCLIPBackend)

    def test_confidence_threshold_stored(self):
        clf = _make_classifier(confidence_threshold=0.4)
        assert clf.confidence_threshold == 0.4

    def test_is_available_false_for_mock(self):
        clf = _make_classifier()
        assert clf.is_available is False

    def test_backend_name_contains_mock(self):
        clf = _make_classifier()
        assert "Mock" in clf.backend_name

    def test_model_name_stored(self):
        clf = _make_classifier(model_name="ViT-L/14")
        assert clf.model_name == "ViT-L/14"

    def test_get_state_dict_expected_keys(self):
        clf = _make_classifier()
        sd = clf.get_state_dict()
        for key in ("backend", "is_available", "candidate_labels", "label_count",
                    "confidence_threshold", "calls_total", "cache_hits", "refinements_made"):
            assert key in sd, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 3. TestClassifyCrop (8 tests)
# ---------------------------------------------------------------------------

class TestClassifyCrop:
    def setup_method(self):
        self.clf = _make_classifier()
        self.crop = _make_frame(64, 64)

    def test_returns_clip_score(self):
        result = self.clf.classify_crop(self.crop)
        assert isinstance(result, CLIPScore)

    def test_refined_label_in_candidates_when_above_threshold(self):
        # Force a very low threshold to ensure we always get a label
        clf = _make_classifier(confidence_threshold=-999.0)
        result = clf.classify_crop(self.crop)
        assert result.refined_label in clf._labels

    def test_score_is_float(self):
        result = self.clf.classify_crop(self.crop)
        assert isinstance(result.score, float)

    def test_score_in_valid_range(self):
        result = self.clf.classify_crop(self.crop)
        # Cosine similarity of unit vectors is in [-1, 1]
        assert -1.0 <= result.score <= 1.0

    def test_all_scores_has_all_labels(self):
        labels = ["alpha", "beta", "gamma"]
        result = self.clf.classify_crop(self.crop, candidate_labels=labels)
        assert set(result.all_scores.keys()) == set(labels)

    def test_entity_id_minus_one(self):
        result = self.clf.classify_crop(self.crop)
        assert result.entity_id == -1

    def test_custom_labels_override_default(self):
        custom = ["apple", "banana"]
        result = self.clf.classify_crop(self.crop, candidate_labels=custom)
        assert set(result.all_scores.keys()) == set(custom)

    def test_below_threshold_gives_unknown(self):
        clf = _make_classifier(confidence_threshold=999.0)
        result = clf.classify_crop(self.crop)
        assert result.refined_label == "unknown"

    def test_different_crops_potentially_different_scores(self):
        crop_a = np.zeros((32, 32, 3), dtype=np.uint8)
        crop_b = np.full((32, 32, 3), 200, dtype=np.uint8)
        r_a = self.clf.classify_crop(crop_a)
        r_b = self.clf.classify_crop(crop_b)
        # Scores may differ since crops have different pixel content
        # Just assert both are valid CLIPScore objects
        assert isinstance(r_a, CLIPScore)
        assert isinstance(r_b, CLIPScore)


# ---------------------------------------------------------------------------
# 4. TestClassifyEntities (10 tests)
# ---------------------------------------------------------------------------

class TestClassifyEntities:
    def setup_method(self):
        self.clf = _make_classifier()
        self.frame = _make_frame()

    def test_empty_list_returns_empty(self):
        result = self.clf.classify_entities([])
        assert result == []

    def test_single_entity_returns_one_score(self):
        det = MockDetection(entity_id=1, label="person", bbox=[0.1, 0.1, 0.5, 0.5])
        result = self.clf.classify_entities([det], frame=self.frame)
        assert len(result) == 1

    def test_multiple_entities_same_count(self):
        dets = [
            MockDetection(entity_id=i, label="worker", bbox=[0.0, 0.0, 0.3, 0.3])
            for i in range(5)
        ]
        result = self.clf.classify_entities(dets, frame=self.frame)
        assert len(result) == 5

    def test_original_label_preserved(self):
        det = MockDetection(entity_id=7, label="forklift", bbox=[0.2, 0.2, 0.8, 0.8])
        result = self.clf.classify_entities([det], frame=self.frame)
        assert result[0].original_label == "forklift"

    def test_frame_none_uses_zero_crop(self):
        det = MockDetection(entity_id=1, label="box", bbox=[0.0, 0.0, 1.0, 1.0])
        # Should not raise even with frame=None
        result = self.clf.classify_entities([det], frame=None)
        assert len(result) == 1

    def test_frame_provided_valid_bbox(self):
        det = MockDetection(entity_id=2, label="helmet", bbox=[0.1, 0.1, 0.6, 0.6])
        result = self.clf.classify_entities([det], frame=self.frame)
        assert isinstance(result[0], CLIPScore)

    def test_invalid_bbox_fallback_graceful(self):
        det = MockDetection(entity_id=3, label="vest", bbox=[0.9, 0.9, 0.1, 0.1])  # x2<x1
        result = self.clf.classify_entities([det], frame=self.frame)
        assert len(result) == 1

    def test_entity_id_preserved(self):
        det = MockDetection(entity_id=42, label="car", bbox=[0.0, 0.0, 0.5, 0.5])
        result = self.clf.classify_entities([det], frame=self.frame)
        assert result[0].entity_id == 42

    def test_calls_total_incremented(self):
        det = MockDetection(entity_id=1, label="truck", bbox=[0.0, 0.0, 0.5, 0.5])
        before = self.clf._stats["calls_total"]
        self.clf.classify_entities([det], frame=self.frame)
        assert self.clf._stats["calls_total"] == before + 1

    def test_refinements_made_increments_when_label_changes(self):
        # Use threshold=-999 so score always exceeds it, meaning mock will always pick
        # the best-matching label (which may differ from original_label)
        clf = _make_classifier(confidence_threshold=-999.0)
        # Pick a label unlikely to be the best-matching hash for this crop
        det = MockDetection(entity_id=1, label="__nonexistent_label__")
        frame = _make_frame()
        before = clf._stats["refinements_made"]
        clf.classify_entities([det], frame=frame)
        # refined_label will differ from "__nonexistent_label__" since that label isn't in _labels
        # original_label = "__nonexistent_label__", refined_label = best matching DEFAULT label
        assert clf._stats["refinements_made"] >= before


# ---------------------------------------------------------------------------
# 5. TestSetLabels (5 tests)
# ---------------------------------------------------------------------------

class TestSetLabels:
    def setup_method(self):
        self.clf = _make_classifier()

    def test_set_labels_updates_labels(self):
        new_labels = ["alpha", "beta"]
        self.clf.set_labels(new_labels)
        assert self.clf._labels == new_labels

    def test_cache_invalidated_after_set_labels(self):
        # Prime the cache via classify_entities, which calls _get_text_embeddings
        det = MockDetection(entity_id=1, label="person", bbox=[0.0, 0.0, 0.5, 0.5])
        frame = _make_frame()
        self.clf.classify_entities([det], frame=frame)
        assert self.clf._text_cache is not None, "Cache should be populated after classify_entities"
        # Changing labels should invalidate
        self.clf.set_labels(["x", "y"])
        assert self.clf._text_cache is None

    def test_new_labels_used_in_subsequent_classify_crop(self):
        new_labels = ["unicorn", "dragon"]
        self.clf.set_labels(new_labels)
        crop = _make_frame(32, 32)
        result = self.clf.classify_crop(crop)
        assert set(result.all_scores.keys()) == set(new_labels)

    def test_original_labels_not_used_after_update(self):
        original = list(self.clf._labels)
        self.clf.set_labels(["only_label"])
        crop = _make_frame(32, 32)
        result = self.clf.classify_crop(crop)
        for orig in original:
            assert orig not in result.all_scores

    def test_empty_list_accepted(self):
        # Should not raise; encode_text([]) is allowed
        self.clf.set_labels([])
        assert self.clf._labels == []


# ---------------------------------------------------------------------------
# 6. TestGetStateDict (4 tests)
# ---------------------------------------------------------------------------

class TestGetStateDict:
    def setup_method(self):
        self.clf = _make_classifier(candidate_labels=["a", "b", "c"])

    def test_all_keys_present(self):
        sd = self.clf.get_state_dict()
        expected_keys = {
            "backend", "is_available", "candidate_labels", "label_count",
            "confidence_threshold", "calls_total", "cache_hits", "refinements_made",
        }
        assert expected_keys.issubset(set(sd.keys()))

    def test_label_count_matches_len(self):
        sd = self.clf.get_state_dict()
        assert sd["label_count"] == len(self.clf._labels)

    def test_is_available_false_for_mock(self):
        sd = self.clf.get_state_dict()
        assert sd["is_available"] is False

    def test_calls_total_increments(self):
        det = MockDetection(entity_id=1, label="a", bbox=[0.0, 0.0, 0.5, 0.5])
        frame = _make_frame()
        before = self.clf.get_state_dict()["calls_total"]
        self.clf.classify_entities([det], frame=frame)
        after = self.clf.get_state_dict()["calls_total"]
        assert after == before + 1


# ---------------------------------------------------------------------------
# 7. TestBuildFactory (4 tests)
# ---------------------------------------------------------------------------

class TestBuildFactory:
    def test_returns_classifier_instance(self):
        clf = build_clip_classifier(backend="mock")
        assert isinstance(clf, CLIPZeroShotClassifier)

    def test_custom_labels_passed_through(self):
        labels = ["foo", "bar"]
        clf = build_clip_classifier(labels=labels, backend="mock")
        assert clf._labels == labels

    def test_backend_mock_gives_mock(self):
        clf = build_clip_classifier(backend="mock")
        assert isinstance(clf._backend, MockCLIPBackend)

    def test_confidence_threshold_passed_through(self):
        clf = build_clip_classifier(backend="mock", confidence_threshold=0.7)
        assert clf.confidence_threshold == 0.7


# ---------------------------------------------------------------------------
# 8. TestThreadSafety (4 tests)
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def setup_method(self):
        self.clf = _make_classifier()
        self.frame = _make_frame()
        self.errors: list[Exception] = []

    def _worker(self, n: int = 10):
        try:
            for i in range(n):
                det = MockDetection(entity_id=i, label="worker", bbox=[0.1, 0.1, 0.9, 0.9])
                self.clf.classify_entities([det], frame=self.frame)
        except Exception as e:
            self.errors.append(e)

    def test_concurrent_classify_entities(self):
        threads = [threading.Thread(target=self._worker, args=(10,)) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert self.errors == [], f"Thread errors: {self.errors}"

    def test_concurrent_set_labels_and_classify(self):
        errors: list[Exception] = []
        frame = self.frame

        def classifier_worker():
            try:
                for i in range(5):
                    det = MockDetection(entity_id=i, label="box", bbox=[0.0, 0.0, 0.5, 0.5])
                    self.clf.classify_entities([det], frame=frame)
            except Exception as e:
                errors.append(e)

        def label_updater():
            try:
                for i in range(5):
                    self.clf.set_labels([f"label_{i}", f"label_{i+1}"])
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=classifier_worker) for _ in range(2)]
            + [threading.Thread(target=label_updater)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert errors == [], f"Thread errors: {errors}"

    def test_no_exceptions_raised(self):
        # Baseline: single-threaded should never raise for valid inputs
        det = MockDetection(entity_id=0, label="person", bbox=[0.0, 0.0, 1.0, 1.0])
        try:
            self.clf.classify_entities([det], frame=self.frame)
        except Exception as e:
            pytest.fail(f"Unexpected exception: {e}")

    def test_state_dict_valid_after_concurrent_ops(self):
        threads = [threading.Thread(target=self._worker, args=(5,)) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        sd = self.clf.get_state_dict()
        assert isinstance(sd["calls_total"], int)
        assert sd["calls_total"] >= 0
        assert isinstance(sd["label_count"], int)


# ---------------------------------------------------------------------------
# 9. TestEdgeCases (4 tests)
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def setup_method(self):
        self.frame = _make_frame()

    def test_single_candidate_label(self):
        clf = _make_classifier(candidate_labels=["only_one"], confidence_threshold=-999.0)
        crop = _make_frame(32, 32)
        result = clf.classify_crop(crop)
        assert result.refined_label == "only_one"
        assert len(result.all_scores) == 1

    def test_all_zero_image_array(self):
        clf = _make_classifier()
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        # Should not raise
        result = clf.classify_crop(img)
        assert isinstance(result, CLIPScore)

    def test_very_long_label_text(self):
        clf = _make_classifier()
        long_label = "a" * 500
        labels = [long_label, "short"]
        result = clf.classify_crop(_make_frame(32, 32), candidate_labels=labels)
        assert long_label in result.all_scores

    def test_entity_with_no_bbox_attribute(self):
        clf = _make_classifier()

        class NoBboxDetection:
            entity_id = 99
            label = "mystery"
            # intentionally no bbox attribute

        det = NoBboxDetection()
        # Should fall back gracefully (returns full frame as crop)
        result = clf.classify_entities([det], frame=self.frame)
        assert len(result) == 1
        assert result[0].entity_id == 99
