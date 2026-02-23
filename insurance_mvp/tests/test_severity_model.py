"""Tests for insurance_mvp.pipeline.stages.severity_model module.

Covers IsotonicCalibrator, PlattScaling, SeverityClassifier,
CalibratedSeverityModel, SeverityFeatures, and edge cases.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from insurance_mvp.pipeline.stages.severity_model import (
    NUM_CLASSES,
    SEVERITY_LABELS,
    CalibratedSeverityModel,
    IsotonicCalibrator,
    PlattScaling,
    SeverityClassifier,
    SeverityFeatures,
    SeverityPrediction,
    _sigmoid,
    _softmax,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic data generators
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Deterministic numpy random generator."""
    return np.random.default_rng(12345)


@pytest.fixture
def synthetic_binary_data(rng):
    """Generate 200 samples with scores in [0,1] and binary labels.

    Higher scores correlate with label=1 so a well-fitted calibrator
    should produce monotonically increasing calibrated values.
    """
    n = 200
    scores = rng.uniform(0, 1, n)
    # label=1 more likely when score is high
    labels = (scores + rng.normal(0, 0.15, n) > 0.5).astype(np.float64)
    labels = np.clip(labels, 0, 1)
    return scores, labels


@pytest.fixture
def synthetic_severity_data(rng):
    """Generate 200 samples of (X, y) for 4-class severity classification.

    Features are 5-D (matching SeverityFeatures.to_array output).
    Labels correlate with the average feature magnitude so the
    classifier can learn a meaningful decision boundary.
    """
    n = 200
    X = rng.uniform(0, 1, (n, 5))
    # Assign severity label based on mean feature magnitude + noise
    avg = X.mean(axis=1) + rng.normal(0, 0.05, n)
    y = np.digitize(avg, bins=[0.3, 0.5, 0.7]) # yields 0,1,2,3
    y = np.clip(y, 0, NUM_CLASSES - 1).astype(np.int64)
    return X, y


@pytest.fixture
def synthetic_features_and_labels(rng):
    """Generate list[SeverityFeatures] + labels for CalibratedSeverityModel."""
    n = 100
    features_list = []
    labels = []
    for _ in range(n):
        sf = SeverityFeatures(
            danger_score=rng.uniform(0, 1),
            motion_score=rng.uniform(0, 1),
            proximity_score=rng.uniform(0, 1),
            vlm_confidence=rng.uniform(0, 1),
            vlm_severity_idx=int(rng.integers(0, NUM_CLASSES)),
        )
        features_list.append(sf)
        # Label correlates with danger_score
        avg = (sf.danger_score + sf.motion_score + sf.proximity_score) / 3
        label = min(int(avg * NUM_CLASSES), NUM_CLASSES - 1)
        labels.append(label)
    return features_list, labels


# ---------------------------------------------------------------------------
# SeverityFeatures
# ---------------------------------------------------------------------------

class TestSeverityFeatures:
    def test_to_array_shape(self):
        sf = SeverityFeatures(
            danger_score=0.8, motion_score=0.6,
            proximity_score=0.4, vlm_confidence=0.9, vlm_severity_idx=3,
        )
        arr = sf.to_array()
        assert arr.shape == (5,)
        assert arr.dtype == np.float64

    def test_to_array_normalizes_severity_idx(self):
        sf = SeverityFeatures(vlm_severity_idx=3)
        arr = sf.to_array()
        # 3 / (4-1) = 1.0
        assert arr[4] == pytest.approx(1.0)

        sf0 = SeverityFeatures(vlm_severity_idx=0)
        assert sf0.to_array()[4] == pytest.approx(0.0)

    def test_defaults_are_zero(self):
        sf = SeverityFeatures()
        arr = sf.to_array()
        np.testing.assert_array_equal(arr, np.zeros(5))


# ---------------------------------------------------------------------------
# IsotonicCalibrator
# ---------------------------------------------------------------------------

class TestIsotonicCalibrator:
    def test_fit_transform_basic(self, synthetic_binary_data):
        scores, labels = synthetic_binary_data
        cal = IsotonicCalibrator()
        cal.fit(scores, labels)

        calibrated = cal.transform(scores)
        assert calibrated.shape == scores.shape
        # All values in [0, 1]
        assert np.all(calibrated >= 0.0)
        assert np.all(calibrated <= 1.0)

    def test_monotonicity_guarantee(self, synthetic_binary_data):
        """Fitted isotonic values at breakpoints must be non-decreasing."""
        scores, labels = synthetic_binary_data
        cal = IsotonicCalibrator()
        cal.fit(scores, labels)

        # The breakpoint y_ values must be non-decreasing
        diffs = np.diff(cal.y_)
        assert np.all(diffs >= -1e-12), (
            f"Isotonic breakpoints violate monotonicity: {cal.y_}"
        )

    def test_transform_monotone_on_sorted_input(self, synthetic_binary_data):
        """transform applied to sorted scores should produce non-decreasing output."""
        scores, labels = synthetic_binary_data
        cal = IsotonicCalibrator().fit(scores, labels)

        sorted_scores = np.sort(scores)
        calibrated = cal.transform(sorted_scores)
        diffs = np.diff(calibrated)
        assert np.all(diffs >= -1e-12)

    def test_edge_case_single_point_raises(self):
        """fit must raise ValueError if fewer than 2 samples."""
        cal = IsotonicCalibrator()
        with pytest.raises(ValueError, match="at least 2"):
            cal.fit(np.array([0.5]), np.array([1.0]))

    def test_edge_case_all_same_score(self):
        """All identical scores -- should still fit without error."""
        cal = IsotonicCalibrator()
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([0, 1, 1, 0])
        cal.fit(scores, labels)
        result = cal.transform(np.array([0.5]))
        assert 0.0 <= result[0] <= 1.0

    def test_edge_case_all_same_label(self):
        """All labels identical -- calibrated output should be constant."""
        cal = IsotonicCalibrator()
        scores = np.linspace(0, 1, 20)
        labels = np.ones(20)
        cal.fit(scores, labels)
        calibrated = cal.transform(scores)
        # All should be close to 1.0
        np.testing.assert_allclose(calibrated, 1.0, atol=1e-10)

    def test_transform_before_fit_raises(self):
        cal = IsotonicCalibrator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            cal.transform(np.array([0.5]))

    def test_length_mismatch_raises(self):
        cal = IsotonicCalibrator()
        with pytest.raises(ValueError, match="same length"):
            cal.fit(np.array([0.1, 0.2, 0.3]), np.array([0, 1]))

    def test_serialization_roundtrip(self, synthetic_binary_data):
        scores, labels = synthetic_binary_data
        cal = IsotonicCalibrator().fit(scores, labels)

        d = cal.to_dict()
        assert d["type"] == "isotonic"
        assert d["fitted"] is True

        cal2 = IsotonicCalibrator.from_dict(d)
        assert cal2._fitted is True
        np.testing.assert_array_equal(cal2.x_, cal.x_)
        np.testing.assert_array_equal(cal2.y_, cal.y_)

        # Same transform results
        test_scores = np.array([0.1, 0.5, 0.9])
        np.testing.assert_array_almost_equal(
            cal.transform(test_scores), cal2.transform(test_scores)
        )

    def test_serialization_unfitted(self):
        cal = IsotonicCalibrator()
        d = cal.to_dict()
        assert d["fitted"] is False

        cal2 = IsotonicCalibrator.from_dict(d)
        assert cal2._fitted is False

    def test_pav_empty(self):
        result = IsotonicCalibrator._pav(np.array([]), np.array([]))
        assert len(result) == 0

    def test_pav_already_monotone(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones(4)
        result = IsotonicCalibrator._pav(y, w)
        np.testing.assert_array_almost_equal(result, y)

    def test_pav_reverse_order(self):
        """Strictly decreasing input should yield a constant (grand mean)."""
        y = np.array([4.0, 3.0, 2.0, 1.0])
        w = np.ones(4)
        result = IsotonicCalibrator._pav(y, w)
        expected_mean = y.mean()
        np.testing.assert_array_almost_equal(result, expected_mean)


# ---------------------------------------------------------------------------
# PlattScaling
# ---------------------------------------------------------------------------

class TestPlattScaling:
    def test_fit_transform_basic(self, synthetic_binary_data):
        scores, labels = synthetic_binary_data
        ps = PlattScaling()
        ps.fit(scores, labels)

        calibrated = ps.transform(scores)
        assert calibrated.shape == scores.shape
        assert np.all(calibrated >= 0.0)
        assert np.all(calibrated <= 1.0)

    def test_output_is_valid_probability(self, synthetic_binary_data):
        scores, labels = synthetic_binary_data
        ps = PlattScaling().fit(scores, labels)

        test_scores = np.linspace(-5, 5, 100)
        result = ps.transform(test_scores)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_sigmoid_shape(self, synthetic_binary_data):
        """Output should be monotone in one direction (sigmoid is monotone)."""
        scores, labels = synthetic_binary_data
        ps = PlattScaling().fit(scores, labels)

        test_scores = np.linspace(0, 1, 50)
        result = ps.transform(test_scores)
        diffs = np.diff(result)
        # All diffs should be same sign (monotone increasing or decreasing)
        if len(diffs) > 0:
            # Platt sigmoid: P = 1/(1+exp(Af+B)). If A < 0 -> increasing
            assert np.all(diffs >= -1e-10) or np.all(diffs <= 1e-10)

    def test_edge_case_single_sample_raises(self):
        ps = PlattScaling()
        with pytest.raises(ValueError, match="at least 2"):
            ps.fit(np.array([0.5]), np.array([1.0]))

    def test_edge_case_length_mismatch_raises(self):
        ps = PlattScaling()
        with pytest.raises(ValueError, match="same length"):
            ps.fit(np.array([0.1, 0.2]), np.array([1]))

    def test_transform_before_fit_raises(self):
        ps = PlattScaling()
        with pytest.raises(RuntimeError, match="not been fitted"):
            ps.transform(np.array([0.5]))

    def test_serialization_roundtrip(self, synthetic_binary_data):
        scores, labels = synthetic_binary_data
        ps = PlattScaling().fit(scores, labels)

        d = ps.to_dict()
        assert d["type"] == "platt"
        assert d["fitted"] is True
        assert isinstance(d["A"], float)
        assert isinstance(d["B"], float)

        ps2 = PlattScaling.from_dict(d)
        assert ps2._fitted is True
        assert ps2.A == pytest.approx(ps.A)
        assert ps2.B == pytest.approx(ps.B)

        test_scores = np.array([0.1, 0.5, 0.9])
        np.testing.assert_array_almost_equal(
            ps.transform(test_scores), ps2.transform(test_scores)
        )

    def test_serialization_unfitted(self):
        ps = PlattScaling()
        d = ps.to_dict()
        assert d["fitted"] is False

        ps2 = PlattScaling.from_dict(d)
        assert ps2._fitted is False

    def test_extreme_scores(self, synthetic_binary_data):
        """Extreme input scores should not cause NaN or inf."""
        scores, labels = synthetic_binary_data
        ps = PlattScaling().fit(scores, labels)

        extreme = np.array([-1000.0, -100.0, 0.0, 100.0, 1000.0])
        result = ps.transform(extreme)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


# ---------------------------------------------------------------------------
# SeverityClassifier
# ---------------------------------------------------------------------------

class TestSeverityClassifier:
    def test_fit_basic(self, synthetic_severity_data):
        X, y = synthetic_severity_data
        clf = SeverityClassifier(max_iter=500)
        clf.fit(X, y)
        assert clf._fitted is True
        assert clf.W.shape == (5, NUM_CLASSES)
        assert clf.b.shape == (NUM_CLASSES,)

    def test_predict_proba_sums_to_one(self, synthetic_severity_data):
        X, y = synthetic_severity_data
        clf = SeverityClassifier(max_iter=500).fit(X, y)

        probs = clf.predict_proba(X)
        assert probs.shape == (len(X), NUM_CLASSES)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_predict_proba_non_negative(self, synthetic_severity_data):
        X, y = synthetic_severity_data
        clf = SeverityClassifier(max_iter=500).fit(X, y)
        probs = clf.predict_proba(X)
        assert np.all(probs >= 0.0)

    def test_predict_proba_single_sample(self, synthetic_severity_data):
        X, y = synthetic_severity_data
        clf = SeverityClassifier(max_iter=500).fit(X, y)

        single = X[0]
        probs = clf.predict_proba(single)
        # Single sample returns 1-D
        assert probs.ndim == 1
        assert probs.shape == (NUM_CLASSES,)
        assert pytest.approx(probs.sum(), abs=1e-6) == 1.0

    def test_predict_returns_valid_labels(self, synthetic_severity_data):
        X, y = synthetic_severity_data
        clf = SeverityClassifier(max_iter=500).fit(X, y)

        preds = clf.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(preds >= 0)
        assert np.all(preds < NUM_CLASSES)

    def test_predict_single_sample(self, synthetic_severity_data):
        X, y = synthetic_severity_data
        clf = SeverityClassifier(max_iter=500).fit(X, y)
        pred = clf.predict(X[0])
        assert isinstance(pred, int)
        assert 0 <= pred < NUM_CLASSES

    def test_predict_before_fit_raises(self):
        clf = SeverityClassifier()
        with pytest.raises(RuntimeError, match="not been fitted"):
            clf.predict_proba(np.zeros((1, 5)))

    def test_invalid_X_shape_raises(self):
        clf = SeverityClassifier()
        with pytest.raises(ValueError, match="2-D"):
            clf.fit(np.zeros(10), np.zeros(10, dtype=np.int64))

    def test_invalid_label_range_raises(self):
        clf = SeverityClassifier()
        X = np.zeros((10, 5))
        y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])  # 4 is out of range
        with pytest.raises(ValueError, match="Labels must be"):
            clf.fit(X, y)

    def test_Xy_length_mismatch_raises(self):
        clf = SeverityClassifier()
        with pytest.raises(ValueError, match="same number"):
            clf.fit(np.zeros((10, 5)), np.zeros(8, dtype=np.int64))

    def test_ordinal_enforcement_on(self, synthetic_severity_data):
        X, y = synthetic_severity_data
        clf = SeverityClassifier(enforce_ordinal=True, max_iter=500).fit(X, y)
        probs = clf.predict_proba(X)

        # After ordinal enforcement, CDF should be non-decreasing for each sample
        for i in range(len(X)):
            cdf = np.cumsum(probs[i])
            diffs = np.diff(cdf)
            assert np.all(diffs >= -1e-10), (
                f"CDF not non-decreasing at sample {i}: {cdf}"
            )

    def test_ordinal_enforcement_off(self, synthetic_severity_data):
        """Without ordinal enforcement, model still produces valid probabilities."""
        X, y = synthetic_severity_data
        clf = SeverityClassifier(enforce_ordinal=False, max_iter=500).fit(X, y)
        probs = clf.predict_proba(X)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_serialization_roundtrip(self, synthetic_severity_data):
        X, y = synthetic_severity_data
        clf = SeverityClassifier(max_iter=300).fit(X, y)

        d = clf.to_dict()
        assert d["fitted"] is True
        assert d["n_features"] == 5

        clf2 = SeverityClassifier.from_dict(d)
        assert clf2._fitted is True
        np.testing.assert_array_almost_equal(clf2.W, clf.W)
        np.testing.assert_array_almost_equal(clf2.b, clf.b)

        # Predictions should match exactly
        probs1 = clf.predict_proba(X[:5])
        probs2 = clf2.predict_proba(X[:5])
        np.testing.assert_array_almost_equal(probs1, probs2)

    def test_all_same_class(self, rng):
        """Training where all samples belong to the same class."""
        n = 50
        X = rng.uniform(0, 1, (n, 5))
        y = np.full(n, 2, dtype=np.int64)  # all MEDIUM
        clf = SeverityClassifier(max_iter=500).fit(X, y)

        probs = clf.predict_proba(X)
        # Model should strongly predict class 2 for all samples
        preds = np.argmax(probs, axis=1)
        assert np.all(preds == 2)

    def test_very_small_dataset(self, rng):
        """Classifier should handle very small datasets (n=4, one per class)."""
        X = rng.uniform(0, 1, (4, 5))
        y = np.array([0, 1, 2, 3])
        clf = SeverityClassifier(max_iter=1000).fit(X, y)
        assert clf._fitted is True
        probs = clf.predict_proba(X)
        assert probs.shape == (4, NUM_CLASSES)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# CalibratedSeverityModel (end-to-end)
# ---------------------------------------------------------------------------

class TestCalibratedSeverityModel:
    def test_fit_and_predict_isotonic(self, synthetic_features_and_labels):
        features, labels = synthetic_features_and_labels
        model = CalibratedSeverityModel(calibration_method="isotonic")
        model.fit_from_data(features, labels)

        pred = model.predict(features[0])
        assert isinstance(pred, SeverityPrediction)
        assert pred.severity in SEVERITY_LABELS
        assert 0.0 <= pred.confidence <= 1.0
        assert pred.severity_idx == SEVERITY_LABELS.index(pred.severity)

        # probabilities dict
        assert set(pred.probabilities.keys()) == set(SEVERITY_LABELS)
        prob_sum = sum(pred.probabilities.values())
        assert pytest.approx(prob_sum, abs=1e-4) == 1.0

    def test_fit_and_predict_platt(self, synthetic_features_and_labels):
        features, labels = synthetic_features_and_labels
        model = CalibratedSeverityModel(calibration_method="platt")
        model.fit_from_data(features, labels)

        pred = model.predict(features[0])
        assert pred.severity in SEVERITY_LABELS
        assert 0.0 <= pred.confidence <= 1.0

    def test_predict_batch(self, synthetic_features_and_labels):
        features, labels = synthetic_features_and_labels
        model = CalibratedSeverityModel(calibration_method="isotonic")
        model.fit_from_data(features, labels)

        batch = features[:10]
        preds = model.predict_batch(batch)
        assert len(preds) == 10
        for p in preds:
            assert isinstance(p, SeverityPrediction)
            assert p.severity in SEVERITY_LABELS

    def test_invalid_calibration_method_raises(self):
        with pytest.raises(ValueError, match="calibration_method"):
            CalibratedSeverityModel(calibration_method="unknown")

    def test_save_and_load(self, synthetic_features_and_labels, tmp_path):
        features, labels = synthetic_features_and_labels
        model = CalibratedSeverityModel(calibration_method="isotonic")
        model.fit_from_data(features, labels)

        save_path = str(tmp_path / "severity_model.json")
        model.save(save_path)

        # File should exist and be valid JSON
        with open(save_path, encoding="utf-8") as f:
            state = json.load(f)
        assert state["version"] == 1
        assert state["calibration_method"] == "isotonic"

        # Load and verify predictions match
        model2 = CalibratedSeverityModel.load(save_path)

        test_features = features[0]
        pred1 = model.predict(test_features)
        pred2 = model2.predict(test_features)

        assert pred1.severity == pred2.severity
        assert pred1.severity_idx == pred2.severity_idx
        assert pred1.confidence == pytest.approx(pred2.confidence, abs=1e-10)

    def test_save_load_platt(self, synthetic_features_and_labels, tmp_path):
        features, labels = synthetic_features_and_labels
        model = CalibratedSeverityModel(calibration_method="platt")
        model.fit_from_data(features, labels)

        save_path = str(tmp_path / "severity_platt.json")
        model.save(save_path)
        model2 = CalibratedSeverityModel.load(save_path)

        pred1 = model.predict(features[5])
        pred2 = model2.predict(features[5])
        assert pred1.severity == pred2.severity
        assert pred1.confidence == pytest.approx(pred2.confidence, abs=1e-10)

    def test_to_dict_from_dict_roundtrip(self, synthetic_features_and_labels):
        features, labels = synthetic_features_and_labels
        model = CalibratedSeverityModel(calibration_method="isotonic")
        model.fit_from_data(features, labels)

        d = model.to_dict()
        model2 = CalibratedSeverityModel.from_dict(d)

        pred1 = model.predict(features[0])
        pred2 = model2.predict(features[0])
        assert pred1.severity == pred2.severity
        assert pred1.confidence == pytest.approx(pred2.confidence, abs=1e-10)

    def test_very_small_dataset(self, rng):
        """With < 10 samples, model should warn but still train."""
        features = [
            SeverityFeatures(
                danger_score=rng.uniform(0, 1),
                motion_score=rng.uniform(0, 1),
                proximity_score=rng.uniform(0, 1),
                vlm_confidence=rng.uniform(0, 1),
                vlm_severity_idx=int(rng.integers(0, NUM_CLASSES)),
            )
            for _ in range(6)
        ]
        labels = [0, 1, 2, 3, 1, 2]

        model = CalibratedSeverityModel(calibration_method="isotonic")
        model.fit_from_data(features, labels)

        pred = model.predict(features[0])
        assert pred.severity in SEVERITY_LABELS

    def test_extreme_feature_values(self, synthetic_features_and_labels):
        """Predict with extreme feature values should not produce NaN/inf."""
        features, labels = synthetic_features_and_labels
        model = CalibratedSeverityModel(calibration_method="isotonic")
        model.fit_from_data(features, labels)

        extreme = SeverityFeatures(
            danger_score=1.0,
            motion_score=1.0,
            proximity_score=1.0,
            vlm_confidence=1.0,
            vlm_severity_idx=3,
        )
        pred = model.predict(extreme)
        assert not np.isnan(pred.confidence)
        assert pred.severity in SEVERITY_LABELS

        zero = SeverityFeatures()
        pred_zero = model.predict(zero)
        assert not np.isnan(pred_zero.confidence)
        assert pred_zero.severity in SEVERITY_LABELS

    def test_classifier_kwargs_forwarded(self):
        """Constructor keyword args should be forwarded to SeverityClassifier."""
        model = CalibratedSeverityModel(
            calibration_method="platt",
            learning_rate=0.05,
            l2_lambda=0.001,
            max_iter=200,
        )
        assert model.classifier.learning_rate == 0.05
        assert model.classifier.l2_lambda == 0.001
        assert model.classifier.max_iter == 200


# ---------------------------------------------------------------------------
# Internal utilities: _sigmoid, _softmax
# ---------------------------------------------------------------------------

class TestInternalUtilities:
    def test_sigmoid_output_range(self):
        z = np.linspace(-10, 10, 100)
        result = _sigmoid(z)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_sigmoid_known_values(self):
        assert _sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)
        # sigmoid(large) -> 1, sigmoid(-large) -> 0
        assert _sigmoid(np.array([100.0]))[0] == pytest.approx(1.0, abs=1e-10)
        assert _sigmoid(np.array([-100.0]))[0] == pytest.approx(0.0, abs=1e-10)

    def test_sigmoid_no_nan_on_extreme(self):
        extreme = np.array([-1e6, -1e3, 0, 1e3, 1e6])
        result = _sigmoid(extreme)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_softmax_sums_to_one(self):
        logits = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]])
        result = _softmax(logits)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-10)

    def test_softmax_1d_input(self):
        logits = np.array([1.0, 2.0, 3.0])
        result = _softmax(logits)
        assert result.shape == (1, 3)
        assert pytest.approx(result.sum(), abs=1e-10) == 1.0

    def test_softmax_large_logits_no_overflow(self):
        logits = np.array([[1000.0, 1001.0, 1002.0, 1003.0]])
        result = _softmax(logits)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert pytest.approx(result.sum(), abs=1e-10) == 1.0
