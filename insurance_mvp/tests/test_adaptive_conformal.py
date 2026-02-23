"""Tests for insurance_mvp.conformal.adaptive module.

Covers:
- generate_calibration_dataset: shape, probabilities, seed determinism
- AdaptiveConformal: fit, predict_set, update cycle, coverage tracking, alpha adaptation
- MondrianConformal: fit, predict_set, per-group coverage
- CoverageMonitor: update, alarm detection, empirical_coverage, should_recalibrate
"""

import numpy as np
import pytest

from insurance_mvp.conformal.adaptive import (
    AdaptiveConformal,
    CoverageMonitor,
    MondrianConformal,
    generate_calibration_dataset,
)


# ---------------------------------------------------------------------------
# generate_calibration_dataset
# ---------------------------------------------------------------------------


class TestGenerateCalibrationDataset:
    """Tests for the synthetic calibration data generator."""

    def test_default_shape(self):
        """Default call returns (500, 4) scores and (500,) labels."""
        scores, y_true = generate_calibration_dataset()
        assert scores.shape == (500, 4)
        assert y_true.shape == (500,)

    def test_custom_shape(self):
        """Custom n_samples and n_classes are respected."""
        scores, y_true = generate_calibration_dataset(n_samples=200, n_classes=6)
        assert scores.shape == (200, 6)
        assert y_true.shape == (200,)

    def test_scores_are_valid_probabilities(self):
        """Each row sums to 1 and all values are in [0, 1]."""
        scores, _ = generate_calibration_dataset(n_samples=300)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)
        np.testing.assert_allclose(scores.sum(axis=1), 1.0, atol=1e-6)

    def test_labels_in_valid_range(self):
        """All labels are valid class indices."""
        _, y_true = generate_calibration_dataset(n_samples=300, n_classes=4)
        assert np.all(y_true >= 0)
        assert np.all(y_true < 4)

    def test_seed_determinism(self):
        """Same seed produces identical outputs."""
        s1, y1 = generate_calibration_dataset(seed=99)
        s2, y2 = generate_calibration_dataset(seed=99)
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        """Different seeds produce different outputs."""
        s1, y1 = generate_calibration_dataset(seed=1)
        s2, y2 = generate_calibration_dataset(seed=2)
        assert not np.array_equal(s1, s2)

    def test_class_priors_respected(self):
        """When class_priors heavily favour class 0, most labels should be 0."""
        priors = np.array([0.9, 0.05, 0.025, 0.025])
        _, y_true = generate_calibration_dataset(n_samples=1000, class_priors=priors, seed=42)
        class_0_frac = (y_true == 0).mean()
        # With 1000 samples and 0.9 prior, expect > 80% class-0
        assert class_0_frac > 0.80


# ---------------------------------------------------------------------------
# AdaptiveConformal
# ---------------------------------------------------------------------------


class TestAdaptiveConformal:
    """Tests for the ACI (Adaptive Conformal Inference) class."""

    @pytest.fixture
    def calibration_data(self):
        """Generate a small calibration dataset."""
        return generate_calibration_dataset(n_samples=200, seed=42)

    @pytest.fixture
    def fitted_aci(self, calibration_data):
        """Return an ACI instance fitted on calibration data."""
        scores, y_true = calibration_data
        aci = AdaptiveConformal(alpha=0.1, gamma=0.01)
        aci.fit(scores, y_true)
        return aci

    def test_predict_set_before_fit_raises(self):
        """predict_set raises RuntimeError if fit() not called."""
        aci = AdaptiveConformal()
        test_scores = np.array([[0.1, 0.2, 0.3, 0.4]])
        with pytest.raises(RuntimeError, match="Not calibrated"):
            aci.predict_set(test_scores)

    def test_fit_stores_calibration_scores(self, calibration_data):
        """After fit(), internal calibration scores are populated."""
        scores, y_true = calibration_data
        aci = AdaptiveConformal()
        aci.fit(scores, y_true)
        assert aci._calibration_scores is not None
        assert len(aci._calibration_scores) == len(scores)

    def test_predict_set_returns_nonempty_sets(self, fitted_aci):
        """Every prediction set contains at least one label."""
        test_scores = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.1, 0.1, 0.7],
        ])
        pred_sets = fitted_aci.predict_set(test_scores)
        assert len(pred_sets) == 2
        for ps in pred_sets:
            assert isinstance(ps, set)
            assert len(ps) >= 1

    def test_predict_set_labels_are_severity_levels(self, fitted_aci):
        """Prediction set labels come from SEVERITY_LEVELS."""
        valid_labels = {"NONE", "LOW", "MEDIUM", "HIGH"}
        test_scores = np.array([[0.6, 0.2, 0.1, 0.1]])
        pred_sets = fitted_aci.predict_set(test_scores)
        for ps in pred_sets:
            assert ps.issubset(valid_labels)

    def test_predict_set_single(self, fitted_aci):
        """predict_set_single returns a single set, not a list."""
        test_scores = np.array([0.7, 0.1, 0.1, 0.1])
        ps = fitted_aci.predict_set_single(test_scores)
        assert isinstance(ps, set)
        assert len(ps) >= 1

    def test_update_cycle_tracks_coverage(self, fitted_aci):
        """After several update() calls, coverage_history is populated."""
        test_scores = np.array([[0.8, 0.1, 0.05, 0.05]])
        ps = fitted_aci.predict_set(test_scores)[0]

        # Update with true_label=0 ("NONE") which should be in the set
        fitted_aci.update(true_label=0, prediction_set=ps)
        assert fitted_aci.state.t == 1
        assert len(fitted_aci.state.coverage_history) == 1

    def test_update_adapts_alpha(self, fitted_aci):
        """Alpha changes after update() reflecting coverage feedback."""
        initial_alpha = fitted_aci.state.alpha_t

        # Simulate a miss: true label not in prediction set
        fitted_aci.update(true_label=0, prediction_set={"HIGH"})

        # After a miss, alpha should decrease (err_t=1, so alpha_t += gamma*(target - 1))
        assert fitted_aci.state.alpha_t < initial_alpha

    def test_running_coverage_after_updates(self, fitted_aci):
        """running_coverage reflects the fraction of correct predictions."""
        # Simulate 10 hits and 0 misses
        for _ in range(10):
            fitted_aci.update(true_label=0, prediction_set={"NONE", "LOW"})
        assert fitted_aci.running_coverage == 1.0

        # Now add 10 misses
        for _ in range(10):
            fitted_aci.update(true_label=0, prediction_set={"HIGH"})
        assert fitted_aci.running_coverage == pytest.approx(0.5)

    def test_mean_set_size_empty(self):
        """mean_set_size is 0.0 when no updates have occurred."""
        aci = AdaptiveConformal()
        assert aci.mean_set_size == 0.0

    def test_mean_set_size_after_updates(self, fitted_aci):
        """mean_set_size reflects average prediction set cardinality."""
        fitted_aci.update(0, {"NONE"})          # size 1
        fitted_aci.update(1, {"LOW", "MEDIUM"})  # size 2
        assert fitted_aci.mean_set_size == pytest.approx(1.5)

    def test_alpha_clamped_to_valid_range(self, fitted_aci):
        """Alpha stays in [0.001, 0.999] even under extreme updates."""
        # Many consecutive misses push alpha down
        for _ in range(500):
            fitted_aci.update(true_label=0, prediction_set={"HIGH"})
        assert fitted_aci.state.alpha_t >= 0.001

        # Many consecutive hits push alpha up
        for _ in range(500):
            fitted_aci.update(true_label=0, prediction_set={"NONE"})
        assert fitted_aci.state.alpha_t <= 0.999


# ---------------------------------------------------------------------------
# MondrianConformal
# ---------------------------------------------------------------------------


class TestMondrianConformal:
    """Tests for Mondrian (group-conditional) conformal prediction."""

    @pytest.fixture
    def calibration_data(self):
        """Generate calibration data with known group structure."""
        return generate_calibration_dataset(n_samples=400, seed=7)

    @pytest.fixture
    def fitted_mondrian(self, calibration_data):
        """Return a Mondrian conformal instance fitted on calibration data."""
        scores, y_true = calibration_data
        mc = MondrianConformal(alpha=0.1)
        mc.fit(scores, y_true)
        return mc

    def test_predict_set_before_fit_raises(self):
        """predict_set raises RuntimeError if fit() not called."""
        mc = MondrianConformal()
        with pytest.raises(RuntimeError, match="Not calibrated"):
            mc.predict_set(np.array([[0.25, 0.25, 0.25, 0.25]]))

    def test_fit_marks_calibrated(self, calibration_data):
        """After fit(), the instance is marked as calibrated."""
        scores, y_true = calibration_data
        mc = MondrianConformal()
        assert not mc._calibrated
        mc.fit(scores, y_true)
        assert mc._calibrated

    def test_predict_set_returns_nonempty(self, fitted_mondrian):
        """Each prediction set has at least one label."""
        test_scores = np.array([
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.7],
        ])
        pred_sets = fitted_mondrian.predict_set(test_scores)
        assert len(pred_sets) == 2
        for ps in pred_sets:
            assert len(ps) >= 1

    def test_per_group_coverage_keys(self, fitted_mondrian, calibration_data):
        """compute_group_coverage returns a dict with severity-level keys."""
        scores, y_true = calibration_data
        coverages = fitted_mondrian.compute_group_coverage(scores, y_true)
        # Each group key should be a severity level
        valid_labels = {"NONE", "LOW", "MEDIUM", "HIGH"}
        assert all(k in valid_labels for k in coverages.keys())

    def test_per_group_coverage_near_target(self, calibration_data):
        """Per-group coverage should be approximately >= 1 - alpha on each group."""
        scores, y_true = calibration_data
        mc = MondrianConformal(alpha=0.1)
        mc.fit(scores, y_true)
        coverages = mc.compute_group_coverage(scores, y_true)
        for group, cov in coverages.items():
            # Allow some slack since calibration and test set overlap
            assert cov >= 0.80, f"Group {group} coverage {cov:.3f} is too low"

    def test_explicit_groups(self, calibration_data):
        """fit() and predict_set() accept explicit group assignments."""
        scores, y_true = calibration_data
        # Use binary groups: even/odd index
        groups = np.array([i % 2 for i in range(len(y_true))])
        mc = MondrianConformal(alpha=0.1, labels=["NONE", "LOW", "MEDIUM", "HIGH"])
        mc.fit(scores, y_true, groups=groups)
        pred_sets = mc.predict_set(scores, groups=groups)
        assert len(pred_sets) == len(scores)


# ---------------------------------------------------------------------------
# CoverageMonitor
# ---------------------------------------------------------------------------


class TestCoverageMonitor:
    """Tests for the CUSUM-based coverage monitor."""

    def test_initial_state(self):
        """Monitor starts with zero observations and no alarm."""
        mon = CoverageMonitor()
        assert mon.n_observations == 0
        assert mon.n_covered == 0
        assert not mon.alarm_triggered
        assert mon.empirical_coverage == 0.0

    def test_update_increments_counts(self):
        """Each update() call increments observation count."""
        mon = CoverageMonitor()
        mon.update(True)
        assert mon.n_observations == 1
        assert mon.n_covered == 1
        mon.update(False)
        assert mon.n_observations == 2
        assert mon.n_covered == 1

    def test_empirical_coverage_correct(self):
        """empirical_coverage = n_covered / n_observations."""
        mon = CoverageMonitor()
        for _ in range(8):
            mon.update(True)
        for _ in range(2):
            mon.update(False)
        assert mon.empirical_coverage == pytest.approx(0.8)

    def test_no_alarm_under_normal_coverage(self):
        """No alarm when coverage matches the target."""
        mon = CoverageMonitor(target_coverage=0.9, h=5.0)
        # Simulate 90% coverage: 9 covered, 1 missed per block
        for _ in range(5):
            for _ in range(9):
                mon.update(True)
            mon.update(False)
        assert not mon.alarm_triggered

    def test_alarm_triggers_on_coverage_drop(self):
        """Sustained coverage drop triggers a CUSUM alarm."""
        mon = CoverageMonitor(target_coverage=0.9, h=3.0, k=0.01)
        alarm_fired = False
        # Simulate all misses -- coverage drops to 0
        for _ in range(100):
            if mon.update(False):
                alarm_fired = True
                break
        assert alarm_fired
        assert mon.alarm_triggered
        assert len(mon.alarm_history) >= 1

    def test_alarm_resets_cusum(self):
        """After alarm, CUSUM statistics are reset to zero."""
        mon = CoverageMonitor(target_coverage=0.9, h=2.0, k=0.01)
        for _ in range(100):
            if mon.update(False):
                break
        # After alarm, CUSUM should be reset
        assert mon.cusum_pos == 0.0
        assert mon.cusum_neg == 0.0

    def test_should_recalibrate_after_alarm(self):
        """should_recalibrate() returns True after an alarm."""
        mon = CoverageMonitor(target_coverage=0.9, h=2.0, k=0.01)
        for _ in range(100):
            if mon.update(False):
                break
        assert mon.should_recalibrate()

    def test_should_recalibrate_on_coverage_gap(self):
        """should_recalibrate() returns True when coverage deviates > 5% after 50 obs."""
        mon = CoverageMonitor(target_coverage=0.9)
        # Feed 50 observations with 70% coverage (gap = 20%)
        for i in range(50):
            mon.update(i % 10 < 7)
        # Even if no alarm yet, the 20% gap should trigger recalibration
        if not mon.alarm_triggered:
            assert mon.should_recalibrate()

    def test_should_not_recalibrate_when_healthy(self):
        """should_recalibrate() returns False under normal conditions (< 50 obs)."""
        mon = CoverageMonitor(target_coverage=0.9)
        for _ in range(10):
            mon.update(True)
        assert not mon.should_recalibrate()

    def test_alarm_history_accumulates(self):
        """Multiple alarms are recorded in alarm_history."""
        mon = CoverageMonitor(target_coverage=0.9, h=1.0, k=0.01)
        alarm_count = 0
        for _ in range(200):
            if mon.update(False):
                alarm_count += 1
            if alarm_count >= 2:
                break
        assert len(mon.alarm_history) >= 2
