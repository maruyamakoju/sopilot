"""Tests for sopilot/perception/adaptive_learner.py

Covers:
- PageHinkleyDetector: init, update, reset, get_state, return type, noise robustness, drift detection
- RecalibrationRecord: to_dict keys/values, rounding
- AdaptiveLearner: init, observe, window capping, total count, force_recalibrate,
  threshold clamping, percentile computation, ensemble application, history, state dict,
  cooldown, drift_count, thread safety
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from sopilot.perception.adaptive_learner import (
    AdaptiveLearner,
    PageHinkleyDetector,
    RecalibrationRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ensemble(sigma: float = 3.0) -> SimpleNamespace:
    config = SimpleNamespace(sigma_threshold=sigma)
    return SimpleNamespace(config=config)


# ===========================================================================
# PageHinkleyDetector
# ===========================================================================

class TestPageHinkleyInit:
    def test_defaults(self):
        ph = PageHinkleyDetector()
        assert ph._delta == 0.005
        assert ph._lambda == 50.0
        assert ph._alpha == 0.9999

    def test_custom_params(self):
        ph = PageHinkleyDetector(delta=0.01, lambda_=100.0, alpha=0.99)
        assert ph._delta == 0.01
        assert ph._lambda == 100.0
        assert ph._alpha == 0.99

    def test_initial_state_zero(self):
        ph = PageHinkleyDetector()
        assert ph._n == 0
        assert ph._sum == 0.0
        assert ph._min_sum == 0.0
        assert ph._x_mean == 0.0


class TestPageHinkleyUpdate:
    def test_stable_stream_no_drift(self):
        """Zero-mean alternating stream should not trigger drift because the
        PH statistic tracks upward drift; alternating 0/0 stays at zero."""
        ph = PageHinkleyDetector(delta=0.005, lambda_=50.0)
        drifts = [ph.update(0.0) for _ in range(200)]
        assert not any(drifts)

    def test_large_spike_triggers_drift(self):
        """A sustained large upward shift should eventually trigger drift."""
        ph = PageHinkleyDetector(delta=0.005, lambda_=50.0)
        # warm up
        for _ in range(50):
            ph.update(0.0)
        # sustained large positive shift
        triggered = False
        for _ in range(200):
            if ph.update(10.0):
                triggered = True
                break
        assert triggered

    def test_n_increments(self):
        ph = PageHinkleyDetector()
        for i in range(5):
            ph.update(float(i))
        assert ph._n == 5


class TestPageHinkleyReset:
    def test_reset_clears_state(self):
        ph = PageHinkleyDetector()
        for _ in range(20):
            ph.update(5.0)
        ph.reset()
        assert ph._n == 0
        assert ph._sum == 0.0
        assert ph._min_sum == 0.0
        assert ph._x_mean == 0.0

    def test_reset_then_update_works(self):
        ph = PageHinkleyDetector()
        for _ in range(10):
            ph.update(1.0)
        ph.reset()
        result = ph.update(1.0)
        assert isinstance(result, bool)
        assert ph._n == 1


class TestPageHinkleyGetState:
    def test_all_required_keys(self):
        ph = PageHinkleyDetector()
        state = ph.get_state()
        for key in ("n", "x_mean", "ph_stat", "threshold"):
            assert key in state, f"Missing key: {key}"

    def test_n_counts_correctly(self):
        ph = PageHinkleyDetector()
        for i in range(7):
            ph.update(float(i))
        assert ph.get_state()["n"] == 7

    def test_threshold_matches_lambda(self):
        ph = PageHinkleyDetector(lambda_=99.0)
        assert ph.get_state()["threshold"] == 99.0

    def test_ph_stat_non_negative(self):
        ph = PageHinkleyDetector()
        for _ in range(10):
            ph.update(1.0)
        assert ph.get_state()["ph_stat"] >= 0.0


class TestPageHinkleyReturnsBool:
    def test_update_returns_bool_true_or_false(self):
        ph = PageHinkleyDetector()
        for val in [0.0, 1.0, -1.0, 100.0, 0.5]:
            result = ph.update(val)
            assert isinstance(result, bool), f"Expected bool, got {type(result)}"


class TestPageHinkleyNoFalsePositiveOnNoise:
    def test_low_noise_no_drift(self):
        """1000 small-noise values should not trigger drift with lambda_=50."""
        rng = np.random.default_rng(42)
        ph = PageHinkleyDetector(delta=0.005, lambda_=50.0)
        noise = rng.normal(0.0, 0.01, size=1000)
        drifts = [ph.update(float(v)) for v in noise]
        assert not any(drifts)


class TestPageHinkleyDriftOnShift:
    def test_shift_triggers_drift(self):
        """100 normal values then 100 shifted by +20 should trigger drift."""
        ph = PageHinkleyDetector(delta=0.005, lambda_=50.0)
        rng = np.random.default_rng(0)
        for v in rng.normal(0.0, 0.1, 100):
            ph.update(float(v))
        triggered = False
        for v in rng.normal(20.0, 0.1, 100):
            if ph.update(float(v)):
                triggered = True
                break
        assert triggered


# ===========================================================================
# RecalibrationRecord
# ===========================================================================

class TestRecalibrationRecordToDict:
    def _make_record(self) -> RecalibrationRecord:
        return RecalibrationRecord(
            timestamp=1234567890.0,
            reason="drift",
            old_threshold=3.0,
            new_threshold=2.5,
            observations_used=100,
            drift_score=5.123456,
        )

    def test_all_keys_present(self):
        d = self._make_record().to_dict()
        for key in ("timestamp", "reason", "old_threshold", "new_threshold",
                    "observations_used", "drift_score"):
            assert key in d, f"Missing key: {key}"

    def test_values_correct(self):
        d = self._make_record().to_dict()
        assert d["timestamp"] == 1234567890.0
        assert d["reason"] == "drift"
        assert d["observations_used"] == 100

    def test_reason_preserved(self):
        for reason in ("drift", "manual", "feedback"):
            r = RecalibrationRecord(1.0, reason, 3.0, 2.5, 10, 0.0)
            assert r.to_dict()["reason"] == reason


class TestRecalibrationRecordRound:
    def test_thresholds_rounded_to_4_decimals(self):
        r = RecalibrationRecord(
            timestamp=0.0,
            reason="manual",
            old_threshold=3.123456789,
            new_threshold=2.987654321,
            observations_used=50,
            drift_score=1.111111111,
        )
        d = r.to_dict()
        assert d["old_threshold"] == round(3.123456789, 4)
        assert d["new_threshold"] == round(2.987654321, 4)
        assert d["drift_score"] == round(1.111111111, 4)


# ===========================================================================
# AdaptiveLearner
# ===========================================================================

class TestAdaptiveLearnerInit:
    def test_defaults(self):
        al = AdaptiveLearner()
        assert al._min_obs == 50
        assert al._cooldown == 300.0
        assert al._window_size == 200
        assert al._percentile == 75.0
        assert al._thresh_min == 1.5
        assert al._thresh_max == 8.0
        assert al._ensemble is None
        assert al._total_observed == 0
        assert al._drift_count == 0

    def test_custom_params(self):
        ens = make_ensemble(4.0)
        al = AdaptiveLearner(
            ensemble=ens,
            min_observations=100,
            recalibration_cooldown_s=60.0,
            score_window_size=500,
            new_threshold_percentile=90.0,
            threshold_min=2.0,
            threshold_max=6.0,
            drift_delta=0.01,
            drift_lambda=30.0,
        )
        assert al._ensemble is ens
        assert al._min_obs == 100
        assert al._cooldown == 60.0
        assert al._window_size == 500
        assert al._percentile == 90.0
        assert al._thresh_min == 2.0
        assert al._thresh_max == 6.0


class TestObserveReturnsBool:
    def test_observe_returns_bool(self):
        al = AdaptiveLearner()
        result = al.observe(1.0)
        assert isinstance(result, bool)

    def test_observe_multiple_returns_bool(self):
        al = AdaptiveLearner()
        for v in [0.0, 0.5, 1.0, -1.0, 100.0]:
            assert isinstance(al.observe(float(v)), bool)


class TestObserveAccumulatesScores:
    def test_five_observations_in_window(self):
        al = AdaptiveLearner(score_window_size=200)
        for i in range(5):
            al.observe(float(i))
        assert len(al._scores) == 5

    def test_window_not_exceeded(self):
        al = AdaptiveLearner(score_window_size=10)
        for i in range(8):
            al.observe(float(i))
        assert len(al._scores) <= 10


class TestWindowCapped:
    def test_window_capped_at_window_size(self):
        al = AdaptiveLearner(score_window_size=200)
        for i in range(300):
            al.observe(float(i % 10))
        assert len(al._scores) == 200

    def test_window_keeps_most_recent(self):
        al = AdaptiveLearner(score_window_size=5)
        for i in range(10):
            al.observe(float(i))
        assert al._scores == [5.0, 6.0, 7.0, 8.0, 9.0]


class TestTotalObservedCounts:
    def test_counts_every_observe_call(self):
        al = AdaptiveLearner()
        for i in range(42):
            al.observe(float(i % 5))
        assert al._total_observed == 42

    def test_count_with_window_overflow(self):
        al = AdaptiveLearner(score_window_size=10)
        for i in range(50):
            al.observe(1.0)
        assert al._total_observed == 50


class TestForceRecalibrateInsufficientData:
    def test_no_data_returns_none(self):
        al = AdaptiveLearner(min_observations=50)
        result = al.force_recalibrate()
        assert result is None

    def test_too_few_scores_returns_none(self):
        al = AdaptiveLearner(min_observations=50)
        # need at least max(10, 50//5) = 10 scores
        for i in range(5):
            al.observe(float(i))
        result = al.force_recalibrate()
        assert result is None

    def test_exactly_at_boundary_returns_none(self):
        al = AdaptiveLearner(min_observations=50)
        # boundary is max(10, 50//5) = 10; feed 9 -> still None
        for i in range(9):
            al.observe(float(i))
        result = al.force_recalibrate()
        assert result is None


class TestForceRecalibrateWithData:
    def test_returns_record_with_sufficient_data(self):
        al = AdaptiveLearner(min_observations=50)
        for i in range(100):
            al.observe(float(i % 10))
        result = al.force_recalibrate()
        assert result is not None
        assert isinstance(result, RecalibrationRecord)

    def test_reason_is_manual(self):
        al = AdaptiveLearner(min_observations=50)
        for i in range(100):
            al.observe(float(i % 5))
        rec = al.force_recalibrate()
        assert rec.reason == "manual"

    def test_observations_used_correct(self):
        al = AdaptiveLearner(min_observations=50, score_window_size=200)
        for i in range(100):
            al.observe(float(i % 5))
        rec = al.force_recalibrate()
        assert rec.observations_used == 100


class TestRecalibrationThresholdRange:
    def test_threshold_within_min_max(self):
        al = AdaptiveLearner(
            min_observations=10,
            threshold_min=1.5,
            threshold_max=8.0,
        )
        for i in range(50):
            al.observe(float(i % 3))
        rec = al.force_recalibrate()
        assert rec is not None
        assert al._thresh_min <= rec.new_threshold <= al._thresh_max

    def test_very_high_scores_clamped_to_max(self):
        al = AdaptiveLearner(
            min_observations=10,
            threshold_min=1.5,
            threshold_max=8.0,
            new_threshold_percentile=75.0,
        )
        for _ in range(50):
            al.observe(1000.0)
        rec = al.force_recalibrate()
        assert rec is not None
        assert rec.new_threshold == 8.0

    def test_very_low_scores_clamped_to_min(self):
        al = AdaptiveLearner(
            min_observations=10,
            threshold_min=1.5,
            threshold_max=8.0,
            new_threshold_percentile=75.0,
        )
        for _ in range(50):
            al.observe(0.0)
        rec = al.force_recalibrate()
        assert rec is not None
        assert rec.new_threshold == 1.5


class TestRecalibrationPercentile:
    def test_75th_percentile_of_uniform(self):
        """100 scores uniformly 0.0..1.0 -> 75th percentile ~0.75, but clamped to threshold_min."""
        al = AdaptiveLearner(
            min_observations=10,
            new_threshold_percentile=75.0,
            threshold_min=0.0,
            threshold_max=10.0,
        )
        scores = list(np.linspace(0.0, 1.0, 100))
        for s in scores:
            al.observe(s)
        rec = al.force_recalibrate()
        assert rec is not None
        expected = np.percentile(scores, 75.0)
        assert abs(rec.new_threshold - expected) < 0.05


class TestRecalibrationAppliedToEnsemble:
    def test_ensemble_sigma_threshold_updated(self):
        ens = make_ensemble(sigma=3.0)
        al = AdaptiveLearner(ensemble=ens, min_observations=10)
        for _ in range(50):
            al.observe(2.0)
        rec = al.force_recalibrate()
        assert rec is not None
        assert ens.config.sigma_threshold == rec.new_threshold

    def test_no_ensemble_still_returns_record(self):
        al = AdaptiveLearner(ensemble=None, min_observations=10)
        for _ in range(50):
            al.observe(2.0)
        rec = al.force_recalibrate()
        assert rec is not None


class TestRecalibrationHistory:
    def test_history_grows_with_calls(self):
        # Use very high drift_lambda so PH never auto-fires during observe();
        # only explicit force_recalibrate() calls add to history.
        al = AdaptiveLearner(
            min_observations=50,
            recalibration_cooldown_s=0.0,
            drift_lambda=1e12,
        )
        for _ in range(50):
            al.observe(1.5)

        for i in range(3):
            al.force_recalibrate(timestamp=float(i * 1000))
        assert len(al._recal_history) == 3

    def test_history_contains_records(self):
        # Same guard: high drift_lambda prevents auto-drift recalibration.
        al = AdaptiveLearner(
            min_observations=50,
            recalibration_cooldown_s=0.0,
            drift_lambda=1e12,
        )
        for _ in range(50):
            al.observe(2.0)
        al.force_recalibrate(timestamp=1000.0)
        history = al.get_recalibration_history()
        assert len(history) == 1
        assert isinstance(history[0], RecalibrationRecord)


class TestGetRecalibrationHistoryN:
    def test_returns_last_n_entries(self):
        # Use high drift_lambda to prevent auto-drift recalibrations
        al = AdaptiveLearner(
            min_observations=50,
            recalibration_cooldown_s=0.0,
            drift_lambda=1e12,
        )
        for _ in range(50):
            al.observe(1.5)
        for i in range(5):
            al.force_recalibrate(timestamp=float(i * 1000))
        history = al.get_recalibration_history(n=3)
        assert len(history) == 3

    def test_n_larger_than_history_returns_all(self):
        # Use high drift_lambda to prevent auto-drift recalibrations
        al = AdaptiveLearner(
            min_observations=50,
            recalibration_cooldown_s=0.0,
            drift_lambda=1e12,
        )
        for _ in range(50):
            al.observe(1.5)
        for i in range(2):
            al.force_recalibrate(timestamp=float(i * 1000))
        history = al.get_recalibration_history(n=20)
        assert len(history) == 2


class TestGetStateDictKeys:
    def test_all_required_keys_present(self):
        al = AdaptiveLearner()
        state = al.get_state_dict()
        required = {
            "total_observed",
            "score_window_size",
            "score_mean",
            "score_std",
            "drift_count",
            "recalibration_count",
            "last_recalibration",
            "ph_state",
        }
        for key in required:
            assert key in state, f"Missing key: {key}"

    def test_ph_state_nested_keys(self):
        al = AdaptiveLearner()
        state = al.get_state_dict()
        for key in ("n", "x_mean", "ph_stat", "threshold"):
            assert key in state["ph_state"], f"ph_state missing key: {key}"

    def test_types_correct(self):
        al = AdaptiveLearner()
        for _ in range(5):
            al.observe(1.0)
        state = al.get_state_dict()
        assert isinstance(state["total_observed"], int)
        assert isinstance(state["score_window_size"], int)
        assert isinstance(state["score_mean"], float)
        assert isinstance(state["score_std"], float)
        assert isinstance(state["drift_count"], int)
        assert isinstance(state["recalibration_count"], int)


class TestGetStateDictEmptyScores:
    def test_works_before_any_observations(self):
        al = AdaptiveLearner()
        state = al.get_state_dict()
        assert state["total_observed"] == 0
        assert state["score_window_size"] == 0
        assert state["last_recalibration"] is None
        # score_mean defaults to 0.0 when no scores
        assert state["score_mean"] == 0.0

    def test_recalibration_count_zero_initially(self):
        al = AdaptiveLearner()
        assert al.get_state_dict()["recalibration_count"] == 0


class TestNoneEnsemble:
    def test_observe_works_with_none_ensemble(self):
        al = AdaptiveLearner(ensemble=None)
        for i in range(10):
            result = al.observe(float(i))
        assert isinstance(result, bool)

    def test_force_recalibrate_works_with_none_ensemble(self):
        al = AdaptiveLearner(ensemble=None, min_observations=10)
        for _ in range(50):
            al.observe(1.5)
        rec = al.force_recalibrate()
        assert rec is not None

    def test_old_threshold_defaults_to_thresh_min(self):
        al = AdaptiveLearner(ensemble=None, min_observations=10, threshold_min=1.5)
        for _ in range(50):
            al.observe(2.0)
        rec = al.force_recalibrate()
        assert rec is not None
        assert rec.old_threshold == 1.5


class TestCooldownPreventsDuplicateRecalibration:
    def test_observe_cooldown_blocks_drift_recalibration(self):
        """Cooldown is enforced on the observe() drift path.
        Set _last_recal_ts manually to simulate a recent recalibration,
        then verify that a subsequent drift does not recalibrate again."""
        al = AdaptiveLearner(
            min_observations=1,
            recalibration_cooldown_s=9999.0,  # very long cooldown
            drift_lambda=5.0,
        )
        for _ in range(30):
            al.observe(0.0, timestamp=0.0)
        # Simulate that a recalibration just happened at t=0
        al._last_recal_ts = 1.0
        al._recal_history.append(RecalibrationRecord(1.0, "manual", 1.5, 1.5, 30, 0.0))
        history_len_before = len(al._recal_history)
        # Now feed large values to trigger PH drift at t=2 (within cooldown)
        triggered = False
        for _ in range(50):
            result = al.observe(100.0, timestamp=2.0)
            if result:
                triggered = True
                break
        # drift_count may increment but recalibration should NOT be added
        assert not triggered
        assert len(al._recal_history) == history_len_before

    def test_after_cooldown_expires_observe_recalibrates(self):
        """After cooldown expires, observe() drift triggers recalibration."""
        al = AdaptiveLearner(
            min_observations=1,
            recalibration_cooldown_s=100.0,
            drift_lambda=5.0,
        )
        for _ in range(30):
            al.observe(0.0, timestamp=0.0)
        # Simulate old recalibration at t=-200 (cooldown long expired)
        al._last_recal_ts = -200.0
        initial_len = len(al._recal_history)
        # Feed large values at t=0 to trigger drift after cooldown
        triggered = False
        for _ in range(100):
            if al.observe(100.0, timestamp=0.0):
                triggered = True
                break
        assert triggered
        assert len(al._recal_history) > initial_len

    def test_force_recalibrate_always_bypasses_cooldown(self):
        """force_recalibrate is a manual override and ignores cooldown."""
        al = AdaptiveLearner(
            min_observations=10,
            recalibration_cooldown_s=9999.0,
        )
        for _ in range(50):
            al.observe(1.5)
        rec1 = al.force_recalibrate(timestamp=0.0)
        rec2 = al.force_recalibrate(timestamp=1.0)   # within "cooldown" but still works
        assert rec1 is not None
        assert rec2 is not None


class TestDriftCountIncrementsOnPH:
    def test_drift_count_increments(self):
        """Trigger PH drift by feeding large values after small ones."""
        al = AdaptiveLearner(
            min_observations=1,
            recalibration_cooldown_s=0.0,
            drift_lambda=5.0,  # low threshold → easy to trigger
        )
        # Warm up with zeros
        for _ in range(20):
            al.observe(0.0, timestamp=0.0)
        initial_drift = al._drift_count
        # Feed large values to trigger drift
        for i in range(20):
            al.observe(100.0, timestamp=float(i * 1000))
        assert al._drift_count > initial_drift

    def test_no_drift_count_on_zero_stream(self):
        """A constant zero stream never accumulates positive PH sum, so no drift."""
        al = AdaptiveLearner(drift_lambda=50.0)
        for _ in range(100):
            al.observe(0.0)
        assert al._drift_count == 0


class TestThreadSafety:
    def test_concurrent_observe_no_errors(self):
        al = AdaptiveLearner(
            min_observations=50,
            recalibration_cooldown_s=0.0,
            score_window_size=500,
        )
        errors: list[Exception] = []

        def worker(thread_id: int) -> None:
            try:
                for i in range(50):
                    al.observe(float((thread_id * 50 + i) % 10))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        # All 500 observations should be counted
        assert al._total_observed == 500

    def test_concurrent_observe_and_recalibrate(self):
        al = AdaptiveLearner(
            min_observations=10,
            recalibration_cooldown_s=0.0,
            score_window_size=200,
        )
        # Pre-fill
        for _ in range(50):
            al.observe(1.5)

        errors: list[Exception] = []

        def observer() -> None:
            try:
                for _ in range(100):
                    al.observe(2.0)
            except Exception as e:
                errors.append(e)

        def recalibrator() -> None:
            try:
                for i in range(10):
                    al.force_recalibrate(timestamp=float(i * 1000))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=observer) for _ in range(5)]
        threads += [threading.Thread(target=recalibrator) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        # get_state_dict should also work cleanly after concurrent ops
        state = al.get_state_dict()
        assert state["total_observed"] >= 550
