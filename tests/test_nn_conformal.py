"""Tests for conformal prediction module."""

import numpy as np
import pytest


class TestSplitConformal:
    def test_coverage_guarantee(self):
        from sopilot.nn.conformal import SplitConformalPredictor

        rng = np.random.default_rng(42)
        n_cal = 200
        n_test = 1000
        alpha = 0.1

        predictions_cal = rng.normal(50, 10, n_cal)
        actuals_cal = predictions_cal + rng.normal(0, 5, n_cal)

        cp = SplitConformalPredictor(alpha=alpha)
        cp.calibrate(predictions_cal, actuals_cal)

        predictions_test = rng.normal(50, 10, n_test)
        actuals_test = predictions_test + rng.normal(0, 5, n_test)

        covered = 0
        for pred, actual in zip(predictions_test, actuals_test, strict=False):
            _, lo, hi = cp.predict(pred)
            if lo <= actual <= hi:
                covered += 1

        empirical_coverage = covered / n_test
        assert empirical_coverage >= (1 - alpha) - 0.05  # Allow small slack

    def test_wider_intervals_with_higher_alpha(self):
        from sopilot.nn.conformal import SplitConformalPredictor

        rng = np.random.default_rng(42)
        preds = rng.normal(50, 10, 100)
        actuals = preds + rng.normal(0, 5, 100)

        cp_tight = SplitConformalPredictor(alpha=0.1)
        cp_tight.calibrate(preds, actuals)

        cp_wide = SplitConformalPredictor(alpha=0.01)
        cp_wide.calibrate(preds, actuals)

        assert cp_wide.interval_width >= cp_tight.interval_width

    def test_batch_predict(self):
        from sopilot.nn.conformal import SplitConformalPredictor

        cp = SplitConformalPredictor(alpha=0.05)
        preds = np.array([50.0, 60.0, 70.0])
        actuals = np.array([52.0, 58.0, 72.0])
        cp.calibrate(preds, actuals)

        lo, hi = cp.predict_batch(np.array([55.0, 65.0]))
        assert lo.shape == (2,)
        assert hi.shape == (2,)
        assert (hi > lo).all()

    def test_uncalibrated_raises(self):
        from sopilot.nn.conformal import SplitConformalPredictor

        cp = SplitConformalPredictor()
        with pytest.raises(RuntimeError):
            cp.predict(50.0)


class TestCQR:
    def test_adaptive_intervals(self):
        from sopilot.nn.conformal import ConformizedQuantileRegression

        cqr = ConformizedQuantileRegression(alpha=0.1)
        lo_q = np.array([40.0, 55.0, 20.0])
        hi_q = np.array([60.0, 65.0, 80.0])
        actuals = np.array([50.0, 60.0, 50.0])

        cqr.calibrate(lo_q, hi_q, actuals)

        lo1, hi1 = cqr.predict(45.0, 55.0)
        lo2, hi2 = cqr.predict(20.0, 80.0)

        # Wider base interval -> wider conformal interval
        assert (hi2 - lo2) >= (hi1 - lo1)


class TestACI:
    def test_converges_to_target(self):
        from sopilot.nn.conformal import AdaptiveConformalInference

        aci = AdaptiveConformalInference(target_alpha=0.1, learning_rate=0.05)

        rng = np.random.default_rng(42)
        for _ in range(200):
            actual = rng.normal(50, 5)
            interval = (45.0, 55.0)
            aci.update(interval, actual)

        assert 0.001 < aci.current_alpha < 0.999
        assert aci.n_updates == 200

    def test_alpha_stays_bounded(self):
        from sopilot.nn.conformal import AdaptiveConformalInference

        aci = AdaptiveConformalInference(target_alpha=0.05, learning_rate=0.1)
        for _ in range(100):
            aci.update((0.0, 100.0), 50.0)  # Always covered
        assert aci.current_alpha >= 0.001

        for _ in range(100):
            aci.update((49.0, 51.0), 1000.0)  # Never covered
        assert aci.current_alpha <= 0.999


class TestMondrian:
    def test_per_group_calibration(self):
        from sopilot.nn.conformal import MondrianConformal

        mc = MondrianConformal(alpha=0.1)
        preds = np.array([50, 60, 70, 20, 30, 40], dtype=float)
        actuals = np.array([52, 58, 72, 22, 28, 42], dtype=float)
        groups = np.array([0, 0, 0, 1, 1, 1])

        mc.calibrate(preds, actuals, groups)

        _, lo0, hi0 = mc.predict(55.0, 0)
        _, lo1, hi1 = mc.predict(25.0, 1)

        assert hi0 > lo0
        assert hi1 > lo1

    def test_unknown_group_uses_max(self):
        from sopilot.nn.conformal import MondrianConformal

        mc = MondrianConformal(alpha=0.1)
        preds = np.array([50.0, 60.0])
        actuals = np.array([52.0, 58.0])
        groups = np.array([0, 0])
        mc.calibrate(preds, actuals, groups)

        # Group 99 doesn't exist, should use max quantile
        _, lo, hi = mc.predict(50.0, 99)
        assert hi > lo
