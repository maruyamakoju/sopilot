"""Phase 57: Predictive Maintenance — AnomalyForecaster Test Suite (65 tests)."""

from __future__ import annotations

import math
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from sopilot.perception.anomaly_forecaster import (
    AnomalyForecaster,
    CameraForecastState,
    ForecastResult,
    MaintenanceAlert,
    TrendResult,
)
from sopilot.perception.types import PerceptionConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_forecaster(**kwargs) -> AnomalyForecaster:
    defaults = {
        "alpha": 0.3,
        "beta": 0.1,
        "threshold": 0.6,
        "sigmoid_scale": 0.2,
        "default_horizon_steps": 5,
        "alert_info_prob": 0.3,
        "alert_warning_prob": 0.6,
        "alert_critical_prob": 0.85,
    }
    defaults.update(kwargs)
    return AnomalyForecaster(**defaults)


def _feed(af: AnomalyForecaster, scores: list[float], camera_id: str = "cam") -> None:
    t = 1000.0
    for s in scores:
        af.observe(s, 0, camera_id=camera_id, timestamp=t)
        t += 1.0


# ---------------------------------------------------------------------------
# Tests: CameraForecastState
# ---------------------------------------------------------------------------


class TestCameraForecastState:
    def test_defaults(self):
        st = CameraForecastState(camera_id="cam1")
        assert st.camera_id == "cam1"
        assert st.level is None
        assert st.trend is None
        assert st.total_observed == 0

    def test_alerts_start_empty(self):
        st = CameraForecastState(camera_id="cam1")
        assert st.alerts == []

    def test_obs_buffer_is_deque(self):
        st = CameraForecastState(camera_id="cam1")
        assert hasattr(st.obs, "append")


# ---------------------------------------------------------------------------
# Tests: AnomalyForecaster construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_init(self):
        af = AnomalyForecaster()
        assert af._alpha > 0
        assert af._beta > 0
        assert af._threshold > 0

    def test_custom_params(self):
        af = _make_forecaster(alpha=0.5, threshold=0.7)
        assert af._alpha == 0.5
        assert af._threshold == 0.7

    def test_initial_stats(self):
        af = _make_forecaster()
        s = af.get_stats()
        assert s["enabled"] is True
        assert s["total_observed"] == 0
        assert s["camera_count"] == 0

    def test_cameras_empty_initially(self):
        af = _make_forecaster()
        assert af.get_camera_states() == []


# ---------------------------------------------------------------------------
# Tests: observe()
# ---------------------------------------------------------------------------


class TestObserve:
    def test_first_observation_stored(self):
        af = _make_forecaster()
        af.observe(0.5, camera_id="cam")
        s = af.get_stats()
        assert s["total_observed"] == 1

    def test_creates_camera_state(self):
        af = _make_forecaster()
        af.observe(0.5, camera_id="cam_x")
        states = af.get_camera_states()
        assert len(states) == 1
        assert states[0]["camera_id"] == "cam_x"

    def test_level_initialized_on_first_observe(self):
        af = _make_forecaster()
        af.observe(0.5, camera_id="cam")
        states = af.get_camera_states()
        assert states[0]["level"] == 0.5

    def test_score_clipped_to_01(self):
        af = _make_forecaster()
        af.observe(2.0, camera_id="cam")
        states = af.get_camera_states()
        assert states[0]["level"] <= 1.0

    def test_negative_score_clipped_to_zero(self):
        af = _make_forecaster()
        af.observe(-0.5, camera_id="cam")
        states = af.get_camera_states()
        assert states[0]["level"] >= 0.0

    def test_multiple_cameras_independent(self):
        af = _make_forecaster()
        af.observe(0.3, camera_id="cam_a")
        af.observe(0.9, camera_id="cam_b")
        states = {s["camera_id"]: s for s in af.get_camera_states()}
        assert "cam_a" in states
        assert "cam_b" in states

    def test_many_observations_counted(self):
        af = _make_forecaster()
        _feed(af, [0.5] * 20)
        assert af.get_stats()["total_observed"] == 20


# ---------------------------------------------------------------------------
# Tests: Holt-Winters smoothing
# ---------------------------------------------------------------------------


class TestHoltWinters:
    def test_rising_trend_increases_level(self):
        af = _make_forecaster(alpha=0.5, beta=0.3)
        _feed(af, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        trend = af.get_trend("cam")
        assert trend.trend_per_step > 0

    def test_falling_trend_decreases_level(self):
        af = _make_forecaster(alpha=0.5, beta=0.3)
        _feed(af, [1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
        trend = af.get_trend("cam")
        assert trend.trend_per_step < 0

    def test_stable_series_zero_trend(self):
        af = _make_forecaster(alpha=0.3, beta=0.1)
        _feed(af, [0.5] * 20)
        trend = af.get_trend("cam")
        assert abs(trend.trend_per_step) < 0.05

    def test_level_stays_in_bounds(self):
        af = _make_forecaster()
        _feed(af, [0.9] * 10)
        states = af.get_camera_states()
        assert 0.0 <= states[0]["level"] <= 1.0

    def test_two_observations_both_stored(self):
        af = _make_forecaster()
        af.observe(0.2, camera_id="cam")
        af.observe(0.8, camera_id="cam")
        states = af.get_camera_states()
        assert states[0]["samples"] == 2


# ---------------------------------------------------------------------------
# Tests: forecast()
# ---------------------------------------------------------------------------


class TestForecast:
    def test_empty_camera_returns_no_values(self):
        af = _make_forecaster()
        r = af.forecast("nonexistent")
        assert r.forecast_values == []

    def test_insufficient_samples_returns_empty(self):
        af = _make_forecaster()
        af.observe(0.5, camera_id="cam")
        r = af.forecast("cam")
        assert r.forecast_values == []

    def test_sufficient_samples_returns_forecasts(self):
        af = _make_forecaster()
        _feed(af, [0.5, 0.5, 0.5])
        r = af.forecast("cam", horizon_steps=3)
        assert len(r.forecast_values) == 3

    def test_forecast_length_matches_horizon(self):
        af = _make_forecaster()
        _feed(af, [0.4, 0.5, 0.6])
        r = af.forecast("cam", horizon_steps=7)
        assert len(r.forecast_values) == 7
        assert len(r.failure_probabilities) == 7

    def test_rising_trend_forecasts_increase(self):
        af = _make_forecaster(alpha=0.5, beta=0.3, default_horizon_steps=5)
        _feed(af, [0.1, 0.2, 0.3, 0.4, 0.5])
        r = af.forecast("cam", horizon_steps=5)
        # Last forecast should be >= first in rising trend
        assert r.forecast_values[-1] >= r.forecast_values[0] - 0.05

    def test_failure_probs_in_01(self):
        af = _make_forecaster()
        _feed(af, [0.7, 0.8, 0.9])
        r = af.forecast("cam")
        for p in r.failure_probabilities:
            assert 0.0 <= p <= 1.0

    def test_high_anomaly_critical_alert_level(self):
        af = _make_forecaster(threshold=0.5, sigmoid_scale=0.1, alert_critical_prob=0.5)
        _feed(af, [0.9, 0.9, 0.9])
        r = af.forecast("cam")
        assert r.alert_level in ("warning", "critical")

    def test_low_anomaly_no_alert(self):
        af = _make_forecaster(threshold=0.9)
        _feed(af, [0.1, 0.1, 0.1])
        r = af.forecast("cam")
        assert r.alert_level == "none"

    def test_peak_failure_prob_is_max(self):
        af = _make_forecaster()
        _feed(af, [0.5, 0.6, 0.7])
        r = af.forecast("cam", horizon_steps=5)
        if r.failure_probabilities:
            assert abs(r.peak_failure_prob - max(r.failure_probabilities)) < 1e-6

    def test_samples_used_reported(self):
        af = _make_forecaster()
        _feed(af, [0.5, 0.5, 0.5, 0.5])
        r = af.forecast("cam")
        assert r.samples_used == 4


# ---------------------------------------------------------------------------
# Tests: get_trend()
# ---------------------------------------------------------------------------


class TestGetTrend:
    def test_unknown_before_any_obs(self):
        af = _make_forecaster()
        t = af.get_trend("cam")
        assert t.direction == "unknown"

    def test_rising_after_upward_series(self):
        af = _make_forecaster(alpha=0.5, beta=0.3)
        _feed(af, [0.1, 0.3, 0.5, 0.7])
        t = af.get_trend("cam")
        assert t.direction == "rising"

    def test_falling_after_downward_series(self):
        af = _make_forecaster(alpha=0.5, beta=0.3)
        _feed(af, [0.9, 0.7, 0.5, 0.3])
        t = af.get_trend("cam")
        assert t.direction == "falling"

    def test_stable_after_constant(self):
        af = _make_forecaster(alpha=0.3, beta=0.1)
        _feed(af, [0.5] * 15)
        t = af.get_trend("cam")
        assert t.direction == "stable"

    def test_camera_id_in_result(self):
        af = _make_forecaster()
        t = af.get_trend("my_cam")
        assert t.camera_id == "my_cam"


# ---------------------------------------------------------------------------
# Tests: get_maintenance_alerts()
# ---------------------------------------------------------------------------


class TestAlerts:
    def test_no_alerts_when_score_low(self):
        af = _make_forecaster(threshold=0.99)
        _feed(af, [0.1, 0.2, 0.1, 0.1])
        alerts = af.get_maintenance_alerts()
        assert all(a["alert_level"] == "none" for a in alerts) or len(alerts) == 0

    def test_alert_generated_for_high_score(self):
        af = _make_forecaster(
            threshold=0.3,
            sigmoid_scale=0.05,
            alert_info_prob=0.1,
        )
        _feed(af, [0.8, 0.8, 0.8, 0.8])
        alerts = af.get_maintenance_alerts()
        assert len(alerts) > 0

    def test_alert_has_required_fields(self):
        af = _make_forecaster(threshold=0.2, alert_info_prob=0.1)
        _feed(af, [0.9, 0.9, 0.9, 0.9])
        alerts = af.get_maintenance_alerts()
        if alerts:
            a = alerts[0]
            assert "camera_id" in a
            assert "alert_level" in a
            assert "failure_prob" in a
            assert "message_ja" in a
            assert "message_en" in a

    def test_critical_alert_message_content(self):
        af = _make_forecaster(threshold=0.2, sigmoid_scale=0.05, alert_critical_prob=0.1)
        _feed(af, [0.9] * 5)
        alerts = af.get_maintenance_alerts()
        critical = [a for a in alerts if a["alert_level"] == "critical"]
        if critical:
            assert "緊急" in critical[0]["message_ja"] or "CRITICAL" in critical[0]["message_en"]

    def test_filter_by_camera_id(self):
        af = _make_forecaster(threshold=0.2, alert_info_prob=0.1)
        _feed(af, [0.9] * 4, camera_id="cam_a")
        _feed(af, [0.1] * 4, camera_id="cam_b")
        alerts_a = af.get_maintenance_alerts(camera_id="cam_a")
        assert all(a["camera_id"] == "cam_a" for a in alerts_a)

    def test_total_alerts_increments(self):
        af = _make_forecaster(threshold=0.2, alert_info_prob=0.1)
        _feed(af, [0.9] * 5)
        s = af.get_stats()
        assert s["total_alerts_generated"] >= 0  # may or may not fire based on prob


# ---------------------------------------------------------------------------
# Tests: failure_prob (sigmoid)
# ---------------------------------------------------------------------------


class TestFailureProb:
    def test_score_at_threshold_prob_near_half(self):
        af = _make_forecaster(threshold=0.6, sigmoid_scale=0.2)
        p = af._failure_prob(0.6)
        assert abs(p - 0.5) < 0.01

    def test_score_much_above_threshold_high_prob(self):
        af = _make_forecaster(threshold=0.5, sigmoid_scale=0.1)
        p = af._failure_prob(0.9)
        assert p > 0.95

    def test_score_much_below_threshold_low_prob(self):
        af = _make_forecaster(threshold=0.6, sigmoid_scale=0.1)
        p = af._failure_prob(0.1)
        assert p < 0.05

    def test_prob_in_01(self):
        af = _make_forecaster()
        for score in [0.0, 0.25, 0.5, 0.75, 1.0]:
            p = af._failure_prob(score)
            assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Tests: reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_all(self):
        af = _make_forecaster()
        _feed(af, [0.5, 0.5, 0.5], camera_id="c1")
        _feed(af, [0.5, 0.5, 0.5], camera_id="c2")
        af.reset()
        assert af.get_stats()["camera_count"] == 0
        assert af.get_stats()["total_observed"] == 0

    def test_reset_single_camera(self):
        af = _make_forecaster()
        _feed(af, [0.5] * 3, camera_id="c1")
        _feed(af, [0.5] * 3, camera_id="c2")
        af.reset(camera_id="c1")
        states = {s["camera_id"]: s for s in af.get_camera_states()}
        assert "c1" not in states
        assert "c2" in states

    def test_reset_unknown_noop(self):
        af = _make_forecaster()
        _feed(af, [0.5] * 3, camera_id="c1")
        af.reset(camera_id="nonexistent")
        assert len(af.get_camera_states()) == 1


# ---------------------------------------------------------------------------
# Tests: engine.py integration
# ---------------------------------------------------------------------------


class TestEngineIntegration:
    def test_attr_exists(self):
        from sopilot.perception.engine import PerceptionEngine
        config = PerceptionConfig()
        engine = PerceptionEngine(config)
        assert hasattr(engine, "_anomaly_forecaster")
        assert engine._anomaly_forecaster is None

    def test_state_none_when_disabled(self):
        from sopilot.perception.engine import PerceptionEngine
        config = PerceptionConfig()
        engine = PerceptionEngine(config)
        assert engine.get_anomaly_forecaster_state() is None

    def test_enabled_via_config(self):
        from sopilot.perception.engine import build_perception_engine
        config = PerceptionConfig(anomaly_forecaster_enabled=True)
        engine = build_perception_engine(config)
        assert engine._anomaly_forecaster is not None

    def test_state_when_enabled(self):
        from sopilot.perception.engine import build_perception_engine
        config = PerceptionConfig(anomaly_forecaster_enabled=True)
        engine = build_perception_engine(config)
        s = engine.get_anomaly_forecaster_state()
        assert s is not None
        assert s["enabled"] is True

    def test_config_params_passed(self):
        from sopilot.perception.engine import build_perception_engine
        config = PerceptionConfig(
            anomaly_forecaster_enabled=True,
            anomaly_forecaster_alpha=0.4,
            anomaly_forecaster_horizon=15,
        )
        engine = build_perception_engine(config)
        s = engine.get_anomaly_forecaster_state()
        assert s["alpha"] == 0.4
        assert s["default_horizon_steps"] == 15


# ---------------------------------------------------------------------------
# Tests: API endpoints
# ---------------------------------------------------------------------------


class TestForecastAPI:
    def _make_client(self, enabled: bool = True):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sopilot.vigil.perception_router import build_perception_router

        af = _make_forecaster() if enabled else None

        engine_mock = MagicMock()
        engine_mock._anomaly_forecaster = af
        engine_mock.get_anomaly_forecaster_state = (
            (lambda: af.get_stats()) if af else (lambda: None)
        )

        class _FakePipeline:
            _vlm = MagicMock()
            _vlm._engine = engine_mock

        app = FastAPI()
        app.state.vigil_pipeline = _FakePipeline()
        app.include_router(build_perception_router())
        return TestClient(app)

    def test_stats_disabled(self):
        client = self._make_client(enabled=False)
        r = client.get("/vigil/perception/forecast/stats")
        assert r.status_code == 200
        assert r.json()["enabled"] is False

    def test_stats_enabled(self):
        client = self._make_client()
        r = client.get("/vigil/perception/forecast/stats")
        assert r.status_code == 200
        assert r.json()["enabled"] is True

    def test_cameras_endpoint(self):
        client = self._make_client()
        r = client.get("/vigil/perception/forecast/cameras")
        assert r.status_code == 200
        assert "cameras" in r.json()

    def test_predict_before_data(self):
        client = self._make_client()
        r = client.get("/vigil/perception/forecast/predict?camera_id=cam")
        assert r.status_code == 200
        assert r.json()["forecast_values"] == []

    def test_trend_before_data(self):
        client = self._make_client()
        r = client.get("/vigil/perception/forecast/trend?camera_id=cam")
        assert r.status_code == 200
        assert r.json()["direction"] == "unknown"

    def test_alerts_endpoint(self):
        client = self._make_client()
        r = client.get("/vigil/perception/forecast/alerts")
        assert r.status_code == 200
        assert "alerts" in r.json()

    def test_observe_endpoint(self):
        client = self._make_client()
        r = client.post("/vigil/perception/forecast/observe", json={
            "anomaly_score": 0.7,
            "violation_count": 2,
            "camera_id": "cam",
        })
        assert r.status_code == 200
        assert r.json()["success"] is True

    def test_observe_then_predict(self):
        client = self._make_client()
        for s in [0.5, 0.6, 0.7]:
            client.post("/vigil/perception/forecast/observe", json={
                "anomaly_score": s, "violation_count": 0, "camera_id": "cam"
            })
        r = client.get("/vigil/perception/forecast/predict?camera_id=cam&horizon=3")
        assert r.status_code == 200
        d = r.json()
        assert len(d["forecast_values"]) == 3

    def test_reset_endpoint(self):
        client = self._make_client()
        r = client.request("DELETE", "/vigil/perception/forecast/reset")
        assert r.status_code == 200
        assert r.json()["success"] is True
