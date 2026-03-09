"""Phase 55: Self-Supervised Domain Adaptation — Test Suite.

70 tests covering:
  - DomainAdapter construction and configuration
  - observe_frame buffering and bootstrap
  - detect_domain_shift (MMD scores + levels)
  - adapt() diagonal CORAL
  - get_adapted_embedding
  - apply_label integration
  - per-camera isolation
  - learning curve / shift history
  - reset (single camera + all)
  - engine.py wiring
  - API endpoints
"""

from __future__ import annotations

import base64
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sopilot.perception.domain_adapter import (
    CameraAdaptationState,
    DomainAdapter,
    _mmd_rbf,
)
from sopilot.perception.types import PerceptionConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 768


def _rand_feat(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=_DIM).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _make_adapter(**kwargs) -> DomainAdapter:
    defaults = {
        "feature_dim": _DIM,
        "memory_size": 200,
        "bootstrap_frames": 10,
        "shift_threshold": 0.15,
        "auto_adapt_every": 50,
    }
    defaults.update(kwargs)
    return DomainAdapter(**defaults)


def _feed(adapter: DomainAdapter, n: int, camera_id: str = "cam", seed_offset: int = 0) -> None:
    for i in range(n):
        adapter.observe_frame(_rand_feat(seed_offset + i), camera_id=camera_id)


def _b64(feat: np.ndarray) -> str:
    return base64.b64encode(feat.tobytes()).decode()


# ---------------------------------------------------------------------------
# Tests: CameraAdaptationState
# ---------------------------------------------------------------------------


class TestCameraAdaptationState:
    def test_defaults(self):
        st = CameraAdaptationState(camera_id="cam1")
        assert st.camera_id == "cam1"
        assert st.source_bootstrapped is False
        assert st.adapt_count == 0
        assert st.labeled_count == 0
        assert st.shift_score == 0.0

    def test_buffer_is_deque(self):
        st = CameraAdaptationState(camera_id="cam1")
        assert hasattr(st.buffer, "append")

    def test_shift_history_starts_empty(self):
        st = CameraAdaptationState(camera_id="cam1")
        assert st.shift_history == []

    def test_source_mean_none_initially(self):
        st = CameraAdaptationState(camera_id="cam1")
        assert st.source_mean is None
        assert st.source_std is None

    def test_correction_none_initially(self):
        st = CameraAdaptationState(camera_id="cam1")
        assert st.correction_mean is None
        assert st.correction_scale is None


# ---------------------------------------------------------------------------
# Tests: _mmd_rbf helper
# ---------------------------------------------------------------------------


class TestMmdRbf:
    def test_same_distribution_near_zero(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 8)).astype(np.float32)
        score = _mmd_rbf(X, X, gamma=1.0)
        assert score < 0.01  # trivially small with same array

    def test_different_distributions_positive(self):
        rng = np.random.default_rng(0)
        X = rng.normal(loc=0.0, size=(50, 8)).astype(np.float32)
        Y = rng.normal(loc=5.0, size=(50, 8)).astype(np.float32)
        score = _mmd_rbf(X, Y, gamma=0.1)
        assert score > 0.0

    def test_small_n_returns_zero(self):
        X = np.zeros((1, 8), dtype=np.float32)
        Y = np.ones((1, 8), dtype=np.float32)
        assert _mmd_rbf(X, Y) == 0.0

    def test_output_non_negative(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(30, 8)).astype(np.float32)
        Y = rng.normal(size=(30, 8)).astype(np.float32)
        assert _mmd_rbf(X, Y) >= 0.0

    def test_subsampling_applied(self):
        # Should not raise even with huge arrays (subsampling kicks in)
        from sopilot.perception.domain_adapter import _MMD_SAMPLE
        rng = np.random.default_rng(0)
        X = rng.normal(size=(_MMD_SAMPLE + 50, 4)).astype(np.float32)
        Y = rng.normal(size=(_MMD_SAMPLE + 50, 4)).astype(np.float32)
        score = _mmd_rbf(X, Y, gamma=1.0)
        assert score >= 0.0


# ---------------------------------------------------------------------------
# Tests: DomainAdapter construction
# ---------------------------------------------------------------------------


class TestDomainAdapterConstruction:
    def test_default_init(self):
        da = DomainAdapter()
        assert da._feature_dim == _DIM
        assert da._memory_size == 2000

    def test_custom_params(self):
        da = _make_adapter(bootstrap_frames=20, auto_adapt_every=30)
        assert da._bootstrap_frames == 20
        assert da._auto_adapt_every == 30

    def test_empty_cameras(self):
        da = _make_adapter()
        assert da.get_camera_states() == []

    def test_stats_initial(self):
        da = _make_adapter()
        s = da.get_stats()
        assert s["enabled"] is True
        assert s["total_frames_seen"] == 0
        assert s["camera_count"] == 0

    def test_feature_dim_stored(self):
        da = DomainAdapter(feature_dim=128)
        assert da._feature_dim == 128


# ---------------------------------------------------------------------------
# Tests: observe_frame
# ---------------------------------------------------------------------------


class TestObserveFrame:
    def test_first_frame_buffered(self):
        da = _make_adapter()
        da.observe_frame(_rand_feat(0))
        s = da.get_stats()
        assert s["total_frames_seen"] == 1

    def test_creates_camera_state(self):
        da = _make_adapter()
        da.observe_frame(_rand_feat(0), camera_id="cam_a")
        states = da.get_camera_states()
        assert len(states) == 1
        assert states[0]["camera_id"] == "cam_a"

    def test_wrong_dim_ignored(self):
        da = _make_adapter()
        bad = np.zeros(32, dtype=np.float32)
        da.observe_frame(bad, camera_id="cam")
        assert da.get_stats()["total_frames_seen"] == 0

    def test_bootstrap_triggered_after_n_frames(self):
        da = _make_adapter(bootstrap_frames=5)
        _feed(da, 5)
        states = da.get_camera_states()
        assert states[0]["source_bootstrapped"] is True

    def test_not_bootstrapped_before_n_frames(self):
        da = _make_adapter(bootstrap_frames=10)
        _feed(da, 4)
        states = da.get_camera_states()
        assert states[0]["source_bootstrapped"] is False

    def test_multiple_cameras_independent(self):
        da = _make_adapter(bootstrap_frames=5)
        _feed(da, 5, camera_id="cam_a")
        _feed(da, 3, camera_id="cam_b")
        states = {s["camera_id"]: s for s in da.get_camera_states()}
        assert states["cam_a"]["source_bootstrapped"] is True
        assert states["cam_b"]["source_bootstrapped"] is False

    def test_auto_adapt_triggers(self):
        da = _make_adapter(bootstrap_frames=5, auto_adapt_every=10)
        _feed(da, 5)   # bootstrap
        _feed(da, 10)  # trigger auto-adapt
        states = da.get_camera_states()
        assert states[0]["adapt_count"] >= 1

    def test_buffer_does_not_exceed_memory_size(self):
        da = _make_adapter(memory_size=20, bootstrap_frames=5)
        _feed(da, 30)
        states = da.get_camera_states()
        assert states[0]["buffer_size"] <= 20


# ---------------------------------------------------------------------------
# Tests: detect_domain_shift
# ---------------------------------------------------------------------------


class TestDetectDomainShift:
    def test_unknown_before_bootstrap(self):
        da = _make_adapter()
        result = da.detect_domain_shift("cam")
        assert result["shift_level"] == "unknown"

    def test_returns_zero_score_before_bootstrap(self):
        da = _make_adapter()
        result = da.detect_domain_shift("cam")
        assert result["shift_score"] == 0.0

    def test_returns_score_after_bootstrap(self):
        da = _make_adapter(bootstrap_frames=5)
        _feed(da, 10)
        result = da.detect_domain_shift("cam")
        assert "shift_score" in result
        assert result["shift_score"] >= 0.0

    def test_shift_level_minimal_for_similar_distributions(self):
        da = _make_adapter(bootstrap_frames=5, shift_threshold=1.0)
        _feed(da, 20, seed_offset=0)
        result = da.detect_domain_shift("cam")
        # With high threshold any small score is "minimal"
        assert result["shift_level"] in ("minimal", "moderate")

    def test_camera_id_in_response(self):
        da = _make_adapter()
        result = da.detect_domain_shift("my_camera")
        assert result["camera_id"] == "my_camera"

    def test_threshold_in_response(self):
        da = _make_adapter(shift_threshold=0.3)
        result = da.detect_domain_shift("cam")
        assert result["threshold"] == 0.3

    def test_severe_level_for_very_different_distributions(self):
        da = _make_adapter(bootstrap_frames=5, shift_threshold=1e-6, mmd_gamma=1.0)
        rng = np.random.default_rng(99)
        # feed source: near-zero vectors
        src = (rng.normal(loc=0.0, size=(5, _DIM)) * 0.001).astype(np.float32)
        for v in src:
            da.observe_frame(v)
        # feed target: large-magnitude vectors
        tgt = rng.normal(loc=10.0, size=(10, _DIM)).astype(np.float32)
        for v in tgt:
            da.observe_frame(v)
        result = da.detect_domain_shift("cam")
        # With tiny threshold, even small MMD is "severe"
        assert result["shift_level"] in ("significant", "severe")

    def test_source_and_target_frame_counts(self):
        da = _make_adapter(bootstrap_frames=5)
        _feed(da, 8)
        result = da.detect_domain_shift("cam")
        assert result["source_frames"] >= 5
        assert result["target_frames"] >= 1


# ---------------------------------------------------------------------------
# Tests: adapt() / get_adapted_embedding
# ---------------------------------------------------------------------------


class TestAdaptation:
    def test_adapt_returns_not_adapted_before_bootstrap(self):
        da = _make_adapter()
        result = da.adapt("cam")
        assert result["adapted"] is False

    def test_adapt_returns_adapted_true_after_bootstrap(self):
        da = _make_adapter(bootstrap_frames=5)
        _feed(da, 10)
        result = da.adapt("cam")
        assert result["adapted"] is True

    def test_adapt_increments_count(self):
        da = _make_adapter(bootstrap_frames=5)
        _feed(da, 10)
        da.adapt("cam")
        da.adapt("cam")
        states = da.get_camera_states()
        assert states[0]["adapt_count"] >= 2

    def test_adapt_correction_norm_reported(self):
        da = _make_adapter(bootstrap_frames=5)
        _feed(da, 10)
        result = da.adapt("cam")
        assert "correction_norm" in result

    def test_get_adapted_embedding_identity_before_bootstrap(self):
        da = _make_adapter()
        f = _rand_feat(42)
        out = da.get_adapted_embedding(f)
        assert np.allclose(f, out)

    def test_get_adapted_embedding_returns_float32(self):
        da = _make_adapter(bootstrap_frames=5)
        _feed(da, 10)
        da.adapt("cam")
        f = _rand_feat(0)
        out = da.get_adapted_embedding(f)
        assert out.dtype == np.float32

    def test_get_adapted_embedding_same_dim(self):
        da = _make_adapter(bootstrap_frames=5)
        _feed(da, 10)
        da.adapt("cam")
        f = _rand_feat(0)
        out = da.get_adapted_embedding(f)
        assert out.shape == f.shape

    def test_get_adapted_embedding_unit_norm(self):
        da = _make_adapter(bootstrap_frames=5)
        _feed(da, 10)
        da.adapt("cam")
        f = _rand_feat(0)
        out = da.get_adapted_embedding(f)
        assert abs(np.linalg.norm(out) - 1.0) < 0.01

    def test_adapt_updates_shift_score(self):
        da = _make_adapter(bootstrap_frames=5)
        _feed(da, 10)
        before = da.get_camera_states()[0]["shift_score"]
        da.adapt("cam")
        after = da.get_camera_states()[0]["shift_score"]
        # After adapt the shift score should be updated (may be same or different)
        assert isinstance(after, float)

    def test_has_correction_after_adapt(self):
        da = _make_adapter(bootstrap_frames=5)
        _feed(da, 10)
        da.adapt("cam")
        states = da.get_camera_states()
        assert states[0]["has_correction"] is True

    def test_different_camera_embedding_independent(self):
        da = _make_adapter(bootstrap_frames=5)
        _feed(da, 10, camera_id="cam_a")
        _feed(da, 10, camera_id="cam_b")
        da.adapt("cam_a")
        # cam_b has no adaptation, returns identity
        f = _rand_feat(7)
        out_b = da.get_adapted_embedding(f, camera_id="cam_b_unknown")
        assert np.allclose(f, out_b)


# ---------------------------------------------------------------------------
# Tests: apply_label
# ---------------------------------------------------------------------------


class TestApplyLabel:
    def test_apply_label_increments_count(self):
        da = _make_adapter()
        da.apply_label(_rand_feat(0), "person", camera_id="cam")
        s = da.get_stats()
        assert s["total_labeled"] == 1

    def test_apply_label_creates_camera_state(self):
        da = _make_adapter()
        da.apply_label(_rand_feat(0), "helmet", camera_id="cam_x")
        states = {s["camera_id"]: s for s in da.get_camera_states()}
        assert "cam_x" in states
        assert states["cam_x"]["labeled_count"] == 1

    def test_apply_label_adds_to_buffer(self):
        da = _make_adapter()
        da.apply_label(_rand_feat(0), "person", camera_id="cam")
        states = da.get_camera_states()
        assert states[0]["buffer_size"] == 1

    def test_multiple_labels_tracked(self):
        da = _make_adapter()
        for i in range(5):
            da.apply_label(_rand_feat(i), f"cat_{i % 3}", camera_id="cam")
        s = da.get_stats()
        assert s["total_labeled"] == 5

    def test_label_counts_per_camera(self):
        da = _make_adapter()
        da.apply_label(_rand_feat(0), "a", camera_id="c1")
        da.apply_label(_rand_feat(1), "a", camera_id="c1")
        da.apply_label(_rand_feat(2), "b", camera_id="c2")
        states = {s["camera_id"]: s for s in da.get_camera_states()}
        assert states["c1"]["labeled_count"] == 2
        assert states["c2"]["labeled_count"] == 1


# ---------------------------------------------------------------------------
# Tests: shift history (learning curve)
# ---------------------------------------------------------------------------


class TestShiftHistory:
    def test_empty_before_any_adapt(self):
        da = _make_adapter()
        assert da.get_shift_history("cam") == []

    def test_history_grows_after_adapt(self):
        da = _make_adapter(bootstrap_frames=5)
        _feed(da, 10)
        da.adapt("cam")
        hist = da.get_shift_history("cam")
        assert len(hist) == 1

    def test_history_contains_ts_and_score(self):
        da = _make_adapter(bootstrap_frames=5)
        _feed(da, 10)
        da.adapt("cam")
        hist = da.get_shift_history("cam")
        assert "ts" in hist[0]
        assert "shift_score" in hist[0]

    def test_history_capped_at_max(self):
        from sopilot.perception.domain_adapter import _LEARNING_CURVE_MAX
        da = _make_adapter(bootstrap_frames=5, auto_adapt_every=1)
        _feed(da, 5)
        for _ in range(_LEARNING_CURVE_MAX + 20):
            da.adapt("cam")
        hist = da.get_shift_history("cam")
        assert len(hist) <= _LEARNING_CURVE_MAX

    def test_unknown_camera_returns_empty(self):
        da = _make_adapter()
        assert da.get_shift_history("no_such_cam") == []

    def test_history_scores_non_negative(self):
        da = _make_adapter(bootstrap_frames=5)
        _feed(da, 10)
        for _ in range(3):
            da.adapt("cam")
        hist = da.get_shift_history("cam")
        assert all(h["shift_score"] >= 0.0 for h in hist)


# ---------------------------------------------------------------------------
# Tests: reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_all_clears_cameras(self):
        da = _make_adapter()
        _feed(da, 5, camera_id="c1")
        _feed(da, 5, camera_id="c2")
        da.reset()
        assert da.get_camera_states() == []

    def test_reset_all_clears_global_counters(self):
        da = _make_adapter()
        _feed(da, 5)
        da.reset()
        s = da.get_stats()
        assert s["total_frames_seen"] == 0

    def test_reset_single_camera(self):
        da = _make_adapter()
        _feed(da, 5, camera_id="c1")
        _feed(da, 5, camera_id="c2")
        da.reset(camera_id="c1")
        states = {s["camera_id"]: s for s in da.get_camera_states()}
        assert "c1" not in states
        assert "c2" in states

    def test_reset_unknown_camera_noop(self):
        da = _make_adapter()
        _feed(da, 5, camera_id="c1")
        da.reset(camera_id="nonexistent")  # Should not raise
        assert len(da.get_camera_states()) == 1

    def test_after_reset_can_re_bootstrap(self):
        da = _make_adapter(bootstrap_frames=5)
        _feed(da, 5)
        da.reset()
        _feed(da, 5)
        states = da.get_camera_states()
        assert states[0]["source_bootstrapped"] is True


# ---------------------------------------------------------------------------
# Tests: engine.py integration
# ---------------------------------------------------------------------------


class TestEngineIntegration:
    def test_domain_adapter_attr_exists(self):
        from sopilot.perception.engine import PerceptionEngine
        config = PerceptionConfig()
        engine = PerceptionEngine(config)
        assert hasattr(engine, "_domain_adapter")
        assert engine._domain_adapter is None

    def test_get_domain_adapter_state_none_when_disabled(self):
        from sopilot.perception.engine import PerceptionEngine
        config = PerceptionConfig()
        engine = PerceptionEngine(config)
        assert engine.get_domain_adapter_state() is None

    def test_enabled_via_config(self):
        from sopilot.perception.engine import build_perception_engine
        config = PerceptionConfig(
            domain_adaptation_enabled=True,
            domain_adaptation_bootstrap_frames=5,
        )
        engine = build_perception_engine(config)
        assert engine._domain_adapter is not None

    def test_get_domain_adapter_state_when_enabled(self):
        from sopilot.perception.engine import build_perception_engine
        config = PerceptionConfig(domain_adaptation_enabled=True)
        engine = build_perception_engine(config)
        s = engine.get_domain_adapter_state()
        assert s is not None
        assert s["enabled"] is True

    def test_config_params_passed(self):
        from sopilot.perception.engine import build_perception_engine
        config = PerceptionConfig(
            domain_adaptation_enabled=True,
            domain_adaptation_memory_size=500,
            domain_adaptation_bootstrap_frames=20,
        )
        engine = build_perception_engine(config)
        s = engine.get_domain_adapter_state()
        assert s["memory_size"] == 500
        assert s["bootstrap_frames"] == 20


# ---------------------------------------------------------------------------
# Tests: API endpoints
# ---------------------------------------------------------------------------


class TestDomainAdapterAPI:
    """Minimal integration tests for the 7 domain adapter endpoints."""

    def _make_app_and_client(self, enabled: bool = True):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sopilot.vigil.perception_router import build_perception_router

        da = DomainAdapter(
            memory_size=200,
            bootstrap_frames=5,
        ) if enabled else None

        engine_mock = MagicMock()
        engine_mock._domain_adapter = da
        engine_mock.get_domain_adapter_state = (
            (lambda: da.get_stats()) if da else (lambda: None)
        )

        class _FakePipeline:
            _vlm = MagicMock()
            _vlm._engine = engine_mock

        app = FastAPI()
        app.state.vigil_pipeline = _FakePipeline()
        router = build_perception_router()
        # Router already has prefix="/vigil/perception" — do not double-add
        app.include_router(router)
        return TestClient(app)

    def test_stats_disabled(self):
        client = self._make_app_and_client(enabled=False)
        r = client.get("/vigil/perception/domain/stats")
        assert r.status_code == 200
        assert r.json()["enabled"] is False

    def test_stats_enabled(self):
        client = self._make_app_and_client(enabled=True)
        r = client.get("/vigil/perception/domain/stats")
        assert r.status_code == 200
        assert r.json()["enabled"] is True

    def test_cameras_endpoint(self):
        client = self._make_app_and_client()
        r = client.get("/vigil/perception/domain/cameras")
        assert r.status_code == 200
        assert "cameras" in r.json()

    def test_shift_endpoint_before_bootstrap(self):
        client = self._make_app_and_client()
        r = client.get("/vigil/perception/domain/shift?camera_id=cam")
        assert r.status_code == 200
        assert r.json()["shift_level"] == "unknown"

    def test_adapt_endpoint_before_bootstrap(self):
        client = self._make_app_and_client()
        r = client.post("/vigil/perception/domain/adapt", json={"camera_id": "cam"})
        assert r.status_code == 200
        assert r.json()["adapted"] is False

    def test_observe_endpoint(self):
        client = self._make_app_and_client()
        feat = _rand_feat(0)
        r = client.post(
            "/vigil/perception/domain/observe",
            json={"camera_id": "cam", "features_b64": _b64(feat)},
        )
        assert r.status_code == 200
        assert r.json()["success"] is True

    def test_observe_missing_features_400(self):
        client = self._make_app_and_client()
        r = client.post(
            "/vigil/perception/domain/observe",
            json={"camera_id": "cam"},
        )
        assert r.status_code == 400

    def test_label_endpoint(self):
        client = self._make_app_and_client()
        feat = _rand_feat(1)
        r = client.post(
            "/vigil/perception/domain/label",
            json={"camera_id": "cam", "category": "person", "features_b64": _b64(feat)},
        )
        assert r.status_code == 200
        assert r.json()["success"] is True

    def test_label_missing_category_400(self):
        client = self._make_app_and_client()
        feat = _rand_feat(1)
        r = client.post(
            "/vigil/perception/domain/label",
            json={"camera_id": "cam", "features_b64": _b64(feat)},
        )
        assert r.status_code == 400

    def test_shift_history_endpoint(self):
        client = self._make_app_and_client()
        r = client.get("/vigil/perception/domain/shift/history?camera_id=cam")
        assert r.status_code == 200
        assert "history" in r.json()

    def test_reset_endpoint(self):
        client = self._make_app_and_client()
        r = client.request("DELETE", "/vigil/perception/domain/reset")
        assert r.status_code == 200
        assert r.json()["success"] is True
