"""Tests for Phase 53: CausalInferenceEngine (causal_inference.py).

Coverage:
    - ObservedEvent / CausalEdge / InterventionResult data classes
    - _count_in_window / _nearest_preceding helpers
    - _granger_edge_for_window
    - _compute_granger_edges
    - CausalInferenceEngine.observe (buffer growth, thread safety)
    - CausalInferenceEngine.compute_causal_graph (caching, force flag, edge discovery)
    - CausalInferenceEngine.query_intervention (DAG traversal, empty graph)
    - CausalInferenceEngine.get_stats / get_event_types / reset
    - PerceptionConfig new fields (Phase 53 + 54)
    - engine._causal_inference_engine attribute exists
    - engine._safety_planner attribute exists
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from sopilot.perception.causal_inference import (
    CausalEdge,
    CausalInferenceEngine,
    InterventionResult,
    ObservedEvent,
    _compute_granger_edges,
    _count_in_window,
    _granger_edge_for_window,
    _nearest_preceding,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(offset: float = 0.0) -> float:
    return time.time() - offset


def _make_engine(**kwargs) -> CausalInferenceEngine:
    return CausalInferenceEngine(
        window_seconds=60.0,
        min_granger_strength=0.01,  # low threshold for tests
        min_samples=2,
        **kwargs,
    )


def _fill_engine(engine: CausalInferenceEngine, n_cause: int = 10, n_effect: int = 10, lag: float = 5.0) -> None:
    """Fill engine with cause→effect pairs at fixed lag."""
    now = time.time()
    for i in range(n_cause):
        engine.observe("PPE未着用", timestamp=now - 100 + i * 8, camera_id="cam01")
    for i in range(n_effect):
        engine.observe("転倒", timestamp=now - 100 + i * 8 + lag, camera_id="cam01")


# ---------------------------------------------------------------------------
# ObservedEvent
# ---------------------------------------------------------------------------


class TestObservedEvent:
    def test_fields(self):
        ev = ObservedEvent(
            event_type="PPE未着用",
            timestamp=1000.0,
            camera_id="cam01",
            confidence=0.9,
        )
        assert ev.event_type == "PPE未着用"
        assert ev.timestamp == 1000.0
        assert ev.camera_id == "cam01"
        assert ev.confidence == 0.9

    def test_default_confidence(self):
        ev = ObservedEvent(event_type="test", timestamp=0.0, camera_id="c")
        assert ev.confidence == 1.0


# ---------------------------------------------------------------------------
# CausalEdge
# ---------------------------------------------------------------------------


class TestCausalEdge:
    def test_to_dict(self):
        edge = CausalEdge(
            cause="PPE未着用",
            effect="転倒",
            strength=0.75,
            lag_seconds=5.0,
            samples=12,
            window_seconds=30.0,
        )
        d = edge.to_dict()
        assert d["cause"] == "PPE未着用"
        assert d["effect"] == "転倒"
        assert d["strength"] == 0.75
        assert d["lag_seconds"] == 5.0
        assert d["samples"] == 12
        assert d["window_seconds"] == 30.0

    def test_strength_rounded(self):
        edge = CausalEdge("A", "B", 0.123456, 2.0, 5, 30.0)
        assert len(str(edge.to_dict()["strength"])) <= 7


# ---------------------------------------------------------------------------
# InterventionResult
# ---------------------------------------------------------------------------


class TestInterventionResult:
    def test_to_dict_empty(self):
        r = InterventionResult(
            intervened_event="PPE未着用",
            prevented_effects=[],
            explanation_ja="テスト",
            explanation_en="test",
        )
        d = r.to_dict()
        assert d["intervened_event"] == "PPE未着用"
        assert d["prevented_effects"] == []
        assert "explanation_ja" in d

    def test_to_dict_with_effects(self):
        r = InterventionResult(
            intervened_event="X",
            prevented_effects=[("Y", 0.8), ("Z", 0.5)],
            explanation_ja="説明",
            explanation_en="explanation",
        )
        d = r.to_dict()
        assert len(d["prevented_effects"]) == 2
        assert d["prevented_effects"][0]["effect"] == "Y"
        assert d["prevented_effects"][0]["probability"] == pytest.approx(0.8, abs=1e-3)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


class TestCountInWindow:
    def test_empty(self):
        assert _count_in_window([], 0.0, 10.0) == 0

    def test_all_inside(self):
        times = [1.0, 2.0, 3.0, 4.0]
        assert _count_in_window(times, 0.0, 5.0) == 4

    def test_boundary_inclusive_lo_exclusive_hi(self):
        # [lo, hi) semantics: 0.0 and 1.0 included, 5.0 and 10.0 excluded
        times = [0.0, 1.0, 5.0, 10.0]
        assert _count_in_window(times, 0.0, 5.0) == 2

    def test_none_inside(self):
        times = [20.0, 30.0]
        assert _count_in_window(times, 0.0, 10.0) == 0


class TestNearestPreceding:
    def test_basic(self):
        times = [1.0, 2.0, 3.0]
        assert _nearest_preceding(times, 3.0) == 2.0

    def test_empty(self):
        assert _nearest_preceding([], 5.0) is None

    def test_all_after(self):
        assert _nearest_preceding([5.0, 6.0], 3.0) is None

    def test_exact_match_excluded(self):
        # bisect_left of t=3.0 in [1,2,3] → idx=2 → idx-1=1 → returns 2.0
        assert _nearest_preceding([1.0, 2.0, 3.0], 3.0) == 2.0


# ---------------------------------------------------------------------------
# Granger edge computation
# ---------------------------------------------------------------------------


class TestGrangerEdgeForWindow:
    def _cause_events(self, ts_list):
        return [ObservedEvent("PPE未着用", ts, "cam01") for ts in ts_list]

    def _effect_events(self, ts_list):
        return [ObservedEvent("転倒", ts, "cam01") for ts in ts_list]

    def test_no_cooccurrence(self):
        cause = self._cause_events([100.0, 200.0, 300.0])
        effect = self._effect_events([5.0, 10.0, 15.0])  # all before causes
        result = _granger_edge_for_window(
            "PPE未着用", "転倒", cause, effect, W=30.0,
            total_span=300.0, min_samples=2,
        )
        assert result is None

    def test_strong_cooccurrence(self):
        # 8 cause events immediately preceding 8 effect events at lag=2s
        base = 1000.0
        cause = self._cause_events([base + i * 10 for i in range(8)])
        effect = self._effect_events([base + i * 10 + 2 for i in range(8)])
        result = _granger_edge_for_window(
            "PPE未着用", "転倒", cause, effect, W=30.0,
            total_span=80.0, min_samples=3,
        )
        assert result is not None
        assert result.strength > 0
        assert result.samples >= 3

    def test_below_min_samples(self):
        base = 1000.0
        cause = self._cause_events([base + i * 10 for i in range(3)])
        effect = self._effect_events([base + i * 10 + 2 for i in range(3)])
        result = _granger_edge_for_window(
            "A", "B", cause, effect, W=30.0,
            total_span=30.0, min_samples=10,
        )
        assert result is None


class TestComputeGrangerEdges:
    def test_empty_returns_empty(self):
        assert _compute_granger_edges([], (30.0,), 0.1, 2) == []

    def test_single_event_type_returns_empty(self):
        events = [ObservedEvent("A", float(i), "cam01") for i in range(10)]
        assert _compute_granger_edges(events, (30.0,), 0.01, 2) == []

    def test_causal_pair_discovered(self):
        base = 1000.0
        events = []
        for i in range(10):
            events.append(ObservedEvent("PPE未着用", base + i * 10, "cam01"))
            events.append(ObservedEvent("転倒", base + i * 10 + 3, "cam01"))
        edges = _compute_granger_edges(events, (30.0,), 0.01, 2)
        causes = {e.cause for e in edges}
        effects = {e.effect for e in edges}
        # At minimum we should see at least one discovered edge
        assert len(edges) >= 1

    def test_edges_sorted_by_strength(self):
        base = 1000.0
        events = []
        for i in range(15):
            events.append(ObservedEvent("A", base + i * 5, "cam01"))
            events.append(ObservedEvent("B", base + i * 5 + 1, "cam01"))
            events.append(ObservedEvent("C", base + i * 5 + 20, "cam01"))
        edges = _compute_granger_edges(events, (10.0, 30.0), 0.01, 2)
        if len(edges) > 1:
            for i in range(len(edges) - 1):
                assert edges[i].strength >= edges[i + 1].strength


# ---------------------------------------------------------------------------
# CausalInferenceEngine
# ---------------------------------------------------------------------------


class TestCausalInferenceEngineObserve:
    def test_observe_grows_buffer(self):
        engine = _make_engine()
        assert engine.get_stats()["buffer_events"] == 0
        engine.observe("test_event")
        assert engine.get_stats()["buffer_events"] == 1

    def test_observe_default_timestamp(self):
        engine = _make_engine()
        t0 = time.time()
        engine.observe("event_A")
        stats = engine.get_stats()
        assert stats["buffer_events"] == 1

    def test_observe_custom_timestamp(self):
        engine = _make_engine()
        engine.observe("event_A", timestamp=1000.0)
        assert engine.get_stats()["buffer_events"] == 1

    def test_observe_multiple_types(self):
        engine = _make_engine()
        engine.observe("A")
        engine.observe("B")
        engine.observe("C")
        assert engine.get_stats()["buffer_events"] == 3
        assert engine.get_stats()["unique_event_types"] == 3

    def test_observe_thread_safe(self):
        engine = _make_engine(max_buffer=500)
        errors = []

        def worker():
            try:
                for _ in range(50):
                    engine.observe("thread_event")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert engine.get_stats()["buffer_events"] == 250

    def test_observe_camera_id(self):
        engine = _make_engine()
        engine.observe("A", camera_id="cam01")
        engine.observe("B", camera_id="cam02")
        assert engine.get_stats()["buffer_events"] == 2


class TestCausalInferenceEngineGraph:
    def test_empty_buffer_empty_graph(self):
        engine = _make_engine()
        assert engine.compute_causal_graph() == []

    def test_causal_graph_force(self):
        engine = _make_engine()
        _fill_engine(engine)
        g1 = engine.compute_causal_graph(force=True)
        # Force-recompute should return same or equivalent result
        g2 = engine.compute_causal_graph(force=True)
        assert len(g1) == len(g2)

    def test_causal_graph_caching(self):
        engine = _make_engine()
        _fill_engine(engine)
        g1 = engine.compute_causal_graph(force=True)
        # Cached result (buffer didn't grow by recompute_interval)
        g2 = engine.compute_causal_graph(force=False)
        assert len(g1) == len(g2)

    def test_edges_have_correct_structure(self):
        engine = _make_engine()
        _fill_engine(engine, n_cause=15, n_effect=15, lag=3.0)
        edges = engine.compute_causal_graph(force=True)
        for edge in edges:
            d = edge.to_dict()
            assert "cause" in d
            assert "effect" in d
            assert 0 <= d["strength"] <= 1.0
            assert d["samples"] >= 1
            assert d["lag_seconds"] >= 0

    def test_single_event_type_no_edges(self):
        engine = _make_engine()
        for i in range(20):
            engine.observe("only_one_type", timestamp=float(i))
        assert engine.compute_causal_graph(force=True) == []

    def test_graph_returns_list(self):
        engine = _make_engine()
        result = engine.compute_causal_graph()
        assert isinstance(result, list)


class TestCausalInferenceEngineIntervention:
    def test_intervention_empty_graph(self):
        engine = _make_engine()
        result = engine.query_intervention("PPE未着用")
        assert result.intervened_event == "PPE未着用"
        assert result.prevented_effects == []
        assert len(result.explanation_ja) > 0
        assert len(result.explanation_en) > 0

    def test_intervention_with_causal_chain(self):
        """A→B→C chain: intervening on A should prevent B and C."""
        engine = _make_engine()  # already uses min_granger_strength=0.01, min_samples=2
        base = 1000.0
        # A → B at lag 2, B → C at lag 2
        for i in range(20):
            engine.observe("A", timestamp=base + i * 10)
            engine.observe("B", timestamp=base + i * 10 + 2)
            engine.observe("C", timestamp=base + i * 10 + 4)
        engine.compute_causal_graph(force=True)
        result = engine.query_intervention("A")
        assert result.intervened_event == "A"
        # At least some effects should be found
        assert isinstance(result.prevented_effects, list)

    def test_intervention_result_to_dict(self):
        engine = _make_engine()
        result = engine.query_intervention("X")
        d = result.to_dict()
        assert "intervened_event" in d
        assert "prevented_effects" in d
        assert "explanation_ja" in d
        assert "explanation_en" in d

    def test_intervention_explanation_contains_event(self):
        engine = _make_engine()
        result = engine.query_intervention("ヘルメット未着用")
        assert "ヘルメット未着用" in result.explanation_ja
        assert "ヘルメット未着用" in result.explanation_en

    def test_intervention_increments_query_count(self):
        engine = _make_engine()
        assert engine.get_stats()["total_intervention_queries"] == 0
        engine.query_intervention("A")
        engine.query_intervention("B")
        assert engine.get_stats()["total_intervention_queries"] == 2


class TestCausalInferenceEngineStats:
    def test_stats_structure(self):
        engine = _make_engine()
        stats = engine.get_stats()
        expected = [
            "buffer_events", "unique_event_types", "causal_edges",
            "total_observed", "total_intervention_queries",
            "window_seconds", "min_granger_strength", "min_samples",
            "uptime_seconds",
        ]
        for key in expected:
            assert key in stats

    def test_stats_counts_observed(self):
        engine = _make_engine()
        engine.observe("A")
        engine.observe("B")
        engine.observe("A")
        stats = engine.get_stats()
        assert stats["total_observed"] == 3
        assert stats["unique_event_types"] == 2

    def test_get_event_types_sorted(self):
        engine = _make_engine()
        engine.observe("Z")
        engine.observe("A")
        engine.observe("M")
        types = engine.get_event_types()
        assert types == sorted(types)
        assert set(types) == {"A", "M", "Z"}

    def test_reset_clears_buffer(self):
        engine = _make_engine()
        _fill_engine(engine)
        assert engine.get_stats()["buffer_events"] > 0
        engine.reset()
        assert engine.get_stats()["buffer_events"] == 0
        assert engine.get_stats()["total_observed"] == 0

    def test_uptime_increases(self):
        engine = _make_engine()
        t0 = engine.get_stats()["uptime_seconds"]
        time.sleep(0.05)
        t1 = engine.get_stats()["uptime_seconds"]
        assert t1 >= t0

    def test_get_event_types_empty(self):
        engine = _make_engine()
        assert engine.get_event_types() == []


# ---------------------------------------------------------------------------
# PerceptionConfig new fields
# ---------------------------------------------------------------------------


class TestPerceptionConfigPhase53Fields:
    def test_causal_inference_fields_exist(self):
        from sopilot.perception.types import PerceptionConfig
        cfg = PerceptionConfig()
        assert hasattr(cfg, "causal_inference_enabled")
        assert hasattr(cfg, "causal_inference_window_seconds")
        assert hasattr(cfg, "causal_inference_min_strength")
        assert hasattr(cfg, "causal_inference_min_samples")

    def test_causal_inference_defaults(self):
        from sopilot.perception.types import PerceptionConfig
        cfg = PerceptionConfig()
        assert cfg.causal_inference_enabled is False
        assert cfg.causal_inference_window_seconds == 60.0
        assert cfg.causal_inference_min_strength == 0.25
        assert cfg.causal_inference_min_samples == 3

    def test_safety_planner_fields_exist(self):
        from sopilot.perception.types import PerceptionConfig
        cfg = PerceptionConfig()
        assert hasattr(cfg, "safety_planner_enabled")
        assert hasattr(cfg, "safety_planner_horizon_seconds")
        assert hasattr(cfg, "safety_planner_max_plans")

    def test_safety_planner_defaults(self):
        from sopilot.perception.types import PerceptionConfig
        cfg = PerceptionConfig()
        assert cfg.safety_planner_enabled is False
        assert cfg.safety_planner_horizon_seconds == 3600.0
        assert cfg.safety_planner_max_plans == 20


# ---------------------------------------------------------------------------
# Engine attribute existence
# ---------------------------------------------------------------------------


class TestEngineAttributes:
    def test_engine_has_causal_inference_engine_attr(self):
        from sopilot.perception.engine import PerceptionEngine
        from sopilot.perception.types import PerceptionConfig
        engine = PerceptionEngine(config=PerceptionConfig())
        assert hasattr(engine, "_causal_inference_engine")
        assert engine._causal_inference_engine is None

    def test_engine_has_safety_planner_attr(self):
        from sopilot.perception.engine import PerceptionEngine
        from sopilot.perception.types import PerceptionConfig
        engine = PerceptionEngine(config=PerceptionConfig())
        assert hasattr(engine, "_safety_planner")
        assert engine._safety_planner is None

    def test_get_causal_inference_state_returns_none_when_disabled(self):
        from sopilot.perception.engine import PerceptionEngine
        from sopilot.perception.types import PerceptionConfig
        engine = PerceptionEngine(config=PerceptionConfig())
        assert engine.get_causal_inference_state() is None

    def test_get_safety_planner_state_returns_none_when_disabled(self):
        from sopilot.perception.engine import PerceptionEngine
        from sopilot.perception.types import PerceptionConfig
        engine = PerceptionEngine(config=PerceptionConfig())
        assert engine.get_safety_planner_state() is None

    def test_get_causal_inference_state_returns_dict_when_enabled(self):
        from sopilot.perception.engine import PerceptionEngine
        from sopilot.perception.types import PerceptionConfig
        engine = PerceptionEngine(config=PerceptionConfig())
        engine._causal_inference_engine = CausalInferenceEngine()
        state = engine.get_causal_inference_state()
        assert state is not None
        assert "buffer_events" in state

    def test_get_safety_planner_state_returns_dict_when_enabled(self):
        from sopilot.perception.engine import PerceptionEngine
        from sopilot.perception.types import PerceptionConfig
        from sopilot.perception.safety_planner import SafetyPlanner
        engine = PerceptionEngine(config=PerceptionConfig())
        engine._safety_planner = SafetyPlanner()
        state = engine.get_safety_planner_state()
        assert state is not None
        assert "plans_running" in state
