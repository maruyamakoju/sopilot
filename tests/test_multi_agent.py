"""Tests for sopilot/perception/multi_agent.py — 60 tests total.

Coverage:
  - TestSharedSpatialMap       (12)
  - TestAgentManagement        (15)
  - TestEntityCoordination     (12)
  - TestEventBroadcast         (8)
  - TestStateAndReset          (5)
  - TestSingleton              (5)
  - TestThreadSafety           (3)
"""
from __future__ import annotations

import threading
import time
from collections import deque
from unittest.mock import patch

import numpy as np
import pytest

from sopilot.perception.multi_agent import (
    MAX_EVENT_BUFFER,
    MAX_GLOBAL_ENTITIES,
    AgentInfo,
    GlobalEntity,
    MultiAgentCoordinator,
    SharedSpatialMap,
    get_coordinator,
    reset_coordinator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ones_grid(h: int = 6, w: int = 8) -> list[list[float]]:
    return [[1.0] * w for _ in range(h)]


def _zeros_grid(h: int = 6, w: int = 8) -> list[list[float]]:
    return [[0.0] * w for _ in range(h)]


def _make_coordinator(**kwargs) -> MultiAgentCoordinator:
    return MultiAgentCoordinator(**kwargs)


# ---------------------------------------------------------------------------
# 1. TestSharedSpatialMap  (12 tests)
# ---------------------------------------------------------------------------

class TestSharedSpatialMap:

    def test_init_correct_dimensions(self):
        ssm = SharedSpatialMap(grid_w=10, grid_h=4)
        assert ssm.grid_w == 10
        assert ssm.grid_h == 4

    def test_init_grid_is_zeros(self):
        ssm = SharedSpatialMap(grid_w=8, grid_h=6)
        grid = ssm.get_global_grid()
        assert grid.shape == (6, 8)
        assert np.all(grid == 0.0)

    def test_update_alpha_one_equals_input(self):
        """With merge_alpha=1.0 and weight=1.0, grid must equal input."""
        ssm = SharedSpatialMap(grid_w=4, grid_h=3, merge_alpha=1.0)
        g = [[float(r * 4 + c) for c in range(4)] for r in range(3)]
        ssm.update_from_agent("a1", g, weight=1.0)
        result = ssm.get_global_grid()
        expected = np.array(g, dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_update_weight_zero_no_change(self):
        """weight=0.0 → effective_alpha=0 → grid unchanged."""
        ssm = SharedSpatialMap(grid_w=4, grid_h=3, merge_alpha=0.5)
        before = ssm.get_global_grid().copy()
        ssm.update_from_agent("a1", _ones_grid(3, 4), weight=0.0)
        after = ssm.get_global_grid()
        np.testing.assert_array_equal(before, after)

    def test_update_mismatched_grid_no_exception(self):
        """Mismatched grid size must be handled without raising."""
        ssm = SharedSpatialMap(grid_w=8, grid_h=6)
        bad_grid = [[1.0] * 3 for _ in range(2)]  # 2x3 vs 6x8
        ssm.update_from_agent("a1", bad_grid)  # must not raise

    def test_get_global_grid_returns_copy(self):
        """Mutating the returned array must not affect internal state."""
        ssm = SharedSpatialMap(grid_w=4, grid_h=3, merge_alpha=1.0)
        ssm.update_from_agent("a1", _ones_grid(3, 4), weight=1.0)
        copy = ssm.get_global_grid()
        copy[:] = 99.0
        internal = ssm.get_global_grid()
        assert not np.any(internal == 99.0)

    def test_get_hotspots_top_n(self):
        ssm = SharedSpatialMap(grid_w=4, grid_h=3, merge_alpha=1.0)
        g = [[0.0] * 4 for _ in range(3)]
        g[1][2] = 0.9
        g[0][0] = 0.5
        g[2][3] = 0.1
        ssm.update_from_agent("a1", g, weight=1.0)
        spots = ssm.get_hotspots(top_n=2)
        assert len(spots) == 2
        assert spots[0]["value"] == pytest.approx(0.9, abs=1e-4)

    def test_get_hotspots_empty_when_all_zeros(self):
        ssm = SharedSpatialMap(grid_w=4, grid_h=3)
        spots = ssm.get_hotspots(top_n=5)
        assert spots == []

    def test_get_state_dict_keys(self):
        ssm = SharedSpatialMap()
        d = ssm.get_state_dict()
        for key in ("grid_w", "grid_h", "merge_alpha", "update_count", "max_occupancy", "mean_occupancy"):
            assert key in d, f"Missing key: {key}"

    def test_reset_zeros_grid_and_count(self):
        ssm = SharedSpatialMap(grid_w=4, grid_h=3, merge_alpha=1.0)
        ssm.update_from_agent("a1", _ones_grid(3, 4), weight=1.0)
        assert ssm.get_state_dict()["update_count"] == 1
        ssm.reset()
        grid = ssm.get_global_grid()
        assert np.all(grid == 0.0)
        assert ssm.get_state_dict()["update_count"] == 0

    def test_multiple_agents_converges(self):
        """Multiple updates should converge: grid values must be > 0."""
        ssm = SharedSpatialMap(grid_w=4, grid_h=3, merge_alpha=0.5)
        for i in range(20):
            ssm.update_from_agent(f"a{i % 3}", _ones_grid(3, 4), weight=1.0)
        grid = ssm.get_global_grid()
        assert np.all(grid > 0.0)

    def test_weight_above_two_clamped_gracefully(self):
        """weight > 2.0 is clamped to 2.0 — no exception, just large alpha."""
        ssm = SharedSpatialMap(grid_w=4, grid_h=3, merge_alpha=0.1)
        ssm.update_from_agent("a1", _ones_grid(3, 4), weight=100.0)
        grid = ssm.get_global_grid()
        # effective_alpha = 0.1 * min(2.0, 100.0) = 0.2 → grid has positive values
        assert np.all(grid >= 0.0)

    def test_thread_safety_concurrent_updates(self):
        ssm = SharedSpatialMap(grid_w=4, grid_h=3, merge_alpha=0.1)
        errors: list[Exception] = []

        def worker(agent_id: str) -> None:
            try:
                for _ in range(50):
                    ssm.update_from_agent(agent_id, _ones_grid(3, 4), weight=1.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"a{i}",)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ---------------------------------------------------------------------------
# 2. TestAgentManagement  (15 tests)
# ---------------------------------------------------------------------------

class TestAgentManagement:

    def test_register_agent_returns_agent_info(self):
        coord = _make_coordinator()
        info = coord.register_agent("ag1", "cam1", "entrance")
        assert isinstance(info, AgentInfo)
        assert info.agent_id == "ag1"
        assert info.camera_id == "cam1"
        assert info.location == "entrance"

    def test_register_agent_initial_frame_count_zero(self):
        coord = _make_coordinator()
        info = coord.register_agent("ag1", "cam1")
        assert info.frame_count == 0

    def test_re_register_updates_fields_keeps_frame_count(self):
        coord = _make_coordinator()
        info = coord.register_agent("ag1", "cam1", "A")
        # Simulate some frames
        grid = _ones_grid()
        coord.submit_spatial_update("ag1", grid)
        coord.submit_spatial_update("ag1", grid)
        assert info.frame_count == 2
        # Re-register
        info2 = coord.register_agent("ag1", "cam2", "B")
        assert info2 is info  # same object
        assert info2.camera_id == "cam2"
        assert info2.location == "B"
        assert info2.frame_count == 2  # preserved

    def test_unregister_returns_true_when_found(self):
        coord = _make_coordinator()
        coord.register_agent("ag1", "cam1")
        result = coord.unregister_agent("ag1")
        assert result is True

    def test_unregister_returns_false_for_unknown(self):
        coord = _make_coordinator()
        result = coord.unregister_agent("nonexistent")
        assert result is False

    def test_heartbeat_returns_true_updates_last_seen(self):
        coord = _make_coordinator()
        coord.register_agent("ag1", "cam1")
        future = time.time() + 1000.0
        result = coord.heartbeat("ag1", _now=future)
        assert result is True
        assert coord.get_agent("ag1").last_seen_at == pytest.approx(future)

    def test_heartbeat_returns_false_for_unknown(self):
        coord = _make_coordinator()
        result = coord.heartbeat("ghost")
        assert result is False

    def test_get_active_agents_excludes_timed_out(self):
        coord = _make_coordinator(agent_timeout_seconds=10.0)
        coord.register_agent("ag1", "cam1")
        # Force last_seen_at to far past
        coord.get_agent("ag1").last_seen_at = time.time() - 100.0
        now = time.time()
        active = coord.get_active_agents(_now=now)
        ids = [a.agent_id for a in active]
        assert "ag1" not in ids

    def test_get_active_agents_includes_recent(self):
        coord = _make_coordinator(agent_timeout_seconds=30.0)
        coord.register_agent("ag1", "cam1")
        now = time.time()
        active = coord.get_active_agents(_now=now)
        ids = [a.agent_id for a in active]
        assert "ag1" in ids

    def test_get_agent_returns_none_for_unknown(self):
        coord = _make_coordinator()
        assert coord.get_agent("nope") is None

    def test_get_agent_returns_agent_info(self):
        coord = _make_coordinator()
        coord.register_agent("ag1", "cam1")
        info = coord.get_agent("ag1")
        assert info is not None
        assert info.agent_id == "ag1"

    def test_max_agents_evicts_oldest(self):
        coord = _make_coordinator(max_agents=3)
        # Register 3 agents with explicit last_seen times
        now = time.time()
        coord.register_agent("oldest", "cam0")
        coord.get_agent("oldest").last_seen_at = now - 100
        coord.register_agent("mid", "cam1")
        coord.get_agent("mid").last_seen_at = now - 50
        coord.register_agent("recent", "cam2")
        coord.get_agent("recent").last_seen_at = now
        # Register a 4th — should evict "oldest"
        coord.register_agent("new4", "cam3")
        assert coord.get_agent("oldest") is None
        assert coord.get_agent("new4") is not None

    def test_register_unregister_register_same_id(self):
        coord = _make_coordinator()
        coord.register_agent("ag1", "cam1")
        coord.unregister_agent("ag1")
        info = coord.register_agent("ag1", "cam2")
        assert info.camera_id == "cam2"
        assert info.frame_count == 0

    def test_agent_timeout_zero_all_inactive(self):
        coord = _make_coordinator(agent_timeout_seconds=0.0)
        coord.register_agent("ag1", "cam1")
        # With timeout=0, agent is active only if now - last_seen <= 0
        now = time.time() + 1.0  # advance time slightly
        active = coord.get_active_agents(_now=now)
        assert all(a.agent_id != "ag1" for a in active)

    def test_agent_info_to_dict_all_fields(self):
        coord = _make_coordinator()
        info = coord.register_agent("ag1", "cam1", "zone-A")
        d = info.to_dict()
        for key in ("agent_id", "camera_id", "location", "registered_at",
                    "last_seen_at", "frame_count", "spatial_contribution"):
            assert key in d, f"Missing key: {key}"

    def test_frame_count_increments_on_spatial_update(self):
        coord = _make_coordinator()
        coord.register_agent("ag1", "cam1")
        for i in range(5):
            coord.submit_spatial_update("ag1", _ones_grid())
        assert coord.get_agent("ag1").frame_count == 5


# ---------------------------------------------------------------------------
# 3. TestEntityCoordination  (12 tests)
# ---------------------------------------------------------------------------

class TestEntityCoordination:

    def _entities(self, count: int, label: str = "person") -> list[dict]:
        return [
            {"entity_id": i, "label": label, "cx": 0.1 * i, "cy": 0.5}
            for i in range(count)
        ]

    def test_submit_entities_returns_correct_length(self):
        coord = _make_coordinator()
        gids = coord.submit_entities("ag1", self._entities(4))
        assert len(gids) == 4

    def test_same_agent_same_local_id_same_global_id(self):
        coord = _make_coordinator()
        ent = [{"entity_id": 7, "label": "person", "cx": 0.5, "cy": 0.5}]
        now = time.time()
        gids1 = coord.submit_entities("ag1", ent, _now=now)
        gids2 = coord.submit_entities("ag1", ent, _now=now + 0.1)
        assert gids1[0] == gids2[0]

    def test_different_agents_same_label_may_merge(self):
        """Two agents reporting same label within 5s can share a global entity."""
        coord = _make_coordinator()
        now = time.time()
        ent = [{"entity_id": 0, "label": "worker", "cx": 0.3, "cy": 0.5}]
        gids1 = coord.submit_entities("ag1", ent, _now=now)
        gids2 = coord.submit_entities("ag2", ent, _now=now + 1.0)
        # They should share the same global entity (merged by proximity heuristic)
        assert gids1[0] == gids2[0]

    def test_submit_empty_entities_returns_empty(self):
        coord = _make_coordinator()
        gids = coord.submit_entities("ag1", [])
        assert gids == []

    def test_get_global_entities_returns_all(self):
        coord = _make_coordinator()
        coord.submit_entities("ag1", self._entities(3))
        entities = coord.get_global_entities()
        assert len(entities) >= 1  # at least some registered

    def test_get_entity_by_global_id_found(self):
        coord = _make_coordinator()
        ent = [{"entity_id": 1, "label": "car", "cx": 0.5, "cy": 0.5}]
        gids = coord.submit_entities("ag1", ent)
        ge = coord.get_entity_by_global_id(gids[0])
        assert ge is not None
        assert ge.global_id == gids[0]

    def test_get_entity_by_global_id_not_found(self):
        coord = _make_coordinator()
        result = coord.get_entity_by_global_id("00000000-0000-0000-0000-000000000000")
        assert result is None

    def test_max_global_entities_prunes_oldest(self):
        coord = _make_coordinator()
        # Fill up to MAX_GLOBAL_ENTITIES + 10 unique entities
        # Each call creates entities with new labels to avoid merging
        for i in range(MAX_GLOBAL_ENTITIES + 10):
            coord.submit_entities(
                "ag1",
                [{"entity_id": i, "label": f"obj_{i}", "cx": 0.5, "cy": 0.5}],
                _now=float(i),  # monotonically increasing so oldest are pruned
            )
        entities = coord.get_global_entities()
        assert len(entities) <= MAX_GLOBAL_ENTITIES

    def test_sighting_count_increments_on_resubmit(self):
        coord = _make_coordinator()
        ent = [{"entity_id": 0, "label": "forklift", "cx": 0.5, "cy": 0.5}]
        now = time.time()
        gids = coord.submit_entities("ag1", ent, _now=now)
        # Re-submit the same entity twice more
        coord.submit_entities("ag1", ent, _now=now + 0.1)
        coord.submit_entities("ag1", ent, _now=now + 0.2)
        ge = coord.get_entity_by_global_id(gids[0])
        assert ge.sighting_count == 3

    def test_global_entity_to_dict_all_keys(self):
        coord = _make_coordinator()
        ent = [{"entity_id": 0, "label": "box", "cx": 0.2, "cy": 0.8}]
        gids = coord.submit_entities("ag1", ent)
        ge = coord.get_entity_by_global_id(gids[0])
        d = ge.to_dict()
        for key in ("global_id", "local_ids", "label", "last_seen_at",
                    "last_seen_agent", "sighting_count"):
            assert key in d, f"Missing key: {key}"

    def test_different_labels_not_merged(self):
        """Entities with different labels must not share the same global_id."""
        coord = _make_coordinator()
        now = time.time()
        gids1 = coord.submit_entities(
            "ag1", [{"entity_id": 0, "label": "cat", "cx": 0.5, "cy": 0.5}], _now=now
        )
        gids2 = coord.submit_entities(
            "ag2", [{"entity_id": 0, "label": "dog", "cx": 0.5, "cy": 0.5}], _now=now + 0.5
        )
        assert gids1[0] != gids2[0]

    def test_unknown_agent_can_submit_entities(self):
        """No authentication: any agent_id can submit without prior registration."""
        coord = _make_coordinator()
        gids = coord.submit_entities(
            "unregistered_agent",
            [{"entity_id": 0, "label": "thing", "cx": 0.5, "cy": 0.5}],
        )
        assert len(gids) == 1


# ---------------------------------------------------------------------------
# 4. TestEventBroadcast  (8 tests)
# ---------------------------------------------------------------------------

class TestEventBroadcast:

    def test_broadcast_adds_source_agent_and_timestamp(self):
        coord = _make_coordinator()
        now = 1_000_000.0
        coord.broadcast_event("ag1", {"type": "violation"}, _now=now)
        events = coord.get_recent_events(10)
        assert len(events) == 1
        assert events[0]["source_agent_id"] == "ag1"
        assert events[0]["broadcast_at"] == pytest.approx(now)

    def test_get_recent_events_returns_last_n(self):
        coord = _make_coordinator()
        for i in range(10):
            coord.broadcast_event("ag1", {"i": i})
        events = coord.get_recent_events(3)
        assert len(events) == 3
        assert events[-1]["i"] == 9

    def test_get_recent_events_zero_returns_empty(self):
        coord = _make_coordinator()
        coord.broadcast_event("ag1", {"x": 1})
        events = coord.get_recent_events(0)
        assert events == []

    def test_event_buffer_max_deque_behaviour(self):
        """After MAX_EVENT_BUFFER events the oldest must be discarded."""
        coord = _make_coordinator()
        for i in range(MAX_EVENT_BUFFER + 50):
            coord.broadcast_event("ag1", {"seq": i})
        events = coord.get_recent_events(MAX_EVENT_BUFFER + 50)
        assert len(events) == MAX_EVENT_BUFFER

    def test_multiple_agents_broadcast_to_same_coordinator(self):
        coord = _make_coordinator()
        coord.broadcast_event("ag1", {"msg": "hello"})
        coord.broadcast_event("ag2", {"msg": "world"})
        events = coord.get_recent_events(10)
        agents = {e["source_agent_id"] for e in events}
        assert "ag1" in agents
        assert "ag2" in agents

    def test_total_events_broadcast_increments(self):
        coord = _make_coordinator()
        for _ in range(7):
            coord.broadcast_event("ag1", {})
        state = coord.get_state_dict()
        assert state["total_events_broadcast"] == 7

    def test_original_event_keys_preserved(self):
        """Keys in the original dict must survive enrichment unchanged."""
        coord = _make_coordinator()
        coord.broadcast_event("ag1", {"severity": "critical", "count": 42})
        event = coord.get_recent_events(1)[0]
        assert event["severity"] == "critical"
        assert event["count"] == 42

    def test_state_dict_event_buffer_size_matches(self):
        coord = _make_coordinator()
        for i in range(5):
            coord.broadcast_event("ag1", {"i": i})
        state = coord.get_state_dict()
        assert state["event_buffer_size"] == 5


# ---------------------------------------------------------------------------
# 5. TestStateAndReset  (5 tests)
# ---------------------------------------------------------------------------

class TestStateAndReset:

    def test_get_state_dict_expected_keys(self):
        coord = _make_coordinator()
        d = coord.get_state_dict()
        for key in ("total_agents", "active_agents", "total_global_entities",
                    "total_events_broadcast", "event_buffer_size", "shared_map", "agents"):
            assert key in d, f"Missing key: {key}"

    def test_reset_clears_all_agents(self):
        coord = _make_coordinator()
        coord.register_agent("ag1", "cam1")
        coord.register_agent("ag2", "cam2")
        coord.reset()
        assert coord.get_state_dict()["total_agents"] == 0

    def test_reset_clears_all_entities(self):
        coord = _make_coordinator()
        coord.submit_entities("ag1", [{"entity_id": 0, "label": "box", "cx": 0.5, "cy": 0.5}])
        coord.reset()
        assert len(coord.get_global_entities()) == 0

    def test_reset_clears_events_and_count(self):
        coord = _make_coordinator()
        coord.broadcast_event("ag1", {"x": 1})
        coord.broadcast_event("ag1", {"x": 2})
        coord.reset()
        assert coord.get_recent_events(100) == []
        assert coord.get_state_dict()["total_events_broadcast"] == 0

    def test_reset_resets_shared_map(self):
        coord = _make_coordinator()
        coord.register_agent("ag1", "cam1")
        coord.submit_spatial_update("ag1", _ones_grid())
        coord.reset()
        grid = coord.get_shared_map().get_global_grid()
        assert np.all(grid == 0.0)


# ---------------------------------------------------------------------------
# 6. TestSingleton  (5 tests)
# ---------------------------------------------------------------------------

class TestSingleton:

    def setup_method(self):
        reset_coordinator()

    def teardown_method(self):
        reset_coordinator()

    def test_get_coordinator_returns_instance(self):
        c = get_coordinator()
        assert isinstance(c, MultiAgentCoordinator)

    def test_get_coordinator_same_instance_twice(self):
        c1 = get_coordinator()
        c2 = get_coordinator()
        assert c1 is c2

    def test_reset_coordinator_new_instance(self):
        c1 = get_coordinator()
        reset_coordinator()
        c2 = get_coordinator()
        assert c1 is not c2

    def test_old_reference_still_works_after_reset(self):
        c1 = get_coordinator()
        c1.register_agent("ag1", "cam1")
        reset_coordinator()
        # c1 is still functional even though singleton was reset
        info = c1.get_agent("ag1")
        assert info is not None

    def test_concurrent_get_coordinator_same_instance(self):
        results: list[MultiAgentCoordinator] = []
        lock = threading.Lock()

        def worker():
            c = get_coordinator()
            with lock:
                results.append(c)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # All references must be identical
        assert all(r is results[0] for r in results)


# ---------------------------------------------------------------------------
# 7. TestThreadSafety  (3 tests)
# ---------------------------------------------------------------------------

class TestThreadSafety:

    def test_concurrent_register_agent(self):
        coord = _make_coordinator(max_agents=100)
        errors: list[Exception] = []

        def worker(idx: int):
            try:
                coord.register_agent(f"ag{idx}", f"cam{idx}", f"loc{idx}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        state = coord.get_state_dict()
        assert state["total_agents"] <= 100

    def test_concurrent_submit_spatial_update(self):
        coord = _make_coordinator()
        for i in range(4):
            coord.register_agent(f"ag{i}", f"cam{i}")
        errors: list[Exception] = []

        def worker(agent_id: str):
            try:
                for _ in range(100):
                    coord.submit_spatial_update(agent_id, _ones_grid())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"ag{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # Shared map must have positive values after all updates
        grid = coord.get_shared_map().get_global_grid()
        assert np.all(grid > 0.0)

    def test_concurrent_broadcast_and_read(self):
        coord = _make_coordinator()
        errors: list[Exception] = []

        def broadcaster(agent_id: str):
            try:
                for i in range(200):
                    coord.broadcast_event(agent_id, {"seq": i})
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(200):
                    coord.get_recent_events(10)
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=broadcaster, args=(f"ag{i}",)) for i in range(4)]
            + [threading.Thread(target=reader) for _ in range(4)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
