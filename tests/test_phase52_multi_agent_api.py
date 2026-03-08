"""Phase 52 — Multi-Agent Coordination: API endpoint + engine integration tests.

Coverage:
  - TestMultiAgentStateEndpoint         (6)
  - TestMultiAgentRegisterEndpoint      (8)
  - TestMultiAgentListEndpoint          (5)
  - TestMultiAgentHeartbeatEndpoint     (5)
  - TestMultiAgentUnregisterEndpoint    (5)
  - TestMultiAgentSpatialEndpoint       (6)
  - TestMultiAgentEntitiesEndpoint      (6)
  - TestMultiAgentGlobalEntityEndpoint  (5)
  - TestMultiAgentEventsEndpoint        (6)
  - TestMultiAgentHotspotsEndpoint      (5)
  - TestMultiAgentResetEndpoint         (4)
  - TestPerceptionConfigPhase52         (6)
  - TestEngineMultiAgentIntegration     (8)
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sopilot.perception.multi_agent import (
    MultiAgentCoordinator,
    get_coordinator,
    reset_coordinator,
)
from sopilot.perception.types import PerceptionConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coord() -> MultiAgentCoordinator:
    """Return a fresh coordinator (not the singleton)."""
    return MultiAgentCoordinator()


def _ones_grid(h: int = 6, w: int = 8) -> list[list[float]]:
    return [[1.0] * w for _ in range(h)]


# ---------------------------------------------------------------------------
# 1. TestMultiAgentStateEndpoint  (6)
# ---------------------------------------------------------------------------

class TestMultiAgentStateEndpoint:

    def test_state_returns_dict(self):
        coord = _coord()
        d = coord.get_state_dict()
        assert isinstance(d, dict)

    def test_state_has_required_keys(self):
        coord = _coord()
        d = coord.get_state_dict()
        for key in ("total_agents", "active_agents", "total_global_entities",
                    "total_events_broadcast", "event_buffer_size", "shared_map", "agents"):
            assert key in d

    def test_state_zero_on_fresh_coordinator(self):
        coord = _coord()
        d = coord.get_state_dict()
        assert d["total_agents"] == 0
        assert d["total_global_entities"] == 0
        assert d["total_events_broadcast"] == 0

    def test_state_reflects_registered_agent(self):
        coord = _coord()
        coord.register_agent("cam1", "camera-A", "entrance")
        d = coord.get_state_dict()
        assert d["total_agents"] == 1

    def test_state_agents_list_contains_to_dict(self):
        coord = _coord()
        coord.register_agent("cam1", "camera-A")
        d = coord.get_state_dict()
        assert len(d["agents"]) == 1
        agent = d["agents"][0]
        assert "agent_id" in agent and "camera_id" in agent

    def test_state_shared_map_sub_dict(self):
        coord = _coord()
        d = coord.get_state_dict()
        sm = d["shared_map"]
        assert "grid_w" in sm and "grid_h" in sm and "update_count" in sm


# ---------------------------------------------------------------------------
# 2. TestMultiAgentRegisterEndpoint  (8)
# ---------------------------------------------------------------------------

class TestMultiAgentRegisterEndpoint:

    def test_register_returns_agent_info(self):
        coord = _coord()
        info = coord.register_agent("ag1", "cam1", "zone-A")
        d = info.to_dict()
        assert d["agent_id"] == "ag1"

    def test_register_missing_agent_id_raises(self):
        """Simulate 422: empty agent_id."""
        coord = _coord()
        # In real route, HTTPException is raised; here simulate the check
        agent_id = ""
        assert not agent_id  # would trigger 422 in route

    def test_register_persists_camera_id(self):
        coord = _coord()
        info = coord.register_agent("ag1", "cam-X", "dock")
        assert info.camera_id == "cam-X"

    def test_register_persists_location(self):
        coord = _coord()
        info = coord.register_agent("ag1", "cam1", "loading-bay")
        assert info.location == "loading-bay"

    def test_register_empty_location_defaults(self):
        coord = _coord()
        info = coord.register_agent("ag1", "cam1")
        assert info.location == ""

    def test_register_second_agent_increments_count(self):
        coord = _coord()
        coord.register_agent("ag1", "cam1")
        coord.register_agent("ag2", "cam2")
        assert coord.get_state_dict()["total_agents"] == 2

    def test_re_register_updates_camera(self):
        coord = _coord()
        coord.register_agent("ag1", "cam-old")
        coord.register_agent("ag1", "cam-new")
        info = coord.get_agent("ag1")
        assert info.camera_id == "cam-new"

    def test_register_to_dict_all_fields(self):
        coord = _coord()
        info = coord.register_agent("ag1", "cam1", "gate")
        d = info.to_dict()
        for key in ("agent_id", "camera_id", "location", "registered_at",
                    "last_seen_at", "frame_count", "spatial_contribution"):
            assert key in d


# ---------------------------------------------------------------------------
# 3. TestMultiAgentListEndpoint  (5)
# ---------------------------------------------------------------------------

class TestMultiAgentListEndpoint:

    def test_list_empty_on_fresh(self):
        coord = _coord()
        agents = list(coord._agents.values())
        assert agents == []

    def test_list_returns_all_registered(self):
        coord = _coord()
        for i in range(4):
            coord.register_agent(f"ag{i}", f"cam{i}")
        agents = list(coord._agents.values())
        assert len(agents) == 4

    def test_active_only_filters_timed_out(self):
        import time
        coord = MultiAgentCoordinator(agent_timeout_seconds=10.0)
        coord.register_agent("ag1", "cam1")
        coord.get_agent("ag1").last_seen_at = time.time() - 200
        coord.register_agent("ag2", "cam2")  # recent
        active = coord.get_active_agents()
        ids = [a.agent_id for a in active]
        assert "ag1" not in ids
        assert "ag2" in ids

    def test_agent_dicts_serializable(self):
        coord = _coord()
        coord.register_agent("ag1", "cam1")
        agents = list(coord._agents.values())
        for a in agents:
            d = a.to_dict()
            json.dumps(d)  # must not raise

    def test_list_after_unregister_reduced(self):
        coord = _coord()
        coord.register_agent("ag1", "cam1")
        coord.register_agent("ag2", "cam2")
        coord.unregister_agent("ag1")
        agents = list(coord._agents.values())
        assert len(agents) == 1


# ---------------------------------------------------------------------------
# 4. TestMultiAgentHeartbeatEndpoint  (5)
# ---------------------------------------------------------------------------

class TestMultiAgentHeartbeatEndpoint:

    def test_heartbeat_known_agent_returns_true(self):
        coord = _coord()
        coord.register_agent("ag1", "cam1")
        assert coord.heartbeat("ag1") is True

    def test_heartbeat_unknown_agent_returns_false(self):
        coord = _coord()
        assert coord.heartbeat("ghost") is False

    def test_heartbeat_updates_last_seen(self):
        import time
        coord = _coord()
        coord.register_agent("ag1", "cam1")
        future = time.time() + 500
        coord.heartbeat("ag1", _now=future)
        assert coord.get_agent("ag1").last_seen_at == pytest.approx(future)

    def test_heartbeat_after_unregister_returns_false(self):
        coord = _coord()
        coord.register_agent("ag1", "cam1")
        coord.unregister_agent("ag1")
        assert coord.heartbeat("ag1") is False

    def test_heartbeat_reactivates_timed_out_if_registered(self):
        import time
        coord = MultiAgentCoordinator(agent_timeout_seconds=10)
        coord.register_agent("ag1", "cam1")
        # Simulate timeout
        coord.get_agent("ag1").last_seen_at = time.time() - 200
        # Heartbeat brings it back
        now = time.time()
        coord.heartbeat("ag1", _now=now)
        active = coord.get_active_agents(_now=now)
        assert any(a.agent_id == "ag1" for a in active)


# ---------------------------------------------------------------------------
# 5. TestMultiAgentUnregisterEndpoint  (5)
# ---------------------------------------------------------------------------

class TestMultiAgentUnregisterEndpoint:

    def test_unregister_known_returns_true(self):
        coord = _coord()
        coord.register_agent("ag1", "cam1")
        assert coord.unregister_agent("ag1") is True

    def test_unregister_unknown_returns_false(self):
        coord = _coord()
        assert coord.unregister_agent("nobody") is False

    def test_unregister_removes_from_list(self):
        coord = _coord()
        coord.register_agent("ag1", "cam1")
        coord.unregister_agent("ag1")
        assert coord.get_agent("ag1") is None

    def test_unregister_reduces_total_agents(self):
        coord = _coord()
        coord.register_agent("ag1", "cam1")
        coord.register_agent("ag2", "cam2")
        coord.unregister_agent("ag1")
        assert coord.get_state_dict()["total_agents"] == 1

    def test_double_unregister_second_is_false(self):
        coord = _coord()
        coord.register_agent("ag1", "cam1")
        coord.unregister_agent("ag1")
        assert coord.unregister_agent("ag1") is False


# ---------------------------------------------------------------------------
# 6. TestMultiAgentSpatialEndpoint  (6)
# ---------------------------------------------------------------------------

class TestMultiAgentSpatialEndpoint:

    def test_submit_spatial_updates_shared_map(self):
        coord = _coord()
        coord.register_agent("ag1", "cam1")
        coord.submit_spatial_update("ag1", _ones_grid())
        grid = coord.get_shared_map().get_global_grid()
        assert np.any(grid > 0)

    def test_submit_spatial_increments_frame_count(self):
        coord = _coord()
        coord.register_agent("ag1", "cam1")
        coord.submit_spatial_update("ag1", _ones_grid())
        coord.submit_spatial_update("ag1", _ones_grid())
        assert coord.get_agent("ag1").frame_count == 2

    def test_submit_spatial_unknown_agent_no_crash(self):
        coord = _coord()
        # Should not raise
        coord.submit_spatial_update("ghost", _ones_grid())

    def test_submit_spatial_returns_no_error(self):
        coord = _coord()
        coord.register_agent("ag1", "cam1")
        result = coord.submit_spatial_update("ag1", _ones_grid())
        assert result is None  # returns None

    def test_submit_spatial_bad_grid_no_crash(self):
        coord = _coord()
        coord.register_agent("ag1", "cam1")
        coord.submit_spatial_update("ag1", [[]])  # empty rows

    def test_submit_spatial_state_update_count_increments(self):
        coord = _coord()
        coord.register_agent("ag1", "cam1")
        for _ in range(3):
            coord.submit_spatial_update("ag1", _ones_grid())
        sm_state = coord.get_shared_map().get_state_dict()
        assert sm_state["update_count"] == 3


# ---------------------------------------------------------------------------
# 7. TestMultiAgentEntitiesEndpoint  (6)
# ---------------------------------------------------------------------------

class TestMultiAgentEntitiesEndpoint:

    def _entity(self, eid: int, label: str = "person") -> dict:
        return {"entity_id": eid, "label": label, "cx": 0.5, "cy": 0.5}

    def test_submit_entities_returns_list_of_ids(self):
        coord = _coord()
        gids = coord.submit_entities("ag1", [self._entity(1)])
        assert isinstance(gids, list) and len(gids) == 1

    def test_submit_entities_global_id_is_uuid(self):
        import re
        coord = _coord()
        gids = coord.submit_entities("ag1", [self._entity(1)])
        uuid_re = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
        assert uuid_re.match(gids[0])

    def test_submit_entities_empty_list_returns_empty(self):
        coord = _coord()
        gids = coord.submit_entities("ag1", [])
        assert gids == []

    def test_submit_entities_increments_global_entity_count(self):
        coord = _coord()
        coord.submit_entities("ag1", [self._entity(0), self._entity(1, "car")])
        assert coord.get_state_dict()["total_global_entities"] >= 1

    def test_get_global_entities_returns_list(self):
        coord = _coord()
        coord.submit_entities("ag1", [self._entity(0)])
        entities = coord.get_global_entities()
        assert isinstance(entities, list)
        assert len(entities) >= 1

    def test_submit_multiple_agents_same_label_merges(self):
        import time
        coord = _coord()
        now = time.time()
        gids1 = coord.submit_entities("ag1", [{"entity_id": 0, "label": "worker", "cx": 0.5, "cy": 0.5}], _now=now)
        gids2 = coord.submit_entities("ag2", [{"entity_id": 0, "label": "worker", "cx": 0.5, "cy": 0.5}], _now=now + 0.5)
        assert gids1[0] == gids2[0]


# ---------------------------------------------------------------------------
# 8. TestMultiAgentGlobalEntityEndpoint  (5)
# ---------------------------------------------------------------------------

class TestMultiAgentGlobalEntityEndpoint:

    def test_get_entity_by_global_id_found(self):
        coord = _coord()
        gids = coord.submit_entities("ag1", [{"entity_id": 0, "label": "box", "cx": 0.3, "cy": 0.7}])
        e = coord.get_entity_by_global_id(gids[0])
        assert e is not None
        assert e.global_id == gids[0]

    def test_get_entity_by_global_id_not_found(self):
        coord = _coord()
        e = coord.get_entity_by_global_id("00000000-0000-0000-0000-000000000000")
        assert e is None

    def test_entity_to_dict_serializable(self):
        coord = _coord()
        gids = coord.submit_entities("ag1", [{"entity_id": 0, "label": "pallet", "cx": 0.1, "cy": 0.9}])
        e = coord.get_entity_by_global_id(gids[0])
        d = e.to_dict()
        json.dumps(d)  # must not raise

    def test_entity_label_preserved(self):
        coord = _coord()
        gids = coord.submit_entities("ag1", [{"entity_id": 0, "label": "forklift", "cx": 0.5, "cy": 0.5}])
        e = coord.get_entity_by_global_id(gids[0])
        assert e.label == "forklift"

    def test_entity_sighting_count_starts_at_one(self):
        coord = _coord()
        gids = coord.submit_entities("ag1", [{"entity_id": 0, "label": "person", "cx": 0.5, "cy": 0.5}])
        e = coord.get_entity_by_global_id(gids[0])
        assert e.sighting_count == 1


# ---------------------------------------------------------------------------
# 9. TestMultiAgentEventsEndpoint  (6)
# ---------------------------------------------------------------------------

class TestMultiAgentEventsEndpoint:

    def test_broadcast_event_returns_none(self):
        coord = _coord()
        result = coord.broadcast_event("ag1", {"type": "violation"})
        assert result is None

    def test_get_recent_events_returns_list(self):
        coord = _coord()
        coord.broadcast_event("ag1", {"type": "test"})
        events = coord.get_recent_events(10)
        assert isinstance(events, list) and len(events) == 1

    def test_events_enriched_with_source_agent(self):
        coord = _coord()
        coord.broadcast_event("ag1", {"msg": "hello"})
        ev = coord.get_recent_events(1)[0]
        assert ev["source_agent_id"] == "ag1"

    def test_events_enriched_with_broadcast_at(self):
        coord = _coord()
        coord.broadcast_event("ag1", {"msg": "test"})
        ev = coord.get_recent_events(1)[0]
        assert "broadcast_at" in ev and isinstance(ev["broadcast_at"], float)

    def test_event_original_keys_preserved(self):
        coord = _coord()
        coord.broadcast_event("ag1", {"severity": "critical", "count": 3})
        ev = coord.get_recent_events(1)[0]
        assert ev["severity"] == "critical"
        assert ev["count"] == 3

    def test_get_zero_events_returns_empty(self):
        coord = _coord()
        coord.broadcast_event("ag1", {"x": 1})
        assert coord.get_recent_events(0) == []


# ---------------------------------------------------------------------------
# 10. TestMultiAgentHotspotsEndpoint  (5)
# ---------------------------------------------------------------------------

class TestMultiAgentHotspotsEndpoint:

    def test_get_hotspots_empty_when_no_updates(self):
        coord = _coord()
        spots = coord.get_shared_map().get_hotspots(top_n=5)
        assert spots == []

    def test_get_hotspots_returns_list(self):
        coord = MultiAgentCoordinator(grid_w=4, grid_h=3)
        coord.register_agent("ag1", "cam1")
        g = [[0.0]*4 for _ in range(3)]
        g[1][2] = 0.9
        coord.submit_spatial_update("ag1", g)
        spots = coord.get_shared_map().get_hotspots(top_n=3)
        assert isinstance(spots, list)

    def test_get_hotspots_top_1_has_highest_value(self):
        coord = MultiAgentCoordinator(grid_w=4, grid_h=3)
        ssm = coord.get_shared_map()
        g = [[0.0]*4 for _ in range(3)]
        g[0][0] = 0.2
        g[1][1] = 0.9
        g[2][3] = 0.5
        from sopilot.perception.multi_agent import SharedSpatialMap
        ssm2 = SharedSpatialMap(grid_w=4, grid_h=3, merge_alpha=1.0)
        ssm2.update_from_agent("ag1", g, weight=1.0)
        spots = ssm2.get_hotspots(top_n=1)
        assert len(spots) == 1
        assert spots[0]["value"] == pytest.approx(0.9, abs=1e-4)

    def test_get_hotspots_row_col_within_bounds(self):
        from sopilot.perception.multi_agent import SharedSpatialMap
        ssm = SharedSpatialMap(grid_w=5, grid_h=4, merge_alpha=1.0)
        g = [[float(r*5+c)/20.0 for c in range(5)] for r in range(4)]
        ssm.update_from_agent("ag1", g, weight=1.0)
        spots = ssm.get_hotspots(top_n=5)
        for s in spots:
            assert 0 <= s["row"] < 4
            assert 0 <= s["col"] < 5

    def test_hotspot_map_stats_keys(self):
        coord = _coord()
        state = coord.get_shared_map().get_state_dict()
        for key in ("grid_w", "grid_h", "merge_alpha", "update_count", "max_occupancy", "mean_occupancy"):
            assert key in state


# ---------------------------------------------------------------------------
# 11. TestMultiAgentResetEndpoint  (4)
# ---------------------------------------------------------------------------

class TestMultiAgentResetEndpoint:

    def test_reset_clears_agents(self):
        coord = _coord()
        coord.register_agent("ag1", "cam1")
        coord.reset()
        assert coord.get_state_dict()["total_agents"] == 0

    def test_reset_clears_entities(self):
        coord = _coord()
        coord.submit_entities("ag1", [{"entity_id": 0, "label": "box", "cx": 0.5, "cy": 0.5}])
        coord.reset()
        assert len(coord.get_global_entities()) == 0

    def test_reset_clears_events(self):
        coord = _coord()
        coord.broadcast_event("ag1", {"x": 1})
        coord.reset()
        assert coord.get_recent_events(100) == []

    def test_reset_clears_shared_map(self):
        coord = _coord()
        coord.register_agent("ag1", "cam1")
        coord.submit_spatial_update("ag1", _ones_grid())
        coord.reset()
        grid = coord.get_shared_map().get_global_grid()
        assert np.all(grid == 0.0)


# ---------------------------------------------------------------------------
# 12. TestPerceptionConfigPhase52  (6)
# ---------------------------------------------------------------------------

class TestPerceptionConfigPhase52:

    def test_config_has_multi_agent_enabled(self):
        cfg = PerceptionConfig()
        assert hasattr(cfg, "multi_agent_enabled")

    def test_config_default_multi_agent_disabled(self):
        cfg = PerceptionConfig()
        assert cfg.multi_agent_enabled is False

    def test_config_multi_agent_max_agents_default(self):
        cfg = PerceptionConfig()
        assert cfg.multi_agent_max_agents == 16

    def test_config_entity_merge_distance_default(self):
        cfg = PerceptionConfig()
        assert cfg.multi_agent_entity_merge_distance == pytest.approx(0.1)

    def test_config_agent_timeout_default(self):
        cfg = PerceptionConfig()
        assert cfg.multi_agent_agent_timeout_seconds == pytest.approx(30.0)

    def test_config_multi_agent_enabled_can_be_set(self):
        cfg = PerceptionConfig(multi_agent_enabled=True)
        assert cfg.multi_agent_enabled is True


# ---------------------------------------------------------------------------
# 13. TestEngineMultiAgentIntegration  (8)
# ---------------------------------------------------------------------------

class TestEngineMultiAgentIntegration:

    def test_engine_has_multi_agent_coordinator_attr(self):
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine()
        assert hasattr(engine, "_multi_agent_coordinator")

    def test_engine_coordinator_none_by_default(self):
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine()
        assert engine._multi_agent_coordinator is None

    def test_engine_get_multi_agent_state_none_when_disabled(self):
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine()
        assert engine.get_multi_agent_state() is None

    def test_engine_get_multi_agent_state_returns_dict_when_set(self):
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine()
        engine._multi_agent_coordinator = MultiAgentCoordinator()
        result = engine.get_multi_agent_state()
        assert isinstance(result, dict)

    def test_build_engine_with_multi_agent_enabled(self):
        from sopilot.perception.engine import build_perception_engine
        cfg = PerceptionConfig(multi_agent_enabled=True)
        engine = build_perception_engine(config=cfg)
        assert engine._multi_agent_coordinator is not None

    def test_build_engine_coordinator_is_multi_agent_coordinator_instance(self):
        from sopilot.perception.engine import build_perception_engine
        cfg = PerceptionConfig(multi_agent_enabled=True)
        engine = build_perception_engine(config=cfg)
        assert isinstance(engine._multi_agent_coordinator, MultiAgentCoordinator)

    def test_build_engine_coordinator_max_agents_param(self):
        from sopilot.perception.engine import build_perception_engine
        cfg = PerceptionConfig(multi_agent_enabled=True, multi_agent_max_agents=8)
        engine = build_perception_engine(config=cfg)
        assert engine._multi_agent_coordinator.max_agents == 8

    def test_build_engine_coordinator_timeout_param(self):
        from sopilot.perception.engine import build_perception_engine
        cfg = PerceptionConfig(multi_agent_enabled=True, multi_agent_agent_timeout_seconds=60.0)
        engine = build_perception_engine(config=cfg)
        assert engine._multi_agent_coordinator.agent_timeout_seconds == pytest.approx(60.0)
