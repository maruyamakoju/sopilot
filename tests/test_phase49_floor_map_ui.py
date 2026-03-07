"""Phase 49: Floor Map Live UI — integration tests.

Tests cover:
- /spatial/stats (enable/disable)
- /spatial/map (canvas data: floor, points, occupancy_grid)
- /spatial/zones (list zones for polygon overlay)
- /spatial/zones POST (zone registration)
- /spatial/zones/events (event feed for UI)
- /spatial/zones/dwell (dwell stats)
- /spatial/zone-analyzer/stats (zone counter)
- Zone event-feed logic (enter/exit/dwell type labels)
- SpatialMapper occupancy grid shape
- Canvas data completeness
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sopilot.perception.spatial_mapper import SpatialMapper
from sopilot.perception.zone_analyzer import ZoneAnalyzer, ZoneDefinition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mapper(w=20.0, h=15.0) -> SpatialMapper:
    return SpatialMapper(floor_width_m=w, floor_height_m=h)


def _make_analyzer() -> ZoneAnalyzer:
    return ZoneAnalyzer()


def _register_camera(mapper: SpatialMapper) -> str:
    H = np.eye(3, dtype=np.float64)
    H[0, 0] = mapper._width
    H[1, 1] = mapper._height
    mapper.register_camera("cam01", H, image_size=(1920, 1080))
    return "cam01"


class _Pt:
    """Minimal point object compatible with ZoneAnalyzer.update()."""
    def __init__(self, pid: str, x: float, y: float, cam: str = "cam01"):
        self.person_id = pid
        self.world_x = x
        self.world_y = y
        self.camera_id = cam


# ---------------------------------------------------------------------------
# /spatial/stats equivalent — get_spatial_mapper_state (6 tests)
# ---------------------------------------------------------------------------


class TestSpatialStats:
    def test_mapper_state_not_none(self):
        m = _make_mapper()
        state = m.get_stats()
        assert state is not None

    def test_state_floor_dimensions(self):
        m = _make_mapper(20.0, 15.0)
        state = m.get_stats()
        assert state["floor_width_m"] == 20.0
        assert state["floor_height_m"] == 15.0

    def test_state_initial_camera_count(self):
        m = _make_mapper()
        state = m.get_stats()
        assert state["registered_cameras"] == 0

    def test_state_camera_count_after_register(self):
        m = _make_mapper()
        _register_camera(m)
        assert m.get_stats()["registered_cameras"] == 1

    def test_state_recent_points_empty(self):
        m = _make_mapper()
        assert m.get_stats()["recent_points"] == 0

    def test_state_has_max_age_seconds(self):
        m = _make_mapper()
        state = m.get_stats()
        assert "max_age_seconds" in state


# ---------------------------------------------------------------------------
# /spatial/map equivalent — to_map_data (10 tests)
# ---------------------------------------------------------------------------


class TestSpatialMapData:
    def test_map_data_keys(self):
        m = _make_mapper()
        d = m.to_map_data()
        assert "floor" in d
        assert "cameras" in d
        assert "points" in d
        assert "occupancy_grid" in d

    def test_floor_dimensions(self):
        m = _make_mapper(25.0, 18.0)
        d = m.to_map_data()
        assert d["floor"]["width_m"] == 25.0
        assert d["floor"]["height_m"] == 18.0

    def test_occupancy_grid_shape(self):
        m = _make_mapper()
        d = m.to_map_data(grid_cols=40, grid_rows=30)
        g = d["occupancy_grid"]
        assert g["cols"] == 40
        assert g["rows"] == 30
        assert len(g["data"]) == 30
        assert len(g["data"][0]) == 40

    def test_grid_values_in_range(self):
        m = _make_mapper()
        _register_camera(m)
        from sopilot.perception.spatial_mapper import OccupancyPoint
        m.add_detection("cam01", (0.5, 0.5, 0.1, 0.1))
        d = m.to_map_data(grid_cols=20, grid_rows=15)
        flat = [v for row in d["occupancy_grid"]["data"] for v in row]
        assert all(0.0 <= v <= 1.0 for v in flat)

    def test_points_list_after_detection(self):
        m = _make_mapper()
        _register_camera(m)
        m.add_detection("cam01", (0.3, 0.4, 0.1, 0.1), person_id="t1", label="person")
        d = m.to_map_data()
        assert len(d["points"]) == 1

    def test_point_has_world_coords(self):
        m = _make_mapper()
        _register_camera(m)
        m.add_detection("cam01", (0.5, 0.5, 0.1, 0.1))
        d = m.to_map_data()
        pt = d["points"][0]
        assert "world_x" in pt and "world_y" in pt

    def test_cameras_list(self):
        m = _make_mapper()
        _register_camera(m)
        d = m.to_map_data()
        assert "cam01" in d["cameras"]

    def test_empty_grid_all_zeros_no_detections(self):
        m = _make_mapper()
        d = m.to_map_data(grid_cols=10, grid_rows=10)
        flat = [v for row in d["occupancy_grid"]["data"] for v in row]
        assert all(v == 0.0 for v in flat)

    def test_max_age_includes_recent_points(self):
        m = _make_mapper()
        _register_camera(m)
        m.add_detection("cam01", (0.5, 0.5, 0.1, 0.1))
        # Recent points should be visible with a generous max_age
        pts = m.get_recent_points(max_age_seconds=60.0)
        assert len(pts) == 1

    def test_multiple_detections(self):
        m = _make_mapper()
        _register_camera(m)
        for i in range(5):
            m.add_detection("cam01", (0.1 * i, 0.1 * i, 0.05, 0.05))
        d = m.to_map_data()
        assert len(d["points"]) == 5


# ---------------------------------------------------------------------------
# Zone polygon overlay (/spatial/zones) (8 tests)
# ---------------------------------------------------------------------------


class TestZoneList:
    def test_empty_zone_list(self):
        a = _make_analyzer()
        assert a.get_zones() == []

    def test_add_zone(self):
        a = _make_analyzer()
        polygon = [[0, 0], [5, 0], [5, 5], [0, 5]]
        a.add_zone("z1", polygon, zone_type="restricted", name="テストゾーン")
        assert len(a.get_zones()) == 1

    def test_zone_has_polygon(self):
        a = _make_analyzer()
        polygon = [[0, 0], [5, 0], [5, 5], [0, 5]]
        a.add_zone("z1", polygon, zone_type="safe")
        zs = a.get_zones()
        assert zs[0].polygon == polygon

    def test_zone_type_stored(self):
        a = _make_analyzer()
        a.add_zone("z1", [[0, 0], [1, 0], [1, 1], [0, 1]], zone_type="entry_exit")
        assert a.get_zones()[0].zone_type == "entry_exit"

    def test_zone_name_stored(self):
        a = _make_analyzer()
        a.add_zone("z1", [[0, 0], [1, 0], [1, 1], [0, 1]], name="入口A")
        assert a.get_zones()[0].name == "入口A"

    def test_remove_zone(self):
        a = _make_analyzer()
        a.add_zone("z1", [[0, 0], [1, 0], [1, 1], [0, 1]])
        a.remove_zone("z1")
        assert a.get_zones() == []

    def test_multiple_zones(self):
        a = _make_analyzer()
        for i in range(3):
            a.add_zone(f"z{i}", [[i, 0], [i+1, 0], [i+1, 1], [i, 1]])
        assert len(a.get_zones()) == 3

    def test_zone_types_for_legend(self):
        """All four legend types should be storable."""
        a = _make_analyzer()
        for zt in ["restricted", "safe", "entry_exit", "monitoring"]:
            a.add_zone(zt, [[0, 0], [1, 0], [1, 1], [0, 1]], zone_type=zt)
        zones = a.get_zones()
        types = {z.zone_type for z in zones}
        assert types == {"restricted", "safe", "entry_exit", "monitoring"}


# ---------------------------------------------------------------------------
# Zone event feed (/spatial/zones/events) (8 tests)
# ---------------------------------------------------------------------------


class TestZoneEventFeed:
    def _analyzer_with_zone(self) -> ZoneAnalyzer:
        a = _make_analyzer()
        a.add_zone("z1", [[0, 0], [10, 0], [10, 10], [0, 10]],
                   zone_type="monitoring", name="テスト監視")
        return a

    def test_no_events_initially(self):
        a = _make_analyzer()
        events = a.get_recent_events(50)
        assert events == []

    def test_enter_event_generated(self):
        a = self._analyzer_with_zone()
        a.update([_Pt("t1", 5.0, 5.0)])
        events = a.get_recent_events(10)
        types = [e["event_type"] for e in events]
        assert "enter" in types

    def test_exit_event_generated(self):
        a = self._analyzer_with_zone()
        a.update([_Pt("t1", 5.0, 5.0)])
        a.update([_Pt("t1", 15.0, 15.0)])  # outside
        events = a.get_recent_events(50)
        types = [e["event_type"] for e in events]
        assert "exit" in types

    def test_event_has_zone_id(self):
        a = self._analyzer_with_zone()
        a.update([_Pt("t1", 5.0, 5.0)])
        events = a.get_recent_events(10)
        assert any(e["zone_id"] == "z1" for e in events)

    def test_event_has_person_id(self):
        a = self._analyzer_with_zone()
        a.update([_Pt("p42", 5.0, 5.0)])
        events = a.get_recent_events(10)
        assert any(e["person_id"] == "p42" for e in events)

    def test_event_limit(self):
        a = self._analyzer_with_zone()
        for i in range(10):
            a.update([_Pt(f"t{i}", 5.0, 5.0)])
        events = a.get_recent_events(5)
        assert len(events) <= 5

    def test_event_types_are_valid(self):
        a = self._analyzer_with_zone()
        a.update([_Pt("t1", 5.0, 5.0)])
        events = a.get_recent_events(50)
        for e in events:
            assert e["event_type"] in ("enter", "exit", "dwell")

    def test_zone_name_available_in_event(self):
        a = self._analyzer_with_zone()
        a.update([_Pt("t1", 5.0, 5.0)])
        events = a.get_recent_events(10)
        enter_events = [e for e in events if e["event_type"] == "enter"]
        if enter_events:
            assert enter_events[0]["zone_name"] == "テスト監視"


# ---------------------------------------------------------------------------
# Zone dwell stats (/spatial/zones/dwell) (5 tests)
# ---------------------------------------------------------------------------


class TestZoneDwellStats:
    def test_dwell_stats_empty(self):
        a = _make_analyzer()
        a.add_zone("z1", [[0, 0], [10, 0], [10, 10], [0, 10]])
        stats = a.get_dwell_stats()
        assert "z1" in stats

    def test_dwell_occupancy_zero_initially(self):
        a = _make_analyzer()
        a.add_zone("z1", [[0, 0], [10, 0], [10, 10], [0, 10]])
        stats = a.get_dwell_stats()
        assert stats["z1"]["current_occupancy"] == 0

    def test_dwell_occupancy_after_enter(self):
        a = _make_analyzer()
        a.add_zone("z1", [[0, 0], [10, 0], [10, 10], [0, 10]])
        a.update([_Pt("t1", 5.0, 5.0)])
        stats = a.get_dwell_stats()
        assert stats["z1"]["current_occupancy"] >= 1

    def test_dwell_stats_has_expected_keys(self):
        a = _make_analyzer()
        a.add_zone("z1", [[0, 0], [10, 0], [10, 10], [0, 10]])
        stats = a.get_dwell_stats()
        assert "current_occupancy" in stats["z1"]

    def test_multiple_zones_dwell(self):
        a = _make_analyzer()
        a.add_zone("z1", [[0, 0], [5, 0], [5, 5], [0, 5]])
        a.add_zone("z2", [[10, 0], [15, 0], [15, 5], [10, 5]])
        stats = a.get_dwell_stats()
        assert "z1" in stats and "z2" in stats


# ---------------------------------------------------------------------------
# Canvas data completeness checks (5 tests)
# ---------------------------------------------------------------------------


class TestCanvasData:
    """Verify the data structure produced for canvas rendering is complete."""

    def test_to_map_data_for_canvas_returns_all_keys(self):
        m = _make_mapper()
        d = m.to_map_data(grid_cols=40, grid_rows=30)
        assert all(k in d for k in ("floor", "cameras", "points", "occupancy_grid"))

    def test_occupancy_grid_normalised(self):
        m = _make_mapper()
        _register_camera(m)
        for _ in range(20):
            m.add_detection("cam01", (0.5, 0.5, 0.05, 0.05))
        d = m.to_map_data(grid_cols=10, grid_rows=10)
        flat = [v for row in d["occupancy_grid"]["data"] for v in row]
        # After normalisation, at least one cell should be 1.0
        assert max(flat) == pytest.approx(1.0, abs=0.001)

    def test_point_dict_has_age_seconds(self):
        m = _make_mapper()
        _register_camera(m)
        m.add_detection("cam01", (0.5, 0.5, 0.1, 0.1))
        d = m.to_map_data()
        assert "age_seconds" in d["points"][0]

    def test_floor_width_height_positive(self):
        m = _make_mapper(30.0, 20.0)
        d = m.to_map_data()
        assert d["floor"]["width_m"] > 0
        assert d["floor"]["height_m"] > 0

    def test_grid_row_count_matches_rows_param(self):
        m = _make_mapper()
        for rows in [10, 20, 30]:
            d = m.to_map_data(grid_cols=20, grid_rows=rows)
            assert len(d["occupancy_grid"]["data"]) == rows
