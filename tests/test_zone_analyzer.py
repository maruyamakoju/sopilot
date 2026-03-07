"""Tests for Phase 46: ZoneAnalyzer (zone_analyzer.py).

Tests cover:
    - ZoneDefinition dataclass + to_dict
    - ZoneEvent dataclass + to_dict
    - _point_in_polygon (ray casting)
    - ZoneAnalyzer.add_zone / remove_zone / get_zones
    - ZoneAnalyzer.update (enter/exit/dwell events)
    - ZoneAnalyzer.get_dwell_stats / get_flow_vectors / get_recent_events
    - ZoneAnalyzer.get_stats
    - Thread safety (concurrent update)
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from sopilot.perception.zone_analyzer import (
    ZoneAnalyzer,
    ZoneDefinition,
    ZoneEvent,
    _point_in_polygon,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_point(x: float, y: float, person_id: str = "p1", camera_id: str = "cam01"):
    return SimpleNamespace(world_x=x, world_y=y, person_id=person_id, camera_id=camera_id)


def _square(cx: float = 0.0, cy: float = 0.0, size: float = 2.0):
    """Axis-aligned square polygon centred at (cx + size/2, cy + size/2)."""
    return [
        (cx, cy),
        (cx + size, cy),
        (cx + size, cy + size),
        (cx, cy + size),
    ]


# ---------------------------------------------------------------------------
# _point_in_polygon
# ---------------------------------------------------------------------------


class TestPointInPolygon:
    def test_centre_of_square_inside(self):
        poly = _square(0, 0, 4)
        assert _point_in_polygon(2.0, 2.0, poly) is True

    def test_point_outside_square(self):
        poly = _square(0, 0, 4)
        assert _point_in_polygon(5.0, 5.0, poly) is False

    def test_point_on_edge_no_crash(self):
        poly = _square(0, 0, 4)
        # Edge case: may be True or False depending on ray, but must not raise
        result = _point_in_polygon(0.0, 2.0, poly)
        assert isinstance(result, bool)

    def test_polygon_too_small_returns_false(self):
        assert _point_in_polygon(1.0, 1.0, [(0, 0), (1, 0)]) is False  # only 2 pts

    def test_triangle_inside(self):
        # Right triangle: (0,0), (4,0), (0,4)
        poly = [(0, 0), (4, 0), (0, 4)]
        assert _point_in_polygon(1.0, 1.0, poly) is True

    def test_triangle_outside(self):
        poly = [(0, 0), (4, 0), (0, 4)]
        assert _point_in_polygon(3.0, 3.0, poly) is False

    def test_concave_polygon_inside(self):
        # L-shape: inside the bottom rectangle
        poly = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
        assert _point_in_polygon(0.5, 0.5, poly) is True

    def test_negative_coords(self):
        poly = [(-2, -2), (2, -2), (2, 2), (-2, 2)]
        assert _point_in_polygon(0.0, 0.0, poly) is True
        assert _point_in_polygon(3.0, 0.0, poly) is False


# ---------------------------------------------------------------------------
# ZoneDefinition
# ---------------------------------------------------------------------------


class TestZoneDefinition:
    def test_fields(self):
        z = ZoneDefinition(
            zone_id="z1",
            name="Zone A",
            polygon=[(0, 0), (5, 0), (5, 3), (0, 3)],
            zone_type="restricted",
            color="#ef4444",
        )
        assert z.zone_id == "z1"
        assert z.zone_type == "restricted"
        assert len(z.polygon) == 4

    def test_to_dict_keys(self):
        z = ZoneDefinition("z1", "Zone A", [(0, 0), (1, 0), (1, 1), (0, 1)])
        d = z.to_dict()
        for key in ("zone_id", "name", "polygon", "zone_type", "color"):
            assert key in d

    def test_polygon_serialised_as_xy_dicts(self):
        z = ZoneDefinition("z1", "A", [(1.5, 2.5), (3.0, 2.5), (3.0, 4.0), (1.5, 4.0)])
        d = z.to_dict()
        assert d["polygon"][0] == {"x": 1.5, "y": 2.5}


# ---------------------------------------------------------------------------
# ZoneEvent
# ---------------------------------------------------------------------------


class TestZoneEvent:
    def _make(self, event_type: str = "enter", dwell: float = 0.0):
        return ZoneEvent(
            event_id="ev1",
            zone_id="z1",
            zone_name="Zone A",
            zone_type="restricted",
            person_id="p1",
            camera_id="cam01",
            event_type=event_type,
            timestamp=time.time(),
            dwell_seconds=dwell,
        )

    def test_fields(self):
        ev = self._make("enter")
        assert ev.event_type == "enter"
        assert ev.zone_type == "restricted"

    def test_to_dict_keys(self):
        ev = self._make("exit", dwell=5.0)
        d = ev.to_dict()
        for key in ("event_id", "zone_id", "zone_name", "zone_type",
                    "person_id", "camera_id", "event_type", "timestamp", "dwell_seconds"):
            assert key in d

    def test_dwell_rounded_in_dict(self):
        ev = self._make("dwell", dwell=12.3456)
        assert abs(ev.to_dict()["dwell_seconds"] - 12.3) < 0.05


# ---------------------------------------------------------------------------
# ZoneAnalyzer.add_zone / remove_zone / get_zones
# ---------------------------------------------------------------------------


class TestZoneManagement:
    def test_add_zone_returns_definition(self):
        a = ZoneAnalyzer()
        z = a.add_zone("z1", _square(), "normal")
        assert isinstance(z, ZoneDefinition)
        assert z.zone_id == "z1"

    def test_add_zone_restricted_color(self):
        a = ZoneAnalyzer()
        z = a.add_zone("z1", _square(), "restricted")
        assert z.color == "#ef4444"

    def test_add_zone_entry_color(self):
        a = ZoneAnalyzer()
        z = a.add_zone("z1", _square(), "entry")
        assert z.color == "#22c55e"

    def test_add_zone_exit_color(self):
        a = ZoneAnalyzer()
        z = a.add_zone("z1", _square(), "exit")
        assert z.color == "#f97316"

    def test_add_zone_custom_name(self):
        a = ZoneAnalyzer()
        z = a.add_zone("z1", _square(), name="立入禁止")
        assert z.name == "立入禁止"

    def test_get_zones_empty(self):
        a = ZoneAnalyzer()
        assert a.get_zones() == []

    def test_get_zones_returns_all(self):
        a = ZoneAnalyzer()
        a.add_zone("z1", _square(0, 0), "normal")
        a.add_zone("z2", _square(5, 5), "restricted")
        zones = a.get_zones()
        assert len(zones) == 2

    def test_remove_zone_existing(self):
        a = ZoneAnalyzer()
        a.add_zone("z1", _square())
        assert a.remove_zone("z1") is True
        assert len(a.get_zones()) == 0

    def test_remove_zone_nonexistent(self):
        a = ZoneAnalyzer()
        assert a.remove_zone("nonexistent") is False


# ---------------------------------------------------------------------------
# ZoneAnalyzer.update — enter / exit / dwell events
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_enter_event_generated(self):
        a = ZoneAnalyzer()
        a.add_zone("z1", _square(0, 0, 4), "normal")
        pts = [_make_point(2.0, 2.0, "p1")]
        events = a.update(pts, time.time())
        enter_events = [e for e in events if e.event_type == "enter"]
        assert len(enter_events) == 1

    def test_enter_event_zone_id(self):
        a = ZoneAnalyzer()
        a.add_zone("z1", _square(0, 0, 4), "normal")
        events = a.update([_make_point(2.0, 2.0, "p1")], time.time())
        assert events[0].zone_id == "z1"

    def test_enter_event_person_id(self):
        a = ZoneAnalyzer()
        a.add_zone("z1", _square(0, 0, 4))
        events = a.update([_make_point(2.0, 2.0, "person_A")], time.time())
        assert events[0].person_id == "person_A"

    def test_no_event_for_outside_point(self):
        a = ZoneAnalyzer()
        a.add_zone("z1", _square(0, 0, 2))
        events = a.update([_make_point(10.0, 10.0, "p1")], time.time())
        assert events == []

    def test_exit_event_generated(self):
        a = ZoneAnalyzer()
        a.add_zone("z1", _square(0, 0, 4))
        ts = time.time()
        # Frame 1: person enters
        a.update([_make_point(2.0, 2.0, "p1")], ts)
        # Frame 2: person leaves
        events = a.update([_make_point(10.0, 10.0, "p1")], ts + 1.0)
        exit_events = [e for e in events if e.event_type == "exit"]
        assert len(exit_events) == 1

    def test_exit_event_dwell_calculated(self):
        a = ZoneAnalyzer()
        a.add_zone("z1", _square(0, 0, 4))
        ts = time.time()
        a.update([_make_point(2.0, 2.0, "p1")], ts)
        events = a.update([_make_point(10.0, 10.0, "p1")], ts + 5.0)
        exit_ev = next(e for e in events if e.event_type == "exit")
        assert abs(exit_ev.dwell_seconds - 5.0) < 0.1

    def test_staying_generates_no_extra_enter(self):
        a = ZoneAnalyzer()
        a.add_zone("z1", _square(0, 0, 4))
        ts = time.time()
        a.update([_make_point(2.0, 2.0, "p1")], ts)
        events = a.update([_make_point(2.5, 2.5, "p1")], ts + 1.0)
        enter_events = [e for e in events if e.event_type == "enter"]
        assert len(enter_events) == 0

    def test_dwell_alert_for_restricted_zone(self):
        a = ZoneAnalyzer(dwell_alert_seconds=5.0)
        a.add_zone("z1", _square(0, 0, 4), "restricted")
        ts = time.time()
        a.update([_make_point(2.0, 2.0, "p1")], ts)
        # Jump 10 seconds ahead
        events = a.update([_make_point(2.0, 2.0, "p1")], ts + 10.0)
        dwell_events = [e for e in events if e.event_type == "dwell"]
        assert len(dwell_events) == 1

    def test_no_dwell_alert_for_normal_zone(self):
        a = ZoneAnalyzer(dwell_alert_seconds=5.0)
        a.add_zone("z1", _square(0, 0, 4), "normal")
        ts = time.time()
        a.update([_make_point(2.0, 2.0, "p1")], ts)
        events = a.update([_make_point(2.0, 2.0, "p1")], ts + 10.0)
        dwell_events = [e for e in events if e.event_type == "dwell"]
        assert len(dwell_events) == 0

    def test_multiple_persons_tracked_independently(self):
        a = ZoneAnalyzer()
        a.add_zone("z1", _square(0, 0, 4))
        ts = time.time()
        pts = [_make_point(2.0, 2.0, "p1"), _make_point(2.0, 2.0, "p2")]
        events = a.update(pts, ts)
        enter_events = [e for e in events if e.event_type == "enter"]
        assert len(enter_events) == 2

    def test_no_zones_returns_empty(self):
        a = ZoneAnalyzer()
        events = a.update([_make_point(2.0, 2.0, "p1")], time.time())
        assert events == []

    def test_empty_points_returns_empty(self):
        a = ZoneAnalyzer()
        a.add_zone("z1", _square(0, 0, 4))
        events = a.update([], time.time())
        assert events == []

    def test_flow_tracked_between_zones(self):
        a = ZoneAnalyzer()
        a.add_zone("z_entry", _square(0, 0, 2), "entry")
        a.add_zone("z_main", _square(3, 0, 2), "normal")
        ts = time.time()
        # Person starts in entry zone
        a.update([_make_point(1.0, 1.0, "p1")], ts)
        # Person moves to main zone
        a.update([_make_point(4.0, 1.0, "p1")], ts + 1.0)
        flows = a.get_flow_vectors()
        assert len(flows) >= 1
        assert flows[0]["from_zone"] == "z_entry"
        assert flows[0]["to_zone"] == "z_main"

    def test_update_count_increments(self):
        a = ZoneAnalyzer()
        a.update([], time.time())
        a.update([], time.time())
        assert a.get_stats()["update_count"] == 2


# ---------------------------------------------------------------------------
# ZoneAnalyzer.get_dwell_stats
# ---------------------------------------------------------------------------


class TestGetDwellStats:
    def test_returns_dict(self):
        a = ZoneAnalyzer()
        a.add_zone("z1", _square(0, 0, 4))
        stats = a.get_dwell_stats()
        assert isinstance(stats, dict)

    def test_zone_included(self):
        a = ZoneAnalyzer()
        a.add_zone("z1", _square(0, 0, 4))
        stats = a.get_dwell_stats()
        assert "z1" in stats

    def test_dwell_stats_keys(self):
        a = ZoneAnalyzer()
        a.add_zone("z1", _square(0, 0, 4))
        stats = a.get_dwell_stats()["z1"]
        for key in ("zone_name", "zone_type", "current_occupancy",
                    "total_visits", "avg_dwell_seconds", "max_dwell_seconds"):
            assert key in stats

    def test_total_visits_after_exit(self):
        a = ZoneAnalyzer()
        a.add_zone("z1", _square(0, 0, 4))
        ts = time.time()
        a.update([_make_point(2.0, 2.0, "p1")], ts)
        a.update([_make_point(10.0, 10.0, "p1")], ts + 3.0)
        stats = a.get_dwell_stats()["z1"]
        assert stats["total_visits"] == 1

    def test_current_occupancy(self):
        a = ZoneAnalyzer()
        a.add_zone("z1", _square(0, 0, 4))
        a.update([_make_point(2.0, 2.0, "p1"), _make_point(1.5, 1.5, "p2")], time.time())
        stats = a.get_dwell_stats()["z1"]
        assert stats["current_occupancy"] == 2


# ---------------------------------------------------------------------------
# ZoneAnalyzer.get_recent_events
# ---------------------------------------------------------------------------


class TestGetRecentEvents:
    def test_returns_list(self):
        a = ZoneAnalyzer()
        a.add_zone("z1", _square(0, 0, 4))
        a.update([_make_point(2.0, 2.0, "p1")], time.time())
        events = a.get_recent_events()
        assert isinstance(events, list)

    def test_events_are_dicts(self):
        a = ZoneAnalyzer()
        a.add_zone("z1", _square(0, 0, 4))
        a.update([_make_point(2.0, 2.0, "p1")], time.time())
        events = a.get_recent_events()
        for ev in events:
            assert isinstance(ev, dict)

    def test_limit_applied(self):
        a = ZoneAnalyzer()
        a.add_zone("z1", _square(0, 0, 4))
        for i in range(10):
            a.update([_make_point(2.0, 2.0, f"p{i}")], time.time())
        events = a.get_recent_events(limit=3)
        assert len(events) <= 3

    def test_event_type_filter(self):
        a = ZoneAnalyzer()
        a.add_zone("z1", _square(0, 0, 4))
        ts = time.time()
        a.update([_make_point(2.0, 2.0, "p1")], ts)
        a.update([_make_point(10.0, 10.0, "p1")], ts + 1.0)
        enter_only = a.get_recent_events(event_type="enter")
        assert all(e["event_type"] == "enter" for e in enter_only)


# ---------------------------------------------------------------------------
# ZoneAnalyzer.get_stats
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_stats_structure(self):
        a = ZoneAnalyzer()
        stats = a.get_stats()
        for key in ("zones_registered", "events_total", "events_buffered",
                    "persons_tracked", "update_count", "dwell_threshold_seconds",
                    "uptime_seconds"):
            assert key in stats

    def test_zones_registered_count(self):
        a = ZoneAnalyzer()
        a.add_zone("z1", _square(0, 0, 2))
        a.add_zone("z2", _square(3, 3, 2))
        assert a.get_stats()["zones_registered"] == 2

    def test_events_total_increments(self):
        a = ZoneAnalyzer()
        a.add_zone("z1", _square(0, 0, 4))
        a.update([_make_point(2.0, 2.0, "p1")], time.time())
        assert a.get_stats()["events_total"] >= 1


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_update(self):
        a = ZoneAnalyzer()
        a.add_zone("z1", _square(0, 0, 4))
        errors: list[Exception] = []

        def worker(i: int):
            try:
                pts = [_make_point(2.0, 2.0, f"p{i}")]
                a.update(pts, time.time())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_concurrent_add_zone_and_update(self):
        a = ZoneAnalyzer()
        errors: list[Exception] = []

        def zone_worker(i: int):
            try:
                a.add_zone(f"z{i}", _square(i, i, 1))
            except Exception as e:
                errors.append(e)

        def update_worker():
            try:
                a.update([_make_point(0.5, 0.5, "p1")], time.time())
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=zone_worker, args=(i,)) for i in range(10)]
            + [threading.Thread(target=update_worker) for _ in range(15)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
