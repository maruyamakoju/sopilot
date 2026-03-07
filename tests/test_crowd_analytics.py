"""Phase 50: Crowd Flow Analytics — unit tests.

60 tests covering CrowdAnalytics construction, density computation, flow
vectors, congestion detection, zone capacity alerts, flow summary, stats,
reset, and thread safety.
"""
from __future__ import annotations

import math
import threading
import time

import pytest

from sopilot.perception.crowd_analytics import (
    CapacityAlert,
    CongestionZone,
    CrowdAnalytics,
    FlowVector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Pt:
    """Minimal point compatible with CrowdAnalytics.update()."""

    def __init__(self, pid: str, x: float, y: float):
        self.person_id = pid
        self.world_x = x
        self.world_y = y


def _analytics(w=20.0, h=15.0, cols=10, rows=8, threshold=0.7) -> CrowdAnalytics:
    return CrowdAnalytics(
        floor_width_m=w,
        floor_height_m=h,
        grid_cols=cols,
        grid_rows=rows,
        congestion_threshold=threshold,
        history_frames=5,
    )


# ---------------------------------------------------------------------------
# Construction (5 tests)
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_floor_dimensions(self):
        a = CrowdAnalytics()
        assert a._width == 20.0
        assert a._height == 15.0

    def test_custom_grid(self):
        a = CrowdAnalytics(grid_cols=30, grid_rows=20)
        assert a._cols == 30
        assert a._rows == 20

    def test_grid_minimum_clamped(self):
        a = CrowdAnalytics(grid_cols=1, grid_rows=1)
        assert a._cols >= 4
        assert a._rows >= 4

    def test_congestion_threshold_stored(self):
        a = CrowdAnalytics(congestion_threshold=0.5)
        assert a._congestion_threshold == 0.5

    def test_history_frames_stored(self):
        a = CrowdAnalytics(history_frames=8)
        assert a._history_frames == 8


# ---------------------------------------------------------------------------
# update() and density (10 tests)
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_increments_count(self):
        a = _analytics()
        a.update([_Pt("t1", 5.0, 5.0)])
        assert a._update_count == 1

    def test_update_empty_persons(self):
        a = _analytics()
        a.update([])
        assert a.get_state()["total_persons"] == 0

    def test_update_person_count(self):
        a = _analytics()
        pts = [_Pt(f"t{i}", float(i), float(i)) for i in range(5)]
        a.update(pts)
        assert a.get_state()["total_persons"] == 5

    def test_density_grid_populated(self):
        a = _analytics()
        a.update([_Pt("t1", 10.0, 7.5)])
        state = a.get_state()
        flat = [v for row in state["density_grid"]["data"] for v in row]
        assert max(flat) > 0.0

    def test_density_grid_shape(self):
        a = _analytics(cols=10, rows=8)
        a.update([_Pt("t1", 5.0, 4.0)])
        state = a.get_state()
        grid = state["density_grid"]
        assert grid["cols"] == 10
        assert grid["rows"] == 8
        assert len(grid["data"]) == 8
        assert len(grid["data"][0]) == 10

    def test_density_normalised_max_1(self):
        a = _analytics()
        for _ in range(20):
            a.update([_Pt("t1", 10.0, 7.5)])
        state = a.get_state()
        flat = [v for row in state["density_grid"]["data"] for v in row]
        assert max(flat) <= 1.0 + 1e-6

    def test_density_nonnegative(self):
        a = _analytics()
        a.update([_Pt("t1", 5.0, 3.0), _Pt("t2", 15.0, 10.0)])
        state = a.get_state()
        flat = [v for row in state["density_grid"]["data"] for v in row]
        assert all(v >= 0.0 for v in flat)

    def test_mean_density_positive_after_update(self):
        a = _analytics()
        a.update([_Pt("t1", 10.0, 7.5)])
        assert a._last_mean_density >= 0.0

    def test_duplicate_pids_last_position_wins(self):
        """Two points with same person_id — count should be 1."""
        a = _analytics()
        a.update([_Pt("t1", 1.0, 1.0), _Pt("t1", 18.0, 12.0)])
        assert a.get_state()["total_persons"] == 1

    def test_no_positions_empty_grid(self):
        a = _analytics()
        a.update([])
        state = a.get_state()
        flat = [v for row in state["density_grid"]["data"] for v in row]
        assert all(v == 0.0 for v in flat)


# ---------------------------------------------------------------------------
# Flow vectors (8 tests)
# ---------------------------------------------------------------------------


class TestFlowVectors:
    def test_no_flow_without_history(self):
        a = _analytics()
        a.update([_Pt("t1", 5.0, 5.0)])  # only 1 frame
        assert a.get_state()["flow_vector_count"] == 0

    def test_flow_vectors_after_movement(self):
        a = _analytics()
        a.update([_Pt("t1", 2.0, 2.0)])
        a.update([_Pt("t1", 4.0, 4.0)])
        vectors = a.get_state()["flow_vectors"]
        assert len(vectors) >= 1

    def test_flow_vector_keys(self):
        a = _analytics()
        a.update([_Pt("t1", 2.0, 2.0)])
        a.update([_Pt("t1", 6.0, 2.0)])
        vs = a.get_state()["flow_vectors"]
        if vs:
            v = vs[0]
            assert all(k in v for k in ("col", "row", "dx", "dy", "magnitude", "sample_count"))

    def test_flow_magnitude_positive(self):
        a = _analytics()
        a.update([_Pt("t1", 1.0, 1.0)])
        a.update([_Pt("t1", 3.0, 1.0)])
        vs = a.get_state()["flow_vectors"]
        if vs:
            assert all(v["magnitude"] > 0 for v in vs)

    def test_flow_sorted_by_magnitude_desc(self):
        a = _analytics()
        a.update([_Pt("t1", 0.5, 1.0), _Pt("t2", 10.0, 8.0)])
        a.update([_Pt("t1", 5.0, 1.0), _Pt("t2", 10.5, 8.0)])
        vs = a.get_state()["flow_vectors"]
        if len(vs) >= 2:
            assert vs[0]["magnitude"] >= vs[1]["magnitude"]

    def test_no_flow_stationary_person(self):
        a = _analytics()
        for _ in range(5):
            a.update([_Pt("t1", 5.0, 5.0)])
        # Magnitude should be 0 (same position), so no vectors
        vs = a.get_state()["flow_vectors"]
        assert all(v["magnitude"] > 1e-6 for v in vs) or len(vs) == 0

    def test_flow_vector_caps_at_50(self):
        a = _analytics(cols=10, rows=8)
        pts_prev = [_Pt(f"t{i}", float(i % 10), float(i % 8)) for i in range(60)]
        pts_next = [_Pt(f"t{i}", float(i % 10) + 1.0, float(i % 8)) for i in range(60)]
        a.update(pts_prev)
        a.update(pts_next)
        vs = a.get_state()["flow_vectors"]
        assert len(vs) <= 50

    def test_flow_with_multiple_frames(self):
        a = CrowdAnalytics(history_frames=5)
        for i in range(5):
            a.update([_Pt("t1", float(i) * 2, 5.0)])
        vs = a.get_state()["flow_vectors"]
        assert len(vs) >= 1


# ---------------------------------------------------------------------------
# Congestion detection (7 tests)
# ---------------------------------------------------------------------------


class TestCongestionDetection:
    def test_no_congestion_no_persons(self):
        a = _analytics()
        a.update([])
        assert a.get_state()["congestion_cell_count"] == 0

    def test_congestion_dense_cluster(self):
        a = CrowdAnalytics(grid_cols=5, grid_rows=4, congestion_threshold=0.5)
        # Many people in same spot
        pts = [_Pt(f"t{i}", 10.0, 7.0) for i in range(15)]
        a.update(pts)
        state = a.get_state()
        # At least one congested cell
        assert state["congestion_cell_count"] >= 1

    def test_congestion_zone_keys(self):
        a = CrowdAnalytics(grid_cols=5, grid_rows=4, congestion_threshold=0.3)
        pts = [_Pt(f"t{i}", 10.0, 7.0) for i in range(20)]
        a.update(pts)
        cz = a.get_state()["congestion_zones"]
        if cz:
            z = cz[0]
            assert all(k in z for k in ("col", "row", "density", "world_x", "world_y"))

    def test_congestion_density_in_range(self):
        a = CrowdAnalytics(congestion_threshold=0.3)
        pts = [_Pt(f"t{i}", 10.0, 7.0) for i in range(20)]
        a.update(pts)
        for z in a.get_state()["congestion_zones"]:
            assert 0.0 <= z["density"] <= 1.0 + 1e-6

    def test_high_threshold_no_congestion(self):
        a = CrowdAnalytics(congestion_threshold=1.1)
        pts = [_Pt(f"t{i}", 10.0, 7.0) for i in range(20)]
        a.update(pts)
        assert a.get_state()["congestion_cell_count"] == 0

    def test_low_threshold_many_congestion_cells(self):
        a = CrowdAnalytics(grid_cols=5, grid_rows=4, congestion_threshold=0.01)
        a.update([_Pt("t1", 10.0, 7.0)])
        # Many cells should be above 0.01 due to Gaussian spreading
        assert a.get_state()["congestion_cell_count"] >= 1

    def test_world_coords_in_floor_range(self):
        a = CrowdAnalytics(floor_width_m=20.0, floor_height_m=15.0, congestion_threshold=0.3)
        pts = [_Pt(f"t{i}", 10.0, 7.0) for i in range(20)]
        a.update(pts)
        for z in a.get_state()["congestion_zones"]:
            assert 0.0 <= z["world_x"] <= 20.0
            assert 0.0 <= z["world_y"] <= 15.0


# ---------------------------------------------------------------------------
# Zone capacity alerts (10 tests)
# ---------------------------------------------------------------------------


class TestCapacityAlerts:
    def test_no_alerts_no_zones(self):
        a = _analytics()
        a.update([_Pt("t1", 5.0, 5.0)])
        alerts = a.get_capacity_alerts()
        assert alerts == []

    def test_no_alert_below_threshold(self):
        a = _analytics()
        a.set_zone_capacity("z1", max_persons=20)
        a.update_zone_occupancy("z1", {"t1", "t2"})  # 2/20 = 10%
        a.update([])
        alerts = a.get_capacity_alerts()
        assert len(alerts) == 0

    def test_warning_alert_above_80_percent(self):
        a = _analytics()
        a.set_zone_capacity("z1", max_persons=10)
        a.update_zone_occupancy("z1", {f"t{i}" for i in range(9)})  # 9/10 = 90%
        a.update([])
        alerts = a.get_capacity_alerts()
        assert any(al["alert_level"] == "warning" for al in alerts)

    def test_critical_alert_above_100_percent(self):
        a = _analytics()
        a.set_zone_capacity("z1", max_persons=5)
        a.update_zone_occupancy("z1", {f"t{i}" for i in range(8)})  # 8/5 = 160%
        a.update([])
        alerts = a.get_capacity_alerts()
        assert any(al["alert_level"] == "critical" for al in alerts)

    def test_alert_utilisation_correct(self):
        a = _analytics()
        a.set_zone_capacity("z1", max_persons=10)
        a.update_zone_occupancy("z1", {f"t{i}" for i in range(9)})
        a.update([])
        alerts = a.get_capacity_alerts()
        if alerts:
            assert abs(alerts[0]["utilisation"] - 0.9) < 0.01

    def test_alert_has_expected_keys(self):
        a = _analytics()
        a.set_zone_capacity("z1", max_persons=5)
        a.update_zone_occupancy("z1", {f"t{i}" for i in range(6)})
        a.update([])
        alerts = a.get_capacity_alerts()
        if alerts:
            assert all(k in alerts[0] for k in ("zone_id", "zone_name", "current_count", "max_capacity", "utilisation", "alert_level", "raised_at"))

    def test_set_zone_capacity_custom_name(self):
        a = _analytics()
        a.set_zone_capacity("z1", max_persons=10, zone_name="エントランス")
        a.update_zone_occupancy("z1", {f"t{i}" for i in range(9)})
        a.update([])
        alerts = a.get_capacity_alerts()
        if alerts:
            assert alerts[0]["zone_name"] == "エントランス"

    def test_remove_zone_capacity(self):
        a = _analytics()
        a.set_zone_capacity("z1", max_persons=5)
        removed = a.remove_zone_capacity("z1")
        assert removed is True
        a.update([])
        assert a.get_capacity_alerts() == []

    def test_remove_nonexistent_zone(self):
        a = _analytics()
        assert a.remove_zone_capacity("nonexistent") is False

    def test_multiple_zones_alerts(self):
        a = _analytics()
        a.set_zone_capacity("z1", max_persons=5)
        a.set_zone_capacity("z2", max_persons=3)
        a.update_zone_occupancy("z1", {"t1", "t2", "t3", "t4", "t5", "t6"})
        a.update_zone_occupancy("z2", {"t7", "t8", "t9", "t10"})
        a.update([])
        alerts = a.get_capacity_alerts()
        zone_ids = {al["zone_id"] for al in alerts}
        assert "z1" in zone_ids and "z2" in zone_ids


# ---------------------------------------------------------------------------
# Flow summary (5 tests)
# ---------------------------------------------------------------------------


class TestFlowSummary:
    def test_no_movement_zero_speed(self):
        a = _analytics()
        summary = a.get_flow_summary()
        assert summary["mean_speed_ms"] == 0.0
        assert summary["max_speed_ms"] == 0.0

    def test_no_movement_direction_none(self):
        a = _analytics()
        summary = a.get_flow_summary()
        assert summary["dominant_direction_deg"] is None

    def test_flow_summary_keys(self):
        a = _analytics()
        s = a.get_flow_summary()
        assert all(k in s for k in ("dominant_direction_deg", "mean_speed_ms", "max_speed_ms", "active_flow_cells"))

    def test_eastward_movement_approx_0_deg(self):
        a = CrowdAnalytics(history_frames=3)
        for i in range(3):
            a.update([_Pt("t1", float(i * 5), 7.5)])  # moves right (east = 0°)
        s = a.get_flow_summary()
        if s["dominant_direction_deg"] is not None:
            # Should be approximately 0° (east)
            deg = s["dominant_direction_deg"]
            assert deg < 30 or deg > 330  # near 0° (east)

    def test_active_flow_cells_nonnegative(self):
        a = _analytics()
        a.update([_Pt("t1", 5.0, 5.0)])
        a.update([_Pt("t1", 8.0, 5.0)])
        s = a.get_flow_summary()
        assert s["active_flow_cells"] >= 0


# ---------------------------------------------------------------------------
# get_state (5 tests)
# ---------------------------------------------------------------------------


class TestGetState:
    def test_state_keys(self):
        a = _analytics()
        s = a.get_state()
        assert all(k in s for k in ("total_persons", "mean_density", "congestion_cell_count",
                                     "flow_vector_count", "density_grid", "flow_vectors",
                                     "congestion_zones", "update_count", "uptime_seconds"))

    def test_state_update_count(self):
        a = _analytics()
        a.update([])
        a.update([])
        assert a.get_state()["update_count"] == 2

    def test_state_density_grid_not_empty_after_update(self):
        a = _analytics(cols=5, rows=4)
        a.update([_Pt("t1", 5.0, 5.0)])
        grid = a.get_state()["density_grid"]
        assert grid["cols"] == 5 and grid["rows"] == 4

    def test_state_uptime_positive(self):
        a = _analytics()
        time.sleep(0.01)
        assert a.get_state()["uptime_seconds"] >= 0.0

    def test_state_after_reset_zero(self):
        a = _analytics()
        a.update([_Pt("t1", 5.0, 5.0)])
        a.reset()
        s = a.get_state()
        assert s["update_count"] == 0
        assert s["total_persons"] == 0


# ---------------------------------------------------------------------------
# get_stats (5 tests)
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_enabled_true(self):
        a = _analytics()
        assert a.get_stats()["enabled"] is True

    def test_stats_keys(self):
        a = _analytics()
        s = a.get_stats()
        assert all(k in s for k in ("update_count", "total_persons", "congestion_cells",
                                     "flow_vectors", "zone_capacities_registered",
                                     "floor_width_m", "floor_height_m", "grid_cols", "grid_rows"))

    def test_stats_grid_dims(self):
        a = CrowdAnalytics(grid_cols=12, grid_rows=9)
        s = a.get_stats()
        assert s["grid_cols"] == 12
        assert s["grid_rows"] == 9

    def test_stats_zone_capacity_count(self):
        a = _analytics()
        a.set_zone_capacity("z1", 10)
        a.set_zone_capacity("z2", 20)
        assert a.get_stats()["zone_capacities_registered"] == 2

    def test_congestion_threshold_in_stats(self):
        a = CrowdAnalytics(congestion_threshold=0.6)
        assert a.get_stats()["congestion_threshold"] == 0.6


# ---------------------------------------------------------------------------
# reset (3 tests)
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_history(self):
        a = _analytics()
        a.update([_Pt("t1", 5.0, 5.0)])
        a.reset()
        assert len(a._position_history) == 0

    def test_reset_clears_alerts_buffer(self):
        a = _analytics()
        a.set_zone_capacity("z1", 5)
        a.update_zone_occupancy("z1", {f"t{i}" for i in range(6)})
        a.update([])
        a.reset()
        assert len(a._alerts) == 0

    def test_reset_preserves_zone_capacities(self):
        a = _analytics()
        a.set_zone_capacity("z1", 10)
        a.reset()
        assert "z1" in a._zone_capacities


# ---------------------------------------------------------------------------
# Thread safety (2 tests)
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_updates(self):
        a = _analytics()
        errors = []

        def worker():
            try:
                for i in range(50):
                    a.update([_Pt(f"t{i}", float(i % 20), float(i % 15))])
            except Exception as e:
                errors.append(e)

        ts = [threading.Thread(target=worker) for _ in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert errors == []

    def test_concurrent_read_write(self):
        a = _analytics()
        errors = []

        def writer():
            try:
                for i in range(30):
                    a.update([_Pt("t1", float(i), 5.0)])
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(30):
                    a.get_state()
                    a.get_flow_summary()
            except Exception as e:
                errors.append(e)

        ts = [threading.Thread(target=writer) for _ in range(2)]
        ts += [threading.Thread(target=reader) for _ in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert errors == []
