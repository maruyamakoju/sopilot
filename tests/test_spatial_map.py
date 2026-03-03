"""Tests for sopilot.perception.spatial_map.

Uses mock WorldState/SceneEntity/Track objects; does NOT import from
sopilot.perception.types.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from sopilot.perception.spatial_map import SpatialCell, SpatialMap, SpatialSnapshot

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def make_entity(entity_id, cx, cy, label="person", w=0.1, h=0.1):
    """Create a mock entity with a bbox whose center is (cx, cy)."""
    bbox = SimpleNamespace(x=cx - w / 2, y=cy - h / 2, w=w, h=h)
    return SimpleNamespace(entity_id=entity_id, label=label, bbox=bbox)


def make_track(entity_id, vx=0.0, vy=0.0):
    return SimpleNamespace(entity_id=entity_id, velocity=(vx, vy))


def make_world_state(entities, tracks=None, timestamp=1000.0, frame_number=0):
    sg = SimpleNamespace(entities=entities)
    active_tracks = {t.entity_id: t for t in (tracks or [])}
    return SimpleNamespace(
        scene_graph=sg,
        active_tracks=active_tracks,
        entity_count=len(entities),
        timestamp=timestamp,
        frame_number=frame_number,
    )


def make_depth(entity_id, depth_relative):
    return SimpleNamespace(entity_id=entity_id, depth_relative=depth_relative)


# ---------------------------------------------------------------------------
# TestSpatialCellToDict
# ---------------------------------------------------------------------------


class TestSpatialCellToDict:
    def test_all_keys_present(self):
        cell = SpatialCell(
            gx=1, gy=2, occupancy=0.5, dominant_label="person",
            last_seen=999.0, depth_mean=0.3, velocity_dx=0.1, velocity_dy=-0.2,
        )
        d = cell.to_dict()
        for key in ("gx", "gy", "occupancy", "dominant_label", "last_seen",
                    "depth_mean", "velocity_dx", "velocity_dy"):
            assert key in d

    def test_values_correct(self):
        cell = SpatialCell(
            gx=3, gy=4, occupancy=0.9, dominant_label="car",
            last_seen=42.0, depth_mean=0.7, velocity_dx=0.05, velocity_dy=0.0,
        )
        d = cell.to_dict()
        assert d["gx"] == 3
        assert d["gy"] == 4
        assert d["dominant_label"] == "car"


# ---------------------------------------------------------------------------
# TestSpatialSnapshotToDict
# ---------------------------------------------------------------------------


class TestSpatialSnapshotToDict:
    def _make_snapshot(self):
        return SpatialSnapshot(
            timestamp=1234.5,
            frame_number=10,
            grid_w=8,
            grid_h=6,
            total_entities=3,
            crowd_density=0.25,
            flow_dx=0.01,
            flow_dy=-0.01,
            hotspots=[{"gx": 2, "gy": 1, "occupancy": 0.3}],
            occupancy_grid=[[0.0] * 8 for _ in range(6)],
        )

    def test_all_keys_present(self):
        snap = self._make_snapshot()
        d = snap.to_dict()
        expected_keys = [
            "timestamp", "frame_number", "grid_w", "grid_h",
            "total_entities", "crowd_density", "flow_dx", "flow_dy",
            "hotspots", "occupancy_grid",
        ]
        for key in expected_keys:
            assert key in d, f"Missing key: {key}"

    def test_values_correct(self):
        snap = self._make_snapshot()
        d = snap.to_dict()
        assert d["timestamp"] == 1234.5
        assert d["frame_number"] == 10
        assert d["grid_w"] == 8
        assert d["grid_h"] == 6
        assert d["total_entities"] == 3
        assert d["crowd_density"] == 0.25

    def test_hotspots_is_list(self):
        snap = self._make_snapshot()
        d = snap.to_dict()
        assert isinstance(d["hotspots"], list)

    def test_occupancy_grid_is_nested_list(self):
        snap = self._make_snapshot()
        d = snap.to_dict()
        assert isinstance(d["occupancy_grid"], list)
        assert len(d["occupancy_grid"]) == 6
        assert len(d["occupancy_grid"][0]) == 8


# ---------------------------------------------------------------------------
# TestSpatialMapInit
# ---------------------------------------------------------------------------


class TestSpatialMapInit:
    def test_default_params(self):
        sm = SpatialMap()
        assert sm._gw == 8
        assert sm._gh == 6
        assert sm._alpha == 0.3
        assert sm._top_n == 3
        assert sm._max_history == 60

    def test_custom_params(self):
        sm = SpatialMap(grid_w=10, grid_h=8, ema_alpha=0.5, hotspot_top_n=5, max_history=20)
        assert sm._gw == 10
        assert sm._gh == 8
        assert sm._alpha == 0.5
        assert sm._top_n == 5
        assert sm._max_history == 20

    def test_alpha_clamped_above_one(self):
        sm = SpatialMap(ema_alpha=2.0)
        assert sm._alpha == 1.0

    def test_alpha_clamped_below_zero(self):
        sm = SpatialMap(ema_alpha=-0.5)
        assert sm._alpha == 0.0

    def test_initial_occupancy_zeros(self):
        sm = SpatialMap()
        arr = sm.get_grid_array()
        assert np.all(arr == 0.0)

    def test_initial_history_empty(self):
        sm = SpatialMap()
        assert sm.get_history() == []


# ---------------------------------------------------------------------------
# TestUpdateEmptyWorldState
# ---------------------------------------------------------------------------


class TestUpdateEmptyWorldState:
    def test_no_entities_crowd_density_zero(self):
        sm = SpatialMap()
        ws = make_world_state([])
        snap = sm.update(ws)
        assert snap.crowd_density == 0.0

    def test_no_entities_entity_count_zero(self):
        sm = SpatialMap()
        ws = make_world_state([])
        snap = sm.update(ws)
        assert snap.total_entities == 0

    def test_no_entities_flow_zero(self):
        sm = SpatialMap()
        ws = make_world_state([])
        snap = sm.update(ws)
        assert snap.flow_dx == 0.0
        assert snap.flow_dy == 0.0

    def test_no_entities_hotspots_empty(self):
        sm = SpatialMap()
        ws = make_world_state([])
        snap = sm.update(ws)
        assert snap.hotspots == []

    def test_snapshot_frame_number_propagated(self):
        sm = SpatialMap()
        ws = make_world_state([], frame_number=42)
        snap = sm.update(ws)
        assert snap.frame_number == 42

    def test_snapshot_timestamp_propagated(self):
        sm = SpatialMap()
        ws = make_world_state([], timestamp=5555.0)
        snap = sm.update(ws, timestamp=5555.0)
        assert snap.timestamp == 5555.0


# ---------------------------------------------------------------------------
# TestUpdateSingleEntity
# ---------------------------------------------------------------------------


class TestUpdateSingleEntity:
    def test_single_entity_occupancy_nonzero(self):
        sm = SpatialMap(grid_w=8, grid_h=6, ema_alpha=0.5)
        # entity at center (0.5, 0.5)
        ws = make_world_state([make_entity(1, 0.5, 0.5)])
        snap = sm.update(ws)
        assert snap.total_entities == 1
        assert snap.crowd_density > 0.0

    def test_single_entity_correct_cell(self):
        """Entity at top-left (0.1, 0.1) should land in cell (0, 0)."""
        sm = SpatialMap(grid_w=8, grid_h=6, ema_alpha=1.0)
        ent = make_entity(1, 0.05, 0.05, w=0.02, h=0.02)
        ws = make_world_state([ent])
        sm.update(ws)
        arr = sm.get_grid_array()
        assert arr[0, 0] > 0.0

    def test_single_entity_total_entities(self):
        sm = SpatialMap()
        ws = make_world_state([make_entity(1, 0.5, 0.5)])
        snap = sm.update(ws)
        assert snap.total_entities == 1


# ---------------------------------------------------------------------------
# TestUpdateMultipleEntities
# ---------------------------------------------------------------------------


class TestUpdateMultipleEntities:
    def test_three_different_cells(self):
        sm = SpatialMap(grid_w=4, grid_h=4, ema_alpha=1.0)
        entities = [
            make_entity(1, 0.1, 0.1, w=0.05, h=0.05),
            make_entity(2, 0.5, 0.5, w=0.05, h=0.05),
            make_entity(3, 0.9, 0.9, w=0.05, h=0.05),
        ]
        ws = make_world_state(entities)
        snap = sm.update(ws)
        assert snap.total_entities == 3
        # At least 3 cells should have occupancy
        arr = sm.get_grid_array()
        assert int(np.sum(arr > 0.0)) >= 3

    def test_snapshot_occupancy_grid_dims(self):
        sm = SpatialMap(grid_w=8, grid_h=6)
        ws = make_world_state([make_entity(1, 0.5, 0.5)])
        snap = sm.update(ws)
        assert len(snap.occupancy_grid) == 6
        assert len(snap.occupancy_grid[0]) == 8


# ---------------------------------------------------------------------------
# TestCrowdDensity
# ---------------------------------------------------------------------------


class TestCrowdDensity:
    def test_no_entities_returns_zero(self):
        sm = SpatialMap()
        ws = make_world_state([])
        sm.update(ws)
        assert sm.get_crowd_density() == 0.0

    def test_density_increases_with_entities(self):
        sm = SpatialMap(grid_w=4, grid_h=4, ema_alpha=1.0)
        # Place one entity per cell
        entities = []
        eid = 0
        for row in range(4):
            for col in range(4):
                cx = (col + 0.5) / 4
                cy = (row + 0.5) / 4
                entities.append(make_entity(eid, cx, cy, w=0.01, h=0.01))
                eid += 1
        ws = make_world_state(entities)
        sm.update(ws)
        density = sm.get_crowd_density()
        assert density == 1.0

    def test_density_between_zero_and_one(self):
        sm = SpatialMap(grid_w=8, grid_h=6)
        ws = make_world_state([make_entity(1, 0.5, 0.5)])
        sm.update(ws)
        d = sm.get_crowd_density()
        assert 0.0 <= d <= 1.0


# ---------------------------------------------------------------------------
# TestFlowVector
# ---------------------------------------------------------------------------


class TestFlowVector:
    def test_entities_with_velocity(self):
        sm = SpatialMap(grid_w=8, grid_h=6, ema_alpha=1.0)
        ent = make_entity(1, 0.5, 0.5)
        track = make_track(1, vx=0.1, vy=0.2)
        ws = make_world_state([ent], tracks=[track])
        snap = sm.update(ws)
        assert snap.flow_dx == pytest.approx(0.1, abs=1e-3)
        assert snap.flow_dy == pytest.approx(0.2, abs=1e-3)

    def test_no_tracks_returns_zero_flow(self):
        sm = SpatialMap()
        ws = make_world_state([make_entity(1, 0.5, 0.5)])
        snap = sm.update(ws)
        assert snap.flow_dx == 0.0
        assert snap.flow_dy == 0.0

    def test_get_flow_vector_matches_snapshot(self):
        sm = SpatialMap(grid_w=8, grid_h=6, ema_alpha=1.0)
        ent = make_entity(1, 0.5, 0.5)
        track = make_track(1, vx=0.05, vy=-0.03)
        ws = make_world_state([ent], tracks=[track])
        snap = sm.update(ws)
        fv = sm.get_flow_vector()
        assert fv[0] == pytest.approx(snap.flow_dx, abs=1e-4)
        assert fv[1] == pytest.approx(snap.flow_dy, abs=1e-4)

    def test_multiple_entities_flow_averaged(self):
        sm = SpatialMap(grid_w=8, grid_h=6, ema_alpha=1.0)
        ent1 = make_entity(1, 0.2, 0.2)
        ent2 = make_entity(2, 0.8, 0.8)
        track1 = make_track(1, vx=0.4, vy=0.0)
        track2 = make_track(2, vx=0.0, vy=0.0)
        ws = make_world_state([ent1, ent2], tracks=[track1, track2])
        snap = sm.update(ws)
        # Average of (0.4, 0.0) and (0.0, 0.0) = (0.2, 0.0)
        assert snap.flow_dx == pytest.approx(0.2, abs=1e-3)
        assert snap.flow_dy == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# TestHotspots
# ---------------------------------------------------------------------------


class TestHotspots:
    def test_entity_appears_in_hotspots(self):
        sm = SpatialMap(grid_w=8, grid_h=6, ema_alpha=1.0, hotspot_top_n=3)
        ent = make_entity(1, 0.5, 0.5)
        ws = make_world_state([ent])
        snap = sm.update(ws)
        assert len(snap.hotspots) >= 1

    def test_hotspot_has_required_keys(self):
        sm = SpatialMap(ema_alpha=1.0)
        ws = make_world_state([make_entity(1, 0.5, 0.5)])
        snap = sm.update(ws)
        h = snap.hotspots[0]
        assert "gx" in h
        assert "gy" in h
        assert "occupancy" in h
        assert "dominant_label" in h

    def test_get_hotspots_custom_n(self):
        sm = SpatialMap(grid_w=8, grid_h=6, ema_alpha=1.0)
        entities = [make_entity(i, (i + 1) * 0.1, 0.5, w=0.02, h=0.02) for i in range(5)]
        ws = make_world_state(entities)
        sm.update(ws)
        spots = sm.get_hotspots(top_n=2)
        assert len(spots) <= 2

    def test_no_entities_no_hotspots(self):
        sm = SpatialMap()
        ws = make_world_state([])
        snap = sm.update(ws)
        assert snap.hotspots == []


# ---------------------------------------------------------------------------
# TestRiskCells
# ---------------------------------------------------------------------------


class TestRiskCells:
    def test_high_occupancy_appears_in_risk(self):
        sm = SpatialMap(grid_w=4, grid_h=4, ema_alpha=1.0)
        # alpha=1.0 means occupancy = alpha per entity per update = 1.0
        ent = make_entity(1, 0.5, 0.5)
        ws = make_world_state([ent])
        sm.update(ws)
        risks = sm.get_risk_cells(threshold=0.5)
        assert len(risks) >= 1

    def test_low_occupancy_below_threshold(self):
        sm = SpatialMap(grid_w=4, grid_h=4, ema_alpha=0.01)
        ent = make_entity(1, 0.5, 0.5)
        ws = make_world_state([ent])
        sm.update(ws)
        # With alpha=0.01, occupancy will be 0.01, below threshold 0.5
        risks = sm.get_risk_cells(threshold=0.5)
        assert len(risks) == 0

    def test_risk_cells_keys(self):
        sm = SpatialMap(grid_w=4, grid_h=4, ema_alpha=1.0)
        ws = make_world_state([make_entity(1, 0.5, 0.5)])
        sm.update(ws)
        risks = sm.get_risk_cells(threshold=0.1)
        if risks:
            assert "gx" in risks[0]
            assert "gy" in risks[0]
            assert "occupancy" in risks[0]


# ---------------------------------------------------------------------------
# TestGridArray
# ---------------------------------------------------------------------------


class TestGridArray:
    def test_returns_numpy_array(self):
        sm = SpatialMap(grid_w=8, grid_h=6)
        arr = sm.get_grid_array()
        assert isinstance(arr, np.ndarray)

    def test_correct_shape(self):
        sm = SpatialMap(grid_w=8, grid_h=6)
        arr = sm.get_grid_array()
        assert arr.shape == (6, 8)

    def test_returns_copy(self):
        sm = SpatialMap(grid_w=8, grid_h=6)
        arr1 = sm.get_grid_array()
        arr2 = sm.get_grid_array()
        arr1[0, 0] = 999.0
        assert sm._occupancy[0, 0] != 999.0
        assert arr2[0, 0] != 999.0


# ---------------------------------------------------------------------------
# TestEMADecay
# ---------------------------------------------------------------------------


class TestEMADecay:
    def test_occupancy_decays_when_entity_absent(self):
        sm = SpatialMap(grid_w=4, grid_h=4, ema_alpha=0.5)
        ent = make_entity(1, 0.5, 0.5)
        ws1 = make_world_state([ent], timestamp=1.0)
        sm.update(ws1)
        occ_after_frame1 = sm.get_grid_array().copy()

        ws2 = make_world_state([], timestamp=2.0)
        sm.update(ws2)
        occ_after_frame2 = sm.get_grid_array()

        # Occupancy should be less after entity leaves
        assert np.all(occ_after_frame2 <= occ_after_frame1 + 1e-9)
        assert np.sum(occ_after_frame2) < np.sum(occ_after_frame1)

    def test_occupancy_builds_with_repeated_entity(self):
        sm = SpatialMap(grid_w=4, grid_h=4, ema_alpha=0.5)
        ent = make_entity(1, 0.5, 0.5)

        prev = 0.0
        for i in range(5):
            ws = make_world_state([ent], timestamp=float(i))
            sm.update(ws)
            arr = sm.get_grid_array()
            cx, cy = 0.5, 0.5
            gx = min(3, int(cx * 4))
            gy = min(3, int(cy * 4))
            current = arr[gy, gx]
            assert current >= prev - 1e-9
            prev = current


# ---------------------------------------------------------------------------
# TestHistoryAccumulation
# ---------------------------------------------------------------------------


class TestHistoryAccumulation:
    def test_history_grows_with_updates(self):
        sm = SpatialMap(max_history=60)
        ws = make_world_state([])
        for i in range(5):
            sm.update(ws, timestamp=float(i))
        assert len(sm.get_history()) == 5

    def test_history_contains_snapshots(self):
        sm = SpatialMap()
        ws = make_world_state([])
        sm.update(ws)
        hist = sm.get_history()
        assert isinstance(hist[0], SpatialSnapshot)


# ---------------------------------------------------------------------------
# TestHistoryLimit
# ---------------------------------------------------------------------------


class TestHistoryLimit:
    def test_history_capped_at_max(self):
        sm = SpatialMap(max_history=5)
        ws = make_world_state([])
        for i in range(10):
            sm.update(ws, timestamp=float(i))
        assert len(sm.get_history(n=100)) == 5

    def test_history_keeps_most_recent(self):
        sm = SpatialMap(max_history=3)
        ws_list = [make_world_state([], timestamp=float(i), frame_number=i) for i in range(5)]
        for ws in ws_list:
            sm.update(ws)
        hist = sm.get_history(n=10)
        # Should have the last 3 frames
        frame_nums = [s.frame_number for s in hist]
        assert 4 in frame_nums
        assert 3 in frame_nums
        assert 2 in frame_nums


# ---------------------------------------------------------------------------
# TestGetHistory
# ---------------------------------------------------------------------------


class TestGetHistory:
    def test_returns_most_recent_n(self):
        sm = SpatialMap(max_history=60)
        ws = make_world_state([])
        for i in range(10):
            sm.update(ws, timestamp=float(i), )
        hist = sm.get_history(n=3)
        assert len(hist) == 3

    def test_returns_all_if_less_than_n(self):
        sm = SpatialMap()
        ws = make_world_state([])
        sm.update(ws)
        sm.update(ws)
        hist = sm.get_history(n=100)
        assert len(hist) == 2


# ---------------------------------------------------------------------------
# TestReset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_occupancy(self):
        sm = SpatialMap(ema_alpha=1.0)
        ws = make_world_state([make_entity(1, 0.5, 0.5)])
        sm.update(ws)
        sm.reset()
        arr = sm.get_grid_array()
        assert np.all(arr == 0.0)

    def test_reset_clears_history(self):
        sm = SpatialMap()
        ws = make_world_state([])
        sm.update(ws)
        sm.reset()
        assert sm.get_history() == []

    def test_reset_clears_label_counts(self):
        sm = SpatialMap(ema_alpha=1.0)
        ws = make_world_state([make_entity(1, 0.5, 0.5, label="car")])
        sm.update(ws)
        sm.reset()
        assert sm._label_counts == {}

    def test_update_after_reset_works(self):
        sm = SpatialMap(ema_alpha=1.0)
        ws = make_world_state([make_entity(1, 0.5, 0.5)])
        sm.update(ws)
        sm.reset()
        snap = sm.update(ws)
        assert snap.total_entities == 1


# ---------------------------------------------------------------------------
# TestGetStateDictKeys
# ---------------------------------------------------------------------------


class TestGetStateDictKeys:
    def test_required_keys_present(self):
        sm = SpatialMap()
        ws = make_world_state([])
        sm.update(ws)
        d = sm.get_state_dict()
        for key in ("grid_w", "grid_h", "crowd_density", "flow_dx", "flow_dy",
                    "hotspots", "history_size"):
            assert key in d, f"Missing key: {key}"

    def test_history_size_matches(self):
        sm = SpatialMap()
        ws = make_world_state([])
        sm.update(ws)
        sm.update(ws)
        d = sm.get_state_dict()
        assert d["history_size"] == 2


# ---------------------------------------------------------------------------
# TestThreadSafety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_updates_no_error(self):
        sm = SpatialMap(grid_w=8, grid_h=6)
        errors = []

        def worker():
            try:
                for i in range(20):
                    ents = [make_entity(j, (j + 1) * 0.1, 0.5, w=0.05, h=0.05)
                            for j in range(3)]
                    ws = make_world_state(ents, timestamp=float(i))
                    sm.update(ws)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_concurrent_reads_no_error(self):
        sm = SpatialMap()
        ws = make_world_state([make_entity(1, 0.5, 0.5)])
        sm.update(ws)
        errors = []

        def reader():
            try:
                for _ in range(20):
                    sm.get_crowd_density()
                    sm.get_flow_vector()
                    sm.get_hotspots()
                    sm.get_grid_array()
                    sm.get_state_dict()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"


# ---------------------------------------------------------------------------
# TestDepthEstimateIntegration
# ---------------------------------------------------------------------------


class TestDepthEstimateIntegration:
    def test_depth_estimate_affects_depth_array(self):
        sm = SpatialMap(grid_w=4, grid_h=4, ema_alpha=1.0)
        ent = make_entity(1, 0.5, 0.5, w=0.05, h=0.05)
        depth_est = make_depth(entity_id=1, depth_relative=0.9)
        ws = make_world_state([ent])
        sm.update(ws, depth_estimates=[depth_est])
        # Depth should have moved from default 0.5 toward 0.9
        gx = min(3, int(0.5 * 4))
        gy = min(3, int(0.5 * 4))
        assert sm._depth[gy, gx] > 0.5

    def test_no_depth_estimates_uses_default(self):
        sm = SpatialMap(grid_w=4, grid_h=4, ema_alpha=0.5)
        ent = make_entity(1, 0.5, 0.5)
        ws = make_world_state([ent])
        sm.update(ws, depth_estimates=None)
        # Default depth is 0.5; EMA with alpha=0.5 from 0.5 stays near 0.5
        gx = min(3, int(0.5 * 4))
        gy = min(3, int(0.5 * 4))
        assert abs(sm._depth[gy, gx] - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# TestEntityCellMapping
# ---------------------------------------------------------------------------


class TestEntityCellMapping:
    def test_entity_at_origin_maps_to_cell_0_0(self):
        sm = SpatialMap(grid_w=8, grid_h=6, ema_alpha=1.0)
        # bbox center at (0.01, 0.01) should land in cell gx=0, gy=0
        ent = make_entity(1, 0.01, 0.01, w=0.01, h=0.01)
        ws = make_world_state([ent])
        sm.update(ws)
        arr = sm.get_grid_array()
        assert arr[0, 0] > 0.0

    def test_entity_at_max_maps_to_last_cell(self):
        sm = SpatialMap(grid_w=8, grid_h=6, ema_alpha=1.0)
        # bbox center at (0.99, 0.99) should land in cell gx=7, gy=5
        ent = make_entity(1, 0.99, 0.99, w=0.01, h=0.01)
        ws = make_world_state([ent])
        sm.update(ws)
        arr = sm.get_grid_array()
        assert arr[5, 7] > 0.0

    def test_entity_at_center_maps_to_middle_cell(self):
        sm = SpatialMap(grid_w=4, grid_h=4, ema_alpha=1.0)
        # Center 0.5 * 4 = 2.0 -> int(2.0) = 2
        ent = make_entity(1, 0.5, 0.5, w=0.01, h=0.01)
        ws = make_world_state([ent])
        sm.update(ws)
        arr = sm.get_grid_array()
        assert arr[2, 2] > 0.0


# ---------------------------------------------------------------------------
# TestDominantLabel
# ---------------------------------------------------------------------------


class TestDominantLabel:
    def test_dominant_label_is_most_frequent(self):
        sm = SpatialMap(grid_w=4, grid_h=4, ema_alpha=1.0)
        # Three "car" entities in same cell, one "person"
        entities = [
            make_entity(1, 0.5, 0.5, label="car", w=0.02, h=0.02),
            make_entity(2, 0.5, 0.5, label="car", w=0.02, h=0.02),
            make_entity(3, 0.5, 0.5, label="car", w=0.02, h=0.02),
            make_entity(4, 0.5, 0.5, label="person", w=0.02, h=0.02),
        ]
        ws = make_world_state(entities)
        snap = sm.update(ws)
        # The hotspot for that cell should have dominant_label = "car"
        if snap.hotspots:
            assert snap.hotspots[0]["dominant_label"] == "car"

    def test_single_entity_label_is_dominant(self):
        sm = SpatialMap(grid_w=4, grid_h=4, ema_alpha=1.0)
        ws = make_world_state([make_entity(1, 0.5, 0.5, label="forklift")])
        snap = sm.update(ws)
        if snap.hotspots:
            assert snap.hotspots[0]["dominant_label"] == "forklift"


# ---------------------------------------------------------------------------
# TestGridDimensions
# ---------------------------------------------------------------------------


class TestGridDimensions:
    def test_4x4_grid_shape(self):
        sm = SpatialMap(grid_w=4, grid_h=4)
        arr = sm.get_grid_array()
        assert arr.shape == (4, 4)

    def test_1x1_grid_shape(self):
        sm = SpatialMap(grid_w=1, grid_h=1)
        arr = sm.get_grid_array()
        assert arr.shape == (1, 1)

    def test_12x10_grid_shape(self):
        sm = SpatialMap(grid_w=12, grid_h=10)
        arr = sm.get_grid_array()
        assert arr.shape == (10, 12)

    def test_snapshot_grid_dims_match(self):
        sm = SpatialMap(grid_w=5, grid_h=3)
        ws = make_world_state([])
        snap = sm.update(ws)
        assert snap.grid_w == 5
        assert snap.grid_h == 3
        assert len(snap.occupancy_grid) == 3
        assert len(snap.occupancy_grid[0]) == 5
