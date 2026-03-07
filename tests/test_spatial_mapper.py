"""Tests for Phase 46: SpatialMapper (spatial_mapper.py).

Tests cover:
    - CameraCalibration dataclass
    - OccupancyPoint dataclass + to_dict
    - SpatialMapper.register_camera / is_registered / get_camera_ids
    - SpatialMapper.project_point (homography, identity, clamp, unregistered)
    - SpatialMapper.add_detection (bbox object, bbox tuple, returns OccupancyPoint)
    - SpatialMapper.get_recent_points (age filter, camera filter)
    - SpatialMapper.get_occupancy_grid (empty, single, multi-point, shape)
    - SpatialMapper.to_map_data (keys, floor info)
    - SpatialMapper.get_stats (structure)
    - Rolling buffer (max_points)
    - Thread safety (concurrent add_detection)
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from sopilot.perception.spatial_mapper import (
    CameraCalibration,
    OccupancyPoint,
    SpatialMapper,
)

_W = 20.0
_H = 15.0


def _make_mapper(**kw) -> SpatialMapper:
    return SpatialMapper(floor_width_m=_W, floor_height_m=_H, **kw)


def _identity_cam(mapper: SpatialMapper, cam_id: str = "cam01") -> None:
    """Register a camera with identity homography (u,v → x*W, y*H)."""
    # Identity H: [u,v,1]^T → [u,v,1]^T  (world = normalised, so scale manually)
    # Use a scale matrix so u=0.5 → x = W/2
    H = np.diag([_W, _H, 1.0])
    mapper.register_camera(cam_id, H, image_size=(1920, 1080))


# ---------------------------------------------------------------------------
# CameraCalibration
# ---------------------------------------------------------------------------


class TestCameraCalibration:
    def test_fields(self):
        H = np.eye(3)
        cal = CameraCalibration(camera_id="cam01", homography=H, image_size=(1920, 1080))
        assert cal.camera_id == "cam01"
        assert cal.homography.shape == (3, 3)
        assert cal.image_size == (1920, 1080)

    def test_registered_at_set(self):
        H = np.eye(3)
        cal = CameraCalibration(camera_id="cam01", homography=H, image_size=(640, 480))
        assert cal.registered_at > 0


# ---------------------------------------------------------------------------
# OccupancyPoint
# ---------------------------------------------------------------------------


class TestOccupancyPoint:
    def _make(self, **kw):
        defaults = dict(
            point_id="p1",
            camera_id="cam01",
            person_id="t1",
            label="person",
            world_x=5.0,
            world_y=3.0,
            timestamp=time.time(),
            confidence=0.9,
        )
        defaults.update(kw)
        return OccupancyPoint(**defaults)

    def test_fields(self):
        pt = self._make()
        assert pt.world_x == 5.0
        assert pt.world_y == 3.0
        assert pt.confidence == 0.9

    def test_to_dict_keys(self):
        pt = self._make()
        d = pt.to_dict()
        for key in ("point_id", "camera_id", "person_id", "label",
                    "world_x", "world_y", "timestamp", "confidence", "age_seconds"):
            assert key in d

    def test_to_dict_coordinates_rounded(self):
        pt = self._make(world_x=1.23456789)
        d = pt.to_dict()
        assert abs(d["world_x"] - 1.235) < 0.001

    def test_age_seconds_nonnegative(self):
        pt = self._make(timestamp=time.time() - 2.0)
        assert pt.to_dict()["age_seconds"] >= 0


# ---------------------------------------------------------------------------
# register_camera / is_registered / get_camera_ids
# ---------------------------------------------------------------------------


class TestRegisterCamera:
    def test_register_stores_camera(self):
        m = _make_mapper()
        _identity_cam(m, "cam01")
        assert m.is_registered("cam01")

    def test_unregistered_returns_false(self):
        m = _make_mapper()
        assert not m.is_registered("nonexistent")

    def test_get_camera_ids_empty(self):
        m = _make_mapper()
        assert m.get_camera_ids() == []

    def test_get_camera_ids_single(self):
        m = _make_mapper()
        _identity_cam(m, "cam01")
        ids = m.get_camera_ids()
        assert "cam01" in ids

    def test_get_camera_ids_multiple(self):
        m = _make_mapper()
        _identity_cam(m, "cam_A")
        _identity_cam(m, "cam_B")
        ids = m.get_camera_ids()
        assert set(ids) == {"cam_A", "cam_B"}

    def test_register_accepts_list_homography(self):
        m = _make_mapper()
        H = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        m.register_camera("cam_list", H)
        assert m.is_registered("cam_list")


# ---------------------------------------------------------------------------
# project_point
# ---------------------------------------------------------------------------


class TestProjectPoint:
    def test_unregistered_returns_none(self):
        m = _make_mapper()
        assert m.project_point("cam_missing", 0.5, 0.5) is None

    def test_identity_homography_center(self):
        m = _make_mapper()
        _identity_cam(m, "cam01")
        # u=0.5, v=0.5 → p = H@[0.5,0.5,1]^T = [10, 7.5, 1] → (10, 7.5)
        result = m.project_point("cam01", 0.5, 0.5)
        assert result is not None
        x, y = result
        assert abs(x - _W / 2) < 0.01
        assert abs(y - _H / 2) < 0.01

    def test_origin_maps_to_origin(self):
        m = _make_mapper()
        _identity_cam(m)
        result = m.project_point("cam01", 0.0, 0.0)
        assert result is not None
        x, y = result
        assert abs(x) < 0.01
        assert abs(y) < 0.01

    def test_corner_maps_to_floor_corner(self):
        m = _make_mapper()
        _identity_cam(m)
        result = m.project_point("cam01", 1.0, 1.0)
        assert result is not None
        x, y = result
        assert abs(x - _W) < 0.01
        assert abs(y - _H) < 0.01

    def test_result_clamped_to_floor(self):
        """Out-of-frame point should be clamped."""
        m = _make_mapper()
        H = np.diag([_W * 2, _H * 2, 1.0])  # maps outside floor
        m.register_camera("cam_big", H)
        result = m.project_point("cam_big", 1.0, 1.0)
        assert result is not None
        x, y = result
        assert x <= _W + 0.001
        assert y <= _H + 0.001

    def test_degenerate_w_returns_none(self):
        """Homography with w≈0 should return None."""
        m = _make_mapper()
        # H where last row = [0, 0, 0]
        H = np.zeros((3, 3))
        m.register_camera("cam_degen", H)
        result = m.project_point("cam_degen", 0.5, 0.5)
        assert result is None

    def test_returns_float_tuple(self):
        m = _make_mapper()
        _identity_cam(m)
        result = m.project_point("cam01", 0.3, 0.7)
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


# ---------------------------------------------------------------------------
# add_detection
# ---------------------------------------------------------------------------


class TestAddDetection:
    def test_returns_occupancy_point(self):
        m = _make_mapper()
        _identity_cam(m)
        bbox = SimpleNamespace(x1=0.4, y1=0.4, x2=0.6, y2=0.6)
        pt = m.add_detection("cam01", bbox, person_id="t1")
        assert isinstance(pt, OccupancyPoint)

    def test_unregistered_camera_returns_none(self):
        m = _make_mapper()
        pt = m.add_detection("no_cam", (0.4, 0.4, 0.6, 0.6))
        assert pt is None

    def test_bbox_tuple(self):
        m = _make_mapper()
        _identity_cam(m)
        pt = m.add_detection("cam01", (0.2, 0.3, 0.4, 0.5), person_id="t2")
        assert pt is not None
        # centre of (0.2,0.3,0.4,0.5) = (0.3, 0.4) → world = (6.0, 6.0)
        assert abs(pt.world_x - 6.0) < 0.01
        assert abs(pt.world_y - 6.0) < 0.01

    def test_bbox_object_with_attrs(self):
        m = _make_mapper()
        _identity_cam(m)
        bbox = SimpleNamespace(x1=0.0, y1=0.0, x2=0.5, y2=0.5)
        pt = m.add_detection("cam01", bbox)
        assert pt is not None

    def test_person_id_stored(self):
        m = _make_mapper()
        _identity_cam(m)
        pt = m.add_detection("cam01", (0.4, 0.4, 0.6, 0.6), person_id="alice")
        assert pt.person_id == "alice"

    def test_auto_person_id_when_empty(self):
        m = _make_mapper()
        _identity_cam(m)
        pt = m.add_detection("cam01", (0.4, 0.4, 0.6, 0.6), person_id="")
        assert pt is not None
        assert len(pt.person_id) > 0

    def test_label_stored(self):
        m = _make_mapper()
        _identity_cam(m)
        pt = m.add_detection("cam01", (0.4, 0.4, 0.6, 0.6), label="helmet")
        assert pt.label == "helmet"

    def test_confidence_stored(self):
        m = _make_mapper()
        _identity_cam(m)
        pt = m.add_detection("cam01", (0.4, 0.4, 0.6, 0.6), confidence=0.75)
        assert abs(pt.confidence - 0.75) < 1e-5

    def test_adds_to_buffer(self):
        m = _make_mapper()
        _identity_cam(m)
        m.add_detection("cam01", (0.3, 0.3, 0.5, 0.5))
        assert len(m._points) == 1


# ---------------------------------------------------------------------------
# get_recent_points
# ---------------------------------------------------------------------------


class TestGetRecentPoints:
    def test_empty_returns_empty(self):
        m = _make_mapper()
        assert m.get_recent_points() == []

    def test_returns_fresh_points(self):
        m = _make_mapper(max_age_seconds=60.0)
        _identity_cam(m)
        m.add_detection("cam01", (0.3, 0.3, 0.5, 0.5))
        pts = m.get_recent_points()
        assert len(pts) == 1

    def test_age_filter_excludes_old(self):
        m = _make_mapper()
        _identity_cam(m)
        old_ts = time.time() - 100.0
        m.add_detection("cam01", (0.3, 0.3, 0.5, 0.5), timestamp=old_ts)
        pts = m.get_recent_points(max_age_seconds=10.0)
        assert len(pts) == 0

    def test_age_filter_includes_recent(self):
        m = _make_mapper()
        _identity_cam(m)
        m.add_detection("cam01", (0.3, 0.3, 0.5, 0.5), timestamp=time.time())
        pts = m.get_recent_points(max_age_seconds=10.0)
        assert len(pts) == 1

    def test_camera_filter(self):
        m = _make_mapper()
        _identity_cam(m, "cam_A")
        _identity_cam(m, "cam_B")
        m.add_detection("cam_A", (0.3, 0.3, 0.5, 0.5))
        m.add_detection("cam_B", (0.3, 0.3, 0.5, 0.5))
        pts = m.get_recent_points(camera_filter="cam_A")
        assert all(p.camera_id == "cam_A" for p in pts)
        assert len(pts) == 1

    def test_rolling_buffer_max_points(self):
        m = _make_mapper(max_points=3)
        _identity_cam(m)
        for _ in range(5):
            m.add_detection("cam01", (0.3, 0.3, 0.5, 0.5))
        assert len(m._points) == 3


# ---------------------------------------------------------------------------
# get_occupancy_grid
# ---------------------------------------------------------------------------


class TestGetOccupancyGrid:
    def test_empty_returns_zeros(self):
        m = _make_mapper()
        grid = m.get_occupancy_grid(cols=10, rows=8)
        assert len(grid) == 8
        assert len(grid[0]) == 10
        assert all(v == 0.0 for row in grid for v in row)

    def test_grid_shape_correct(self):
        m = _make_mapper()
        _identity_cam(m)
        m.add_detection("cam01", (0.5, 0.5, 0.6, 0.6))
        grid = m.get_occupancy_grid(cols=20, rows=15)
        assert len(grid) == 15
        assert len(grid[0]) == 20

    def test_values_in_unit_range(self):
        m = _make_mapper()
        _identity_cam(m)
        for i in range(5):
            m.add_detection("cam01", (i * 0.1, i * 0.1, i * 0.1 + 0.1, i * 0.1 + 0.1))
        grid = m.get_occupancy_grid()
        for row in grid:
            for v in row:
                assert 0.0 <= v <= 1.0 + 1e-8

    def test_max_value_is_one(self):
        m = _make_mapper()
        _identity_cam(m)
        m.add_detection("cam01", (0.5, 0.5, 0.6, 0.6))
        grid = m.get_occupancy_grid(cols=40, rows=30)
        flat = [v for row in grid for v in row]
        assert max(flat) == pytest.approx(1.0, abs=1e-8)

    def test_old_points_excluded(self):
        m = _make_mapper()
        _identity_cam(m)
        m.add_detection("cam01", (0.5, 0.5, 0.6, 0.6), timestamp=time.time() - 200)
        grid = m.get_occupancy_grid(max_age_seconds=10.0)
        flat = [v for row in grid for v in row]
        assert max(flat) == 0.0


# ---------------------------------------------------------------------------
# to_map_data
# ---------------------------------------------------------------------------


class TestToMapData:
    def test_keys_present(self):
        m = _make_mapper()
        data = m.to_map_data()
        for key in ("floor", "cameras", "points", "occupancy_grid", "stats"):
            assert key in data

    def test_floor_dimensions(self):
        m = _make_mapper()
        data = m.to_map_data()
        assert abs(data["floor"]["width_m"] - _W) < 0.01
        assert abs(data["floor"]["height_m"] - _H) < 0.01

    def test_cameras_list(self):
        m = _make_mapper()
        _identity_cam(m, "cam01")
        data = m.to_map_data()
        assert "cam01" in data["cameras"]

    def test_occupancy_grid_keys(self):
        m = _make_mapper()
        data = m.to_map_data()
        og = data["occupancy_grid"]
        assert "cols" in og
        assert "rows" in og
        assert "data" in og

    def test_points_serialised(self):
        m = _make_mapper()
        _identity_cam(m)
        m.add_detection("cam01", (0.4, 0.4, 0.6, 0.6), person_id="t1")
        data = m.to_map_data()
        assert len(data["points"]) == 1
        assert "world_x" in data["points"][0]


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_stats_structure(self):
        m = _make_mapper()
        stats = m.get_stats()
        for key in ("registered_cameras", "cameras_active", "buffer_size",
                    "recent_points", "total_added", "floor_width_m",
                    "floor_height_m", "max_age_seconds", "uptime_seconds"):
            assert key in stats

    def test_total_added_increments(self):
        m = _make_mapper()
        _identity_cam(m)
        m.add_detection("cam01", (0.3, 0.3, 0.5, 0.5))
        m.add_detection("cam01", (0.4, 0.4, 0.6, 0.6))
        assert m.get_stats()["total_added"] == 2

    def test_registered_cameras_count(self):
        m = _make_mapper()
        _identity_cam(m, "cam01")
        _identity_cam(m, "cam02")
        assert m.get_stats()["registered_cameras"] == 2


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_add_detection(self):
        m = _make_mapper()
        _identity_cam(m)
        errors: list[Exception] = []

        def worker():
            try:
                m.add_detection("cam01", (0.4, 0.4, 0.6, 0.6))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_concurrent_register_and_query(self):
        m = _make_mapper()
        errors: list[Exception] = []

        def reg_worker(i: int):
            try:
                H = np.diag([_W, _H, 1.0])
                m.register_camera(f"cam_{i}", H)
            except Exception as e:
                errors.append(e)

        def query_worker():
            try:
                m.get_camera_ids()
                m.get_recent_points()
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=reg_worker, args=(i,)) for i in range(15)]
            + [threading.Thread(target=query_worker) for _ in range(10)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
