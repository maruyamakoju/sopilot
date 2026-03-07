"""Spatial Intelligence — Multi-camera 2D floor map.

Phase 46: Projects detections from multiple cameras onto a unified 2D floor
plan using per-camera homography transforms. Computes occupancy grids and
maintains a rolling window of world-coordinate positions.

Architecture::

    Camera A (image coords) ──H_A──→ World (floor) coords
    Camera B (image coords) ──H_B──→ World (floor) coords
    Camera C (image coords) ──H_C──→ World (floor) coords
                                         ↓
                              SpatialMapper.get_occupancy_grid()
                                         ↓
                              2D heat map + live person dots

Homography::

    [xw]   [H00 H01 H02]   [u]
    [yw] = [H10 H11 H12] × [v]   then  x = xw/w,  y = yw/w
    [w ]   [H20 H21 H22]   [1]

    u, v ∈ [0, 1]  (normalized image coords: bbox center)
    x, y ∈ [0, floor_width_m] × [0, floor_height_m]

Usage::

    mapper = SpatialMapper(floor_width_m=20.0, floor_height_m=15.0)

    # Register cameras (identity = no transform, for testing)
    mapper.register_camera("cam01", np.eye(3), image_size=(1920, 1080))

    # Add detection each frame
    mapper.add_detection("cam01", bbox, person_id="t1", label="person",
                         confidence=0.9)

    # Get occupancy grid for UI
    grid = mapper.get_occupancy_grid(cols=40, rows=30)

    # Full map data for API
    data = mapper.to_map_data()
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_MAX_POINTS = 5_000        # rolling window
_DEFAULT_WIDTH_M = 20.0   # floor plan width in metres
_DEFAULT_HEIGHT_M = 15.0  # floor plan height in metres
_OCCUPANCY_SIGMA_M = 1.0  # Gaussian spread for KDE in metres


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraCalibration:
    """Homography matrix mapping image → floor coordinates for one camera."""

    camera_id: str
    homography: np.ndarray          # (3, 3) float64
    image_size: tuple[int, int]     # (width, height) in pixels
    registered_at: float = field(default_factory=time.time)


@dataclass
class OccupancyPoint:
    """A single projected person/object position on the floor plan."""

    point_id: str
    camera_id: str
    person_id: str
    label: str
    world_x: float          # metres from floor origin
    world_y: float          # metres from floor origin
    timestamp: float
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "camera_id": self.camera_id,
            "person_id": self.person_id,
            "label": self.label,
            "world_x": round(self.world_x, 3),
            "world_y": round(self.world_y, 3),
            "timestamp": self.timestamp,
            "confidence": round(self.confidence, 3),
            "age_seconds": round(time.time() - self.timestamp, 1),
        }


# ---------------------------------------------------------------------------
# SpatialMapper
# ---------------------------------------------------------------------------


class SpatialMapper:
    """Multi-camera 2D floor-plan mapper.

    Parameters
    ----------
    floor_width_m:
        Physical width of the floor plan in metres.
    floor_height_m:
        Physical height of the floor plan in metres.
    max_age_seconds:
        Drop points older than this from the rolling buffer.
    max_points:
        Maximum points in the rolling buffer.
    """

    def __init__(
        self,
        floor_width_m: float = _DEFAULT_WIDTH_M,
        floor_height_m: float = _DEFAULT_HEIGHT_M,
        max_age_seconds: float = 60.0,
        max_points: int = _MAX_POINTS,
    ) -> None:
        self._width = max(0.1, floor_width_m)
        self._height = max(0.1, floor_height_m)
        self._max_age = max_age_seconds
        self._max_points = max_points

        self._lock = threading.Lock()
        self._cameras: dict[str, CameraCalibration] = {}
        self._points: deque[OccupancyPoint] = deque(maxlen=max_points)
        self._add_count = 0
        self._created_at = time.time()

        logger.info(
            "SpatialMapper created (floor=%.1f×%.1fm, max_age=%.0fs)",
            floor_width_m,
            floor_height_m,
            max_age_seconds,
        )

    # ── Camera registration ────────────────────────────────────────────

    def register_camera(
        self,
        camera_id: str,
        homography: np.ndarray,
        image_size: tuple[int, int] = (1920, 1080),
    ) -> None:
        """Register a camera with its floor homography.

        Parameters
        ----------
        camera_id:
            Unique camera identifier.
        homography:
            3×3 homography matrix transforming normalised image coordinates
            ``(u, v)`` → world floor coordinates ``(x, y)`` in metres.
            Pass ``np.eye(3)`` for a camera that already outputs world coords.
        image_size:
            (width, height) in pixels (informational only).
        """
        H = np.asarray(homography, dtype=np.float64).reshape(3, 3)
        cal = CameraCalibration(
            camera_id=camera_id,
            homography=H,
            image_size=image_size,
        )
        with self._lock:
            self._cameras[camera_id] = cal
        logger.info(
            "SpatialMapper: registered camera %s (image=%dx%d)",
            camera_id,
            image_size[0],
            image_size[1],
        )

    def is_registered(self, camera_id: str) -> bool:
        with self._lock:
            return camera_id in self._cameras

    def get_camera_ids(self) -> list[str]:
        with self._lock:
            return list(self._cameras.keys())

    # ── Projection ─────────────────────────────────────────────────────

    def project_point(
        self, camera_id: str, u: float, v: float
    ) -> tuple[float, float] | None:
        """Project a normalised image point to world floor coordinates.

        Parameters
        ----------
        camera_id:
            Registered camera identifier.
        u, v:
            Normalised image coordinates in ``[0, 1]``.

        Returns
        -------
        tuple[float, float] | None
            ``(world_x, world_y)`` in metres, or ``None`` if unregistered
            or degenerate transform.
        """
        with self._lock:
            cal = self._cameras.get(camera_id)
        if cal is None:
            return None

        # Apply homography: [xw, yw, w]^T = H @ [u, v, 1]^T
        p = cal.homography @ np.array([u, v, 1.0], dtype=np.float64)
        w = p[2]
        if abs(w) < 1e-10:
            return None
        x, y = p[0] / w, p[1] / w

        # Clamp to floor bounds
        x = max(0.0, min(self._width, x))
        y = max(0.0, min(self._height, y))
        return float(x), float(y)

    def add_detection(
        self,
        camera_id: str,
        bbox: Any,  # BBox with x1/y1/x2/y2 in [0,1], or tuple
        person_id: str = "",
        label: str = "person",
        confidence: float = 1.0,
        timestamp: float | None = None,
    ) -> OccupancyPoint | None:
        """Project a bounding box centre to the floor and store the point.

        Parameters
        ----------
        camera_id:
            Source camera.
        bbox:
            Object with ``.x1``, ``.y1``, ``.x2``, ``.y2`` in ``[0, 1]``,
            or a 4-tuple ``(x1, y1, x2, y2)``.
        person_id:
            Track or person identifier.
        label:
            Object class label.
        confidence:
            Detection confidence.
        timestamp:
            Unix timestamp (default: now).

        Returns
        -------
        OccupancyPoint | None
            The projected point, or None if camera is unregistered.
        """
        # Extract bbox centre
        if hasattr(bbox, "x1"):
            cx = (bbox.x1 + bbox.x2) / 2.0
            cy = (bbox.y1 + bbox.y2) / 2.0
        else:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

        world = self.project_point(camera_id, cx, cy)
        if world is None:
            return None

        world_x, world_y = world
        ts = timestamp or time.time()
        pid = person_id or str(uuid.uuid4())[:8]

        pt = OccupancyPoint(
            point_id=str(uuid.uuid4()),
            camera_id=camera_id,
            person_id=pid,
            label=label,
            world_x=world_x,
            world_y=world_y,
            timestamp=ts,
            confidence=float(confidence),
        )

        with self._lock:
            self._points.append(pt)
            self._add_count += 1

        return pt

    # ── Query ──────────────────────────────────────────────────────────

    def get_recent_points(
        self,
        max_age_seconds: float | None = None,
        camera_filter: str | None = None,
    ) -> list[OccupancyPoint]:
        """Return points within the age window."""
        cutoff = time.time() - (max_age_seconds or self._max_age)
        with self._lock:
            pts = list(self._points)
        result = [
            p for p in pts
            if p.timestamp >= cutoff
            and (camera_filter is None or p.camera_id == camera_filter)
        ]
        return result

    def get_occupancy_grid(
        self,
        cols: int = 40,
        rows: int = 30,
        max_age_seconds: float | None = None,
        sigma_cells: float | None = None,
    ) -> list[list[float]]:
        """Compute a KDE-smoothed occupancy grid.

        Parameters
        ----------
        cols, rows:
            Grid resolution (number of cells).
        max_age_seconds:
            Age window (default: self._max_age).
        sigma_cells:
            Gaussian sigma in cells (default: auto from floor scale).

        Returns
        -------
        list[list[float]]
            ``rows × cols`` grid of density values in ``[0, 1]``.
        """
        points = self.get_recent_points(max_age_seconds)
        grid = np.zeros((rows, cols), dtype=np.float64)

        if not points:
            return grid.tolist()

        # Sigma in cells (1 metre in grid units)
        sigma = sigma_cells or max(
            1.0,
            _OCCUPANCY_SIGMA_M / self._width * cols,
        )

        for pt in points:
            # Fractional cell position
            cx = pt.world_x / self._width * cols
            cy = pt.world_y / self._height * rows

            # Gaussian splat onto nearby cells
            i0 = max(0, int(cy) - int(sigma * 3))
            i1 = min(rows - 1, int(cy) + int(sigma * 3))
            j0 = max(0, int(cx) - int(sigma * 3))
            j1 = min(cols - 1, int(cx) + int(sigma * 3))

            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    d2 = (j - cx) ** 2 + (i - cy) ** 2
                    grid[i, j] += math.exp(-d2 / (2 * sigma ** 2))

        # Normalise to [0, 1]
        gmax = grid.max()
        if gmax > 0:
            grid /= gmax

        return grid.tolist()

    # ── Map data for frontend ──────────────────────────────────────────

    def to_map_data(
        self,
        grid_cols: int = 40,
        grid_rows: int = 30,
        max_age_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Return all data needed to render the floor map in the UI.

        Returns
        -------
        dict
            Keys: ``floor``, ``cameras``, ``points``, ``occupancy_grid``,
            ``stats``.
        """
        pts = self.get_recent_points(max_age_seconds)
        grid = self.get_occupancy_grid(grid_cols, grid_rows, max_age_seconds)

        with self._lock:
            camera_ids = list(self._cameras.keys())

        return {
            "floor": {
                "width_m": self._width,
                "height_m": self._height,
            },
            "cameras": camera_ids,
            "points": [p.to_dict() for p in pts],
            "occupancy_grid": {
                "cols": grid_cols,
                "rows": grid_rows,
                "data": grid,
            },
            "stats": self.get_stats(),
        }

    # ── Stats ──────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            n_cams = len(self._cameras)
            n_pts = len(self._points)
        recent = self.get_recent_points()
        cameras_active = len({p.camera_id for p in recent})
        return {
            "registered_cameras": n_cams,
            "cameras_active": cameras_active,
            "buffer_size": n_pts,
            "recent_points": len(recent),
            "total_added": self._add_count,
            "floor_width_m": self._width,
            "floor_height_m": self._height,
            "max_age_seconds": self._max_age,
            "uptime_seconds": round(time.time() - self._created_at, 1),
        }
