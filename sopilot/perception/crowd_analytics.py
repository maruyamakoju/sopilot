"""Crowd Flow Analytics — Phase 50.

Computes real-time crowd density, flow direction vectors, congestion
detection, and zone-capacity alerts from occupancy point streams.

Pure Python + stdlib (numpy for vector math). No external ML dependencies.

Architecture::

    SpatialMapper.get_recent_points()
              ↓
    CrowdAnalytics.update(points)
              ↓
    ┌─────────────────────────────────────┐
    │  density_grid  (NxM heat map)       │
    │  flow_vectors  (dx, dy per cell)    │
    │  congestion_zones  (threshold alert)│
    │  zone_capacity_alerts               │
    └─────────────────────────────────────┘

Usage::

    analytics = CrowdAnalytics(floor_width_m=20, floor_height_m=15)
    analytics.set_zone_capacity("lobby", max_persons=10)
    analytics.update(points)                     # call each frame
    state = analytics.get_state()                # UI snapshot
    alerts = analytics.get_capacity_alerts()     # active capacity violations
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GRID_COLS = 20          # default density-grid columns
_GRID_ROWS = 15          # default density-grid rows
_HISTORY_FRAMES = 10     # how many frames to keep for flow computation
_MAX_ALERTS = 100        # capacity-alert ring buffer size


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FlowVector:
    """Mean displacement vector for one grid cell over the last N frames."""

    col: int
    row: int
    dx: float            # world-space displacement (metres) per frame
    dy: float
    magnitude: float     # sqrt(dx² + dy²)
    sample_count: int


@dataclass
class CongestionZone:
    """A grid cell that has exceeded the congestion density threshold."""

    col: int
    row: int
    density: float       # normalised [0, 1]
    world_x: float       # centre in world coords
    world_y: float
    detected_at: float   # unix timestamp


@dataclass
class CapacityAlert:
    """Alert for a named zone whose occupant count exceeds its capacity."""

    zone_id: str
    zone_name: str
    current_count: int
    max_capacity: int
    utilisation: float   # current_count / max_capacity
    alert_level: str     # "warning" (>80%) | "critical" (>100%)
    raised_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "current_count": self.current_count,
            "max_capacity": self.max_capacity,
            "utilisation": round(self.utilisation, 3),
            "alert_level": self.alert_level,
            "raised_at": self.raised_at,
        }


# ---------------------------------------------------------------------------
# CrowdAnalytics
# ---------------------------------------------------------------------------


class CrowdAnalytics:
    """Real-time crowd density, flow, and congestion analytics.

    Parameters
    ----------
    floor_width_m, floor_height_m:
        Floor plan dimensions in metres.
    grid_cols, grid_rows:
        Density-grid resolution.
    congestion_threshold:
        Normalised density [0, 1] above which a cell is flagged as congested.
    history_frames:
        Number of frames retained for optical-flow-style vector computation.
    """

    def __init__(
        self,
        floor_width_m: float = 20.0,
        floor_height_m: float = 15.0,
        grid_cols: int = _GRID_COLS,
        grid_rows: int = _GRID_ROWS,
        congestion_threshold: float = 0.7,
        history_frames: int = _HISTORY_FRAMES,
    ) -> None:
        self._width = float(floor_width_m)
        self._height = float(floor_height_m)
        self._cols = max(4, int(grid_cols))
        self._rows = max(4, int(grid_rows))
        self._congestion_threshold = float(congestion_threshold)
        self._history_frames = max(2, int(history_frames))

        self._lock = threading.Lock()

        # Rolling person-position history: list of {pid: (x, y)} snapshots
        self._position_history: deque[dict[str, tuple[float, float]]] = deque(
            maxlen=self._history_frames
        )

        # Zone capacities: zone_id → (zone_name, max_persons)
        self._zone_capacities: dict[str, tuple[str, int]] = {}

        # Zone occupancy tracker: zone_id → current person set
        self._zone_persons: dict[str, set[str]] = {}

        # Capacity alerts ring buffer
        self._alerts: deque[CapacityAlert] = deque(maxlen=_MAX_ALERTS)

        # Cached state (updated each frame)
        self._last_density_grid: list[list[float]] = []
        self._last_flow_vectors: list[FlowVector] = []
        self._last_congestion_zones: list[CongestionZone] = []
        self._last_total_persons: int = 0
        self._last_mean_density: float = 0.0
        self._created_at = time.time()
        self._update_count = 0

    # ── Zone capacity management ───────────────────────────────────────────

    def set_zone_capacity(
        self, zone_id: str, max_persons: int, zone_name: str | None = None
    ) -> None:
        """Register or update the capacity limit for a named zone."""
        with self._lock:
            self._zone_capacities[zone_id] = (zone_name or zone_id, max(1, int(max_persons)))
            if zone_id not in self._zone_persons:
                self._zone_persons[zone_id] = set()

    def remove_zone_capacity(self, zone_id: str) -> bool:
        with self._lock:
            if zone_id in self._zone_capacities:
                del self._zone_capacities[zone_id]
                self._zone_persons.pop(zone_id, None)
                return True
        return False

    def update_zone_occupancy(
        self, zone_id: str, person_ids: set[str]
    ) -> None:
        """Update which persons are currently inside a zone.

        Called externally (e.g. from ZoneAnalyzer.update()) each frame.
        """
        with self._lock:
            if zone_id in self._zone_capacities:
                self._zone_persons[zone_id] = set(person_ids)

    # ── Main update ────────────────────────────────────────────────────────

    def update(
        self,
        points: list[Any],          # OccupancyPoint-like objects
        zone_occupancies: dict[str, set[str]] | None = None,
    ) -> None:
        """Process one frame of occupancy points.

        Parameters
        ----------
        points:
            Objects with ``person_id``, ``world_x``, ``world_y`` attributes.
        zone_occupancies:
            Optional {zone_id: {person_id, ...}} for capacity checking.
        """
        # Build person → position map
        current_positions: dict[str, tuple[float, float]] = {}
        for pt in points:
            pid = getattr(pt, "person_id", str(id(pt)))
            x = float(getattr(pt, "world_x", 0.0))
            y = float(getattr(pt, "world_y", 0.0))
            # Last write wins if duplicates exist
            current_positions[pid] = (x, y)

        with self._lock:
            self._position_history.append(current_positions)
            if zone_occupancies:
                for zid, pids in zone_occupancies.items():
                    if zid in self._zone_capacities:
                        self._zone_persons[zid] = set(pids)
            self._update_count += 1

        # Compute analytics (outside lock for performance)
        density = self._compute_density(list(current_positions.values()))
        flow = self._compute_flow()
        congestion = self._compute_congestion(density)
        alerts = self._evaluate_capacity_alerts()

        with self._lock:
            self._last_density_grid = density.tolist()
            self._last_flow_vectors = flow
            self._last_congestion_zones = congestion
            self._last_total_persons = len(current_positions)
            flat = density.flatten()
            self._last_mean_density = float(flat.mean())
            for alert in alerts:
                self._alerts.append(alert)

    # ── Internal computations ──────────────────────────────────────────────

    def _compute_density(
        self, positions: list[tuple[float, float]]
    ) -> np.ndarray:
        """KDE Gaussian density grid from world positions."""
        grid = np.zeros((self._rows, self._cols), dtype=np.float32)
        if not positions:
            return grid

        sigma_cells = max(
            1.0,
            min(self._cols, self._rows) * 0.1,
        )

        for (x, y) in positions:
            cx = x / self._width * self._cols
            cy = y / self._height * self._rows
            i0 = max(0, int(cy) - int(sigma_cells * 3))
            i1 = min(self._rows - 1, int(cy) + int(sigma_cells * 3))
            j0 = max(0, int(cx) - int(sigma_cells * 3))
            j1 = min(self._cols - 1, int(cx) + int(sigma_cells * 3))
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    d2 = (j - cx) ** 2 + (i - cy) ** 2
                    grid[i, j] += math.exp(-d2 / (2 * sigma_cells ** 2))

        gmax = grid.max()
        if gmax > 0:
            grid /= gmax
        return grid

    def _compute_flow(self) -> list[FlowVector]:
        """Estimate per-cell flow vectors from position history.

        Uses the centroid displacement of persons present in consecutive
        frames — a simple but fast optical-flow proxy.
        """
        with self._lock:
            history = list(self._position_history)

        if len(history) < 2:
            return []

        # Accumulate dx, dy per grid cell across all consecutive frame pairs
        dx_grid = np.zeros((self._rows, self._cols), dtype=np.float64)
        dy_grid = np.zeros((self._rows, self._cols), dtype=np.float64)
        count_grid = np.zeros((self._rows, self._cols), dtype=np.int32)

        for prev, curr in zip(history[:-1], history[1:]):
            common_pids = set(prev) & set(curr)
            for pid in common_pids:
                px, py = prev[pid]
                cx, cy = curr[pid]
                ddx = cx - px
                ddy = cy - py
                # Map current position to grid cell
                col = min(self._cols - 1, max(0, int(cx / self._width * self._cols)))
                row = min(self._rows - 1, max(0, int(cy / self._height * self._rows)))
                dx_grid[row, col] += ddx
                dy_grid[row, col] += ddy
                count_grid[row, col] += 1

        vectors: list[FlowVector] = []
        for r in range(self._rows):
            for c in range(self._cols):
                n = int(count_grid[r, c])
                if n == 0:
                    continue
                dx = float(dx_grid[r, c] / n)
                dy = float(dy_grid[r, c] / n)
                mag = math.sqrt(dx * dx + dy * dy)
                if mag > 1e-6:
                    vectors.append(FlowVector(col=c, row=r, dx=dx, dy=dy,
                                              magnitude=mag, sample_count=n))

        vectors.sort(key=lambda v: -v.magnitude)
        return vectors[:50]   # top-50 strongest vectors

    def _compute_congestion(
        self, density: np.ndarray
    ) -> list[CongestionZone]:
        """Identify grid cells above the congestion threshold."""
        zones: list[CongestionZone] = []
        now = time.time()
        cell_w = self._width / self._cols
        cell_h = self._height / self._rows
        for r in range(self._rows):
            for c in range(self._cols):
                v = float(density[r, c])
                if v >= self._congestion_threshold:
                    zones.append(CongestionZone(
                        col=c, row=r, density=round(v, 3),
                        world_x=round((c + 0.5) * cell_w, 2),
                        world_y=round((r + 0.5) * cell_h, 2),
                        detected_at=now,
                    ))
        return zones

    def _evaluate_capacity_alerts(self) -> list[CapacityAlert]:
        """Check zone capacities and produce new alert objects."""
        alerts: list[CapacityAlert] = []
        now = time.time()
        with self._lock:
            caps = dict(self._zone_capacities)
            persons = {k: set(v) for k, v in self._zone_persons.items()}

        for zone_id, (zone_name, max_cap) in caps.items():
            count = len(persons.get(zone_id, set()))
            util = count / max_cap
            if util > 0.8:
                level = "critical" if util > 1.0 else "warning"
                alerts.append(CapacityAlert(
                    zone_id=zone_id,
                    zone_name=zone_name,
                    current_count=count,
                    max_capacity=max_cap,
                    utilisation=round(util, 3),
                    alert_level=level,
                    raised_at=now,
                ))
        return alerts

    # ── State / query ──────────────────────────────────────────────────────

    def get_state(self) -> dict[str, Any]:
        """Full analytics snapshot for the UI/API."""
        with self._lock:
            density = self._last_density_grid
            flow = self._last_flow_vectors
            congestion = self._last_congestion_zones
            total = self._last_total_persons
            mean_dens = self._last_mean_density
            upd = self._update_count

        return {
            "total_persons": total,
            "mean_density": round(mean_dens, 4),
            "congestion_cell_count": len(congestion),
            "flow_vector_count": len(flow),
            "density_grid": {
                "cols": self._cols,
                "rows": self._rows,
                "data": density,
            },
            "flow_vectors": [
                {
                    "col": v.col, "row": v.row,
                    "dx": round(v.dx, 3), "dy": round(v.dy, 3),
                    "magnitude": round(v.magnitude, 3),
                    "sample_count": v.sample_count,
                }
                for v in flow
            ],
            "congestion_zones": [
                {
                    "col": z.col, "row": z.row,
                    "density": z.density,
                    "world_x": z.world_x, "world_y": z.world_y,
                }
                for z in congestion
            ],
            "update_count": upd,
            "uptime_seconds": round(time.time() - self._created_at, 1),
        }

    def get_capacity_alerts(self, active_only: bool = True) -> list[dict[str, Any]]:
        """Recent capacity alerts.

        Parameters
        ----------
        active_only:
            If True (default), only return alerts for zones currently over-
            capacity. Otherwise return all buffered alerts (newest first).
        """
        if active_only:
            return [a.to_dict() for a in self._evaluate_capacity_alerts()]
        with self._lock:
            all_alerts = list(self._alerts)
        all_alerts.sort(key=lambda a: -a.raised_at)
        return [a.to_dict() for a in all_alerts]

    def get_flow_summary(self) -> dict[str, Any]:
        """Dominant flow direction and overall crowd motion speed."""
        with self._lock:
            vectors = list(self._last_flow_vectors)

        if not vectors:
            return {
                "dominant_direction_deg": None,
                "mean_speed_ms": 0.0,
                "max_speed_ms": 0.0,
                "active_flow_cells": 0,
            }

        # Weighted mean direction
        total_weight = sum(v.magnitude for v in vectors)
        wx = sum(v.dx * v.magnitude for v in vectors) / total_weight
        wy = sum(v.dy * v.magnitude for v in vectors) / total_weight
        angle_deg = round(math.degrees(math.atan2(wy, wx)) % 360, 1)
        mean_speed = round(sum(v.magnitude for v in vectors) / len(vectors), 4)
        max_speed = round(max(v.magnitude for v in vectors), 4)

        return {
            "dominant_direction_deg": angle_deg,
            "mean_speed_ms": mean_speed,
            "max_speed_ms": max_speed,
            "active_flow_cells": len(vectors),
        }

    def get_stats(self) -> dict[str, Any]:
        """Compact statistics for the /status endpoint."""
        with self._lock:
            caps = len(self._zone_capacities)
            upd = self._update_count
            total = self._last_total_persons
            cong = len(self._last_congestion_zones)
            flow_n = len(self._last_flow_vectors)

        return {
            "enabled": True,
            "update_count": upd,
            "total_persons": total,
            "congestion_cells": cong,
            "flow_vectors": flow_n,
            "zone_capacities_registered": caps,
            "uptime_seconds": round(time.time() - self._created_at, 1),
            "floor_width_m": self._width,
            "floor_height_m": self._height,
            "grid_cols": self._cols,
            "grid_rows": self._rows,
            "congestion_threshold": self._congestion_threshold,
        }

    def reset(self) -> None:
        """Clear all history and cached state."""
        with self._lock:
            self._position_history.clear()
            self._alerts.clear()
            self._zone_persons = {k: set() for k in self._zone_capacities}
            self._last_density_grid = []
            self._last_flow_vectors = []
            self._last_congestion_zones = []
            self._last_total_persons = 0
            self._last_mean_density = 0.0
            self._update_count = 0
