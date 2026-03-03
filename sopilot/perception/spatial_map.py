"""2D spatial occupancy grid for surveillance scenes.

Maps normalized [0,1] entity bbox centers onto a configurable grid.
Maintains EMA-smoothed occupancy counts, dominant labels per cell,
and aggregates velocity vectors for flow estimation.

Thread-safe under RLock. No external dependencies beyond stdlib + numpy.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SpatialCell:
    gx: int               # grid column index
    gy: int               # grid row index
    occupancy: float      # EMA-smoothed entity count [0+]
    dominant_label: str   # most frequent label seen recently
    last_seen: float      # timestamp of last entity in this cell
    depth_mean: float     # mean depth_relative of entities in cell (0=near, 1=far)
    velocity_dx: float    # mean horizontal velocity
    velocity_dy: float    # mean vertical velocity

    def to_dict(self) -> dict:
        return {
            "gx": self.gx,
            "gy": self.gy,
            "occupancy": self.occupancy,
            "dominant_label": self.dominant_label,
            "last_seen": self.last_seen,
            "depth_mean": self.depth_mean,
            "velocity_dx": self.velocity_dx,
            "velocity_dy": self.velocity_dy,
        }


@dataclass
class SpatialSnapshot:
    timestamp: float
    frame_number: int
    grid_w: int
    grid_h: int
    total_entities: int
    crowd_density: float                    # fraction of cells occupied
    flow_dx: float                          # dominant flow direction x
    flow_dy: float                          # dominant flow direction y
    hotspots: list[dict]                    # top-N cells by occupancy
    occupancy_grid: list[list[float]]       # [gy][gx] occupancy values

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "grid_w": self.grid_w,
            "grid_h": self.grid_h,
            "total_entities": self.total_entities,
            "crowd_density": self.crowd_density,
            "flow_dx": self.flow_dx,
            "flow_dy": self.flow_dy,
            "hotspots": self.hotspots,
            "occupancy_grid": self.occupancy_grid,
        }


class SpatialMap:
    """Entity spatial distribution tracker on a 2D grid.

    Updates per frame with entity positions. Maintains EMA-smoothed
    occupancy and aggregates velocity vectors for flow estimation.

    Parameters
    ----------
    grid_w, grid_h : int
        Grid dimensions. Default 8x6.
    ema_alpha : float
        EMA smoothing factor [0,1]. Default 0.3.
    hotspot_top_n : int
        Number of hotspot cells to report. Default 3.
    max_history : int
        Maximum snapshots to retain in history. Default 60.
    """

    def __init__(
        self,
        grid_w: int = 8,
        grid_h: int = 6,
        ema_alpha: float = 0.3,
        hotspot_top_n: int = 3,
        max_history: int = 60,
    ) -> None:
        self._gw = grid_w
        self._gh = grid_h
        self._alpha = max(0.0, min(1.0, ema_alpha))
        self._top_n = hotspot_top_n
        self._max_history = max_history
        self._lock = threading.RLock()
        # occupancy[gy][gx]: EMA entity count
        self._occupancy = np.zeros((grid_h, grid_w), dtype=np.float64)
        # label counts: {(gy,gx): {label: count}}
        self._label_counts: dict[tuple[int, int], dict[str, int]] = {}
        # last_seen[gy][gx]
        self._last_seen = np.zeros((grid_h, grid_w), dtype=np.float64)
        # depth[gy][gx]: mean depth_relative
        self._depth = np.full((grid_h, grid_w), 0.5, dtype=np.float64)
        # velocity accumulator per cell (dx, dy sums and count)
        self._vel_sum = np.zeros((grid_h, grid_w, 2), dtype=np.float64)
        self._vel_count = np.zeros((grid_h, grid_w), dtype=np.int32)
        self._history: list[SpatialSnapshot] = []

    def update(
        self,
        world_state: Any,
        depth_estimates: list | None = None,
        timestamp: float | None = None,
    ) -> SpatialSnapshot:
        """Update grid from world_state and return current snapshot."""
        ts = timestamp if timestamp is not None else getattr(world_state, "timestamp", time.time())
        frame_num = getattr(world_state, "frame_number", 0)

        with self._lock:
            # Build depth lookup: entity_id -> depth_relative
            depth_by_id: dict[int, float] = {}
            if depth_estimates:
                for de in depth_estimates:
                    depth_by_id[getattr(de, "entity_id", -1)] = getattr(de, "depth_relative", 0.5)

            # Build velocity lookup: entity_id -> (dx, dy)
            vel_by_id: dict[int, tuple[float, float]] = {}
            active_tracks = getattr(world_state, "active_tracks", {})
            for tid, track in active_tracks.items():
                vel_by_id[getattr(track, "entity_id", tid)] = getattr(track, "velocity", (0.0, 0.0))

            # Current frame entity positions
            entities = []
            sg = getattr(world_state, "scene_graph", None)
            if sg is not None:
                entities = list(getattr(sg, "entities", []))

            # Decay existing occupancy (EMA toward 0 for unoccupied cells)
            self._occupancy *= (1.0 - self._alpha)
            self._vel_sum[:] = 0.0
            self._vel_count[:] = 0

            # Update from entities
            for ent in entities:
                bbox = getattr(ent, "bbox", None)
                if bbox is None:
                    continue
                cx = getattr(bbox, "x", 0.5) + getattr(bbox, "w", 0.1) / 2.0
                cy = getattr(bbox, "y", 0.5) + getattr(bbox, "h", 0.1) / 2.0
                gx = min(self._gw - 1, max(0, int(cx * self._gw)))
                gy = min(self._gh - 1, max(0, int(cy * self._gh)))

                self._occupancy[gy, gx] += self._alpha
                self._last_seen[gy, gx] = ts

                label = getattr(ent, "label", "")
                key = (gy, gx)
                if key not in self._label_counts:
                    self._label_counts[key] = {}
                self._label_counts[key][label] = self._label_counts[key].get(label, 0) + 1

                eid = getattr(ent, "entity_id", 0)
                dr = depth_by_id.get(eid, 0.5)
                # EMA depth
                self._depth[gy, gx] = (1 - self._alpha) * self._depth[gy, gx] + self._alpha * dr

                vx, vy = vel_by_id.get(eid, (0.0, 0.0))
                self._vel_sum[gy, gx, 0] += vx
                self._vel_sum[gy, gx, 1] += vy
                self._vel_count[gy, gx] += 1

            # Build snapshot
            entity_count = len(entities)
            occupied = int(np.sum(self._occupancy > 0.05))
            crowd_density = occupied / (self._gw * self._gh)

            # Flow vector: mean velocity of all entities
            flow_dx, flow_dy = 0.0, 0.0
            total_tracked = int(np.sum(self._vel_count))
            if total_tracked > 0:
                flow_dx = float(np.sum(self._vel_sum[:, :, 0])) / total_tracked
                flow_dy = float(np.sum(self._vel_sum[:, :, 1])) / total_tracked

            # Hotspots: top N cells by occupancy
            flat_occ = self._occupancy.flatten()
            top_indices = np.argsort(flat_occ)[::-1][: self._top_n]
            hotspots = []
            for idx in top_indices:
                gy_h = int(idx // self._gw)
                gx_h = int(idx % self._gw)
                occ_val = float(self._occupancy[gy_h, gx_h])
                if occ_val < 0.01:
                    break
                key = (gy_h, gx_h)
                lc = self._label_counts.get(key, {})
                dom = max(lc, key=lc.get) if lc else ""
                hotspots.append(
                    {
                        "gx": gx_h,
                        "gy": gy_h,
                        "occupancy": round(occ_val, 4),
                        "dominant_label": dom,
                    }
                )

            # occupancy grid as nested list
            occ_grid = self._occupancy.tolist()

            snapshot = SpatialSnapshot(
                timestamp=ts,
                frame_number=frame_num,
                grid_w=self._gw,
                grid_h=self._gh,
                total_entities=entity_count,
                crowd_density=round(crowd_density, 4),
                flow_dx=round(flow_dx, 4),
                flow_dy=round(flow_dy, 4),
                hotspots=hotspots,
                occupancy_grid=occ_grid,
            )
            self._history.append(snapshot)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]
            return snapshot

    def get_crowd_density(self) -> float:
        """Fraction of cells with non-trivial occupancy (>0.05)."""
        with self._lock:
            occupied = int(np.sum(self._occupancy > 0.05))
            return round(occupied / (self._gw * self._gh), 4)

    def get_flow_vector(self) -> tuple[float, float]:
        """Return the mean velocity vector from the last update."""
        with self._lock:
            total = int(np.sum(self._vel_count))
            if total == 0:
                return (0.0, 0.0)
            dx = float(np.sum(self._vel_sum[:, :, 0])) / total
            dy = float(np.sum(self._vel_sum[:, :, 1])) / total
            return (round(dx, 4), round(dy, 4))

    def get_hotspots(self, top_n: int | None = None) -> list[dict]:
        """Return top N cells by occupancy (default: hotspot_top_n from init)."""
        with self._lock:
            n = top_n if top_n is not None else self._top_n
            flat = self._occupancy.flatten()
            indices = np.argsort(flat)[::-1][:n]
            result = []
            for idx in indices:
                gy = int(idx // self._gw)
                gx = int(idx % self._gw)
                occ = float(self._occupancy[gy, gx])
                if occ < 0.01:
                    break
                result.append({"gx": gx, "gy": gy, "occupancy": round(occ, 4)})
            return result

    def get_risk_cells(self, threshold: float = 0.5) -> list[dict]:
        """Return cells with occupancy >= threshold."""
        with self._lock:
            result = []
            for gy in range(self._gh):
                for gx in range(self._gw):
                    occ = float(self._occupancy[gy, gx])
                    if occ >= threshold:
                        result.append({"gx": gx, "gy": gy, "occupancy": round(occ, 4)})
            return result

    def get_grid_array(self) -> np.ndarray:
        """Return grid_h x grid_w occupancy array (copy)."""
        with self._lock:
            return self._occupancy.copy()

    def get_history(self, n: int = 10) -> list[SpatialSnapshot]:
        with self._lock:
            return list(self._history[-n:])

    def get_state_dict(self) -> dict:
        with self._lock:
            total_vel = max(1, int(np.sum(self._vel_count)))
            return {
                "grid_w": self._gw,
                "grid_h": self._gh,
                "crowd_density": self.get_crowd_density(),
                "flow_dx": round(float(np.sum(self._vel_sum[:, :, 0])) / total_vel, 4),
                "flow_dy": round(float(np.sum(self._vel_sum[:, :, 1])) / total_vel, 4),
                "hotspots": self.get_hotspots(),
                "history_size": len(self._history),
            }

    def reset(self) -> None:
        with self._lock:
            self._occupancy[:] = 0.0
            self._last_seen[:] = 0.0
            self._depth[:] = 0.5
            self._vel_sum[:] = 0.0
            self._vel_count[:] = 0
            self._label_counts.clear()
            self._history.clear()
