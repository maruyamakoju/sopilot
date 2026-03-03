"""Multi-agent perception coordination.

Manages multiple named perception engine instances and coordinates their
shared understanding of the environment.

- SharedSpatialMap: merges occupancy grids from multiple camera agents
- GlobalEntityRegistry: cross-camera entity ID resolution
- MultiAgentCoordinator: orchestrates coordination, event broadcasting
- Module-level singleton for easy access across sessions
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MAX_GLOBAL_ENTITIES = 1000
MAX_EVENT_BUFFER = 500


@dataclass
class AgentInfo:
    agent_id: str
    camera_id: str
    location: str
    registered_at: float
    last_seen_at: float
    frame_count: int
    spatial_contribution: float = 1.0  # weight in shared map merge

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "camera_id": self.camera_id,
            "location": self.location,
            "registered_at": self.registered_at,
            "last_seen_at": self.last_seen_at,
            "frame_count": self.frame_count,
            "spatial_contribution": self.spatial_contribution,
        }


@dataclass
class GlobalEntity:
    global_id: str
    local_ids: dict[str, int]   # agent_id -> local entity_id
    label: str
    last_seen_at: float
    last_seen_agent: str
    sighting_count: int = 1

    def to_dict(self) -> dict:
        return {
            "global_id": self.global_id,
            "local_ids": dict(self.local_ids),
            "label": self.label,
            "last_seen_at": self.last_seen_at,
            "last_seen_agent": self.last_seen_agent,
            "sighting_count": self.sighting_count,
        }


class SharedSpatialMap:
    """Merges occupancy grids from multiple agents using weighted EMA."""

    def __init__(
        self,
        grid_w: int = 8,
        grid_h: int = 6,
        merge_alpha: float = 0.3,
    ) -> None:
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.merge_alpha = merge_alpha
        self._grid = np.zeros((grid_h, grid_w), dtype=np.float32)
        self._lock = threading.RLock()
        self._update_count = 0

    def update_from_agent(
        self,
        agent_id: str,
        occupancy_grid: list[list[float]],
        weight: float = 1.0,
    ) -> None:
        """Merge agent's occupancy grid into global map via weighted EMA."""
        try:
            arr = np.array(occupancy_grid, dtype=np.float32)
            if arr.shape != (self.grid_h, self.grid_w):
                # Resize if mismatch (nearest-neighbor)
                from numpy import interp
                arr_resized = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
                for r in range(self.grid_h):
                    for c in range(self.grid_w):
                        sr = int(r * arr.shape[0] / self.grid_h)
                        sc = int(c * arr.shape[1] / self.grid_w)
                        arr_resized[r, c] = arr[min(sr, arr.shape[0]-1), min(sc, arr.shape[1]-1)]
                arr = arr_resized
        except Exception as e:
            logger.warning("SharedSpatialMap: invalid grid from %s: %s", agent_id, e)
            return
        with self._lock:
            effective_alpha = self.merge_alpha * max(0.0, min(2.0, weight))
            self._grid = (1.0 - effective_alpha) * self._grid + effective_alpha * arr
            self._update_count += 1

    def get_global_grid(self) -> np.ndarray:
        with self._lock:
            return self._grid.copy()

    def get_hotspots(self, top_n: int = 5) -> list[dict]:
        with self._lock:
            flat = self._grid.flatten()
            indices = np.argsort(flat)[::-1][:top_n]
            return [
                {
                    "row": int(idx // self.grid_w),
                    "col": int(idx % self.grid_w),
                    "value": float(flat[idx]),
                }
                for idx in indices
                if flat[idx] > 0
            ]

    def get_state_dict(self) -> dict:
        with self._lock:
            return {
                "grid_w": self.grid_w,
                "grid_h": self.grid_h,
                "merge_alpha": self.merge_alpha,
                "update_count": self._update_count,
                "max_occupancy": float(self._grid.max()),
                "mean_occupancy": float(self._grid.mean()),
            }

    def reset(self) -> None:
        with self._lock:
            self._grid[:] = 0.0
            self._update_count = 0


class MultiAgentCoordinator:
    """Registers agents, coordinates shared spatial map, resolves entities."""

    def __init__(
        self,
        max_agents: int = 16,
        entity_merge_distance: float = 0.1,
        agent_timeout_seconds: float = 30.0,
        grid_w: int = 8,
        grid_h: int = 6,
    ) -> None:
        self.max_agents = max_agents
        self.entity_merge_distance = entity_merge_distance
        self.agent_timeout_seconds = agent_timeout_seconds
        self._lock = threading.RLock()
        self._agents: dict[str, AgentInfo] = {}
        self._global_entities: dict[str, GlobalEntity] = {}
        self._events: deque[dict] = deque(maxlen=MAX_EVENT_BUFFER)
        self._shared_map = SharedSpatialMap(grid_w=grid_w, grid_h=grid_h)
        self._total_events_broadcast = 0

    # ── Agent management ──────────────────────────────────────────────────────

    def register_agent(
        self, agent_id: str, camera_id: str, location: str = ""
    ) -> AgentInfo:
        with self._lock:
            now = time.time()
            if agent_id in self._agents:
                # Re-registration: update fields but keep frame_count
                info = self._agents[agent_id]
                info.camera_id = camera_id
                info.location = location
                info.last_seen_at = now
                return info
            if len(self._agents) >= self.max_agents:
                # Evict oldest inactive agent
                oldest = min(self._agents.values(), key=lambda a: a.last_seen_at)
                del self._agents[oldest.agent_id]
                logger.warning("MultiAgentCoordinator: evicted agent %s (max_agents=%d)", oldest.agent_id, self.max_agents)
            info = AgentInfo(
                agent_id=agent_id,
                camera_id=camera_id,
                location=location,
                registered_at=now,
                last_seen_at=now,
                frame_count=0,
            )
            self._agents[agent_id] = info
            logger.info("MultiAgentCoordinator: registered agent %s (camera=%s)", agent_id, camera_id)
            return info

    def unregister_agent(self, agent_id: str) -> bool:
        with self._lock:
            if agent_id in self._agents:
                del self._agents[agent_id]
                logger.info("MultiAgentCoordinator: unregistered agent %s", agent_id)
                return True
            return False

    def heartbeat(self, agent_id: str, _now: float | None = None) -> bool:
        with self._lock:
            if agent_id not in self._agents:
                return False
            self._agents[agent_id].last_seen_at = _now or time.time()
            return True

    def get_active_agents(self, _now: float | None = None) -> list[AgentInfo]:
        now = _now or time.time()
        with self._lock:
            return [
                a for a in self._agents.values()
                if (now - a.last_seen_at) <= self.agent_timeout_seconds
            ]

    def get_agent(self, agent_id: str) -> AgentInfo | None:
        with self._lock:
            return self._agents.get(agent_id)

    # ── Spatial coordination ──────────────────────────────────────────────────

    def submit_spatial_update(
        self,
        agent_id: str,
        occupancy_grid: list[list[float]],
        _now: float | None = None,
    ) -> None:
        """Register agent spatial contribution and update shared map."""
        with self._lock:
            info = self._agents.get(agent_id)
            if info is None:
                return
            info.last_seen_at = _now or time.time()
            info.frame_count += 1
            weight = info.spatial_contribution
        self._shared_map.update_from_agent(agent_id, occupancy_grid, weight)

    def get_shared_map(self) -> SharedSpatialMap:
        return self._shared_map

    # ── Entity coordination ───────────────────────────────────────────────────

    def submit_entities(
        self,
        agent_id: str,
        entities: list[dict],
        _now: float | None = None,
    ) -> list[str]:
        """Register local entities. Returns list of global_ids assigned.

        entities: list of dicts with keys: entity_id (int), label (str),
                  cx (float, normalized), cy (float, normalized)
        """
        now = _now or time.time()
        global_ids: list[str] = []
        with self._lock:
            for ent in entities:
                local_id = int(ent.get("entity_id", 0))
                label = str(ent.get("label", "unknown"))
                cx = float(ent.get("cx", 0.5))
                cy = float(ent.get("cy", 0.5))
                gid = self._find_or_create_global_entity(agent_id, local_id, label, cx, cy, now)
                global_ids.append(gid)
            # Prune oldest if over limit
            if len(self._global_entities) > MAX_GLOBAL_ENTITIES:
                sorted_ids = sorted(
                    self._global_entities.keys(),
                    key=lambda k: self._global_entities[k].last_seen_at,
                )
                for old_id in sorted_ids[: len(self._global_entities) - MAX_GLOBAL_ENTITIES]:
                    del self._global_entities[old_id]
        return global_ids

    def get_global_entities(self) -> list[GlobalEntity]:
        with self._lock:
            return list(self._global_entities.values())

    def get_entity_by_global_id(self, global_id: str) -> GlobalEntity | None:
        with self._lock:
            return self._global_entities.get(global_id)

    # ── Event broadcasting ────────────────────────────────────────────────────

    def broadcast_event(self, agent_id: str, event_dict: dict, _now: float | None = None) -> None:
        """Buffer an event from an agent for cross-agent subscribers."""
        with self._lock:
            enriched = dict(event_dict)
            enriched["source_agent_id"] = agent_id
            enriched["broadcast_at"] = _now or time.time()
            self._events.append(enriched)
            self._total_events_broadcast += 1

    def get_recent_events(self, n: int = 20) -> list[dict]:
        with self._lock:
            items = list(self._events)
            return items[-n:] if n > 0 else []

    # ── State ─────────────────────────────────────────────────────────────────

    def get_state_dict(self, _now: float | None = None) -> dict:
        now = _now or time.time()
        with self._lock:
            active = [
                a for a in self._agents.values()
                if (now - a.last_seen_at) <= self.agent_timeout_seconds
            ]
            return {
                "total_agents": len(self._agents),
                "active_agents": len(active),
                "total_global_entities": len(self._global_entities),
                "total_events_broadcast": self._total_events_broadcast,
                "event_buffer_size": len(self._events),
                "shared_map": self._shared_map.get_state_dict(),
                "agents": [a.to_dict() for a in self._agents.values()],
            }

    def reset(self) -> None:
        with self._lock:
            self._agents.clear()
            self._global_entities.clear()
            self._events.clear()
            self._total_events_broadcast = 0
            self._shared_map.reset()

    # ── Private ───────────────────────────────────────────────────────────────

    def _find_or_create_global_entity(
        self,
        agent_id: str,
        local_id: int,
        label: str,
        cx: float,
        cy: float,
        now: float,
    ) -> str:
        import math
        # Check if this local_id for this agent is already mapped
        for ge in self._global_entities.values():
            if agent_id in ge.local_ids and ge.local_ids[agent_id] == local_id:
                ge.last_seen_at = now
                ge.last_seen_agent = agent_id
                ge.sighting_count += 1
                return ge.global_id
        # Try spatial proximity merge with same-label entities from other agents
        for ge in self._global_entities.values():
            if ge.label != label:
                continue
            if agent_id in ge.local_ids:
                continue
            # Check if any known position is close (we don't store positions, use heuristic)
            # Use sighting_count as proxy: only merge if recently seen
            if (now - ge.last_seen_at) < 5.0:
                ge.local_ids[agent_id] = local_id
                ge.last_seen_at = now
                ge.last_seen_agent = agent_id
                ge.sighting_count += 1
                return ge.global_id
        # Create new global entity
        gid = str(uuid.uuid4())
        self._global_entities[gid] = GlobalEntity(
            global_id=gid,
            local_ids={agent_id: local_id},
            label=label,
            last_seen_at=now,
            last_seen_agent=agent_id,
        )
        return gid


# ── Module-level singleton ────────────────────────────────────────────────────

_coordinator: MultiAgentCoordinator | None = None
_coordinator_lock = threading.Lock()


def get_coordinator() -> MultiAgentCoordinator:
    global _coordinator
    with _coordinator_lock:
        if _coordinator is None:
            _coordinator = MultiAgentCoordinator()
        return _coordinator


def reset_coordinator() -> None:
    global _coordinator
    with _coordinator_lock:
        _coordinator = None
