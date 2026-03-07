"""Zone Analyzer — polygon zones, dwell time, and flow analytics.

Phase 46: Defines floor-plan zones (restricted, entry, exit, etc.) and
tracks per-person dwell time, zone enter/exit events, and inter-zone
flow vectors.

Architecture::

    ZoneAnalyzer.add_zone("restricted_A", polygon=[(0,0),(5,0),(5,3),(0,3)])
    ZoneAnalyzer.update(occupancy_points, timestamp)
        → detect enter/exit events per person
        → accumulate dwell time
    ZoneAnalyzer.get_flow_vectors()   → movement direction between zones
    ZoneAnalyzer.get_dwell_stats()    → per-zone avg/max dwell, occupancy

Point-in-polygon uses the ray casting algorithm (O(n) per point per zone).

Usage::

    analyzer = ZoneAnalyzer()
    analyzer.add_zone("entry", [(0,0),(3,0),(3,15),(0,15)], "entry")
    analyzer.add_zone("restricted", [(8,5),(12,5),(12,10),(8,10)], "restricted")

    events = analyzer.update(points, time.time())
    for ev in events:
        if ev.event_type == "enter" and ev.zone_type == "restricted":
            print(f"ALERT: {ev.person_id} entered restricted zone!")
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_MAX_EVENTS = 500     # rolling event buffer
_DWELL_ALERT_S = 30.0  # loitering threshold for restricted zones


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ZoneDefinition:
    """A named polygon zone on the floor plan."""

    zone_id: str
    name: str
    polygon: list[tuple[float, float]]   # world (x, y) in metres
    zone_type: str = "normal"            # "restricted", "normal", "entry", "exit"
    color: str = "#64748b"               # hex colour for UI

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "polygon": [{"x": x, "y": y} for x, y in self.polygon],
            "zone_type": self.zone_type,
            "color": self.color,
        }


@dataclass
class ZoneEvent:
    """A person entering, dwelling in, or exiting a zone."""

    event_id: str
    zone_id: str
    zone_name: str
    zone_type: str
    person_id: str
    camera_id: str
    event_type: str          # "enter", "dwell", "exit"
    timestamp: float
    dwell_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "zone_type": self.zone_type,
            "person_id": self.person_id,
            "camera_id": self.camera_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "dwell_seconds": round(self.dwell_seconds, 1),
        }


# ---------------------------------------------------------------------------
# ZoneAnalyzer
# ---------------------------------------------------------------------------


class ZoneAnalyzer:
    """Polygon-based zone detection, dwell time tracking, and flow analytics.

    Parameters
    ----------
    dwell_alert_seconds:
        Minimum dwell time (seconds) before a ``dwell`` event is emitted
        for persons in restricted zones.
    max_events:
        Maximum zone events to keep in the rolling buffer.
    """

    def __init__(
        self,
        dwell_alert_seconds: float = _DWELL_ALERT_S,
        max_events: int = _MAX_EVENTS,
    ) -> None:
        self._dwell_threshold = dwell_alert_seconds
        self._max_events = max_events

        self._lock = threading.Lock()
        self._zones: dict[str, ZoneDefinition] = {}
        self._events: deque[ZoneEvent] = deque(maxlen=max_events)

        # Tracking state: person_id → {zone_id → enter_time}
        self._person_zones: dict[str, dict[str, float]] = defaultdict(dict)
        # Per-zone cumulative dwell time: zone_id → list[float]
        self._dwell_history: dict[str, list[float]] = defaultdict(list)
        # Flow counter: (from_zone, to_zone) → count
        self._flow_counts: dict[tuple[str, str], int] = defaultdict(int)
        # Previous zone per person: person_id → set[zone_id]
        self._prev_zones: dict[str, set[str]] = defaultdict(set)

        self._update_count = 0
        self._event_count = 0
        self._created_at = time.time()

        logger.info("ZoneAnalyzer created (dwell_threshold=%.0fs)", dwell_alert_seconds)

    # ── Zone management ────────────────────────────────────────────────

    def add_zone(
        self,
        zone_id: str,
        polygon: list[tuple[float, float]],
        zone_type: str = "normal",
        name: str | None = None,
        color: str | None = None,
    ) -> ZoneDefinition:
        """Register a polygon zone.

        Parameters
        ----------
        zone_id:
            Unique zone identifier.
        polygon:
            List of ``(x, y)`` world coordinates forming the zone boundary
            (min 3 vertices).
        zone_type:
            One of ``"restricted"``, ``"normal"``, ``"entry"``, ``"exit"``.
        name:
            Human-readable name (default: zone_id).
        color:
            Hex colour for UI rendering.
        """
        _COLOR_BY_TYPE = {
            "restricted": "#ef4444",
            "entry": "#22c55e",
            "exit": "#f97316",
            "normal": "#64748b",
        }
        zone = ZoneDefinition(
            zone_id=zone_id,
            name=name or zone_id,
            polygon=list(polygon),
            zone_type=zone_type,
            color=color or _COLOR_BY_TYPE.get(zone_type, "#64748b"),
        )
        with self._lock:
            self._zones[zone_id] = zone
        logger.info(
            "ZoneAnalyzer: registered zone %r (%s, %d vertices)",
            zone_id,
            zone_type,
            len(polygon),
        )
        return zone

    def remove_zone(self, zone_id: str) -> bool:
        with self._lock:
            if zone_id in self._zones:
                del self._zones[zone_id]
                return True
        return False

    def get_zones(self) -> list[ZoneDefinition]:
        with self._lock:
            return list(self._zones.values())

    # ── Update (called each frame) ─────────────────────────────────────

    def update(
        self,
        points: list[Any],  # list[OccupancyPoint]
        timestamp: float | None = None,
    ) -> list[ZoneEvent]:
        """Detect zone events for the current set of occupancy points.

        Parameters
        ----------
        points:
            List of :class:`OccupancyPoint` objects (from SpatialMapper).
        timestamp:
            Frame timestamp (default: now).

        Returns
        -------
        list[ZoneEvent]
            New zone events (enter / exit / dwell) generated this frame.
        """
        ts = timestamp or time.time()
        new_events: list[ZoneEvent] = []

        with self._lock:
            zones = dict(self._zones)

        # Group points by person_id (use latest point per person)
        person_positions: dict[str, Any] = {}
        for pt in points:
            pid = getattr(pt, "person_id", str(id(pt)))
            person_positions[pid] = pt

        # Determine current zone membership per person
        current_zone_membership: dict[str, set[str]] = {}
        for person_id, pt in person_positions.items():
            x, y = getattr(pt, "world_x", 0.0), getattr(pt, "world_y", 0.0)
            in_zones: set[str] = set()
            for zid, zone in zones.items():
                if _point_in_polygon(x, y, zone.polygon):
                    in_zones.add(zid)
            current_zone_membership[person_id] = in_zones

        with self._lock:
            for person_id, current_zones in current_zone_membership.items():
                prev_zones = self._prev_zones.get(person_id, set())
                pt = person_positions[person_id]
                cam = getattr(pt, "camera_id", "cam")

                # Enter events
                for zid in current_zones - prev_zones:
                    zone = zones.get(zid)
                    if zone is None:
                        continue
                    self._person_zones[person_id][zid] = ts
                    ev = ZoneEvent(
                        event_id=str(uuid.uuid4()),
                        zone_id=zid,
                        zone_name=zone.name,
                        zone_type=zone.zone_type,
                        person_id=person_id,
                        camera_id=cam,
                        event_type="enter",
                        timestamp=ts,
                        dwell_seconds=0.0,
                    )
                    self._events.append(ev)
                    self._event_count += 1
                    new_events.append(ev)

                    # Flow: from previous zones → this zone
                    for prev_z in prev_zones:
                        self._flow_counts[(prev_z, zid)] += 1

                # Exit events
                for zid in prev_zones - current_zones:
                    zone = zones.get(zid)
                    enter_ts = self._person_zones[person_id].pop(zid, ts)
                    dwell = ts - enter_ts
                    if zone:
                        self._dwell_history[zid].append(dwell)
                        if len(self._dwell_history[zid]) > 1000:
                            self._dwell_history[zid] = self._dwell_history[zid][-1000:]
                    ev = ZoneEvent(
                        event_id=str(uuid.uuid4()),
                        zone_id=zid,
                        zone_name=zone.name if zone else zid,
                        zone_type=zone.zone_type if zone else "normal",
                        person_id=person_id,
                        camera_id=cam,
                        event_type="exit",
                        timestamp=ts,
                        dwell_seconds=dwell,
                    )
                    self._events.append(ev)
                    self._event_count += 1
                    new_events.append(ev)

                # Dwell alerts (restricted zones only)
                for zid in current_zones:
                    zone = zones.get(zid)
                    if zone and zone.zone_type == "restricted":
                        enter_ts = self._person_zones[person_id].get(zid, ts)
                        dwell = ts - enter_ts
                        if dwell >= self._dwell_threshold:
                            ev = ZoneEvent(
                                event_id=str(uuid.uuid4()),
                                zone_id=zid,
                                zone_name=zone.name,
                                zone_type=zone.zone_type,
                                person_id=person_id,
                                camera_id=cam,
                                event_type="dwell",
                                timestamp=ts,
                                dwell_seconds=dwell,
                            )
                            self._events.append(ev)
                            self._event_count += 1
                            new_events.append(ev)

                self._prev_zones[person_id] = current_zones

            self._update_count += 1

        return new_events

    # ── Analytics ──────────────────────────────────────────────────────

    def get_dwell_stats(self) -> dict[str, Any]:
        """Per-zone dwell time statistics."""
        with self._lock:
            zones = dict(self._zones)
            dwell_history = {k: list(v) for k, v in self._dwell_history.items()}
            current_persons = {
                zid: sum(1 for pid, zones_map in self._person_zones.items() if zid in zones_map)
                for zid in zones
            }

        result: dict[str, Any] = {}
        for zid, zone in zones.items():
            hist = dwell_history.get(zid, [])
            result[zid] = {
                "zone_name": zone.name,
                "zone_type": zone.zone_type,
                "current_occupancy": current_persons.get(zid, 0),
                "total_visits": len(hist),
                "avg_dwell_seconds": round(sum(hist) / max(1, len(hist)), 1),
                "max_dwell_seconds": round(max(hist, default=0.0), 1),
            }
        return result

    def get_flow_vectors(self) -> list[dict[str, Any]]:
        """Inter-zone movement counts (from_zone → to_zone)."""
        with self._lock:
            flow = dict(self._flow_counts)
            zones = dict(self._zones)
        result = []
        for (from_z, to_z), count in sorted(flow.items(), key=lambda x: -x[1]):
            result.append({
                "from_zone": from_z,
                "from_name": zones[from_z].name if from_z in zones else from_z,
                "to_zone": to_z,
                "to_name": zones[to_z].name if to_z in zones else to_z,
                "count": count,
            })
        return result

    def get_recent_events(
        self, limit: int = 50, event_type: str | None = None
    ) -> list[dict[str, Any]]:
        with self._lock:
            evs = list(self._events)
        if event_type:
            evs = [e for e in evs if e.event_type == event_type]
        evs.sort(key=lambda e: e.timestamp, reverse=True)
        return [e.to_dict() for e in evs[:limit]]

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "zones_registered": len(self._zones),
                "events_total": self._event_count,
                "events_buffered": len(self._events),
                "persons_tracked": len(self._person_zones),
                "update_count": self._update_count,
                "dwell_threshold_seconds": self._dwell_threshold,
                "uptime_seconds": round(time.time() - self._created_at, 1),
            }


# ---------------------------------------------------------------------------
# Ray-casting point-in-polygon
# ---------------------------------------------------------------------------


def _point_in_polygon(
    x: float, y: float, polygon: list[tuple[float, float]]
) -> bool:
    """Ray casting algorithm for point-in-polygon test.

    Parameters
    ----------
    x, y:
        Test point in world coordinates.
    polygon:
        List of ``(x, y)`` vertices (min 3).

    Returns
    -------
    bool
        True if ``(x, y)`` is inside the polygon.
    """
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside
