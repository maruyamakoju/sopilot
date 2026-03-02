"""Continuous world state management with temporal memory.

The WorldModel is the core of the perception engine's "human-like"
understanding.  It maintains continuous state about every entity in
the scene, remembers what happened over time, and detects when things
change.  It produces a WorldState on every frame that downstream
components (HybridReasoner, alerting, etc.) consume.

Flow per frame:
    SceneGraph  -->  WorldModel.update()  -->  WorldState
                     1. entity registry (enter/exit events)
                     2. zone monitor     (zone enter/exit events)
                     3. state changes    (attribute changes)
                     4. prolonged presence
                     5. anomaly baseline
                     6. assemble WorldState
                     7. push to temporal memory

Thread safety:
    All public methods acquire a reentrant lock so the world model
    can safely be updated from a background video-processing thread
    while the main thread reads state.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from sopilot.perception.types import (
    BBox,
    EntityEvent,
    EntityEventType,
    PerceptionConfig,
    SceneEntity,
    SceneGraph,
    Track,
    TrackState,
    WorldState,
    Zone,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EntityHistory — full lifecycle record of a single entity
# ---------------------------------------------------------------------------


@dataclass
class EntityHistory:
    """Full history of an entity in the world.

    Created when an entity first appears in the scene graph and
    updated every frame it remains visible.  Retained even after the
    entity exits so that callers can query past presence.
    """

    entity_id: int
    label: str
    first_seen: float  # timestamp of first observation
    last_seen: float  # timestamp of most recent observation
    total_frames: int  # total frames this entity was visible
    states: list[tuple[float, TrackState]]  # (timestamp, state) history
    zones_visited: set[str]  # every zone_id the entity has been in
    current_zone_ids: list[str]  # zone_ids the entity is currently in
    attributes_history: list[tuple[float, dict[str, Any]]]  # (timestamp, attrs)
    bbox_trajectory: list[tuple[float, BBox]]  # (timestamp, bbox) for motion analysis


# ---------------------------------------------------------------------------
# EntityRegistry — maintains the full registry of entities
# ---------------------------------------------------------------------------


class EntityRegistry:
    """Maintains the full history and current state of every entity ever seen.

    When a new entity appears that was not previously known, an ENTERED
    event is generated.  When a known entity disappears from the scene
    graph, an EXITED event is generated.
    """

    def __init__(self) -> None:
        self._histories: dict[int, EntityHistory] = {}
        # Set of entity IDs present in the *previous* frame.
        self._previous_ids: set[int] = set()

    # -- public API --------------------------------------------------------

    def update(
        self,
        scene_graph: SceneGraph,
    ) -> list[EntityEvent]:
        """Process a new scene graph and return enter/exit events."""
        events: list[EntityEvent] = []
        current_ids: set[int] = set()

        for entity in scene_graph.entities:
            eid = entity.entity_id
            current_ids.add(eid)

            if eid not in self._histories:
                # Brand-new entity
                self._histories[eid] = EntityHistory(
                    entity_id=eid,
                    label=entity.label,
                    first_seen=scene_graph.timestamp,
                    last_seen=scene_graph.timestamp,
                    total_frames=1,
                    states=[(scene_graph.timestamp, TrackState.ACTIVE)],
                    zones_visited=set(entity.zone_ids),
                    current_zone_ids=list(entity.zone_ids),
                    attributes_history=(
                        [(scene_graph.timestamp, dict(entity.attributes))]
                        if entity.attributes
                        else []
                    ),
                    bbox_trajectory=[(scene_graph.timestamp, entity.bbox)],
                )
                events.append(
                    EntityEvent(
                        event_type=EntityEventType.ENTERED,
                        entity_id=eid,
                        timestamp=scene_graph.timestamp,
                        frame_number=scene_graph.frame_number,
                        details={"label": entity.label},
                        confidence=entity.confidence,
                    )
                )
                logger.info(
                    "Entity %d (%s) ENTERED at t=%.3f frame=%d",
                    eid,
                    entity.label,
                    scene_graph.timestamp,
                    scene_graph.frame_number,
                )
            else:
                # Update existing history
                hist = self._histories[eid]
                hist.last_seen = scene_graph.timestamp
                hist.total_frames += 1
                hist.current_zone_ids = list(entity.zone_ids)
                hist.zones_visited.update(entity.zone_ids)
                hist.bbox_trajectory.append(
                    (scene_graph.timestamp, entity.bbox)
                )
                if entity.attributes:
                    hist.attributes_history.append(
                        (scene_graph.timestamp, dict(entity.attributes))
                    )

        # Entities that were present last frame but are gone now
        exited_ids = self._previous_ids - current_ids
        for eid in exited_ids:
            hist = self._histories.get(eid)
            label = hist.label if hist else "unknown"
            if hist:
                hist.states.append(
                    (scene_graph.timestamp, TrackState.EXITED)
                )
                hist.current_zone_ids = []
            events.append(
                EntityEvent(
                    event_type=EntityEventType.EXITED,
                    entity_id=eid,
                    timestamp=scene_graph.timestamp,
                    frame_number=scene_graph.frame_number,
                    details={"label": label},
                )
            )
            logger.info(
                "Entity %d (%s) EXITED at t=%.3f frame=%d",
                eid,
                label,
                scene_graph.timestamp,
                scene_graph.frame_number,
            )

        self._previous_ids = current_ids
        return events

    def get(self, entity_id: int) -> EntityHistory | None:
        return self._histories.get(entity_id)

    def all_histories(self) -> dict[int, EntityHistory]:
        return dict(self._histories)

    def reset(self) -> None:
        self._histories.clear()
        self._previous_ids.clear()


# ---------------------------------------------------------------------------
# ZoneMonitor — tracks zone entry/exit for every entity
# ---------------------------------------------------------------------------


class ZoneMonitor:
    """Tracks zone occupancy and generates entry/exit events.

    Also monitors for prolonged presence in restricted zones and
    generates PROLONGED_PRESENCE events when a configurable threshold
    is exceeded.
    """

    def __init__(
        self,
        zones: list[Zone],
        prolonged_presence_seconds: float = 60.0,
    ) -> None:
        self._zones = {z.zone_id: z for z in zones}
        self._prolonged_seconds = prolonged_presence_seconds
        # entity_id -> set of zone_ids the entity was in last frame
        self._previous_zone_map: dict[int, set[str]] = {}
        # (entity_id, zone_id) -> timestamp when entity first entered that zone
        self._zone_entry_times: dict[tuple[int, str], float] = {}
        # (entity_id, zone_id) set for which we already emitted PROLONGED_PRESENCE
        self._prolonged_alerted: set[tuple[int, str]] = set()

    # -- public API --------------------------------------------------------

    def update(
        self,
        scene_graph: SceneGraph,
    ) -> tuple[dict[str, list[int]], list[EntityEvent]]:
        """Process a scene graph and return (zone_occupancy, events).

        Returns:
            zone_occupancy: zone_id -> list of entity_ids currently inside
            events: ZONE_ENTERED / ZONE_EXITED / PROLONGED_PRESENCE events
        """
        events: list[EntityEvent] = []
        zone_occupancy: dict[str, list[int]] = {zid: [] for zid in self._zones}

        current_zone_map: dict[int, set[str]] = {}

        for entity in scene_graph.entities:
            eid = entity.entity_id
            current_zones: set[str] = set()

            # Use entity's zone_ids if already assigned by scene graph builder
            if entity.zone_ids:
                for zid in entity.zone_ids:
                    if zid in self._zones:
                        current_zones.add(zid)
            else:
                # Fallback: compute from zone polygons
                for zid, zone in self._zones.items():
                    if zone.contains_bbox(entity.bbox):
                        current_zones.add(zid)

            current_zone_map[eid] = current_zones

            # Populate occupancy
            for zid in current_zones:
                zone_occupancy[zid].append(eid)

            # Compare with previous frame
            prev_zones = self._previous_zone_map.get(eid, set())

            entered_zones = current_zones - prev_zones
            exited_zones = prev_zones - current_zones

            for zid in entered_zones:
                zone = self._zones.get(zid)
                zone_name = zone.name if zone else zid
                events.append(
                    EntityEvent(
                        event_type=EntityEventType.ZONE_ENTERED,
                        entity_id=eid,
                        timestamp=scene_graph.timestamp,
                        frame_number=scene_graph.frame_number,
                        details={
                            "zone_id": zid,
                            "zone_name": zone_name,
                            "label": entity.label,
                        },
                    )
                )
                self._zone_entry_times[(eid, zid)] = scene_graph.timestamp
                logger.info(
                    "Entity %d entered zone '%s' at t=%.3f",
                    eid,
                    zone_name,
                    scene_graph.timestamp,
                )

            for zid in exited_zones:
                zone = self._zones.get(zid)
                zone_name = zone.name if zone else zid
                entry_ts = self._zone_entry_times.pop((eid, zid), None)
                duration = (
                    scene_graph.timestamp - entry_ts if entry_ts is not None else 0.0
                )
                events.append(
                    EntityEvent(
                        event_type=EntityEventType.ZONE_EXITED,
                        entity_id=eid,
                        timestamp=scene_graph.timestamp,
                        frame_number=scene_graph.frame_number,
                        details={
                            "zone_id": zid,
                            "zone_name": zone_name,
                            "label": entity.label,
                            "duration_seconds": round(duration, 2),
                        },
                    )
                )
                # Clean up prolonged-presence alert state
                self._prolonged_alerted.discard((eid, zid))
                logger.info(
                    "Entity %d exited zone '%s' at t=%.3f (duration=%.1fs)",
                    eid,
                    zone_name,
                    scene_graph.timestamp,
                    duration,
                )

        # Clean up previous_zone_map entries for entities that disappeared
        current_entity_ids = {e.entity_id for e in scene_graph.entities}
        departed = set(self._previous_zone_map.keys()) - current_entity_ids
        for eid in departed:
            prev_zones = self._previous_zone_map.pop(eid, set())
            for zid in prev_zones:
                self._zone_entry_times.pop((eid, zid), None)
                self._prolonged_alerted.discard((eid, zid))

        self._previous_zone_map = current_zone_map
        return zone_occupancy, events

    def check_prolonged_presence(
        self,
        scene_graph: SceneGraph,
    ) -> list[EntityEvent]:
        """Check if any entity has been in a restricted zone too long.

        Only generates an event once per (entity, zone) pair until the
        entity leaves the zone and re-enters.
        """
        events: list[EntityEvent] = []

        for (eid, zid), entry_ts in list(self._zone_entry_times.items()):
            if (eid, zid) in self._prolonged_alerted:
                continue

            zone = self._zones.get(zid)
            if zone is None:
                continue

            # Only check restricted/hazard zones
            if zone.zone_type not in ("restricted", "hazard"):
                continue

            elapsed = scene_graph.timestamp - entry_ts
            if elapsed >= self._prolonged_seconds:
                events.append(
                    EntityEvent(
                        event_type=EntityEventType.PROLONGED_PRESENCE,
                        entity_id=eid,
                        timestamp=scene_graph.timestamp,
                        frame_number=scene_graph.frame_number,
                        details={
                            "zone_id": zid,
                            "zone_name": zone.name,
                            "zone_type": zone.zone_type,
                            "duration_seconds": round(elapsed, 2),
                            "threshold_seconds": self._prolonged_seconds,
                        },
                    )
                )
                self._prolonged_alerted.add((eid, zid))
                logger.info(
                    "PROLONGED PRESENCE: entity %d in zone '%s' for %.1fs "
                    "(threshold=%.1fs)",
                    eid,
                    zone.name,
                    elapsed,
                    self._prolonged_seconds,
                )

        return events

    def reset(self) -> None:
        self._previous_zone_map.clear()
        self._zone_entry_times.clear()
        self._prolonged_alerted.clear()


# ---------------------------------------------------------------------------
# StateChangeDetector — detects attribute changes on entities
# ---------------------------------------------------------------------------


class StateChangeDetector:
    """Detects when entity attributes change between frames.

    Focuses on safety-relevant attribute changes (e.g., helmet removed,
    gloves on/off) and generates STATE_CHANGED events with details about
    what changed.
    """

    # Attributes considered safety-relevant.  Changes to these get
    # higher-confidence events and INFO-level logging.
    SAFETY_ATTRIBUTES: frozenset[str] = frozenset(
        {
            "has_helmet",
            "has_hardhat",
            "has_safety_vest",
            "has_gloves",
            "has_goggles",
            "has_mask",
            "has_harness",
            "is_sitting",
            "is_standing",
            "is_running",
            "is_lying_down",
            "is_carrying_load",
        }
    )

    def __init__(self) -> None:
        # entity_id -> attribute dict from previous frame
        self._previous_attrs: dict[int, dict[str, Any]] = {}

    def update(
        self,
        scene_graph: SceneGraph,
    ) -> list[EntityEvent]:
        """Compare current attributes with previous frame and emit events."""
        events: list[EntityEvent] = []
        current_attrs: dict[int, dict[str, Any]] = {}

        for entity in scene_graph.entities:
            eid = entity.entity_id
            attrs = dict(entity.attributes) if entity.attributes else {}
            current_attrs[eid] = attrs

            prev = self._previous_attrs.get(eid)
            if prev is None:
                # First time seeing this entity — no comparison possible
                continue

            # Find changed keys
            all_keys = set(prev.keys()) | set(attrs.keys())
            for key in all_keys:
                old_val = prev.get(key)
                new_val = attrs.get(key)

                if old_val == new_val:
                    continue

                is_safety = key in self.SAFETY_ATTRIBUTES
                confidence = 0.9 if is_safety else 0.7

                events.append(
                    EntityEvent(
                        event_type=EntityEventType.STATE_CHANGED,
                        entity_id=eid,
                        timestamp=scene_graph.timestamp,
                        frame_number=scene_graph.frame_number,
                        details={
                            "attribute": key,
                            "old_value": old_val,
                            "new_value": new_val,
                            "label": entity.label,
                            "safety_relevant": is_safety,
                        },
                        confidence=confidence,
                    )
                )

                if is_safety:
                    logger.info(
                        "STATE CHANGE (safety): entity %d (%s) "
                        "%s: %s -> %s at t=%.3f",
                        eid,
                        entity.label,
                        key,
                        old_val,
                        new_val,
                        scene_graph.timestamp,
                    )
                else:
                    logger.debug(
                        "STATE CHANGE: entity %d (%s) %s: %s -> %s",
                        eid,
                        entity.label,
                        key,
                        old_val,
                        new_val,
                    )

        # Clean up departed entities from the previous-attrs map to
        # avoid unbounded growth.
        current_ids = {e.entity_id for e in scene_graph.entities}
        departed = set(self._previous_attrs.keys()) - current_ids
        for eid in departed:
            del self._previous_attrs[eid]

        self._previous_attrs = current_attrs
        return events

    def reset(self) -> None:
        self._previous_attrs.clear()


# ---------------------------------------------------------------------------
# TemporalMemoryBuffer — rolling buffer of recent WorldStates
# ---------------------------------------------------------------------------


class TemporalMemoryBuffer:
    """Rolling buffer of recent WorldStates for temporal reasoning.

    Stores up to ``max_seconds`` worth of world states and supports
    efficient queries by time window and entity ID.  Memory-bounded:
    stores only WorldState metadata, not raw frames.
    """

    def __init__(self, max_seconds: float = 300.0) -> None:
        self._max_seconds = max_seconds
        self._buffer: deque[WorldState] = deque()
        # entity_id -> list of (timestamp, bbox) for fast trajectory lookup
        self._entity_bboxes: dict[int, deque[tuple[float, BBox]]] = {}

    # -- public API --------------------------------------------------------

    def push(self, state: WorldState) -> None:
        """Add a new WorldState, evicting stale entries."""
        self._buffer.append(state)

        # Index entity bboxes for trajectory queries
        for entity in state.scene_graph.entities:
            eid = entity.entity_id
            if eid not in self._entity_bboxes:
                self._entity_bboxes[eid] = deque()
            self._entity_bboxes[eid].append(
                (state.timestamp, entity.bbox)
            )

        self._evict(state.timestamp)

    def get_states_in_window(self, seconds: float) -> list[WorldState]:
        """Return all WorldStates within the last *seconds*."""
        if not self._buffer:
            return []
        cutoff = self._buffer[-1].timestamp - seconds
        return [s for s in self._buffer if s.timestamp >= cutoff]

    def get_entity_trajectory(
        self, entity_id: int
    ) -> list[tuple[float, BBox]]:
        """Return the bbox trajectory of an entity within the buffer window."""
        bboxes = self._entity_bboxes.get(entity_id)
        if bboxes is None:
            return []
        return list(bboxes)

    def get_avg_entity_count(self, seconds: float = 60.0) -> float:
        """Average entity count over the last *seconds*."""
        states = self.get_states_in_window(seconds)
        if not states:
            return 0.0
        return sum(s.entity_count for s in states) / len(states)

    def clear(self) -> None:
        self._buffer.clear()
        self._entity_bboxes.clear()

    # -- internal ----------------------------------------------------------

    def _evict(self, current_timestamp: float) -> None:
        """Drop entries older than max_seconds."""
        cutoff = current_timestamp - self._max_seconds

        while self._buffer and self._buffer[0].timestamp < cutoff:
            self._buffer.popleft()

        # Also trim per-entity bbox deques
        stale_eids: list[int] = []
        for eid, dq in self._entity_bboxes.items():
            while dq and dq[0][0] < cutoff:
                dq.popleft()
            if not dq:
                stale_eids.append(eid)
        for eid in stale_eids:
            del self._entity_bboxes[eid]

    @property
    def size(self) -> int:
        """Number of WorldStates currently buffered."""
        return len(self._buffer)


# ---------------------------------------------------------------------------
# AnomalyBaseline — learns "normal" and flags deviations
# ---------------------------------------------------------------------------


class AnomalyBaseline:
    """Learns what "normal" looks like and flags deviations.

    Uses exponential moving averages (EMA) for adaptability.  Generates
    ANOMALY events when a metric deviates more than ``sigma_threshold``
    standard deviations from its running mean.

    Metrics tracked:
        - entity_count: total entities per frame
        - zone occupancy: per-zone entity count
    """

    _SIGMA_THRESHOLD: float = 2.0
    _ALPHA: float = 0.05  # EMA smoothing factor (lower = slower adaptation)
    _MIN_OBSERVATIONS: int = 30  # minimum frames before we start checking

    def __init__(self) -> None:
        # entity_count statistics
        self._count_mean: float = 0.0
        self._count_var: float = 0.0  # running variance (EMA)
        self._observations: int = 0

        # Per-zone occupancy statistics: zone_id -> (mean, var)
        self._zone_mean: dict[str, float] = {}
        self._zone_var: dict[str, float] = {}

    # -- public API --------------------------------------------------------

    def observe(self, world_state: WorldState) -> None:
        """Update running statistics with a new observation."""
        n = world_state.entity_count
        self._observations += 1

        if self._observations == 1:
            self._count_mean = float(n)
            self._count_var = 0.0
        else:
            alpha = self._ALPHA
            delta = n - self._count_mean
            self._count_mean += alpha * delta
            # EMA of variance: Var_new = (1-alpha) * (Var_old + alpha * delta^2)
            self._count_var = (1 - alpha) * (self._count_var + alpha * delta * delta)

        # Per-zone occupancy
        for zid, eids in world_state.zone_occupancy.items():
            occ = float(len(eids))
            if zid not in self._zone_mean:
                self._zone_mean[zid] = occ
                self._zone_var[zid] = 0.0
            else:
                alpha = self._ALPHA
                delta_z = occ - self._zone_mean[zid]
                self._zone_mean[zid] += alpha * delta_z
                self._zone_var[zid] = (1 - alpha) * (
                    self._zone_var[zid] + alpha * delta_z * delta_z
                )

    def check_anomalies(
        self, world_state: WorldState
    ) -> list[EntityEvent]:
        """Check whether the current world state is anomalous."""
        if self._observations < self._MIN_OBSERVATIONS:
            return []

        events: list[EntityEvent] = []

        # Check entity count
        count_std = math.sqrt(max(self._count_var, 1e-8))
        z_score = (
            abs(world_state.entity_count - self._count_mean) / count_std
        )
        if z_score > self._SIGMA_THRESHOLD:
            events.append(
                EntityEvent(
                    event_type=EntityEventType.ANOMALY,
                    entity_id=-1,  # scene-level anomaly
                    timestamp=world_state.timestamp,
                    frame_number=world_state.frame_number,
                    details={
                        "metric": "entity_count",
                        "current": world_state.entity_count,
                        "expected_mean": round(self._count_mean, 2),
                        "expected_std": round(count_std, 2),
                        "z_score": round(z_score, 2),
                    },
                    confidence=min(1.0, z_score / 5.0),
                )
            )
            logger.info(
                "ANOMALY: entity_count=%d (mean=%.1f, std=%.1f, z=%.1f)",
                world_state.entity_count,
                self._count_mean,
                count_std,
                z_score,
            )

        # Check per-zone occupancy
        for zid, eids in world_state.zone_occupancy.items():
            if zid not in self._zone_mean:
                continue
            zone_std = math.sqrt(max(self._zone_var.get(zid, 0.0), 1e-8))
            occ = float(len(eids))
            z_zone = abs(occ - self._zone_mean[zid]) / zone_std
            if z_zone > self._SIGMA_THRESHOLD:
                events.append(
                    EntityEvent(
                        event_type=EntityEventType.ANOMALY,
                        entity_id=-1,
                        timestamp=world_state.timestamp,
                        frame_number=world_state.frame_number,
                        details={
                            "metric": "zone_occupancy",
                            "zone_id": zid,
                            "current": len(eids),
                            "expected_mean": round(self._zone_mean[zid], 2),
                            "expected_std": round(zone_std, 2),
                            "z_score": round(z_zone, 2),
                        },
                        confidence=min(1.0, z_zone / 5.0),
                    )
                )
                logger.info(
                    "ANOMALY: zone '%s' occupancy=%d (mean=%.1f, std=%.1f, z=%.1f)",
                    zid,
                    len(eids),
                    self._zone_mean[zid],
                    zone_std,
                    z_zone,
                )

        return events

    def reset(self) -> None:
        self._count_mean = 0.0
        self._count_var = 0.0
        self._observations = 0
        self._zone_mean.clear()
        self._zone_var.clear()


# ---------------------------------------------------------------------------
# WorldModel — the main orchestrator
# ---------------------------------------------------------------------------


class WorldModel:
    """Continuous world state management with temporal memory.

    Consumes SceneGraph snapshots produced by the scene-graph builder
    and produces WorldState objects enriched with:
        - entity lifecycle events (enter/exit)
        - zone occupancy events (zone enter/exit/prolonged presence)
        - attribute state-change events (safety-relevant changes)
        - statistical anomaly events (entity count, zone occupancy)
        - temporal memory for trajectory and trend queries

    All public methods are thread-safe.

    Usage::

        model = WorldModel(config)
        for scene_graph in scene_graphs:
            world_state = model.update(scene_graph)
            for event in world_state.events:
                handle(event)
    """

    def __init__(self, config: PerceptionConfig) -> None:
        self._config = config
        self._lock = threading.RLock()

        # Sub-components
        self._entity_registry = EntityRegistry()
        self._zone_monitor = ZoneMonitor(
            zones=config.zone_definitions,
            prolonged_presence_seconds=config.prolonged_presence_seconds,
        )
        self._state_detector = StateChangeDetector()
        self._temporal_memory = TemporalMemoryBuffer(
            max_seconds=config.temporal_memory_seconds,
        )
        self._anomaly_baseline = AnomalyBaseline()

        # Frame counter for diagnostics
        self._frames_processed: int = 0
        self._last_timestamp: float = 0.0

        logger.info(
            "WorldModel initialized (temporal_memory=%.0fs, zones=%d, "
            "prolonged_presence=%.0fs)",
            config.temporal_memory_seconds,
            len(config.zone_definitions),
            config.prolonged_presence_seconds,
        )

    # -- main update -------------------------------------------------------

    def update(self, scene_graph: SceneGraph) -> WorldState:
        """Process a new scene graph and produce the current WorldState.

        This is the main entry point called once per frame (or once per
        processed frame if skip_frames > 0).  It runs the full
        perception pipeline:

        1. Update entity registry (detect new/departed entities)
        2. Update zone monitor (detect zone entry/exit)
        3. Detect state changes (attribute changes)
        4. Check prolonged presence in restricted zones
        5. Update anomaly baseline + check anomalies
        6. Build WorldState with all events
        7. Push to temporal memory buffer
        8. Return WorldState

        Args:
            scene_graph: The current frame's scene graph snapshot.

        Returns:
            A WorldState containing all events generated this frame.
        """
        with self._lock:
            all_events: list[EntityEvent] = []

            # 1. Entity registry — enter/exit events
            registry_events = self._entity_registry.update(scene_graph)
            all_events.extend(registry_events)

            # 2. Zone monitor — zone enter/exit events
            zone_occupancy, zone_events = self._zone_monitor.update(scene_graph)
            all_events.extend(zone_events)

            # 3. State change detection — attribute changes
            state_events = self._state_detector.update(scene_graph)
            all_events.extend(state_events)

            # 4. Prolonged presence check
            prolonged_events = self._zone_monitor.check_prolonged_presence(
                scene_graph
            )
            all_events.extend(prolonged_events)

            # 5. Build active tracks dict from scene entities.
            # We synthesize Track objects from SceneEntity data since
            # the world model operates on scene graphs, not raw tracker
            # output.  Downstream code that needs full Track objects
            # should use the tracker directly.
            active_tracks: dict[int, Track] = {}
            for entity in scene_graph.entities:
                hist = self._entity_registry.get(entity.entity_id)
                active_tracks[entity.entity_id] = Track(
                    track_id=entity.entity_id,
                    label=entity.label,
                    state=TrackState.ACTIVE,
                    bbox=entity.bbox,
                    confidence=entity.confidence,
                    attributes=dict(entity.attributes) if entity.attributes else {},
                    hits=hist.total_frames if hist else 1,
                )

            # 6. Assemble WorldState
            world_state = WorldState(
                timestamp=scene_graph.timestamp,
                frame_number=scene_graph.frame_number,
                scene_graph=scene_graph,
                active_tracks=active_tracks,
                events=all_events,
                zone_occupancy=zone_occupancy,
                entity_count=scene_graph.entity_count,
                person_count=scene_graph.person_count,
            )

            # 5 (cont'd). Anomaly baseline — observe then check
            self._anomaly_baseline.observe(world_state)
            anomaly_events = self._anomaly_baseline.check_anomalies(world_state)
            all_events.extend(anomaly_events)
            # Re-assign events since we appended after construction
            world_state.events = all_events

            # 7. Push to temporal memory
            self._temporal_memory.push(world_state)

            # Update counters
            self._frames_processed += 1
            self._last_timestamp = scene_graph.timestamp

            if all_events:
                logger.debug(
                    "Frame %d: %d events generated (%s)",
                    scene_graph.frame_number,
                    len(all_events),
                    ", ".join(e.event_type.value for e in all_events),
                )

            return world_state

    # -- query methods -----------------------------------------------------

    def get_entity_history(self, entity_id: int) -> EntityHistory | None:
        """Return the full history of an entity, or None if unknown.

        Thread-safe.  The returned EntityHistory is a mutable object
        owned by the registry — callers should not modify it.
        """
        with self._lock:
            return self._entity_registry.get(entity_id)

    def get_zone_occupancy(self) -> dict[str, list[int]]:
        """Return the current zone occupancy map.

        Returns:
            A dict mapping zone_id to a list of entity_ids currently
            inside that zone.  Returns an empty dict if no frames have
            been processed yet.
        """
        with self._lock:
            if not self._temporal_memory.size:
                return {}
            # Return the occupancy from the most recent WorldState
            states = self._temporal_memory.get_states_in_window(0.001)
            if states:
                return dict(states[-1].zone_occupancy)
            return {}

    def get_recent_events(
        self, seconds: float = 60.0
    ) -> list[EntityEvent]:
        """Return all events from the last *seconds*.

        Thread-safe.  Returns events in chronological order.

        Args:
            seconds: Time window in seconds (default 60).

        Returns:
            List of EntityEvent objects, oldest first.
        """
        with self._lock:
            states = self._temporal_memory.get_states_in_window(seconds)
            events: list[EntityEvent] = []
            for s in states:
                events.extend(s.events)
            return events

    def get_entity_trajectory(
        self, entity_id: int
    ) -> list[tuple[float, BBox]]:
        """Return the bbox trajectory for an entity from temporal memory.

        Useful for motion analysis and speed estimation.
        """
        with self._lock:
            return self._temporal_memory.get_entity_trajectory(entity_id)

    def get_temporal_memory(self) -> TemporalMemoryBuffer:
        """Return a reference to the temporal memory buffer.

        Callers should acquire the world model's lock if performing
        multi-step reads, or accept eventual-consistency semantics.
        """
        return self._temporal_memory

    def get_anomaly_baseline(self) -> AnomalyBaseline:
        """Return a reference to the anomaly baseline component."""
        return self._anomaly_baseline

    @property
    def frames_processed(self) -> int:
        """Total number of frames processed since creation or last reset."""
        with self._lock:
            return self._frames_processed

    @property
    def last_timestamp(self) -> float:
        """Timestamp of the most recently processed frame."""
        with self._lock:
            return self._last_timestamp

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        """Reset all state, clearing history and temporal memory.

        Thread-safe.  After reset, the world model behaves as if
        freshly constructed.
        """
        with self._lock:
            self._entity_registry.reset()
            self._zone_monitor.reset()
            self._state_detector.reset()
            self._temporal_memory.clear()
            self._anomaly_baseline.reset()
            self._frames_processed = 0
            self._last_timestamp = 0.0
            logger.info("WorldModel reset — all state cleared")
