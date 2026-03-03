"""Long-Horizon Context Memory for session-level understanding.

Maintains compressed summaries of entity histories, zone statistics,
and session-level aggregates.  Designed to answer questions about what
has happened over the *entire* session — not just the last few minutes
kept in TemporalMemoryBuffer.

ContextMemory sits *above* the WorldModel:

    WorldModel.update()  →  WorldState
          ↓
    ContextMemory.update(world_state)
          ↓
    EntitySummary / SessionSummary / ZoneStats / query()

Key design choices:
    - Summaries are incrementally updated (O(1) per frame), not
      recomputed from scratch.
    - Old detailed events are pruned after ``event_retention_seconds``
      but compressed summaries persist for the full session.
    - ``query()`` uses keyword matching (no LLM) for fast, deterministic
      answers to common questions.  Returns Japanese by default.
"""

from __future__ import annotations

import logging
import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field

from sopilot.perception.types import (
    EntityEvent,
    EntityEventType,
    WorldState,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EntitySummary:
    """Compressed summary of an entity's history."""

    entity_id: int
    label: str
    first_seen: float  # timestamp
    last_seen: float  # timestamp
    total_frames: int  # frames where entity was present
    zones_visited: list[str]  # ordered list of zones visited
    zone_durations: dict[str, float]  # zone_id -> total seconds spent
    activities: list[str]  # ordered list of activity types observed
    violations: list[str]  # rules violated by this entity
    current_activity: str  # latest activity classification
    current_zone: str | None  # current zone or None
    total_distance: float  # total distance traveled (normalized coords)


@dataclass
class ZoneStats:
    """Statistics for a zone over the session."""

    zone_id: str
    total_entries: int
    total_exits: int
    unique_visitors: int
    current_occupancy: int
    max_occupancy: int
    average_dwell_time_seconds: float
    violations_in_zone: int


@dataclass
class SessionSummary:
    """High-level summary of what has happened in the session."""

    start_time: float
    current_time: float
    duration_seconds: float
    total_frames_processed: int
    unique_entities_seen: int
    current_entity_count: int
    total_violations: int
    violations_by_severity: dict[str, int]  # "critical": 3, "warning": 7
    violations_by_rule: dict[str, int]  # rule -> count
    zone_summary: dict[str, ZoneStats]  # zone_id -> stats
    entity_summaries: dict[int, EntitySummary]  # entity_id -> summary
    notable_events: list[str]  # human-readable list of important events


# ---------------------------------------------------------------------------
# Internal tracking state (not exposed)
# ---------------------------------------------------------------------------


@dataclass
class _EntityState:
    """Mutable internal state for incremental entity tracking."""

    entity_id: int
    label: str
    first_seen: float
    last_seen: float
    total_frames: int = 0
    zones_visited: list[str] = field(default_factory=list)
    zone_durations: dict[str, float] = field(default_factory=dict)
    # zone_id -> timestamp when entity entered that zone (still inside)
    zone_entry_times: dict[str, float] = field(default_factory=dict)
    current_zone_ids: set[str] = field(default_factory=set)
    activities: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    current_activity: str = "unknown"
    total_distance: float = 0.0
    prev_center: tuple[float, float] | None = None
    is_present: bool = True


@dataclass
class _ZoneState:
    """Mutable internal state for incremental zone tracking."""

    zone_id: str
    total_entries: int = 0
    total_exits: int = 0
    unique_visitors: set[int] = field(default_factory=set)
    current_occupancy: int = 0
    max_occupancy: int = 0
    # Completed dwell times for averaging
    completed_dwell_times: list[float] = field(default_factory=list)
    violations_in_zone: int = 0


# ---------------------------------------------------------------------------
# ContextMemory
# ---------------------------------------------------------------------------


class ContextMemory:
    """Long-horizon session memory with compressed entity and zone summaries.

    Incrementally updated on every WorldState.  Old detailed events are
    pruned after ``event_retention_seconds`` but entity/zone summaries
    persist for the full session.

    Thread safety: This class is *not* internally locked.  The caller
    (typically the perception engine) should ensure sequential calls to
    ``update()`` and can read summaries concurrently only after an
    update finishes.
    """

    def __init__(
        self,
        max_entity_summaries: int = 200,
        event_retention_seconds: float = 7200.0,
    ) -> None:
        """
        Args:
            max_entity_summaries: Max entities to track.  When exceeded,
                the least-recently-seen entity is evicted.
            event_retention_seconds: How long to keep detailed events
                (default 2 hours).
        """
        self._max_entity_summaries = max_entity_summaries
        self._event_retention_seconds = event_retention_seconds

        # Entity tracking — OrderedDict for LRU eviction
        self._entities: OrderedDict[int, _EntityState] = OrderedDict()

        # Zone tracking
        self._zones: dict[str, _ZoneState] = {}

        # Flat event log (pruned by retention window)
        self._events: list[EntityEvent] = []

        # Session-level counters
        self._start_time: float | None = None
        self._current_time: float = 0.0
        self._total_frames: int = 0
        self._total_violations: int = 0
        self._violations_by_severity: dict[str, int] = {}
        self._violations_by_rule: dict[str, int] = {}
        self._notable_events: list[str] = []
        self._current_entity_ids: set[int] = set()

    # -- main update -------------------------------------------------------

    def update(self, world_state: WorldState) -> None:
        """Ingest a new world state snapshot and update all summaries."""
        ts = world_state.timestamp
        if self._start_time is None:
            self._start_time = ts
        self._current_time = ts
        self._total_frames += 1

        # 1. Register entities from scene graph FIRST so that event
        #    processing (step 2) can reference them (e.g., appending
        #    violations to an entity's record).
        current_entity_ids: set[int] = set()
        for entity in world_state.scene_graph.entities:
            eid = entity.entity_id
            current_entity_ids.add(eid)

            if eid not in self._entities:
                self._entities[eid] = _EntityState(
                    entity_id=eid,
                    label=entity.label,
                    first_seen=ts,
                    last_seen=ts,
                    total_frames=1,
                    is_present=True,
                )
                self._enforce_entity_limit()
            else:
                state = self._entities[eid]
                state.last_seen = ts
                state.total_frames += 1
                state.is_present = True
                # Move to end for LRU
                self._entities.move_to_end(eid)

            # Update distance
            state = self._entities[eid]
            cx, cy = entity.bbox.center
            if state.prev_center is not None:
                dx = cx - state.prev_center[0]
                dy = cy - state.prev_center[1]
                state.total_distance += math.hypot(dx, dy)
            state.prev_center = (cx, cy)

            # Update zone presence from zone_occupancy
            new_zone_ids: set[str] = set()
            for zid, eids in world_state.zone_occupancy.items():
                if eid in eids:
                    new_zone_ids.add(zid)

            # Detect zone entry (newly in zone)
            for zid in new_zone_ids - state.current_zone_ids:
                if zid not in state.zone_entry_times:
                    state.zone_entry_times[zid] = ts
                if zid not in state.zones_visited:
                    state.zones_visited.append(zid)

            # Detect zone exit (was in zone, no longer)
            for zid in state.current_zone_ids - new_zone_ids:
                entry_ts = state.zone_entry_times.pop(zid, None)
                if entry_ts is not None:
                    duration = ts - entry_ts
                    state.zone_durations[zid] = (
                        state.zone_durations.get(zid, 0.0) + duration
                    )

            state.current_zone_ids = new_zone_ids
            if new_zone_ids:
                state.current_activity = "in_zone"

            # Update activity from attributes
            attrs = entity.attributes
            if attrs:
                activity = attrs.get("activity", attrs.get("action"))
                if activity:
                    state.current_activity = str(activity)
                    if not state.activities or state.activities[-1] != str(activity):
                        state.activities.append(str(activity))

        # Mark entities no longer present
        for eid, state in self._entities.items():
            if eid not in current_entity_ids and state.is_present:
                state.is_present = False
                # Close any open zone durations
                for zid in list(state.zone_entry_times.keys()):
                    entry_ts = state.zone_entry_times.pop(zid)
                    duration = ts - entry_ts
                    state.zone_durations[zid] = (
                        state.zone_durations.get(zid, 0.0) + duration
                    )
                state.current_zone_ids = set()

        self._current_entity_ids = current_entity_ids

        # 2. Process events from this frame (after entity registration
        #    so that violations can be appended to entity records).
        for event in world_state.events:
            self._events.append(event)
            self._process_event(event, ts)

        # 3. Update zone stats from zone_occupancy
        for zid, eids in world_state.zone_occupancy.items():
            if zid not in self._zones:
                self._zones[zid] = _ZoneState(zone_id=zid)
            zs = self._zones[zid]
            zs.current_occupancy = len(eids)
            if zs.current_occupancy > zs.max_occupancy:
                zs.max_occupancy = zs.current_occupancy
            for eid in eids:
                zs.unique_visitors.add(eid)

        # 4. Prune old events
        self._prune_events(ts)

    # -- queries -----------------------------------------------------------

    def get_entity_summary(self, entity_id: int) -> EntitySummary | None:
        """Get compressed history for a specific entity."""
        state = self._entities.get(entity_id)
        if state is None:
            return None
        return self._build_entity_summary(state)

    def get_session_summary(self) -> SessionSummary:
        """Get high-level session summary."""
        start = self._start_time if self._start_time is not None else 0.0
        current = self._current_time
        duration = current - start if start > 0 else 0.0

        entity_summaries = {
            eid: self._build_entity_summary(state)
            for eid, state in self._entities.items()
        }

        zone_summary = {
            zid: self._build_zone_stats(zs)
            for zid, zs in self._zones.items()
        }

        return SessionSummary(
            start_time=start,
            current_time=current,
            duration_seconds=duration,
            total_frames_processed=self._total_frames,
            unique_entities_seen=len(self._entities),
            current_entity_count=len(self._current_entity_ids),
            total_violations=self._total_violations,
            violations_by_severity=dict(self._violations_by_severity),
            violations_by_rule=dict(self._violations_by_rule),
            zone_summary=zone_summary,
            entity_summaries=entity_summaries,
            notable_events=list(self._notable_events),
        )

    def query(self, question: str) -> str:
        """Answer a natural-language question about the session.

        Uses keyword matching (no LLM) for fast, deterministic answers.
        Returns Japanese by default.

        Supports questions like:
        - "How many people have entered the restricted zone?"
        - "制限エリアに何人入った？"
        - "Has worker 5 taken a break?"
        - "What's the violation rate in the last hour?"
        """
        q = question.lower().strip()

        # --- Person/entity count ---
        if _matches_any(q, ["何人", "how many people", "how many person"]):
            if _matches_any(q, ["制限エリア", "restricted", "制限区域"]):
                return self._query_restricted_zone_visitors()
            # General person count
            persons = [
                s for s in self._entities.values()
                if "person" in s.label.lower()
            ]
            return f"セッション中に {len(persons)} 人を検出しました。現在 {self._count_present_persons()} 人がシーン内にいます。"

        # --- Violations ---
        if _matches_any(q, ["違反", "violation"]):
            # "rate" is more specific — check before "last N minutes" count
            if _matches_any(q, ["率", "rate"]):
                minutes = _extract_minutes(q)
                return self._query_violation_rate(minutes)
            if _matches_any(q, ["最後の", "last"]):
                minutes = _extract_minutes(q)
                if minutes is not None:
                    return self._query_violations_in_window(minutes)
            return self._query_violations_total()

        # --- Zone occupancy (specific zone query — checked before the
        #     broader "restricted zone" catch-all so that "zone restricted_area
        #     occupancy?" routes here) ---
        if _matches_any(q, ["ゾーン", "zone"]):
            if _matches_any(q, ["占有", "occupancy", "何人", "how many"]):
                zone_id = _extract_zone_id(q)
                if zone_id:
                    return self._query_zone_occupancy(zone_id)
            # If no specific occupancy query, list all zones
            if not _matches_any(q, ["制限エリア", "restricted", "制限区域"]):
                return self._query_all_zones()

        # --- Restricted zone ---
        if _matches_any(q, ["制限エリア", "restricted", "制限区域"]):
            return self._query_restricted_zone_visitors()

        # --- Anomaly queries ---
        if _matches_any(q, ["異常", "anomaly", "anomalies", "アノマリ"]):
            anomaly_events = [
                e for e in self._events
                if e.event_type == EntityEventType.ANOMALY
            ]
            if not anomaly_events:
                return "セッション中に自律検知された異常はありません。"
            # Group by detector
            by_detector: dict[str, int] = {}
            for e in anomaly_events:
                det = e.details.get("detector", "unknown")
                by_detector[det] = by_detector.get(det, 0) + 1
            parts = [f"{det}: {cnt}件" for det, cnt in sorted(by_detector.items())]
            latest = anomaly_events[-1]
            latest_desc = latest.details.get("description_ja", "詳細なし")
            return (
                f"自律検知された異常: 合計{len(anomaly_events)}件 "
                f"({', '.join(parts)})。"
                f"最新: {latest_desc}"
            )

        # --- Break / stationary check ---
        if _matches_any(q, ["休憩", "break"]):
            entity_id = _extract_entity_id(q)
            if entity_id is not None:
                return self._query_break(entity_id)
            return "エンティティIDを指定してください。例: 「worker 5 は休憩しましたか？」"

        # --- Zone / area listing (without "zone" keyword) ---
        if _matches_any(q, ["エリア", "area"]):
            return self._query_all_zones()

        # --- Entity-specific ---
        entity_id = _extract_entity_id(q)
        if entity_id is not None:
            summary = self.get_entity_summary(entity_id)
            if summary is None:
                return f"エンティティ {entity_id} は記録にありません。"
            return self._format_entity_summary_ja(summary)

        # --- Fallback ---
        session = self.get_session_summary()
        return (
            f"セッション概要: {session.duration_seconds:.0f}秒経過、"
            f"フレーム数 {session.total_frames_processed}、"
            f"検出エンティティ {session.unique_entities_seen}、"
            f"違反数 {session.total_violations}。"
            f"より具体的な質問をお願いします。"
        )

    def get_timeline(
        self,
        entity_id: int | None = None,
        zone_id: str | None = None,
        event_types: list[EntityEventType] | None = None,
        last_n_minutes: float | None = None,
    ) -> list[dict]:
        """Get a filtered timeline of events.

        All filters are combined with AND logic.  Pass None to skip a filter.

        Returns:
            List of dicts with keys: event_type, entity_id, timestamp,
            frame_number, details, confidence.
        """
        events = self._events

        if last_n_minutes is not None:
            cutoff = self._current_time - last_n_minutes * 60.0
            events = [e for e in events if e.timestamp >= cutoff]

        if entity_id is not None:
            events = [e for e in events if e.entity_id == entity_id]

        if zone_id is not None:
            events = [
                e for e in events
                if e.details.get("zone_id") == zone_id
            ]

        if event_types is not None:
            type_set = set(event_types)
            events = [e for e in events if e.event_type in type_set]

        return [
            {
                "event_type": e.event_type.value,
                "entity_id": e.entity_id,
                "timestamp": e.timestamp,
                "frame_number": e.frame_number,
                "details": dict(e.details),
                "confidence": e.confidence,
            }
            for e in events
        ]

    def reset(self) -> None:
        """Clear all memory."""
        self._entities.clear()
        self._zones.clear()
        self._events.clear()
        self._start_time = None
        self._current_time = 0.0
        self._total_frames = 0
        self._total_violations = 0
        self._violations_by_severity.clear()
        self._violations_by_rule.clear()
        self._notable_events.clear()
        self._current_entity_ids.clear()

    # -- internal helpers --------------------------------------------------

    def _process_event(self, event: EntityEvent, ts: float) -> None:
        """Process a single event and update counters/summaries."""
        etype = event.event_type
        eid = event.entity_id
        details = event.details

        if etype == EntityEventType.RULE_VIOLATION:
            self._total_violations += 1
            severity = details.get("severity", "warning")
            self._violations_by_severity[severity] = (
                self._violations_by_severity.get(severity, 0) + 1
            )
            rule = details.get("rule", "unknown")
            self._violations_by_rule[rule] = (
                self._violations_by_rule.get(rule, 0) + 1
            )
            # Track on entity
            if eid in self._entities:
                self._entities[eid].violations.append(rule)
            # Track on zone
            zone_id = details.get("zone_id")
            if zone_id:
                if zone_id not in self._zones:
                    self._zones[zone_id] = _ZoneState(zone_id=zone_id)
                self._zones[zone_id].violations_in_zone += 1

            # Notable event
            label = details.get("label", f"entity {eid}")
            self._notable_events.append(
                f"[{ts:.1f}] 違反: {rule} — {label} (severity={severity})"
            )

        elif etype == EntityEventType.ZONE_ENTERED:
            zone_id = details.get("zone_id")
            if zone_id:
                if zone_id not in self._zones:
                    self._zones[zone_id] = _ZoneState(zone_id=zone_id)
                self._zones[zone_id].total_entries += 1
                self._zones[zone_id].unique_visitors.add(eid)

        elif etype == EntityEventType.ZONE_EXITED:
            zone_id = details.get("zone_id")
            if zone_id:
                if zone_id not in self._zones:
                    self._zones[zone_id] = _ZoneState(zone_id=zone_id)
                self._zones[zone_id].total_exits += 1
                duration = details.get("duration_seconds", 0.0)
                if duration > 0:
                    self._zones[zone_id].completed_dwell_times.append(duration)

        elif etype == EntityEventType.ENTERED:
            label = details.get("label", "unknown")
            self._notable_events.append(
                f"[{ts:.1f}] 入場: {label} (ID={eid})"
            )

        elif etype == EntityEventType.EXITED:
            label = details.get("label", "unknown")
            self._notable_events.append(
                f"[{ts:.1f}] 退場: {label} (ID={eid})"
            )

    def _build_entity_summary(self, state: _EntityState) -> EntitySummary:
        """Convert internal mutable state to an immutable EntitySummary."""
        # Determine current zone
        current_zone: str | None = None
        if state.current_zone_ids:
            current_zone = next(iter(state.current_zone_ids))

        return EntitySummary(
            entity_id=state.entity_id,
            label=state.label,
            first_seen=state.first_seen,
            last_seen=state.last_seen,
            total_frames=state.total_frames,
            zones_visited=list(state.zones_visited),
            zone_durations=dict(state.zone_durations),
            activities=list(state.activities),
            violations=list(state.violations),
            current_activity=state.current_activity,
            current_zone=current_zone,
            total_distance=state.total_distance,
        )

    def _build_zone_stats(self, zs: _ZoneState) -> ZoneStats:
        """Convert internal mutable zone state to a ZoneStats."""
        avg_dwell = 0.0
        if zs.completed_dwell_times:
            avg_dwell = sum(zs.completed_dwell_times) / len(zs.completed_dwell_times)

        return ZoneStats(
            zone_id=zs.zone_id,
            total_entries=zs.total_entries,
            total_exits=zs.total_exits,
            unique_visitors=len(zs.unique_visitors),
            current_occupancy=zs.current_occupancy,
            max_occupancy=zs.max_occupancy,
            average_dwell_time_seconds=avg_dwell,
            violations_in_zone=zs.violations_in_zone,
        )

    def _enforce_entity_limit(self) -> None:
        """Evict the oldest entity if we exceed max_entity_summaries."""
        while len(self._entities) > self._max_entity_summaries:
            # Pop from front (least recently used)
            self._entities.popitem(last=False)

    def _prune_events(self, current_ts: float) -> None:
        """Remove events older than the retention window."""
        cutoff = current_ts - self._event_retention_seconds
        # Events are in chronological order — find the first non-stale index
        prune_count = 0
        for event in self._events:
            if event.timestamp >= cutoff:
                break
            prune_count += 1
        if prune_count > 0:
            del self._events[:prune_count]

    # -- query helpers (Japanese responses) --------------------------------

    def _count_present_persons(self) -> int:
        """Count persons currently present in the scene."""
        count = 0
        for eid in self._current_entity_ids:
            state = self._entities.get(eid)
            if state and "person" in state.label.lower():
                count += 1
        return count

    def _query_restricted_zone_visitors(self) -> str:
        """Count unique visitors to restricted zones."""
        restricted_visitors: set[int] = set()
        for zid, zs in self._zones.items():
            if "restricted" in zid.lower() or "制限" in zid.lower():
                restricted_visitors.update(zs.unique_visitors)
        if not restricted_visitors:
            return "制限エリアへの訪問者はいません。"
        return f"制限エリアに合計 {len(restricted_visitors)} 人が入りました。"

    def _query_violations_total(self) -> str:
        """Return total violation count and breakdown."""
        if self._total_violations == 0:
            return "違反は記録されていません。"
        parts = [f"合計 {self._total_violations} 件の違反が記録されています。"]
        if self._violations_by_severity:
            severity_str = ", ".join(
                f"{k}: {v}件" for k, v in self._violations_by_severity.items()
            )
            parts.append(f"重要度別: {severity_str}。")
        return " ".join(parts)

    def _query_violations_in_window(self, minutes: float) -> str:
        """Count violations in the last N minutes."""
        cutoff = self._current_time - minutes * 60.0
        count = sum(
            1
            for e in self._events
            if e.event_type == EntityEventType.RULE_VIOLATION
            and e.timestamp >= cutoff
        )
        return f"最後の {minutes:.0f} 分間に {count} 件の違反がありました。"

    def _query_violation_rate(self, minutes: float | None) -> str:
        """Compute violation rate per minute."""
        if minutes is not None:
            cutoff = self._current_time - minutes * 60.0
            violations = sum(
                1
                for e in self._events
                if e.event_type == EntityEventType.RULE_VIOLATION
                and e.timestamp >= cutoff
            )
            if minutes > 0:
                rate = violations / minutes
            else:
                rate = 0.0
            return f"最後の {minutes:.0f} 分間の違反率: {rate:.2f} 件/分。"
        else:
            start = self._start_time if self._start_time is not None else 0.0
            duration_min = (self._current_time - start) / 60.0
            if duration_min > 0:
                rate = self._total_violations / duration_min
            else:
                rate = 0.0
            return f"セッション全体の違反率: {rate:.2f} 件/分。"

    def _query_break(self, entity_id: int) -> str:
        """Check if an entity appears to have taken a break (stationary)."""
        state = self._entities.get(entity_id)
        if state is None:
            return f"エンティティ {entity_id} は記録にありません。"

        # Heuristic: if total_distance is very small relative to session
        # duration, the entity has been mostly stationary.
        duration = state.last_seen - state.first_seen
        if duration <= 0:
            return f"エンティティ {entity_id} のデータが不十分です。"

        # Also check if entity was absent for a stretch (gap between frames)
        avg_speed = state.total_distance / duration if duration > 0 else 0.0
        stationary_threshold = 0.001  # normalized units per second

        if avg_speed < stationary_threshold:
            return f"エンティティ {entity_id} ({state.label}) は休憩中と推定されます（ほぼ静止状態）。"
        return f"エンティティ {entity_id} ({state.label}) は活動中です（平均移動速度: {avg_speed:.4f}/秒）。"

    def _query_zone_occupancy(self, zone_id: str) -> str:
        """Report occupancy for a specific zone."""
        zs = self._zones.get(zone_id)
        if zs is None:
            return f"ゾーン '{zone_id}' の記録はありません。"
        stats = self._build_zone_stats(zs)
        return (
            f"ゾーン '{zone_id}': "
            f"現在 {stats.current_occupancy} 人、"
            f"最大 {stats.max_occupancy} 人、"
            f"入場回数 {stats.total_entries}、"
            f"退場回数 {stats.total_exits}、"
            f"訪問者数 {stats.unique_visitors} 人。"
        )

    def _query_all_zones(self) -> str:
        """Summarize all zones."""
        if not self._zones:
            return "ゾーン情報はありません。"
        parts: list[str] = []
        for zid, zs in self._zones.items():
            stats = self._build_zone_stats(zs)
            parts.append(
                f"  {zid}: 現在{stats.current_occupancy}人, "
                f"最大{stats.max_occupancy}人, "
                f"違反{stats.violations_in_zone}件"
            )
        return "ゾーン一覧:\n" + "\n".join(parts)

    def _format_entity_summary_ja(self, summary: EntitySummary) -> str:
        """Format an entity summary as Japanese text."""
        duration = summary.last_seen - summary.first_seen
        parts = [
            f"エンティティ {summary.entity_id} ({summary.label}):",
            f"  検出期間: {duration:.1f}秒 ({summary.total_frames}フレーム)",
            f"  移動距離: {summary.total_distance:.4f} (正規化座標)",
            f"  現在のアクティビティ: {summary.current_activity}",
        ]
        if summary.zones_visited:
            parts.append(f"  訪問ゾーン: {', '.join(summary.zones_visited)}")
        if summary.violations:
            parts.append(f"  違反: {', '.join(summary.violations)}")
        if summary.current_zone:
            parts.append(f"  現在のゾーン: {summary.current_zone}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Keyword-matching helpers for query()
# ---------------------------------------------------------------------------


def _matches_any(text: str, keywords: list[str]) -> bool:
    """True if *text* contains any of the *keywords* (case-insensitive)."""
    return any(kw.lower() in text for kw in keywords)


def _extract_minutes(text: str) -> float | None:
    """Extract a number of minutes from text like 'last 30 minutes' or '最後の30分'."""
    # English: "last 30 minutes"
    m = re.search(r"last\s+(\d+(?:\.\d+)?)\s*(?:minute|min|分)", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # Japanese: "最後の30分"
    m = re.search(r"(\d+(?:\.\d+)?)\s*分", text)
    if m:
        return float(m.group(1))
    # "last hour" / "1 hour"
    m = re.search(r"(?:last\s+)?(\d+)?\s*hour", text, re.IGNORECASE)
    if m:
        hours = float(m.group(1)) if m.group(1) else 1.0
        return hours * 60.0
    return None


def _extract_entity_id(text: str) -> int | None:
    """Extract an entity/worker ID from text."""
    # "worker 5", "entity 12", "エンティティ 3", "ID=7"
    m = re.search(r"(?:worker|entity|エンティティ|id\s*=?\s*)\s*(\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _extract_zone_id(text: str) -> str | None:
    """Extract a zone ID from text."""
    # "zone 'work_area'" or "zone work_area" or "ゾーンwork_area"
    m = re.search(r"(?:zone|ゾーン|エリア)\s*['\"]?(\w+)['\"]?", text, re.IGNORECASE)
    if m:
        return m.group(1)
    return None
