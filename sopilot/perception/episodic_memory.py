"""Episodic memory: segments the event stream into coherent episodes and
detects recurring temporal patterns across long time horizons.

An episode is a coherent "scene" — a contiguous period of activity that is
separated from neighbouring episodes by a quiet period (no events for
`episode_gap_seconds`).  Episodes enable cross-episode pattern matching and
long-horizon narrative construction.
"""

from __future__ import annotations

import math
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sopilot.perception.types import EntityEvent, EntityEventType, WorldState


# ── Public data classes ────────────────────────────────────────────────────


@dataclass
class Episode:
    """A coherent segment of the event stream."""

    id: str  # UUID
    start_time: float  # unix timestamp
    end_time: float | None  # None if still open
    start_frame: int
    end_frame: int
    entity_ids: list[int]  # unique entity IDs observed
    event_count: int
    event_type_counts: dict[str, int]  # EntityEventType.name → count
    summary_ja: str  # NL summary in Japanese
    summary_en: str  # NL summary in English
    severity: str  # "normal" | "notable" | "critical"
    tags: list[str]  # searchable keyword tags
    duration_seconds: float  # 0 if still open


@dataclass
class TemporalPattern:
    """A recurring pattern detected across episodes."""

    pattern_id: str
    description_ja: str
    description_en: str
    occurrences: int
    typical_hours: list[float]  # hours of day (0.0–23.99) when pattern occurs
    event_sequence: list[str]  # sequence of EntityEventType names
    confidence: float  # 0–1
    last_seen: float  # unix timestamp


# ── Internal mutable episode state ────────────────────────────────────────


@dataclass
class _OpenEpisode:
    """Mutable state for an episode that is still accumulating events."""

    id: str
    start_time: float
    start_frame: int
    last_event_time: float
    entity_ids: set[int]
    events: list[EntityEvent]
    # derived counters — updated incrementally
    anomaly_count: int = 0
    violation_count: int = 0
    zone_events: int = 0


# ── Main class ─────────────────────────────────────────────────────────────


class EpisodicMemoryStore:
    """Segments events into episodes and detects temporal patterns.

    Design:
    - An episode stays "open" as long as events keep arriving.
    - After `episode_gap_seconds` of silence, the episode is "closed" and a
      new one starts on the next event.
    - Episodes are stored in a bounded deque (max_episodes).
    - Temporal patterns are detected by looking for recurring event-type
      sequences across closed episodes.
    """

    def __init__(
        self,
        max_episodes: int = 500,
        episode_gap_seconds: float = 120.0,
    ) -> None:
        self._max_episodes = max_episodes
        self._episode_gap_seconds = episode_gap_seconds
        self._episodes: deque[Episode] = deque(maxlen=max_episodes)
        self._current: _OpenEpisode | None = None
        self._patterns: list[TemporalPattern] = []
        self._pattern_check_interval = 50  # check patterns every N closed episodes
        self._closed_since_last_check = 0

    # ── Public API ────────────────────────────────────────────────────────

    def push_event(self, event: EntityEvent, world_state: WorldState) -> Episode | None:
        """Feed one event.

        Returns a closed Episode if this call caused one to close, else None.
        """
        closed: Episode | None = None

        if self._current is None:
            # First event ever (or after reset): start a fresh episode.
            self._current = self._start_new_episode(event)
        elif (
            event.timestamp - self._current.last_event_time
            > self._episode_gap_seconds
        ):
            # Gap detected: close the current episode and open a new one.
            closed = self._close_episode(self._current, self._current.events[-1])
            self._episodes.append(closed)
            self._closed_since_last_check += 1
            if self._closed_since_last_check >= self._pattern_check_interval:
                self._detect_patterns()
                self._closed_since_last_check = 0
            self._current = self._start_new_episode(event)

        # Accumulate the event into the (possibly just-started) current episode.
        ep = self._current
        ep.events.append(event)
        ep.last_event_time = event.timestamp
        ep.entity_ids.add(event.entity_id)

        # Update specialised counters.
        if event.event_type == EntityEventType.ANOMALY:
            ep.anomaly_count += 1
        elif event.event_type == EntityEventType.RULE_VIOLATION:
            ep.violation_count += 1
        elif event.event_type in (
            EntityEventType.ZONE_ENTERED,
            EntityEventType.ZONE_EXITED,
        ):
            ep.zone_events += 1

        return closed

    def get_current_episode(self) -> Episode | None:
        """Return a snapshot of the currently-open episode (or None)."""
        if self._current is None:
            return None
        ep = self._current
        # Build a snapshot without closing.
        ja, en = self._build_summary(ep)
        duration = ep.last_event_time - ep.start_time
        type_counts: dict[str, int] = defaultdict(int)
        for e in ep.events:
            type_counts[e.event_type.name] += 1
        return Episode(
            id=ep.id,
            start_time=ep.start_time,
            end_time=None,
            start_frame=ep.start_frame,
            end_frame=ep.events[-1].frame_number if ep.events else ep.start_frame,
            entity_ids=sorted(ep.entity_ids),
            event_count=len(ep.events),
            event_type_counts=dict(type_counts),
            summary_ja=ja,
            summary_en=en,
            severity=self._compute_severity(ep),
            tags=self._extract_tags(ep),
            duration_seconds=duration,
        )

    def get_recent_episodes(self, n: int = 10) -> list[Episode]:
        """Return the n most-recent closed episodes (newest first)."""
        episodes = list(self._episodes)
        # _episodes is oldest-first (deque appended at right); reverse for newest-first.
        return list(reversed(episodes))[:n]

    def search(self, keywords: list[str], max_results: int = 5) -> list[Episode]:
        """Return episodes whose tags or summary contain ANY keyword (case-insensitive)."""
        if not keywords:
            return []
        lower_kw = [kw.lower() for kw in keywords]
        results: list[Episode] = []
        for ep in reversed(list(self._episodes)):  # newest first
            haystack = (
                " ".join(ep.tags)
                + " "
                + ep.summary_ja.lower()
                + " "
                + ep.summary_en.lower()
            )
            if any(kw in haystack for kw in lower_kw):
                results.append(ep)
            if len(results) >= max_results:
                break
        return results

    def get_temporal_patterns(self) -> list[TemporalPattern]:
        """Return all detected temporal patterns, sorted by confidence desc."""
        return sorted(self._patterns, key=lambda p: p.confidence, reverse=True)

    def get_cross_episode_summary(self, hours: int = 24) -> str:
        """Return a Japanese NL summary of the last `hours` hours of episodes."""
        now = _now_ts()
        cutoff = now - hours * 3600.0
        relevant = [ep for ep in self._episodes if ep.start_time >= cutoff]

        if not relevant:
            return f"過去{hours}時間はエピソードが記録されていません。"

        total_violations = sum(
            ep.event_type_counts.get(EntityEventType.RULE_VIOLATION.name, 0)
            for ep in relevant
        )
        total_anomalies = sum(
            ep.event_type_counts.get(EntityEventType.ANOMALY.name, 0)
            for ep in relevant
        )
        unique_entities: set[int] = set()
        for ep in relevant:
            unique_entities.update(ep.entity_ids)

        critical = sum(1 for ep in relevant if ep.severity == "critical")
        notable = sum(1 for ep in relevant if ep.severity == "notable")

        parts = [
            f"過去{hours}時間のサマリー: {len(relevant)}エピソード、"
            f"{len(unique_entities)}エンティティ観測、"
            f"違反{total_violations}件、異常{total_anomalies}件。"
        ]
        if critical:
            parts.append(f"重大エピソード: {critical}件。")
        if notable:
            parts.append(f"注目エピソード: {notable}件。")
        return "".join(parts)

    def all_episodes(self) -> list[Episode]:
        """Return all closed episodes (oldest first)."""
        return list(self._episodes)

    def reset(self) -> None:
        """Clear all memory."""
        self._episodes.clear()
        self._current = None
        self._patterns = []
        self._closed_since_last_check = 0

    # ── Internal helpers ──────────────────────────────────────────────────

    def _start_new_episode(self, event: EntityEvent) -> _OpenEpisode:
        return _OpenEpisode(
            id=str(uuid.uuid4()),
            start_time=event.timestamp,
            start_frame=event.frame_number,
            last_event_time=event.timestamp,
            entity_ids=set(),
            events=[],
        )

    def _close_episode(self, open_ep: _OpenEpisode, last_event: EntityEvent) -> Episode:
        """Convert an open episode to an immutable closed Episode."""
        end_time = open_ep.last_event_time
        duration = end_time - open_ep.start_time

        type_counts: dict[str, int] = defaultdict(int)
        for e in open_ep.events:
            type_counts[e.event_type.name] += 1

        ja, en = self._build_summary(open_ep)
        severity = self._compute_severity(open_ep)
        tags = self._extract_tags(open_ep)
        end_frame = open_ep.events[-1].frame_number if open_ep.events else open_ep.start_frame

        return Episode(
            id=open_ep.id,
            start_time=open_ep.start_time,
            end_time=end_time,
            start_frame=open_ep.start_frame,
            end_frame=end_frame,
            entity_ids=sorted(open_ep.entity_ids),
            event_count=len(open_ep.events),
            event_type_counts=dict(type_counts),
            summary_ja=ja,
            summary_en=en,
            severity=severity,
            tags=tags,
            duration_seconds=duration,
        )

    def _build_summary(self, open_ep: _OpenEpisode) -> tuple[str, str]:
        """Generate compact Japanese and English summaries."""
        duration_s = open_ep.last_event_time - open_ep.start_time
        minutes = duration_s / 60.0
        entity_count = len(open_ep.entity_ids)
        violations = open_ep.violation_count
        anomalies = open_ep.anomaly_count

        if minutes < 1.0:
            dur_ja = f"{int(duration_s)}秒間"
            dur_en = f"{int(duration_s)}-second"
        else:
            dur_ja = f"{minutes:.1f}分間"
            dur_en = f"{minutes:.1f}-minute"

        ja = (
            f"{dur_ja}の活動: {entity_count}エンティティ検出、"
            f"{violations}件の違反、{anomalies}件の異常"
        )
        en = (
            f"{dur_en} activity: {entity_count} "
            f"{'entity' if entity_count == 1 else 'entities'} detected, "
            f"{violations} {'violation' if violations == 1 else 'violations'}, "
            f"{anomalies} {'anomaly' if anomalies == 1 else 'anomalies'}"
        )
        return ja, en

    def _compute_severity(self, open_ep: _OpenEpisode) -> str:
        """Map episode counters to severity label."""
        if open_ep.anomaly_count > 0 or open_ep.violation_count > 2:
            return "critical"
        if open_ep.violation_count > 0:
            return "notable"
        return "normal"

    def _extract_tags(self, open_ep: _OpenEpisode) -> list[str]:
        """Derive searchable keyword tags from the episode's events."""
        tags: list[str] = []
        seen_types: set[str] = set()
        for e in open_ep.events:
            name = e.event_type.name.lower()
            if name not in seen_types:
                seen_types.add(name)
                tags.append(name)
            # Include detail labels (e.g. entity label from detection)
            label = e.details.get("label") or e.details.get("entity_label")
            if label and isinstance(label, str):
                lbl = label.lower()
                if lbl not in tags:
                    tags.append(lbl)

        # Severity as tag
        severity = self._compute_severity(open_ep)
        if severity not in tags:
            tags.append(severity)

        # Zone names
        for e in open_ep.events:
            zone = e.details.get("zone_id") or e.details.get("zone_name")
            if zone and isinstance(zone, str):
                z = zone.lower()
                if z not in tags:
                    tags.append(z)

        return tags

    def _detect_patterns(self) -> None:
        """Detect recurring event-type bigrams across closed episodes.

        A pattern is a bigram (pair of consecutive event types) that:
        - appears in at least 3 distinct episodes
        - tends to occur at similar times of day (within ±2 hours of the mean)
        """
        episodes = list(self._episodes)
        if len(episodes) < 3:
            return

        # Build bigram → list of (episode_index, hour_of_day, last_seen_ts)
        bigram_occurrences: dict[
            tuple[str, str], list[tuple[int, float, float]]
        ] = defaultdict(list)

        for idx, ep in enumerate(episodes):
            # Reconstruct event-type sequence for this episode from type_counts
            # (we only store counts in closed episodes, not the full sequence).
            # Use the start_time to get hour of day.
            hour = _hour_of_day(ep.start_time)
            # Extract bigrams from the ordered event list (we use event_type_counts
            # keys as a proxy; for real sequence we need the events — but they are
            # only in _OpenEpisode.  Use the key order of event_type_counts as a
            # rough sequence proxy, which is insertion-ordered in Python 3.7+.)
            type_sequence = list(ep.event_type_counts.keys())
            for i in range(len(type_sequence) - 1):
                bigram = (type_sequence[i], type_sequence[i + 1])
                bigram_occurrences[bigram].append((idx, hour, ep.start_time))

        new_patterns: list[TemporalPattern] = []
        for bigram, occurrences in bigram_occurrences.items():
            if len(occurrences) < 3:
                continue

            hours = [h for _, h, _ in occurrences]
            mean_hour = _circular_mean_hour(hours)
            # Check that at least 3 occurrences fall within ±2 hours of the mean.
            nearby = [
                h for h in hours if _hour_distance(h, mean_hour) <= 2.0
            ]
            if len(nearby) < 3:
                continue

            last_seen = max(ts for _, _, ts in occurrences)
            occ_count = len(occurrences)
            # Confidence: fraction of episodes that contain this bigram, capped to 1.
            confidence = min(1.0, occ_count / len(episodes))

            pattern_id = f"bigram_{bigram[0]}_{bigram[1]}"
            seq_pretty_a = bigram[0].replace("_", " ")
            seq_pretty_b = bigram[1].replace("_", " ")
            description_ja = (
                f"パターン: 「{seq_pretty_a}」→「{seq_pretty_b}」が"
                f"{occ_count}回繰り返し検出 (信頼度 {confidence:.0%})"
            )
            description_en = (
                f"Recurring pattern: '{seq_pretty_a}' → '{seq_pretty_b}' "
                f"detected {occ_count} times (confidence {confidence:.0%})"
            )

            new_patterns.append(
                TemporalPattern(
                    pattern_id=pattern_id,
                    description_ja=description_ja,
                    description_en=description_en,
                    occurrences=occ_count,
                    typical_hours=sorted(set(round(h, 1) for h in hours)),
                    event_sequence=list(bigram),
                    confidence=confidence,
                    last_seen=last_seen,
                )
            )

        # Merge with existing patterns (replace by pattern_id).
        existing_ids = {p.pattern_id: i for i, p in enumerate(self._patterns)}
        for np_ in new_patterns:
            if np_.pattern_id in existing_ids:
                self._patterns[existing_ids[np_.pattern_id]] = np_
            else:
                self._patterns.append(np_)

    @staticmethod
    def _find_ngrams(
        episodes: list[Episode], n: int
    ) -> dict[tuple[str, ...], int]:
        """Count event-type n-grams across all provided episodes."""
        counts: dict[tuple[str, ...], int] = defaultdict(int)
        for ep in episodes:
            seq = list(ep.event_type_counts.keys())
            for i in range(len(seq) - n + 1):
                gram = tuple(seq[i : i + n])
                counts[gram] += 1
        return dict(counts)


# ── Utility helpers ────────────────────────────────────────────────────────


def _now_ts() -> float:
    return datetime.now(tz=timezone.utc).timestamp()


def _hour_of_day(ts: float) -> float:
    """Return fractional hour of day (0.0–23.99) for a UTC unix timestamp."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0


def _circular_mean_hour(hours: list[float]) -> float:
    """Compute the circular mean of a list of hours (0–24 range)."""
    if not hours:
        return 0.0
    radians = [h * 2 * math.pi / 24.0 for h in hours]
    sin_mean = sum(math.sin(r) for r in radians) / len(radians)
    cos_mean = sum(math.cos(r) for r in radians) / len(radians)
    mean_rad = math.atan2(sin_mean, cos_mean)
    mean_hour = mean_rad * 24.0 / (2 * math.pi)
    if mean_hour < 0:
        mean_hour += 24.0
    return mean_hour


def _hour_distance(h1: float, h2: float) -> float:
    """Shortest circular distance between two hours (0–24)."""
    diff = abs(h1 - h2) % 24.0
    return min(diff, 24.0 - diff)
