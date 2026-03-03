"""Causal Reasoning from Scene Sequences.

Infers cause-effect relationships between events produced by the world
model.  The CausalReasoner scans a rolling buffer of recent events
and, for each new event, looks backwards for plausible causes according
to registered CausalPatterns.

The module ships with a set of built-in safety-relevant patterns
(equipment removal -> violation, zone entry -> prolonged presence,
approach -> zone entry, activity change -> incident, proximity ->
interaction).  Custom patterns can be registered at runtime via
``add_pattern()``.

Thread safety:
    All public methods on CausalReasoner acquire an RLock so the
    reasoner can be called from background processing threads while
    the main thread reads causal chains.

Typical usage::

    reasoner = CausalReasoner()
    # ... in the frame loop:
    world_state = world_model.update(scene_graph)
    causal_links = reasoner.analyze(world_state)
    for link in causal_links:
        print(link.explanation_ja)
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from sopilot.perception.types import (
    EntityEvent,
    EntityEventType,
    WorldState,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CausalLink:
    """A causal relationship between two events."""

    cause_event: EntityEvent
    effect_event: EntityEvent
    cause_type: str  # e.g., "zone_entry", "equipment_removal", "proximity"
    confidence: float  # 0-1, how confident we are in the causal link
    explanation_ja: str  # Japanese explanation
    explanation_en: str  # English explanation
    time_delta_seconds: float  # time between cause and effect


@dataclass
class CausalPattern:
    """A learned causal pattern from observed event sequences."""

    pattern_id: str
    cause_type: EntityEventType
    effect_type: EntityEventType
    cause_conditions: dict[str, Any] = field(default_factory=dict)
    effect_conditions: dict[str, Any] = field(default_factory=dict)
    max_time_gap_seconds: float = 30.0  # maximum time between cause and effect
    occurrences: int = 0  # how many times observed
    confidence: float = 0.5  # prior confidence, updated with observations


# ---------------------------------------------------------------------------
# Built-in pattern definitions
# ---------------------------------------------------------------------------

_BUILTIN_PATTERNS: list[CausalPattern] = [
    # 1. Equipment removal -> Violation
    #    If an entity has a STATE_CHANGED (equipment attribute removed) and
    #    a RULE_VIOLATION follows, the removal likely caused the violation.
    CausalPattern(
        pattern_id="equipment_removal_violation",
        cause_type=EntityEventType.STATE_CHANGED,
        effect_type=EntityEventType.RULE_VIOLATION,
        cause_conditions={"safety_relevant": True},
        effect_conditions={},
        max_time_gap_seconds=30.0,
        occurrences=0,
        confidence=0.8,
    ),
    # 2. Zone entry -> Prolonged presence
    #    Entering a restricted zone eventually leads to prolonged presence.
    CausalPattern(
        pattern_id="zone_entry_prolonged",
        cause_type=EntityEventType.ZONE_ENTERED,
        effect_type=EntityEventType.PROLONGED_PRESENCE,
        cause_conditions={},
        effect_conditions={},
        max_time_gap_seconds=300.0,  # up to 5 minutes for prolonged presence
        occurrences=0,
        confidence=0.9,
    ),
    # 3. Approach (zone entry predicted) -> Zone entry
    #    Trajectory prediction of zone entry followed by actual entry.
    CausalPattern(
        pattern_id="approach_zone_entry",
        cause_type=EntityEventType.ZONE_ENTRY_PREDICTED,
        effect_type=EntityEventType.ZONE_ENTERED,
        cause_conditions={},
        effect_conditions={},
        max_time_gap_seconds=15.0,
        occurrences=0,
        confidence=0.85,
    ),
    # 4. Activity change (anomaly) -> Violation
    #    An anomaly event followed by a rule violation or another anomaly.
    CausalPattern(
        pattern_id="anomaly_violation",
        cause_type=EntityEventType.ANOMALY,
        effect_type=EntityEventType.RULE_VIOLATION,
        cause_conditions={},
        effect_conditions={},
        max_time_gap_seconds=20.0,
        occurrences=0,
        confidence=0.6,
    ),
    # 5. State change -> State change (proximity interaction)
    #    Two entities near each other, both experiencing state changes.
    CausalPattern(
        pattern_id="proximity_interaction",
        cause_type=EntityEventType.STATE_CHANGED,
        effect_type=EntityEventType.STATE_CHANGED,
        cause_conditions={},
        effect_conditions={},
        max_time_gap_seconds=5.0,
        occurrences=0,
        confidence=0.5,
    ),
]


# ---------------------------------------------------------------------------
# Helper: Japanese explanation generators
# ---------------------------------------------------------------------------

_CAUSE_TYPE_JA: dict[str, str] = {
    "equipment_removal_violation": "安全装備が外された",
    "zone_entry_prolonged": "制限区域に入った",
    "approach_zone_entry": "接近が検出された",
    "anomaly_violation": "異常な行動パターンが検出された",
    "proximity_interaction": "近接状態での属性変化が発生した",
}

_CAUSE_TYPE_EN: dict[str, str] = {
    "equipment_removal_violation": "safety equipment was removed",
    "zone_entry_prolonged": "entered a restricted zone",
    "approach_zone_entry": "approach trajectory was detected",
    "anomaly_violation": "anomalous activity pattern was detected",
    "proximity_interaction": "state change occurred in proximity",
}


def _generate_explanation_ja(
    cause: EntityEvent,
    effect: EntityEvent,
    pattern_id: str,
    time_delta: float,
) -> str:
    """Generate a Japanese explanation for a causal link."""
    cause_desc = _CAUSE_TYPE_JA.get(pattern_id, cause.event_type.value)
    return (
        f"この{effect.event_type.value}は、"
        f"エンティティ{cause.entity_id}が"
        f"{cause_desc}"
        f"（{time_delta:.1f}秒前）"
        f"ことが原因と推測されます"
    )


def _generate_explanation_en(
    cause: EntityEvent,
    effect: EntityEvent,
    pattern_id: str,
    time_delta: float,
) -> str:
    """Generate an English explanation for a causal link."""
    cause_desc = _CAUSE_TYPE_EN.get(pattern_id, cause.event_type.value)
    return (
        f"This {effect.event_type.value} was likely caused by "
        f"entity {cause.entity_id}: {cause_desc} "
        f"({time_delta:.1f}s ago)"
    )


# ---------------------------------------------------------------------------
# CausalReasoner
# ---------------------------------------------------------------------------


class CausalReasoner:
    """Infers causal relationships between events in the world model.

    Maintains a rolling buffer of recent events and matches incoming
    events against registered causal patterns to produce CausalLink
    objects.  All public methods are thread-safe.

    Args:
        max_history_events: Maximum event history to maintain.
        max_time_gap_seconds: Default maximum time gap for causal links.
    """

    def __init__(
        self,
        max_history_events: int = 500,
        max_time_gap_seconds: float = 30.0,
    ) -> None:
        self._max_time_gap = max_time_gap_seconds
        self._lock = threading.RLock()

        # Rolling event buffer.
        self._event_buffer: deque[EntityEvent] = deque(
            maxlen=max_history_events,
        )

        # Registered causal patterns.
        self._patterns: list[CausalPattern] = []

        # Discovered causal links (effect event -> list of CausalLink).
        # Keyed by (entity_id, timestamp, event_type) for fast lookup.
        self._links: dict[tuple[int, float, str], list[CausalLink]] = {}

        # Set of already-processed event identity tuples to avoid
        # re-analyzing events that were already in the buffer.
        self._processed_events: set[tuple[int, float, str]] = set()

        # Load built-in patterns.
        for p in _BUILTIN_PATTERNS:
            self._patterns.append(
                CausalPattern(
                    pattern_id=p.pattern_id,
                    cause_type=p.cause_type,
                    effect_type=p.effect_type,
                    cause_conditions=dict(p.cause_conditions),
                    effect_conditions=dict(p.effect_conditions),
                    max_time_gap_seconds=p.max_time_gap_seconds,
                    occurrences=p.occurrences,
                    confidence=p.confidence,
                )
            )

        logger.info(
            "CausalReasoner initialized (buffer=%d, default_gap=%.1fs, "
            "patterns=%d)",
            max_history_events,
            max_time_gap_seconds,
            len(self._patterns),
        )

    # -- public API --------------------------------------------------------

    def analyze(self, world_state: WorldState) -> list[CausalLink]:
        """Analyze current world state events for causal relationships.

        Checks new events against recent event history for potential
        cause-effect relationships.

        Args:
            world_state: The current WorldState produced by the world model.

        Returns:
            List of newly discovered CausalLink objects.
        """
        with self._lock:
            new_links: list[CausalLink] = []

            for event in world_state.events:
                event_key = _event_key(event)

                # Skip events we have already processed as potential effects.
                if event_key in self._processed_events:
                    continue

                # Try to find causes in the buffer.
                links = self._find_causes(event)
                if links:
                    new_links.extend(links)
                    self._links[event_key] = links

                self._processed_events.add(event_key)

                # Add to the rolling buffer for future cause lookups.
                self._event_buffer.append(event)

            # Trim processed-events set to match buffer bounds.
            self._trim_processed()

            if new_links:
                logger.info(
                    "Frame %d: %d causal links discovered",
                    world_state.frame_number,
                    len(new_links),
                )

            return new_links

    def add_pattern(self, pattern: CausalPattern) -> None:
        """Register a causal pattern to watch for.

        Args:
            pattern: The CausalPattern to register.
        """
        with self._lock:
            self._patterns.append(pattern)
            logger.info(
                "Registered causal pattern '%s' (%s -> %s)",
                pattern.pattern_id,
                pattern.cause_type.value,
                pattern.effect_type.value,
            )

    def get_causal_chain(self, event: EntityEvent) -> list[CausalLink]:
        """Trace the causal chain leading to a specific event.

        Follows cause links backwards recursively to build the full
        chain from root cause to the given event.

        Args:
            event: The event to trace backwards from.

        Returns:
            List of CausalLink objects ordered from root cause to the
            given event (oldest first).
        """
        with self._lock:
            chain: list[CausalLink] = []
            visited: set[tuple[int, float, str]] = set()
            self._trace_chain(event, chain, visited)
            # Reverse so root cause is first.
            chain.reverse()
            return chain

    def explain_violation(
        self,
        violation_event: EntityEvent,
        recent_events: list[EntityEvent],
    ) -> str:
        """Generate a natural-language explanation of why a violation occurred.

        Uses the causal chain if available, otherwise scans recent_events
        for plausible causes.

        Args:
            violation_event: The violation event to explain.
            recent_events: Additional recent events to consider.

        Returns:
            A Japanese explanation string.
        """
        with self._lock:
            # First try the stored causal chain.
            chain = self.get_causal_chain(violation_event)
            if chain:
                parts: list[str] = []
                for i, link in enumerate(chain, 1):
                    parts.append(
                        f"{i}. {link.explanation_ja}"
                    )
                return (
                    f"違反の因果連鎖（{len(chain)}ステップ）:\n"
                    + "\n".join(parts)
                )

            # Fallback: scan recent_events for plausible causes.
            best_link = self._find_best_cause_from_list(
                violation_event, recent_events,
            )
            if best_link is not None:
                return best_link.explanation_ja

            # No causal link found.
            return (
                f"エンティティ{violation_event.entity_id}による"
                f"{violation_event.event_type.value}が発生しましたが、"
                f"直接的な原因は特定できませんでした"
            )

    # -- internal ----------------------------------------------------------

    def _find_causes(self, effect: EntityEvent) -> list[CausalLink]:
        """Scan the event buffer for events that could be causes of *effect*."""
        links: list[CausalLink] = []

        for pattern in self._patterns:
            if pattern.effect_type != effect.event_type:
                continue

            # Check effect conditions.
            if not _matches_conditions(effect.details, pattern.effect_conditions):
                continue

            # Scan buffer backwards (most recent first) for matching causes.
            for cause in reversed(self._event_buffer):
                if cause.event_type != pattern.cause_type:
                    continue

                time_delta = effect.timestamp - cause.timestamp
                if time_delta < 0:
                    continue
                if time_delta > pattern.max_time_gap_seconds:
                    continue

                # Check cause conditions.
                if not _matches_conditions(cause.details, pattern.cause_conditions):
                    continue

                # Entity relationship boosts confidence.
                same_entity = cause.entity_id == effect.entity_id
                entity_factor = 1.0 if same_entity else 0.7

                # For proximity_interaction, we specifically want
                # *different* entities (same entity is less interesting).
                if pattern.pattern_id == "proximity_interaction":
                    if same_entity:
                        continue
                    entity_factor = 0.8

                # For zone-related patterns, check zone_id match.
                if pattern.pattern_id in (
                    "zone_entry_prolonged",
                    "approach_zone_entry",
                ):
                    cause_zone = cause.details.get("zone_id")
                    effect_zone = effect.details.get("zone_id")
                    if cause_zone and effect_zone and cause_zone != effect_zone:
                        continue
                    # Same entity required for zone patterns.
                    if not same_entity:
                        continue

                # Temporal proximity factor: closer in time = higher confidence.
                max_gap = pattern.max_time_gap_seconds
                temporal_factor = 1.0 - (time_delta / max_gap) if max_gap > 0 else 1.0

                confidence = min(
                    1.0,
                    pattern.confidence * entity_factor * temporal_factor,
                )

                explanation_ja = _generate_explanation_ja(
                    cause, effect, pattern.pattern_id, time_delta,
                )
                explanation_en = _generate_explanation_en(
                    cause, effect, pattern.pattern_id, time_delta,
                )

                link = CausalLink(
                    cause_event=cause,
                    effect_event=effect,
                    cause_type=pattern.pattern_id,
                    confidence=confidence,
                    explanation_ja=explanation_ja,
                    explanation_en=explanation_en,
                    time_delta_seconds=time_delta,
                )
                links.append(link)

                # Update pattern statistics.
                pattern.occurrences += 1

                logger.debug(
                    "Causal link: %s (entity %d, t=%.3f) -> %s (entity %d, "
                    "t=%.3f), confidence=%.2f, pattern=%s",
                    cause.event_type.value,
                    cause.entity_id,
                    cause.timestamp,
                    effect.event_type.value,
                    effect.entity_id,
                    effect.timestamp,
                    confidence,
                    pattern.pattern_id,
                )

                # Take only the best (most recent) cause per pattern.
                break

        return links

    def _find_best_cause_from_list(
        self,
        effect: EntityEvent,
        candidates: list[EntityEvent],
    ) -> CausalLink | None:
        """Find the best plausible cause from a list of candidate events."""
        best: CausalLink | None = None

        for cause in reversed(candidates):
            time_delta = effect.timestamp - cause.timestamp
            if time_delta < 0 or time_delta > self._max_time_gap:
                continue

            # Look for any pattern that matches.
            for pattern in self._patterns:
                if pattern.cause_type != cause.event_type:
                    continue
                if pattern.effect_type != effect.event_type:
                    continue
                if not _matches_conditions(cause.details, pattern.cause_conditions):
                    continue
                if not _matches_conditions(effect.details, pattern.effect_conditions):
                    continue

                same_entity = cause.entity_id == effect.entity_id
                entity_factor = 1.0 if same_entity else 0.7
                max_gap = pattern.max_time_gap_seconds
                temporal_factor = (
                    1.0 - (time_delta / max_gap) if max_gap > 0 else 1.0
                )
                confidence = min(
                    1.0,
                    pattern.confidence * entity_factor * temporal_factor,
                )

                link = CausalLink(
                    cause_event=cause,
                    effect_event=effect,
                    cause_type=pattern.pattern_id,
                    confidence=confidence,
                    explanation_ja=_generate_explanation_ja(
                        cause, effect, pattern.pattern_id, time_delta,
                    ),
                    explanation_en=_generate_explanation_en(
                        cause, effect, pattern.pattern_id, time_delta,
                    ),
                    time_delta_seconds=time_delta,
                )

                if best is None or link.confidence > best.confidence:
                    best = link
                break

        return best

    def _trace_chain(
        self,
        event: EntityEvent,
        chain: list[CausalLink],
        visited: set[tuple[int, float, str]],
    ) -> None:
        """Recursively trace causal chain backwards from *event*."""
        key = _event_key(event)
        if key in visited:
            return
        visited.add(key)

        links = self._links.get(key)
        if not links:
            return

        # Follow the highest-confidence link.
        best = max(links, key=lambda lk: lk.confidence)
        chain.append(best)
        self._trace_chain(best.cause_event, chain, visited)

    def _trim_processed(self) -> None:
        """Keep processed-events set bounded to buffer contents."""
        if len(self._processed_events) > len(self._event_buffer) * 2:
            buffer_keys = {_event_key(e) for e in self._event_buffer}
            link_keys = set(self._links.keys())
            self._processed_events = buffer_keys | link_keys

    # -- diagnostics -------------------------------------------------------

    @property
    def pattern_count(self) -> int:
        """Number of registered patterns."""
        with self._lock:
            return len(self._patterns)

    @property
    def buffer_size(self) -> int:
        """Number of events in the rolling buffer."""
        with self._lock:
            return len(self._event_buffer)

    @property
    def link_count(self) -> int:
        """Total number of discovered causal links."""
        with self._lock:
            return sum(len(v) for v in self._links.values())

    def get_patterns(self) -> list[CausalPattern]:
        """Return a copy of all registered patterns."""
        with self._lock:
            return list(self._patterns)

    def reset(self) -> None:
        """Clear all state."""
        with self._lock:
            self._event_buffer.clear()
            self._links.clear()
            self._processed_events.clear()
            logger.info("CausalReasoner reset")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _event_key(event: EntityEvent) -> tuple[int, float, str]:
    """Produce a hashable identity tuple for an event."""
    return (event.entity_id, event.timestamp, event.event_type.value)


def _matches_conditions(
    details: dict[str, Any],
    conditions: dict[str, Any],
) -> bool:
    """Check whether event details satisfy all conditions.

    An empty conditions dict matches everything.
    """
    for key, expected in conditions.items():
        if key not in details:
            return False
        if details[key] != expected:
            return False
    return True


# ── Causal Graph ──────────────────────────────────────────────────────────────


@dataclass
class CausalNode:
    """A node in the causal graph representing a single event."""

    node_id: str                    # UUID
    event: EntityEvent
    caused_by: list[str] = field(default_factory=list)   # node_ids of cause nodes
    causes: list[str] = field(default_factory=list)      # node_ids of effect nodes
    importance: float = 0.0         # computed centrality


class CausalGraph:
    """Directed causal graph over recent events.

    Maintains a DAG where edges represent causal links detected by
    CausalAnalyzer.  Provides:
    - Root cause identification (follow chains backward to events with no cause)
    - Intervention target ranking (which events, if prevented, would eliminate the most violations)
    - Natural-language causal narrative generation in Japanese
    - Automatic pruning of old nodes (configurable max age)

    Usage:
        graph = CausalGraph()
        for link in causal_links:
            graph.add_link(link)
        roots = graph.get_root_causes()
        narrative = graph.get_causal_narrative()
    """

    def __init__(self, max_nodes: int = 200, max_age_seconds: float = 600.0) -> None:
        self._max_nodes = max_nodes
        self._max_age_seconds = max_age_seconds
        self._nodes: dict[str, CausalNode] = {}          # node_id → CausalNode
        # Track edges separately for fast lookup
        self._cause_to_effects: dict[str, list[str]] = defaultdict(list)  # cause_node_id → effect_node_ids
        self._effect_to_causes: dict[str, list[str]] = defaultdict(list)  # effect_node_id → cause_node_ids

    def add_link(self, link: CausalLink) -> tuple[str, str]:
        """Register a CausalLink as two nodes + one edge.

        Creates nodes for cause_event and effect_event if they don't exist.
        Matches events by timestamp+entity_id to avoid duplicates.
        Returns (cause_node_id, effect_node_id).
        """
        cause_id = self._find_or_create_node(link.cause_event)
        effect_id = self._find_or_create_node(link.effect_event)

        cause_node = self._nodes[cause_id]
        effect_node = self._nodes[effect_id]

        # Add the edge only if it does not already exist.
        if effect_id not in cause_node.causes:
            cause_node.causes.append(effect_id)
        if cause_id not in effect_node.caused_by:
            effect_node.caused_by.append(cause_id)

        # Keep the fast-lookup edge indices in sync.
        if effect_id not in self._cause_to_effects[cause_id]:
            self._cause_to_effects[cause_id].append(effect_id)
        if cause_id not in self._effect_to_causes[effect_id]:
            self._effect_to_causes[effect_id].append(cause_id)

        # Enforce max_nodes limit by discarding the oldest nodes (by timestamp).
        while len(self._nodes) > self._max_nodes:
            self._evict_oldest()

        return cause_id, effect_id

    def get_root_causes(self, max_depth: int = 5) -> list[CausalNode]:
        """Return nodes that have no incoming causal edges (root causes).
        Sorted by importance desc (number of downstream effects).
        """
        self.compute_importance()
        roots = [
            node for node in self._nodes.values()
            if not node.caused_by
        ]
        roots.sort(key=lambda n: n.importance, reverse=True)
        return roots

    def get_consequences(self, node_id: str, max_depth: int = 5) -> list[CausalNode]:
        """Follow causal chain forward from node_id. BFS up to max_depth.
        Returns list of consequence nodes (excluding the start node).
        """
        if node_id not in self._nodes:
            return []

        visited: set[str] = {node_id}
        queue: list[tuple[str, int]] = [(node_id, 0)]
        results: list[CausalNode] = []

        while queue:
            current_id, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            for effect_id in self._cause_to_effects.get(current_id, []):
                if effect_id not in visited:
                    visited.add(effect_id)
                    results.append(self._nodes[effect_id])
                    queue.append((effect_id, depth + 1))

        return results

    def get_intervention_targets(self) -> list[tuple[CausalNode, int]]:
        """Return (node, downstream_violation_count) pairs sorted by violation_count desc.

        For each node, count how many RULE_VIOLATION events are reachable
        downstream (via get_consequences). Nodes with more downstream violations
        are higher-leverage intervention points.
        """
        results: list[tuple[CausalNode, int]] = []
        for node_id, node in self._nodes.items():
            consequences = self.get_consequences(node_id)
            violation_count = sum(
                1 for c in consequences
                if c.event.event_type == EntityEventType.RULE_VIOLATION
            )
            # Also count the node itself if it is a violation.
            if node.event.event_type == EntityEventType.RULE_VIOLATION:
                violation_count += 1
            results.append((node, violation_count))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_causal_narrative(self, max_chains: int = 3) -> str:
        """Generate a Japanese NL description of the top causal chains.

        For each root cause (up to max_chains):
          「{cause_event_type}」が発生し、その結果「{effect1}」「{effect2}」...に繋がりました。

        Example:
          「STATE_CHANGED」が発生し、その結果「RULE_VIOLATION」「ANOMALY」に繋がりました。
          「ZONE_ENTERED」が発生し、その結果「PROLONGED_PRESENCE」に繋がりました。
        """
        roots = self.get_root_causes()
        if not roots:
            return "因果関係のあるイベントはまだ記録されていません。"

        lines: list[str] = []
        for root in roots[:max_chains]:
            consequences = self.get_consequences(root.node_id)
            if not consequences:
                continue
            effect_types = list(dict.fromkeys(
                c.event.event_type.value for c in consequences
            ))
            effect_str = "".join(f"「{et}」" for et in effect_types)
            cause_type = root.event.event_type.value
            lines.append(
                f"「{cause_type}」が発生し、その結果{effect_str}に繋がりました。"
            )

        if not lines:
            return "因果関係のあるイベントはまだ記録されていません。"

        return "\n".join(lines)

    def prune_old_nodes(self, current_time: float) -> int:
        """Remove nodes older than max_age_seconds. Returns count removed."""
        cutoff = current_time - self._max_age_seconds
        to_remove = [
            nid for nid, node in self._nodes.items()
            if node.event.timestamp < cutoff
        ]
        for nid in to_remove:
            self._remove_node(nid)
        return len(to_remove)

    def compute_importance(self) -> None:
        """Recompute importance scores for all nodes.
        importance = (number of descendants) / total_nodes (normalized centrality).
        """
        total = len(self._nodes)
        if total == 0:
            return
        for node_id, node in self._nodes.items():
            descendants = self.get_consequences(node_id)
            node.importance = len(descendants) / total

    def get_stats(self) -> dict:
        """Return graph statistics for API/debugging."""
        edge_count = sum(
            len(effects) for effects in self._cause_to_effects.values()
        )
        root_count = sum(
            1 for node in self._nodes.values() if not node.caused_by
        )
        leaf_count = sum(
            1 for node in self._nodes.values() if not node.causes
        )
        return {
            "node_count": len(self._nodes),
            "edge_count": edge_count,
            "root_count": root_count,
            "leaf_count": leaf_count,
            "max_nodes": self._max_nodes,
            "max_age_seconds": self._max_age_seconds,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _find_or_create_node(self, event: EntityEvent) -> str:
        """Find existing node matching this event, or create a new one. Returns node_id."""
        for node_id, node in self._nodes.items():
            if self._event_matches(node, event):
                return node_id
        # Create a new node.
        node_id = str(uuid.uuid4())
        self._nodes[node_id] = CausalNode(node_id=node_id, event=event)
        return node_id

    def _event_matches(self, node: CausalNode, event: EntityEvent) -> bool:
        """True if node.event and event refer to the same event instance.
        Match by: event_type, entity_id, and abs(timestamp - node.event.timestamp) < 1.0
        """
        return (
            node.event.event_type == event.event_type
            and node.event.entity_id == event.entity_id
            and abs(node.event.timestamp - event.timestamp) < 1.0
        )

    def _remove_node(self, node_id: str) -> None:
        """Remove a node and clean up all edge references to/from it."""
        if node_id not in self._nodes:
            return
        node = self._nodes.pop(node_id)

        # Remove this node from its causes' effect lists.
        for cause_id in node.caused_by:
            if cause_id in self._nodes:
                cause_node = self._nodes[cause_id]
                if node_id in cause_node.causes:
                    cause_node.causes.remove(node_id)
            if cause_id in self._cause_to_effects:
                if node_id in self._cause_to_effects[cause_id]:
                    self._cause_to_effects[cause_id].remove(node_id)

        # Remove this node from its effects' cause lists.
        for effect_id in node.causes:
            if effect_id in self._nodes:
                effect_node = self._nodes[effect_id]
                if node_id in effect_node.caused_by:
                    effect_node.caused_by.remove(node_id)
            if effect_id in self._effect_to_causes:
                if node_id in self._effect_to_causes[effect_id]:
                    self._effect_to_causes[effect_id].remove(node_id)

        # Remove edge-index entries for this node.
        self._cause_to_effects.pop(node_id, None)
        self._effect_to_causes.pop(node_id, None)

    def _evict_oldest(self) -> None:
        """Remove the oldest node (by event timestamp) to stay within max_nodes."""
        if not self._nodes:
            return
        oldest_id = min(
            self._nodes,
            key=lambda nid: self._nodes[nid].event.timestamp,
        )
        self._remove_node(oldest_id)
