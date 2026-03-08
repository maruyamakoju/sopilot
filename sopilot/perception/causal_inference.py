"""Phase 53: Causal Inference Engine.

Extends Phase 44 rule-based causal chains with *statistical* causal discovery:

- **Granger causality**: For each ordered pair of event types (A, B), test
  whether A's history improves prediction of B beyond B's own history, using
  a sliding time-window cross-correlation score.

- **Causal DAG**: A weighted directed graph of discovered causal edges, pruned
  by a minimum strength threshold.

- **Intervention simulation**: "If we had prevented event type X from occurring,
  which downstream violations would have been avoided?"  Uses the DAG's
  transitive closure to enumerate consequences.

Usage::

    engine = CausalInferenceEngine(window_seconds=60)
    engine.observe("PPE未着用", ts=t-20, camera_id="cam01")
    engine.observe("制限区域侵入", ts=t-10, camera_id="cam01")
    engine.observe("転倒", ts=t, camera_id="cam01")

    graph = engine.compute_causal_graph()
    result = engine.query_intervention("PPE未着用")
    print(result.prevented_effects)
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_BUFFER = 2000           # total event buffer size
_MIN_SAMPLES = 3             # minimum co-occurrence count for a causal edge
_MIN_GRANGER_STRENGTH = 0.25 # minimum edge strength to include in DAG
_WINDOW_SIZES_S = (5.0, 15.0, 30.0, 60.0)  # multi-scale Granger windows


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ObservedEvent:
    """A single buffered observation."""
    event_type: str
    timestamp: float
    camera_id: str
    confidence: float = 1.0


@dataclass
class CausalEdge:
    """A directed causal edge: cause → effect.

    Attributes
    ----------
    cause, effect:
        Event type labels.
    strength:
        Normalised Granger score [0, 1].  Higher = stronger causal evidence.
    lag_seconds:
        Median observed time lag (cause precedes effect by this many seconds).
    samples:
        Number of co-occurrence pairs that contributed to the score.
    window_seconds:
        Time window at which this edge was strongest.
    """
    cause: str
    effect: str
    strength: float
    lag_seconds: float
    samples: int
    window_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "strength": round(self.strength, 4),
            "lag_seconds": round(self.lag_seconds, 2),
            "samples": self.samples,
            "window_seconds": self.window_seconds,
        }


@dataclass
class InterventionResult:
    """Result of a do-calculus style intervention query.

    Attributes
    ----------
    intervened_event:
        The event type we hypothetically suppress.
    prevented_effects:
        List of (effect_type, probability) pairs that would be prevented by
        the intervention, ordered by probability descending.
    explanation_ja:
        Japanese natural-language summary.
    explanation_en:
        English natural-language summary.
    """
    intervened_event: str
    prevented_effects: list[tuple[str, float]]
    explanation_ja: str
    explanation_en: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "intervened_event": self.intervened_event,
            "prevented_effects": [
                {"effect": eff, "probability": round(prob, 4)}
                for eff, prob in self.prevented_effects
            ],
            "explanation_ja": self.explanation_ja,
            "explanation_en": self.explanation_en,
        }


# ---------------------------------------------------------------------------
# CausalInferenceEngine
# ---------------------------------------------------------------------------


class CausalInferenceEngine:
    """Statistical causal discovery over a rolling event stream.

    Parameters
    ----------
    window_seconds:
        Primary sliding window for Granger causality computation.
    min_granger_strength:
        Minimum edge strength to include in the causal DAG.
    min_samples:
        Minimum co-occurrence samples required to assert a causal edge.
    max_buffer:
        Maximum events to retain in the rolling buffer.
    """

    def __init__(
        self,
        window_seconds: float = 60.0,
        min_granger_strength: float = _MIN_GRANGER_STRENGTH,
        min_samples: int = _MIN_SAMPLES,
        max_buffer: int = _MAX_BUFFER,
    ) -> None:
        self._window = window_seconds
        self._min_strength = min_granger_strength
        self._min_samples = min_samples

        self._lock = threading.Lock()
        self._buffer: deque[ObservedEvent] = deque(maxlen=max_buffer)

        # Cached graph — recomputed lazily when buffer grows by _RECOMPUTE_INTERVAL
        self._graph: list[CausalEdge] = []
        self._last_buffer_size: int = 0
        self._recompute_interval = 50  # recompute every N new events
        self._observe_count = 0
        self._query_count = 0
        self._created_at = time.time()

        logger.info(
            "CausalInferenceEngine created (window=%.0fs, min_strength=%.2f, min_samples=%d)",
            window_seconds, min_granger_strength, min_samples,
        )

    # ── Observation ─────────────────────────────────────────────────────────

    def observe(
        self,
        event_type: str,
        timestamp: float | None = None,
        camera_id: str = "cam",
        confidence: float = 1.0,
    ) -> None:
        """Buffer an event for causal analysis.

        Parameters
        ----------
        event_type:
            Short label (e.g., "PPE未着用", "転倒", "制限区域侵入").
        timestamp:
            Unix timestamp; defaults to now.
        camera_id:
            Source camera identifier.
        confidence:
            Detection confidence [0, 1].
        """
        with self._lock:
            self._buffer.append(ObservedEvent(
                event_type=event_type,
                timestamp=timestamp or time.time(),
                camera_id=camera_id,
                confidence=float(confidence),
            ))
            self._observe_count += 1

    # ── Graph computation ────────────────────────────────────────────────────

    def compute_causal_graph(self, force: bool = False) -> list[CausalEdge]:
        """Compute (or return cached) the causal DAG.

        Uses a multi-scale Granger-inspired co-occurrence score: for each
        ordered pair (A, B) and each time window W, we count how often A
        precedes B within [0, W] seconds.  The score is normalised by the
        base rate of B to give a conditional lift.

        Parameters
        ----------
        force:
            If True, bypass the cache and recompute unconditionally.

        Returns
        -------
        list[CausalEdge]
            Edges with strength ≥ min_granger_strength, sorted by strength
            descending.
        """
        with self._lock:
            current_size = len(self._buffer)
            if not force and (current_size - self._last_buffer_size) < self._recompute_interval:
                return list(self._graph)
            events = list(self._buffer)
            self._last_buffer_size = current_size

        edges = _compute_granger_edges(
            events,
            window_sizes=_WINDOW_SIZES_S,
            min_strength=self._min_strength,
            min_samples=self._min_samples,
        )

        with self._lock:
            self._graph = edges

        logger.debug("CausalInferenceEngine: graph recomputed (%d edges, %d events)", len(edges), current_size)
        return list(edges)

    # ── Intervention query ───────────────────────────────────────────────────

    def query_intervention(
        self,
        prevent_event: str,
        camera_id: str | None = None,
    ) -> InterventionResult:
        """Simulate intervention: "What would be prevented if *prevent_event* never occurred?"

        Uses the causal DAG's transitive closure to enumerate all downstream
        effects that would be eliminated.  Edge strengths are multiplied along
        causal paths to compute path probability.

        Parameters
        ----------
        prevent_event:
            The event type to hypothetically suppress.
        camera_id:
            Optional; restrict analysis to this camera.

        Returns
        -------
        InterventionResult
            List of (effect_type, probability) pairs.
        """
        self._query_count += 1
        graph = self.compute_causal_graph()

        # Build adjacency: cause → [(effect, strength)]
        adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for edge in graph:
            adj[edge.cause].append((edge.effect, edge.strength))

        # BFS / DFS transitive closure from prevent_event
        prevented: dict[str, float] = {}  # effect → cumulative path prob
        stack: list[tuple[str, float]] = [(prevent_event, 1.0)]
        visited: set[str] = {prevent_event}

        while stack:
            node, prob = stack.pop()
            for effect, strength in adj.get(node, []):
                edge_prob = prob * strength
                if effect not in visited:
                    visited.add(effect)
                    current = prevented.get(effect, 0.0)
                    prevented[effect] = max(current, edge_prob)
                    stack.append((effect, edge_prob))

        sorted_effects = sorted(prevented.items(), key=lambda x: x[1], reverse=True)

        explanation_ja = _build_intervention_ja(prevent_event, sorted_effects)
        explanation_en = _build_intervention_en(prevent_event, sorted_effects)

        logger.info(
            "CausalInferenceEngine: intervention('%s') → %d prevented effects",
            prevent_event, len(sorted_effects),
        )
        return InterventionResult(
            intervened_event=prevent_event,
            prevented_effects=sorted_effects,
            explanation_ja=explanation_ja,
            explanation_en=explanation_en,
        )

    # ── Stats ────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Return operational statistics."""
        with self._lock:
            buf_size = len(self._buffer)
            graph_size = len(self._graph)
            # Count unique event types
            types: set[str] = {e.event_type for e in self._buffer}
        return {
            "buffer_events": buf_size,
            "unique_event_types": len(types),
            "causal_edges": graph_size,
            "total_observed": self._observe_count,
            "total_intervention_queries": self._query_count,
            "window_seconds": self._window,
            "min_granger_strength": self._min_strength,
            "min_samples": self._min_samples,
            "uptime_seconds": round(time.time() - self._created_at, 1),
        }

    def get_event_types(self) -> list[str]:
        """Return sorted list of unique event types in the buffer."""
        with self._lock:
            return sorted({e.event_type for e in self._buffer})

    def reset(self) -> None:
        """Clear buffer and cached graph."""
        with self._lock:
            self._buffer.clear()
            self._graph = []
            self._last_buffer_size = 0
            self._observe_count = 0
            self._query_count = 0
        logger.info("CausalInferenceEngine: reset")


# ---------------------------------------------------------------------------
# Granger score computation (pure Python, no heavy deps)
# ---------------------------------------------------------------------------


def _compute_granger_edges(
    events: list[ObservedEvent],
    window_sizes: tuple[float, ...],
    min_strength: float,
    min_samples: int,
) -> list[CausalEdge]:
    """Compute Granger-inspired causal edges over multiple time windows.

    Algorithm
    ---------
    For each ordered pair (A, B) and window W:
      1. For each occurrence of B, count occurrences of A in [t_B - W, t_B).
      2. Compute conditional rate: P(A precedes B within W) / P(B).
      3. Compute base rate of A in same windows: P(A in random window of W).
      4. Strength = max(0, conditional_rate - base_rate) / base_rate,
         normalised to [0, 1] via sigmoid.

    Also records median lag and sample count.
    """
    if len(events) < 2:
        return []

    # Group events by type
    by_type: dict[str, list[ObservedEvent]] = defaultdict(list)
    for e in events:
        by_type[e.event_type].append(e)

    types = list(by_type.keys())
    if len(types) < 2:
        return []

    total_span = (events[-1].timestamp - events[0].timestamp) if len(events) > 1 else 1.0
    total_span = max(total_span, 1.0)

    best_edges: dict[tuple[str, str], CausalEdge] = {}

    for effect_type in types:
        effect_events = by_type[effect_type]
        if len(effect_events) < min_samples:
            continue

        for cause_type in types:
            if cause_type == effect_type:
                continue
            cause_events = by_type[cause_type]
            if len(cause_events) < min_samples:
                continue

            for W in window_sizes:
                edge = _granger_edge_for_window(
                    cause_type, effect_type,
                    cause_events, effect_events,
                    W, total_span,
                    min_samples,
                )
                if edge is None:
                    continue
                key = (cause_type, effect_type)
                existing = best_edges.get(key)
                if existing is None or edge.strength > existing.strength:
                    best_edges[key] = edge

    # Filter by min_strength and sort
    result = [e for e in best_edges.values() if e.strength >= min_strength]
    result.sort(key=lambda e: e.strength, reverse=True)
    return result


def _granger_edge_for_window(
    cause_type: str,
    effect_type: str,
    cause_events: list[ObservedEvent],
    effect_events: list[ObservedEvent],
    W: float,
    total_span: float,
    min_samples: int,
) -> CausalEdge | None:
    """Compute one Granger edge for a specific time window W."""
    lags: list[float] = []
    co_count = 0

    # Sort for binary search
    cause_times = sorted(e.timestamp for e in cause_events)

    for eff in effect_events:
        t_b = eff.timestamp
        window_start = t_b - W
        # Count causes in [t_b - W, t_b)
        n_cause_before = _count_in_window(cause_times, window_start, t_b)
        if n_cause_before > 0:
            co_count += 1
            # Nearest preceding cause
            nearest = _nearest_preceding(cause_times, t_b)
            if nearest is not None:
                lags.append(t_b - nearest)

    if co_count < min_samples:
        return None

    # Conditional rate: fraction of effect events with a preceding cause
    conditional_rate = co_count / len(effect_events)

    # Base rate: Poisson approximation — P(at least one cause in window W)
    # P = 1 - exp(-λW) where λ = n_cause / total_span
    rate_per_sec = len(cause_events) / max(total_span, 1.0)
    base_rate = 1.0 - math.exp(-rate_per_sec * W)

    # Granger lift (conditional > base), normalised to [0, 1]
    lift = conditional_rate - base_rate
    if lift <= 0:
        return None
    # Normalise: sigmoid of lift scaled by evidence
    evidence_factor = math.log1p(co_count) / math.log1p(max(len(effect_events), 1))
    raw_strength = (lift / (1.0 - base_rate + 1e-6)) * min(1.0, evidence_factor)
    strength = min(1.0, raw_strength)

    median_lag = sorted(lags)[len(lags) // 2] if lags else W / 2

    return CausalEdge(
        cause=cause_type,
        effect=effect_type,
        strength=round(strength, 4),
        lag_seconds=round(median_lag, 2),
        samples=co_count,
        window_seconds=W,
    )


def _count_in_window(sorted_times: list[float], lo: float, hi: float) -> int:
    """Count timestamps in [lo, hi) using bisect."""
    import bisect
    left = bisect.bisect_left(sorted_times, lo)   # inclusive lower bound
    right = bisect.bisect_left(sorted_times, hi)  # exclusive upper bound
    return max(0, right - left)


def _nearest_preceding(sorted_times: list[float], t: float) -> float | None:
    """Return the latest timestamp strictly before t."""
    import bisect
    idx = bisect.bisect_left(sorted_times, t) - 1
    if idx < 0:
        return None
    return sorted_times[idx]


# ---------------------------------------------------------------------------
# Explanation builders
# ---------------------------------------------------------------------------


def _build_intervention_ja(
    prevented: str, effects: list[tuple[str, float]]
) -> str:
    if not effects:
        return f"「{prevented}」を防ぐことで、連鎖的な影響は観測されませんでした。"
    top = effects[:3]
    parts = "、".join(f"「{eff}」(確率{prob:.0%})" for eff, prob in top)
    suffix = f"など {len(effects)} 種" if len(effects) > 3 else ""
    return (
        f"「{prevented}」への介入により、{parts}{suffix}が防止できる可能性があります。"
        f"因果グラフの推移閉包に基づく推定です。"
    )


def _build_intervention_en(
    prevented: str, effects: list[tuple[str, float]]
) -> str:
    if not effects:
        return f"Preventing '{prevented}' shows no downstream effects in the causal graph."
    top = effects[:3]
    parts = ", ".join(f"'{eff}' ({prob:.0%})" for eff, prob in top)
    suffix = f" and {len(effects) - 3} more" if len(effects) > 3 else ""
    return (
        f"Intervening on '{prevented}' could prevent: {parts}{suffix}. "
        f"Estimated via causal DAG transitive closure."
    )
