"""Attention-Based Dynamic Frame Sampling.

Computes scene-level attention scores based on how "interesting" the
current frame is (entity changes, zone events, motion, novelty) and
uses those scores to adaptively control the frame sampling rate.

When nothing is happening, sampling drops to a minimum rate (saving
compute).  When entities enter, zones change, or motion spikes, the
sampler ramps up to capture every relevant frame.

Usage::

    scorer = SceneAttentionScorer()
    sampler = AdaptiveSampler(attention_scorer=scorer)

    for frame_ts, world_state in video_stream:
        decision = sampler.should_sample(frame_ts, world_state)
        if decision.should_analyze:
            result = engine.process_frame(frame, frame_ts, ...)
            sampler.update(scorer.score(result.world_state, previous_state))

Design:
    - AttentionScore is frozen (immutable snapshot per frame)
    - SamplingDecision is a lightweight value object per candidate frame
    - SceneAttentionScorer is stateless — all context passed via arguments
    - AdaptiveSampler is stateful — tracks current_fps and counters
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from sopilot.perception.types import (
    EntityEvent,
    EntityEventType,
    Track,
    WorldState,
)


# ── Data types ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AttentionScore:
    """Attention score for a single frame.

    ``total`` is the weighted sum of all components, clamped to [0, 1].
    ``components`` breaks down contributions for explainability.
    ``reason`` is a human-readable summary (Japanese OK for this project).
    """

    total: float  # 0.0 to 1.0, overall attention level
    components: dict[str, float]  # breakdown by factor
    reason: str  # human-readable explanation


@dataclass
class SamplingDecision:
    """Result of an adaptive sampling decision for a single timestamp.

    ``should_analyze`` tells the caller whether to send this frame to
    the VLM / full perception pipeline.
    """

    should_analyze: bool  # whether to send this frame to VLM
    attention_score: float  # most recent attention score (0 if unavailable)
    current_fps: float  # effective sampling rate at decision time
    reason: str  # human-readable explanation


# ── SceneAttentionScorer ──────────────────────────────────────────────────


class SceneAttentionScorer:
    """Computes an attention score (0-1) for how much the scene deserves analysis.

    High attention = more frames should be analyzed.
    Low attention = can skip frames.

    Four components are scored independently and combined via weighted sum:

    1. **entity_change** — new entities appeared or existing ones left.
    2. **zone_event** — zone entries/exits occurred.
    3. **motion** — average velocity magnitude of active tracks.
    4. **novelty** — scene looks different from previous state.

    The scorer itself is stateless: all context is passed via
    ``world_state`` and ``previous_state`` arguments.
    """

    # Velocity normalization constant: empirical max in normalised
    # coords / frame.  Velocities above this saturate to 1.0.
    _VELOCITY_MAX: float = 0.1

    def __init__(
        self,
        entity_change_weight: float = 0.3,
        zone_event_weight: float = 0.3,
        motion_weight: float = 0.2,
        novelty_weight: float = 0.2,
        decay_rate: float = 0.9,
    ) -> None:
        if not math.isclose(
            entity_change_weight + zone_event_weight + motion_weight + novelty_weight,
            1.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "Attention weights must sum to 1.0, got "
                f"{entity_change_weight + zone_event_weight + motion_weight + novelty_weight}"
            )

        self._entity_change_weight = entity_change_weight
        self._zone_event_weight = zone_event_weight
        self._motion_weight = motion_weight
        self._novelty_weight = novelty_weight
        self._decay_rate = decay_rate

    def score(
        self,
        world_state: WorldState,
        previous_state: WorldState | None = None,
    ) -> AttentionScore:
        """Compute attention score based on scene dynamics.

        Args:
            world_state: Current world state snapshot.
            previous_state: Previous world state (None on first frame).

        Returns:
            AttentionScore with total in [0, 1] and per-component breakdown.
        """
        entity_change = self._score_entity_change(world_state)
        zone_event = self._score_zone_event(world_state)
        motion = self._score_motion(world_state)
        novelty = self._score_novelty(world_state, previous_state)

        total = (
            self._entity_change_weight * entity_change
            + self._zone_event_weight * zone_event
            + self._motion_weight * motion
            + self._novelty_weight * novelty
        )
        total = max(0.0, min(1.0, total))

        components = {
            "entity_change": round(entity_change, 4),
            "zone_event": round(zone_event, 4),
            "motion": round(motion, 4),
            "novelty": round(novelty, 4),
        }

        reason = self._build_reason(components, total)

        return AttentionScore(
            total=round(total, 4),
            components=components,
            reason=reason,
        )

    # ── Component scoring ─────────────────────────────────────────────

    def _score_entity_change(self, world_state: WorldState) -> float:
        """Score based on entity enter/exit events this frame.

        Returns min(1.0, num_events / 3) where events are ENTERED + EXITED.
        """
        count = sum(
            1
            for e in world_state.events
            if e.event_type in (EntityEventType.ENTERED, EntityEventType.EXITED)
        )
        return min(1.0, count / 3.0)

    def _score_zone_event(self, world_state: WorldState) -> float:
        """Score based on zone entry/exit events this frame.

        Returns min(1.0, num_zone_events / 2).
        """
        count = sum(
            1
            for e in world_state.events
            if e.event_type
            in (EntityEventType.ZONE_ENTERED, EntityEventType.ZONE_EXITED)
        )
        return min(1.0, count / 2.0)

    def _score_motion(self, world_state: WorldState) -> float:
        """Score based on average velocity magnitude of active tracks.

        Velocity is in normalised coordinates per frame.  Normalised to
        [0, 1] using _VELOCITY_MAX as saturation point.
        """
        tracks = world_state.active_tracks
        if not tracks:
            return 0.0

        total_speed = 0.0
        count = 0
        for track in tracks.values():
            vx, vy = track.velocity
            speed = math.hypot(vx, vy)
            total_speed += speed
            count += 1

        if count == 0:
            return 0.0

        avg_speed = total_speed / count
        return min(1.0, avg_speed / self._VELOCITY_MAX)

    def _score_novelty(
        self,
        world_state: WorldState,
        previous_state: WorldState | None,
    ) -> float:
        """Score based on how different the scene looks from the previous state.

        Measures:
        - Entity count change
        - New labels appearing
        - Zone occupancy change

        Returns min(1.0, total_changes / 5).
        """
        if previous_state is None:
            # First frame — everything is novel
            return 1.0

        changes = 0.0

        # Entity count change
        count_diff = abs(world_state.entity_count - previous_state.entity_count)
        changes += count_diff

        # New labels appearing
        current_labels = {
            e.label for e in world_state.scene_graph.entities
        }
        previous_labels = {
            e.label for e in previous_state.scene_graph.entities
        }
        new_labels = current_labels - previous_labels
        changes += len(new_labels)

        # Zone occupancy change
        for zone_id in set(world_state.zone_occupancy) | set(
            previous_state.zone_occupancy
        ):
            curr_occ = len(world_state.zone_occupancy.get(zone_id, []))
            prev_occ = len(previous_state.zone_occupancy.get(zone_id, []))
            changes += abs(curr_occ - prev_occ)

        return min(1.0, changes / 5.0)

    # ── Reason builder ────────────────────────────────────────────────

    def _build_reason(
        self, components: dict[str, float], total: float
    ) -> str:
        """Build a human-readable reason string."""
        if total < 0.1:
            return "シーン安定 — 変化なし"

        parts: list[str] = []
        if components.get("entity_change", 0) > 0.1:
            parts.append("エンティティ変化")
        if components.get("zone_event", 0) > 0.1:
            parts.append("ゾーンイベント")
        if components.get("motion", 0) > 0.1:
            parts.append("動き検出")
        if components.get("novelty", 0) > 0.1:
            parts.append("シーン変化")

        if not parts:
            return f"注意スコア低 ({total:.2f})"

        return "注意要因: " + ", ".join(parts)


# ── AdaptiveSampler ───────────────────────────────────────────────────────


class AdaptiveSampler:
    """Controls frame sampling rate based on attention scores.

    When the scene is calm, sampling drops toward ``min_fps``.
    When attention spikes, sampling ramps up quickly toward ``max_fps``.
    Ramp-down is deliberately slower than ramp-up (conservative — we
    don't want to miss a follow-up event right after a spike).

    The sampler tracks total analyzed/skipped frames for efficiency
    reporting.
    """

    def __init__(
        self,
        min_fps: float = 0.1,
        max_fps: float = 2.0,
        base_fps: float = 0.5,
        ramp_up_speed: float = 0.5,
        ramp_down_speed: float = 0.1,
        attention_scorer: SceneAttentionScorer | None = None,
    ) -> None:
        if min_fps <= 0:
            raise ValueError(f"min_fps must be positive, got {min_fps}")
        if max_fps < min_fps:
            raise ValueError(
                f"max_fps ({max_fps}) must be >= min_fps ({min_fps})"
            )
        if not (min_fps <= base_fps <= max_fps):
            raise ValueError(
                f"base_fps ({base_fps}) must be between "
                f"min_fps ({min_fps}) and max_fps ({max_fps})"
            )

        self._min_fps = min_fps
        self._max_fps = max_fps
        self._base_fps = base_fps
        self._ramp_up_speed = ramp_up_speed
        self._ramp_down_speed = ramp_down_speed
        self._attention_scorer = attention_scorer

        # Internal state
        self._current_fps = base_fps
        self._last_sample_time: float | None = None
        self._last_attention_score: float = 0.0
        self._frames_analyzed: int = 0
        self._frames_skipped: int = 0

    # ── Main decision API ─────────────────────────────────────────────

    def should_sample(
        self,
        timestamp: float,
        world_state: WorldState | None = None,
    ) -> SamplingDecision:
        """Decide whether the current frame should be analyzed.

        If ``world_state`` is provided and an attention_scorer is
        configured, uses attention-based scoring to decide.  Otherwise
        falls back to time-based sampling at ``current_fps``.

        The first frame is always sampled regardless of timing.

        Args:
            timestamp: Current frame timestamp in seconds.
            world_state: Optional current world state for attention scoring.

        Returns:
            SamplingDecision indicating whether to analyze.
        """
        # First frame is always sampled
        if self._last_sample_time is None:
            self._last_sample_time = timestamp
            self._frames_analyzed += 1
            return SamplingDecision(
                should_analyze=True,
                attention_score=self._last_attention_score,
                current_fps=self._current_fps,
                reason="初回フレーム — 必ず解析",
            )

        # Compute time since last sample
        elapsed = timestamp - self._last_sample_time
        interval = 1.0 / self._current_fps if self._current_fps > 0 else float("inf")

        if elapsed >= interval:
            # Time to sample
            self._last_sample_time = timestamp
            self._frames_analyzed += 1
            return SamplingDecision(
                should_analyze=True,
                attention_score=self._last_attention_score,
                current_fps=self._current_fps,
                reason=f"サンプリング間隔到達 (間隔={interval:.2f}s, 経過={elapsed:.2f}s)",
            )
        else:
            # Skip this frame
            self._frames_skipped += 1
            return SamplingDecision(
                should_analyze=False,
                attention_score=self._last_attention_score,
                current_fps=self._current_fps,
                reason=f"スキップ (次回まで{interval - elapsed:.2f}s)",
            )

    def update(self, attention_score: AttentionScore) -> None:
        """Update internal sampling rate based on latest attention score.

        Adjusts ``current_fps`` toward a target derived from the
        attention score.  Ramp-up is fast, ramp-down is slow.

        Args:
            attention_score: Latest attention score from SceneAttentionScorer.
        """
        self._last_attention_score = attention_score.total

        # Compute target fps from attention
        target_fps = (
            self._base_fps
            + (self._max_fps - self._base_fps) * attention_score.total
        )

        # Asymmetric ramp: fast up, slow down
        if target_fps > self._current_fps:
            self._current_fps += self._ramp_up_speed * (
                target_fps - self._current_fps
            )
        else:
            self._current_fps -= self._ramp_down_speed * (
                self._current_fps - target_fps
            )

        # Clamp to valid range
        self._current_fps = max(
            self._min_fps, min(self._max_fps, self._current_fps)
        )

    # ── Properties ────────────────────────────────────────────────────

    @property
    def current_fps(self) -> float:
        """Current effective sampling rate."""
        return self._current_fps

    @property
    def frames_analyzed(self) -> int:
        """Total frames that were analyzed."""
        return self._frames_analyzed

    @property
    def frames_skipped(self) -> int:
        """Total frames that were skipped."""
        return self._frames_skipped

    @property
    def efficiency_ratio(self) -> float:
        """Ratio of analyzed to total frames (lower = more efficient).

        Returns 0.0 if no frames have been processed.
        """
        total = self._frames_analyzed + self._frames_skipped
        if total == 0:
            return 0.0
        return self._frames_analyzed / total
