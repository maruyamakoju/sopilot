"""Tests for sopilot.perception.attention — Attention-Based Dynamic Frame Sampling.

Covers:
    - SceneAttentionScorer component scoring (entity_change, zone_event, motion, novelty)
    - AttentionScore data type invariants
    - AdaptiveSampler ramp-up / ramp-down behaviour
    - Efficiency ratio tracking
    - Edge cases: empty world state, no previous state, first-frame guarantee
    - Time-based fallback when no world_state is provided
"""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

from sopilot.perception.attention import (
    AdaptiveSampler,
    AttentionScore,
    SamplingDecision,
    SceneAttentionScorer,
)
from sopilot.perception.types import (
    BBox,
    EntityEvent,
    EntityEventType,
    SceneEntity,
    SceneGraph,
    Track,
    TrackState,
    WorldState,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_scene_graph(
    timestamp: float = 0.0,
    frame_number: int = 0,
    entities: list[SceneEntity] | None = None,
) -> SceneGraph:
    return SceneGraph(
        timestamp=timestamp,
        frame_number=frame_number,
        entities=entities or [],
        relations=[],
    )


def _make_world_state(
    timestamp: float = 0.0,
    frame_number: int = 0,
    entities: list[SceneEntity] | None = None,
    events: list[EntityEvent] | None = None,
    active_tracks: dict[int, Track] | None = None,
    zone_occupancy: dict[str, list[int]] | None = None,
    entity_count: int | None = None,
) -> WorldState:
    ents = entities or []
    sg = _make_scene_graph(
        timestamp=timestamp,
        frame_number=frame_number,
        entities=ents,
    )
    return WorldState(
        timestamp=timestamp,
        frame_number=frame_number,
        scene_graph=sg,
        active_tracks=active_tracks or {},
        events=events or [],
        zone_occupancy=zone_occupancy or {},
        entity_count=entity_count if entity_count is not None else len(ents),
        person_count=0,
    )


def _make_entity(
    entity_id: int = 1,
    label: str = "person",
    zone_ids: list[str] | None = None,
) -> SceneEntity:
    return SceneEntity(
        entity_id=entity_id,
        label=label,
        bbox=BBox(0.1, 0.1, 0.3, 0.3),
        confidence=0.9,
        zone_ids=zone_ids or [],
    )


def _make_track(
    track_id: int = 1,
    velocity: tuple[float, float] = (0.0, 0.0),
) -> Track:
    return Track(
        track_id=track_id,
        label="person",
        state=TrackState.ACTIVE,
        bbox=BBox(0.1, 0.1, 0.3, 0.3),
        velocity=velocity,
        confidence=0.9,
    )


def _make_event(
    event_type: EntityEventType,
    entity_id: int = 1,
    timestamp: float = 0.0,
    frame_number: int = 0,
) -> EntityEvent:
    return EntityEvent(
        event_type=event_type,
        entity_id=entity_id,
        timestamp=timestamp,
        frame_number=frame_number,
    )


# ── SceneAttentionScorer Tests ────────────────────────────────────────────


class TestSceneAttentionScorer(unittest.TestCase):
    """Tests for the SceneAttentionScorer class."""

    def setUp(self) -> None:
        self.scorer = SceneAttentionScorer()

    def test_calm_scene_low_attention(self) -> None:
        """Calm scene with no events, no motion, and a previous state yields low score."""
        prev = _make_world_state(timestamp=0.0)
        curr = _make_world_state(timestamp=1.0)

        score = self.scorer.score(curr, prev)

        self.assertLessEqual(score.total, 0.1)
        self.assertAlmostEqual(score.components["entity_change"], 0.0)
        self.assertAlmostEqual(score.components["zone_event"], 0.0)
        self.assertAlmostEqual(score.components["motion"], 0.0)

    def test_entity_entries_high_attention(self) -> None:
        """Entities entering the scene drive entity_change component high."""
        events = [
            _make_event(EntityEventType.ENTERED, entity_id=1),
            _make_event(EntityEventType.ENTERED, entity_id=2),
            _make_event(EntityEventType.ENTERED, entity_id=3),
        ]
        curr = _make_world_state(events=events)
        prev = _make_world_state()

        score = self.scorer.score(curr, prev)

        self.assertAlmostEqual(score.components["entity_change"], 1.0)
        self.assertGreater(score.total, 0.2)

    def test_entity_exits_contribute(self) -> None:
        """Entity EXITED events also contribute to entity_change."""
        events = [
            _make_event(EntityEventType.EXITED, entity_id=1),
        ]
        curr = _make_world_state(events=events)
        prev = _make_world_state()

        score = self.scorer.score(curr, prev)

        self.assertGreater(score.components["entity_change"], 0.0)
        # 1/3 = 0.333...
        self.assertAlmostEqual(score.components["entity_change"], 1.0 / 3.0, places=3)

    def test_zone_events_high_attention(self) -> None:
        """Zone entries/exits drive zone_event component."""
        events = [
            _make_event(EntityEventType.ZONE_ENTERED, entity_id=1),
            _make_event(EntityEventType.ZONE_EXITED, entity_id=2),
        ]
        curr = _make_world_state(events=events)
        prev = _make_world_state()

        score = self.scorer.score(curr, prev)

        self.assertAlmostEqual(score.components["zone_event"], 1.0)
        self.assertGreater(score.total, 0.2)

    def test_high_motion_increases_attention(self) -> None:
        """High velocity tracks increase the motion component."""
        tracks = {
            1: _make_track(track_id=1, velocity=(0.08, 0.06)),  # speed = 0.1
        }
        curr = _make_world_state(active_tracks=tracks)
        prev = _make_world_state()

        score = self.scorer.score(curr, prev)

        self.assertGreater(score.components["motion"], 0.9)

    def test_zero_motion_when_no_tracks(self) -> None:
        """Motion is 0 when there are no active tracks."""
        curr = _make_world_state()
        prev = _make_world_state()

        score = self.scorer.score(curr, prev)

        self.assertAlmostEqual(score.components["motion"], 0.0)

    def test_novelty_high_on_first_frame(self) -> None:
        """First frame (no previous state) gets max novelty."""
        curr = _make_world_state()

        score = self.scorer.score(curr, previous_state=None)

        self.assertAlmostEqual(score.components["novelty"], 1.0)

    def test_novelty_from_entity_count_change(self) -> None:
        """Entity count change contributes to novelty."""
        prev = _make_world_state(entity_count=0)
        curr = _make_world_state(entity_count=3)

        score = self.scorer.score(curr, prev)

        # 3 changes / 5 = 0.6
        self.assertGreater(score.components["novelty"], 0.5)

    def test_novelty_from_new_labels(self) -> None:
        """New labels appearing contributes to novelty."""
        prev_entities = [_make_entity(entity_id=1, label="person")]
        curr_entities = [
            _make_entity(entity_id=1, label="person"),
            _make_entity(entity_id=2, label="forklift"),
            _make_entity(entity_id=3, label="helmet"),
        ]
        prev = _make_world_state(entities=prev_entities, entity_count=1)
        curr = _make_world_state(entities=curr_entities, entity_count=3)

        score = self.scorer.score(curr, prev)

        # new labels: forklift, helmet = 2; entity count change: 2; total changes = 4; 4/5 = 0.8
        self.assertGreater(score.components["novelty"], 0.5)

    def test_novelty_from_zone_occupancy_change(self) -> None:
        """Zone occupancy change contributes to novelty."""
        prev = _make_world_state(
            zone_occupancy={"zone_a": [1, 2]}
        )
        curr = _make_world_state(
            zone_occupancy={"zone_a": [1, 2, 3, 4, 5]}
        )

        score = self.scorer.score(curr, prev)

        # zone change: |5 - 2| = 3 occupancy diff; 3/5 = 0.6
        self.assertGreater(score.components["novelty"], 0.5)

    def test_total_clamped_to_0_1(self) -> None:
        """Total score is always in [0, 1] even with extreme inputs."""
        events = [
            _make_event(EntityEventType.ENTERED, entity_id=i)
            for i in range(10)
        ] + [
            _make_event(EntityEventType.ZONE_ENTERED, entity_id=i)
            for i in range(10)
        ]
        tracks = {
            i: _make_track(track_id=i, velocity=(0.5, 0.5))
            for i in range(10)
        }
        curr = _make_world_state(events=events, active_tracks=tracks, entity_count=10)

        score = self.scorer.score(curr, previous_state=None)

        self.assertGreaterEqual(score.total, 0.0)
        self.assertLessEqual(score.total, 1.0)

    def test_score_has_reason(self) -> None:
        """AttentionScore always has a non-empty reason string."""
        curr = _make_world_state()
        prev = _make_world_state()

        score = self.scorer.score(curr, prev)

        self.assertIsInstance(score.reason, str)
        self.assertTrue(len(score.reason) > 0)

    def test_score_is_frozen(self) -> None:
        """AttentionScore is immutable (frozen dataclass)."""
        score = AttentionScore(total=0.5, components={}, reason="test")

        with self.assertRaises(AttributeError):
            score.total = 0.9  # type: ignore[misc]

    def test_weights_must_sum_to_one(self) -> None:
        """Weights that don't sum to 1.0 raise ValueError."""
        with self.assertRaises(ValueError):
            SceneAttentionScorer(
                entity_change_weight=0.5,
                zone_event_weight=0.5,
                motion_weight=0.5,
                novelty_weight=0.5,
            )

    def test_reason_shows_calm_scene(self) -> None:
        """Calm scene reason indicates stability."""
        prev = _make_world_state()
        curr = _make_world_state()

        score = self.scorer.score(curr, prev)

        self.assertIn("安定", score.reason)

    def test_reason_shows_contributing_factors(self) -> None:
        """Active scene reason lists contributing factors."""
        events = [
            _make_event(EntityEventType.ENTERED, entity_id=1),
            _make_event(EntityEventType.ENTERED, entity_id=2),
            _make_event(EntityEventType.ENTERED, entity_id=3),
            _make_event(EntityEventType.ZONE_ENTERED, entity_id=1),
            _make_event(EntityEventType.ZONE_ENTERED, entity_id=2),
        ]
        tracks = {
            1: _make_track(track_id=1, velocity=(0.08, 0.06)),
        }
        curr = _make_world_state(events=events, active_tracks=tracks, entity_count=5)

        score = self.scorer.score(curr, previous_state=None)

        self.assertIn("注意要因", score.reason)

    def test_mixed_events_combined_score(self) -> None:
        """Score combines multiple components correctly."""
        events = [
            _make_event(EntityEventType.ENTERED, entity_id=1),
            _make_event(EntityEventType.ZONE_ENTERED, entity_id=1),
        ]
        tracks = {
            1: _make_track(track_id=1, velocity=(0.05, 0.0)),
        }
        curr = _make_world_state(
            events=events,
            active_tracks=tracks,
            entity_count=1,
        )
        prev = _make_world_state(entity_count=0)

        score = self.scorer.score(curr, prev)

        # entity_change: 1/3, zone_event: 1/2, motion: 0.05/0.1=0.5, novelty > 0
        self.assertGreater(score.components["entity_change"], 0.0)
        self.assertGreater(score.components["zone_event"], 0.0)
        self.assertGreater(score.components["motion"], 0.0)
        self.assertGreater(score.total, 0.2)

    def test_motion_saturates_at_velocity_max(self) -> None:
        """Motion component saturates at 1.0 for very high velocities."""
        tracks = {
            1: _make_track(track_id=1, velocity=(1.0, 1.0)),
        }
        curr = _make_world_state(active_tracks=tracks)
        prev = _make_world_state()

        score = self.scorer.score(curr, prev)

        self.assertAlmostEqual(score.components["motion"], 1.0)


# ── AdaptiveSampler Tests ─────────────────────────────────────────────────


class TestAdaptiveSampler(unittest.TestCase):
    """Tests for the AdaptiveSampler class."""

    def test_first_frame_always_sampled(self) -> None:
        """First call to should_sample always returns should_analyze=True."""
        sampler = AdaptiveSampler()

        decision = sampler.should_sample(0.0)

        self.assertTrue(decision.should_analyze)
        self.assertEqual(sampler.frames_analyzed, 1)
        self.assertEqual(sampler.frames_skipped, 0)

    def test_time_based_sampling_without_world_state(self) -> None:
        """Without world_state, uses time-based sampling at current_fps."""
        sampler = AdaptiveSampler(base_fps=1.0)

        # First frame
        d1 = sampler.should_sample(0.0)
        self.assertTrue(d1.should_analyze)

        # 0.5s later — too soon for 1 fps
        d2 = sampler.should_sample(0.5)
        self.assertFalse(d2.should_analyze)

        # 1.0s later — exactly at interval
        d3 = sampler.should_sample(1.0)
        self.assertTrue(d3.should_analyze)

    def test_ramp_up_on_high_attention(self) -> None:
        """FPS increases when attention score is high."""
        sampler = AdaptiveSampler(
            base_fps=0.5,
            max_fps=2.0,
            ramp_up_speed=0.5,
        )
        initial_fps = sampler.current_fps

        # Simulate high attention
        high_score = AttentionScore(
            total=1.0,
            components={"entity_change": 1.0, "zone_event": 0.0, "motion": 0.0, "novelty": 0.0},
            reason="test",
        )
        sampler.update(high_score)

        self.assertGreater(sampler.current_fps, initial_fps)

    def test_ramp_down_slower_than_ramp_up(self) -> None:
        """FPS decreases more slowly than it increases."""
        sampler = AdaptiveSampler(
            base_fps=0.5,
            max_fps=2.0,
            ramp_up_speed=0.5,
            ramp_down_speed=0.1,
        )

        # Ramp up
        high_score = AttentionScore(total=1.0, components={}, reason="up")
        sampler.update(high_score)
        fps_after_up = sampler.current_fps
        delta_up = fps_after_up - 0.5  # change from base

        # Now ramp down from the elevated fps
        low_score = AttentionScore(total=0.0, components={}, reason="down")
        sampler.update(low_score)
        fps_after_down = sampler.current_fps
        delta_down = fps_after_up - fps_after_down  # change downward

        # Ramp-down step should be smaller than ramp-up step
        self.assertGreater(delta_up, delta_down)

    def test_fps_clamped_to_range(self) -> None:
        """FPS stays within [min_fps, max_fps]."""
        sampler = AdaptiveSampler(min_fps=0.1, max_fps=2.0, base_fps=0.5)

        # Many ramp-ups
        high_score = AttentionScore(total=1.0, components={}, reason="max")
        for _ in range(100):
            sampler.update(high_score)
        self.assertLessEqual(sampler.current_fps, 2.0)

        # Many ramp-downs
        low_score = AttentionScore(total=0.0, components={}, reason="min")
        for _ in range(100):
            sampler.update(low_score)
        self.assertGreaterEqual(sampler.current_fps, 0.1)

    def test_efficiency_ratio_tracking(self) -> None:
        """Efficiency ratio correctly tracks analyzed vs total."""
        sampler = AdaptiveSampler(base_fps=1.0)

        # Frame 0: analyzed (first frame)
        sampler.should_sample(0.0)
        # Frame 0.3: skipped
        sampler.should_sample(0.3)
        # Frame 0.6: skipped
        sampler.should_sample(0.6)
        # Frame 1.0: analyzed
        sampler.should_sample(1.0)
        # Frame 1.3: skipped
        sampler.should_sample(1.3)

        self.assertEqual(sampler.frames_analyzed, 2)
        self.assertEqual(sampler.frames_skipped, 3)
        self.assertAlmostEqual(sampler.efficiency_ratio, 2.0 / 5.0)

    def test_efficiency_ratio_zero_when_no_frames(self) -> None:
        """Efficiency ratio is 0.0 before any frames are processed."""
        sampler = AdaptiveSampler()

        self.assertAlmostEqual(sampler.efficiency_ratio, 0.0)

    def test_current_fps_starts_at_base(self) -> None:
        """Current FPS starts at the configured base_fps."""
        sampler = AdaptiveSampler(base_fps=0.75)

        self.assertAlmostEqual(sampler.current_fps, 0.75)

    def test_sampling_decision_has_reason(self) -> None:
        """SamplingDecision always has a non-empty reason."""
        sampler = AdaptiveSampler()

        d1 = sampler.should_sample(0.0)
        self.assertTrue(len(d1.reason) > 0)

        d2 = sampler.should_sample(0.1)
        self.assertTrue(len(d2.reason) > 0)

    def test_invalid_min_fps_raises(self) -> None:
        """min_fps <= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            AdaptiveSampler(min_fps=0.0)

        with self.assertRaises(ValueError):
            AdaptiveSampler(min_fps=-1.0)

    def test_invalid_max_fps_raises(self) -> None:
        """max_fps < min_fps raises ValueError."""
        with self.assertRaises(ValueError):
            AdaptiveSampler(min_fps=2.0, max_fps=1.0)

    def test_invalid_base_fps_raises(self) -> None:
        """base_fps outside [min_fps, max_fps] raises ValueError."""
        with self.assertRaises(ValueError):
            AdaptiveSampler(min_fps=0.5, max_fps=2.0, base_fps=0.1)

        with self.assertRaises(ValueError):
            AdaptiveSampler(min_fps=0.5, max_fps=2.0, base_fps=3.0)

    def test_gradual_ramp_up_over_multiple_updates(self) -> None:
        """FPS gradually approaches max_fps with sustained high attention."""
        sampler = AdaptiveSampler(
            base_fps=0.5,
            max_fps=2.0,
            ramp_up_speed=0.3,
        )
        high_score = AttentionScore(total=1.0, components={}, reason="high")

        fps_history: list[float] = [sampler.current_fps]
        for _ in range(20):
            sampler.update(high_score)
            fps_history.append(sampler.current_fps)

        # FPS should be monotonically increasing
        for i in range(1, len(fps_history)):
            self.assertGreaterEqual(fps_history[i], fps_history[i - 1])

        # Should approach max_fps
        self.assertGreater(sampler.current_fps, 1.5)

    def test_gradual_ramp_down_over_multiple_updates(self) -> None:
        """FPS gradually decreases with sustained zero attention."""
        sampler = AdaptiveSampler(
            min_fps=0.1,
            base_fps=0.5,
            max_fps=2.0,
            ramp_up_speed=1.0,
            ramp_down_speed=0.1,
        )

        # First ramp up to near max
        high_score = AttentionScore(total=1.0, components={}, reason="high")
        for _ in range(50):
            sampler.update(high_score)

        elevated_fps = sampler.current_fps
        self.assertGreater(elevated_fps, 1.5)

        # Now ramp down
        low_score = AttentionScore(total=0.0, components={}, reason="low")
        fps_history: list[float] = [sampler.current_fps]
        for _ in range(20):
            sampler.update(low_score)
            fps_history.append(sampler.current_fps)

        # FPS should be monotonically decreasing
        for i in range(1, len(fps_history)):
            self.assertLessEqual(fps_history[i], fps_history[i - 1])

        # Should have decreased but still be above min
        self.assertLess(sampler.current_fps, elevated_fps)

    def test_update_adjusts_sampling_interval(self) -> None:
        """After ramp-up, frames are sampled more frequently."""
        sampler = AdaptiveSampler(
            base_fps=0.5,
            max_fps=2.0,
            ramp_up_speed=1.0,
        )

        # First frame
        sampler.should_sample(0.0)

        # At base_fps=0.5, interval is 2.0s.
        # 1.5s should be skipped.
        d_before = sampler.should_sample(1.5)
        self.assertFalse(d_before.should_analyze)

        # Now ramp up to high fps
        high_score = AttentionScore(total=1.0, components={}, reason="high")
        for _ in range(50):
            sampler.update(high_score)

        # Reset last_sample_time by sampling at 2.0
        sampler.should_sample(2.0)

        # At ~2.0 fps, interval is ~0.5s.  0.6s should be enough.
        d_after = sampler.should_sample(2.6)
        self.assertTrue(d_after.should_analyze)


# ── Integration Tests ─────────────────────────────────────────────────────


class TestAttentionIntegration(unittest.TestCase):
    """End-to-end tests combining scorer and sampler."""

    def test_full_lifecycle(self) -> None:
        """Scorer + sampler work together through a calm-active-calm cycle."""
        scorer = SceneAttentionScorer()
        sampler = AdaptiveSampler(
            base_fps=0.5,
            max_fps=2.0,
            min_fps=0.1,
            attention_scorer=scorer,
        )

        # Calm phase
        prev_state = None
        for i in range(5):
            ts = float(i) * 2.0
            ws = _make_world_state(timestamp=ts, frame_number=i)
            decision = sampler.should_sample(ts, ws)
            if decision.should_analyze:
                score = scorer.score(ws, prev_state)
                sampler.update(score)
            prev_state = ws

        calm_fps = sampler.current_fps

        # Active phase — entities entering
        for i in range(5, 10):
            ts = float(i) * 2.0
            events = [
                _make_event(EntityEventType.ENTERED, entity_id=100 + i, timestamp=ts),
                _make_event(EntityEventType.ZONE_ENTERED, entity_id=100 + i, timestamp=ts),
            ]
            entities = [_make_entity(entity_id=100 + j) for j in range(i - 4)]
            ws = _make_world_state(
                timestamp=ts,
                frame_number=i,
                events=events,
                entities=entities,
                entity_count=len(entities),
            )
            decision = sampler.should_sample(ts, ws)
            if decision.should_analyze:
                score = scorer.score(ws, prev_state)
                sampler.update(score)
            prev_state = ws

        active_fps = sampler.current_fps

        # FPS should have increased during active phase
        self.assertGreater(active_fps, calm_fps)

        # Calm-down phase
        for i in range(10, 30):
            ts = float(i) * 2.0
            ws = _make_world_state(timestamp=ts, frame_number=i)
            decision = sampler.should_sample(ts, ws)
            if decision.should_analyze:
                score = scorer.score(ws, prev_state)
                sampler.update(score)
            prev_state = ws

        # FPS should have decreased
        self.assertLess(sampler.current_fps, active_fps)

    def test_empty_world_state(self) -> None:
        """Scorer handles completely empty world state gracefully."""
        scorer = SceneAttentionScorer()
        ws = _make_world_state()

        score = scorer.score(ws, previous_state=None)

        self.assertGreaterEqual(score.total, 0.0)
        self.assertLessEqual(score.total, 1.0)
        self.assertIsInstance(score.components, dict)

    def test_sampler_with_scorer_attribute(self) -> None:
        """AdaptiveSampler stores scorer reference for external coordination."""
        scorer = SceneAttentionScorer()
        sampler = AdaptiveSampler(attention_scorer=scorer)

        self.assertIs(sampler._attention_scorer, scorer)

    def test_sampling_decision_dataclass(self) -> None:
        """SamplingDecision fields are accessible."""
        d = SamplingDecision(
            should_analyze=True,
            attention_score=0.75,
            current_fps=1.5,
            reason="test reason",
        )
        self.assertTrue(d.should_analyze)
        self.assertAlmostEqual(d.attention_score, 0.75)
        self.assertAlmostEqual(d.current_fps, 1.5)
        self.assertEqual(d.reason, "test reason")


if __name__ == "__main__":
    unittest.main()
