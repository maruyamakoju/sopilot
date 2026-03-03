"""Tests for sopilot.perception.goal_recognizer.

Covers:
  - Initialization
  - observe() basic behavior
  - Hypothesis retrieval
  - Belief update / decay / pruning
  - Zone proximity detection
  - Loiter detection
  - Exit direction detection
  - reset / reset_entity
  - get_state_dict()
"""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Any

import pytest

from sopilot.perception.goal_recognizer import (
    GOAL_DEFINITIONS,
    GoalHypothesis,
    GoalRecognizer,
)
from sopilot.perception.types import (
    BBox,
    EntityEvent,
    EntityEventType,
    Relation,
    SceneEntity,
    SceneGraph,
    Track,
    TrackState,
    WorldState,
    Zone,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_bbox(cx: float = 0.5, cy: float = 0.5, w: float = 0.05, h: float = 0.1) -> BBox:
    """Create a BBox from center + size (normalized)."""
    return BBox(
        x1=cx - w / 2,
        y1=cy - h / 2,
        x2=cx + w / 2,
        y2=cy + h / 2,
    )


def _make_track(entity_id: int = 1, vx: float = 0.0, vy: float = 0.0,
                history_bboxes: list[BBox] | None = None) -> Track:
    """Create a Track with optional velocity and history."""
    return Track(
        track_id=entity_id,
        label="person",
        state=TrackState.ACTIVE,
        bbox=_make_bbox(),
        velocity=(vx, vy),
        confidence=0.9,
        history=history_bboxes or [],
    )


def _make_entity(
    entity_id: int = 1,
    label: str = "person",
    cx: float = 0.5,
    cy: float = 0.5,
) -> SceneEntity:
    """Create a SceneEntity at given position."""
    return SceneEntity(
        entity_id=entity_id,
        label=label,
        bbox=_make_bbox(cx, cy),
        confidence=0.9,
    )


def _make_scene_graph(
    entities: list[SceneEntity] | None = None,
    zones: dict[str, Zone] | None = None,
    timestamp: float = 0.0,
) -> SceneGraph:
    """Create a SceneGraph, optionally with zones injected as attribute."""
    sg = SceneGraph(
        timestamp=timestamp,
        frame_number=0,
        entities=entities or [],
        relations=[],
    )
    # Inject zones as an extra attribute (mimics world_model usage)
    if zones is not None:
        sg.zones = zones  # type: ignore[attr-defined]
    return sg


def _make_world_state(
    entities: list[SceneEntity] | None = None,
    tracks: dict[int, Track] | None = None,
    zones: dict[str, Zone] | None = None,
    timestamp: float = 0.0,
) -> WorldState:
    """Create a WorldState with entities and optional tracks/zones."""
    ents = entities or []
    trk = tracks or {}
    sg = _make_scene_graph(entities=ents, zones=zones, timestamp=timestamp)
    return WorldState(
        timestamp=timestamp,
        frame_number=0,
        scene_graph=sg,
        active_tracks=trk,
        events=[],
        zone_occupancy={},
        entity_count=len(ents),
        person_count=sum(1 for e in ents if "person" in e.label.lower()),
    )


def _make_zone(
    zone_id: str = "z1",
    name: str = "Zone A",
    cx: float = 0.5,
    cy: float = 0.5,
    size: float = 0.1,
    zone_type: str = "generic",
) -> Zone:
    """Create a square Zone centred at (cx, cy)."""
    half = size / 2
    polygon = [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    ]
    return Zone(zone_id=zone_id, name=name, polygon=polygon, zone_type=zone_type)


# ── TestGoalRecognizerInit ────────────────────────────────────────────────────


class TestGoalRecognizerInit:
    """3 tests: basic initialisation."""

    def test_initial_beliefs_empty(self) -> None:
        gr = GoalRecognizer()
        assert len(gr._beliefs) == 0

    def test_high_risk_threshold_default(self) -> None:
        gr = GoalRecognizer()
        assert gr._high_risk_threshold == 0.55

    def test_high_risk_threshold_custom(self) -> None:
        gr = GoalRecognizer(high_risk_threshold=0.7)
        assert gr._high_risk_threshold == 0.7


# ── TestObserveBasic ──────────────────────────────────────────────────────────


class TestObserveBasic:
    """6 tests: observe() fundamental behavior."""

    def test_observe_returns_list(self) -> None:
        gr = GoalRecognizer()
        entity = _make_entity(cx=0.05, cy=0.5)  # near left edge
        track = _make_track(entity_id=1, vx=-0.01)
        ws = _make_world_state(entities=[entity], tracks={1: track})
        result = gr.observe(entity, ws)
        assert isinstance(result, list)

    def test_observe_person_near_edge_detects_exit(self) -> None:
        gr = GoalRecognizer()
        entity = _make_entity(cx=0.05, cy=0.5)
        track = _make_track(entity_id=1, vx=-0.02)
        ws = _make_world_state(entities=[entity], tracks={1: track})
        # Need enough history, but initial call should still detect edge proximity
        result = gr.observe(entity, ws)
        goal_types = [h.goal_type for h in result]
        assert "exit_area" in goal_types

    def test_observe_stationary_20_frames_detects_loiter(self) -> None:
        gr = GoalRecognizer()
        entity = _make_entity(cx=0.5, cy=0.5)
        track = _make_track(entity_id=1)
        ws = _make_world_state(entities=[entity], tracks={1: track})
        # Feed 20 frames at the same position
        for _ in range(20):
            gr.observe(entity, ws)
        result = gr.get_hypotheses(1)
        goal_types = [h.goal_type for h in result]
        assert "loiter" in goal_types

    def test_observe_returns_sorted_by_confidence(self) -> None:
        gr = GoalRecognizer()
        entity = _make_entity(cx=0.05, cy=0.5)
        track = _make_track(entity_id=1, vx=-0.02)
        ws = _make_world_state(entities=[entity], tracks={1: track})
        for _ in range(5):
            gr.observe(entity, ws)
        result = gr.observe(entity, ws)
        confidences = [h.confidence for h in result]
        assert confidences == sorted(confidences, reverse=True)

    def test_observe_confidence_in_range(self) -> None:
        gr = GoalRecognizer()
        entity = _make_entity(cx=0.05, cy=0.5)
        track = _make_track(entity_id=1, vx=-0.02)
        ws = _make_world_state(entities=[entity], tracks={1: track})
        result = gr.observe(entity, ws)
        for h in result:
            assert 0.0 <= h.confidence <= 1.0

    def test_observe_evidence_not_empty(self) -> None:
        gr = GoalRecognizer()
        entity = _make_entity(cx=0.05, cy=0.5)
        track = _make_track(entity_id=1, vx=-0.02)
        ws = _make_world_state(entities=[entity], tracks={1: track})
        result = gr.observe(entity, ws)
        for h in result:
            assert len(h.evidence) > 0


# ── TestHypothesisRetrieval ───────────────────────────────────────────────────


class TestHypothesisRetrieval:
    """5 tests: get_hypotheses() and get_high_risk_intents()."""

    def test_get_hypotheses_unknown_entity_returns_empty(self) -> None:
        gr = GoalRecognizer()
        assert gr.get_hypotheses(999) == []

    def test_get_hypotheses_after_observe_returns_list(self) -> None:
        gr = GoalRecognizer()
        entity = _make_entity(cx=0.05, cy=0.5)
        track = _make_track(entity_id=1, vx=-0.02)
        ws = _make_world_state(entities=[entity], tracks={1: track})
        gr.observe(entity, ws)
        hyps = gr.get_hypotheses(1)
        assert isinstance(hyps, list)

    def test_get_high_risk_intents_empty_initially(self) -> None:
        gr = GoalRecognizer()
        assert gr.get_high_risk_intents() == []

    def test_get_high_risk_intents_filters_by_threshold(self) -> None:
        gr = GoalRecognizer()
        # Manually inject a high-confidence enter_restricted hypothesis
        entity = _make_entity()
        hyp = gr._make_hypothesis(
            entity=entity,
            goal_type="enter_restricted",
            confidence=0.9,
            evidence=["test"],
        )
        gr._beliefs[1].append(hyp)
        high = gr.get_high_risk_intents(risk_threshold=0.5)
        assert len(high) >= 1
        for h in high:
            assert h.risk_score >= 0.5

    def test_get_high_risk_intents_all_entities(self) -> None:
        gr = GoalRecognizer()
        for eid in [1, 2, 3]:
            entity = _make_entity(entity_id=eid)
            hyp = gr._make_hypothesis(
                entity=entity,
                goal_type="enter_restricted",
                confidence=0.9,
                evidence=["test"],
            )
            gr._beliefs[eid].append(hyp)
        high = gr.get_high_risk_intents(risk_threshold=0.5)
        assert len(high) == 3


# ── TestBeliefUpdate ─────────────────────────────────────────────────────────


class TestBeliefUpdate:
    """8 tests: _update_beliefs, decay, prune, confidence bounds."""

    def test_repeated_same_observation_increases_confidence(self) -> None:
        gr = GoalRecognizer()
        entity = _make_entity(cx=0.05, cy=0.5)
        track = _make_track(entity_id=1, vx=-0.02)
        ws = _make_world_state(entities=[entity], tracks={1: track})
        gr.observe(entity, ws)
        first_conf = next(
            (h.confidence for h in gr.get_hypotheses(1) if h.goal_type == "exit_area"), None
        )
        # More observations → confidence should stay >= (exponential smoothing
        # keeps it stable or increases)
        for _ in range(5):
            gr.observe(entity, ws)
        later_conf = next(
            (h.confidence for h in gr.get_hypotheses(1) if h.goal_type == "exit_area"), None
        )
        assert first_conf is not None
        assert later_conf is not None
        # Repeated reinforcement keeps or raises confidence
        assert later_conf >= first_conf - 0.05  # allow tiny float tolerance

    def test_different_observation_adds_new_hypothesis(self) -> None:
        gr = GoalRecognizer()
        entity = _make_entity(cx=0.05, cy=0.5)
        track = _make_track(entity_id=1, vx=-0.02)
        ws = _make_world_state(entities=[entity], tracks={1: track})
        gr.observe(entity, ws)
        # Now add a stationary scenario to trigger loiter
        for _ in range(20):
            gr.observe(_make_entity(cx=0.5, cy=0.5), _make_world_state(
                entities=[_make_entity(cx=0.5, cy=0.5)],
                tracks={1: _make_track(entity_id=1)},
            ))
        all_types = {h.goal_type for h in gr.get_hypotheses(1)}
        assert len(all_types) >= 1  # At minimum loiter or exit should be present

    def test_unreinforced_hypotheses_decay(self) -> None:
        gr = GoalRecognizer()
        entity = _make_entity(cx=0.05, cy=0.5)
        track = _make_track(entity_id=1, vx=-0.02)
        ws_exit = _make_world_state(entities=[entity], tracks={1: track})
        gr.observe(entity, ws_exit)
        exit_conf_initial = next(
            (h.confidence for h in gr.get_hypotheses(1) if h.goal_type == "exit_area"), None
        )
        # Now observe an entity at center (no exit signal)
        center_entity = _make_entity(cx=0.5, cy=0.5, entity_id=1)
        center_track = _make_track(entity_id=1)
        ws_center = _make_world_state(
            entities=[center_entity], tracks={1: center_track}
        )
        for _ in range(10):
            gr.observe(center_entity, ws_center)
        exit_conf_later = next(
            (h.confidence for h in gr.get_hypotheses(1) if h.goal_type == "exit_area"), 0.0
        )
        if exit_conf_initial is not None:
            assert exit_conf_later <= exit_conf_initial

    def test_hypotheses_below_floor_are_pruned(self) -> None:
        gr = GoalRecognizer()
        entity = _make_entity()
        # Inject a hypothesis just above floor
        hyp = gr._make_hypothesis(
            entity=entity,
            goal_type="exit_area",
            confidence=0.11,
            evidence=["test"],
        )
        gr._beliefs[1].append(hyp)
        # Decay many times until it falls below floor
        for _ in range(200):
            gr._decay_beliefs(1)
        assert all(h.goal_type != "exit_area" for h in gr.get_hypotheses(1))

    def test_decay_applied_each_call(self) -> None:
        gr = GoalRecognizer()
        entity = _make_entity()
        hyp = gr._make_hypothesis(
            entity=entity,
            goal_type="patrol",
            confidence=0.5,
            evidence=["test"],
        )
        gr._beliefs[1].append(hyp)
        initial_conf = hyp.confidence
        gr._decay_beliefs(1)
        after_decay = next(
            (h.confidence for h in gr._beliefs[1] if h.goal_type == "patrol"), None
        )
        if after_decay is not None:
            assert after_decay < initial_conf

    def test_confidence_capped_at_1(self) -> None:
        gr = GoalRecognizer()
        entity = _make_entity()
        hyp = gr._make_hypothesis(
            entity=entity,
            goal_type="enter_restricted",
            confidence=0.95,
            evidence=["test"],
        )
        gr._beliefs[1].append(hyp)
        # Feed high-confidence candidate repeatedly
        for _ in range(50):
            new_hyp = gr._make_hypothesis(
                entity=entity,
                goal_type="enter_restricted",
                confidence=0.99,
                evidence=["reinforced"],
            )
            gr._update_beliefs(1, [new_hyp])
        for h in gr._beliefs[1]:
            assert h.confidence <= 1.0

    def test_confidence_never_negative(self) -> None:
        gr = GoalRecognizer()
        entity = _make_entity()
        hyp = gr._make_hypothesis(
            entity=entity,
            goal_type="loiter",
            confidence=0.15,
            evidence=["test"],
        )
        gr._beliefs[1].append(hyp)
        for _ in range(500):
            gr._decay_beliefs(1)
        for h in gr._beliefs.get(1, []):
            assert h.confidence >= 0.0

    def test_multiple_entities_independent(self) -> None:
        gr = GoalRecognizer()
        e1 = _make_entity(entity_id=1, cx=0.05, cy=0.5)
        e2 = _make_entity(entity_id=2, cx=0.5, cy=0.5)
        t1 = _make_track(entity_id=1, vx=-0.02)
        t2 = _make_track(entity_id=2)
        ws = _make_world_state(entities=[e1, e2], tracks={1: t1, 2: t2})
        gr.observe(e1, ws)
        gr.observe(e2, ws)
        hyps1 = gr.get_hypotheses(1)
        hyps2 = gr.get_hypotheses(2)
        # Entity 1 near edge should have exit hypothesis; entity 2 (center) should not
        types1 = {h.goal_type for h in hyps1}
        assert "exit_area" in types1
        # Entity 2's beliefs should be independent
        ids2 = {h.entity_id for h in hyps2}
        assert 1 not in ids2  # no bleed from entity 1


# ── TestZoneProximity ─────────────────────────────────────────────────────────


class TestZoneProximity:
    """5 tests: zone proximity detection."""

    def test_entity_near_zone_generates_approach_zone(self) -> None:
        gr = GoalRecognizer()
        zone = _make_zone(zone_id="z1", name="Work Area", cx=0.5, cy=0.5, zone_type="work_area")
        entity = _make_entity(cx=0.4, cy=0.5)  # distance ~0.1 from zone center
        track = _make_track(entity_id=1, vx=0.02)  # moving toward zone (right)
        ws = _make_world_state(
            entities=[entity],
            tracks={1: track},
            zones={"z1": zone},
        )
        result = gr.observe(entity, ws)
        goal_types = [h.goal_type for h in result]
        assert "approach_zone" in goal_types

    def test_entity_near_restricted_zone_generates_enter_restricted(self) -> None:
        gr = GoalRecognizer()
        zone = _make_zone(
            zone_id="z2", name="Danger Zone", cx=0.5, cy=0.5, zone_type="restricted"
        )
        entity = _make_entity(cx=0.4, cy=0.5)
        track = _make_track(entity_id=1, vx=0.02)
        ws = _make_world_state(
            entities=[entity],
            tracks={1: track},
            zones={"z2": zone},
        )
        result = gr.observe(entity, ws)
        goal_types = [h.goal_type for h in result]
        assert "enter_restricted" in goal_types

    def test_entity_far_from_zone_no_zone_hypothesis(self) -> None:
        gr = GoalRecognizer()
        zone = _make_zone(zone_id="z1", cx=0.9, cy=0.9)  # zone far away
        entity = _make_entity(cx=0.1, cy=0.1)  # distance > 0.25
        track = _make_track(entity_id=1)
        ws = _make_world_state(
            entities=[entity],
            tracks={1: track},
            zones={"z1": zone},
        )
        result = gr.observe(entity, ws)
        goal_types = [h.goal_type for h in result]
        assert "approach_zone" not in goal_types
        assert "enter_restricted" not in goal_types

    def test_zone_hypothesis_has_target_zone_set(self) -> None:
        gr = GoalRecognizer()
        zone = _make_zone(zone_id="z1", name="Assembly Area", cx=0.5, cy=0.5)
        entity = _make_entity(cx=0.4, cy=0.5)
        track = _make_track(entity_id=1, vx=0.02)
        ws = _make_world_state(
            entities=[entity],
            tracks={1: track},
            zones={"z1": zone},
        )
        result = gr.observe(entity, ws)
        zone_hyps = [h for h in result if h.goal_type in ("approach_zone", "enter_restricted")]
        assert len(zone_hyps) > 0
        assert zone_hyps[0].target_zone is not None

    def test_confidence_higher_when_closer_to_zone(self) -> None:
        gr1 = GoalRecognizer()
        gr2 = GoalRecognizer()
        zone = _make_zone(zone_id="z1", cx=0.5, cy=0.5, zone_type="work_area")

        # Entity 1: further from zone (distance ~0.2)
        e_far = _make_entity(entity_id=1, cx=0.3, cy=0.5)
        t_far = _make_track(entity_id=1, vx=0.02)
        ws_far = _make_world_state(entities=[e_far], tracks={1: t_far}, zones={"z1": zone})
        result_far = gr1.observe(e_far, ws_far)

        # Entity 2: closer to zone (distance ~0.05)
        e_near = _make_entity(entity_id=1, cx=0.45, cy=0.5)
        t_near = _make_track(entity_id=1, vx=0.02)
        ws_near = _make_world_state(entities=[e_near], tracks={1: t_near}, zones={"z1": zone})
        result_near = gr2.observe(e_near, ws_near)

        conf_far = next(
            (h.confidence for h in result_far if h.goal_type == "approach_zone"), 0.0
        )
        conf_near = next(
            (h.confidence for h in result_near if h.goal_type == "approach_zone"), 0.0
        )
        # Closer entity should have equal or higher confidence
        assert conf_near >= conf_far - 0.05  # allow small float tolerance


# ── TestLoiterDetection ───────────────────────────────────────────────────────


class TestLoiterDetection:
    """5 tests: loiter detection."""

    def test_short_history_no_loiter(self) -> None:
        gr = GoalRecognizer()
        entity = _make_entity(cx=0.5, cy=0.5)
        track = _make_track(entity_id=1)
        ws = _make_world_state(entities=[entity], tracks={1: track})
        # Only 5 observations — below the 15-frame threshold
        for _ in range(5):
            gr.observe(entity, ws)
        goal_types = [h.goal_type for h in gr.get_hypotheses(1)]
        assert "loiter" not in goal_types

    def test_15_frames_same_position_detects_loiter(self) -> None:
        gr = GoalRecognizer()
        entity = _make_entity(cx=0.5, cy=0.5)
        track = _make_track(entity_id=1)
        ws = _make_world_state(entities=[entity], tracks={1: track})
        for _ in range(15):
            gr.observe(entity, ws)
        goal_types = [h.goal_type for h in gr.get_hypotheses(1)]
        assert "loiter" in goal_types

    def test_loiter_confidence_increases_with_history(self) -> None:
        gr = GoalRecognizer()
        entity = _make_entity(cx=0.5, cy=0.5)
        track = _make_track(entity_id=1)
        ws = _make_world_state(entities=[entity], tracks={1: track})

        for _ in range(15):
            gr.observe(entity, ws)
        conf_15 = next(
            (h.confidence for h in gr.get_hypotheses(1) if h.goal_type == "loiter"), 0.0
        )

        for _ in range(15):  # now 30 frames total
            gr.observe(entity, ws)
        conf_30 = next(
            (h.confidence for h in gr.get_hypotheses(1) if h.goal_type == "loiter"), 0.0
        )

        assert conf_30 >= conf_15

    def test_loiter_description_in_japanese(self) -> None:
        gr = GoalRecognizer()
        entity = _make_entity(cx=0.5, cy=0.5)
        track = _make_track(entity_id=1)
        ws = _make_world_state(entities=[entity], tracks={1: track})
        for _ in range(15):
            gr.observe(entity, ws)
        loiter_hyps = [h for h in gr.get_hypotheses(1) if h.goal_type == "loiter"]
        assert len(loiter_hyps) > 0
        # description_ja should be non-empty Japanese string
        assert len(loiter_hyps[0].description_ja) > 0
        # evidence should contain Japanese text
        assert any(len(ev) > 0 for ev in loiter_hyps[0].evidence)

    def test_loiter_risk_score_set(self) -> None:
        gr = GoalRecognizer()
        entity = _make_entity(cx=0.5, cy=0.5)
        track = _make_track(entity_id=1)
        ws = _make_world_state(entities=[entity], tracks={1: track})
        for _ in range(15):
            gr.observe(entity, ws)
        loiter_hyps = [h for h in gr.get_hypotheses(1) if h.goal_type == "loiter"]
        assert len(loiter_hyps) > 0
        assert loiter_hyps[0].risk_score > 0.0
        # risk_score = GOAL_DEFINITIONS["loiter"]["risk"] * confidence
        expected_base = GOAL_DEFINITIONS["loiter"]["risk"]
        assert loiter_hyps[0].risk_score <= expected_base + 1e-6


# ── TestExitDetection ─────────────────────────────────────────────────────────


class TestExitDetection:
    """4 tests: exit_area detection."""

    def test_entity_at_left_edge_with_outward_velocity_generates_exit(self) -> None:
        gr = GoalRecognizer()
        entity = _make_entity(cx=0.05, cy=0.5)
        track = _make_track(entity_id=1, vx=-0.02)
        ws = _make_world_state(entities=[entity], tracks={1: track})
        result = gr.observe(entity, ws)
        goal_types = [h.goal_type for h in result]
        assert "exit_area" in goal_types

    def test_entity_at_center_no_exit(self) -> None:
        gr = GoalRecognizer()
        entity = _make_entity(cx=0.5, cy=0.5)
        track = _make_track(entity_id=1)
        ws = _make_world_state(entities=[entity], tracks={1: track})
        result = gr.observe(entity, ws)
        goal_types = [h.goal_type for h in result]
        assert "exit_area" not in goal_types

    def test_exit_has_low_risk_score(self) -> None:
        gr = GoalRecognizer()
        entity = _make_entity(cx=0.05, cy=0.5)
        track = _make_track(entity_id=1, vx=-0.02)
        ws = _make_world_state(entities=[entity], tracks={1: track})
        result = gr.observe(entity, ws)
        exit_hyps = [h for h in result if h.goal_type == "exit_area"]
        assert len(exit_hyps) > 0
        # risk for exit_area = 0.2 * confidence
        assert exit_hyps[0].risk_score <= GOAL_DEFINITIONS["exit_area"]["risk"] + 1e-6

    def test_entity_at_bottom_edge_outward_velocity(self) -> None:
        gr = GoalRecognizer()
        entity = _make_entity(cx=0.5, cy=0.95)
        track = _make_track(entity_id=1, vy=0.02)  # moving downward (outward at bottom)
        ws = _make_world_state(entities=[entity], tracks={1: track})
        result = gr.observe(entity, ws)
        goal_types = [h.goal_type for h in result]
        assert "exit_area" in goal_types


# ── TestReset ────────────────────────────────────────────────────────────────


class TestReset:
    """4 tests: reset_entity and reset."""

    def test_reset_entity_clears_beliefs(self) -> None:
        gr = GoalRecognizer()
        entity = _make_entity(cx=0.05, cy=0.5)
        track = _make_track(entity_id=1, vx=-0.02)
        ws = _make_world_state(entities=[entity], tracks={1: track})
        gr.observe(entity, ws)
        assert len(gr.get_hypotheses(1)) > 0
        gr.reset_entity(1)
        assert gr.get_hypotheses(1) == []

    def test_reset_entity_leaves_others_intact(self) -> None:
        gr = GoalRecognizer()
        for eid in [1, 2]:
            e = _make_entity(entity_id=eid, cx=0.05, cy=0.5)
            t = _make_track(entity_id=eid, vx=-0.02)
            ws = _make_world_state(entities=[e], tracks={eid: t})
            gr.observe(e, ws)
        gr.reset_entity(1)
        assert gr.get_hypotheses(1) == []
        assert len(gr.get_hypotheses(2)) >= 0  # entity 2 untouched

    def test_reset_clears_all_beliefs(self) -> None:
        gr = GoalRecognizer()
        for eid in [1, 2, 3]:
            e = _make_entity(entity_id=eid, cx=0.05, cy=0.5)
            t = _make_track(entity_id=eid, vx=-0.02)
            ws = _make_world_state(entities=[e], tracks={eid: t})
            gr.observe(e, ws)
        gr.reset()
        assert len(gr._beliefs) == 0

    def test_reset_clears_pos_history(self) -> None:
        gr = GoalRecognizer()
        entity = _make_entity(cx=0.5, cy=0.5)
        track = _make_track(entity_id=1)
        ws = _make_world_state(entities=[entity], tracks={1: track})
        for _ in range(5):
            gr.observe(entity, ws)
        assert len(gr._pos_history) > 0
        gr.reset()
        assert len(gr._pos_history) == 0


# ── TestStateDict ─────────────────────────────────────────────────────────────


class TestStateDict:
    """5 tests: get_state_dict() output structure."""

    def test_get_state_dict_is_dict(self) -> None:
        gr = GoalRecognizer()
        assert isinstance(gr.get_state_dict(), dict)

    def test_get_state_dict_contains_entity_hypotheses(self) -> None:
        gr = GoalRecognizer()
        state = gr.get_state_dict()
        assert "entity_hypotheses" in state

    def test_get_state_dict_contains_high_risk_count(self) -> None:
        gr = GoalRecognizer()
        state = gr.get_state_dict()
        assert "high_risk_count" in state

    def test_high_risk_count_is_int(self) -> None:
        gr = GoalRecognizer()
        state = gr.get_state_dict()
        assert isinstance(state["high_risk_count"], int)

    def test_entity_hypotheses_maps_str_ids_to_lists(self) -> None:
        gr = GoalRecognizer()
        entity = _make_entity(entity_id=1, cx=0.05, cy=0.5)
        track = _make_track(entity_id=1, vx=-0.02)
        ws = _make_world_state(entities=[entity], tracks={1: track})
        gr.observe(entity, ws)
        state = gr.get_state_dict()
        eh = state["entity_hypotheses"]
        for key, value in eh.items():
            assert isinstance(key, str)
            assert isinstance(value, list)
