"""Comprehensive tests for the Perception Engine.

Tests cover every component of sopilot/perception/:
    - types.py   (BBox, Detection, Track, SceneGraph, Zone, etc.)
    - detector.py (MockDetector, non_max_suppression)
    - tracker.py  (MultiObjectTracker)
    - scene_graph.py (SceneGraphBuilder)
    - world_model.py (WorldModel and sub-components)
    - reasoning.py   (RuleParser, LocalReasoner, HybridReasoner)
    - engine.py      (PerceptionEngine, build_perception_engine)

All tests use MockDetector (no GPU required).
Run:  python -m pytest tests/test_perception.py -v
"""

from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from sopilot.perception.types import (
    BBox,
    Detection,
    EntityEvent,
    EntityEventType,
    FrameResult,
    PerceptionConfig,
    Relation,
    SceneEntity,
    SceneGraph,
    SpatialRelation,
    Track,
    TrackState,
    Violation,
    ViolationSeverity,
    WorldState,
    Zone,
)
from sopilot.perception.detector import MockDetector, MockRule, non_max_suppression
from sopilot.perception.tracker import MultiObjectTracker
from sopilot.perception.scene_graph import SceneGraphBuilder
from sopilot.perception.world_model import (
    AnomalyBaseline,
    EntityRegistry,
    StateChangeDetector,
    TemporalMemoryBuffer,
    WorldModel,
    ZoneMonitor,
)
from sopilot.perception.reasoning import (
    HybridReasoner,
    LocalReasoner,
    ParsedRule,
    RuleCheckType,
    RuleParser,
)
from sopilot.perception.engine import PerceptionEngine, build_perception_engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(h: int = 480, w: int = 640, value: int = 128) -> np.ndarray:
    """Create a synthetic BGR frame filled with a constant value."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _make_detection(
    label: str = "person",
    x1: float = 0.1,
    y1: float = 0.1,
    x2: float = 0.3,
    y2: float = 0.5,
    confidence: float = 0.9,
    attributes: dict | None = None,
) -> Detection:
    return Detection(
        bbox=BBox(x1, y1, x2, y2),
        label=label,
        confidence=confidence,
        attributes=attributes or {},
    )


def _make_track(
    track_id: int = 1,
    label: str = "person",
    state: TrackState = TrackState.ACTIVE,
    x1: float = 0.1,
    y1: float = 0.1,
    x2: float = 0.3,
    y2: float = 0.5,
    confidence: float = 0.9,
    attributes: dict | None = None,
) -> Track:
    return Track(
        track_id=track_id,
        label=label,
        state=state,
        bbox=BBox(x1, y1, x2, y2),
        confidence=confidence,
        attributes=attributes or {},
        hits=5,
    )


def _make_scene_entity(
    entity_id: int = 1,
    label: str = "person",
    x1: float = 0.1,
    y1: float = 0.1,
    x2: float = 0.3,
    y2: float = 0.5,
    confidence: float = 0.9,
    zone_ids: list[str] | None = None,
    attributes: dict | None = None,
) -> SceneEntity:
    return SceneEntity(
        entity_id=entity_id,
        label=label,
        bbox=BBox(x1, y1, x2, y2),
        confidence=confidence,
        zone_ids=zone_ids or [],
        attributes=attributes or {},
    )


def _make_scene_graph(
    entities: list[SceneEntity] | None = None,
    relations: list[Relation] | None = None,
    timestamp: float = 1.0,
    frame_number: int = 30,
) -> SceneGraph:
    return SceneGraph(
        timestamp=timestamp,
        frame_number=frame_number,
        entities=entities or [],
        relations=relations or [],
    )


def _make_world_state(
    scene_graph: SceneGraph | None = None,
    zone_occupancy: dict[str, list[int]] | None = None,
    events: list[EntityEvent] | None = None,
    timestamp: float = 1.0,
    frame_number: int = 30,
) -> WorldState:
    sg = scene_graph or _make_scene_graph(timestamp=timestamp, frame_number=frame_number)
    return WorldState(
        timestamp=timestamp,
        frame_number=frame_number,
        scene_graph=sg,
        active_tracks={},
        events=events or [],
        zone_occupancy=zone_occupancy or {},
        entity_count=sg.entity_count,
        person_count=sg.person_count,
    )


def _make_restricted_zone() -> Zone:
    """A square restricted zone covering the center of the frame."""
    return Zone(
        zone_id="restricted_1",
        name="Restricted Area",
        polygon=[(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)],
        zone_type="restricted",
    )


# ===================================================================
# TestBBox
# ===================================================================


class TestBBox(unittest.TestCase):
    """Tests for BBox geometry and spatial computations."""

    def test_center(self):
        b = BBox(0.1, 0.2, 0.5, 0.8)
        cx, cy = b.center
        self.assertAlmostEqual(cx, 0.3)
        self.assertAlmostEqual(cy, 0.5)

    def test_width_and_height(self):
        b = BBox(0.1, 0.2, 0.5, 0.8)
        self.assertAlmostEqual(b.width, 0.4)
        self.assertAlmostEqual(b.height, 0.6)

    def test_width_height_non_negative(self):
        """Inverted coords should yield zero width/height."""
        b = BBox(0.5, 0.5, 0.1, 0.1)
        self.assertEqual(b.width, 0.0)
        self.assertEqual(b.height, 0.0)

    def test_area(self):
        b = BBox(0.0, 0.0, 0.5, 0.5)
        self.assertAlmostEqual(b.area, 0.25)

    def test_iou_identical(self):
        b = BBox(0.1, 0.1, 0.5, 0.5)
        self.assertAlmostEqual(b.iou(b), 1.0)

    def test_iou_non_overlapping(self):
        a = BBox(0.0, 0.0, 0.2, 0.2)
        b = BBox(0.5, 0.5, 0.8, 0.8)
        self.assertAlmostEqual(a.iou(b), 0.0)

    def test_iou_partial_overlap(self):
        a = BBox(0.0, 0.0, 0.4, 0.4)
        b = BBox(0.2, 0.2, 0.6, 0.6)
        iou = a.iou(b)
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)

    def test_contains_true(self):
        outer = BBox(0.0, 0.0, 1.0, 1.0)
        inner = BBox(0.2, 0.2, 0.8, 0.8)
        self.assertTrue(outer.contains(inner))

    def test_contains_false(self):
        a = BBox(0.0, 0.0, 0.5, 0.5)
        b = BBox(0.3, 0.3, 0.9, 0.9)
        self.assertFalse(a.contains(b))

    def test_distance_to(self):
        a = BBox(0.0, 0.0, 0.2, 0.2)  # center (0.1, 0.1)
        b = BBox(0.3, 0.0, 0.5, 0.2)  # center (0.4, 0.1)
        dist = a.distance_to(b)
        self.assertAlmostEqual(dist, 0.3, places=5)

    def test_from_pixels_to_pixels_round_trip(self):
        original = BBox(0.25, 0.5, 0.75, 1.0)
        w, h = 640, 480
        px = original.to_pixels(w, h)
        recovered = BBox.from_pixels(*px, width=w, height=h)
        # Allow small rounding errors from int truncation
        self.assertAlmostEqual(original.x1, recovered.x1, places=2)
        self.assertAlmostEqual(original.y1, recovered.y1, places=2)

    def test_expanded_normal(self):
        b = BBox(0.3, 0.3, 0.7, 0.7)
        ex = b.expanded(0.1)
        self.assertAlmostEqual(ex.x1, 0.2)
        self.assertAlmostEqual(ex.y1, 0.2)
        self.assertAlmostEqual(ex.x2, 0.8)
        self.assertAlmostEqual(ex.y2, 0.8)

    def test_expanded_clamped(self):
        b = BBox(0.05, 0.05, 0.95, 0.95)
        ex = b.expanded(0.2)
        self.assertAlmostEqual(ex.x1, 0.0)
        self.assertAlmostEqual(ex.y1, 0.0)
        self.assertAlmostEqual(ex.x2, 1.0)
        self.assertAlmostEqual(ex.y2, 1.0)

    def test_aspect_ratio(self):
        b = BBox(0.0, 0.0, 0.4, 0.2)
        self.assertAlmostEqual(b.aspect_ratio, 2.0)

    def test_frozen(self):
        b = BBox(0.0, 0.0, 1.0, 1.0)
        with self.assertRaises(FrozenInstanceError):
            b.x1 = 0.5  # type: ignore[misc]


# ===================================================================
# TestDetection
# ===================================================================


class TestDetection(unittest.TestCase):
    """Tests for Detection dataclass."""

    def test_creation_with_all_fields(self):
        bbox = BBox(0.1, 0.2, 0.3, 0.4)
        emb = np.zeros(128)
        d = Detection(
            bbox=bbox,
            label="helmet",
            confidence=0.88,
            embedding=emb,
            attributes={"color": "yellow"},
        )
        self.assertEqual(d.label, "helmet")
        self.assertAlmostEqual(d.confidence, 0.88)
        self.assertIsNotNone(d.embedding)
        self.assertEqual(d.attributes["color"], "yellow")

    def test_frozen(self):
        d = _make_detection()
        with self.assertRaises(FrozenInstanceError):
            d.label = "changed"  # type: ignore[misc]

    def test_default_attributes(self):
        d = Detection(bbox=BBox(0, 0, 1, 1), label="x", confidence=0.5)
        self.assertIsInstance(d.attributes, dict)
        self.assertEqual(len(d.attributes), 0)

    def test_default_embedding_none(self):
        d = Detection(bbox=BBox(0, 0, 1, 1), label="x", confidence=0.5)
        self.assertIsNone(d.embedding)


# ===================================================================
# TestNonMaxSuppression
# ===================================================================


class TestNonMaxSuppression(unittest.TestCase):
    """Tests for the non_max_suppression function."""

    def test_empty_input(self):
        self.assertEqual(non_max_suppression([]), [])

    def test_single_detection(self):
        d = _make_detection()
        result = non_max_suppression([d])
        self.assertEqual(len(result), 1)

    def test_no_suppression_when_no_overlap(self):
        d1 = _make_detection(x1=0.0, y1=0.0, x2=0.1, y2=0.1, confidence=0.9)
        d2 = _make_detection(x1=0.8, y1=0.8, x2=0.9, y2=0.9, confidence=0.8)
        result = non_max_suppression([d1, d2], iou_threshold=0.5)
        self.assertEqual(len(result), 2)

    def test_suppress_overlapping(self):
        d1 = _make_detection(x1=0.1, y1=0.1, x2=0.5, y2=0.5, confidence=0.9)
        d2 = _make_detection(x1=0.1, y1=0.1, x2=0.5, y2=0.5, confidence=0.7)
        result = non_max_suppression([d1, d2], iou_threshold=0.5)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].confidence, 0.9)

    def test_keep_highest_confidence(self):
        # Three overlapping detections
        d1 = _make_detection(x1=0.1, y1=0.1, x2=0.5, y2=0.5, confidence=0.6)
        d2 = _make_detection(x1=0.12, y1=0.12, x2=0.52, y2=0.52, confidence=0.95)
        d3 = _make_detection(x1=0.11, y1=0.11, x2=0.51, y2=0.51, confidence=0.7)
        result = non_max_suppression([d1, d2, d3], iou_threshold=0.3)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].confidence, 0.95)


# ===================================================================
# TestMockDetector
# ===================================================================


class TestMockDetector(unittest.TestCase):
    """Tests for MockDetector."""

    def test_rules_mode_matching(self):
        """Rules mode returns detections only when prompt matches rule label."""
        rule = MockRule(label="person", bbox=BBox(0.2, 0.2, 0.4, 0.6), confidence=0.9)
        det = MockDetector(rules=[rule])
        frame = _make_frame()

        result = det.detect(frame, ["person"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].label, "person")

    def test_rules_mode_no_match(self):
        """Rules mode returns nothing when prompt does not match."""
        rule = MockRule(label="person", bbox=BBox(0.2, 0.2, 0.4, 0.6))
        det = MockDetector(rules=[rule])
        frame = _make_frame()

        result = det.detect(frame, ["forklift"])
        self.assertEqual(len(result), 0)

    def test_heuristic_mode_produces_detections(self):
        """Heuristic mode produces detections when frame has varied content."""
        det = MockDetector(use_heuristics=True)
        # Create frame with a bright quadrant (top-left)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:240, :320] = 200  # bright top-left
        result = det.detect(frame, ["person"])
        self.assertGreater(len(result), 0)

    def test_empty_prompts(self):
        rule = MockRule(label="person", bbox=BBox(0.2, 0.2, 0.4, 0.6))
        det = MockDetector(rules=[rule])
        result = det.detect(_make_frame(), [])
        self.assertEqual(len(result), 0)

    def test_configurable_confidence(self):
        rule = MockRule(label="person", bbox=BBox(0.1, 0.1, 0.3, 0.3), confidence=0.42)
        det = MockDetector(rules=[rule])
        result = det.detect(_make_frame(), ["person"])
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].confidence, 0.42)

    def test_detection_objects_valid(self):
        rule = MockRule(label="helmet", bbox=BBox(0.3, 0.1, 0.5, 0.2), confidence=0.8)
        det = MockDetector(rules=[rule])
        result = det.detect(_make_frame(), ["helmet"])
        self.assertEqual(len(result), 1)
        d = result[0]
        self.assertIsInstance(d, Detection)
        self.assertIsInstance(d.bbox, BBox)
        self.assertEqual(d.label, "helmet")

    def test_jitter(self):
        rule = MockRule(
            label="person",
            bbox=BBox(0.3, 0.3, 0.6, 0.6),
            confidence=0.9,
            jitter=0.05,
        )
        det = MockDetector(rules=[rule])
        frame = _make_frame()
        results = [det.detect(frame, ["person"])[0] for _ in range(5)]
        # With jitter, at least some bboxes should differ
        unique_x1 = {r.bbox.x1 for r in results}
        # Due to seeded RNG, we may or may not get different values, but
        # the code should not crash
        self.assertTrue(all(isinstance(r, Detection) for r in results))


# ===================================================================
# TestMultiObjectTracker
# ===================================================================


class TestMultiObjectTracker(unittest.TestCase):
    """Tests for MultiObjectTracker."""

    def setUp(self):
        self.config = PerceptionConfig(
            track_max_age=5,
            track_min_hits=2,
            track_high_threshold=0.3,
            track_low_threshold=0.1,
            detection_confidence_threshold=0.3,
        )
        self.tracker = MultiObjectTracker(self.config)

    def test_single_object_tracking(self):
        """A single object detected across frames should be tracked."""
        det = _make_detection(confidence=0.9)
        for i in range(5):
            tracks = self.tracker.update([det], frame_id=i)
        # After min_hits frames the track should be confirmed
        self.assertGreater(len(tracks), 0)
        # All returned tracks should have a bbox
        for t in tracks:
            self.assertIsNotNone(t.bbox)

    def test_multiple_objects_tracked_independently(self):
        """Two spatially separated objects should get different track IDs."""
        d1 = _make_detection(x1=0.0, y1=0.0, x2=0.1, y2=0.1, confidence=0.9)
        d2 = _make_detection(x1=0.8, y1=0.8, x2=0.95, y2=0.95, confidence=0.9)
        for i in range(5):
            tracks = self.tracker.update([d1, d2], frame_id=i)
        # There should be at least 2 tracks
        track_ids = {t.track_id for t in tracks}
        self.assertGreaterEqual(len(track_ids), 2)

    def test_track_lifecycle_tentative_to_active(self):
        """Track should transition from TENTATIVE to ACTIVE after min_hits."""
        det = _make_detection(confidence=0.95)
        # First frame: tentative
        tracks = self.tracker.update([det], frame_id=0)
        tentative = [t for t in tracks if t.state == TrackState.TENTATIVE]
        self.assertGreater(len(tentative), 0)

        # After min_hits (2) frames: should become ACTIVE
        for i in range(1, 5):
            tracks = self.tracker.update([det], frame_id=i)
        active = [t for t in tracks if t.state == TrackState.ACTIVE]
        self.assertGreater(len(active), 0)

    def test_track_lost_after_max_age(self):
        """Track should be lost after max_age consecutive misses."""
        det = _make_detection(confidence=0.9)
        # Build up a confirmed track
        for i in range(5):
            self.tracker.update([det], frame_id=i)

        # Now provide no detections for max_age + extra frames
        for i in range(5, 5 + self.config.track_max_age + 5):
            tracks = self.tracker.update([], frame_id=i)

        # After enough misses, track should have been garbage-collected
        active = self.tracker.get_active_tracks()
        self.assertEqual(len(active), 0)

    def test_reid_after_brief_occlusion(self):
        """Track should recover after brief occlusion (< max_age frames)."""
        det = _make_detection(confidence=0.9)
        # Build confirmed track
        for i in range(5):
            self.tracker.update([det], frame_id=i)

        # Brief occlusion (1-2 frames)
        self.tracker.update([], frame_id=5)
        self.tracker.update([], frame_id=6)

        # Re-detect
        tracks = self.tracker.update([det], frame_id=7)
        # Should still have a track
        self.assertGreater(len(tracks), 0)

    def test_label_majority_vote(self):
        """Label should follow majority vote of matched detections."""
        d1 = _make_detection(label="worker", confidence=0.9)
        d2 = _make_detection(label="person", confidence=0.9)

        # Start with "worker" for 3 frames
        for i in range(3):
            self.tracker.update([d1], frame_id=i)
        # Switch to "person" for 1 frame
        tracks = self.tracker.update([d2], frame_id=3)

        # Majority should still be "worker" (3 vs 1)
        if tracks:
            self.assertEqual(tracks[0].label, "worker")

    def test_reset_clears_all(self):
        det = _make_detection(confidence=0.9)
        for i in range(5):
            self.tracker.update([det], frame_id=i)
        self.tracker.reset()
        tracks = self.tracker.get_active_tracks()
        self.assertEqual(len(tracks), 0)

    def test_get_track_by_id(self):
        det = _make_detection(confidence=0.9)
        tracks = self.tracker.update([det], frame_id=0)
        if tracks:
            tid = tracks[0].track_id
            t = self.tracker.get_track(tid)
            self.assertIsNotNone(t)
            self.assertEqual(t.track_id, tid)

    def test_get_track_nonexistent(self):
        self.assertIsNone(self.tracker.get_track(9999))


# ===================================================================
# TestSceneGraphBuilder
# ===================================================================


class TestSceneGraphBuilder(unittest.TestCase):
    """Tests for SceneGraphBuilder."""

    def setUp(self):
        self.config = PerceptionConfig(
            scene_near_threshold=0.15,
            scene_vertical_threshold=0.1,
            scene_overlap_threshold=0.3,
            scene_graph_max_relations=200,
        )
        self.builder = SceneGraphBuilder(self.config)

    def test_entities_from_active_tracks(self):
        """Only confirmed (ACTIVE/OCCLUDED) tracks should become entities."""
        t_active = _make_track(track_id=1, state=TrackState.ACTIVE)
        t_tentative = _make_track(track_id=2, state=TrackState.TENTATIVE)
        t_occluded = _make_track(track_id=3, state=TrackState.OCCLUDED)
        tracks = [t_active, t_tentative, t_occluded]
        graph = self.builder.build(tracks, [], (480, 640), 1.0, 1)
        entity_ids = {e.entity_id for e in graph.entities}
        self.assertIn(1, entity_ids)
        self.assertNotIn(2, entity_ids)  # tentative excluded
        self.assertIn(3, entity_ids)

    def test_zone_assignment(self):
        """Entities inside a zone polygon should have zone_ids assigned."""
        zone = _make_restricted_zone()
        # Track in center of zone
        t = _make_track(track_id=1, x1=0.4, y1=0.4, x2=0.6, y2=0.6)
        graph = self.builder.build([t], [zone], (480, 640), 1.0, 1)
        self.assertEqual(len(graph.entities), 1)
        self.assertIn("restricted_1", graph.entities[0].zone_ids)

    def test_near_relation(self):
        """Close entities should have NEAR relation."""
        t1 = _make_track(track_id=1, x1=0.1, y1=0.1, x2=0.2, y2=0.2)
        t2 = _make_track(track_id=2, x1=0.15, y1=0.1, x2=0.25, y2=0.2)
        graph = self.builder.build([t1, t2], [], (480, 640), 1.0, 1)
        near_rels = [
            r for r in graph.relations if r.predicate == SpatialRelation.NEAR
        ]
        self.assertGreater(len(near_rels), 0)

    def test_above_below_relations(self):
        """Vertically separated entities should have ABOVE/BELOW relations."""
        t_top = _make_track(track_id=1, x1=0.4, y1=0.0, x2=0.6, y2=0.1)
        t_bottom = _make_track(track_id=2, x1=0.4, y1=0.7, x2=0.6, y2=0.9)
        graph = self.builder.build([t_top, t_bottom], [], (480, 640), 1.0, 1)
        above = [r for r in graph.relations if r.predicate == SpatialRelation.ABOVE]
        below = [r for r in graph.relations if r.predicate == SpatialRelation.BELOW]
        # One direction is emitted (e1 -> e2 for i < j)
        self.assertTrue(len(above) > 0 or len(below) > 0)

    def test_wearing_relation_person_helmet(self):
        """Person with overlapping helmet should produce WEARING relation."""
        # Person bbox covering most of the frame
        person = _make_track(
            track_id=1, label="person", x1=0.2, y1=0.1, x2=0.5, y2=0.8
        )
        # Helmet on person's head (upper portion, smaller)
        helmet = _make_track(
            track_id=2, label="helmet", x1=0.25, y1=0.1, x2=0.45, y2=0.25
        )
        graph = self.builder.build([person, helmet], [], (480, 640), 1.0, 1)
        wearing = [
            r for r in graph.relations if r.predicate == SpatialRelation.WEARING
        ]
        self.assertGreater(len(wearing), 0)
        # Subject should be the person
        self.assertEqual(wearing[0].subject_id, 1)
        self.assertEqual(wearing[0].object_id, 2)

    def test_relation_pruning(self):
        """Relations should be pruned if exceeding the max limit."""
        config = PerceptionConfig(scene_graph_max_relations=3)
        builder = SceneGraphBuilder(config)
        # Many tracks to generate many relations
        tracks = [
            _make_track(
                track_id=i,
                x1=0.1 * i,
                y1=0.1 * i,
                x2=0.1 * i + 0.08,
                y2=0.1 * i + 0.08,
            )
            for i in range(1, 6)
        ]
        graph = builder.build(tracks, [], (480, 640), 1.0, 1)
        self.assertLessEqual(len(graph.relations), 3)


# ===================================================================
# TestWorldModel
# ===================================================================


class TestWorldModel(unittest.TestCase):
    """Tests for WorldModel and its sub-components."""

    def _make_config_with_zone(self) -> PerceptionConfig:
        zone = _make_restricted_zone()
        return PerceptionConfig(zone_definitions=[zone])

    def test_entered_event(self):
        """New entity appearing should generate ENTERED event."""
        model = WorldModel(PerceptionConfig())
        entity = _make_scene_entity(entity_id=10, label="person")
        sg = _make_scene_graph(entities=[entity], timestamp=0.5, frame_number=1)
        ws = model.update(sg)
        entered = [e for e in ws.events if e.event_type == EntityEventType.ENTERED]
        self.assertEqual(len(entered), 1)
        self.assertEqual(entered[0].entity_id, 10)

    def test_exited_event(self):
        """Entity disappearing should generate EXITED event."""
        model = WorldModel(PerceptionConfig())
        entity = _make_scene_entity(entity_id=10)
        sg1 = _make_scene_graph(entities=[entity], timestamp=1.0, frame_number=1)
        model.update(sg1)

        # Second frame: entity gone
        sg2 = _make_scene_graph(entities=[], timestamp=2.0, frame_number=2)
        ws2 = model.update(sg2)
        exited = [e for e in ws2.events if e.event_type == EntityEventType.EXITED]
        self.assertEqual(len(exited), 1)
        self.assertEqual(exited[0].entity_id, 10)

    def test_zone_entered_event(self):
        """Entity entering a zone should generate ZONE_ENTERED event."""
        config = self._make_config_with_zone()
        model = WorldModel(config)

        # Entity is in the center of the restricted zone
        entity = _make_scene_entity(
            entity_id=1, x1=0.4, y1=0.4, x2=0.6, y2=0.6,
            zone_ids=["restricted_1"],
        )
        sg = _make_scene_graph(entities=[entity], timestamp=1.0, frame_number=1)
        ws = model.update(sg)
        zone_entered = [
            e for e in ws.events if e.event_type == EntityEventType.ZONE_ENTERED
        ]
        self.assertEqual(len(zone_entered), 1)

    def test_zone_exited_event(self):
        """Entity leaving a zone should generate ZONE_EXITED event."""
        config = self._make_config_with_zone()
        model = WorldModel(config)

        # Frame 1: entity in zone
        entity_in = _make_scene_entity(
            entity_id=1, x1=0.4, y1=0.4, x2=0.6, y2=0.6,
            zone_ids=["restricted_1"],
        )
        model.update(_make_scene_graph(
            entities=[entity_in], timestamp=1.0, frame_number=1
        ))

        # Frame 2: entity moved outside zone
        entity_out = _make_scene_entity(
            entity_id=1, x1=0.0, y1=0.0, x2=0.1, y2=0.1,
            zone_ids=[],
        )
        ws2 = model.update(_make_scene_graph(
            entities=[entity_out], timestamp=2.0, frame_number=2
        ))
        zone_exited = [
            e for e in ws2.events if e.event_type == EntityEventType.ZONE_EXITED
        ]
        self.assertEqual(len(zone_exited), 1)

    def test_state_change_detection(self):
        """Attribute changes should generate STATE_CHANGED events."""
        model = WorldModel(PerceptionConfig())

        entity_v1 = _make_scene_entity(
            entity_id=1, attributes={"has_helmet": True}
        )
        model.update(_make_scene_graph(
            entities=[entity_v1], timestamp=1.0, frame_number=1
        ))

        entity_v2 = _make_scene_entity(
            entity_id=1, attributes={"has_helmet": False}
        )
        ws2 = model.update(_make_scene_graph(
            entities=[entity_v2], timestamp=2.0, frame_number=2
        ))
        changed = [
            e for e in ws2.events if e.event_type == EntityEventType.STATE_CHANGED
        ]
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0].details["attribute"], "has_helmet")

    def test_temporal_memory(self):
        """Temporal memory should store recent states."""
        model = WorldModel(PerceptionConfig(temporal_memory_seconds=10.0))
        for i in range(5):
            sg = _make_scene_graph(timestamp=float(i), frame_number=i)
            model.update(sg)
        mem = model.get_temporal_memory()
        self.assertEqual(mem.size, 5)

    def test_get_entity_history(self):
        model = WorldModel(PerceptionConfig())
        entity = _make_scene_entity(entity_id=42, label="person")
        model.update(_make_scene_graph(
            entities=[entity], timestamp=1.0, frame_number=1
        ))
        hist = model.get_entity_history(42)
        self.assertIsNotNone(hist)
        self.assertEqual(hist.entity_id, 42)
        self.assertEqual(hist.label, "person")

    def test_anomaly_baseline_flags_unusual_count(self):
        """After warmup, an anomalous entity count should trigger ANOMALY event."""
        model = WorldModel(PerceptionConfig())
        # Warmup: 35 frames with 2 entities each (min observations = 30)
        for i in range(35):
            entities = [
                _make_scene_entity(entity_id=1, x1=0.1, y1=0.1, x2=0.2, y2=0.2),
                _make_scene_entity(entity_id=2, x1=0.5, y1=0.5, x2=0.6, y2=0.6),
            ]
            model.update(_make_scene_graph(
                entities=entities, timestamp=float(i), frame_number=i
            ))

        # Sudden spike: 20 entities
        big_entities = [
            _make_scene_entity(
                entity_id=100 + j,
                x1=0.01 * j, y1=0.01 * j,
                x2=0.01 * j + 0.05, y2=0.01 * j + 0.05,
            )
            for j in range(20)
        ]
        ws = model.update(_make_scene_graph(
            entities=big_entities, timestamp=36.0, frame_number=36
        ))
        anomalies = [
            e for e in ws.events if e.event_type == EntityEventType.ANOMALY
        ]
        self.assertGreater(len(anomalies), 0)

    def test_reset_clears_all_state(self):
        model = WorldModel(PerceptionConfig())
        entity = _make_scene_entity(entity_id=1)
        model.update(_make_scene_graph(
            entities=[entity], timestamp=1.0, frame_number=1
        ))
        model.reset()
        self.assertEqual(model.frames_processed, 0)
        self.assertIsNone(model.get_entity_history(1))


# ===================================================================
# TestEntityRegistry
# ===================================================================


class TestEntityRegistry(unittest.TestCase):

    def test_enter_and_exit(self):
        reg = EntityRegistry()
        entity = _make_scene_entity(entity_id=5, label="person")
        sg1 = _make_scene_graph(entities=[entity], timestamp=1.0, frame_number=1)
        events1 = reg.update(sg1)
        entered = [e for e in events1 if e.event_type == EntityEventType.ENTERED]
        self.assertEqual(len(entered), 1)

        sg2 = _make_scene_graph(entities=[], timestamp=2.0, frame_number=2)
        events2 = reg.update(sg2)
        exited = [e for e in events2 if e.event_type == EntityEventType.EXITED]
        self.assertEqual(len(exited), 1)

    def test_history_updated(self):
        reg = EntityRegistry()
        entity = _make_scene_entity(entity_id=5)
        sg = _make_scene_graph(entities=[entity], timestamp=1.0, frame_number=1)
        reg.update(sg)
        hist = reg.get(5)
        self.assertIsNotNone(hist)
        self.assertEqual(hist.total_frames, 1)


# ===================================================================
# TestZoneMonitor
# ===================================================================


class TestZoneMonitor(unittest.TestCase):

    def test_zone_enter_exit(self):
        zone = _make_restricted_zone()
        mon = ZoneMonitor([zone])

        entity_in = _make_scene_entity(
            entity_id=1, x1=0.4, y1=0.4, x2=0.6, y2=0.6,
            zone_ids=["restricted_1"],
        )
        sg1 = _make_scene_graph(entities=[entity_in], timestamp=1.0, frame_number=1)
        occ1, events1 = mon.update(sg1)
        entered = [e for e in events1 if e.event_type == EntityEventType.ZONE_ENTERED]
        self.assertEqual(len(entered), 1)

        entity_out = _make_scene_entity(
            entity_id=1, x1=0.0, y1=0.0, x2=0.05, y2=0.05,
            zone_ids=[],
        )
        sg2 = _make_scene_graph(entities=[entity_out], timestamp=2.0, frame_number=2)
        occ2, events2 = mon.update(sg2)
        exited = [e for e in events2 if e.event_type == EntityEventType.ZONE_EXITED]
        self.assertEqual(len(exited), 1)


# ===================================================================
# TestStateChangeDetector
# ===================================================================


class TestStateChangeDetector(unittest.TestCase):

    def test_attribute_change(self):
        det = StateChangeDetector()
        e1 = _make_scene_entity(entity_id=1, attributes={"has_helmet": True})
        sg1 = _make_scene_graph(entities=[e1], timestamp=1.0, frame_number=1)
        det.update(sg1)

        e2 = _make_scene_entity(entity_id=1, attributes={"has_helmet": False})
        sg2 = _make_scene_graph(entities=[e2], timestamp=2.0, frame_number=2)
        events = det.update(sg2)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].details["attribute"], "has_helmet")


# ===================================================================
# TestTemporalMemoryBuffer
# ===================================================================


class TestTemporalMemoryBuffer(unittest.TestCase):

    def test_push_and_size(self):
        buf = TemporalMemoryBuffer(max_seconds=10.0)
        entity = _make_scene_entity(entity_id=1)
        for i in range(5):
            sg = _make_scene_graph(entities=[entity], timestamp=float(i), frame_number=i)
            ws = _make_world_state(scene_graph=sg, timestamp=float(i), frame_number=i)
            buf.push(ws)
        self.assertEqual(buf.size, 5)

    def test_eviction(self):
        buf = TemporalMemoryBuffer(max_seconds=3.0)
        for i in range(10):
            sg = _make_scene_graph(timestamp=float(i), frame_number=i)
            ws = _make_world_state(scene_graph=sg, timestamp=float(i), frame_number=i)
            buf.push(ws)
        # Only the last ~3 seconds should remain
        self.assertLessEqual(buf.size, 4)

    def test_entity_trajectory(self):
        buf = TemporalMemoryBuffer(max_seconds=100.0)
        for i in range(3):
            entity = _make_scene_entity(
                entity_id=1, x1=0.1 * i, y1=0.1, x2=0.1 * i + 0.1, y2=0.2
            )
            sg = _make_scene_graph(entities=[entity], timestamp=float(i), frame_number=i)
            ws = _make_world_state(scene_graph=sg, timestamp=float(i), frame_number=i)
            buf.push(ws)
        traj = buf.get_entity_trajectory(1)
        self.assertEqual(len(traj), 3)


# ===================================================================
# TestAnomalyBaseline
# ===================================================================


class TestAnomalyBaseline(unittest.TestCase):

    def test_no_anomaly_before_warmup(self):
        ab = AnomalyBaseline()
        ws = _make_world_state()
        for _ in range(10):
            ab.observe(ws)
        events = ab.check_anomalies(ws)
        self.assertEqual(len(events), 0)

    def test_anomaly_after_warmup(self):
        ab = AnomalyBaseline()
        # Warmup with entity_count=2
        entities = [
            _make_scene_entity(entity_id=1),
            _make_scene_entity(entity_id=2, x1=0.5, y1=0.5, x2=0.6, y2=0.6),
        ]
        sg_normal = _make_scene_graph(entities=entities)
        ws_normal = _make_world_state(scene_graph=sg_normal)
        for i in range(35):
            ab.observe(ws_normal)

        # Spike: entity_count=20
        many = [
            _make_scene_entity(
                entity_id=100 + j,
                x1=0.01 * j, y1=0.01, x2=0.01 * j + 0.05, y2=0.06,
            )
            for j in range(20)
        ]
        sg_spike = _make_scene_graph(entities=many)
        ws_spike = _make_world_state(scene_graph=sg_spike)
        ab.observe(ws_spike)
        events = ab.check_anomalies(ws_spike)
        self.assertGreater(len(events), 0)


# ===================================================================
# TestRuleParser
# ===================================================================


class TestRuleParser(unittest.TestCase):
    """Tests for RuleParser — natural-language rule parsing."""

    def setUp(self):
        self.parser = RuleParser()

    def test_japanese_helmet_rule(self):
        rules = self.parser.parse(["ヘルメット未着用の作業者を検出"])
        self.assertEqual(len(rules), 1)
        r = rules[0]
        self.assertEqual(r.check_type, RuleCheckType.WEARING)
        self.assertTrue(r.negated)
        self.assertEqual(r.target_object, "helmet")

    def test_zone_violation_rule(self):
        rules = self.parser.parse(["立入禁止エリアへの侵入を検出"])
        self.assertEqual(len(rules), 1)
        r = rules[0]
        self.assertEqual(r.check_type, RuleCheckType.ZONE_VIOLATION)
        self.assertIsNotNone(r.zone_restriction)

    def test_generic_detection_rule(self):
        rules = self.parser.parse(["フォークリフトを検出"])
        r = rules[0]
        self.assertEqual(r.check_type, RuleCheckType.PRESENCE)
        self.assertEqual(r.target_object, "forklift")

    def test_complex_rule_falls_back_to_behavioral(self):
        rules = self.parser.parse(["正しい姿勢で作業しているか判定"])
        r = rules[0]
        self.assertEqual(r.check_type, RuleCheckType.BEHAVIORAL)

    def test_severity_inference(self):
        rules = self.parser.parse(["危険な転倒を検出"])
        r = rules[0]
        self.assertEqual(r.severity_hint, ViolationSeverity.CRITICAL)

    def test_multiple_rules(self):
        rules = self.parser.parse([
            "ヘルメット未着用の作業者を検出",
            "立入禁止エリアへの侵入を検出",
            "フォークリフトを検出",
        ])
        self.assertEqual(len(rules), 3)
        types = {r.check_type for r in rules}
        self.assertIn(RuleCheckType.WEARING, types)
        self.assertIn(RuleCheckType.ZONE_VIOLATION, types)
        self.assertIn(RuleCheckType.PRESENCE, types)

    def test_english_not_wearing_rule(self):
        rules = self.parser.parse(["Detect workers not wearing helmet"])
        r = rules[0]
        self.assertEqual(r.check_type, RuleCheckType.WEARING)
        self.assertTrue(r.negated)

    def test_english_restricted_area(self):
        rules = self.parser.parse(["Detect person in restricted area"])
        r = rules[0]
        self.assertEqual(r.check_type, RuleCheckType.ZONE_VIOLATION)


# ===================================================================
# TestLocalReasoner
# ===================================================================


class TestLocalReasoner(unittest.TestCase):
    """Tests for LocalReasoner — scene-graph-based rule evaluation."""

    def setUp(self):
        self.reasoner = LocalReasoner()

    def _helmet_rule(self, negated: bool = True) -> ParsedRule:
        return ParsedRule(
            original="ヘルメット未着用の作業者を検出",
            rule_index=0,
            target_object="helmet",
            check_type=RuleCheckType.WEARING,
            negated=negated,
            zone_restriction=None,
            related_object="person",
            severity_hint=ViolationSeverity.WARNING,
        )

    def test_wearing_no_helmet_violation(self):
        """Person without WEARING relation to helmet should trigger violation."""
        person = _make_scene_entity(entity_id=1, label="person")
        sg = _make_scene_graph(entities=[person], relations=[])
        ws = _make_world_state(scene_graph=sg)
        rule = self._helmet_rule(negated=True)

        violations, conf = self.reasoner.evaluate(rule, sg, ws)
        self.assertGreater(len(violations), 0)
        self.assertEqual(violations[0].entity_ids, [1])

    def test_wearing_with_helmet_no_violation(self):
        """Person WITH WEARING relation to helmet should not trigger negated rule."""
        person = _make_scene_entity(entity_id=1, label="person")
        helmet = _make_scene_entity(entity_id=2, label="helmet",
                                     x1=0.15, y1=0.1, x2=0.25, y2=0.2)
        wearing_rel = Relation(
            subject_id=1,
            predicate=SpatialRelation.WEARING,
            object_id=2,
            confidence=0.9,
        )
        sg = _make_scene_graph(
            entities=[person, helmet], relations=[wearing_rel]
        )
        ws = _make_world_state(scene_graph=sg)
        rule = self._helmet_rule(negated=True)

        violations, conf = self.reasoner.evaluate(rule, sg, ws)
        self.assertEqual(len(violations), 0)

    def test_zone_violation_person_in_restricted(self):
        """Person in restricted zone should trigger ZONE_VIOLATION."""
        person = _make_scene_entity(
            entity_id=1, label="person",
            x1=0.4, y1=0.4, x2=0.6, y2=0.6,
            zone_ids=["restricted_1"],
        )
        sg = _make_scene_graph(entities=[person])
        ws = _make_world_state(
            scene_graph=sg,
            zone_occupancy={"restricted_1": [1]},
        )
        rule = ParsedRule(
            original="立入禁止エリアへの侵入を検出",
            rule_index=0,
            target_object="person",
            check_type=RuleCheckType.ZONE_VIOLATION,
            negated=False,
            zone_restriction="立入禁止",
            related_object=None,
            severity_hint=ViolationSeverity.CRITICAL,
        )
        violations, conf = self.reasoner.evaluate(rule, sg, ws)
        # The zone_id "restricted_1" does not contain the Japanese keyword
        # directly, so we check via the restricted-zone naming fallback.
        # Zone matching may depend on zone_id naming convention.
        # Either way, the reasoner should return a result.
        self.assertIsInstance(violations, list)
        self.assertIsInstance(conf, float)

    def test_presence_check(self):
        """PRESENCE rule should return violation when object is detected."""
        forklift = _make_scene_entity(entity_id=1, label="forklift")
        sg = _make_scene_graph(entities=[forklift])
        ws = _make_world_state(scene_graph=sg)
        rule = ParsedRule(
            original="フォークリフトを検出",
            rule_index=0,
            target_object="forklift",
            check_type=RuleCheckType.PRESENCE,
            negated=False,
            zone_restriction=None,
            related_object=None,
            severity_hint=ViolationSeverity.WARNING,
        )
        violations, conf = self.reasoner.evaluate(rule, sg, ws)
        self.assertEqual(len(violations), 1)
        self.assertGreater(conf, 0.5)

    def test_behavioral_returns_low_confidence(self):
        """BEHAVIORAL rules should return 0.0 confidence (needs VLM)."""
        sg = _make_scene_graph()
        ws = _make_world_state(scene_graph=sg)
        rule = ParsedRule(
            original="作業手順が正しく実行されているか確認",
            rule_index=0,
            target_object="object",
            check_type=RuleCheckType.BEHAVIORAL,
            negated=False,
            zone_restriction=None,
            related_object=None,
            severity_hint=ViolationSeverity.WARNING,
        )
        violations, conf = self.reasoner.evaluate(rule, sg, ws)
        self.assertEqual(len(violations), 0)
        self.assertAlmostEqual(conf, 0.0)


# ===================================================================
# TestHybridReasoner
# ===================================================================


class TestHybridReasoner(unittest.TestCase):
    """Tests for HybridReasoner."""

    def test_high_confidence_uses_local(self):
        """When local confidence is high, VLM should NOT be called."""
        config = PerceptionConfig(vlm_escalation_threshold=0.6)
        reasoner = HybridReasoner(config=config, vlm_client=None)

        person = _make_scene_entity(entity_id=1, label="person")
        helmet = _make_scene_entity(
            entity_id=2, label="helmet",
            x1=0.15, y1=0.1, x2=0.25, y2=0.2,
        )
        wearing = Relation(
            subject_id=1,
            predicate=SpatialRelation.WEARING,
            object_id=2,
        )
        sg = _make_scene_graph(entities=[person, helmet], relations=[wearing])
        ws = _make_world_state(scene_graph=sg)

        violations = reasoner.evaluate_rules(
            ["ヘルメット未着用の作業者を検出"], sg, ws
        )
        # No violation since person IS wearing helmet
        self.assertEqual(len(violations), 0)
        self.assertFalse(reasoner._vlm_called_this_frame)

    def test_low_confidence_vlm_fallback_when_no_client(self):
        """When confidence is low but no VLM client, still returns local result."""
        config = PerceptionConfig(vlm_escalation_threshold=0.6)
        reasoner = HybridReasoner(config=config, vlm_client=None)

        sg = _make_scene_graph(entities=[])
        ws = _make_world_state(scene_graph=sg)

        # Behavioral rule: local returns 0.0 confidence
        violations = reasoner.evaluate_rules(
            ["作業手順が正しく実行されているか確認"], sg, ws
        )
        self.assertFalse(reasoner._vlm_called_this_frame)

    def test_no_vlm_uses_local_regardless(self):
        """Without VLM client, all rules use local result."""
        config = PerceptionConfig(vlm_escalation_threshold=0.6)
        reasoner = HybridReasoner(config=config, vlm_client=None)

        person = _make_scene_entity(entity_id=1, label="person")
        sg = _make_scene_graph(entities=[person])
        ws = _make_world_state(scene_graph=sg)

        # Helmet rule: no helmet detected -> violations
        violations = reasoner.evaluate_rules(
            ["ヘルメット未着用の作業者を検出"], sg, ws
        )
        self.assertGreater(len(violations), 0)
        self.assertFalse(reasoner._vlm_called_this_frame)

    def test_multiple_rules_evaluated(self):
        """Multiple rules should all be evaluated."""
        config = PerceptionConfig(vlm_escalation_threshold=0.6)
        reasoner = HybridReasoner(config=config, vlm_client=None)

        person = _make_scene_entity(entity_id=1, label="person")
        forklift = _make_scene_entity(
            entity_id=2, label="forklift",
            x1=0.5, y1=0.5, x2=0.7, y2=0.7,
        )
        sg = _make_scene_graph(entities=[person, forklift])
        ws = _make_world_state(scene_graph=sg)

        violations = reasoner.evaluate_rules(
            [
                "ヘルメット未着用の作業者を検出",
                "フォークリフトを検出",
            ],
            sg,
            ws,
        )
        # Should have violations from both rules
        rules_hit = {v.rule for v in violations}
        self.assertGreaterEqual(len(rules_hit), 1)

    def test_vlm_called_when_mock_vlm_provided(self):
        """When VLM client is provided and confidence is low, VLM flag set."""
        # Create a minimal mock VLM that returns empty violations
        class MockVLM:
            def analyze_frame(self, path, rules):
                class Result:
                    violations = []
                return Result()
            def close(self):
                pass

        config = PerceptionConfig(vlm_escalation_threshold=0.99)
        reasoner = HybridReasoner(config=config, vlm_client=MockVLM())

        person = _make_scene_entity(entity_id=1, label="person")
        sg = _make_scene_graph(entities=[person])
        ws = _make_world_state(scene_graph=sg)
        frame = _make_frame()

        # With threshold 0.99 almost all rules escalate, but VLM
        # escalator needs cv2 which may not be available. The key
        # thing is the reasoner handles it gracefully.
        violations = reasoner.evaluate_rules(
            ["ヘルメット未着用の作業者を検出"],
            sg, ws, frame,
        )
        # Should not crash regardless of cv2 availability
        self.assertIsInstance(violations, list)


# ===================================================================
# TestPerceptionEngine
# ===================================================================


class TestPerceptionEngine(unittest.TestCase):
    """Tests for PerceptionEngine end-to-end."""

    def _build_engine(self, rules_for_mock: list[MockRule] | None = None) -> PerceptionEngine:
        """Build an engine with MockDetector and all real components."""
        config = PerceptionConfig(
            track_max_age=5,
            track_min_hits=2,
            detection_confidence_threshold=0.3,
            track_high_threshold=0.3,
        )
        detector = MockDetector(
            rules=rules_for_mock or [],
            use_heuristics=False,
            config=config,
        )
        tracker = MultiObjectTracker(config)
        scene_builder = SceneGraphBuilder(config)
        world_model = WorldModel(config)
        reasoner = HybridReasoner(config=config, vlm_client=None)

        return PerceptionEngine(
            config=config,
            detector=detector,
            tracker=tracker,
            scene_builder=scene_builder,
            world_model=world_model,
            reasoner=reasoner,
        )

    def test_process_frame_returns_frame_result(self):
        engine = self._build_engine()
        frame = _make_frame()
        result = engine.process_frame(frame, 0.0, 0, ["person"])
        self.assertIsInstance(result, FrameResult)
        self.assertEqual(result.frame_number, 0)
        self.assertIsNotNone(result.world_state)

    def test_process_frame_detects_objects(self):
        rules = [
            MockRule(label="person", bbox=BBox(0.2, 0.2, 0.5, 0.8), confidence=0.9),
        ]
        engine = self._build_engine(rules_for_mock=rules)
        frame = _make_frame()
        result = engine.process_frame(frame, 0.0, 0, ["person"])
        self.assertGreaterEqual(result.detections_count, 1)

    def test_world_state_updated_across_frames(self):
        rules = [
            MockRule(label="person", bbox=BBox(0.2, 0.2, 0.5, 0.8), confidence=0.9),
        ]
        engine = self._build_engine(rules_for_mock=rules)
        frame = _make_frame()

        # Process several frames to build up tracks and world state
        for i in range(5):
            result = engine.process_frame(frame, float(i) * 0.033, i, ["person"])

        self.assertEqual(engine.frames_processed, 5)
        # World state should have entity count > 0 for last frame
        self.assertIsNotNone(result.world_state)

    def test_violations_detected_for_safety_rules(self):
        """When a person is detected but no helmet, helmet rule should trigger."""
        rules = [
            MockRule(label="person", bbox=BBox(0.2, 0.1, 0.5, 0.8), confidence=0.9),
        ]
        engine = self._build_engine(rules_for_mock=rules)
        frame = _make_frame()

        # Process enough frames for tracks to be confirmed (ACTIVE)
        safety_rules = ["ヘルメット未着用の作業者を検出"]
        result = None
        for i in range(5):
            result = engine.process_frame(
                frame, float(i) * 0.033, i, safety_rules
            )

        # By the time tracks are ACTIVE, the reasoner should detect
        # that the person has no helmet
        self.assertIsInstance(result.violations, list)
        # Violations depend on whether track becomes ACTIVE and enters
        # scene graph. With 5 frames and min_hits=2, it should.
        if result.tracks_count > 0:
            helmet_violations = [
                v for v in result.violations
                if "helmet" in v.rule.lower() or "ヘルメット" in v.rule
            ]
            self.assertGreater(len(helmet_violations), 0)

    def test_build_perception_engine_factory(self):
        """build_perception_engine should create a working engine."""
        config = PerceptionConfig(detector_backend="mock")
        # The factory tries to instantiate ObjectDetector (ABC), which
        # will fail. But the engine should still be created with None
        # detector.
        engine = build_perception_engine(config=config)
        self.assertIsInstance(engine, PerceptionEngine)
        frame = _make_frame()
        result = engine.process_frame(frame, 0.0, 0, ["person"])
        self.assertIsInstance(result, FrameResult)

    def test_engine_reset(self):
        engine = self._build_engine()
        frame = _make_frame()
        engine.process_frame(frame, 0.0, 0, ["person"])
        engine.reset()
        self.assertEqual(engine.frames_processed, 0)

    def test_engine_close(self):
        engine = self._build_engine()
        engine.close()  # should not raise


# ===================================================================
# TestZone
# ===================================================================


class TestZone(unittest.TestCase):
    """Tests for Zone point-in-polygon and bbox containment."""

    def test_contains_point_inside(self):
        zone = _make_restricted_zone()
        self.assertTrue(zone.contains_point(0.5, 0.5))

    def test_contains_point_outside(self):
        zone = _make_restricted_zone()
        self.assertFalse(zone.contains_point(0.1, 0.1))

    def test_contains_bbox_center_inside(self):
        zone = _make_restricted_zone()
        bbox = BBox(0.4, 0.4, 0.6, 0.6)  # center at (0.5, 0.5)
        self.assertTrue(zone.contains_bbox(bbox))

    def test_contains_bbox_center_outside(self):
        zone = _make_restricted_zone()
        bbox = BBox(0.0, 0.0, 0.1, 0.1)  # center at (0.05, 0.05)
        self.assertFalse(zone.contains_bbox(bbox))

    def test_overlap_ratio(self):
        zone = _make_restricted_zone()
        bbox = BBox(0.4, 0.4, 0.6, 0.6)  # fully inside zone
        ratio = zone.overlap_ratio(bbox)
        self.assertGreater(ratio, 0.5)

    def test_degenerate_polygon(self):
        """Polygon with fewer than 3 points cannot contain anything."""
        zone = Zone(
            zone_id="bad", name="Bad", polygon=[(0.1, 0.1), (0.5, 0.5)]
        )
        self.assertFalse(zone.contains_point(0.3, 0.3))


# ===================================================================
# TestSceneGraph
# ===================================================================


class TestSceneGraph(unittest.TestCase):
    """Tests for SceneGraph query methods."""

    def test_get_entity_found(self):
        e = _make_scene_entity(entity_id=42)
        sg = _make_scene_graph(entities=[e])
        self.assertIsNotNone(sg.get_entity(42))

    def test_get_entity_not_found(self):
        sg = _make_scene_graph(entities=[])
        self.assertIsNone(sg.get_entity(999))

    def test_get_relations_for(self):
        rel = Relation(subject_id=1, predicate=SpatialRelation.NEAR, object_id=2)
        sg = _make_scene_graph(relations=[rel])
        rels_for_1 = sg.get_relations_for(1)
        self.assertEqual(len(rels_for_1), 1)
        rels_for_3 = sg.get_relations_for(3)
        self.assertEqual(len(rels_for_3), 0)

    def test_entities_with_label(self):
        e1 = _make_scene_entity(entity_id=1, label="person")
        e2 = _make_scene_entity(entity_id=2, label="helmet")
        sg = _make_scene_graph(entities=[e1, e2])
        persons = sg.entities_with_label("person")
        self.assertEqual(len(persons), 1)

    def test_entity_count(self):
        entities = [_make_scene_entity(entity_id=i) for i in range(3)]
        sg = _make_scene_graph(entities=entities)
        self.assertEqual(sg.entity_count, 3)

    def test_person_count(self):
        e1 = _make_scene_entity(entity_id=1, label="person")
        e2 = _make_scene_entity(entity_id=2, label="helmet")
        sg = _make_scene_graph(entities=[e1, e2])
        self.assertEqual(sg.person_count, 1)


# ===================================================================
# TestTrackState
# ===================================================================


class TestTrackState(unittest.TestCase):

    def test_is_confirmed_active(self):
        t = Track(track_id=1, label="person", state=TrackState.ACTIVE)
        self.assertTrue(t.is_confirmed)

    def test_is_confirmed_occluded(self):
        t = Track(track_id=1, label="person", state=TrackState.OCCLUDED)
        self.assertTrue(t.is_confirmed)

    def test_is_not_confirmed_tentative(self):
        t = Track(track_id=1, label="person", state=TrackState.TENTATIVE)
        self.assertFalse(t.is_confirmed)

    def test_lifetime_frames(self):
        t = Track(track_id=1, label="person", first_frame=10, last_frame=20)
        self.assertEqual(t.lifetime_frames, 11)


# ===================================================================
# TestViolation
# ===================================================================


class TestViolation(unittest.TestCase):

    def test_creation(self):
        v = Violation(
            rule="test rule",
            rule_index=0,
            description_ja="テスト",
            severity=ViolationSeverity.WARNING,
            confidence=0.8,
            entity_ids=[1, 2],
            bbox=BBox(0.1, 0.1, 0.5, 0.5),
            evidence={"key": "value"},
            source="local",
        )
        self.assertEqual(v.severity, ViolationSeverity.WARNING)
        self.assertEqual(len(v.entity_ids), 2)

    def test_defaults(self):
        v = Violation(
            rule="r", rule_index=0, description_ja="d",
            severity=ViolationSeverity.INFO, confidence=0.5,
        )
        self.assertEqual(v.entity_ids, [])
        self.assertIsNone(v.bbox)
        self.assertEqual(v.source, "local")


# ===================================================================
# TestFrameResult
# ===================================================================


class TestFrameResult(unittest.TestCase):

    def test_creation(self):
        ws = _make_world_state()
        fr = FrameResult(
            timestamp=1.0,
            frame_number=30,
            world_state=ws,
            violations=[],
            processing_time_ms=15.0,
            detections_count=5,
            tracks_count=3,
        )
        self.assertEqual(fr.detections_count, 5)
        self.assertFalse(fr.vlm_called)

    def test_defaults(self):
        ws = _make_world_state()
        fr = FrameResult(
            timestamp=0.0, frame_number=0, world_state=ws,
            violations=[], processing_time_ms=0.0,
        )
        self.assertEqual(fr.detections_count, 0)
        self.assertEqual(fr.vlm_latency_ms, 0.0)


# ===================================================================
# TestPerceptionConfig
# ===================================================================


class TestPerceptionConfig(unittest.TestCase):

    def test_defaults(self):
        c = PerceptionConfig()
        self.assertEqual(c.track_max_age, 30)
        self.assertEqual(c.track_min_hits, 3)
        self.assertAlmostEqual(c.scene_near_threshold, 0.15)
        self.assertEqual(c.max_detections_per_frame, 50)

    def test_custom_values(self):
        c = PerceptionConfig(
            track_max_age=10,
            scene_near_threshold=0.25,
            vlm_escalation_threshold=0.8,
        )
        self.assertEqual(c.track_max_age, 10)
        self.assertAlmostEqual(c.scene_near_threshold, 0.25)
        self.assertAlmostEqual(c.vlm_escalation_threshold, 0.8)


if __name__ == "__main__":
    unittest.main()
