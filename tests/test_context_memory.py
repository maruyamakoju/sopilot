"""Tests for ContextMemory — long-horizon session memory.

Covers:
    - Entity summary creation and updates
    - Session summary aggregation
    - Zone stats (entries, exits, dwell time, occupancy)
    - Timeline filtering by entity, zone, event type, time
    - Simple query answering (person count, violation count, zone visitors)
    - Memory retention (old events pruned)
    - Reset clears all state
    - Edge cases: empty session, single frame
"""

from __future__ import annotations

import unittest
from dataclasses import field

from sopilot.perception.context_memory import (
    ContextMemory,
    EntitySummary,
    SessionSummary,
    ZoneStats,
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
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_entity(
    entity_id: int,
    label: str = "person",
    x: float = 0.5,
    y: float = 0.5,
    zone_ids: list[str] | None = None,
    attributes: dict | None = None,
) -> SceneEntity:
    """Create a SceneEntity at a given position."""
    bbox = BBox(x1=x - 0.05, y1=y - 0.05, x2=x + 0.05, y2=y + 0.05)
    return SceneEntity(
        entity_id=entity_id,
        label=label,
        bbox=bbox,
        confidence=0.9,
        attributes=attributes or {},
        zone_ids=zone_ids or [],
    )


def _make_scene_graph(
    timestamp: float,
    frame_number: int,
    entities: list[SceneEntity],
) -> SceneGraph:
    return SceneGraph(
        timestamp=timestamp,
        frame_number=frame_number,
        entities=entities,
        relations=[],
    )


def _make_world_state(
    timestamp: float,
    frame_number: int,
    entities: list[SceneEntity],
    events: list[EntityEvent] | None = None,
    zone_occupancy: dict[str, list[int]] | None = None,
) -> WorldState:
    """Create a WorldState snapshot."""
    sg = _make_scene_graph(timestamp, frame_number, entities)
    active_tracks: dict[int, Track] = {}
    for e in entities:
        active_tracks[e.entity_id] = Track(
            track_id=e.entity_id,
            label=e.label,
            state=TrackState.ACTIVE,
            bbox=e.bbox,
            confidence=e.confidence,
        )
    return WorldState(
        timestamp=timestamp,
        frame_number=frame_number,
        scene_graph=sg,
        active_tracks=active_tracks,
        events=events or [],
        zone_occupancy=zone_occupancy or {},
        entity_count=len(entities),
        person_count=sum(1 for e in entities if "person" in e.label.lower()),
    )


def _make_zone_entered_event(
    entity_id: int,
    zone_id: str,
    timestamp: float,
    frame_number: int,
    label: str = "person",
) -> EntityEvent:
    return EntityEvent(
        event_type=EntityEventType.ZONE_ENTERED,
        entity_id=entity_id,
        timestamp=timestamp,
        frame_number=frame_number,
        details={"zone_id": zone_id, "zone_name": zone_id, "label": label},
    )


def _make_zone_exited_event(
    entity_id: int,
    zone_id: str,
    timestamp: float,
    frame_number: int,
    duration_seconds: float = 10.0,
    label: str = "person",
) -> EntityEvent:
    return EntityEvent(
        event_type=EntityEventType.ZONE_EXITED,
        entity_id=entity_id,
        timestamp=timestamp,
        frame_number=frame_number,
        details={
            "zone_id": zone_id,
            "zone_name": zone_id,
            "label": label,
            "duration_seconds": duration_seconds,
        },
    )


def _make_violation_event(
    entity_id: int,
    rule: str,
    timestamp: float,
    frame_number: int,
    severity: str = "warning",
    zone_id: str | None = None,
) -> EntityEvent:
    details: dict = {
        "rule": rule,
        "severity": severity,
        "label": f"entity {entity_id}",
    }
    if zone_id:
        details["zone_id"] = zone_id
    return EntityEvent(
        event_type=EntityEventType.RULE_VIOLATION,
        entity_id=entity_id,
        timestamp=timestamp,
        frame_number=frame_number,
        details=details,
    )


def _make_entered_event(
    entity_id: int,
    timestamp: float,
    frame_number: int,
    label: str = "person",
) -> EntityEvent:
    return EntityEvent(
        event_type=EntityEventType.ENTERED,
        entity_id=entity_id,
        timestamp=timestamp,
        frame_number=frame_number,
        details={"label": label},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestContextMemoryEntitySummary(unittest.TestCase):
    """Entity summary creation and updates."""

    def test_single_entity_across_frames(self):
        cm = ContextMemory()
        # Frame 1: entity 1 at (0.3, 0.3)
        ws1 = _make_world_state(
            1.0, 1,
            [_make_entity(1, "person", x=0.3, y=0.3)],
            events=[_make_entered_event(1, 1.0, 1)],
        )
        cm.update(ws1)

        # Frame 2: entity 1 moves to (0.5, 0.5)
        ws2 = _make_world_state(
            2.0, 2,
            [_make_entity(1, "person", x=0.5, y=0.5)],
        )
        cm.update(ws2)

        summary = cm.get_entity_summary(1)
        self.assertIsNotNone(summary)
        self.assertEqual(summary.entity_id, 1)
        self.assertEqual(summary.label, "person")
        self.assertAlmostEqual(summary.first_seen, 1.0)
        self.assertAlmostEqual(summary.last_seen, 2.0)
        self.assertEqual(summary.total_frames, 2)
        self.assertGreater(summary.total_distance, 0.0)

    def test_nonexistent_entity(self):
        cm = ContextMemory()
        self.assertIsNone(cm.get_entity_summary(999))

    def test_entity_with_zones(self):
        cm = ContextMemory()
        # Entity enters restricted zone
        ws1 = _make_world_state(
            0.0, 0,
            [_make_entity(1, "person", x=0.5, y=0.5)],
            events=[_make_zone_entered_event(1, "restricted_area", 0.0, 0)],
            zone_occupancy={"restricted_area": [1]},
        )
        cm.update(ws1)

        summary = cm.get_entity_summary(1)
        self.assertIsNotNone(summary)
        self.assertIn("restricted_area", summary.zones_visited)
        self.assertEqual(summary.current_zone, "restricted_area")

    def test_entity_with_violations(self):
        cm = ContextMemory()
        ws = _make_world_state(
            1.0, 1,
            [_make_entity(1, "person")],
            events=[_make_violation_event(1, "no_helmet", 1.0, 1)],
        )
        cm.update(ws)

        summary = cm.get_entity_summary(1)
        self.assertIn("no_helmet", summary.violations)

    def test_entity_with_activity(self):
        cm = ContextMemory()
        ws = _make_world_state(
            1.0, 1,
            [_make_entity(1, "person", attributes={"activity": "walking"})],
        )
        cm.update(ws)

        summary = cm.get_entity_summary(1)
        self.assertEqual(summary.current_activity, "walking")
        self.assertIn("walking", summary.activities)

    def test_entity_zone_duration(self):
        """Zone duration is accumulated when entity exits a zone."""
        cm = ContextMemory()

        # Frame 1: entity in zone
        ws1 = _make_world_state(
            10.0, 1,
            [_make_entity(1, "person")],
            events=[_make_zone_entered_event(1, "work_area", 10.0, 1)],
            zone_occupancy={"work_area": [1]},
        )
        cm.update(ws1)

        # Frame 2: still in zone
        ws2 = _make_world_state(
            20.0, 2,
            [_make_entity(1, "person")],
            zone_occupancy={"work_area": [1]},
        )
        cm.update(ws2)

        # Frame 3: entity exits zone (no longer in zone_occupancy)
        ws3 = _make_world_state(
            30.0, 3,
            [_make_entity(1, "person")],
            events=[_make_zone_exited_event(1, "work_area", 30.0, 3, duration_seconds=20.0)],
            zone_occupancy={},
        )
        cm.update(ws3)

        summary = cm.get_entity_summary(1)
        # Duration should be ~20 seconds (from 10 to 30)
        self.assertAlmostEqual(summary.zone_durations.get("work_area", 0.0), 20.0, places=1)

    def test_entity_exits_scene_closes_zone_duration(self):
        """When entity disappears, open zone durations are closed."""
        cm = ContextMemory()

        ws1 = _make_world_state(
            10.0, 1,
            [_make_entity(1, "person")],
            zone_occupancy={"zone_a": [1]},
        )
        cm.update(ws1)

        # Frame 2: entity gone
        ws2 = _make_world_state(20.0, 2, [])
        cm.update(ws2)

        summary = cm.get_entity_summary(1)
        # Duration from 10 to 20
        self.assertAlmostEqual(summary.zone_durations.get("zone_a", 0.0), 10.0, places=1)
        self.assertIsNone(summary.current_zone)


class TestContextMemorySessionSummary(unittest.TestCase):
    """Session summary aggregation."""

    def test_empty_session(self):
        cm = ContextMemory()
        summary = cm.get_session_summary()
        self.assertEqual(summary.total_frames_processed, 0)
        self.assertEqual(summary.unique_entities_seen, 0)
        self.assertEqual(summary.current_entity_count, 0)
        self.assertEqual(summary.total_violations, 0)
        self.assertEqual(summary.duration_seconds, 0.0)

    def test_session_with_entities_and_violations(self):
        cm = ContextMemory()

        entities = [
            _make_entity(1, "person", x=0.2, y=0.2),
            _make_entity(2, "person", x=0.8, y=0.8),
        ]
        events = [
            _make_entered_event(1, 100.0, 1),
            _make_entered_event(2, 100.0, 1),
            _make_violation_event(1, "no_helmet", 100.0, 1, severity="critical"),
            _make_violation_event(2, "no_vest", 100.0, 1, severity="warning"),
        ]
        ws1 = _make_world_state(100.0, 1, entities, events=events)
        cm.update(ws1)

        # Second frame, entity 2 leaves
        ws2 = _make_world_state(110.0, 2, [_make_entity(1, "person", x=0.3, y=0.3)])
        cm.update(ws2)

        summary = cm.get_session_summary()
        self.assertEqual(summary.total_frames_processed, 2)
        self.assertAlmostEqual(summary.start_time, 100.0)
        self.assertAlmostEqual(summary.current_time, 110.0)
        self.assertAlmostEqual(summary.duration_seconds, 10.0)
        self.assertEqual(summary.unique_entities_seen, 2)
        self.assertEqual(summary.current_entity_count, 1)
        self.assertEqual(summary.total_violations, 2)
        self.assertEqual(summary.violations_by_severity.get("critical"), 1)
        self.assertEqual(summary.violations_by_severity.get("warning"), 1)
        self.assertEqual(summary.violations_by_rule.get("no_helmet"), 1)
        self.assertEqual(summary.violations_by_rule.get("no_vest"), 1)
        self.assertIn(1, summary.entity_summaries)
        self.assertIn(2, summary.entity_summaries)
        self.assertTrue(len(summary.notable_events) > 0)

    def test_single_frame_session(self):
        cm = ContextMemory()
        ws = _make_world_state(50.0, 1, [_make_entity(1, "person")])
        cm.update(ws)

        summary = cm.get_session_summary()
        self.assertEqual(summary.total_frames_processed, 1)
        self.assertEqual(summary.unique_entities_seen, 1)
        self.assertEqual(summary.current_entity_count, 1)
        self.assertAlmostEqual(summary.duration_seconds, 0.0)


class TestContextMemoryZoneStats(unittest.TestCase):
    """Zone stats: entries, exits, dwell time, occupancy."""

    def test_zone_entries_and_exits(self):
        cm = ContextMemory()

        # Entity 1 enters zone
        ws1 = _make_world_state(
            0.0, 0,
            [_make_entity(1, "person")],
            events=[_make_zone_entered_event(1, "work_area", 0.0, 0)],
            zone_occupancy={"work_area": [1]},
        )
        cm.update(ws1)

        # Entity 1 exits zone
        ws2 = _make_world_state(
            10.0, 1,
            [_make_entity(1, "person")],
            events=[_make_zone_exited_event(1, "work_area", 10.0, 1, duration_seconds=10.0)],
            zone_occupancy={"work_area": []},
        )
        cm.update(ws2)

        summary = cm.get_session_summary()
        zs = summary.zone_summary.get("work_area")
        self.assertIsNotNone(zs)
        self.assertEqual(zs.total_entries, 1)
        self.assertEqual(zs.total_exits, 1)
        self.assertEqual(zs.unique_visitors, 1)
        self.assertAlmostEqual(zs.average_dwell_time_seconds, 10.0)

    def test_zone_max_occupancy(self):
        cm = ContextMemory()

        # 3 people in zone at once
        entities = [
            _make_entity(i, "person", x=0.1 * i, y=0.1 * i)
            for i in range(1, 4)
        ]
        ws1 = _make_world_state(
            0.0, 0, entities,
            zone_occupancy={"area_a": [1, 2, 3]},
        )
        cm.update(ws1)

        # Only 1 person remains
        ws2 = _make_world_state(
            1.0, 1,
            [_make_entity(1, "person")],
            zone_occupancy={"area_a": [1]},
        )
        cm.update(ws2)

        summary = cm.get_session_summary()
        zs = summary.zone_summary["area_a"]
        self.assertEqual(zs.max_occupancy, 3)
        self.assertEqual(zs.current_occupancy, 1)
        self.assertEqual(zs.unique_visitors, 3)

    def test_zone_violations_tracked(self):
        cm = ContextMemory()
        ws = _make_world_state(
            0.0, 0,
            [_make_entity(1, "person")],
            events=[
                _make_violation_event(1, "no_helmet", 0.0, 0, zone_id="hazard_zone"),
                _make_violation_event(1, "no_vest", 0.0, 0, zone_id="hazard_zone"),
            ],
            zone_occupancy={"hazard_zone": [1]},
        )
        cm.update(ws)

        summary = cm.get_session_summary()
        zs = summary.zone_summary.get("hazard_zone")
        self.assertIsNotNone(zs)
        self.assertEqual(zs.violations_in_zone, 2)


class TestContextMemoryTimeline(unittest.TestCase):
    """Timeline filtering by entity, zone, event type, time."""

    def setUp(self):
        self.cm = ContextMemory()

        # Frame 1: entities enter, zone events
        events1 = [
            _make_entered_event(1, 100.0, 1, "person"),
            _make_entered_event(2, 100.0, 1, "person"),
            _make_zone_entered_event(1, "zone_a", 100.0, 1),
        ]
        entities1 = [
            _make_entity(1, "person", x=0.2, y=0.2),
            _make_entity(2, "person", x=0.8, y=0.8),
        ]
        ws1 = _make_world_state(100.0, 1, entities1, events=events1, zone_occupancy={"zone_a": [1]})
        self.cm.update(ws1)

        # Frame 2: violation
        events2 = [
            _make_violation_event(1, "no_helmet", 200.0, 2),
        ]
        ws2 = _make_world_state(200.0, 2, entities1, events=events2, zone_occupancy={"zone_a": [1]})
        self.cm.update(ws2)

        # Frame 3: entity 1 exits zone
        events3 = [
            _make_zone_exited_event(1, "zone_a", 300.0, 3, duration_seconds=200.0),
        ]
        ws3 = _make_world_state(300.0, 3, entities1, events=events3)
        self.cm.update(ws3)

    def test_unfiltered_timeline(self):
        tl = self.cm.get_timeline()
        self.assertEqual(len(tl), 5)  # 2 entered + 1 zone_entered + 1 violation + 1 zone_exited

    def test_filter_by_entity(self):
        tl = self.cm.get_timeline(entity_id=1)
        self.assertTrue(all(e["entity_id"] == 1 for e in tl))
        self.assertEqual(len(tl), 4)  # entered, zone_entered, violation, zone_exited

    def test_filter_by_zone(self):
        tl = self.cm.get_timeline(zone_id="zone_a")
        self.assertTrue(all(e["details"].get("zone_id") == "zone_a" for e in tl))
        self.assertEqual(len(tl), 2)  # zone_entered + zone_exited

    def test_filter_by_event_type(self):
        tl = self.cm.get_timeline(event_types=[EntityEventType.RULE_VIOLATION])
        self.assertEqual(len(tl), 1)
        self.assertEqual(tl[0]["event_type"], "rule_violation")

    def test_filter_by_time_window(self):
        # Only last ~2 minutes (120s) from current_time=300
        # Should include events at t=200 and t=300
        tl = self.cm.get_timeline(last_n_minutes=2.0)
        timestamps = [e["timestamp"] for e in tl]
        self.assertTrue(all(t >= 180.0 for t in timestamps))
        # t=100 events should be excluded
        self.assertNotIn(100.0, timestamps)

    def test_combined_filters(self):
        tl = self.cm.get_timeline(
            entity_id=1,
            event_types=[EntityEventType.ZONE_ENTERED, EntityEventType.ZONE_EXITED],
        )
        self.assertEqual(len(tl), 2)
        types = {e["event_type"] for e in tl}
        self.assertEqual(types, {"zone_entered", "zone_exited"})


class TestContextMemoryQuery(unittest.TestCase):
    """Simple query answering."""

    def setUp(self):
        self.cm = ContextMemory()
        entities = [
            _make_entity(1, "person", x=0.2, y=0.2),
            _make_entity(2, "person", x=0.5, y=0.5),
            _make_entity(3, "forklift", x=0.8, y=0.8),
        ]
        events = [
            _make_entered_event(1, 1000.0, 1, "person"),
            _make_entered_event(2, 1000.0, 1, "person"),
            _make_entered_event(3, 1000.0, 1, "forklift"),
            _make_zone_entered_event(1, "restricted_area", 1000.0, 1),
            _make_violation_event(1, "unauthorized_access", 1000.0, 1, severity="critical"),
            _make_violation_event(2, "no_helmet", 1000.0, 1, severity="warning"),
        ]
        ws = _make_world_state(
            1000.0, 1, entities,
            events=events,
            zone_occupancy={"restricted_area": [1]},
        )
        self.cm.update(ws)

    def test_query_person_count(self):
        answer = self.cm.query("何人いますか？")
        self.assertIn("2", answer)  # 2 persons (forklift is not a person)

    def test_query_person_count_english(self):
        answer = self.cm.query("How many people are there?")
        self.assertIn("2", answer)

    def test_query_restricted_zone(self):
        answer = self.cm.query("制限エリアに何人入った？")
        self.assertIn("1", answer)

    def test_query_restricted_zone_english(self):
        answer = self.cm.query("How many people entered the restricted zone?")
        self.assertIn("1", answer)

    def test_query_violations(self):
        answer = self.cm.query("違反は何件ありますか？")
        self.assertIn("2", answer)

    def test_query_violations_english(self):
        answer = self.cm.query("What are the violations?")
        self.assertIn("2", answer)

    def test_query_entity_specific(self):
        answer = self.cm.query("entity 1 の情報を教えて")
        self.assertIn("person", answer)
        self.assertIn("1", answer)

    def test_query_worker_break(self):
        """Break query for a nearly-stationary entity."""
        answer = self.cm.query("worker 1 は休憩しましたか？")
        # Entity just appeared in 1 frame, so data insufficient or stationary
        self.assertIsInstance(answer, str)
        self.assertTrue(len(answer) > 0)

    def test_query_fallback(self):
        """Unknown query returns session overview."""
        answer = self.cm.query("xyzzy")
        self.assertIn("セッション概要", answer)

    def test_query_zone_occupancy(self):
        answer = self.cm.query("zone restricted_area の占有状況は？")
        self.assertIn("restricted_area", answer)

    def test_query_all_zones(self):
        answer = self.cm.query("ゾーン一覧")
        self.assertIn("restricted_area", answer)


class TestContextMemoryRetention(unittest.TestCase):
    """Memory retention — old events are pruned."""

    def test_old_events_pruned(self):
        # Very short retention
        cm = ContextMemory(event_retention_seconds=100.0)

        # Old event at t=0
        ws1 = _make_world_state(
            0.0, 0,
            [_make_entity(1, "person")],
            events=[_make_entered_event(1, 0.0, 0)],
        )
        cm.update(ws1)

        # Much later event at t=200 (100 > retention)
        ws2 = _make_world_state(
            200.0, 1,
            [_make_entity(1, "person")],
            events=[_make_violation_event(1, "no_vest", 200.0, 1)],
        )
        cm.update(ws2)

        # The old event at t=0 should be pruned
        tl = cm.get_timeline()
        timestamps = [e["timestamp"] for e in tl]
        self.assertNotIn(0.0, timestamps)
        self.assertIn(200.0, timestamps)

    def test_summaries_persist_after_pruning(self):
        """Entity summaries survive event pruning."""
        cm = ContextMemory(event_retention_seconds=50.0)

        ws1 = _make_world_state(0.0, 0, [_make_entity(1, "person")])
        cm.update(ws1)

        ws2 = _make_world_state(100.0, 1, [_make_entity(1, "person")])
        cm.update(ws2)

        # Events may be pruned, but entity summary persists
        summary = cm.get_entity_summary(1)
        self.assertIsNotNone(summary)
        self.assertEqual(summary.total_frames, 2)


class TestContextMemoryReset(unittest.TestCase):
    """Reset clears all state."""

    def test_reset(self):
        cm = ContextMemory()
        ws = _make_world_state(
            1.0, 1,
            [_make_entity(1, "person")],
            events=[_make_entered_event(1, 1.0, 1)],
        )
        cm.update(ws)

        # Verify state exists
        self.assertIsNotNone(cm.get_entity_summary(1))
        self.assertEqual(cm.get_session_summary().total_frames_processed, 1)

        # Reset
        cm.reset()

        # Verify everything is cleared
        self.assertIsNone(cm.get_entity_summary(1))
        summary = cm.get_session_summary()
        self.assertEqual(summary.total_frames_processed, 0)
        self.assertEqual(summary.unique_entities_seen, 0)
        self.assertEqual(summary.total_violations, 0)
        self.assertEqual(len(cm.get_timeline()), 0)


class TestContextMemoryEdgeCases(unittest.TestCase):
    """Edge cases: empty session, single frame, entity limit."""

    def test_empty_timeline(self):
        cm = ContextMemory()
        self.assertEqual(cm.get_timeline(), [])

    def test_empty_query(self):
        cm = ContextMemory()
        answer = cm.query("何人いますか？")
        self.assertIn("0", answer)

    def test_entity_limit_eviction(self):
        cm = ContextMemory(max_entity_summaries=3)

        # Add 5 entities across frames
        for i in range(1, 6):
            ws = _make_world_state(
                float(i), i,
                [_make_entity(i, "person", x=0.1 * i, y=0.1 * i)],
            )
            cm.update(ws)

        # Should only have the most recent 3
        summary = cm.get_session_summary()
        self.assertLessEqual(len(summary.entity_summaries), 3)
        # Oldest entities should be evicted
        self.assertIsNone(cm.get_entity_summary(1))
        self.assertIsNone(cm.get_entity_summary(2))
        # Most recent should be present
        self.assertIsNotNone(cm.get_entity_summary(5))

    def test_multiple_zones_same_entity(self):
        """Entity in multiple zones simultaneously."""
        cm = ContextMemory()
        ws = _make_world_state(
            0.0, 0,
            [_make_entity(1, "person")],
            events=[
                _make_zone_entered_event(1, "zone_a", 0.0, 0),
                _make_zone_entered_event(1, "zone_b", 0.0, 0),
            ],
            zone_occupancy={"zone_a": [1], "zone_b": [1]},
        )
        cm.update(ws)

        summary = cm.get_entity_summary(1)
        self.assertIn("zone_a", summary.zones_visited)
        self.assertIn("zone_b", summary.zones_visited)
        # current_zone should be one of them
        self.assertIn(summary.current_zone, ["zone_a", "zone_b"])

    def test_distance_accumulation(self):
        """Distance accumulates correctly across frames."""
        cm = ContextMemory()

        # Move entity from (0.0, 0.0) to (0.3, 0.4) in 2 steps
        ws1 = _make_world_state(0.0, 0, [_make_entity(1, "person", x=0.0, y=0.0)])
        cm.update(ws1)

        ws2 = _make_world_state(1.0, 1, [_make_entity(1, "person", x=0.3, y=0.0)])
        cm.update(ws2)

        ws3 = _make_world_state(2.0, 2, [_make_entity(1, "person", x=0.3, y=0.4)])
        cm.update(ws3)

        summary = cm.get_entity_summary(1)
        # Should be 0.3 + 0.4 = 0.7 (approximately, accounting for bbox center)
        self.assertAlmostEqual(summary.total_distance, 0.7, places=1)

    def test_violation_rate_query(self):
        cm = ContextMemory()

        # 3 violations over 10 minutes
        entities = [_make_entity(1, "person")]
        for i in range(3):
            ts = float(i * 200)  # 0, 200, 400 seconds
            events = [_make_violation_event(1, f"rule_{i}", ts, i)]
            ws = _make_world_state(ts, i, entities, events=events)
            cm.update(ws)

        answer = cm.query("違反率は？")
        self.assertIn("件/分", answer)

    def test_last_n_minutes_violation_query(self):
        cm = ContextMemory()
        entities = [_make_entity(1, "person")]

        # Old violation at t=0
        ws1 = _make_world_state(
            0.0, 0, entities,
            events=[_make_violation_event(1, "old_rule", 0.0, 0)],
        )
        cm.update(ws1)

        # Recent violation at t=3600 (1 hour later)
        ws2 = _make_world_state(
            3600.0, 1, entities,
            events=[_make_violation_event(1, "new_rule", 3600.0, 1)],
        )
        cm.update(ws2)

        answer = cm.query("最後の30分間の違反は？")
        self.assertIn("1", answer)  # only the recent one


class TestContextMemoryQueryMinutesExtraction(unittest.TestCase):
    """Test that time extraction from queries works correctly."""

    def test_japanese_minutes(self):
        cm = ContextMemory()
        entities = [_make_entity(1, "person")]
        ws = _make_world_state(
            600.0, 1, entities,
            events=[_make_violation_event(1, "test", 600.0, 1)],
        )
        cm.update(ws)

        answer = cm.query("最後の10分間の違反は？")
        self.assertIn("10", answer)
        self.assertIn("分", answer)

    def test_english_minutes(self):
        cm = ContextMemory()
        entities = [_make_entity(1, "person")]
        ws = _make_world_state(
            600.0, 1, entities,
            events=[_make_violation_event(1, "test", 600.0, 1)],
        )
        cm.update(ws)

        answer = cm.query("What's the violation rate in the last 60 minutes?")
        self.assertIn("60", answer)
        self.assertIn("件/分", answer)


if __name__ == "__main__":
    unittest.main()
