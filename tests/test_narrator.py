"""Tests for sopilot.perception.narrator — Natural Language Scene Narration.

Covers:
    - Empty scene narration
    - Scene with persons and equipment
    - Violation alert generation
    - Event narration for each event type
    - Entity narration (stationary, moving, in zone)
    - Change summarization between two world states
    - BRIEF / STANDARD / DETAILED style differences
    - English language output
    - Edge cases: unknown labels, zero entities
"""

from __future__ import annotations

import unittest

from sopilot.perception.narrator import (
    NarrationStyle,
    SceneNarration,
    SceneNarrator,
)
from sopilot.perception.types import (
    BBox,
    EntityEvent,
    EntityEventType,
    Relation,
    SceneEntity,
    SceneGraph,
    SpatialRelation,
    Track,
    TrackState,
    Violation,
    ViolationSeverity,
    WorldState,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_entity(
    entity_id: int,
    label: str = "person",
    confidence: float = 0.9,
    zone_ids: list[str] | None = None,
    attributes: dict | None = None,
) -> SceneEntity:
    return SceneEntity(
        entity_id=entity_id,
        label=label,
        bbox=BBox(0.1, 0.1, 0.3, 0.3),
        confidence=confidence,
        zone_ids=zone_ids or [],
        attributes=attributes or {},
    )


def _make_track(
    track_id: int,
    label: str = "person",
    velocity: tuple[float, float] = (0.0, 0.0),
    state: TrackState = TrackState.ACTIVE,
) -> Track:
    return Track(
        track_id=track_id,
        label=label,
        state=state,
        bbox=BBox(0.1, 0.1, 0.3, 0.3),
        velocity=velocity,
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
    entities: list[SceneEntity] | None = None,
    tracks: dict[int, Track] | None = None,
    events: list[EntityEvent] | None = None,
    zone_occupancy: dict[str, list[int]] | None = None,
    timestamp: float = 1.0,
    frame_number: int = 30,
    relations: list[Relation] | None = None,
) -> WorldState:
    entities = entities or []
    tracks = tracks or {}
    person_count = sum(1 for e in entities if e.label.lower() in ("person", "worker"))
    sg = _make_scene_graph(
        entities=entities,
        relations=relations or [],
        timestamp=timestamp,
        frame_number=frame_number,
    )
    return WorldState(
        timestamp=timestamp,
        frame_number=frame_number,
        scene_graph=sg,
        active_tracks=tracks,
        events=events or [],
        zone_occupancy=zone_occupancy or {},
        entity_count=len(entities),
        person_count=person_count,
    )


def _make_violation(
    rule: str = "helmet_required",
    description_ja: str = "ヘルメット未着用",
    severity: ViolationSeverity = ViolationSeverity.WARNING,
    entity_ids: list[int] | None = None,
) -> Violation:
    return Violation(
        rule=rule,
        rule_index=0,
        description_ja=description_ja,
        severity=severity,
        confidence=0.9,
        entity_ids=entity_ids or [],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNarrationStyleEnum(unittest.TestCase):
    """NarrationStyle enum values."""

    def test_values(self):
        self.assertEqual(NarrationStyle.BRIEF.value, "brief")
        self.assertEqual(NarrationStyle.STANDARD.value, "standard")
        self.assertEqual(NarrationStyle.DETAILED.value, "detailed")


class TestSceneNarrationDataclass(unittest.TestCase):
    """SceneNarration is frozen and has the right fields."""

    def test_frozen(self):
        narration = SceneNarration(
            text_ja="テスト",
            text_en="test",
            style=NarrationStyle.BRIEF,
            timestamp=1.0,
            frame_number=30,
            key_facts=["fact"],
            entity_mentions=[1],
        )
        with self.assertRaises(AttributeError):
            narration.text_ja = "changed"  # type: ignore[misc]

    def test_fields(self):
        narration = SceneNarration(
            text_ja="日本語",
            text_en="English",
            style=NarrationStyle.STANDARD,
            timestamp=2.5,
            frame_number=75,
            key_facts=["a", "b"],
            entity_mentions=[1, 2, 3],
        )
        self.assertEqual(narration.text_ja, "日本語")
        self.assertEqual(narration.text_en, "English")
        self.assertEqual(narration.style, NarrationStyle.STANDARD)
        self.assertAlmostEqual(narration.timestamp, 2.5)
        self.assertEqual(narration.frame_number, 75)
        self.assertEqual(narration.key_facts, ["a", "b"])
        self.assertEqual(narration.entity_mentions, [1, 2, 3])


class TestEmptyScene(unittest.TestCase):
    """Narration for an empty scene (no entities, no violations)."""

    def setUp(self):
        self.narrator = SceneNarrator()
        self.ws = _make_world_state()

    def test_brief_empty(self):
        result = self.narrator.narrate(self.ws, style=NarrationStyle.BRIEF)
        self.assertIn("検出されていません", result.text_ja)
        self.assertIn("No entities", result.text_en)
        self.assertEqual(result.entity_mentions, [])

    def test_standard_empty(self):
        result = self.narrator.narrate(self.ws, style=NarrationStyle.STANDARD)
        self.assertIn("検出されていません", result.text_ja)

    def test_detailed_empty(self):
        result = self.narrator.narrate(self.ws, style=NarrationStyle.DETAILED)
        self.assertIn("検出されていません", result.text_ja)


class TestSceneWithPersonsAndEquipment(unittest.TestCase):
    """Narration for a scene with persons and equipment."""

    def setUp(self):
        self.narrator = SceneNarrator()
        self.entities = [
            _make_entity(1, "person"),
            _make_entity(2, "person"),
            _make_entity(3, "person"),
            _make_entity(10, "helmet"),
            _make_entity(11, "helmet"),
        ]
        self.tracks = {
            1: _make_track(1, velocity=(0.0, 0.0)),
            2: _make_track(2, velocity=(0.01, 0.005)),
            3: _make_track(3, velocity=(0.0, 0.0)),
        }
        self.ws = _make_world_state(
            entities=self.entities,
            tracks=self.tracks,
        )

    def test_person_count_in_ja(self):
        result = self.narrator.narrate(self.ws, style=NarrationStyle.BRIEF)
        self.assertIn("3名", result.text_ja)
        self.assertIn("作業員", result.text_ja)

    def test_person_count_in_en(self):
        result = self.narrator.narrate(self.ws, style=NarrationStyle.BRIEF)
        self.assertIn("3 workers", result.text_en)

    def test_equipment_in_standard(self):
        result = self.narrator.narrate(self.ws, style=NarrationStyle.STANDARD)
        self.assertIn("ヘルメット", result.text_ja)
        self.assertIn("helmet", result.text_en)

    def test_entity_mentions(self):
        result = self.narrator.narrate(self.ws, style=NarrationStyle.STANDARD)
        # All entity IDs should be mentioned
        for eid in [1, 2, 3, 10, 11]:
            self.assertIn(eid, result.entity_mentions)

    def test_activity_descriptions(self):
        result = self.narrator.narrate(self.ws, style=NarrationStyle.STANDARD)
        # Stationary workers
        self.assertIn("静止", result.text_ja)
        # Moving worker
        self.assertIn("移動中", result.text_ja)


class TestViolationAlertGeneration(unittest.TestCase):
    """Violation alert generation."""

    def setUp(self):
        self.narrator = SceneNarrator()
        self.entities = [_make_entity(5, "person")]
        self.tracks = {5: _make_track(5)}
        self.ws = _make_world_state(entities=self.entities, tracks=self.tracks)

    def test_warning_alert(self):
        v = _make_violation(
            description_ja="ヘルメット未着用で作業エリアに入りました",
            severity=ViolationSeverity.WARNING,
            entity_ids=[5],
        )
        alert = self.narrator.generate_alert(v, self.ws)
        self.assertTrue(alert.startswith("\u26a0"))
        self.assertIn("ID: 5", alert)
        self.assertIn("ヘルメット未着用", alert)

    def test_critical_alert(self):
        v = _make_violation(
            description_ja="危険エリアに無許可で侵入",
            severity=ViolationSeverity.CRITICAL,
            entity_ids=[5],
        )
        alert = self.narrator.generate_alert(v, self.ws)
        self.assertTrue(alert.startswith("\U0001f6a8"))

    def test_info_alert_no_marker(self):
        v = _make_violation(
            description_ja="情報通知",
            severity=ViolationSeverity.INFO,
            entity_ids=[5],
        )
        alert = self.narrator.generate_alert(v, self.ws)
        self.assertFalse(alert.startswith("\u26a0"))
        self.assertFalse(alert.startswith("\U0001f6a8"))

    def test_alert_without_entity(self):
        v = _make_violation(
            description_ja="一般的な違反",
            severity=ViolationSeverity.WARNING,
            entity_ids=[],
        )
        alert = self.narrator.generate_alert(v, self.ws)
        self.assertIn("一般的な違反", alert)

    def test_violations_in_narration(self):
        violations = [
            _make_violation(
                description_ja="ヘルメット未着用",
                severity=ViolationSeverity.WARNING,
                entity_ids=[5],
            ),
        ]
        result = self.narrator.narrate(self.ws, violations=violations)
        self.assertIn("1件の違反", result.text_ja)
        self.assertIn("1 violation", result.text_en)


class TestEventNarration(unittest.TestCase):
    """narrate_event for each event type."""

    def setUp(self):
        self.narrator = SceneNarrator()
        self.entities = [_make_entity(1, "person")]
        self.tracks = {1: _make_track(1)}
        self.ws = _make_world_state(entities=self.entities, tracks=self.tracks)

    def _narrate(self, event_type: EntityEventType, details: dict | None = None) -> str:
        evt = EntityEvent(
            event_type=event_type,
            entity_id=1,
            timestamp=1.0,
            frame_number=30,
            details=details or {},
        )
        return self.narrator.narrate_event(evt, self.ws)

    def test_entered(self):
        text = self._narrate(EntityEventType.ENTERED)
        self.assertIn("入場", text)
        self.assertIn("ID: 1", text)

    def test_exited(self):
        text = self._narrate(EntityEventType.EXITED)
        self.assertIn("退場", text)

    def test_zone_entered(self):
        text = self._narrate(
            EntityEventType.ZONE_ENTERED,
            {"zone_id": "制限エリアA"},
        )
        self.assertIn("制限エリアA", text)
        self.assertIn("入りました", text)

    def test_zone_exited(self):
        text = self._narrate(
            EntityEventType.ZONE_EXITED,
            {"zone_id": "制限エリアA"},
        )
        self.assertIn("制限エリアA", text)
        self.assertIn("出ました", text)

    def test_state_changed_activity(self):
        text = self._narrate(
            EntityEventType.STATE_CHANGED,
            {"change_type": "activity", "old_activity": "stationary", "new_activity": "walking"},
        )
        self.assertIn("静止中", text)
        self.assertIn("歩行中", text)

    def test_state_changed_generic(self):
        text = self._narrate(EntityEventType.STATE_CHANGED, {"change_type": "other"})
        self.assertIn("状態が変化", text)

    def test_anomaly(self):
        text = self._narrate(EntityEventType.ANOMALY)
        self.assertIn("異常", text)

    def test_rule_violation(self):
        text = self._narrate(
            EntityEventType.RULE_VIOLATION,
            {"rule": "helmet_required"},
        )
        self.assertIn("helmet_required", text)
        self.assertIn("違反", text)

    def test_prolonged_presence(self):
        text = self._narrate(
            EntityEventType.PROLONGED_PRESENCE,
            {"zone_id": "危険エリア", "duration_seconds": 120},
        )
        self.assertIn("危険エリア", text)
        self.assertIn("120", text)
        self.assertIn("滞在", text)

    def test_zone_entry_predicted(self):
        text = self._narrate(
            EntityEventType.ZONE_ENTRY_PREDICTED,
            {"zone_id": "制限エリア", "eta_seconds": 5},
        )
        self.assertIn("制限エリア", text)
        self.assertIn("5", text)
        self.assertIn("予測", text)

    def test_collision_predicted(self):
        text = self._narrate(
            EntityEventType.COLLISION_PREDICTED,
            {"other_entity_id": 7, "eta_seconds": 3},
        )
        self.assertIn("衝突", text)
        self.assertIn("3", text)


class TestEntityNarration(unittest.TestCase):
    """narrate_entity for various entity states."""

    def setUp(self):
        self.narrator = SceneNarrator()

    def test_stationary_entity(self):
        entities = [_make_entity(1, "person")]
        tracks = {1: _make_track(1, velocity=(0.0, 0.0))}
        ws = _make_world_state(entities=entities, tracks=tracks)
        text = self.narrator.narrate_entity(1, ws)
        self.assertIn("作業員", text)
        self.assertIn("静止", text)

    def test_moving_entity(self):
        entities = [_make_entity(2, "person")]
        tracks = {2: _make_track(2, velocity=(0.02, 0.01))}
        ws = _make_world_state(entities=entities, tracks=tracks)
        text = self.narrator.narrate_entity(2, ws)
        self.assertIn("移動中", text)

    def test_entity_in_zone(self):
        entities = [_make_entity(3, "person", zone_ids=["制限エリアA"])]
        tracks = {3: _make_track(3)}
        ws = _make_world_state(entities=entities, tracks=tracks)
        text = self.narrator.narrate_entity(3, ws)
        self.assertIn("制限エリアA", text)

    def test_unknown_entity(self):
        ws = _make_world_state()
        text = self.narrator.narrate_entity(999, ws)
        self.assertIn("検出されていません", text)

    def test_entity_no_track(self):
        entities = [_make_entity(4, "person")]
        ws = _make_world_state(entities=entities, tracks={})
        text = self.narrator.narrate_entity(4, ws)
        self.assertIn("追跡データなし", text)


class TestChangeSummarization(unittest.TestCase):
    """summarize_changes between two world states."""

    def setUp(self):
        self.narrator = SceneNarrator()

    def test_person_entered(self):
        prev = _make_world_state(
            entities=[_make_entity(1, "person")],
            tracks={1: _make_track(1)},
            timestamp=0.0,
        )
        curr = _make_world_state(
            entities=[_make_entity(1, "person"), _make_entity(2, "person")],
            tracks={1: _make_track(1), 2: _make_track(2)},
            timestamp=30.0,
        )
        text = self.narrator.summarize_changes(curr, prev)
        self.assertIn("1名", text)
        self.assertIn("入場", text)

    def test_person_exited(self):
        prev = _make_world_state(
            entities=[_make_entity(1, "person"), _make_entity(2, "person")],
            tracks={1: _make_track(1), 2: _make_track(2)},
            timestamp=0.0,
        )
        curr = _make_world_state(
            entities=[_make_entity(1, "person")],
            tracks={1: _make_track(1)},
            timestamp=60.0,
        )
        text = self.narrator.summarize_changes(curr, prev)
        self.assertIn("1名", text)
        self.assertIn("退場", text)

    def test_no_changes(self):
        ws = _make_world_state(
            entities=[_make_entity(1, "person")],
            tracks={1: _make_track(1)},
            timestamp=0.0,
        )
        ws2 = _make_world_state(
            entities=[_make_entity(1, "person")],
            tracks={1: _make_track(1)},
            timestamp=10.0,
        )
        text = self.narrator.summarize_changes(ws2, ws)
        self.assertIn("変化はありませんでした", text)

    def test_minutes_time_format(self):
        prev = _make_world_state(timestamp=0.0, tracks={})
        curr = _make_world_state(
            timestamp=120.0,
            tracks={1: _make_track(1)},
            entities=[_make_entity(1, "person")],
        )
        text = self.narrator.summarize_changes(curr, prev)
        self.assertIn("分間", text)

    def test_violation_events_counted(self):
        prev = _make_world_state(timestamp=0.0, tracks={})
        violation_event = EntityEvent(
            event_type=EntityEventType.RULE_VIOLATION,
            entity_id=1,
            timestamp=10.0,
            frame_number=300,
        )
        curr = _make_world_state(
            timestamp=10.0,
            tracks={},
            events=[violation_event],
        )
        text = self.narrator.summarize_changes(curr, prev)
        self.assertIn("1件の違反", text)


class TestStyleDifferences(unittest.TestCase):
    """BRIEF, STANDARD, and DETAILED produce different amounts of text."""

    def setUp(self):
        self.narrator = SceneNarrator()
        entities = [
            _make_entity(1, "person", zone_ids=["zoneA"]),
            _make_entity(2, "person"),
            _make_entity(10, "helmet"),
        ]
        tracks = {
            1: _make_track(1, velocity=(0.0, 0.0)),
            2: _make_track(2, velocity=(0.015, 0.01)),
        }
        zone_occ = {"zoneA": [1]}
        self.ws = _make_world_state(
            entities=entities,
            tracks=tracks,
            zone_occupancy=zone_occ,
            relations=[
                Relation(
                    subject_id=1,
                    predicate=SpatialRelation.NEAR,
                    object_id=2,
                ),
            ],
        )
        self.violations = [
            _make_violation(entity_ids=[2]),
        ]

    def test_brief_shortest(self):
        brief = self.narrator.narrate(self.ws, self.violations, NarrationStyle.BRIEF)
        standard = self.narrator.narrate(self.ws, self.violations, NarrationStyle.STANDARD)
        self.assertLess(len(brief.text_ja), len(standard.text_ja))

    def test_standard_shorter_than_detailed(self):
        standard = self.narrator.narrate(self.ws, self.violations, NarrationStyle.STANDARD)
        detailed = self.narrator.narrate(self.ws, self.violations, NarrationStyle.DETAILED)
        self.assertLess(len(standard.text_ja), len(detailed.text_ja))

    def test_brief_style_field(self):
        result = self.narrator.narrate(self.ws, style=NarrationStyle.BRIEF)
        self.assertEqual(result.style, NarrationStyle.BRIEF)

    def test_standard_style_field(self):
        result = self.narrator.narrate(self.ws, style=NarrationStyle.STANDARD)
        self.assertEqual(result.style, NarrationStyle.STANDARD)

    def test_detailed_style_field(self):
        result = self.narrator.narrate(self.ws, style=NarrationStyle.DETAILED)
        self.assertEqual(result.style, NarrationStyle.DETAILED)

    def test_detailed_includes_relations(self):
        result = self.narrator.narrate(self.ws, style=NarrationStyle.DETAILED)
        self.assertIn("関係", result.text_ja)

    def test_detailed_includes_per_entity(self):
        result = self.narrator.narrate(self.ws, style=NarrationStyle.DETAILED)
        # Should mention confidence for entities
        self.assertIn("信頼度", result.text_ja)

    def test_brief_no_zone_details(self):
        result = self.narrator.narrate(self.ws, style=NarrationStyle.BRIEF)
        # BRIEF should not contain zone occupancy details
        self.assertNotIn("zoneA", result.text_ja)


class TestEnglishOutput(unittest.TestCase):
    """English narration is always generated alongside Japanese."""

    def setUp(self):
        self.narrator = SceneNarrator(language="en")

    def test_english_empty(self):
        ws = _make_world_state()
        result = self.narrator.narrate(ws, style=NarrationStyle.BRIEF)
        self.assertIn("No entities", result.text_en)

    def test_english_with_workers(self):
        ws = _make_world_state(
            entities=[_make_entity(1, "person"), _make_entity(2, "person")],
            tracks={1: _make_track(1), 2: _make_track(2)},
        )
        result = self.narrator.narrate(ws, style=NarrationStyle.BRIEF)
        self.assertIn("2 workers", result.text_en)
        self.assertIn("are present", result.text_en)

    def test_english_single_worker(self):
        ws = _make_world_state(
            entities=[_make_entity(1, "person")],
            tracks={1: _make_track(1)},
        )
        result = self.narrator.narrate(ws, style=NarrationStyle.BRIEF)
        self.assertIn("1 worker", result.text_en)
        self.assertIn("is present", result.text_en)

    def test_english_violations(self):
        ws = _make_world_state(
            entities=[_make_entity(1, "person")],
            tracks={1: _make_track(1)},
        )
        violations = [_make_violation(entity_ids=[1])]
        result = self.narrator.narrate(ws, violations=violations, style=NarrationStyle.STANDARD)
        self.assertIn("Violation", result.text_en)

    def test_english_equipment(self):
        ws = _make_world_state(
            entities=[_make_entity(10, "helmet")],
        )
        result = self.narrator.narrate(ws, style=NarrationStyle.STANDARD)
        self.assertIn("helmet", result.text_en)


class TestEdgeCases(unittest.TestCase):
    """Edge cases: unknown labels, zero entities, etc."""

    def setUp(self):
        self.narrator = SceneNarrator()

    def test_unknown_label(self):
        entities = [_make_entity(1, "xyzzy_widget_99")]
        tracks = {1: _make_track(1, label="xyzzy_widget_99")}
        ws = _make_world_state(entities=entities, tracks=tracks)
        # Should not crash, and label used as-is
        result = self.narrator.narrate(ws, style=NarrationStyle.STANDARD)
        self.assertIsInstance(result, SceneNarration)

    def test_entity_narration_unknown_label(self):
        entities = [_make_entity(1, "alien_artifact")]
        ws = _make_world_state(entities=entities)
        text = self.narrator.narrate_entity(1, ws)
        # Should produce some output (label falls through to raw string)
        self.assertIn("ID: 1", text)

    def test_only_non_person_entities(self):
        entities = [_make_entity(10, "helmet"), _make_entity(11, "cone")]
        ws = _make_world_state(entities=entities)
        result = self.narrator.narrate(ws, style=NarrationStyle.BRIEF)
        self.assertIn("オブジェクト", result.text_ja)
        self.assertIn("object", result.text_en)

    def test_event_narration_missing_entity(self):
        """Event references an entity not in the scene graph."""
        ws = _make_world_state()
        evt = EntityEvent(
            event_type=EntityEventType.ENTERED,
            entity_id=999,
            timestamp=1.0,
            frame_number=30,
        )
        text = self.narrator.narrate_event(evt, ws)
        self.assertIn("エンティティ", text)
        self.assertIn("999", text)

    def test_narration_timestamp_and_frame(self):
        ws = _make_world_state(timestamp=12.5, frame_number=375)
        result = self.narrator.narrate(ws, style=NarrationStyle.BRIEF)
        self.assertAlmostEqual(result.timestamp, 12.5)
        self.assertEqual(result.frame_number, 375)

    def test_default_style_used(self):
        narrator = SceneNarrator(style=NarrationStyle.DETAILED)
        ws = _make_world_state(
            entities=[_make_entity(1, "person")],
            tracks={1: _make_track(1)},
        )
        result = narrator.narrate(ws)
        self.assertEqual(result.style, NarrationStyle.DETAILED)

    def test_zone_occupancy_empty_zone(self):
        ws = _make_world_state(
            zone_occupancy={"restrictedA": []},
        )
        result = self.narrator.narrate(ws, style=NarrationStyle.STANDARD)
        self.assertIn("誰もいません", result.text_ja)

    def test_zone_occupancy_populated(self):
        entities = [_make_entity(1, "person")]
        tracks = {1: _make_track(1)}
        ws = _make_world_state(
            entities=entities,
            tracks=tracks,
            zone_occupancy={"workArea": [1]},
        )
        result = self.narrator.narrate(ws, style=NarrationStyle.STANDARD)
        self.assertIn("workArea", result.text_ja)
        self.assertIn("1名", result.text_ja)

    def test_multiple_violations(self):
        entities = [_make_entity(1, "person"), _make_entity(2, "person")]
        tracks = {1: _make_track(1), 2: _make_track(2)}
        ws = _make_world_state(entities=entities, tracks=tracks)
        violations = [
            _make_violation(entity_ids=[1]),
            _make_violation(
                rule="zone_restricted",
                description_ja="制限エリア侵入",
                entity_ids=[2],
            ),
        ]
        result = self.narrator.narrate(ws, violations=violations, style=NarrationStyle.BRIEF)
        self.assertIn("2件の違反", result.text_ja)

    def test_key_facts_populated(self):
        entities = [_make_entity(1, "person")]
        tracks = {1: _make_track(1)}
        ws = _make_world_state(entities=entities, tracks=tracks)
        result = self.narrator.narrate(ws, style=NarrationStyle.BRIEF)
        self.assertTrue(len(result.key_facts) > 0)

    def test_occluded_track_not_narrated_in_activity(self):
        """LOST tracks should not appear in activity narration."""
        entities = [_make_entity(1, "person")]
        tracks = {1: _make_track(1, state=TrackState.LOST)}
        ws = _make_world_state(entities=entities, tracks=tracks)
        result = self.narrator.narrate(ws, style=NarrationStyle.STANDARD)
        # LOST tracks should be excluded from the activity lines
        # (The overview line still counts persons via entities)
        self.assertNotIn("ID: 1", result.text_ja.split("作業員がいます。")[-1].split("検出機器")[0] if "作業員がいます" in result.text_ja else "")


class TestDetailedEvents(unittest.TestCase):
    """DETAILED style includes events from world state."""

    def setUp(self):
        self.narrator = SceneNarrator()

    def test_detailed_includes_events(self):
        entities = [_make_entity(1, "person")]
        tracks = {1: _make_track(1)}
        events = [
            EntityEvent(
                event_type=EntityEventType.ZONE_ENTRY_PREDICTED,
                entity_id=1,
                timestamp=1.0,
                frame_number=30,
                details={"zone_id": "危険エリア", "eta_seconds": 5},
            ),
        ]
        ws = _make_world_state(entities=entities, tracks=tracks, events=events)
        result = self.narrator.narrate(ws, style=NarrationStyle.DETAILED)
        self.assertIn("危険エリア", result.text_ja)
        self.assertIn("予測", result.text_ja)

    def test_detailed_english_events(self):
        entities = [_make_entity(1, "person")]
        tracks = {1: _make_track(1)}
        events = [
            EntityEvent(
                event_type=EntityEventType.ENTERED,
                entity_id=1,
                timestamp=1.0,
                frame_number=30,
            ),
        ]
        ws = _make_world_state(entities=entities, tracks=tracks, events=events)
        result = self.narrator.narrate(ws, style=NarrationStyle.DETAILED)
        self.assertIn("entered the scene", result.text_en)


if __name__ == "__main__":
    unittest.main()
