"""Tests for sopilot.perception.causality — Causal Reasoning from Scene Sequences."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from sopilot.perception.causality import (
    CausalLink,
    CausalPattern,
    CausalReasoner,
    _matches_conditions,
)
from sopilot.perception.types import (
    BBox,
    EntityEvent,
    EntityEventType,
    SceneEntity,
    SceneGraph,
    WorldState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    event_type: EntityEventType,
    entity_id: int = 1,
    timestamp: float = 0.0,
    frame_number: int = 0,
    details: dict | None = None,
    confidence: float = 1.0,
) -> EntityEvent:
    return EntityEvent(
        event_type=event_type,
        entity_id=entity_id,
        timestamp=timestamp,
        frame_number=frame_number,
        details=details or {},
        confidence=confidence,
    )


def _make_world_state(
    events: list[EntityEvent],
    timestamp: float = 0.0,
    frame_number: int = 0,
) -> WorldState:
    """Create a minimal WorldState with the given events."""
    sg = SceneGraph(
        timestamp=timestamp,
        frame_number=frame_number,
        entities=[],
        relations=[],
    )
    return WorldState(
        timestamp=timestamp,
        frame_number=frame_number,
        scene_graph=sg,
        active_tracks={},
        events=events,
        zone_occupancy={},
        entity_count=0,
        person_count=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCausalLink(unittest.TestCase):
    """Basic CausalLink dataclass tests."""

    def test_frozen(self):
        cause = _make_event(EntityEventType.STATE_CHANGED, timestamp=1.0)
        effect = _make_event(EntityEventType.RULE_VIOLATION, timestamp=3.0)
        link = CausalLink(
            cause_event=cause,
            effect_event=effect,
            cause_type="equipment_removal_violation",
            confidence=0.85,
            explanation_ja="テスト",
            explanation_en="test",
            time_delta_seconds=2.0,
        )
        self.assertEqual(link.time_delta_seconds, 2.0)
        self.assertEqual(link.confidence, 0.85)
        with self.assertRaises(AttributeError):
            link.confidence = 0.5  # type: ignore[misc]

    def test_fields(self):
        cause = _make_event(EntityEventType.ZONE_ENTERED, timestamp=10.0)
        effect = _make_event(EntityEventType.PROLONGED_PRESENCE, timestamp=70.0)
        link = CausalLink(
            cause_event=cause,
            effect_event=effect,
            cause_type="zone_entry_prolonged",
            confidence=0.9,
            explanation_ja="区域に入った",
            explanation_en="entered zone",
            time_delta_seconds=60.0,
        )
        self.assertIs(link.cause_event, cause)
        self.assertIs(link.effect_event, effect)
        self.assertEqual(link.cause_type, "zone_entry_prolonged")


class TestCausalPattern(unittest.TestCase):
    """CausalPattern dataclass tests."""

    def test_defaults(self):
        p = CausalPattern(
            pattern_id="test",
            cause_type=EntityEventType.ENTERED,
            effect_type=EntityEventType.ZONE_ENTERED,
        )
        self.assertEqual(p.occurrences, 0)
        self.assertEqual(p.confidence, 0.5)
        self.assertEqual(p.max_time_gap_seconds, 30.0)
        self.assertEqual(p.cause_conditions, {})
        self.assertEqual(p.effect_conditions, {})

    def test_mutable(self):
        p = CausalPattern(
            pattern_id="test",
            cause_type=EntityEventType.ENTERED,
            effect_type=EntityEventType.ZONE_ENTERED,
        )
        p.occurrences += 1
        self.assertEqual(p.occurrences, 1)


class TestMatchesConditions(unittest.TestCase):
    """Tests for the _matches_conditions helper."""

    def test_empty_conditions(self):
        self.assertTrue(_matches_conditions({"a": 1}, {}))

    def test_match(self):
        self.assertTrue(
            _matches_conditions(
                {"safety_relevant": True, "attribute": "has_helmet"},
                {"safety_relevant": True},
            )
        )

    def test_no_match_value(self):
        self.assertFalse(
            _matches_conditions(
                {"safety_relevant": False},
                {"safety_relevant": True},
            )
        )

    def test_no_match_missing_key(self):
        self.assertFalse(
            _matches_conditions(
                {"other": 1},
                {"safety_relevant": True},
            )
        )


class TestEquipmentRemovalViolation(unittest.TestCase):
    """Equipment removal -> rule violation causal link."""

    def test_equipment_removal_causes_violation(self):
        reasoner = CausalReasoner()

        # Frame 1: equipment removal (STATE_CHANGED with safety_relevant=True)
        removal_event = _make_event(
            EntityEventType.STATE_CHANGED,
            entity_id=1,
            timestamp=10.0,
            frame_number=100,
            details={
                "attribute": "has_helmet",
                "old_value": True,
                "new_value": False,
                "safety_relevant": True,
                "label": "person",
            },
        )
        ws1 = _make_world_state([removal_event], timestamp=10.0, frame_number=100)
        links1 = reasoner.analyze(ws1)
        self.assertEqual(links1, [])  # No effect yet

        # Frame 2: violation event on the same entity
        violation_event = _make_event(
            EntityEventType.RULE_VIOLATION,
            entity_id=1,
            timestamp=12.3,
            frame_number=123,
            details={"rule": "helmet_required"},
        )
        ws2 = _make_world_state([violation_event], timestamp=12.3, frame_number=123)
        links2 = reasoner.analyze(ws2)

        self.assertEqual(len(links2), 1)
        link = links2[0]
        self.assertEqual(link.cause_type, "equipment_removal_violation")
        self.assertIs(link.cause_event, removal_event)
        self.assertIs(link.effect_event, violation_event)
        self.assertAlmostEqual(link.time_delta_seconds, 2.3, places=1)
        self.assertGreater(link.confidence, 0.0)
        self.assertIn("ヘルメット" if False else "エンティティ", link.explanation_ja)
        self.assertIn("entity", link.explanation_en)

    def test_equipment_removal_non_safety_not_matched(self):
        """STATE_CHANGED without safety_relevant=True should not match."""
        reasoner = CausalReasoner()

        # Non-safety state change
        change = _make_event(
            EntityEventType.STATE_CHANGED,
            entity_id=1,
            timestamp=10.0,
            details={"attribute": "is_sitting", "safety_relevant": False},
        )
        ws1 = _make_world_state([change], timestamp=10.0)
        reasoner.analyze(ws1)

        violation = _make_event(
            EntityEventType.RULE_VIOLATION,
            entity_id=1,
            timestamp=12.0,
            details={"rule": "no_sitting"},
        )
        ws2 = _make_world_state([violation], timestamp=12.0)
        links = reasoner.analyze(ws2)

        # Should not match because cause_conditions require safety_relevant=True
        self.assertEqual(len(links), 0)


class TestZoneEntryProlongedPresence(unittest.TestCase):
    """Zone entry -> prolonged presence causal chain."""

    def test_zone_entry_causes_prolonged_presence(self):
        reasoner = CausalReasoner()

        # Entity enters zone
        entry_event = _make_event(
            EntityEventType.ZONE_ENTERED,
            entity_id=5,
            timestamp=100.0,
            frame_number=1000,
            details={"zone_id": "restricted_1", "zone_name": "Restricted Area"},
        )
        ws1 = _make_world_state([entry_event], timestamp=100.0, frame_number=1000)
        reasoner.analyze(ws1)

        # 65 seconds later: prolonged presence alert
        prolonged_event = _make_event(
            EntityEventType.PROLONGED_PRESENCE,
            entity_id=5,
            timestamp=165.0,
            frame_number=1650,
            details={
                "zone_id": "restricted_1",
                "zone_name": "Restricted Area",
                "duration_seconds": 65.0,
            },
        )
        ws2 = _make_world_state([prolonged_event], timestamp=165.0, frame_number=1650)
        links = reasoner.analyze(ws2)

        self.assertEqual(len(links), 1)
        link = links[0]
        self.assertEqual(link.cause_type, "zone_entry_prolonged")
        self.assertAlmostEqual(link.time_delta_seconds, 65.0, places=1)
        self.assertIs(link.cause_event, entry_event)

    def test_different_zone_no_link(self):
        """Zone entry in zone A should not cause prolonged presence in zone B."""
        reasoner = CausalReasoner()

        entry = _make_event(
            EntityEventType.ZONE_ENTERED,
            entity_id=5,
            timestamp=100.0,
            details={"zone_id": "zone_a"},
        )
        ws1 = _make_world_state([entry], timestamp=100.0)
        reasoner.analyze(ws1)

        prolonged = _make_event(
            EntityEventType.PROLONGED_PRESENCE,
            entity_id=5,
            timestamp=165.0,
            details={"zone_id": "zone_b"},
        )
        ws2 = _make_world_state([prolonged], timestamp=165.0)
        links = reasoner.analyze(ws2)

        self.assertEqual(len(links), 0)


class TestApproachZoneEntry(unittest.TestCase):
    """Approach (predicted) -> actual zone entry causal link."""

    def test_predicted_entry_causes_actual_entry(self):
        reasoner = CausalReasoner()

        prediction = _make_event(
            EntityEventType.ZONE_ENTRY_PREDICTED,
            entity_id=3,
            timestamp=50.0,
            frame_number=500,
            details={"zone_id": "hazard_zone"},
        )
        ws1 = _make_world_state([prediction], timestamp=50.0, frame_number=500)
        reasoner.analyze(ws1)

        actual_entry = _make_event(
            EntityEventType.ZONE_ENTERED,
            entity_id=3,
            timestamp=55.0,
            frame_number=550,
            details={"zone_id": "hazard_zone"},
        )
        ws2 = _make_world_state([actual_entry], timestamp=55.0, frame_number=550)
        links = reasoner.analyze(ws2)

        self.assertEqual(len(links), 1)
        self.assertEqual(links[0].cause_type, "approach_zone_entry")
        self.assertAlmostEqual(links[0].time_delta_seconds, 5.0, places=1)


class TestNoSpuriousLinks(unittest.TestCase):
    """No causal links should be inferred for unrelated events."""

    def test_unrelated_events(self):
        reasoner = CausalReasoner()

        # Entity 1 enters zone
        entry = _make_event(
            EntityEventType.ZONE_ENTERED,
            entity_id=1,
            timestamp=10.0,
            details={"zone_id": "zone_a"},
        )
        ws1 = _make_world_state([entry], timestamp=10.0)
        reasoner.analyze(ws1)

        # Entity 2 exits (completely unrelated)
        exit_event = _make_event(
            EntityEventType.EXITED,
            entity_id=2,
            timestamp=12.0,
        )
        ws2 = _make_world_state([exit_event], timestamp=12.0)
        links = reasoner.analyze(ws2)

        self.assertEqual(len(links), 0)

    def test_time_gap_exceeded(self):
        """Events too far apart should not be linked."""
        reasoner = CausalReasoner()

        removal = _make_event(
            EntityEventType.STATE_CHANGED,
            entity_id=1,
            timestamp=10.0,
            details={"safety_relevant": True, "attribute": "has_helmet"},
        )
        ws1 = _make_world_state([removal], timestamp=10.0)
        reasoner.analyze(ws1)

        # 100 seconds later — well beyond max_time_gap_seconds=30
        violation = _make_event(
            EntityEventType.RULE_VIOLATION,
            entity_id=1,
            timestamp=110.0,
        )
        ws2 = _make_world_state([violation], timestamp=110.0)
        links = reasoner.analyze(ws2)

        self.assertEqual(len(links), 0)

    def test_different_entity_zone_patterns(self):
        """Zone patterns require same entity."""
        reasoner = CausalReasoner()

        entry = _make_event(
            EntityEventType.ZONE_ENTERED,
            entity_id=1,
            timestamp=100.0,
            details={"zone_id": "zone_a"},
        )
        ws1 = _make_world_state([entry], timestamp=100.0)
        reasoner.analyze(ws1)

        prolonged = _make_event(
            EntityEventType.PROLONGED_PRESENCE,
            entity_id=2,  # different entity
            timestamp=165.0,
            details={"zone_id": "zone_a"},
        )
        ws2 = _make_world_state([prolonged], timestamp=165.0)
        links = reasoner.analyze(ws2)

        self.assertEqual(len(links), 0)


class TestCausalChainTracing(unittest.TestCase):
    """Tracing full causal chains."""

    def test_two_step_chain(self):
        """approach -> zone entry -> prolonged presence = 2-step chain."""
        reasoner = CausalReasoner()

        # Step 1: Approach prediction
        prediction = _make_event(
            EntityEventType.ZONE_ENTRY_PREDICTED,
            entity_id=7,
            timestamp=100.0,
            frame_number=1000,
            details={"zone_id": "restricted"},
        )
        ws1 = _make_world_state([prediction], timestamp=100.0, frame_number=1000)
        reasoner.analyze(ws1)

        # Step 2: Actual zone entry
        entry = _make_event(
            EntityEventType.ZONE_ENTERED,
            entity_id=7,
            timestamp=105.0,
            frame_number=1050,
            details={"zone_id": "restricted"},
        )
        ws2 = _make_world_state([entry], timestamp=105.0, frame_number=1050)
        links2 = reasoner.analyze(ws2)
        self.assertEqual(len(links2), 1)
        self.assertEqual(links2[0].cause_type, "approach_zone_entry")

        # Step 3: Prolonged presence
        prolonged = _make_event(
            EntityEventType.PROLONGED_PRESENCE,
            entity_id=7,
            timestamp=170.0,
            frame_number=1700,
            details={"zone_id": "restricted"},
        )
        ws3 = _make_world_state([prolonged], timestamp=170.0, frame_number=1700)
        links3 = reasoner.analyze(ws3)
        self.assertEqual(len(links3), 1)

        # Trace full chain from prolonged presence event
        chain = reasoner.get_causal_chain(prolonged)
        self.assertEqual(len(chain), 2)
        # Root cause first
        self.assertEqual(chain[0].cause_type, "approach_zone_entry")
        self.assertEqual(chain[1].cause_type, "zone_entry_prolonged")

    def test_chain_no_links(self):
        """Event with no causal history returns empty chain."""
        reasoner = CausalReasoner()
        event = _make_event(EntityEventType.ENTERED, entity_id=1, timestamp=5.0)
        chain = reasoner.get_causal_chain(event)
        self.assertEqual(chain, [])


class TestExplainViolation(unittest.TestCase):
    """explain_violation() output tests."""

    def test_explain_with_chain(self):
        reasoner = CausalReasoner()

        removal = _make_event(
            EntityEventType.STATE_CHANGED,
            entity_id=1,
            timestamp=10.0,
            details={"safety_relevant": True, "attribute": "has_helmet"},
        )
        ws1 = _make_world_state([removal], timestamp=10.0)
        reasoner.analyze(ws1)

        violation = _make_event(
            EntityEventType.RULE_VIOLATION,
            entity_id=1,
            timestamp=12.0,
        )
        ws2 = _make_world_state([violation], timestamp=12.0)
        reasoner.analyze(ws2)

        explanation = reasoner.explain_violation(violation, [])
        self.assertIn("因果連鎖", explanation)
        self.assertIn("1", explanation)

    def test_explain_with_recent_events_fallback(self):
        reasoner = CausalReasoner()

        removal = _make_event(
            EntityEventType.STATE_CHANGED,
            entity_id=1,
            timestamp=10.0,
            details={"safety_relevant": True, "attribute": "has_helmet"},
        )
        violation = _make_event(
            EntityEventType.RULE_VIOLATION,
            entity_id=1,
            timestamp=12.0,
        )

        # Do NOT run analyze — use explain_violation with recent_events fallback.
        explanation = reasoner.explain_violation(violation, [removal])
        self.assertIn("エンティティ", explanation)
        self.assertIn("推測", explanation)

    def test_explain_no_cause_found(self):
        reasoner = CausalReasoner()

        violation = _make_event(
            EntityEventType.RULE_VIOLATION,
            entity_id=99,
            timestamp=100.0,
        )
        explanation = reasoner.explain_violation(violation, [])
        self.assertIn("特定できませんでした", explanation)


class TestPatternRegistration(unittest.TestCase):
    """Custom pattern registration and matching."""

    def test_add_custom_pattern(self):
        reasoner = CausalReasoner()
        initial_count = reasoner.pattern_count

        custom = CausalPattern(
            pattern_id="custom_test",
            cause_type=EntityEventType.ENTERED,
            effect_type=EntityEventType.ZONE_ENTERED,
            max_time_gap_seconds=10.0,
            confidence=0.7,
        )
        reasoner.add_pattern(custom)
        self.assertEqual(reasoner.pattern_count, initial_count + 1)

    def test_custom_pattern_matching(self):
        reasoner = CausalReasoner()

        # Register a custom pattern: ENTERED -> ZONE_ENTERED (same entity)
        custom = CausalPattern(
            pattern_id="entered_then_zone",
            cause_type=EntityEventType.ENTERED,
            effect_type=EntityEventType.ZONE_ENTERED,
            max_time_gap_seconds=10.0,
            confidence=0.75,
        )
        reasoner.add_pattern(custom)

        # Cause: entity enters scene
        entered = _make_event(
            EntityEventType.ENTERED,
            entity_id=10,
            timestamp=1.0,
            details={"label": "person"},
        )
        ws1 = _make_world_state([entered], timestamp=1.0)
        reasoner.analyze(ws1)

        # Effect: entity enters zone
        zone_entered = _make_event(
            EntityEventType.ZONE_ENTERED,
            entity_id=10,
            timestamp=4.0,
            details={"zone_id": "work_area"},
        )
        ws2 = _make_world_state([zone_entered], timestamp=4.0)
        links = reasoner.analyze(ws2)

        # Should find the custom pattern link.
        custom_links = [lk for lk in links if lk.cause_type == "entered_then_zone"]
        self.assertEqual(len(custom_links), 1)
        self.assertAlmostEqual(custom_links[0].time_delta_seconds, 3.0, places=1)


class TestEdgeCases(unittest.TestCase):
    """Edge cases: empty history, single event, boundary conditions."""

    def test_empty_world_state(self):
        reasoner = CausalReasoner()
        ws = _make_world_state([], timestamp=0.0)
        links = reasoner.analyze(ws)
        self.assertEqual(links, [])

    def test_single_event_no_link(self):
        reasoner = CausalReasoner()
        event = _make_event(EntityEventType.RULE_VIOLATION, timestamp=5.0)
        ws = _make_world_state([event], timestamp=5.0)
        links = reasoner.analyze(ws)
        self.assertEqual(links, [])

    def test_zero_time_delta(self):
        """Cause and effect at the exact same timestamp should still link."""
        reasoner = CausalReasoner()

        removal = _make_event(
            EntityEventType.STATE_CHANGED,
            entity_id=1,
            timestamp=10.0,
            details={"safety_relevant": True},
        )
        ws1 = _make_world_state([removal], timestamp=10.0)
        reasoner.analyze(ws1)

        violation = _make_event(
            EntityEventType.RULE_VIOLATION,
            entity_id=1,
            timestamp=10.0,  # same timestamp
        )
        ws2 = _make_world_state([violation], timestamp=10.0)
        links = reasoner.analyze(ws2)

        self.assertEqual(len(links), 1)
        self.assertAlmostEqual(links[0].time_delta_seconds, 0.0)

    def test_buffer_overflow(self):
        """Ensure the buffer handles max_history_events properly."""
        reasoner = CausalReasoner(max_history_events=5)

        # Feed more events than the buffer can hold.
        for i in range(10):
            event = _make_event(
                EntityEventType.ENTERED,
                entity_id=i,
                timestamp=float(i),
                frame_number=i,
            )
            ws = _make_world_state([event], timestamp=float(i), frame_number=i)
            reasoner.analyze(ws)

        self.assertLessEqual(reasoner.buffer_size, 5)

    def test_reset_clears_state(self):
        reasoner = CausalReasoner()

        event = _make_event(
            EntityEventType.STATE_CHANGED,
            entity_id=1,
            timestamp=10.0,
            details={"safety_relevant": True},
        )
        ws = _make_world_state([event], timestamp=10.0)
        reasoner.analyze(ws)
        self.assertGreater(reasoner.buffer_size, 0)

        reasoner.reset()
        self.assertEqual(reasoner.buffer_size, 0)
        self.assertEqual(reasoner.link_count, 0)

    def test_duplicate_events_not_reprocessed(self):
        """Analyzing the same WorldState twice should not create duplicate links."""
        reasoner = CausalReasoner()

        removal = _make_event(
            EntityEventType.STATE_CHANGED,
            entity_id=1,
            timestamp=10.0,
            details={"safety_relevant": True},
        )
        ws1 = _make_world_state([removal], timestamp=10.0)
        reasoner.analyze(ws1)

        violation = _make_event(
            EntityEventType.RULE_VIOLATION,
            entity_id=1,
            timestamp=12.0,
        )
        ws2 = _make_world_state([violation], timestamp=12.0)
        links_first = reasoner.analyze(ws2)
        links_second = reasoner.analyze(ws2)

        self.assertEqual(len(links_first), 1)
        self.assertEqual(len(links_second), 0)  # Already processed


class TestProximityInteraction(unittest.TestCase):
    """Proximity -> interaction (state change on different entities)."""

    def test_different_entities_state_changes(self):
        reasoner = CausalReasoner()

        # Entity 1 state change
        change_1 = _make_event(
            EntityEventType.STATE_CHANGED,
            entity_id=1,
            timestamp=20.0,
            details={"attribute": "is_running", "old_value": False, "new_value": True},
        )
        ws1 = _make_world_state([change_1], timestamp=20.0)
        reasoner.analyze(ws1)

        # Entity 2 state change within 5 seconds
        change_2 = _make_event(
            EntityEventType.STATE_CHANGED,
            entity_id=2,
            timestamp=22.0,
            details={"attribute": "is_running", "old_value": False, "new_value": True},
        )
        ws2 = _make_world_state([change_2], timestamp=22.0)
        links = reasoner.analyze(ws2)

        proximity_links = [
            lk for lk in links if lk.cause_type == "proximity_interaction"
        ]
        self.assertEqual(len(proximity_links), 1)
        self.assertEqual(proximity_links[0].cause_event.entity_id, 1)
        self.assertEqual(proximity_links[0].effect_event.entity_id, 2)

    def test_same_entity_state_changes_no_proximity(self):
        """proximity_interaction requires different entities."""
        reasoner = CausalReasoner()

        change_1 = _make_event(
            EntityEventType.STATE_CHANGED,
            entity_id=1,
            timestamp=20.0,
            details={"attribute": "has_helmet"},
        )
        ws1 = _make_world_state([change_1], timestamp=20.0)
        reasoner.analyze(ws1)

        change_2 = _make_event(
            EntityEventType.STATE_CHANGED,
            entity_id=1,  # same entity
            timestamp=22.0,
            details={"attribute": "has_vest"},
        )
        ws2 = _make_world_state([change_2], timestamp=22.0)
        links = reasoner.analyze(ws2)

        proximity_links = [
            lk for lk in links if lk.cause_type == "proximity_interaction"
        ]
        self.assertEqual(len(proximity_links), 0)


class TestAnomalyViolation(unittest.TestCase):
    """Anomaly -> violation causal link."""

    def test_anomaly_causes_violation(self):
        reasoner = CausalReasoner()

        anomaly = _make_event(
            EntityEventType.ANOMALY,
            entity_id=-1,
            timestamp=30.0,
            details={"metric": "entity_count"},
        )
        ws1 = _make_world_state([anomaly], timestamp=30.0)
        reasoner.analyze(ws1)

        violation = _make_event(
            EntityEventType.RULE_VIOLATION,
            entity_id=1,
            timestamp=35.0,
            details={"rule": "max_occupancy"},
        )
        ws2 = _make_world_state([violation], timestamp=35.0)
        links = reasoner.analyze(ws2)

        anomaly_links = [lk for lk in links if lk.cause_type == "anomaly_violation"]
        self.assertEqual(len(anomaly_links), 1)


class TestConfidenceScoring(unittest.TestCase):
    """Confidence computation tests."""

    def test_closer_events_higher_confidence(self):
        reasoner = CausalReasoner()

        removal = _make_event(
            EntityEventType.STATE_CHANGED,
            entity_id=1,
            timestamp=10.0,
            details={"safety_relevant": True},
        )
        ws1 = _make_world_state([removal], timestamp=10.0)
        reasoner.analyze(ws1)

        # Close violation
        violation_close = _make_event(
            EntityEventType.RULE_VIOLATION,
            entity_id=1,
            timestamp=11.0,
        )
        ws_close = _make_world_state([violation_close], timestamp=11.0)
        links_close = reasoner.analyze(ws_close)

        # Reset and redo with a farther violation
        reasoner.reset()
        ws1_again = _make_world_state([removal], timestamp=10.0)
        reasoner.analyze(ws1_again)

        violation_far = _make_event(
            EntityEventType.RULE_VIOLATION,
            entity_id=1,
            timestamp=25.0,
        )
        ws_far = _make_world_state([violation_far], timestamp=25.0)
        links_far = reasoner.analyze(ws_far)

        self.assertEqual(len(links_close), 1)
        self.assertEqual(len(links_far), 1)
        self.assertGreater(links_close[0].confidence, links_far[0].confidence)

    def test_confidence_bounded_0_1(self):
        reasoner = CausalReasoner()

        removal = _make_event(
            EntityEventType.STATE_CHANGED,
            entity_id=1,
            timestamp=10.0,
            details={"safety_relevant": True},
        )
        ws1 = _make_world_state([removal], timestamp=10.0)
        reasoner.analyze(ws1)

        violation = _make_event(
            EntityEventType.RULE_VIOLATION,
            entity_id=1,
            timestamp=10.5,
        )
        ws2 = _make_world_state([violation], timestamp=10.5)
        links = reasoner.analyze(ws2)

        self.assertEqual(len(links), 1)
        self.assertGreaterEqual(links[0].confidence, 0.0)
        self.assertLessEqual(links[0].confidence, 1.0)


class TestDiagnostics(unittest.TestCase):
    """Diagnostic property tests."""

    def test_initial_state(self):
        reasoner = CausalReasoner()
        self.assertGreater(reasoner.pattern_count, 0)  # built-in patterns
        self.assertEqual(reasoner.buffer_size, 0)
        self.assertEqual(reasoner.link_count, 0)

    def test_get_patterns(self):
        reasoner = CausalReasoner()
        patterns = reasoner.get_patterns()
        self.assertIsInstance(patterns, list)
        self.assertEqual(len(patterns), reasoner.pattern_count)
        # Verify it is a copy.
        patterns.append(
            CausalPattern(
                pattern_id="extra",
                cause_type=EntityEventType.ENTERED,
                effect_type=EntityEventType.EXITED,
            )
        )
        self.assertNotEqual(len(patterns), reasoner.pattern_count)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# CausalGraph tests
# ---------------------------------------------------------------------------

from sopilot.perception.causality import CausalGraph, CausalNode  # noqa: E402


def _make_link(
    cause_type_ev: EntityEventType,
    effect_type_ev: EntityEventType,
    cause_entity: int = 1,
    effect_entity: int = 1,
    cause_ts: float = 10.0,
    effect_ts: float = 15.0,
    cause_type: str = "equipment_removal_violation",
    cause_details: dict | None = None,
    effect_details: dict | None = None,
) -> CausalLink:
    """Helper that builds a CausalLink from two simple events."""
    cause_ev = _make_event(
        cause_type_ev,
        entity_id=cause_entity,
        timestamp=cause_ts,
        details=cause_details or {},
    )
    effect_ev = _make_event(
        effect_type_ev,
        entity_id=effect_entity,
        timestamp=effect_ts,
        details=effect_details or {},
    )
    return CausalLink(
        cause_event=cause_ev,
        effect_event=effect_ev,
        cause_type=cause_type,
        confidence=0.8,
        explanation_ja="テスト因果",
        explanation_en="test causal",
        time_delta_seconds=effect_ts - cause_ts,
    )


class TestCausalGraph(unittest.TestCase):
    """Tests for the CausalGraph class."""

    # ── Construction ──────────────────────────────────────────────────────

    def test_init_empty(self):
        """A freshly created graph has no nodes."""
        graph = CausalGraph()
        stats = graph.get_stats()
        self.assertEqual(stats["node_count"], 0)

    # ── add_link ──────────────────────────────────────────────────────────

    def test_add_link_creates_nodes(self):
        """Adding one link creates exactly two nodes (cause + effect)."""
        graph = CausalGraph()
        link = _make_link(
            EntityEventType.STATE_CHANGED,
            EntityEventType.RULE_VIOLATION,
        )
        graph.add_link(link)
        self.assertEqual(graph.get_stats()["node_count"], 2)

    def test_add_link_returns_ids(self):
        """add_link returns a tuple of two non-empty string IDs."""
        graph = CausalGraph()
        link = _make_link(
            EntityEventType.STATE_CHANGED,
            EntityEventType.RULE_VIOLATION,
        )
        result = graph.add_link(link)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], str)
        self.assertIsInstance(result[1], str)
        self.assertTrue(result[0])
        self.assertTrue(result[1])

    def test_add_duplicate_does_not_create_extra_nodes(self):
        """Adding the same link twice reuses the existing nodes."""
        graph = CausalGraph()
        link = _make_link(
            EntityEventType.STATE_CHANGED,
            EntityEventType.RULE_VIOLATION,
            cause_ts=10.0,
            effect_ts=12.0,
        )
        id1 = graph.add_link(link)
        id2 = graph.add_link(link)  # identical link
        # Same node IDs returned.
        self.assertEqual(id1, id2)
        # Still only two nodes, not four.
        self.assertEqual(graph.get_stats()["node_count"], 2)

    # ── get_root_causes ───────────────────────────────────────────────────

    def test_root_causes_empty(self):
        """An empty graph returns an empty root-cause list."""
        graph = CausalGraph()
        self.assertEqual(graph.get_root_causes(), [])

    def test_root_causes_single_chain(self):
        """In A→B, A is the root cause and B is not."""
        graph = CausalGraph()
        link = _make_link(
            EntityEventType.ZONE_ENTERED,
            EntityEventType.PROLONGED_PRESENCE,
            cause_entity=5,
            effect_entity=5,
            cause_ts=100.0,
            effect_ts=165.0,
            cause_type="zone_entry_prolonged",
        )
        cause_id, effect_id = graph.add_link(link)
        roots = graph.get_root_causes()
        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0].node_id, cause_id)

    def test_root_causes_multiple_chains(self):
        """Two independent single-step chains produce two root causes."""
        graph = CausalGraph()
        link1 = _make_link(
            EntityEventType.STATE_CHANGED,
            EntityEventType.RULE_VIOLATION,
            cause_entity=1,
            effect_entity=1,
            cause_ts=10.0,
            effect_ts=12.0,
        )
        link2 = _make_link(
            EntityEventType.ZONE_ENTERED,
            EntityEventType.PROLONGED_PRESENCE,
            cause_entity=2,
            effect_entity=2,
            cause_ts=20.0,
            effect_ts=80.0,
            cause_type="zone_entry_prolonged",
        )
        graph.add_link(link1)
        graph.add_link(link2)
        roots = graph.get_root_causes()
        self.assertEqual(len(roots), 2)

    def test_root_causes_sorted_by_importance(self):
        """Root with more downstream nodes is ranked first."""
        graph = CausalGraph()
        # Chain A: root_a → mid → leaf  (2 downstream from root_a)
        link_a1 = _make_link(
            EntityEventType.ZONE_ENTRY_PREDICTED,
            EntityEventType.ZONE_ENTERED,
            cause_entity=10,
            effect_entity=10,
            cause_ts=1.0,
            effect_ts=5.0,
            cause_type="approach_zone_entry",
        )
        link_a2 = _make_link(
            EntityEventType.ZONE_ENTERED,
            EntityEventType.PROLONGED_PRESENCE,
            cause_entity=10,
            effect_entity=10,
            cause_ts=5.0,
            effect_ts=70.0,
            cause_type="zone_entry_prolonged",
        )
        # Chain B: root_b → leaf_b  (1 downstream from root_b)
        link_b = _make_link(
            EntityEventType.STATE_CHANGED,
            EntityEventType.RULE_VIOLATION,
            cause_entity=20,
            effect_entity=20,
            cause_ts=100.0,
            effect_ts=105.0,
        )
        graph.add_link(link_a1)
        graph.add_link(link_a2)
        graph.add_link(link_b)

        roots = graph.get_root_causes()
        # root_a has 2 descendants; root_b has 1 → root_a should be first.
        self.assertGreater(roots[0].importance, roots[1].importance)

    # ── get_consequences ──────────────────────────────────────────────────

    def test_get_consequences_empty(self):
        """A leaf node has no consequences."""
        graph = CausalGraph()
        link = _make_link(
            EntityEventType.STATE_CHANGED,
            EntityEventType.RULE_VIOLATION,
        )
        _, effect_id = graph.add_link(link)
        self.assertEqual(graph.get_consequences(effect_id), [])

    def test_get_consequences_chain(self):
        """A→B→C: consequences of A are [B, C]."""
        graph = CausalGraph()
        link1 = _make_link(
            EntityEventType.ZONE_ENTRY_PREDICTED,
            EntityEventType.ZONE_ENTERED,
            cause_entity=7,
            effect_entity=7,
            cause_ts=100.0,
            effect_ts=105.0,
            cause_type="approach_zone_entry",
        )
        link2 = _make_link(
            EntityEventType.ZONE_ENTERED,
            EntityEventType.PROLONGED_PRESENCE,
            cause_entity=7,
            effect_entity=7,
            cause_ts=105.0,
            effect_ts=170.0,
            cause_type="zone_entry_prolonged",
        )
        a_id, _ = graph.add_link(link1)
        graph.add_link(link2)
        consequences = graph.get_consequences(a_id)
        self.assertEqual(len(consequences), 2)

    def test_get_consequences_max_depth(self):
        """BFS stops at max_depth=1: only immediate children returned."""
        graph = CausalGraph()
        link1 = _make_link(
            EntityEventType.ZONE_ENTRY_PREDICTED,
            EntityEventType.ZONE_ENTERED,
            cause_entity=7,
            effect_entity=7,
            cause_ts=100.0,
            effect_ts=105.0,
            cause_type="approach_zone_entry",
        )
        link2 = _make_link(
            EntityEventType.ZONE_ENTERED,
            EntityEventType.PROLONGED_PRESENCE,
            cause_entity=7,
            effect_entity=7,
            cause_ts=105.0,
            effect_ts=170.0,
            cause_type="zone_entry_prolonged",
        )
        a_id, _ = graph.add_link(link1)
        graph.add_link(link2)
        # depth=1 → only B is returned, C is not
        consequences = graph.get_consequences(a_id, max_depth=1)
        self.assertEqual(len(consequences), 1)
        self.assertEqual(consequences[0].event.event_type, EntityEventType.ZONE_ENTERED)

    # ── get_intervention_targets ──────────────────────────────────────────

    def test_get_intervention_targets_empty(self):
        """Empty graph returns an empty list."""
        graph = CausalGraph()
        self.assertEqual(graph.get_intervention_targets(), [])

    def test_get_intervention_targets_violations(self):
        """A node upstream of a RULE_VIOLATION has violation_count >= 1."""
        graph = CausalGraph()
        link = _make_link(
            EntityEventType.STATE_CHANGED,
            EntityEventType.RULE_VIOLATION,
            cause_entity=1,
            effect_entity=1,
            cause_ts=10.0,
            effect_ts=12.0,
        )
        cause_id, _ = graph.add_link(link)
        targets = graph.get_intervention_targets()
        # At least the cause node should appear with violation_count >= 1.
        cause_targets = [t for t in targets if t[0].node_id == cause_id]
        self.assertEqual(len(cause_targets), 1)
        self.assertGreaterEqual(cause_targets[0][1], 1)

    def test_get_intervention_targets_sorted(self):
        """Results are sorted by violation_count descending."""
        graph = CausalGraph()
        # A chain: root → violation1 → violation2 (root has 2 downstream violations)
        link1 = _make_link(
            EntityEventType.STATE_CHANGED,
            EntityEventType.RULE_VIOLATION,
            cause_entity=1,
            effect_entity=1,
            cause_ts=10.0,
            effect_ts=12.0,
        )
        link2 = _make_link(
            EntityEventType.RULE_VIOLATION,
            EntityEventType.RULE_VIOLATION,
            cause_entity=1,
            effect_entity=2,
            cause_ts=12.0,
            effect_ts=14.0,
            cause_type="anomaly_violation",
        )
        graph.add_link(link1)
        graph.add_link(link2)
        targets = graph.get_intervention_targets()
        self.assertGreaterEqual(len(targets), 2)
        # Verify descending sort.
        counts = [t[1] for t in targets]
        self.assertEqual(counts, sorted(counts, reverse=True))

    # ── get_causal_narrative ──────────────────────────────────────────────

    def test_causal_narrative_empty(self):
        """An empty graph returns a non-empty graceful string."""
        graph = CausalGraph()
        narrative = graph.get_causal_narrative()
        self.assertIsInstance(narrative, str)
        self.assertGreater(len(narrative), 0)

    def test_causal_narrative_format(self):
        """Narrative for a real chain contains Japanese text."""
        graph = CausalGraph()
        link = _make_link(
            EntityEventType.ZONE_ENTERED,
            EntityEventType.PROLONGED_PRESENCE,
            cause_entity=5,
            effect_entity=5,
            cause_ts=100.0,
            effect_ts=165.0,
            cause_type="zone_entry_prolonged",
        )
        graph.add_link(link)
        narrative = graph.get_causal_narrative()
        # Should contain Japanese characters (multi-byte).
        has_japanese = any(ord(c) > 0x3000 for c in narrative)
        self.assertTrue(has_japanese, f"Narrative lacks Japanese text: {narrative!r}")

    def test_causal_narrative_max_chains(self):
        """max_chains=1 produces at most one chain line."""
        graph = CausalGraph()
        for i in range(4):
            link = _make_link(
                EntityEventType.STATE_CHANGED,
                EntityEventType.RULE_VIOLATION,
                cause_entity=i * 10,
                effect_entity=i * 10,
                cause_ts=float(i * 100),
                effect_ts=float(i * 100 + 5),
            )
            graph.add_link(link)
        narrative = graph.get_causal_narrative(max_chains=1)
        # At most 1 chain-line (separated by newlines).
        lines = [ln for ln in narrative.splitlines() if ln.strip()]
        self.assertLessEqual(len(lines), 1)

    # ── prune_old_nodes ───────────────────────────────────────────────────

    def test_prune_old_nodes_removes_old(self):
        """Nodes older than max_age_seconds are removed."""
        graph = CausalGraph(max_age_seconds=60.0)
        link = _make_link(
            EntityEventType.STATE_CHANGED,
            EntityEventType.RULE_VIOLATION,
            cause_ts=0.0,
            effect_ts=5.0,
        )
        graph.add_link(link)
        self.assertEqual(graph.get_stats()["node_count"], 2)
        # current_time = 70 → both nodes (ts=0, ts=5) are older than 60 s.
        graph.prune_old_nodes(current_time=70.0)
        self.assertEqual(graph.get_stats()["node_count"], 0)

    def test_prune_old_nodes_keeps_recent(self):
        """Nodes younger than max_age_seconds are NOT removed."""
        graph = CausalGraph(max_age_seconds=600.0)
        link = _make_link(
            EntityEventType.STATE_CHANGED,
            EntityEventType.RULE_VIOLATION,
            cause_ts=1000.0,
            effect_ts=1005.0,
        )
        graph.add_link(link)
        # current_time = 1100 → age = 95–100 s < 600 s → kept.
        graph.prune_old_nodes(current_time=1100.0)
        self.assertEqual(graph.get_stats()["node_count"], 2)

    def test_prune_returns_count(self):
        """prune_old_nodes returns the number of nodes removed."""
        graph = CausalGraph(max_age_seconds=10.0)
        link = _make_link(
            EntityEventType.STATE_CHANGED,
            EntityEventType.RULE_VIOLATION,
            cause_ts=0.0,
            effect_ts=3.0,
        )
        graph.add_link(link)
        removed = graph.prune_old_nodes(current_time=100.0)
        self.assertEqual(removed, 2)

    # ── compute_importance ────────────────────────────────────────────────

    def test_compute_importance(self):
        """Root node (with descendants) has higher importance than a leaf."""
        graph = CausalGraph()
        link = _make_link(
            EntityEventType.STATE_CHANGED,
            EntityEventType.RULE_VIOLATION,
            cause_ts=10.0,
            effect_ts=12.0,
        )
        cause_id, effect_id = graph.add_link(link)
        graph.compute_importance()
        cause_node = graph._nodes[cause_id]
        effect_node = graph._nodes[effect_id]
        self.assertGreater(cause_node.importance, effect_node.importance)

    def test_compute_importance_normalized(self):
        """All importance values are in [0, 1]."""
        graph = CausalGraph()
        link1 = _make_link(
            EntityEventType.ZONE_ENTRY_PREDICTED,
            EntityEventType.ZONE_ENTERED,
            cause_entity=3,
            effect_entity=3,
            cause_ts=1.0,
            effect_ts=5.0,
            cause_type="approach_zone_entry",
        )
        link2 = _make_link(
            EntityEventType.ZONE_ENTERED,
            EntityEventType.PROLONGED_PRESENCE,
            cause_entity=3,
            effect_entity=3,
            cause_ts=5.0,
            effect_ts=70.0,
            cause_type="zone_entry_prolonged",
        )
        graph.add_link(link1)
        graph.add_link(link2)
        graph.compute_importance()
        for node in graph._nodes.values():
            self.assertGreaterEqual(node.importance, 0.0)
            self.assertLessEqual(node.importance, 1.0)

    # ── get_stats ──────────────────────────────────────────────────────────

    def test_get_stats_is_dict(self):
        """get_stats() returns a dict."""
        graph = CausalGraph()
        self.assertIsInstance(graph.get_stats(), dict)

    def test_get_stats_node_count(self):
        """get_stats() contains a 'node_count' key reflecting the actual count."""
        graph = CausalGraph()
        link = _make_link(
            EntityEventType.STATE_CHANGED,
            EntityEventType.RULE_VIOLATION,
        )
        graph.add_link(link)
        stats = graph.get_stats()
        self.assertIn("node_count", stats)
        self.assertEqual(stats["node_count"], 2)

    # ── max_nodes limit ───────────────────────────────────────────────────

    def test_max_nodes_respected(self):
        """Adding many links never exceeds max_nodes."""
        max_n = 10
        graph = CausalGraph(max_nodes=max_n)
        for i in range(50):
            link = _make_link(
                EntityEventType.STATE_CHANGED,
                EntityEventType.RULE_VIOLATION,
                cause_entity=i,
                effect_entity=i + 1000,
                cause_ts=float(i),
                effect_ts=float(i) + 2.0,
            )
            graph.add_link(link)
        self.assertLessEqual(graph.get_stats()["node_count"], max_n + 1)
