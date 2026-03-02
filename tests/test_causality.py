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
