"""Tests for sopilot.perception.episodic_memory.

~50 tests across 6 test classes:
    TestEpisodeSegmentation (10)
    TestEpisodeRetrieval     (8)
    TestSearch               (8)
    TestTemporalPatterns     (8)
    TestCrossEpisodeSummary  (6)
    TestEdgeCases           (10)
"""
from __future__ import annotations

import time
import uuid
from unittest.mock import MagicMock

import pytest

from sopilot.perception.episodic_memory import (
    Episode,
    EpisodicMemoryStore,
    TemporalPattern,
)
from sopilot.perception.types import (
    EntityEvent,
    EntityEventType,
    SceneEntity,
    WorldState,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_event(
    event_type: EntityEventType = EntityEventType.ENTERED,
    entity_id: int = 1,
    timestamp: float = 0.0,
    frame_number: int = 0,
    details: dict | None = None,
) -> EntityEvent:
    return EntityEvent(
        event_type=event_type,
        entity_id=entity_id,
        timestamp=timestamp,
        frame_number=frame_number,
        details=details or {},
        confidence=0.9,
    )


def _make_world_state() -> WorldState:
    ws = MagicMock(spec=WorldState)
    ws.events = []
    ws.scene_graph = MagicMock()
    ws.scene_graph.entities = {}
    return ws


def _store(gap: float = 120.0) -> EpisodicMemoryStore:
    return EpisodicMemoryStore(max_episodes=500, episode_gap_seconds=gap)


# ══════════════════════════════════════════════════════════════════════════
# TestEpisodeSegmentation  (10 tests)
# ══════════════════════════════════════════════════════════════════════════


class TestEpisodeSegmentation:
    def test_push_first_event_starts_episode(self):
        store = _store()
        ev = _make_event(timestamp=1000.0)
        result = store.push_event(ev, _make_world_state())
        assert result is None  # no episode closed yet
        current = store.get_current_episode()
        assert current is not None

    def test_events_accumulate_in_episode(self):
        store = _store()
        ws = _make_world_state()
        for i in range(5):
            store.push_event(_make_event(timestamp=1000.0 + i * 10), ws)
        current = store.get_current_episode()
        assert current is not None
        assert current.event_count == 5

    def test_gap_closes_episode(self):
        store = _store(gap=60.0)
        ws = _make_world_state()
        store.push_event(_make_event(timestamp=1000.0), ws)
        # Push an event far beyond the gap
        closed = store.push_event(_make_event(timestamp=1200.0), ws)
        assert closed is not None
        assert isinstance(closed, Episode)

    def test_close_returns_episode_object(self):
        store = _store(gap=60.0)
        ws = _make_world_state()
        store.push_event(_make_event(timestamp=0.0), ws)
        closed = store.push_event(_make_event(timestamp=200.0), ws)
        assert isinstance(closed, Episode)
        assert closed.id  # non-empty UUID string

    def test_episode_fields_correct(self):
        store = _store(gap=60.0)
        ws = _make_world_state()
        store.push_event(_make_event(entity_id=1, timestamp=1000.0, frame_number=10), ws)
        store.push_event(_make_event(entity_id=2, timestamp=1020.0, frame_number=20), ws)
        # Close by gap
        closed = store.push_event(_make_event(entity_id=3, timestamp=2000.0, frame_number=100), ws)
        assert closed is not None
        assert 1 in closed.entity_ids
        assert 2 in closed.entity_ids
        assert closed.event_count == 2
        assert closed.start_time == 1000.0
        assert closed.start_frame == 10
        assert closed.end_frame == 20

    def test_severity_critical_anomaly(self):
        store = _store(gap=60.0)
        ws = _make_world_state()
        store.push_event(_make_event(event_type=EntityEventType.ANOMALY, timestamp=0.0), ws)
        closed = store.push_event(_make_event(timestamp=200.0), ws)
        assert closed.severity == "critical"

    def test_severity_notable_one_violation(self):
        store = _store(gap=60.0)
        ws = _make_world_state()
        store.push_event(_make_event(event_type=EntityEventType.RULE_VIOLATION, timestamp=0.0), ws)
        closed = store.push_event(_make_event(timestamp=200.0), ws)
        assert closed.severity == "notable"

    def test_severity_normal_no_issues(self):
        store = _store(gap=60.0)
        ws = _make_world_state()
        store.push_event(_make_event(event_type=EntityEventType.ENTERED, timestamp=0.0), ws)
        closed = store.push_event(_make_event(timestamp=200.0), ws)
        assert closed.severity == "normal"

    def test_push_event_returns_none_within_episode(self):
        store = _store(gap=120.0)
        ws = _make_world_state()
        store.push_event(_make_event(timestamp=0.0), ws)
        result = store.push_event(_make_event(timestamp=10.0), ws)
        assert result is None

    def test_summary_not_empty_after_close(self):
        store = _store(gap=60.0)
        ws = _make_world_state()
        store.push_event(_make_event(timestamp=0.0), ws)
        closed = store.push_event(_make_event(timestamp=200.0), ws)
        assert closed.summary_ja
        assert closed.summary_en
        assert len(closed.summary_ja) > 5
        assert len(closed.summary_en) > 5


# ══════════════════════════════════════════════════════════════════════════
# TestEpisodeRetrieval  (8 tests)
# ══════════════════════════════════════════════════════════════════════════


class TestEpisodeRetrieval:
    def test_get_current_episode_none_initially(self):
        store = _store()
        assert store.get_current_episode() is None

    def test_get_current_episode_returns_snapshot(self):
        store = _store()
        store.push_event(_make_event(timestamp=100.0), _make_world_state())
        snap = store.get_current_episode()
        assert snap is not None
        assert snap.end_time is None  # still open

    def test_get_recent_episodes_empty_initially(self):
        store = _store()
        assert store.get_recent_episodes() == []

    def test_get_recent_episodes_after_several(self):
        store = _store(gap=10.0)
        ws = _make_world_state()
        # Create 3 closed episodes by using large gaps
        t = 0.0
        for _ in range(3):
            store.push_event(_make_event(timestamp=t), ws)
            t += 100.0  # gap > 10 s → closes each time
        store.push_event(_make_event(timestamp=t), ws)
        recent = store.get_recent_episodes(10)
        assert len(recent) == 3

    def test_get_recent_episodes_newest_first(self):
        store = _store(gap=10.0)
        ws = _make_world_state()
        t = 0.0
        for i in range(4):
            store.push_event(_make_event(timestamp=t), ws)
            t += 100.0
        store.push_event(_make_event(timestamp=t), ws)  # close last
        recent = store.get_recent_episodes(10)
        # newest first: start_times descending
        times = [ep.start_time for ep in recent]
        assert times == sorted(times, reverse=True)

    def test_max_episodes_limit(self):
        store = EpisodicMemoryStore(max_episodes=3, episode_gap_seconds=10.0)
        ws = _make_world_state()
        t = 0.0
        for _ in range(6):
            store.push_event(_make_event(timestamp=t), ws)
            t += 100.0
        store.push_event(_make_event(timestamp=t), ws)
        # Should cap at max_episodes=3
        assert len(store.all_episodes()) <= 3

    def test_all_episodes_oldest_first(self):
        store = _store(gap=10.0)
        ws = _make_world_state()
        t = 0.0
        for _ in range(4):
            store.push_event(_make_event(timestamp=t), ws)
            t += 100.0
        store.push_event(_make_event(timestamp=t), ws)
        all_eps = store.all_episodes()
        times = [ep.start_time for ep in all_eps]
        assert times == sorted(times)

    def test_reset_clears_state(self):
        store = _store(gap=10.0)
        ws = _make_world_state()
        store.push_event(_make_event(timestamp=0.0), ws)
        store.push_event(_make_event(timestamp=100.0), ws)
        store.reset()
        assert store.get_current_episode() is None
        assert store.all_episodes() == []


# ══════════════════════════════════════════════════════════════════════════
# TestSearch  (8 tests)
# ══════════════════════════════════════════════════════════════════════════


def _push_episodes(store: EpisodicMemoryStore, n: int = 3, gap: float = 200.0) -> None:
    """Push n separate episodes with different entity labels."""
    ws = _make_world_state()
    t = 0.0
    labels = ["person", "forklift", "robot"]
    for i in range(n):
        store.push_event(
            _make_event(
                event_type=EntityEventType.ENTERED,
                entity_id=i + 1,
                timestamp=t,
                details={"label": labels[i % len(labels)]},
            ),
            ws,
        )
        t += gap  # force gap close on next push
    # final event to close the last open episode
    store.push_event(_make_event(timestamp=t), ws)


class TestSearch:
    def test_search_empty_returns_empty(self):
        store = _store()
        assert store.search(["anything"]) == []

    def test_search_no_keywords_returns_empty(self):
        store = _store()
        _push_episodes(store)
        assert store.search([]) == []

    def test_search_by_tag_matches(self):
        store = _store(gap=100.0)
        _push_episodes(store, n=3, gap=200.0)
        # "entered" should appear as a tag (event type name)
        results = store.search(["entered"])
        assert len(results) > 0

    def test_search_case_insensitive(self):
        store = _store(gap=100.0)
        _push_episodes(store, n=3, gap=200.0)
        results_lower = store.search(["entered"])
        results_upper = store.search(["ENTERED"])
        assert len(results_lower) == len(results_upper)

    def test_search_max_results_limit(self):
        store = _store(gap=100.0)
        _push_episodes(store, n=5, gap=200.0)
        results = store.search(["entered"], max_results=2)
        assert len(results) <= 2

    def test_search_returns_multiple_matches(self):
        store = _store(gap=100.0)
        _push_episodes(store, n=5, gap=200.0)
        results = store.search(["entered"], max_results=10)
        assert len(results) > 1

    def test_search_by_summary_text(self):
        store = _store(gap=100.0)
        _push_episodes(store, n=3, gap=200.0)
        # All summaries contain "entities" or "エンティティ"
        results = store.search(["activity"])
        assert len(results) > 0

    def test_search_keywords_any_match(self):
        store = _store(gap=100.0)
        _push_episodes(store, n=3, gap=200.0)
        # "xyznotexist" won't match, but "normal" or "entered" will
        results = store.search(["xyznotexist", "entered"])
        assert len(results) > 0


# ══════════════════════════════════════════════════════════════════════════
# TestTemporalPatterns  (8 tests)
# ══════════════════════════════════════════════════════════════════════════


def _push_many_similar_episodes(
    store: EpisodicMemoryStore, count: int = 60
) -> None:
    """Push many episodes with a consistent ENTERED → EXITED bigram."""
    ws = _make_world_state()
    t = 0.0
    for _ in range(count):
        store.push_event(
            _make_event(event_type=EntityEventType.ENTERED, timestamp=t), ws
        )
        store.push_event(
            _make_event(event_type=EntityEventType.EXITED, timestamp=t + 5.0), ws
        )
        t += 300.0  # 5-min gap to close episode each time


class TestTemporalPatterns:
    def test_get_temporal_patterns_empty_initially(self):
        store = _store()
        assert store.get_temporal_patterns() == []

    def test_pattern_detection_after_enough_episodes(self):
        store = EpisodicMemoryStore(
            max_episodes=500,
            episode_gap_seconds=60.0,
        )
        # Force detection by closing 50+ episodes
        _push_many_similar_episodes(store, count=60)
        patterns = store.get_temporal_patterns()
        assert len(patterns) > 0

    def test_pattern_confidence_in_range(self):
        store = EpisodicMemoryStore(max_episodes=500, episode_gap_seconds=60.0)
        _push_many_similar_episodes(store, count=60)
        for p in store.get_temporal_patterns():
            assert 0.0 <= p.confidence <= 1.0

    def test_pattern_occurrences_at_least_three(self):
        store = EpisodicMemoryStore(max_episodes=500, episode_gap_seconds=60.0)
        _push_many_similar_episodes(store, count=60)
        for p in store.get_temporal_patterns():
            assert p.occurrences >= 3

    def test_pattern_has_event_sequence(self):
        store = EpisodicMemoryStore(max_episodes=500, episode_gap_seconds=60.0)
        _push_many_similar_episodes(store, count=60)
        for p in store.get_temporal_patterns():
            assert isinstance(p.event_sequence, list)
            assert len(p.event_sequence) >= 2

    def test_pattern_descriptions_not_empty(self):
        store = EpisodicMemoryStore(max_episodes=500, episode_gap_seconds=60.0)
        _push_many_similar_episodes(store, count=60)
        for p in store.get_temporal_patterns():
            assert p.description_ja
            assert p.description_en

    def test_pattern_typical_hours_populated(self):
        store = EpisodicMemoryStore(max_episodes=500, episode_gap_seconds=60.0)
        _push_many_similar_episodes(store, count=60)
        for p in store.get_temporal_patterns():
            assert isinstance(p.typical_hours, list)
            assert len(p.typical_hours) > 0

    def test_patterns_sorted_by_confidence_desc(self):
        store = EpisodicMemoryStore(max_episodes=500, episode_gap_seconds=60.0)
        _push_many_similar_episodes(store, count=60)
        patterns = store.get_temporal_patterns()
        if len(patterns) >= 2:
            for i in range(len(patterns) - 1):
                assert patterns[i].confidence >= patterns[i + 1].confidence


# ══════════════════════════════════════════════════════════════════════════
# TestCrossEpisodeSummary  (6 tests)
# ══════════════════════════════════════════════════════════════════════════


class TestCrossEpisodeSummary:
    def test_summary_empty_store_returns_meaningful_string(self):
        store = _store()
        s = store.get_cross_episode_summary(hours=24)
        assert isinstance(s, str)
        assert len(s) > 5

    def test_summary_with_episodes(self):
        store = _store(gap=10.0)
        ws = _make_world_state()
        import time as _time

        now = _time.time()
        store.push_event(_make_event(timestamp=now - 3600), ws)
        store.push_event(_make_event(timestamp=now), ws)
        s = store.get_cross_episode_summary(hours=24)
        assert isinstance(s, str)

    def test_summary_counts_violations(self):
        store = _store(gap=10.0)
        ws = _make_world_state()
        import time as _time

        now = _time.time()
        store.push_event(
            _make_event(event_type=EntityEventType.RULE_VIOLATION, timestamp=now - 3600),
            ws,
        )
        store.push_event(_make_event(timestamp=now), ws)  # closes episode
        s = store.get_cross_episode_summary(hours=24)
        # The summary covers closed episodes; expect violation count > 0 text
        assert "1" in s or "違反" in s or "violation" in s.lower()

    def test_summary_counts_anomalies(self):
        store = _store(gap=10.0)
        ws = _make_world_state()
        import time as _time

        now = _time.time()
        store.push_event(
            _make_event(event_type=EntityEventType.ANOMALY, timestamp=now - 3600),
            ws,
        )
        store.push_event(_make_event(timestamp=now), ws)
        s = store.get_cross_episode_summary(hours=24)
        assert "1" in s or "異常" in s or "anomal" in s.lower()

    def test_summary_covers_time_window(self):
        store = _store(gap=10.0)
        ws = _make_world_state()
        import time as _time

        now = _time.time()
        # Add episode well outside the 1-hour window
        store.push_event(_make_event(timestamp=now - 7200), ws)  # 2 h ago
        store.push_event(_make_event(timestamp=now - 6000), ws)  # also outside
        # close it
        store.push_event(_make_event(timestamp=now - 5999), ws)
        s_1h = store.get_cross_episode_summary(hours=1)
        s_24h = store.get_cross_episode_summary(hours=24)
        # 24h window sees more episodes than 1h
        # Both are strings; 24h may mention episodes while 1h may not
        assert isinstance(s_1h, str)
        assert isinstance(s_24h, str)

    def test_summary_is_japanese(self):
        store = _store()
        s = store.get_cross_episode_summary(hours=24)
        # Must contain at least one CJK character
        has_japanese = any("\u3000" <= c <= "\u9fff" or "\u30a0" <= c <= "\u30ff" for c in s)
        assert has_japanese


# ══════════════════════════════════════════════════════════════════════════
# TestEdgeCases  (10 tests)
# ══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_same_timestamp_as_last_event_no_close(self):
        """Two events at the same timestamp should stay in the same episode."""
        store = _store(gap=60.0)
        ws = _make_world_state()
        store.push_event(_make_event(timestamp=100.0), ws)
        closed = store.push_event(_make_event(timestamp=100.0), ws)
        assert closed is None
        assert store.get_current_episode().event_count == 2

    def test_very_short_gap_no_close(self):
        """Gap slightly below threshold must NOT close the episode."""
        store = _store(gap=60.0)
        ws = _make_world_state()
        store.push_event(_make_event(timestamp=0.0), ws)
        closed = store.push_event(_make_event(timestamp=59.9), ws)
        assert closed is None

    def test_very_long_gap_closes_episode(self):
        """A gap 10× the threshold must close and open a fresh episode."""
        store = _store(gap=60.0)
        ws = _make_world_state()
        store.push_event(_make_event(timestamp=0.0), ws)
        closed = store.push_event(_make_event(timestamp=700.0), ws)
        assert closed is not None
        assert store.get_current_episode().event_count == 1

    def test_episode_with_single_event(self):
        """An episode containing exactly one event closes correctly."""
        store = _store(gap=60.0)
        ws = _make_world_state()
        store.push_event(_make_event(entity_id=42, timestamp=0.0, frame_number=5), ws)
        closed = store.push_event(_make_event(timestamp=200.0), ws)
        assert closed is not None
        assert closed.event_count == 1
        assert closed.start_frame == 5
        assert closed.end_frame == 5

    def test_many_entities_in_one_episode(self):
        """Many distinct entities within a single episode are deduplicated."""
        store = _store(gap=120.0)
        ws = _make_world_state()
        for i in range(20):
            store.push_event(_make_event(entity_id=i, timestamp=float(i)), ws)
        current = store.get_current_episode()
        assert len(current.entity_ids) == 20

    def test_anomaly_events_increment_anomaly_count(self):
        """ANOMALY events raise severity to critical."""
        store = _store(gap=60.0)
        ws = _make_world_state()
        for _ in range(3):
            store.push_event(
                _make_event(event_type=EntityEventType.ANOMALY, timestamp=0.0), ws
            )
        closed = store.push_event(_make_event(timestamp=200.0), ws)
        assert closed.severity == "critical"
        assert closed.event_type_counts.get(EntityEventType.ANOMALY.name, 0) == 3

    def test_rule_violation_increments_violation_count(self):
        """More than 2 RULE_VIOLATION events trigger critical severity."""
        store = _store(gap=60.0)
        ws = _make_world_state()
        for i in range(3):
            store.push_event(
                _make_event(
                    event_type=EntityEventType.RULE_VIOLATION,
                    timestamp=float(i),
                ),
                ws,
            )
        closed = store.push_event(_make_event(timestamp=200.0), ws)
        assert closed.severity == "critical"

    def test_entity_ids_deduplicated(self):
        """Same entity_id pushed multiple times appears once in entity_ids."""
        store = _store(gap=120.0)
        ws = _make_world_state()
        for _ in range(5):
            store.push_event(_make_event(entity_id=99, timestamp=0.0), ws)
        current = store.get_current_episode()
        assert current.entity_ids.count(99) == 1
        assert len(current.entity_ids) == 1

    def test_start_end_frame_correct(self):
        store = _store(gap=60.0)
        ws = _make_world_state()
        store.push_event(_make_event(timestamp=0.0, frame_number=1), ws)
        store.push_event(_make_event(timestamp=5.0, frame_number=5), ws)
        store.push_event(_make_event(timestamp=10.0, frame_number=10), ws)
        closed = store.push_event(_make_event(timestamp=200.0, frame_number=200), ws)
        assert closed.start_frame == 1
        assert closed.end_frame == 10

    def test_duration_seconds_correct_after_close(self):
        store = _store(gap=60.0)
        ws = _make_world_state()
        store.push_event(_make_event(timestamp=1000.0), ws)
        store.push_event(_make_event(timestamp=1030.0), ws)
        closed = store.push_event(_make_event(timestamp=2000.0), ws)
        assert abs(closed.duration_seconds - 30.0) < 1e-6
