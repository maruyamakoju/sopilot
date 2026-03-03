"""Tests for sopilot/perception/long_term_memory.py

All DB paths use tempfile.TemporaryDirectory() to avoid pollution between tests.
Uses ignore_cleanup_errors=True because SQLite WAL mode may hold file handles
briefly on Windows after connection close.
"""

import sqlite3
import tempfile
import threading
import time
from pathlib import Path

import pytest

from sopilot.perception.long_term_memory import FactRecord, LongTermMemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmpdir: str) -> LongTermMemoryStore:
    db_path = Path(tmpdir) / "ltm_test.db"
    return LongTermMemoryStore(db_path=db_path)


def _closed_episode(
    hour: int = 9,
    event_count: int = 5,
    entity_ids: list | None = None,
    severity: str = "warning",
    event_type_counts: dict | None = None,
    duration_seconds: float = 60.0,
) -> dict:
    """Build a minimal closed episode dict for a specific hour."""
    now = time.time()
    t = time.localtime(now)
    local_base = time.mktime(
        time.struct_time((t.tm_year, t.tm_mon, t.tm_mday, hour, 0, 0, t.tm_wday, t.tm_yday, t.tm_isdst))
    )
    return {
        "start_time": local_base,
        "end_time": local_base + duration_seconds,
        "event_count": event_count,
        "entity_ids": entity_ids if entity_ids is not None else [1, 2, 3],
        "severity": severity,
        "event_type_counts": event_type_counts or {},
        "duration_seconds": duration_seconds,
    }


# ---------------------------------------------------------------------------
# 1. TestInit
# ---------------------------------------------------------------------------

class TestInit:
    def test_creates_db_file(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = Path(tmpdir) / "subdir" / "ltm.db"
            store = LongTermMemoryStore(db_path=db_path)
            assert db_path.exists()
            store.close()

    def test_ltm_facts_table_exists(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            db_path = Path(tmpdir) / "ltm_test.db"
            conn = sqlite3.connect(str(db_path))
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='ltm_facts'"
            ).fetchone()
            conn.close()
            store.close()
            assert row is not None
            assert row[0] == "ltm_facts"

    def test_init_idempotent(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = Path(tmpdir) / "ltm_test.db"
            store1 = LongTermMemoryStore(db_path=db_path)
            # Second instance on same path must not raise
            store2 = LongTermMemoryStore(db_path=db_path)
            assert store2 is not None
            store1.close()
            store2.close()


# ---------------------------------------------------------------------------
# 2. TestUpsertFact
# ---------------------------------------------------------------------------

class TestUpsertFact:
    def test_insert_new_fact_observations(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("hourly_activity", "entity_count", 5.0, time_slot=9)
            facts = store.get_hourly_pattern(9)
            store.close()
            assert len(facts) == 1
            assert facts[0].observations == 1

    def test_insert_new_fact_confidence(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("hourly_activity", "entity_count", 5.0, time_slot=9)
            facts = store.get_hourly_pattern(9)
            store.close()
            assert facts[0].confidence == pytest.approx(0.5)

    def test_insert_new_fact_metric_value(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("hourly_activity", "entity_count", 7.0, time_slot=10)
            facts = store.get_hourly_pattern(10)
            store.close()
            assert facts[0].metric_value == pytest.approx(7.0)

    def test_upsert_same_key_increments_observations(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("hourly_activity", "entity_count", 5.0, time_slot=9)
            store.upsert_fact("hourly_activity", "entity_count", 5.0, time_slot=9)
            facts = store.get_hourly_pattern(9)
            store.close()
            assert facts[0].observations == 2

    def test_upsert_same_key_increases_confidence(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("hourly_activity", "entity_count", 5.0, time_slot=9)
            store.upsert_fact("hourly_activity", "entity_count", 5.0, time_slot=9)
            facts = store.get_hourly_pattern(9)
            store.close()
            assert facts[0].confidence > 0.5

    def test_ema_blending(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            # First insert: value = 0.0
            store.upsert_fact("hourly_activity", "entity_count", 0.0, time_slot=9)
            # Second upsert: new_value = 10.0 → EMA = 0.2*10 + 0.8*0 = 2.0
            store.upsert_fact("hourly_activity", "entity_count", 10.0, time_slot=9)
            facts = store.get_hourly_pattern(9)
            store.close()
            assert 0.0 < facts[0].metric_value < 10.0

    def test_different_fact_type_creates_separate_rows(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("hourly_activity", "entity_count", 5.0, time_slot=9)
            store.upsert_fact("severity_pattern", "entity_count", 5.0, time_slot=9)
            state = store.get_state_dict()
            store.close()
            assert state["total_facts"] == 2

    def test_different_metric_name_creates_separate_rows(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("hourly_activity", "entity_count", 5.0, time_slot=9)
            store.upsert_fact("hourly_activity", "event_count", 3.0, time_slot=9)
            state = store.get_state_dict()
            store.close()
            assert state["total_facts"] == 2

    def test_different_time_slot_creates_separate_rows(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("hourly_activity", "entity_count", 5.0, time_slot=9)
            store.upsert_fact("hourly_activity", "entity_count", 5.0, time_slot=10)
            state = store.get_state_dict()
            store.close()
            assert state["total_facts"] == 2

    def test_different_location_creates_separate_rows(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("hourly_activity", "entity_count", 5.0, location="zone_a", time_slot=9)
            store.upsert_fact("hourly_activity", "entity_count", 5.0, location="zone_b", time_slot=9)
            state = store.get_state_dict()
            store.close()
            assert state["total_facts"] == 2

    def test_different_entity_type_creates_separate_rows(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("entity_frequency", "count", 2.0, entity_type="ANOMALY")
            store.upsert_fact("entity_frequency", "count", 1.0, entity_type="RULE_VIOLATION")
            state = store.get_state_dict()
            store.close()
            assert state["total_facts"] == 2


# ---------------------------------------------------------------------------
# 3. TestRecordEpisodeFacts
# ---------------------------------------------------------------------------

class TestRecordEpisodeFacts:
    def test_open_episode_creates_no_facts(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            ep = {"start_time": time.time(), "end_time": None, "event_count": 3}
            store.record_episode_facts(ep)
            result = []
            for h in range(24):
                result.extend(store.get_hourly_pattern(h))
            store.close()
            assert result == []

    def test_closed_episode_creates_hourly_activity(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            ep = _closed_episode(hour=14, entity_ids=[1, 2, 3], event_count=5)
            store.record_episode_facts(ep)
            facts = store.get_hourly_pattern(14)
            store.close()
            metric_names = {f.metric_name for f in facts}
            assert "entity_count" in metric_names
            assert "event_count" in metric_names

    def test_closed_episode_creates_entity_frequency(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            ep = _closed_episode(
                hour=10,
                event_type_counts={"ANOMALY": 2, "RULE_VIOLATION": 1}
            )
            store.record_episode_facts(ep)
            state = store.get_state_dict()
            store.close()
            assert "entity_frequency" in state["by_type"]
            assert state["by_type"]["entity_frequency"] == 2

    def test_two_episodes_same_hour_applies_ema(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            ep1 = _closed_episode(hour=8, entity_ids=[1, 2])
            ep2 = _closed_episode(hour=8, entity_ids=[3, 4])
            store.record_episode_facts(ep1)
            store.record_episode_facts(ep2)
            facts = store.get_hourly_pattern(8)
            store.close()
            entity_count_facts = [f for f in facts if f.metric_name == "entity_count"]
            assert len(entity_count_facts) == 1
            assert entity_count_facts[0].observations == 2

    def test_empty_entity_ids_upserts_zero(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            ep = _closed_episode(hour=6, entity_ids=[])
            store.record_episode_facts(ep)
            facts = store.get_hourly_pattern(6)
            store.close()
            entity_count_facts = [f for f in facts if f.metric_name == "entity_count"]
            assert len(entity_count_facts) == 1
            assert entity_count_facts[0].metric_value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. TestGetHourlyPattern
# ---------------------------------------------------------------------------

class TestGetHourlyPattern:
    def test_returns_facts_for_correct_hour(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("hourly_activity", "entity_count", 3.0, time_slot=7)
            facts = store.get_hourly_pattern(7)
            store.close()
            assert len(facts) == 1
            assert facts[0].time_slot == 7

    def test_empty_list_for_hour_with_no_data(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            facts = store.get_hourly_pattern(23)
            store.close()
            assert facts == []

    def test_returns_only_facts_for_requested_hour(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("hourly_activity", "entity_count", 3.0, time_slot=7)
            store.upsert_fact("hourly_activity", "entity_count", 5.0, time_slot=8)
            facts = store.get_hourly_pattern(7)
            store.close()
            assert all(f.time_slot == 7 for f in facts)
            assert len(facts) == 1

    def test_returns_fact_record_instances(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("hourly_activity", "entity_count", 4.0, time_slot=12)
            facts = store.get_hourly_pattern(12)
            store.close()
            assert isinstance(facts[0], FactRecord)


# ---------------------------------------------------------------------------
# 5. TestGetLocationFacts
# ---------------------------------------------------------------------------

class TestGetLocationFacts:
    def test_returns_facts_for_correct_location(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("hourly_activity", "entity_count", 3.0, location="entrance", time_slot=9)
            facts = store.get_location_facts("entrance")
            store.close()
            assert len(facts) == 1
            assert facts[0].location == "entrance"

    def test_empty_for_unknown_location(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("hourly_activity", "entity_count", 3.0, location="entrance", time_slot=9)
            facts = store.get_location_facts("unknown_location")
            store.close()
            assert facts == []

    def test_returns_only_facts_for_requested_location(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("hourly_activity", "entity_count", 3.0, location="zone_a", time_slot=9)
            store.upsert_fact("hourly_activity", "entity_count", 5.0, location="zone_b", time_slot=9)
            facts = store.get_location_facts("zone_a")
            store.close()
            assert all(f.location == "zone_a" for f in facts)
            assert len(facts) == 1


# ---------------------------------------------------------------------------
# 6. TestGetExpectedActivity
# ---------------------------------------------------------------------------

class TestGetExpectedActivity:
    def test_returns_zero_for_no_data(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            result = store.get_expected_activity(hour=9)
            store.close()
            assert result == pytest.approx(0.0)

    def test_returns_correct_value_after_upsert(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("hourly_activity", "entity_count", 8.0, time_slot=9)
            result = store.get_expected_activity(hour=9)
            store.close()
            assert result == pytest.approx(8.0)

    def test_entity_type_param_filters_correctly(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("hourly_activity", "entity_count", 5.0, entity_type="worker", time_slot=10)
            store.upsert_fact("hourly_activity", "entity_count", 2.0, entity_type="vehicle", time_slot=10)
            result = store.get_expected_activity(hour=10, entity_type="worker")
            store.close()
            assert result == pytest.approx(5.0)

    def test_entity_type_returns_zero_if_not_found(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("hourly_activity", "entity_count", 5.0, time_slot=10)
            result = store.get_expected_activity(hour=10, entity_type="nonexistent")
            store.close()
            assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 7. TestGenerateSummaryJa
# ---------------------------------------------------------------------------

class TestGenerateSummaryJa:
    def test_empty_db_returns_no_data_message(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            summary = store.generate_summary_ja()
            store.close()
            assert "長期記憶にデータなし" in summary

    def test_after_data_contains_long_term_memory_header(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("hourly_activity", "entity_count", 3.0, time_slot=9)
            summary = store.generate_summary_ja()
            store.close()
            assert "長期記憶" in summary

    def test_after_data_contains_fact_count(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("hourly_activity", "entity_count", 3.0, time_slot=9)
            summary = store.generate_summary_ja()
            store.close()
            assert "件の事実" in summary

    def test_after_hourly_data_contains_hour_reference(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("hourly_activity", "entity_count", 10.0, time_slot=14)
            summary = store.generate_summary_ja()
            store.close()
            assert "時頃" in summary

    def test_returns_string(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            summary = store.generate_summary_ja()
            store.close()
            assert isinstance(summary, str)


# ---------------------------------------------------------------------------
# 8. TestGetStateDict
# ---------------------------------------------------------------------------

class TestGetStateDict:
    def test_has_required_keys(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            state = store.get_state_dict()
            store.close()
            assert "total_facts" in state
            assert "by_type" in state
            assert "db_path" in state

    def test_total_facts_zero_for_empty_db(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            state = store.get_state_dict()
            store.close()
            assert state["total_facts"] == 0

    def test_by_type_contains_correct_fact_types(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("hourly_activity", "entity_count", 3.0, time_slot=9)
            store.upsert_fact("severity_pattern", "severity_score", 0.5, time_slot=9)
            state = store.get_state_dict()
            store.close()
            assert "hourly_activity" in state["by_type"]
            assert "severity_pattern" in state["by_type"]

    def test_db_path_is_string(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            state = store.get_state_dict()
            store.close()
            assert isinstance(state["db_path"], str)

    def test_total_facts_increments_correctly(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("hourly_activity", "entity_count", 3.0, time_slot=9)
            store.upsert_fact("hourly_activity", "event_count", 2.0, time_slot=9)
            state = store.get_state_dict()
            store.close()
            assert state["total_facts"] == 2


# ---------------------------------------------------------------------------
# 9. TestClear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_resets_total_facts(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("hourly_activity", "entity_count", 3.0, time_slot=9)
            store.clear()
            state = store.get_state_dict()
            store.close()
            assert state["total_facts"] == 0

    def test_clear_empty_hourly_pattern(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("hourly_activity", "entity_count", 3.0, time_slot=9)
            store.clear()
            facts = store.get_hourly_pattern(9)
            store.close()
            assert facts == []

    def test_can_upsert_after_clear(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            store.upsert_fact("hourly_activity", "entity_count", 3.0, time_slot=9)
            store.clear()
            store.upsert_fact("hourly_activity", "entity_count", 5.0, time_slot=9)
            state = store.get_state_dict()
            store.close()
            assert state["total_facts"] == 1


# ---------------------------------------------------------------------------
# 10. TestThreadSafety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_upserts_no_exceptions(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)
            errors = []

            def worker(thread_id: int):
                try:
                    for i in range(20):
                        store.upsert_fact(
                            "hourly_activity",
                            "entity_count",
                            float(i),
                            time_slot=thread_id % 24,
                            entity_type=f"type_{thread_id}",
                        )
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            store.close()
            assert errors == [], f"Thread errors: {errors}"

    def test_concurrent_upserts_produces_facts(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = _make_store(tmpdir)

            def worker(thread_id: int):
                for i in range(20):
                    store.upsert_fact(
                        "hourly_activity",
                        "entity_count",
                        float(i),
                        time_slot=thread_id % 24,
                        entity_type=f"type_{thread_id}",
                    )

            threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            state = store.get_state_dict()
            store.close()
            assert state["total_facts"] > 0
