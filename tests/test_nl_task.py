"""Tests for sopilot/perception/nl_task.py — NL Task Specification module."""

import time
import pytest

from sopilot.perception.nl_task import (
    NLTask,
    NLTaskParser,
    NLTaskManager,
    TaskTrigger,
    TASK_TYPES,
    SEVERITY_MAP,
    ENTITY_KEYWORDS,
)


# ---------------------------------------------------------------------------
# TestNLTaskDataclass
# ---------------------------------------------------------------------------

class TestNLTaskDataclass:
    def test_to_dict_all_keys_present(self):
        task = NLTask(raw_text="test rule")
        d = task.to_dict()
        expected_keys = {"id", "raw_text", "task_type", "parameters", "active",
                         "created_at", "description_ja", "description_en", "severity"}
        assert expected_keys == set(d.keys())

    def test_to_dict_round_trip_values(self):
        task = NLTask(raw_text="hello", task_type="zone_entry", severity="critical")
        d = task.to_dict()
        assert d["raw_text"] == "hello"
        assert d["task_type"] == "zone_entry"
        assert d["severity"] == "critical"

    def test_default_id_generated(self):
        task1 = NLTask()
        task2 = NLTask()
        assert task1.id != ""
        assert task2.id != ""
        assert task1.id != task2.id

    def test_default_active_true(self):
        task = NLTask()
        assert task.active is True

    def test_default_task_type_loiter(self):
        task = NLTask()
        assert task.task_type == "loiter"

    def test_default_severity_warning(self):
        task = NLTask()
        assert task.severity == "warning"

    def test_created_at_is_recent(self):
        before = time.time()
        task = NLTask()
        after = time.time()
        assert before <= task.created_at <= after


# ---------------------------------------------------------------------------
# TestNLTaskParser_Loiter
# ---------------------------------------------------------------------------

class TestNLTaskParser_Loiter:
    def setup_method(self):
        self.parser = NLTaskParser()

    def test_loiter_2min(self):
        task = self.parser.parse_rule("2分以上滞留したら警告")
        assert task.task_type == "loiter"
        assert task.parameters["duration_seconds"] == 120.0

    def test_loiter_30sec(self):
        task = self.parser.parse_rule("30秒以上いたら")
        assert task.task_type == "loiter"
        assert task.parameters["duration_seconds"] == 30.0

    def test_loiter_zone_a_3min(self):
        task = self.parser.parse_rule("Zone Aで3分以上滞留")
        assert task.task_type == "loiter"
        assert task.parameters.get("zone") == "Zone A"
        assert task.parameters["duration_seconds"] == 180.0

    def test_loiter_default_duration(self):
        task = self.parser.parse_rule("誰かが滞留しているかチェック")
        assert task.task_type == "loiter"
        assert task.parameters["duration_seconds"] == 120.0

    def test_loiter_area_radius_set(self):
        task = self.parser.parse_rule("2分以上滞留したら")
        assert "area_radius" in task.parameters
        assert task.parameters["area_radius"] > 0

    def test_loiter_descriptions_not_empty(self):
        task = self.parser.parse_rule("2分以上滞留したら警告")
        assert task.description_ja != ""
        assert task.description_en != ""


# ---------------------------------------------------------------------------
# TestNLTaskParser_ZoneEntry
# ---------------------------------------------------------------------------

class TestNLTaskParser_ZoneEntry:
    def setup_method(self):
        self.parser = NLTaskParser()

    def test_zone_entry_zone_a(self):
        task = self.parser.parse_rule("Zone Aに入ったら")
        assert task.task_type == "zone_entry"
        assert task.parameters.get("zone") == "Zone A"

    def test_zone_entry_critical_severity(self):
        task = self.parser.parse_rule("制限区域に侵入したら緊急")
        assert task.task_type == "zone_entry"
        assert task.severity == "critical"

    def test_zone_entry_area_b(self):
        task = self.parser.parse_rule("エリアBに進入")
        assert task.task_type == "zone_entry"
        assert task.parameters.get("zone") == "Zone B"

    def test_zone_entry_shinyu_keyword(self):
        task = self.parser.parse_rule("危険区域に侵入")
        assert task.task_type == "zone_entry"

    def test_zone_entry_enter_english(self):
        task = self.parser.parse_rule("Alert when someone enters Zone C")
        assert task.task_type == "zone_entry"
        assert task.parameters.get("zone") == "Zone C"


# ---------------------------------------------------------------------------
# TestNLTaskParser_Approach
# ---------------------------------------------------------------------------

class TestNLTaskParser_Approach:
    def setup_method(self):
        self.parser = NLTaskParser()

    def test_approach_basic(self):
        task = self.parser.parse_rule("Zone Aに接近したら")
        assert task.task_type == "approach"

    def test_approach_default_distance(self):
        task = self.parser.parse_rule("機械に近づく")
        assert task.task_type == "approach"
        assert task.parameters.get("distance_threshold") == 0.15

    def test_approach_custom_distance(self):
        task = self.parser.parse_rule("2mより近づいたら警告")
        assert task.task_type == "approach"
        assert task.parameters.get("distance_threshold") == 2.0

    def test_approach_description_set(self):
        task = self.parser.parse_rule("Zone Aに接近したら")
        assert task.description_ja != ""
        assert task.description_en != ""


# ---------------------------------------------------------------------------
# TestNLTaskParser_CountThreshold
# ---------------------------------------------------------------------------

class TestNLTaskParser_CountThreshold:
    def setup_method(self):
        self.parser = NLTaskParser()

    def test_count_3_persons(self):
        task = self.parser.parse_rule("3人以上いたら")
        assert task.task_type == "count_threshold"
        assert task.parameters["threshold"] == 3

    def test_count_5_vehicles(self):
        task = self.parser.parse_rule("5台以上")
        assert task.task_type == "count_threshold"
        assert task.parameters["threshold"] == 5

    def test_count_entity_type_from_person(self):
        task = self.parser.parse_rule("3人以上同時にいたら警告")
        assert task.task_type == "count_threshold"
        assert task.parameters.get("entity_type") == "person"

    def test_count_description_set(self):
        task = self.parser.parse_rule("3人以上いたら")
        assert task.description_ja != ""
        assert task.description_en != ""

    def test_count_yori_oku(self):
        task = self.parser.parse_rule("10名より多い場合")
        assert task.task_type == "count_threshold"
        assert task.parameters["threshold"] == 10


# ---------------------------------------------------------------------------
# TestNLTaskParser_TimeFilter
# ---------------------------------------------------------------------------

class TestNLTaskParser_TimeFilter:
    def setup_method(self):
        self.parser = NLTaskParser()

    def test_time_filter_japanese(self):
        task = self.parser.parse_rule("9時から17時の間監視")
        assert task.task_type == "time_filter"
        assert task.parameters["start_hour"] == 9
        assert task.parameters["end_hour"] == 17

    def test_time_filter_colon_format(self):
        task = self.parser.parse_rule("9:00〜18:00")
        assert task.task_type == "time_filter"
        assert task.parameters["start_hour"] == 9
        assert task.parameters["end_hour"] == 18

    def test_time_filter_descriptions_set(self):
        task = self.parser.parse_rule("9時から17時の間監視")
        assert task.description_ja != ""
        assert task.description_en != ""


# ---------------------------------------------------------------------------
# TestNLTaskParser_Severity
# ---------------------------------------------------------------------------

class TestNLTaskParser_Severity:
    def setup_method(self):
        self.parser = NLTaskParser()

    def test_severity_kinkyu_critical(self):
        task = self.parser.parse_rule("制限区域に侵入したら緊急")
        assert task.severity == "critical"

    def test_severity_tsuchi_info(self):
        task = self.parser.parse_rule("通知レベルで人を検出")
        assert task.severity == "info"

    def test_severity_default_warning(self):
        task = self.parser.parse_rule("誰かが滞留している")
        assert task.severity == "warning"

    def test_severity_kiken_critical(self):
        task = self.parser.parse_rule("危険なエリアに接近")
        assert task.severity == "critical"

    def test_severity_alert_warning(self):
        task = self.parser.parse_rule("アラートを出す")
        assert task.severity == "warning"


# ---------------------------------------------------------------------------
# TestNLTaskParser_EntityType
# ---------------------------------------------------------------------------

class TestNLTaskParser_EntityType:
    def setup_method(self):
        self.parser = NLTaskParser()

    def test_entity_person_japanese(self):
        task = self.parser.parse_rule("人がエリアに滞留")
        assert task.parameters.get("entity_type") == "person"

    def test_entity_vehicle_japanese(self):
        task = self.parser.parse_rule("車が近づく")
        assert task.parameters.get("entity_type") == "vehicle"

    def test_entity_worker_english(self):
        task = self.parser.parse_rule("worker enters zone")
        assert task.parameters.get("entity_type") == "person"

    def test_entity_equipment(self):
        task = self.parser.parse_rule("機械が制限区域に侵入")
        assert task.parameters.get("entity_type") == "equipment"


# ---------------------------------------------------------------------------
# TestNLTaskManagerAddRemove
# ---------------------------------------------------------------------------

class TestNLTaskManagerAddRemove:
    def setup_method(self):
        self.mgr = NLTaskManager()

    def test_add_and_get_task(self):
        task = NLTask(raw_text="test")
        self.mgr.add_task(task)
        retrieved = self.mgr.get_task(task.id)
        assert retrieved is task

    def test_get_tasks_returns_list(self):
        task1 = NLTask(raw_text="a")
        task2 = NLTask(raw_text="b")
        self.mgr.add_task(task1)
        self.mgr.add_task(task2)
        tasks = self.mgr.get_tasks()
        assert isinstance(tasks, list)
        assert len(tasks) == 2

    def test_parse_and_add_creates_and_stores(self):
        task = self.mgr.parse_and_add("2分以上滞留したら警告")
        assert isinstance(task, NLTask)
        assert self.mgr.get_task(task.id) is task

    def test_remove_task_returns_true(self):
        task = self.mgr.add_task(NLTask(raw_text="x"))
        result = self.mgr.remove_task(task.id)
        assert result is True

    def test_remove_task_get_returns_none(self):
        task = self.mgr.add_task(NLTask(raw_text="x"))
        self.mgr.remove_task(task.id)
        assert self.mgr.get_task(task.id) is None

    def test_remove_nonexistent_returns_false(self):
        result = self.mgr.remove_task("nonexistent-id")
        assert result is False

    def test_set_active_changes_flag(self):
        task = self.mgr.add_task(NLTask(raw_text="x"))
        assert task.active is True
        result = self.mgr.set_active(task.id, False)
        assert result is True
        assert task.active is False

    def test_set_active_nonexistent_returns_false(self):
        result = self.mgr.set_active("bad-id", False)
        assert result is False


# ---------------------------------------------------------------------------
# TestNLTaskManagerLoiter
# ---------------------------------------------------------------------------

class TestNLTaskManagerLoiter:
    def setup_method(self):
        self.mgr = NLTaskManager()
        # Add a loiter task with short duration for testing
        self.task = NLTask(
            raw_text="test loiter",
            task_type="loiter",
            parameters={"duration_seconds": 10.0, "area_radius": 0.10},
        )
        self.mgr.add_task(self.task)

    def test_first_call_no_trigger(self):
        triggers = self.mgr.check_entity(1, "person", (0.5, 0.5), current_time=0.0)
        assert triggers == []

    def test_trigger_after_duration_exceeded(self):
        # First call to record start
        self.mgr.check_entity(1, "person", (0.5, 0.5), current_time=0.0)
        # Second call after duration exceeded
        triggers = self.mgr.check_entity(1, "person", (0.5, 0.5), current_time=15.0)
        assert len(triggers) == 1
        assert triggers[0].task_type == "loiter"
        assert triggers[0].entity_id == 1

    def test_entity_moved_resets_timer_no_trigger(self):
        # Record start position
        self.mgr.check_entity(1, "person", (0.5, 0.5), current_time=0.0)
        # Move far away (beyond radius 0.10)
        triggers = self.mgr.check_entity(1, "person", (0.8, 0.8), current_time=15.0)
        assert triggers == []

    def test_entity_type_filter_blocks_wrong_type(self):
        # Task filtered to person only
        person_task = NLTask(
            raw_text="person loiter",
            task_type="loiter",
            parameters={"duration_seconds": 5.0, "area_radius": 0.10, "entity_type": "person"},
        )
        mgr = NLTaskManager()
        mgr.add_task(person_task)
        # First call with vehicle entity
        mgr.check_entity(2, "vehicle", (0.5, 0.5), current_time=0.0)
        # Second call after duration — vehicle should NOT trigger person task
        triggers = mgr.check_entity(2, "vehicle", (0.5, 0.5), current_time=10.0)
        assert triggers == []

    def test_inactive_task_no_trigger(self):
        self.mgr.set_active(self.task.id, False)
        self.mgr.check_entity(1, "person", (0.5, 0.5), current_time=0.0)
        triggers = self.mgr.check_entity(1, "person", (0.5, 0.5), current_time=15.0)
        assert triggers == []

    def test_reset_clears_loiter_start(self):
        # Record start
        self.mgr.check_entity(1, "person", (0.5, 0.5), current_time=0.0)
        self.mgr.reset()
        # After reset, first call again → no trigger
        triggers = self.mgr.check_entity(1, "person", (0.5, 0.5), current_time=15.0)
        assert triggers == []

    def test_trigger_severity_propagated(self):
        critical_task = NLTask(
            raw_text="critical loiter",
            task_type="loiter",
            parameters={"duration_seconds": 5.0, "area_radius": 0.10},
            severity="critical",
        )
        mgr = NLTaskManager()
        mgr.add_task(critical_task)
        mgr.check_entity(3, "person", (0.5, 0.5), current_time=0.0)
        triggers = mgr.check_entity(3, "person", (0.5, 0.5), current_time=10.0)
        assert len(triggers) == 1
        assert triggers[0].severity == "critical"


# ---------------------------------------------------------------------------
# TestNLTaskManagerCount
# ---------------------------------------------------------------------------

class TestNLTaskManagerCount:
    def setup_method(self):
        self.mgr = NLTaskManager()
        self.task = NLTask(
            raw_text="3 person threshold",
            task_type="count_threshold",
            parameters={"threshold": 3, "entity_type": "person"},
        )
        self.mgr.add_task(self.task)

    def test_below_threshold_no_trigger(self):
        triggers = self.mgr.update_entity_counts({"person": 2})
        assert triggers == []

    def test_at_threshold_trigger_returned(self):
        triggers = self.mgr.update_entity_counts({"person": 3})
        assert len(triggers) == 1
        assert triggers[0].task_type == "count_threshold"

    def test_above_threshold_trigger_returned(self):
        triggers = self.mgr.update_entity_counts({"person": 5})
        assert len(triggers) == 1

    def test_entity_type_filter_in_count(self):
        # Only person counts; vehicle count should not trigger person threshold
        triggers = self.mgr.update_entity_counts({"vehicle": 10})
        assert triggers == []

    def test_no_entity_type_sums_all(self):
        # Task without entity_type filter
        all_task = NLTask(
            raw_text="total count",
            task_type="count_threshold",
            parameters={"threshold": 5},
        )
        mgr = NLTaskManager()
        mgr.add_task(all_task)
        triggers = mgr.update_entity_counts({"person": 3, "vehicle": 2})
        assert len(triggers) == 1

    def test_inactive_count_task_no_trigger(self):
        self.mgr.set_active(self.task.id, False)
        triggers = self.mgr.update_entity_counts({"person": 10})
        assert triggers == []


# ---------------------------------------------------------------------------
# TestNLTaskManagerTimeFilter
# ---------------------------------------------------------------------------

class TestNLTaskManagerTimeFilter:
    def test_no_time_filter_tasks_returns_true(self):
        mgr = NLTaskManager()
        assert mgr.is_within_monitor_hours() is True

    def test_only_inactive_time_filter_returns_true(self):
        mgr = NLTaskManager()
        task = NLTask(
            raw_text="time filter",
            task_type="time_filter",
            parameters={"start_hour": 0, "end_hour": 1},
            active=False,
        )
        mgr.add_task(task)
        # Even if current hour != 0-1, inactive task should not block
        assert mgr.is_within_monitor_hours() is True

    def test_time_filter_with_full_day_range_returns_true(self):
        mgr = NLTaskManager()
        task = NLTask(
            raw_text="all day",
            task_type="time_filter",
            parameters={"start_hour": 0, "end_hour": 24},
        )
        mgr.add_task(task)
        # Current hour is always in [0, 24)
        assert mgr.is_within_monitor_hours() is True


# ---------------------------------------------------------------------------
# TestNLTaskManagerStateDict
# ---------------------------------------------------------------------------

class TestNLTaskManagerStateDict:
    def test_state_dict_has_required_keys(self):
        mgr = NLTaskManager()
        d = mgr.get_state_dict()
        assert "task_count" in d
        assert "active_count" in d
        assert "tasks" in d

    def test_state_dict_task_count(self):
        mgr = NLTaskManager()
        mgr.add_task(NLTask(raw_text="a"))
        mgr.add_task(NLTask(raw_text="b"))
        d = mgr.get_state_dict()
        assert d["task_count"] == 2

    def test_state_dict_active_count(self):
        mgr = NLTaskManager()
        t1 = mgr.add_task(NLTask(raw_text="a"))
        t2 = mgr.add_task(NLTask(raw_text="b"))
        mgr.set_active(t1.id, False)
        d = mgr.get_state_dict()
        assert d["active_count"] == 1

    def test_state_dict_tasks_is_list_of_dicts(self):
        mgr = NLTaskManager()
        mgr.add_task(NLTask(raw_text="x"))
        d = mgr.get_state_dict()
        assert isinstance(d["tasks"], list)
        assert len(d["tasks"]) == 1
        assert isinstance(d["tasks"][0], dict)

    def test_empty_manager_state_dict(self):
        mgr = NLTaskManager()
        d = mgr.get_state_dict()
        assert d["task_count"] == 0
        assert d["active_count"] == 0
        assert d["tasks"] == []
