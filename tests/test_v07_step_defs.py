"""Tests for SOPilot v0.7.0 SOP Step Definition feature.

Covers:
- Database.upsert_sop_steps, get_sop_steps, delete_sop_steps + sop_steps table
- StepDefinitionService.get_steps, upsert_steps, delete_steps, compute_time_compliance
- scoring.compute_time_compliance_per_step + step_definitions param in compute_step_contributions
- GET/PUT/DELETE /tasks/steps HTTP endpoints
- SOPilotService.get_sop_steps, upsert_sop_steps, delete_sop_steps delegation
"""
from __future__ import annotations

import os
import pathlib
import tempfile
import unittest

from sopilot.core.scoring import (
    ScoreWeights,
    compute_step_contributions,
    compute_time_compliance_per_step,
)
from sopilot.database import Database
from sopilot.services.step_definition_service import StepDefinitionService

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path: pathlib.Path) -> Database:
    """Return a fresh, empty Database pointing at *tmp_path*."""
    return Database(tmp_path / "test.db")


# ---------------------------------------------------------------------------
# Class 1: TestDatabaseStepDefs
# ---------------------------------------------------------------------------

class TestDatabaseStepDefs(unittest.TestCase):
    """Tests for the database layer upsert_sop_steps / get_sop_steps / delete_sop_steps."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = _make_db(pathlib.Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_sop_steps_table_created(self) -> None:
        """sop_steps table must exist after Database initialisation."""
        import sqlite3
        conn = sqlite3.connect(self.db.path)
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sop_steps'"
            )
            tables = [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()
        self.assertIn("sop_steps", tables)

    def test_upsert_and_get_steps(self) -> None:
        """Upsert 2 steps and retrieve them; verify all fields round-trip correctly."""
        steps = [
            {
                "step_index": 0,
                "name_ja": "消毒",
                "name_en": "Sanitize",
                "expected_duration_sec": 10.0,
                "min_duration_sec": 5.0,
                "max_duration_sec": 20.0,
                "is_critical": True,
                "description": "Disinfect hands",
            },
            {
                "step_index": 1,
                "name_ja": "器具準備",
                "name_en": "Prep tools",
                "expected_duration_sec": 15.0,
                "is_critical": False,
            },
        ]
        self.db.upsert_sop_steps("task-a", steps)
        result = self.db.get_sop_steps("task-a")

        self.assertEqual(len(result), 2)
        # Verify ordering: step_index ASC
        self.assertEqual(result[0]["step_index"], 0)
        self.assertEqual(result[1]["step_index"], 1)

        step0 = result[0]
        self.assertEqual(step0["name_ja"], "消毒")
        self.assertEqual(step0["name_en"], "Sanitize")
        self.assertAlmostEqual(step0["min_duration_sec"], 5.0)
        self.assertAlmostEqual(step0["max_duration_sec"], 20.0)
        self.assertEqual(step0["description"], "Disinfect hands")
        # is_critical is stored as integer in SQLite; check it is truthy
        self.assertTrue(step0["is_critical"])

    def test_upsert_updates_existing(self) -> None:
        """Upserting a step_index that already exists should update the name."""
        self.db.upsert_sop_steps("task-a", [{"step_index": 0, "name_ja": "初回名称"}])
        # Upsert again with a different name
        self.db.upsert_sop_steps("task-a", [{"step_index": 0, "name_ja": "更新後名称"}])

        result = self.db.get_sop_steps("task-a")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name_ja"], "更新後名称")

    def test_delete_steps(self) -> None:
        """After deleting all steps, get_sop_steps must return an empty list."""
        self.db.upsert_sop_steps("task-a", [
            {"step_index": 0, "name_ja": "手順1"},
            {"step_index": 1, "name_ja": "手順2"},
        ])
        self.db.delete_sop_steps("task-a")
        result = self.db.get_sop_steps("task-a")
        self.assertEqual(result, [])

    def test_upsert_returns_count(self) -> None:
        """upsert_sop_steps must return the number of steps upserted."""
        steps = [
            {"step_index": 0, "name_ja": "A"},
            {"step_index": 1, "name_ja": "B"},
            {"step_index": 2, "name_ja": "C"},
        ]
        count = self.db.upsert_sop_steps("task-a", steps)
        self.assertEqual(count, 3)

    def test_delete_returns_count(self) -> None:
        """delete_sop_steps must return the number of rows deleted."""
        self.db.upsert_sop_steps("task-a", [
            {"step_index": 0, "name_ja": "X"},
            {"step_index": 1, "name_ja": "Y"},
        ])
        deleted = self.db.delete_sop_steps("task-a")
        self.assertEqual(deleted, 2)

    def test_steps_are_task_scoped(self) -> None:
        """Steps for task-a must not appear under task-b."""
        self.db.upsert_sop_steps("task-a", [{"step_index": 0, "name_ja": "A only"}])
        self.db.upsert_sop_steps("task-b", [{"step_index": 0, "name_ja": "B only"}])

        result_a = self.db.get_sop_steps("task-a")
        result_b = self.db.get_sop_steps("task-b")
        self.assertEqual(len(result_a), 1)
        self.assertEqual(result_a[0]["name_ja"], "A only")
        self.assertEqual(len(result_b), 1)
        self.assertEqual(result_b[0]["name_ja"], "B only")

    def test_get_empty_returns_empty_list(self) -> None:
        """get_sop_steps for a task with no rows returns an empty list."""
        result = self.db.get_sop_steps("nonexistent-task")
        self.assertEqual(result, [])

    def test_delete_nonexistent_returns_zero(self) -> None:
        """Deleting from a task that has no steps returns 0."""
        count = self.db.delete_sop_steps("no-such-task")
        self.assertEqual(count, 0)


# ---------------------------------------------------------------------------
# Class 2: TestStepDefinitionService
# ---------------------------------------------------------------------------

class TestStepDefinitionService(unittest.TestCase):
    """Tests for StepDefinitionService business logic."""

    TASK_ID = "svc-task"

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = _make_db(pathlib.Path(self._tmp.name))
        self.svc = StepDefinitionService(self.db)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_get_empty(self) -> None:
        """Before any steps are defined, get_steps returns step_count=0 and empty list."""
        result = self.svc.get_steps(self.TASK_ID)
        self.assertEqual(result["task_id"], self.TASK_ID)
        self.assertEqual(result["step_count"], 0)
        self.assertEqual(result["steps"], [])

    def test_upsert_and_get(self) -> None:
        """Upsert 3 steps and verify get_steps returns them ordered by step_index."""
        steps = [
            {"step_index": 2, "name_ja": "手順3"},
            {"step_index": 0, "name_ja": "手順1"},
            {"step_index": 1, "name_ja": "手順2"},
        ]
        result = self.svc.upsert_steps(self.TASK_ID, steps)
        self.assertEqual(result["step_count"], 3)

        fetched = self.svc.get_steps(self.TASK_ID)
        self.assertEqual(fetched["step_count"], 3)
        indices = [s["step_index"] for s in fetched["steps"]]
        self.assertEqual(indices, [0, 1, 2])

    def test_validation_negative_step_index(self) -> None:
        """step_index=-1 must raise ValueError."""
        with self.assertRaises(ValueError):
            self.svc.upsert_steps(self.TASK_ID, [{"step_index": -1, "name_ja": "bad"}])

    def test_validation_non_integer_step_index(self) -> None:
        """step_index as a string must raise ValueError."""
        with self.assertRaises(ValueError):
            self.svc.upsert_steps(self.TASK_ID, [{"step_index": "0", "name_ja": "bad"}])

    def test_validation_min_gt_max(self) -> None:
        """min_duration_sec > max_duration_sec must raise ValueError."""
        with self.assertRaises(ValueError):
            self.svc.upsert_steps(self.TASK_ID, [{
                "step_index": 0,
                "min_duration_sec": 20.0,
                "max_duration_sec": 5.0,
            }])

    def test_validation_expected_duration_non_positive(self) -> None:
        """expected_duration_sec=0 must raise ValueError."""
        with self.assertRaises(ValueError):
            self.svc.upsert_steps(self.TASK_ID, [{
                "step_index": 0,
                "expected_duration_sec": 0.0,
            }])

    def test_delete_steps(self) -> None:
        """delete_steps removes all definitions and returns the correct deleted_count."""
        self.svc.upsert_steps(self.TASK_ID, [
            {"step_index": 0, "name_ja": "X"},
            {"step_index": 1, "name_ja": "Y"},
        ])
        result = self.svc.delete_steps(self.TASK_ID)
        self.assertEqual(result["deleted_count"], 2)
        self.assertEqual(self.svc.get_steps(self.TASK_ID)["step_count"], 0)

    def test_compute_time_compliance_ok(self) -> None:
        """boundaries=[0,3,8], clip_seconds=4, step 0 -> 12s (min=8, max=20) -> 'ok'."""
        self.svc.upsert_steps(self.TASK_ID, [{
            "step_index": 0,
            "name_ja": "手順A",
            "min_duration_sec": 8.0,
            "max_duration_sec": 20.0,
        }])
        results = self.svc.compute_time_compliance(
            self.TASK_ID, [0, 3], clip_seconds=4
        )
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0]["actual_duration_sec"], 12.0)
        self.assertEqual(results[0]["compliance"], "ok")

    def test_compute_time_compliance_too_fast(self) -> None:
        """1 clip * 4s = 4s < min=8s -> 'too_fast'."""
        self.svc.upsert_steps(self.TASK_ID, [{
            "step_index": 0,
            "name_ja": "速い手順",
            "min_duration_sec": 8.0,
        }])
        results = self.svc.compute_time_compliance(
            self.TASK_ID, [0, 1], clip_seconds=4
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["compliance"], "too_fast")
        self.assertAlmostEqual(results[0]["actual_duration_sec"], 4.0)

    def test_compute_time_compliance_too_slow(self) -> None:
        """10 clips * 4s = 40s > max=20s -> 'too_slow'."""
        self.svc.upsert_steps(self.TASK_ID, [{
            "step_index": 0,
            "name_ja": "遅い手順",
            "max_duration_sec": 20.0,
        }])
        results = self.svc.compute_time_compliance(
            self.TASK_ID, [0, 10], clip_seconds=4
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["compliance"], "too_slow")
        self.assertAlmostEqual(results[0]["actual_duration_sec"], 40.0)

    def test_compute_time_compliance_undefined(self) -> None:
        """Step with no min or max -> compliance='undefined'."""
        self.svc.upsert_steps(self.TASK_ID, [{
            "step_index": 0,
            "name_ja": "未定義手順",
        }])
        results = self.svc.compute_time_compliance(
            self.TASK_ID, [0, 5], clip_seconds=4
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["compliance"], "undefined")

    def test_step_names_in_compliance(self) -> None:
        """Compliance results must carry name_ja / name_en from the step definitions."""
        self.svc.upsert_steps(self.TASK_ID, [{
            "step_index": 0,
            "name_ja": "カスタム名",
            "name_en": "Custom Name",
            "min_duration_sec": 1.0,
            "max_duration_sec": 100.0,
        }])
        results = self.svc.compute_time_compliance(
            self.TASK_ID, [0, 3], clip_seconds=4
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name_ja"], "カスタム名")
        self.assertEqual(results[0]["name_en"], "Custom Name")

    def test_compute_time_compliance_no_steps_defined(self) -> None:
        """When no step defs are stored, compliance defaults to 'undefined' for all steps."""
        results = self.svc.compute_time_compliance(
            self.TASK_ID, [0, 2, 5], clip_seconds=4
        )
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(r["compliance"], "undefined")

    def test_compute_time_compliance_empty_boundaries(self) -> None:
        """An empty or single-element boundaries list -> 0 steps returned."""
        self.svc.upsert_steps(self.TASK_ID, [{"step_index": 0, "name_ja": "手順"}])
        result_none = self.svc.compute_time_compliance(self.TASK_ID, [], clip_seconds=4)
        self.assertEqual(len(result_none), 0)
        result_one = self.svc.compute_time_compliance(self.TASK_ID, [0], clip_seconds=4)
        self.assertEqual(len(result_one), 0)


# ---------------------------------------------------------------------------
# Class 3: TestTimeComplianceScoring
# ---------------------------------------------------------------------------

class TestTimeComplianceScoring(unittest.TestCase):
    """Tests for compute_time_compliance_per_step and compute_step_contributions."""

    def test_compute_time_compliance_basic(self) -> None:
        """boundaries=[0,2,5], clip_seconds=4 -> step0=8s (ok), step1=12s (ok)."""
        step_defs = [
            {
                "step_index": 0,
                "name_ja": "手順A",
                "min_duration_sec": 5.0,
                "max_duration_sec": 20.0,
            },
            {
                "step_index": 1,
                "name_ja": "手順B",
                "min_duration_sec": 10.0,
                "max_duration_sec": 30.0,
            },
        ]
        result = compute_time_compliance_per_step([0, 2, 5], step_defs, clip_seconds=4)
        self.assertEqual(len(result), 2)

        self.assertEqual(result[0]["step_index"], 0)
        self.assertAlmostEqual(result[0]["actual_duration_sec"], 8.0)
        self.assertEqual(result[0]["compliance"], "ok")
        self.assertEqual(result[0]["name_ja"], "手順A")

        self.assertEqual(result[1]["step_index"], 1)
        self.assertAlmostEqual(result[1]["actual_duration_sec"], 12.0)
        self.assertEqual(result[1]["compliance"], "ok")
        self.assertEqual(result[1]["name_ja"], "手順B")

    def test_compute_time_compliance_too_fast(self) -> None:
        """2 clips * 4s = 8s < min=20s -> 'too_fast'."""
        step_defs = [{"step_index": 0, "min_duration_sec": 20.0}]
        result = compute_time_compliance_per_step([0, 2], step_defs, clip_seconds=4)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["compliance"], "too_fast")

    def test_compute_time_compliance_too_slow(self) -> None:
        """10 clips * 4s = 40s > max=20s -> 'too_slow'."""
        step_defs = [{"step_index": 0, "max_duration_sec": 20.0}]
        result = compute_time_compliance_per_step([0, 10], step_defs, clip_seconds=4)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["compliance"], "too_slow")

    def test_compute_time_compliance_undefined_no_bounds(self) -> None:
        """Step def with neither min nor max -> compliance='undefined'."""
        step_defs = [{"step_index": 0, "name_ja": "未定義"}]
        result = compute_time_compliance_per_step([0, 3], step_defs, clip_seconds=4)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["compliance"], "undefined")

    def test_compute_time_compliance_empty_step_defs(self) -> None:
        """No step definitions -> every step gets compliance='undefined'."""
        result = compute_time_compliance_per_step([0, 2, 5], [], clip_seconds=4)
        self.assertEqual(len(result), 2)
        for r in result:
            self.assertEqual(r["compliance"], "undefined")

    def test_compute_time_compliance_empty_boundaries(self) -> None:
        """Fewer than 2 boundary points -> zero steps returned."""
        result = compute_time_compliance_per_step([], [], clip_seconds=4)
        self.assertEqual(result, [])
        result_one = compute_time_compliance_per_step([0], [], clip_seconds=4)
        self.assertEqual(result_one, [])

    def test_compute_time_compliance_default_names(self) -> None:
        """When a step def lacks name_ja, the result uses the default fallback name."""
        step_defs = [{"step_index": 0}]
        result = compute_time_compliance_per_step([0, 3], step_defs, clip_seconds=4)
        self.assertEqual(len(result), 1)
        # Default fallback for step 0 is '手順1'
        self.assertEqual(result[0]["name_ja"], "手順1")
        self.assertEqual(result[0]["name_en"], "Step 1")

    def test_compute_time_compliance_is_critical_field(self) -> None:
        """is_critical from step def must propagate to compliance result."""
        step_defs = [{"step_index": 0, "is_critical": True}]
        result = compute_time_compliance_per_step([0, 1], step_defs, clip_seconds=4)
        self.assertTrue(result[0]["is_critical"])

    def test_step_contributions_include_names_when_defs_provided(self) -> None:
        """compute_step_contributions with step_definitions merges name_ja, name_en, is_critical."""
        step_defs = [
            {
                "step_index": 0,
                "name_ja": "消毒作業",
                "name_en": "Sanitize",
                "is_critical": True,
            },
        ]
        weights = ScoreWeights()
        contribs = compute_step_contributions(
            deviations=[],
            boundaries=[3],
            gold_len=3,
            weights=weights,
            step_definitions=step_defs,
        )
        self.assertGreater(len(contribs), 0)
        self.assertEqual(contribs[0]["name_ja"], "消毒作業")
        self.assertEqual(contribs[0]["name_en"], "Sanitize")
        self.assertTrue(contribs[0]["is_critical"])

    def test_step_contributions_no_defs(self) -> None:
        """Without step_definitions, contributions still work and have no name_ja field."""
        weights = ScoreWeights()
        contribs = compute_step_contributions(
            deviations=[],
            boundaries=[3],
            gold_len=3,
            weights=weights,
        )
        self.assertGreater(len(contribs), 0)
        # No name fields when step_definitions is not provided
        self.assertNotIn("name_ja", contribs[0])
        self.assertNotIn("name_en", contribs[0])

    def test_step_contributions_multiple_steps_with_defs(self) -> None:
        """Names should be merged correctly for each step when multiple defs are provided."""
        step_defs = [
            {"step_index": 0, "name_ja": "ステップ0", "name_en": "Step0", "is_critical": False},
            {"step_index": 1, "name_ja": "ステップ1", "name_en": "Step1", "is_critical": True},
        ]
        weights = ScoreWeights()
        contribs = compute_step_contributions(
            deviations=[],
            boundaries=[3],
            gold_len=6,
            weights=weights,
            step_definitions=step_defs,
        )
        self.assertEqual(len(contribs), 2)
        self.assertEqual(contribs[0]["name_ja"], "ステップ0")
        self.assertFalse(contribs[0]["is_critical"])
        self.assertEqual(contribs[1]["name_ja"], "ステップ1")
        self.assertTrue(contribs[1]["is_critical"])

    def test_step_contributions_deviation_reduces_points_with_defs(self) -> None:
        """A missing_step deviation still reduces points when step_definitions is provided."""
        step_defs = [{"step_index": 0, "name_ja": "重要手順", "is_critical": True}]
        deviations = [{"step_index": 0, "type": "missing_step"}]
        weights = ScoreWeights()
        contribs = compute_step_contributions(
            deviations=deviations,
            boundaries=[3],
            gold_len=3,
            weights=weights,
            step_definitions=step_defs,
        )
        self.assertEqual(len(contribs), 1)
        self.assertLess(contribs[0]["points_earned"], contribs[0]["points_possible"])
        self.assertEqual(contribs[0]["name_ja"], "重要手順")


# ---------------------------------------------------------------------------
# Class 4: TestStepDefsEndpoints
# ---------------------------------------------------------------------------

class TestStepDefsEndpoints(unittest.TestCase):
    """HTTP endpoint tests for GET/PUT/DELETE /tasks/steps."""

    _ENV_KEYS = ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_RATE_LIMIT_RPM")

    def _make_client(self, tmp_dir: str):
        from fastapi.testclient import TestClient

        from sopilot.main import create_app
        os.environ["SOPILOT_DATA_DIR"] = str(pathlib.Path(tmp_dir) / "data")
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_RATE_LIMIT_RPM"] = "0"
        app = create_app()
        return TestClient(app)

    def _cleanup_env(self) -> None:
        for key in self._ENV_KEYS:
            os.environ.pop(key, None)

    def test_get_steps_empty(self) -> None:
        """GET /tasks/steps on a fresh database returns step_count=0 and empty steps list."""
        with tempfile.TemporaryDirectory() as tmp:
            try:
                client = self._make_client(tmp)
                r = client.get("/tasks/steps")
                self.assertEqual(r.status_code, 200)
                data = r.json()
                self.assertEqual(data["step_count"], 0)
                self.assertEqual(data["steps"], [])
            finally:
                self._cleanup_env()

    def test_put_and_get_steps(self) -> None:
        """PUT /tasks/steps then GET /tasks/steps returns all upserted steps."""
        with tempfile.TemporaryDirectory() as tmp:
            try:
                client = self._make_client(tmp)
                steps = [
                    {
                        "step_index": 0,
                        "name_ja": "消毒",
                        "expected_duration_sec": 10.0,
                        "min_duration_sec": 5.0,
                        "max_duration_sec": 20.0,
                        "is_critical": True,
                    },
                    {
                        "step_index": 1,
                        "name_ja": "器具準備",
                        "expected_duration_sec": 15.0,
                    },
                ]
                r = client.put("/tasks/steps", json={"steps": steps})
                self.assertEqual(r.status_code, 200)
                put_data = r.json()
                self.assertEqual(put_data["step_count"], 2)

                r2 = client.get("/tasks/steps")
                self.assertEqual(r2.status_code, 200)
                get_data = r2.json()
                self.assertEqual(get_data["step_count"], 2)
                names = [s["name_ja"] for s in get_data["steps"]]
                self.assertIn("消毒", names)
                self.assertIn("器具準備", names)
            finally:
                self._cleanup_env()

    def test_put_updates_existing_step(self) -> None:
        """Calling PUT twice with the same step_index updates the name."""
        with tempfile.TemporaryDirectory() as tmp:
            try:
                client = self._make_client(tmp)
                client.put("/tasks/steps", json={"steps": [{"step_index": 0, "name_ja": "旧名称"}]})
                client.put("/tasks/steps", json={"steps": [{"step_index": 0, "name_ja": "新名称"}]})

                r = client.get("/tasks/steps")
                self.assertEqual(r.status_code, 200)
                data = r.json()
                self.assertEqual(data["step_count"], 1)
                self.assertEqual(data["steps"][0]["name_ja"], "新名称")
            finally:
                self._cleanup_env()

    def test_delete_steps(self) -> None:
        """DELETE /tasks/steps removes all steps and returns deleted_count."""
        with tempfile.TemporaryDirectory() as tmp:
            try:
                client = self._make_client(tmp)
                client.put(
                    "/tasks/steps",
                    json={"steps": [{"step_index": 0, "name_ja": "テスト"}]},
                )
                r = client.delete("/tasks/steps")
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json()["deleted_count"], 1)

                # Verify it is now empty
                r2 = client.get("/tasks/steps")
                self.assertEqual(r2.status_code, 200)
                self.assertEqual(r2.json()["step_count"], 0)
            finally:
                self._cleanup_env()

    def test_delete_empty_returns_zero(self) -> None:
        """DELETE /tasks/steps on an empty task returns deleted_count=0 with 200."""
        with tempfile.TemporaryDirectory() as tmp:
            try:
                client = self._make_client(tmp)
                r = client.delete("/tasks/steps")
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json()["deleted_count"], 0)
            finally:
                self._cleanup_env()

    def test_put_invalid_payload_steps_not_list(self) -> None:
        """PUT /tasks/steps with steps as a string must return 422."""
        with tempfile.TemporaryDirectory() as tmp:
            try:
                client = self._make_client(tmp)
                r = client.put("/tasks/steps", json={"steps": "not-a-list"})
                self.assertEqual(r.status_code, 422)
            finally:
                self._cleanup_env()

    def test_put_empty_steps_list(self) -> None:
        """PUT /tasks/steps with an empty list is valid and returns step_count=0."""
        with tempfile.TemporaryDirectory() as tmp:
            try:
                client = self._make_client(tmp)
                r = client.put("/tasks/steps", json={"steps": []})
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json()["step_count"], 0)
            finally:
                self._cleanup_env()

    def test_get_steps_response_structure(self) -> None:
        """GET /tasks/steps response must include task_id, step_count, and steps fields."""
        with tempfile.TemporaryDirectory() as tmp:
            try:
                client = self._make_client(tmp)
                r = client.get("/tasks/steps")
                self.assertEqual(r.status_code, 200)
                data = r.json()
                self.assertIn("task_id", data)
                self.assertIn("step_count", data)
                self.assertIn("steps", data)
                self.assertIsInstance(data["steps"], list)
            finally:
                self._cleanup_env()

    def test_put_preserves_step_fields(self) -> None:
        """PUT then GET must preserve is_critical, min/max/expected duration."""
        with tempfile.TemporaryDirectory() as tmp:
            try:
                client = self._make_client(tmp)
                step = {
                    "step_index": 0,
                    "name_ja": "詳細手順",
                    "name_en": "Detailed Step",
                    "expected_duration_sec": 12.0,
                    "min_duration_sec": 6.0,
                    "max_duration_sec": 24.0,
                    "is_critical": True,
                }
                client.put("/tasks/steps", json={"steps": [step]})
                r = client.get("/tasks/steps")
                returned = r.json()["steps"][0]
                self.assertEqual(returned["name_en"], "Detailed Step")
                self.assertAlmostEqual(returned["expected_duration_sec"], 12.0)
                self.assertAlmostEqual(returned["min_duration_sec"], 6.0)
                self.assertAlmostEqual(returned["max_duration_sec"], 24.0)
                self.assertTrue(returned["is_critical"])
            finally:
                self._cleanup_env()


if __name__ == "__main__":
    unittest.main()
