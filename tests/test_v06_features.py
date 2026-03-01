"""Comprehensive tests for SOPilot v0.6 features.

Covers:
- Database.get_operator_trend
- Database.get_step_performance
- Database.get_compliance_overview
- RecommendationService.get_recommendations
- scoring.compute_step_contributions / compute_score_confidence
- /analytics/report/pdf HTTP endpoint
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING

from sopilot.core.scoring import ScoreWeights, compute_score_confidence, compute_step_contributions
from sopilot.database import Database
from sopilot.services.recommendation_service import RecommendationService

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path: Path) -> Database:
    """Return a fresh, empty Database pointing at *tmp_path*."""
    return Database(tmp_path / "test.db")


def _insert_gold(db: Database, task_id: str = "task-1") -> int:
    return db.insert_video(
        task_id=task_id,
        site_id="site-a",
        camera_id=None,
        operator_id_hash=None,
        recorded_at=None,
        is_gold=True,
    )


def _insert_trainee(
    db: Database,
    *,
    task_id: str = "task-1",
    operator_id_hash: str | None = "op-1",
    site_id: str | None = "site-a",
) -> int:
    return db.insert_video(
        task_id=task_id,
        site_id=site_id,
        camera_id=None,
        operator_id_hash=operator_id_hash,
        recorded_at=None,
        is_gold=False,
    )


def _completed_job(
    db: Database,
    gold_id: int,
    trainee_id: int,
    score: float,
    decision: str,
    deviations: list[dict] | None = None,
) -> int:
    """Create a completed score job with optional deviations in score_json."""
    job_id = db.create_score_job(gold_id, trainee_id)
    db.claim_score_job(job_id)
    payload: dict = {
        "score": score,
        "summary": {"decision": decision},
        "metrics": {},
        "deviations": deviations or [],
    }
    db.complete_score_job(job_id, payload)
    return job_id


# ---------------------------------------------------------------------------
# 1.  TestOperatorTrend
# ---------------------------------------------------------------------------

class TestOperatorTrend(unittest.TestCase):
    """Tests for Database.get_operator_trend."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = _make_db(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_empty_operator_returns_no_jobs(self) -> None:
        """An operator with no completed jobs returns job_count=0 and empty jobs list."""
        result = self.db.get_operator_trend("nonexistent-op")
        self.assertEqual(result["job_count"], 0)
        self.assertEqual(result["jobs"], [])
        self.assertIsNone(result["avg_score"])
        self.assertIsNone(result["latest_decision"])

    def test_operator_with_one_job(self) -> None:
        """One completed job returns job_count=1 with a valid score but no slope/improvement."""
        gold = _insert_gold(self.db)
        trainee = _insert_trainee(self.db, operator_id_hash="op-solo")
        _completed_job(self.db, gold, trainee, 85.0, "pass")

        result = self.db.get_operator_trend("op-solo")
        self.assertEqual(result["job_count"], 1)
        self.assertEqual(len(result["jobs"]), 1)
        self.assertEqual(result["jobs"][0]["score"], 85.0)
        # With a single data point the denominator in the linear regression is 0
        # so trend_slope comes back None.
        self.assertIsNone(result["trend_slope"])
        # improvement_pct = (last - first) / first * 100; for one job that is 0.0
        self.assertEqual(result["improvement_pct"], 0.0)

    def test_operator_trend_oldest_first(self) -> None:
        """Multiple completed jobs are returned in chronological (oldest-first) order."""
        gold = _insert_gold(self.db)
        trainee = _insert_trainee(self.db, operator_id_hash="op-chrono")

        # Insert 3 jobs; the DB orders by created_at ASC
        _completed_job(self.db, gold, trainee, 60.0, "fail")
        _completed_job(self.db, gold, trainee, 75.0, "needs_review")
        _completed_job(self.db, gold, trainee, 90.0, "pass")

        result = self.db.get_operator_trend("op-chrono")
        self.assertEqual(result["job_count"], 3)
        scores = [j["score"] for j in result["jobs"]]
        # The scores should be returned in insertion (chronological) order
        self.assertEqual(scores, [60.0, 75.0, 90.0])

    def test_operator_trend_task_filter(self) -> None:
        """task_id filter returns only jobs whose gold video belongs to that task."""
        gold_a = _insert_gold(self.db, task_id="task-A")
        gold_b = _insert_gold(self.db, task_id="task-B")
        trainee_a = _insert_trainee(self.db, task_id="task-A", operator_id_hash="op-filter")
        trainee_b = _insert_trainee(self.db, task_id="task-B", operator_id_hash="op-filter")

        _completed_job(self.db, gold_a, trainee_a, 80.0, "pass")
        _completed_job(self.db, gold_a, trainee_a, 85.0, "pass")
        _completed_job(self.db, gold_b, trainee_b, 50.0, "fail")

        # Filter to task-A only — should see 2 jobs
        result = self.db.get_operator_trend("op-filter", task_id="task-A")
        self.assertEqual(result["job_count"], 2)
        for job in result["jobs"]:
            self.assertGreaterEqual(job["score"], 80.0)

        # Filter to task-B only — should see 1 job
        result_b = self.db.get_operator_trend("op-filter", task_id="task-B")
        self.assertEqual(result_b["job_count"], 1)
        self.assertEqual(result_b["jobs"][0]["score"], 50.0)


# ---------------------------------------------------------------------------
# 2.  TestStepPerformance
# ---------------------------------------------------------------------------

class TestStepPerformance(unittest.TestCase):
    """Tests for Database.get_step_performance."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = _make_db(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_no_jobs_returns_empty_steps(self) -> None:
        """An empty database returns total_jobs=0 and an empty steps list."""
        result = self.db.get_step_performance()
        self.assertEqual(result["total_jobs"], 0)
        self.assertEqual(result["steps"], [])
        self.assertIsNone(result["hardest_step_index"])

    def test_steps_aggregated_correctly(self) -> None:
        """A completed job with a deviation at step_index=2 appears in the steps list."""
        gold = _insert_gold(self.db)
        trainee = _insert_trainee(self.db)
        devs = [{"step_index": 2, "type": "missing_step", "severity": "critical"}]
        _completed_job(self.db, gold, trainee, 72.0, "retrain", deviations=devs)

        result = self.db.get_step_performance()
        self.assertEqual(result["total_jobs"], 1)
        self.assertTrue(len(result["steps"]) > 0)

        step_map = {s["step_index"]: s for s in result["steps"]}
        self.assertIn(2, step_map)
        # miss_rate for step 2 = 1 miss / 1 total job = 1.0
        self.assertAlmostEqual(step_map[2]["miss_rate"], 1.0)

    def test_hardest_step_identified(self) -> None:
        """When step 2 has more issues than step 0, hardest_step_index should be 2."""
        gold = _insert_gold(self.db)
        trainee = _insert_trainee(self.db)

        # Job 1: deviation at step 0 only
        devs1 = [{"step_index": 0, "type": "step_deviation"}]
        _completed_job(self.db, gold, trainee, 85.0, "pass", deviations=devs1)

        # Job 2: deviations at step 2 (two issues makes it the hardest)
        devs2 = [
            {"step_index": 2, "type": "missing_step"},
            {"step_index": 2, "type": "missing_step"},
        ]
        _completed_job(self.db, gold, trainee, 55.0, "fail", deviations=devs2)

        # Job 3: another deviation at step 2 to widen the gap
        devs3 = [{"step_index": 2, "type": "missing_step"}]
        _completed_job(self.db, gold, trainee, 60.0, "fail", deviations=devs3)

        result = self.db.get_step_performance()
        # step 2 has issues in more jobs (2 out of 3) vs step 0 (1 out of 3)
        self.assertEqual(result["hardest_step_index"], 2)


# ---------------------------------------------------------------------------
# 3.  TestComplianceOverview
# ---------------------------------------------------------------------------

class TestComplianceOverview(unittest.TestCase):
    """Tests for Database.get_compliance_overview."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = _make_db(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_empty_returns_zero_compliance(self) -> None:
        """No completed jobs → compliance_rate = 0.0."""
        result = self.db.get_compliance_overview()
        self.assertEqual(result["compliance_rate"], 0.0)

    def test_compliance_rate_calculation(self) -> None:
        """8 passes out of 10 completed jobs → compliance_rate = 0.8."""
        gold = _insert_gold(self.db)
        trainee = _insert_trainee(self.db)

        for _ in range(8):
            _completed_job(self.db, gold, trainee, 90.0, "pass")
        for _ in range(2):
            _completed_job(self.db, gold, trainee, 40.0, "fail")

        result = self.db.get_compliance_overview()
        self.assertAlmostEqual(result["compliance_rate"], 0.8, places=4)

    def test_trend_direction(self) -> None:
        """The trend_direction field reflects whether compliance is improving or declining."""
        gold = _insert_gold(self.db)
        trainee = _insert_trainee(self.db)

        # First 2 jobs fail (early, low compliance)
        _completed_job(self.db, gold, trainee, 40.0, "fail")
        _completed_job(self.db, gold, trainee, 45.0, "fail")
        # Last 2 jobs pass (recent, high compliance)
        _completed_job(self.db, gold, trainee, 90.0, "pass")
        _completed_job(self.db, gold, trainee, 95.0, "pass")

        result = self.db.get_compliance_overview()
        # The result must have a trend_direction key
        self.assertIn("trend_direction", result)
        # Recent half (2 passes) is better than earlier half (2 fails) → improving
        self.assertEqual(result["trend_direction"], "improving")


# ---------------------------------------------------------------------------
# 4.  TestRecommendationService
# ---------------------------------------------------------------------------

class TestRecommendationService(unittest.TestCase):
    """Tests for RecommendationService.get_recommendations."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = _make_db(Path(self._tmp.name))
        self.svc = RecommendationService(self.db)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_no_jobs_returns_empty_recommendations(self) -> None:
        """An operator with no jobs gets job_count=0 and empty recommendations."""
        result = self.svc.get_recommendations("op-nobody")
        self.assertEqual(result["job_count"], 0)
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(result["overall_assessment"], "データなし")

    def test_high_frequency_deviation_generates_recommendation(self) -> None:
        """5 jobs all with missing_step at step_index=2 → recommendation for step 2 appears."""
        gold = _insert_gold(self.db)
        trainee = _insert_trainee(self.db, operator_id_hash="op-repeat")

        devs = [{"step_index": 2, "type": "missing_step"}]
        for _ in range(5):
            _completed_job(self.db, gold, trainee, 60.0, "fail", deviations=devs)

        result = self.svc.get_recommendations("op-repeat")
        self.assertEqual(result["job_count"], 5)
        self.assertGreater(len(result["recommendations"]), 0)
        step_indices = [r["step_index"] for r in result["recommendations"]]
        self.assertIn(2, step_indices)

    def test_priority_critical_for_high_frequency(self) -> None:
        """When frequency >= 0.5, the recommendation priority must be 'critical'."""
        gold = _insert_gold(self.db)
        trainee = _insert_trainee(self.db, operator_id_hash="op-critical")

        # 5/5 = 100% frequency → critical
        devs = [{"step_index": 1, "type": "missing_step"}]
        for _ in range(5):
            _completed_job(self.db, gold, trainee, 55.0, "fail", deviations=devs)

        result = self.svc.get_recommendations("op-critical")
        recs = result["recommendations"]
        self.assertGreater(len(recs), 0)
        # The recommendation for step 1 must be critical
        step1_recs = [r for r in recs if r["step_index"] == 1]
        self.assertTrue(len(step1_recs) > 0)
        self.assertEqual(step1_recs[0]["priority"], "critical")

    def test_overall_assessment_requires_retrain(self) -> None:
        """A critical recommendation causes overall_assessment='要再研修'."""
        gold = _insert_gold(self.db)
        trainee = _insert_trainee(self.db, operator_id_hash="op-retrain")

        devs = [{"step_index": 0, "type": "missing_step"}]
        for _ in range(5):
            _completed_job(self.db, gold, trainee, 50.0, "fail", deviations=devs)

        result = self.svc.get_recommendations("op-retrain")
        self.assertEqual(result["overall_assessment"], "要再研修")


# ---------------------------------------------------------------------------
# 5.  TestStepContributions
# ---------------------------------------------------------------------------

class TestStepContributions(unittest.TestCase):
    """Tests for compute_step_contributions and compute_score_confidence."""

    def _default_weights(self) -> ScoreWeights:
        return ScoreWeights()

    def test_compute_step_contributions_empty(self) -> None:
        """No deviations → each step gets full points_earned == points_possible."""
        weights = self._default_weights()
        # 2 steps: boundaries=[5], gold_len=10
        contributions = compute_step_contributions(
            deviations=[],
            boundaries=[5],
            gold_len=10,
            weights=weights,
        )
        self.assertEqual(len(contributions), 2)
        for step in contributions:
            self.assertIn("step_index", step)
            self.assertIn("points_possible", step)
            self.assertIn("points_earned", step)
            self.assertIn("deductions", step)
            self.assertAlmostEqual(step["points_earned"], step["points_possible"], places=4)
            self.assertEqual(step["deductions"], [])

    def test_missing_step_reduces_points(self) -> None:
        """A missing_step deviation at step 0 → points_earned < points_possible for step 0."""
        weights = self._default_weights()
        deviations = [{"step_index": 0, "type": "missing_step"}]
        contributions = compute_step_contributions(
            deviations=deviations,
            boundaries=[5],
            gold_len=10,
            weights=weights,
        )
        self.assertEqual(len(contributions), 2)
        step0 = contributions[0]
        self.assertEqual(step0["step_index"], 0)
        self.assertLess(step0["points_earned"], step0["points_possible"])
        self.assertGreater(len(step0["deductions"]), 0)
        # Step 1 should be unaffected
        step1 = contributions[1]
        self.assertAlmostEqual(step1["points_earned"], step1["points_possible"], places=4)

    def test_compute_score_confidence_returns_dict(self) -> None:
        """compute_score_confidence with valid args returns a dict with required keys."""
        metrics = {"dtw_normalized_cost": 0.1}
        result = compute_score_confidence(
            result_score=85.0,
            deviations=[],
            metrics=metrics,
            gold_len=20,
            trainee_len=22,
        )
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("ci_low", result)
        self.assertIn("ci_high", result)
        self.assertIn("stability", result)
        self.assertGreaterEqual(result["ci_low"], 0.0)
        self.assertLessEqual(result["ci_high"], 100.0)
        self.assertIn(result["stability"], ("high", "medium", "low"))


# ---------------------------------------------------------------------------
# 6.  TestAnalyticsPDFEndpoint
# ---------------------------------------------------------------------------

def _make_test_client(tmp_dir: Path) -> TestClient:
    """Create a FastAPI TestClient wired to a temp directory."""
    from fastapi.testclient import TestClient

    from sopilot.main import create_app

    os.environ["SOPILOT_DATA_DIR"] = str(tmp_dir / "data")
    os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
    os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-pdf-test"
    os.environ.pop("SOPILOT_API_KEY", None)
    return TestClient(create_app())


class TestAnalyticsPDFEndpoint(unittest.TestCase):
    """Tests for GET /analytics/report/pdf."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._root = Path(self._tmp.name)
        self._client = _make_test_client(self._root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_pdf_endpoint_returns_pdf_bytes(self) -> None:
        """GET /analytics/report/pdf returns HTTP 200 with Content-Type application/pdf."""
        resp = self._client.get("/analytics/report/pdf")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertIn("application/pdf", resp.headers.get("content-type", ""))
        # PDF magic bytes
        self.assertTrue(resp.content[:4] == b"%PDF", "Response body does not start with %PDF")

    def test_pdf_endpoint_with_days_filter(self) -> None:
        """GET /analytics/report/pdf?days=30 also returns a valid PDF."""
        resp = self._client.get("/analytics/report/pdf?days=30")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertIn("application/pdf", resp.headers.get("content-type", ""))
        self.assertTrue(resp.content[:4] == b"%PDF", "Response body does not start with %PDF")


if __name__ == "__main__":
    unittest.main()
