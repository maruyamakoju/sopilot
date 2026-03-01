"""Tests for Database.get_analytics() — aggregate scoring analytics."""

import tempfile
import unittest
from pathlib import Path

from sopilot.database import Database


class _AnalyticsTestBase(unittest.TestCase):
    """Base class with helpers for setting up a database with controlled score data."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.db = Database(self.tmp_path / "test.db")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _insert_gold_video(
        self,
        task_id: str = "task-1",
        site_id: str | None = "site-a",
    ) -> int:
        """Insert a gold video and return its id."""
        return self.db.insert_video(
            task_id=task_id,
            site_id=site_id,
            camera_id=None,
            operator_id_hash=None,
            recorded_at=None,
            is_gold=True,
        )

    def _insert_trainee_video(
        self,
        task_id: str = "task-1",
        site_id: str | None = "site-a",
        operator_id_hash: str | None = "op-1",
    ) -> int:
        """Insert a trainee video and return its id."""
        return self.db.insert_video(
            task_id=task_id,
            site_id=site_id,
            camera_id=None,
            operator_id_hash=operator_id_hash,
            recorded_at=None,
            is_gold=False,
        )

    def _create_completed_job(
        self,
        gold_video_id: int,
        trainee_video_id: int,
        score: float,
        decision: str,
    ) -> int:
        """Create a score job and immediately complete it with given score/decision."""
        job_id = self.db.create_score_job(gold_video_id, trainee_video_id)
        self.db.claim_score_job(job_id)
        score_payload = {
            "score": score,
            "summary": {"decision": decision},
        }
        self.db.complete_score_job(job_id, score_payload)
        return job_id

    def _create_pending_job(
        self,
        gold_video_id: int,
        trainee_video_id: int,
    ) -> int:
        """Create a queued (pending) score job."""
        return self.db.create_score_job(gold_video_id, trainee_video_id)

    def _create_failed_job(
        self,
        gold_video_id: int,
        trainee_video_id: int,
    ) -> int:
        """Create a score job that has failed."""
        job_id = self.db.create_score_job(gold_video_id, trainee_video_id)
        self.db.claim_score_job(job_id)
        self.db.fail_score_job(job_id, "test error")
        return job_id


class TestAnalyticsEmptyDatabase(_AnalyticsTestBase):
    """Analytics on an empty database should return sensible zero defaults."""

    def test_empty_database_returns_zeros(self) -> None:
        result = self.db.get_analytics()
        self.assertEqual(result["total_jobs"], 0)
        self.assertEqual(result["completed_jobs"], 0)
        self.assertEqual(result["pass_count"], 0)
        self.assertEqual(result["fail_count"], 0)
        self.assertEqual(result["needs_review_count"], 0)
        self.assertEqual(result["retrain_count"], 0)

    def test_empty_database_scores_are_none(self) -> None:
        result = self.db.get_analytics()
        self.assertIsNone(result["avg_score"])
        self.assertIsNone(result["min_score"])
        self.assertIsNone(result["max_score"])

    def test_empty_database_aggregations_are_empty(self) -> None:
        result = self.db.get_analytics()
        self.assertEqual(result["by_operator"], [])
        self.assertEqual(result["by_site"], [])
        self.assertEqual(result["recent_trend"], [])

    def test_empty_database_score_distribution_all_zero(self) -> None:
        result = self.db.get_analytics()
        dist = {d["bucket"]: d["count"] for d in result["score_distribution"]}
        self.assertEqual(dist["90-100"], 0)
        self.assertEqual(dist["80-89"], 0)
        self.assertEqual(dist["70-79"], 0)
        self.assertEqual(dist["0-69"], 0)

    def test_empty_database_distribution_has_four_buckets(self) -> None:
        result = self.db.get_analytics()
        self.assertEqual(len(result["score_distribution"]), 4)
        buckets = [d["bucket"] for d in result["score_distribution"]]
        self.assertEqual(buckets, ["90-100", "80-89", "70-79", "0-69"])


class TestAnalyticsOverallCounts(_AnalyticsTestBase):
    """Verify total_jobs, completed_jobs, and decision counts."""

    def test_completed_jobs_counted(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()
        self._create_completed_job(gold, trainee, 95.0, "pass")
        self._create_completed_job(gold, trainee, 85.0, "pass")
        self._create_completed_job(gold, trainee, 60.0, "fail")

        result = self.db.get_analytics()
        self.assertEqual(result["total_jobs"], 3)
        self.assertEqual(result["completed_jobs"], 3)
        self.assertEqual(result["pass_count"], 2)
        self.assertEqual(result["fail_count"], 1)
        self.assertEqual(result["needs_review_count"], 0)
        self.assertEqual(result["retrain_count"], 0)

    def test_all_decision_types_counted(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()
        self._create_completed_job(gold, trainee, 95.0, "pass")
        self._create_completed_job(gold, trainee, 50.0, "fail")
        self._create_completed_job(gold, trainee, 75.0, "needs_review")
        self._create_completed_job(gold, trainee, 40.0, "retrain")

        result = self.db.get_analytics()
        self.assertEqual(result["total_jobs"], 4)
        self.assertEqual(result["completed_jobs"], 4)
        self.assertEqual(result["pass_count"], 1)
        self.assertEqual(result["fail_count"], 1)
        self.assertEqual(result["needs_review_count"], 1)
        self.assertEqual(result["retrain_count"], 1)


class TestAnalyticsScoreStatistics(_AnalyticsTestBase):
    """Verify avg, min, max score calculations."""

    def test_single_completed_job_scores(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()
        self._create_completed_job(gold, trainee, 88.5, "pass")

        result = self.db.get_analytics()
        self.assertEqual(result["avg_score"], 88.5)
        self.assertEqual(result["min_score"], 88.5)
        self.assertEqual(result["max_score"], 88.5)

    def test_multiple_completed_jobs_score_stats(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()
        self._create_completed_job(gold, trainee, 90.0, "pass")
        self._create_completed_job(gold, trainee, 80.0, "pass")
        self._create_completed_job(gold, trainee, 70.0, "needs_review")

        result = self.db.get_analytics()
        self.assertEqual(result["avg_score"], 80.0)
        self.assertEqual(result["min_score"], 70.0)
        self.assertEqual(result["max_score"], 90.0)

    def test_avg_score_rounded_to_two_decimals(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()
        # 91 + 82 + 73 = 246 / 3 = 82.0
        self._create_completed_job(gold, trainee, 91.0, "pass")
        self._create_completed_job(gold, trainee, 82.0, "pass")
        self._create_completed_job(gold, trainee, 73.0, "needs_review")

        result = self.db.get_analytics()
        self.assertEqual(result["avg_score"], 82.0)

    def test_avg_score_with_thirds(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()
        # 100 + 90 + 80 = 270 / 3 = 90.0
        self._create_completed_job(gold, trainee, 100.0, "pass")
        self._create_completed_job(gold, trainee, 90.0, "pass")
        self._create_completed_job(gold, trainee, 80.0, "pass")

        result = self.db.get_analytics()
        self.assertEqual(result["avg_score"], 90.0)


class TestAnalyticsScoreDistribution(_AnalyticsTestBase):
    """Verify score distribution buckets (90-100, 80-89, 70-79, 0-69)."""

    def test_all_buckets_populated(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()
        # 90-100 bucket
        self._create_completed_job(gold, trainee, 95.0, "pass")
        self._create_completed_job(gold, trainee, 100.0, "pass")
        # 80-89 bucket
        self._create_completed_job(gold, trainee, 85.0, "pass")
        # 70-79 bucket
        self._create_completed_job(gold, trainee, 75.0, "needs_review")
        self._create_completed_job(gold, trainee, 70.0, "needs_review")
        self._create_completed_job(gold, trainee, 79.0, "needs_review")
        # 0-69 bucket
        self._create_completed_job(gold, trainee, 50.0, "fail")

        result = self.db.get_analytics()
        dist = {d["bucket"]: d["count"] for d in result["score_distribution"]}
        self.assertEqual(dist["90-100"], 2)
        self.assertEqual(dist["80-89"], 1)
        self.assertEqual(dist["70-79"], 3)
        self.assertEqual(dist["0-69"], 1)

    def test_boundary_score_90_in_upper_bucket(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()
        self._create_completed_job(gold, trainee, 90.0, "pass")

        result = self.db.get_analytics()
        dist = {d["bucket"]: d["count"] for d in result["score_distribution"]}
        self.assertEqual(dist["90-100"], 1)
        self.assertEqual(dist["80-89"], 0)

    def test_boundary_score_80_in_correct_bucket(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()
        self._create_completed_job(gold, trainee, 80.0, "pass")

        result = self.db.get_analytics()
        dist = {d["bucket"]: d["count"] for d in result["score_distribution"]}
        self.assertEqual(dist["90-100"], 0)
        self.assertEqual(dist["80-89"], 1)
        self.assertEqual(dist["70-79"], 0)

    def test_boundary_score_70_in_correct_bucket(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()
        self._create_completed_job(gold, trainee, 70.0, "needs_review")

        result = self.db.get_analytics()
        dist = {d["bucket"]: d["count"] for d in result["score_distribution"]}
        self.assertEqual(dist["80-89"], 0)
        self.assertEqual(dist["70-79"], 1)
        self.assertEqual(dist["0-69"], 0)

    def test_boundary_score_69_in_low_bucket(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()
        self._create_completed_job(gold, trainee, 69.0, "fail")

        result = self.db.get_analytics()
        dist = {d["bucket"]: d["count"] for d in result["score_distribution"]}
        self.assertEqual(dist["70-79"], 0)
        self.assertEqual(dist["0-69"], 1)

    def test_zero_score_in_low_bucket(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()
        self._create_completed_job(gold, trainee, 0.0, "fail")

        result = self.db.get_analytics()
        dist = {d["bucket"]: d["count"] for d in result["score_distribution"]}
        self.assertEqual(dist["0-69"], 1)


class TestAnalyticsByOperator(_AnalyticsTestBase):
    """Verify by_operator aggregation groups by trainee video's operator_id_hash."""

    def test_single_operator(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video(operator_id_hash="op-alpha")
        self._create_completed_job(gold, trainee, 90.0, "pass")
        self._create_completed_job(gold, trainee, 80.0, "fail")

        result = self.db.get_analytics()
        self.assertEqual(len(result["by_operator"]), 1)
        op = result["by_operator"][0]
        self.assertEqual(op["operator_id"], "op-alpha")
        self.assertEqual(op["job_count"], 2)
        self.assertEqual(op["avg_score"], 85.0)
        self.assertEqual(op["pass_count"], 1)
        self.assertEqual(op["fail_count"], 1)

    def test_multiple_operators(self) -> None:
        gold = self._insert_gold_video()
        t1 = self._insert_trainee_video(operator_id_hash="op-A")
        t2 = self._insert_trainee_video(operator_id_hash="op-B")
        self._create_completed_job(gold, t1, 95.0, "pass")
        self._create_completed_job(gold, t1, 85.0, "pass")
        self._create_completed_job(gold, t2, 60.0, "fail")

        result = self.db.get_analytics()
        self.assertEqual(len(result["by_operator"]), 2)
        op_map = {o["operator_id"]: o for o in result["by_operator"]}

        self.assertEqual(op_map["op-A"]["job_count"], 2)
        self.assertEqual(op_map["op-A"]["pass_count"], 2)
        self.assertEqual(op_map["op-A"]["fail_count"], 0)
        self.assertEqual(op_map["op-A"]["avg_score"], 90.0)

        self.assertEqual(op_map["op-B"]["job_count"], 1)
        self.assertEqual(op_map["op-B"]["pass_count"], 0)
        self.assertEqual(op_map["op-B"]["fail_count"], 1)
        self.assertEqual(op_map["op-B"]["avg_score"], 60.0)

    def test_null_operator_becomes_unknown(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video(operator_id_hash=None)
        self._create_completed_job(gold, trainee, 75.0, "needs_review")

        result = self.db.get_analytics()
        self.assertEqual(len(result["by_operator"]), 1)
        self.assertEqual(result["by_operator"][0]["operator_id"], "unknown")

    def test_operators_ordered_by_job_count_desc(self) -> None:
        gold = self._insert_gold_video()
        t_few = self._insert_trainee_video(operator_id_hash="op-few")
        t_many = self._insert_trainee_video(operator_id_hash="op-many")
        # op-many gets 3 jobs
        self._create_completed_job(gold, t_many, 90.0, "pass")
        self._create_completed_job(gold, t_many, 85.0, "pass")
        self._create_completed_job(gold, t_many, 80.0, "pass")
        # op-few gets 1 job
        self._create_completed_job(gold, t_few, 70.0, "needs_review")

        result = self.db.get_analytics()
        self.assertEqual(result["by_operator"][0]["operator_id"], "op-many")
        self.assertEqual(result["by_operator"][1]["operator_id"], "op-few")


class TestAnalyticsBySite(_AnalyticsTestBase):
    """Verify by_site aggregation groups by trainee video's site_id."""

    def test_single_site(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video(site_id="site-X")
        self._create_completed_job(gold, trainee, 92.0, "pass")

        result = self.db.get_analytics()
        self.assertEqual(len(result["by_site"]), 1)
        site = result["by_site"][0]
        self.assertEqual(site["site_id"], "site-X")
        self.assertEqual(site["job_count"], 1)
        self.assertEqual(site["pass_count"], 1)
        self.assertEqual(site["fail_count"], 0)

    def test_multiple_sites(self) -> None:
        gold = self._insert_gold_video()
        t_site1 = self._insert_trainee_video(site_id="warehouse-1")
        t_site2 = self._insert_trainee_video(site_id="warehouse-2")
        self._create_completed_job(gold, t_site1, 95.0, "pass")
        self._create_completed_job(gold, t_site1, 88.0, "pass")
        self._create_completed_job(gold, t_site2, 55.0, "fail")

        result = self.db.get_analytics()
        self.assertEqual(len(result["by_site"]), 2)
        site_map = {s["site_id"]: s for s in result["by_site"]}

        self.assertEqual(site_map["warehouse-1"]["job_count"], 2)
        self.assertEqual(site_map["warehouse-1"]["pass_count"], 2)
        self.assertEqual(site_map["warehouse-2"]["job_count"], 1)
        self.assertEqual(site_map["warehouse-2"]["fail_count"], 1)

    def test_null_site_becomes_unknown(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video(site_id=None)
        self._create_completed_job(gold, trainee, 80.0, "pass")

        result = self.db.get_analytics()
        self.assertEqual(len(result["by_site"]), 1)
        self.assertEqual(result["by_site"][0]["site_id"], "unknown")

    def test_sites_ordered_by_job_count_desc(self) -> None:
        gold = self._insert_gold_video()
        t_big = self._insert_trainee_video(site_id="big-site")
        t_small = self._insert_trainee_video(site_id="small-site")
        self._create_completed_job(gold, t_big, 90.0, "pass")
        self._create_completed_job(gold, t_big, 85.0, "pass")
        self._create_completed_job(gold, t_big, 80.0, "pass")
        self._create_completed_job(gold, t_small, 70.0, "needs_review")

        result = self.db.get_analytics()
        self.assertEqual(result["by_site"][0]["site_id"], "big-site")
        self.assertEqual(result["by_site"][1]["site_id"], "small-site")


class TestAnalyticsTaskIdFilter(_AnalyticsTestBase):
    """Verify task_id filtering limits results to a specific task."""

    def test_filter_by_task_id(self) -> None:
        gold_a = self._insert_gold_video(task_id="task-A")
        trainee_a = self._insert_trainee_video(task_id="task-A")
        gold_b = self._insert_gold_video(task_id="task-B")
        trainee_b = self._insert_trainee_video(task_id="task-B")

        self._create_completed_job(gold_a, trainee_a, 95.0, "pass")
        self._create_completed_job(gold_a, trainee_a, 85.0, "pass")
        self._create_completed_job(gold_b, trainee_b, 60.0, "fail")

        # Filter for task-A only
        result_a = self.db.get_analytics(task_id="task-A")
        self.assertEqual(result_a["total_jobs"], 2)
        self.assertEqual(result_a["completed_jobs"], 2)
        self.assertEqual(result_a["pass_count"], 2)
        self.assertEqual(result_a["fail_count"], 0)

        # Filter for task-B only
        result_b = self.db.get_analytics(task_id="task-B")
        self.assertEqual(result_b["total_jobs"], 1)
        self.assertEqual(result_b["completed_jobs"], 1)
        self.assertEqual(result_b["pass_count"], 0)
        self.assertEqual(result_b["fail_count"], 1)

    def test_filter_nonexistent_task_returns_zeros(self) -> None:
        gold = self._insert_gold_video(task_id="task-exists")
        trainee = self._insert_trainee_video(task_id="task-exists")
        self._create_completed_job(gold, trainee, 90.0, "pass")

        result = self.db.get_analytics(task_id="no-such-task")
        self.assertEqual(result["total_jobs"], 0)
        self.assertEqual(result["completed_jobs"], 0)
        self.assertIsNone(result["avg_score"])
        self.assertEqual(result["by_operator"], [])
        self.assertEqual(result["by_site"], [])
        self.assertEqual(result["recent_trend"], [])

    def test_no_filter_returns_all_tasks(self) -> None:
        gold_a = self._insert_gold_video(task_id="task-A")
        trainee_a = self._insert_trainee_video(task_id="task-A")
        gold_b = self._insert_gold_video(task_id="task-B")
        trainee_b = self._insert_trainee_video(task_id="task-B")
        self._create_completed_job(gold_a, trainee_a, 90.0, "pass")
        self._create_completed_job(gold_b, trainee_b, 80.0, "pass")

        result = self.db.get_analytics()
        self.assertEqual(result["total_jobs"], 2)
        self.assertEqual(result["completed_jobs"], 2)

    def test_task_filter_affects_score_distribution(self) -> None:
        gold_a = self._insert_gold_video(task_id="task-A")
        trainee_a = self._insert_trainee_video(task_id="task-A")
        gold_b = self._insert_gold_video(task_id="task-B")
        trainee_b = self._insert_trainee_video(task_id="task-B")
        # task-A: score 95 (90-100 bucket)
        self._create_completed_job(gold_a, trainee_a, 95.0, "pass")
        # task-B: score 50 (0-69 bucket)
        self._create_completed_job(gold_b, trainee_b, 50.0, "fail")

        result_a = self.db.get_analytics(task_id="task-A")
        dist_a = {d["bucket"]: d["count"] for d in result_a["score_distribution"]}
        self.assertEqual(dist_a["90-100"], 1)
        self.assertEqual(dist_a["0-69"], 0)

        result_b = self.db.get_analytics(task_id="task-B")
        dist_b = {d["bucket"]: d["count"] for d in result_b["score_distribution"]}
        self.assertEqual(dist_b["90-100"], 0)
        self.assertEqual(dist_b["0-69"], 1)

    def test_task_filter_affects_by_operator(self) -> None:
        gold_a = self._insert_gold_video(task_id="task-A")
        trainee_a = self._insert_trainee_video(task_id="task-A", operator_id_hash="op-X")
        gold_b = self._insert_gold_video(task_id="task-B")
        trainee_b = self._insert_trainee_video(task_id="task-B", operator_id_hash="op-Y")
        self._create_completed_job(gold_a, trainee_a, 90.0, "pass")
        self._create_completed_job(gold_b, trainee_b, 60.0, "fail")

        result_a = self.db.get_analytics(task_id="task-A")
        self.assertEqual(len(result_a["by_operator"]), 1)
        self.assertEqual(result_a["by_operator"][0]["operator_id"], "op-X")


class TestAnalyticsPartialData(_AnalyticsTestBase):
    """Verify handling when some jobs are completed and some are pending/failed."""

    def test_pending_jobs_counted_in_total_not_completed(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()
        self._create_completed_job(gold, trainee, 90.0, "pass")
        self._create_pending_job(gold, trainee)
        self._create_pending_job(gold, trainee)

        result = self.db.get_analytics()
        self.assertEqual(result["total_jobs"], 3)
        self.assertEqual(result["completed_jobs"], 1)
        self.assertEqual(result["pass_count"], 1)

    def test_failed_jobs_counted_in_total_not_completed(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()
        self._create_completed_job(gold, trainee, 85.0, "pass")
        self._create_failed_job(gold, trainee)

        result = self.db.get_analytics()
        self.assertEqual(result["total_jobs"], 2)
        self.assertEqual(result["completed_jobs"], 1)

    def test_pending_jobs_not_in_score_stats(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()
        self._create_completed_job(gold, trainee, 80.0, "pass")
        self._create_pending_job(gold, trainee)

        result = self.db.get_analytics()
        # avg/min/max should only consider the one completed job
        self.assertEqual(result["avg_score"], 80.0)
        self.assertEqual(result["min_score"], 80.0)
        self.assertEqual(result["max_score"], 80.0)

    def test_pending_jobs_not_in_score_distribution(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()
        self._create_completed_job(gold, trainee, 95.0, "pass")
        self._create_pending_job(gold, trainee)

        result = self.db.get_analytics()
        total_in_dist = sum(d["count"] for d in result["score_distribution"])
        self.assertEqual(total_in_dist, 1)

    def test_failed_jobs_not_in_by_operator(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video(operator_id_hash="op-1")
        self._create_completed_job(gold, trainee, 90.0, "pass")
        self._create_failed_job(gold, trainee)

        result = self.db.get_analytics()
        self.assertEqual(len(result["by_operator"]), 1)
        self.assertEqual(result["by_operator"][0]["job_count"], 1)

    def test_failed_jobs_not_in_by_site(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video(site_id="site-1")
        self._create_completed_job(gold, trainee, 90.0, "pass")
        self._create_failed_job(gold, trainee)

        result = self.db.get_analytics()
        self.assertEqual(len(result["by_site"]), 1)
        self.assertEqual(result["by_site"][0]["job_count"], 1)

    def test_only_pending_jobs_gives_zero_completed(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()
        self._create_pending_job(gold, trainee)
        self._create_pending_job(gold, trainee)

        result = self.db.get_analytics()
        self.assertEqual(result["total_jobs"], 2)
        self.assertEqual(result["completed_jobs"], 0)
        self.assertIsNone(result["avg_score"])
        self.assertEqual(result["by_operator"], [])
        self.assertEqual(result["by_site"], [])
        self.assertEqual(result["recent_trend"], [])


class TestAnalyticsRecentTrend(_AnalyticsTestBase):
    """Verify recent_trend returns at most 30 completed jobs."""

    def test_recent_trend_returns_completed_jobs(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()
        job_id = self._create_completed_job(gold, trainee, 88.0, "pass")

        result = self.db.get_analytics()
        self.assertEqual(len(result["recent_trend"]), 1)
        trend = result["recent_trend"][0]
        self.assertEqual(trend["job_id"], job_id)
        self.assertEqual(trend["score"], 88.0)
        self.assertEqual(trend["decision"], "pass")
        self.assertIn("created_at", trend)

    def test_recent_trend_excludes_pending_jobs(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()
        self._create_completed_job(gold, trainee, 90.0, "pass")
        self._create_pending_job(gold, trainee)

        result = self.db.get_analytics()
        self.assertEqual(len(result["recent_trend"]), 1)

    def test_recent_trend_capped_at_30(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()
        for i in range(35):
            self._create_completed_job(gold, trainee, 80.0 + (i % 10), "pass")

        result = self.db.get_analytics()
        self.assertEqual(len(result["recent_trend"]), 30)

    def test_recent_trend_ordered_by_id_desc(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()
        id1 = self._create_completed_job(gold, trainee, 70.0, "needs_review")
        id2 = self._create_completed_job(gold, trainee, 80.0, "pass")
        id3 = self._create_completed_job(gold, trainee, 90.0, "pass")

        result = self.db.get_analytics()
        trend_ids = [t["job_id"] for t in result["recent_trend"]]
        self.assertEqual(trend_ids, [id3, id2, id1])


class TestAnalyticsResponseStructure(_AnalyticsTestBase):
    """Verify the overall response structure and key presence."""

    def test_all_top_level_keys_present(self) -> None:
        result = self.db.get_analytics()
        expected_keys = {
            "total_jobs",
            "completed_jobs",
            "pass_count",
            "fail_count",
            "needs_review_count",
            "retrain_count",
            "avg_score",
            "min_score",
            "max_score",
            "by_operator",
            "by_site",
            "score_distribution",
            "recent_trend",
            "reviewer_agreement",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_by_operator_entry_structure(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video(operator_id_hash="op-1")
        self._create_completed_job(gold, trainee, 85.0, "pass")

        result = self.db.get_analytics()
        op = result["by_operator"][0]
        self.assertIn("operator_id", op)
        self.assertIn("job_count", op)
        self.assertIn("avg_score", op)
        self.assertIn("pass_count", op)
        self.assertIn("fail_count", op)

    def test_by_site_entry_structure(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video(site_id="site-Z")
        self._create_completed_job(gold, trainee, 85.0, "pass")

        result = self.db.get_analytics()
        site = result["by_site"][0]
        self.assertIn("site_id", site)
        self.assertIn("job_count", site)
        self.assertIn("avg_score", site)
        self.assertIn("pass_count", site)
        self.assertIn("fail_count", site)

    def test_score_distribution_entry_structure(self) -> None:
        result = self.db.get_analytics()
        for entry in result["score_distribution"]:
            self.assertIn("bucket", entry)
            self.assertIn("count", entry)
            self.assertIsInstance(entry["count"], int)

    def test_recent_trend_entry_structure(self) -> None:
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()
        self._create_completed_job(gold, trainee, 90.0, "pass")

        result = self.db.get_analytics()
        entry = result["recent_trend"][0]
        self.assertIn("job_id", entry)
        self.assertIn("score", entry)
        self.assertIn("decision", entry)
        self.assertIn("created_at", entry)


class TestAnalyticsReviewerAgreementEmpty(_AnalyticsTestBase):
    """Verify reviewer_agreement on an empty database returns sensible defaults."""

    def test_empty_database_reviewer_agreement_structure(self) -> None:
        result = self.db.get_analytics()
        self.assertIn("reviewer_agreement", result)
        ra = result["reviewer_agreement"]
        self.assertEqual(ra["reviewed_count"], 0)
        self.assertEqual(ra["agree_count"], 0)
        self.assertIsNone(ra["agreement_rate"])
        self.assertIn("by_verdict", ra)

    def test_empty_database_reviewer_agreement_by_verdict_all_zero(self) -> None:
        result = self.db.get_analytics()
        by_verdict = result["reviewer_agreement"]["by_verdict"]
        self.assertEqual(by_verdict["pass"], 0)
        self.assertEqual(by_verdict["fail"], 0)
        self.assertEqual(by_verdict["retrain"], 0)
        self.assertEqual(by_verdict["needs_review"], 0)


class TestAnalyticsReviewerAgreementCounts(_AnalyticsTestBase):
    """Verify reviewer_agreement counts and agreement_rate calculations."""

    def test_single_agreeing_review(self) -> None:
        """One completed job with decision=pass, reviewer verdict=pass => agree."""
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()
        job_id = self._create_completed_job(gold, trainee, 95.0, "pass")
        self.db.upsert_score_review(job_id, "pass", "Looks good")

        result = self.db.get_analytics()
        ra = result["reviewer_agreement"]
        self.assertEqual(ra["reviewed_count"], 1)
        self.assertEqual(ra["agree_count"], 1)
        self.assertEqual(ra["agreement_rate"], 1.0)

    def test_single_disagreeing_review(self) -> None:
        """One completed job with decision=pass, reviewer verdict=fail => disagree."""
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()
        job_id = self._create_completed_job(gold, trainee, 95.0, "pass")
        self.db.upsert_score_review(job_id, "fail", "Missed a step")

        result = self.db.get_analytics()
        ra = result["reviewer_agreement"]
        self.assertEqual(ra["reviewed_count"], 1)
        self.assertEqual(ra["agree_count"], 0)
        self.assertEqual(ra["agreement_rate"], 0.0)

    def test_mixed_agreement(self) -> None:
        """Multiple reviews with mixed agreement."""
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()

        # Job 1: decision=pass, review=pass => agree
        j1 = self._create_completed_job(gold, trainee, 95.0, "pass")
        self.db.upsert_score_review(j1, "pass", None)

        # Job 2: decision=fail, review=fail => agree
        j2 = self._create_completed_job(gold, trainee, 50.0, "fail")
        self.db.upsert_score_review(j2, "fail", None)

        # Job 3: decision=pass, review=needs_review => disagree
        j3 = self._create_completed_job(gold, trainee, 85.0, "pass")
        self.db.upsert_score_review(j3, "needs_review", None)

        # Job 4: decision=retrain, review=retrain => agree
        j4 = self._create_completed_job(gold, trainee, 40.0, "retrain")
        self.db.upsert_score_review(j4, "retrain", None)

        result = self.db.get_analytics()
        ra = result["reviewer_agreement"]
        self.assertEqual(ra["reviewed_count"], 4)
        self.assertEqual(ra["agree_count"], 3)
        self.assertEqual(ra["agreement_rate"], 0.75)

    def test_by_verdict_breakdown(self) -> None:
        """Verify by_verdict counts match the reviewer verdicts submitted."""
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()

        j1 = self._create_completed_job(gold, trainee, 95.0, "pass")
        self.db.upsert_score_review(j1, "pass", None)

        j2 = self._create_completed_job(gold, trainee, 50.0, "fail")
        self.db.upsert_score_review(j2, "fail", None)

        j3 = self._create_completed_job(gold, trainee, 85.0, "pass")
        self.db.upsert_score_review(j3, "fail", None)

        j4 = self._create_completed_job(gold, trainee, 40.0, "retrain")
        self.db.upsert_score_review(j4, "retrain", None)

        j5 = self._create_completed_job(gold, trainee, 75.0, "needs_review")
        self.db.upsert_score_review(j5, "needs_review", None)

        result = self.db.get_analytics()
        by_verdict = result["reviewer_agreement"]["by_verdict"]
        self.assertEqual(by_verdict["pass"], 1)
        self.assertEqual(by_verdict["fail"], 2)
        self.assertEqual(by_verdict["retrain"], 1)
        self.assertEqual(by_verdict["needs_review"], 1)

    def test_unreviewed_jobs_not_counted(self) -> None:
        """Completed jobs without reviews should not be counted in reviewer_agreement."""
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()

        # Create 3 completed jobs but only review 1
        self._create_completed_job(gold, trainee, 90.0, "pass")
        self._create_completed_job(gold, trainee, 80.0, "pass")
        j3 = self._create_completed_job(gold, trainee, 70.0, "needs_review")
        self.db.upsert_score_review(j3, "needs_review", None)

        result = self.db.get_analytics()
        ra = result["reviewer_agreement"]
        self.assertEqual(ra["reviewed_count"], 1)
        self.assertEqual(ra["agree_count"], 1)

    def test_pending_jobs_reviews_not_counted(self) -> None:
        """Reviews on non-completed jobs should not be counted (if any exist)."""
        gold = self._insert_gold_video()
        trainee = self._insert_trainee_video()

        # Create a completed job with review
        j1 = self._create_completed_job(gold, trainee, 90.0, "pass")
        self.db.upsert_score_review(j1, "pass", None)

        # Create a pending job (no score_json)
        self._create_pending_job(gold, trainee)

        result = self.db.get_analytics()
        ra = result["reviewer_agreement"]
        self.assertEqual(ra["reviewed_count"], 1)


class TestAnalyticsReviewerAgreementTaskFilter(_AnalyticsTestBase):
    """Verify reviewer_agreement respects task_id filter."""

    def test_task_filter_limits_reviewer_agreement(self) -> None:
        """Reviews from different tasks should be separated by task_id filter."""
        gold_a = self._insert_gold_video(task_id="task-A")
        trainee_a = self._insert_trainee_video(task_id="task-A")
        gold_b = self._insert_gold_video(task_id="task-B")
        trainee_b = self._insert_trainee_video(task_id="task-B")

        # Task A: review agrees
        j1 = self._create_completed_job(gold_a, trainee_a, 90.0, "pass")
        self.db.upsert_score_review(j1, "pass", None)

        # Task B: review disagrees
        j2 = self._create_completed_job(gold_b, trainee_b, 60.0, "fail")
        self.db.upsert_score_review(j2, "pass", "Override")

        # Filter for task-A: 1 reviewed, 1 agree
        result_a = self.db.get_analytics(task_id="task-A")
        ra_a = result_a["reviewer_agreement"]
        self.assertEqual(ra_a["reviewed_count"], 1)
        self.assertEqual(ra_a["agree_count"], 1)
        self.assertEqual(ra_a["agreement_rate"], 1.0)

        # Filter for task-B: 1 reviewed, 0 agree
        result_b = self.db.get_analytics(task_id="task-B")
        ra_b = result_b["reviewer_agreement"]
        self.assertEqual(ra_b["reviewed_count"], 1)
        self.assertEqual(ra_b["agree_count"], 0)
        self.assertEqual(ra_b["agreement_rate"], 0.0)

        # No filter: 2 reviewed, 1 agree
        result_all = self.db.get_analytics()
        ra_all = result_all["reviewer_agreement"]
        self.assertEqual(ra_all["reviewed_count"], 2)
        self.assertEqual(ra_all["agree_count"], 1)
        self.assertEqual(ra_all["agreement_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
