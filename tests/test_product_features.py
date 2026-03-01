"""Tests for v0.8 product features.

Covers:
- make_decision() new return fields: decision_basis, score_band
- _score_band() helper
- enrich_score_result() back-fill for old stored results
- POST /admin/rescore endpoint (dry_run, live, idempotent, task_id filter)
- ScoringService.rescore_decisions (via the API layer)
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from sopilot.core.score_pipeline import _score_band, make_decision
from sopilot.core.score_result import enrich_score_result
from sopilot.database import Database


# ---------------------------------------------------------------------------
# Shared DB helpers (mirrors pattern from test_analytics.py)
# ---------------------------------------------------------------------------

def _make_db(tmp_path: Path) -> Database:
    return Database(tmp_path / "test.db")


def _insert_gold(db: Database, task_id: str = "task-A") -> int:
    return db.insert_video(
        task_id=task_id,
        site_id="site-1",
        camera_id=None,
        operator_id_hash=None,
        recorded_at=None,
        is_gold=True,
    )


def _insert_trainee(db: Database, task_id: str = "task-A") -> int:
    return db.insert_video(
        task_id=task_id,
        site_id="site-1",
        camera_id=None,
        operator_id_hash="op-1",
        recorded_at=None,
        is_gold=False,
    )


def _create_completed_job(
    db: Database,
    gold_id: int,
    trainee_id: int,
    score: float,
    decision: str,
    deviations: list | None = None,
    severity_counts: dict | None = None,
    pass_score: float = 60.0,
    retrain_score: float = 50.0,
) -> int:
    """Create and complete a score job with given score/decision in the DB."""
    job_id = db.create_score_job(gold_id, trainee_id)
    db.claim_score_job(job_id)
    payload = {
        "score": score,
        "deviations": deviations or [],
        "summary": {
            "decision": decision,
            "severity_counts": severity_counts or {"critical": 0, "quality": 0, "efficiency": 0},
            "pass_score": pass_score,
            "retrain_score": retrain_score,
        },
    }
    db.complete_score_job(job_id, payload)
    return job_id


# ===========================================================================
# 1. TestMakeDecision — unit tests for make_decision() and _score_band()
# ===========================================================================

class TestMakeDecision(unittest.TestCase):
    """Unit tests for make_decision() and _score_band()."""

    # ------------------------------------------------------------------
    # decision_basis tests
    # ------------------------------------------------------------------

    def test_decision_basis_critical_deviation(self) -> None:
        """Any critical severity deviation must yield decision_basis='critical_deviation'."""
        deviations = [{"type": "missing_step", "severity": "critical"}]
        result = make_decision(
            score=80.0,
            deviations=deviations,
            pass_score=60.0,
            retrain_score=50.0,
        )
        self.assertEqual(result["decision_basis"], "critical_deviation")
        self.assertEqual(result["decision"], "fail")

    def test_decision_basis_score_above_threshold(self) -> None:
        """No critical deviations, score >= pass_score → 'score_above_threshold'."""
        result = make_decision(
            score=80.0,
            deviations=[],
            pass_score=60.0,
            retrain_score=50.0,
        )
        self.assertEqual(result["decision_basis"], "score_above_threshold")
        self.assertEqual(result["decision"], "pass")

    def test_decision_basis_score_below_retrain(self) -> None:
        """Score below retrain_score → 'score_below_retrain' and decision='retrain'."""
        result = make_decision(
            score=30.0,
            deviations=[],
            pass_score=60.0,
            retrain_score=50.0,
        )
        self.assertEqual(result["decision_basis"], "score_below_retrain")
        self.assertEqual(result["decision"], "retrain")

    def test_decision_basis_between_thresholds(self) -> None:
        """Score between retrain_score and pass_score → 'score_between_thresholds'."""
        result = make_decision(
            score=55.0,
            deviations=[],
            pass_score=60.0,
            retrain_score=50.0,
        )
        self.assertEqual(result["decision_basis"], "score_between_thresholds")
        self.assertEqual(result["decision"], "needs_review")

    def test_decision_basis_score_exactly_at_pass_threshold(self) -> None:
        """Score exactly equal to pass_score → 'score_above_threshold'."""
        result = make_decision(
            score=60.0,
            deviations=[],
            pass_score=60.0,
            retrain_score=50.0,
        )
        self.assertEqual(result["decision_basis"], "score_above_threshold")
        self.assertEqual(result["decision"], "pass")

    def test_decision_basis_score_exactly_at_retrain_threshold(self) -> None:
        """Score exactly equal to retrain_score → 'score_between_thresholds' (not below)."""
        result = make_decision(
            score=50.0,
            deviations=[],
            pass_score=60.0,
            retrain_score=50.0,
        )
        self.assertEqual(result["decision_basis"], "score_between_thresholds")
        self.assertEqual(result["decision"], "needs_review")

    # ------------------------------------------------------------------
    # score_band tests (via make_decision return value)
    # ------------------------------------------------------------------

    def test_score_band_excellent(self) -> None:
        """score >= pass_score + pass_score * 0.2 → 'excellent'.

        With pass_score=60: threshold = 60 + 12 = 72. score=80 qualifies.
        """
        result = make_decision(
            score=80.0,
            deviations=[],
            pass_score=60.0,
            retrain_score=50.0,
        )
        self.assertEqual(result["score_band"], "excellent")

    def test_score_band_passing(self) -> None:
        """pass_score <= score < pass_score * 1.2 → 'passing'.

        With pass_score=60: excellent threshold = 72. score=65 is between 60 and 72.
        """
        result = make_decision(
            score=65.0,
            deviations=[],
            pass_score=60.0,
            retrain_score=50.0,
        )
        self.assertEqual(result["score_band"], "passing")

    def test_score_band_needs_review(self) -> None:
        """retrain_score <= score < pass_score → 'needs_review'."""
        result = make_decision(
            score=55.0,
            deviations=[],
            pass_score=60.0,
            retrain_score=50.0,
        )
        self.assertEqual(result["score_band"], "needs_review")

    def test_score_band_poor(self) -> None:
        """score < retrain_score → 'poor'."""
        result = make_decision(
            score=40.0,
            deviations=[],
            pass_score=60.0,
            retrain_score=50.0,
        )
        self.assertEqual(result["score_band"], "poor")

    # ------------------------------------------------------------------
    # _score_band helper (direct)
    # ------------------------------------------------------------------

    def test_score_band_helper_excellent_boundary(self) -> None:
        """_score_band at exactly the excellent boundary."""
        band = _score_band(72.0, pass_score=60.0, retrain_score=50.0)
        self.assertEqual(band, "excellent")

    def test_score_band_helper_just_below_excellent(self) -> None:
        """_score_band just below excellent boundary is 'passing'."""
        band = _score_band(71.9, pass_score=60.0, retrain_score=50.0)
        self.assertEqual(band, "passing")

    def test_score_band_helper_zero_score(self) -> None:
        """_score_band with score=0 is 'poor'."""
        band = _score_band(0.0, pass_score=60.0, retrain_score=50.0)
        self.assertEqual(band, "poor")

    def test_score_band_helper_perfect_score(self) -> None:
        """_score_band with score=100 is 'excellent'."""
        band = _score_band(100.0, pass_score=60.0, retrain_score=50.0)
        self.assertEqual(band, "excellent")

    # ------------------------------------------------------------------
    # Critical deviation forces fail even at high score
    # ------------------------------------------------------------------

    def test_critical_deviation_forces_fail_regardless_of_score(self) -> None:
        """A critical deviation must override a high score and produce decision='fail'
        with decision_basis='critical_deviation'."""
        deviations = [{"type": "missing_step", "severity": "critical"}]
        result = make_decision(
            score=90.0,
            deviations=deviations,
            pass_score=60.0,
            retrain_score=50.0,
        )
        self.assertEqual(result["decision"], "fail")
        self.assertEqual(result["decision_basis"], "critical_deviation")

    def test_non_critical_deviation_does_not_override_pass(self) -> None:
        """Quality-only deviations must not prevent a pass decision."""
        deviations = [{"type": "step_deviation", "severity": "quality"}]
        result = make_decision(
            score=80.0,
            deviations=deviations,
            pass_score=60.0,
            retrain_score=50.0,
        )
        self.assertEqual(result["decision"], "pass")
        self.assertEqual(result["decision_basis"], "score_above_threshold")

    def test_return_contains_required_keys(self) -> None:
        """make_decision() must always return all expected keys."""
        result = make_decision(
            score=70.0,
            deviations=[],
            pass_score=60.0,
            retrain_score=50.0,
        )
        for key in ("decision", "decision_reason", "decision_basis", "score_band",
                    "severity_counts", "pass_score", "retrain_score"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_severity_counts_accumulated_correctly(self) -> None:
        """severity_counts must reflect all deviation severity types."""
        deviations = [
            {"type": "a", "severity": "quality"},
            {"type": "b", "severity": "quality"},
            {"type": "c", "severity": "efficiency"},
        ]
        result = make_decision(
            score=80.0,
            deviations=deviations,
            pass_score=60.0,
            retrain_score=50.0,
        )
        self.assertEqual(result["severity_counts"]["quality"], 2)
        self.assertEqual(result["severity_counts"]["efficiency"], 1)
        self.assertEqual(result["severity_counts"]["critical"], 0)


# ===========================================================================
# 2. TestEnrichScoreResult — unit tests for enrich_score_result()
# ===========================================================================

class TestEnrichScoreResult(unittest.TestCase):
    """Unit tests for enrich_score_result() back-fill function."""

    def _make_old_result(
        self,
        score: float = 80.0,
        decision: str = "pass",
        severity_counts: dict | None = None,
        pass_score: float = 60.0,
        retrain_score: float = 50.0,
    ) -> dict:
        """Build a score result dict that lacks decision_basis and score_band
        (simulating data stored before v0.8)."""
        return {
            "score": score,
            "summary": {
                "decision": decision,
                "severity_counts": severity_counts or {"critical": 0, "quality": 0, "efficiency": 0},
                "pass_score": pass_score,
                "retrain_score": retrain_score,
            },
        }

    def test_enrich_adds_decision_basis_to_old_result(self) -> None:
        """enrich_score_result() must add decision_basis when absent."""
        result = self._make_old_result(score=80.0, decision="pass")
        enrich_score_result(result)
        self.assertIn("decision_basis", result["summary"])

    def test_enrich_adds_score_band_to_old_result(self) -> None:
        """enrich_score_result() must add score_band when absent."""
        result = self._make_old_result(score=80.0, decision="pass")
        enrich_score_result(result)
        self.assertIn("score_band", result["summary"])

    def test_enrich_pass_decision_maps_to_score_above_threshold(self) -> None:
        """Old result with decision='pass' → decision_basis='score_above_threshold'."""
        result = self._make_old_result(score=80.0, decision="pass")
        enrich_score_result(result)
        self.assertEqual(result["summary"]["decision_basis"], "score_above_threshold")

    def test_enrich_retrain_decision_maps_to_score_below_retrain(self) -> None:
        """Old result with decision='retrain' → decision_basis='score_below_retrain'."""
        result = self._make_old_result(score=30.0, decision="retrain")
        enrich_score_result(result)
        self.assertEqual(result["summary"]["decision_basis"], "score_below_retrain")

    def test_enrich_needs_review_decision_maps_to_between_thresholds(self) -> None:
        """Old result with decision='needs_review' → decision_basis='score_between_thresholds'."""
        result = self._make_old_result(score=55.0, decision="needs_review")
        enrich_score_result(result)
        self.assertEqual(result["summary"]["decision_basis"], "score_between_thresholds")

    def test_enrich_score_band_excellent(self) -> None:
        """Enriched score_band for a high-score old result must be 'excellent'."""
        result = self._make_old_result(score=80.0, decision="pass", pass_score=60.0, retrain_score=50.0)
        enrich_score_result(result)
        self.assertEqual(result["summary"]["score_band"], "excellent")

    def test_enrich_score_band_passing(self) -> None:
        """Enriched score_band for a just-passing old result must be 'passing'."""
        result = self._make_old_result(score=65.0, decision="pass", pass_score=60.0, retrain_score=50.0)
        enrich_score_result(result)
        self.assertEqual(result["summary"]["score_band"], "passing")

    def test_enrich_score_band_poor(self) -> None:
        """Enriched score_band for a low score must be 'poor'."""
        result = self._make_old_result(score=30.0, decision="retrain", pass_score=60.0, retrain_score=50.0)
        enrich_score_result(result)
        self.assertEqual(result["summary"]["score_band"], "poor")

    def test_enrich_does_not_overwrite_existing_decision_basis(self) -> None:
        """If decision_basis already exists, enrich_score_result must not overwrite it."""
        result = self._make_old_result(score=80.0, decision="pass")
        result["summary"]["decision_basis"] = "score_above_threshold"
        result["summary"]["score_band"] = "excellent"
        enrich_score_result(result)
        # Should remain unchanged
        self.assertEqual(result["summary"]["decision_basis"], "score_above_threshold")
        self.assertEqual(result["summary"]["score_band"], "excellent")

    def test_enrich_does_not_overwrite_existing_custom_values(self) -> None:
        """Pre-existing non-standard values in decision_basis must not be overwritten."""
        result = self._make_old_result(score=80.0, decision="pass")
        result["summary"]["decision_basis"] = "custom_value"
        result["summary"]["score_band"] = "custom_band"
        enrich_score_result(result)
        self.assertEqual(result["summary"]["decision_basis"], "custom_value")
        self.assertEqual(result["summary"]["score_band"], "custom_band")

    def test_enrich_handles_missing_summary(self) -> None:
        """enrich_score_result() must not crash when summary is absent."""
        result: dict = {"score": 80.0}
        # Should return without raising
        returned = enrich_score_result(result)
        self.assertIs(returned, result)
        # No summary added
        self.assertNotIn("summary", result)

    def test_enrich_handles_none_summary(self) -> None:
        """enrich_score_result() must not crash when summary is None."""
        result: dict = {"score": 80.0, "summary": None}
        returned = enrich_score_result(result)
        self.assertIs(returned, result)

    def test_enrich_returns_same_dict(self) -> None:
        """enrich_score_result() must return the same dict object (in-place mutation)."""
        result = self._make_old_result()
        returned = enrich_score_result(result)
        self.assertIs(returned, result)

    def test_enrich_critical_deviation_in_old_result(self) -> None:
        """Old result with decision='fail' and critical severity_count > 0
        must back-fill decision_basis='critical_deviation'."""
        result = self._make_old_result(
            score=80.0,
            decision="fail",
            severity_counts={"critical": 1, "quality": 0, "efficiency": 0},
        )
        enrich_score_result(result)
        self.assertEqual(result["summary"]["decision_basis"], "critical_deviation")

    def test_enrich_fail_without_critical_severity_yields_unknown(self) -> None:
        """Old result with decision='fail' but no critical counts → decision_basis='unknown'."""
        result = self._make_old_result(
            score=80.0,
            decision="fail",
            severity_counts={"critical": 0, "quality": 1, "efficiency": 0},
        )
        enrich_score_result(result)
        self.assertEqual(result["summary"]["decision_basis"], "unknown")

    def test_enrich_uses_defaults_when_pass_score_missing(self) -> None:
        """When pass_score/retrain_score are absent from summary, defaults (60/50) are used."""
        result: dict = {
            "score": 80.0,
            "summary": {
                "decision": "pass",
                "severity_counts": {"critical": 0, "quality": 0, "efficiency": 0},
                # No pass_score or retrain_score
            },
        }
        enrich_score_result(result)
        # 80 >= 60 + 60*0.2 = 72 → excellent
        self.assertEqual(result["summary"]["score_band"], "excellent")


# ===========================================================================
# 3. TestAdminRescore — API integration tests for POST /admin/rescore
# ===========================================================================

def _make_env(root: Path, task_id: str = "task-rescore") -> dict:
    """Return a dict of env var changes needed for a test app instance."""
    return {
        "SOPILOT_DATA_DIR": str(root / "data"),
        "SOPILOT_EMBEDDER_BACKEND": "color-motion",
        "SOPILOT_PRIMARY_TASK_ID": task_id,
        "SOPILOT_RATE_LIMIT_RPM": "0",
    }


class TestAdminRescore(unittest.TestCase):
    """Integration tests for POST /admin/rescore using synchronous TestClient."""

    def _setup_env(self, root: Path, task_id: str = "task-rescore") -> dict:
        """Set env vars and return original values so teardown can restore them."""
        originals: dict = {}
        new_vals = _make_env(root, task_id)
        for key, val in new_vals.items():
            originals[key] = os.environ.get(key)
            os.environ[key] = val
        return originals

    def _restore_env(self, originals: dict) -> None:
        for key, val in originals.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _seed_db_with_old_pass_job(self, db: Database, gold_id: int, trainee_id: int) -> int:
        """Insert a completed job whose stored decision was computed with old thresholds.

        We deliberately store decision='needs_review' for a score=75.0 job.
        After rescore with pass_score=60, the new decision should be 'pass'.
        This simulates a job stored before a threshold change.
        """
        job_id = db.create_score_job(gold_id, trainee_id)
        db.claim_score_job(job_id)
        # Stored with wrong (old) decision to make rescore detect a change
        payload = {
            "score": 75.0,
            "deviations": [],
            "summary": {
                "decision": "needs_review",  # wrong — should be 'pass' at pass_score=60
                "severity_counts": {"critical": 0, "quality": 0, "efficiency": 0},
                "pass_score": 60.0,
                "retrain_score": 50.0,
            },
        }
        db.complete_score_job(job_id, payload)
        return job_id

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_rescore_dry_run_returns_expected_shape(self) -> None:
        """POST /admin/rescore?dry_run=true must return the correct response shape."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            originals = self._setup_env(root)
            try:
                from fastapi.testclient import TestClient
                from sopilot.main import create_app
                app = create_app()
                with TestClient(app) as client:
                    resp = client.post("/admin/rescore?dry_run=true")
                self.assertEqual(resp.status_code, 200, resp.text)
                body = resp.json()
                self.assertIn("total_jobs_processed", body)
                self.assertIn("decisions_changed", body)
                self.assertIn("breakdown", body)
                self.assertIn("dry_run", body)
                self.assertTrue(body["dry_run"])
            finally:
                self._restore_env(originals)

    def test_rescore_dry_run_empty_db_returns_zeros(self) -> None:
        """On an empty database, dry_run rescore reports zero jobs processed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            originals = self._setup_env(root)
            try:
                from fastapi.testclient import TestClient
                from sopilot.main import create_app
                app = create_app()
                with TestClient(app) as client:
                    resp = client.post("/admin/rescore?dry_run=true")
                self.assertEqual(resp.status_code, 200, resp.text)
                body = resp.json()
                self.assertEqual(body["total_jobs_processed"], 0)
                self.assertEqual(body["decisions_changed"], 0)
                self.assertTrue(body["dry_run"])
            finally:
                self._restore_env(originals)

    def test_rescore_dry_run_detects_changes_without_writing(self) -> None:
        """dry_run=true must report decisions_changed > 0 without persisting changes to the DB."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            originals = self._setup_env(root, task_id="task-rescore")
            try:
                from fastapi.testclient import TestClient
                from sopilot.database import Database
                from sopilot.main import create_app

                # Pre-populate the database with a stale decision before the app starts
                (root / "data").mkdir(parents=True, exist_ok=True)
                db_path = root / "data" / "sopilot.db"
                db = Database(db_path)
                gold_id = _insert_gold(db, task_id="task-rescore")
                trainee_id = _insert_trainee(db, task_id="task-rescore")
                # Job stored with wrong decision (needs_review for score=75 at pass=60)
                self._seed_db_with_old_pass_job(db, gold_id, trainee_id)
                db.close()

                app = create_app()
                with TestClient(app) as client:
                    dry_resp = client.post("/admin/rescore?dry_run=true")
                self.assertEqual(dry_resp.status_code, 200, dry_resp.text)
                dry_body = dry_resp.json()
                self.assertTrue(dry_body["dry_run"])
                self.assertGreaterEqual(dry_body["total_jobs_processed"], 1)
                self.assertGreaterEqual(dry_body["decisions_changed"], 1)

                # Verify DB was NOT modified: re-read and check original wrong decision
                db2 = Database(db_path)
                jobs = db2.list_completed_score_jobs()
                db2.close()
                self.assertEqual(len(jobs), 1)
                stored_decision = jobs[0]["score"]["summary"]["decision"]
                self.assertEqual(stored_decision, "needs_review",
                                 "dry_run must not write changes to DB")
            finally:
                self._restore_env(originals)

    def test_rescore_applies_changes_when_not_dry_run(self) -> None:
        """POST /admin/rescore?dry_run=false must persist corrected decisions to the DB."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            originals = self._setup_env(root, task_id="task-rescore")
            try:
                from fastapi.testclient import TestClient
                from sopilot.database import Database
                from sopilot.main import create_app

                (root / "data").mkdir(parents=True, exist_ok=True)
                db_path = root / "data" / "sopilot.db"
                db = Database(db_path)
                gold_id = _insert_gold(db, task_id="task-rescore")
                trainee_id = _insert_trainee(db, task_id="task-rescore")
                self._seed_db_with_old_pass_job(db, gold_id, trainee_id)
                db.close()

                app = create_app()
                with TestClient(app) as client:
                    live_resp = client.post("/admin/rescore?dry_run=false")
                self.assertEqual(live_resp.status_code, 200, live_resp.text)
                body = live_resp.json()
                self.assertFalse(body["dry_run"])
                self.assertGreaterEqual(body["decisions_changed"], 1)

                # Verify DB WAS modified: the decision should now be 'pass'
                db2 = Database(db_path)
                jobs = db2.list_completed_score_jobs()
                db2.close()
                self.assertEqual(len(jobs), 1)
                stored_decision = jobs[0]["score"]["summary"]["decision"]
                self.assertEqual(stored_decision, "pass",
                                 "live rescore must update the stored decision to 'pass'")
            finally:
                self._restore_env(originals)

    def test_rescore_idempotent_second_run_changes_nothing(self) -> None:
        """Running rescore twice: first run applies changes, second run finds 0 changes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            originals = self._setup_env(root, task_id="task-rescore")
            try:
                from fastapi.testclient import TestClient
                from sopilot.database import Database
                from sopilot.main import create_app

                (root / "data").mkdir(parents=True, exist_ok=True)
                db_path = root / "data" / "sopilot.db"
                db = Database(db_path)
                gold_id = _insert_gold(db, task_id="task-rescore")
                trainee_id = _insert_trainee(db, task_id="task-rescore")
                self._seed_db_with_old_pass_job(db, gold_id, trainee_id)
                db.close()

                app = create_app()
                with TestClient(app) as client:
                    # First run: apply changes
                    first_resp = client.post("/admin/rescore?dry_run=false")
                    self.assertEqual(first_resp.status_code, 200)
                    first_body = first_resp.json()
                    self.assertGreaterEqual(first_body["decisions_changed"], 1)

                    # Second run: decisions are already correct, nothing to change
                    second_resp = client.post("/admin/rescore?dry_run=false")
                    self.assertEqual(second_resp.status_code, 200)
                    second_body = second_resp.json()
                    self.assertEqual(second_body["decisions_changed"], 0,
                                     "second rescore must find 0 decisions to change")
            finally:
                self._restore_env(originals)

    def test_rescore_with_task_id_filter_restricts_scope(self) -> None:
        """task_id= query parameter must limit rescore to only that task's jobs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            # Use enforce_primary_task=False so both tasks can coexist
            originals = self._setup_env(root, task_id="task-A")
            # Disable task enforcement so we can insert videos from two tasks
            originals["SOPILOT_ENFORCE_PRIMARY_TASK"] = os.environ.get("SOPILOT_ENFORCE_PRIMARY_TASK")
            os.environ["SOPILOT_ENFORCE_PRIMARY_TASK"] = "false"
            try:
                from fastapi.testclient import TestClient
                from sopilot.database import Database
                from sopilot.main import create_app

                (root / "data").mkdir(parents=True, exist_ok=True)
                db_path = root / "data" / "sopilot.db"
                db = Database(db_path)

                # Task A: stale decision (should be corrected by rescore)
                gold_a = _insert_gold(db, task_id="task-A")
                trainee_a = _insert_trainee(db, task_id="task-A")
                self._seed_db_with_old_pass_job(db, gold_a, trainee_a)

                # Task B: also stale decision (should NOT be corrected when filtering by task-A)
                gold_b = _insert_gold(db, task_id="task-B")
                trainee_b = _insert_trainee(db, task_id="task-B")
                self._seed_db_with_old_pass_job(db, gold_b, trainee_b)
                db.close()

                app = create_app()
                with TestClient(app) as client:
                    # Rescore only task-A
                    resp = client.post("/admin/rescore?task_id=task-A&dry_run=false")
                self.assertEqual(resp.status_code, 200, resp.text)
                body = resp.json()
                self.assertEqual(body["total_jobs_processed"], 1,
                                 "task_id filter must restrict to 1 job")
                self.assertGreaterEqual(body["decisions_changed"], 1)

                # Task B job must remain stale (not updated)
                db2 = Database(db_path)
                all_jobs = db2.list_completed_score_jobs()
                db2.close()
                task_b_jobs = [j for j in all_jobs if j.get("task_id") == "task-B"]
                self.assertEqual(len(task_b_jobs), 1)
                self.assertEqual(
                    task_b_jobs[0]["score"]["summary"]["decision"],
                    "needs_review",
                    "task-B job must NOT be updated when filtering by task-A",
                )
            finally:
                self._restore_env(originals)

    def test_rescore_breakdown_records_transition(self) -> None:
        """After a live rescore, the breakdown dict must record the old→new transition."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            originals = self._setup_env(root, task_id="task-rescore")
            try:
                from fastapi.testclient import TestClient
                from sopilot.database import Database
                from sopilot.main import create_app

                (root / "data").mkdir(parents=True, exist_ok=True)
                db_path = root / "data" / "sopilot.db"
                db = Database(db_path)
                gold_id = _insert_gold(db, task_id="task-rescore")
                trainee_id = _insert_trainee(db, task_id="task-rescore")
                self._seed_db_with_old_pass_job(db, gold_id, trainee_id)
                db.close()

                app = create_app()
                with TestClient(app) as client:
                    resp = client.post("/admin/rescore?dry_run=false")
                self.assertEqual(resp.status_code, 200, resp.text)
                body = resp.json()
                breakdown = body["breakdown"]
                # We seeded a needs_review job that should become 'pass'
                self.assertIn("needs_review -> pass", breakdown,
                              f"Expected 'needs_review -> pass' in breakdown, got: {breakdown}")
                self.assertEqual(breakdown["needs_review -> pass"], 1)
            finally:
                self._restore_env(originals)

    def test_rescore_default_dry_run_is_false(self) -> None:
        """Calling POST /admin/rescore with no dry_run parameter defaults to dry_run=False."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            originals = self._setup_env(root)
            try:
                from fastapi.testclient import TestClient
                from sopilot.main import create_app
                app = create_app()
                with TestClient(app) as client:
                    resp = client.post("/admin/rescore")
                self.assertEqual(resp.status_code, 200, resp.text)
                body = resp.json()
                self.assertFalse(body["dry_run"])
            finally:
                self._restore_env(originals)
