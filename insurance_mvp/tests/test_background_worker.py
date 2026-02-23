"""Tests for insurance_mvp.api.background — background task processing.

Tests cover:
- BackgroundWorker: submit, duplicate prevention, error handling, progress, status
- Worker lifecycle: init, active, shutdown
- Module-level functions: initialize_worker, get_worker, shutdown_worker
- PipelineProcessor._assessment_to_dict conversion
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from insurance_mvp.api.database import (
    AssessmentRepository,
    ClaimRepository,
    DatabaseManager,
)
from insurance_mvp.api.models import ClaimStatus

import insurance_mvp.api.background as bg_module
from insurance_mvp.api.background import (
    BackgroundWorker,
    PipelineProcessor,
    get_worker,
    initialize_worker,
    shutdown_worker,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db():
    """In-memory SQLite database with tables created."""
    db_manager = DatabaseManager("sqlite:///:memory:")
    db_manager.create_tables()
    return db_manager


@pytest.fixture
def _seed_claim(db):
    """Seed a single claim and return its ID."""
    claim_id = "claim_test_001"
    with db.get_session() as session:
        repo = ClaimRepository(session)
        repo.create(claim_id, "/videos/test.mp4")
    return claim_id


@pytest.fixture
def seeded_db(db, _seed_claim):
    """Return (db_manager, claim_id) with a pre-seeded claim."""
    return db, _seed_claim


def _make_mock_process(result: dict[str, Any] | None = None, error: Exception | None = None):
    """Create a mock processing function.

    If error is set, the function raises it.
    Otherwise returns result (or a default assessment dict).
    """
    default_result = {
        "severity": "MEDIUM",
        "confidence": 0.85,
        "prediction_set": ["MEDIUM", "HIGH"],
        "review_priority": "STANDARD",
        "fault_assessment": {
            "fault_ratio": 40.0,
            "reasoning": "Test fault reasoning",
            "applicable_rules": ["Rule A"],
            "scenario_type": "rear_end",
            "traffic_signal": None,
            "right_of_way": None,
        },
        "fraud_risk": {
            "risk_score": 0.1,
            "indicators": [],
            "reasoning": "No fraud indicators",
        },
        "hazards": [],
        "evidence": [],
        "causal_reasoning": "Test causal reasoning",
        "recommended_action": "REVIEW",
        "processing_time_sec": 0.5,
        "model_version": "test-v1",
    }

    def fn(claim_id, update_progress):
        if error is not None:
            raise error
        return result or default_result

    return fn


# ---------------------------------------------------------------------------
# BackgroundWorker basics
# ---------------------------------------------------------------------------


class TestBackgroundWorker:

    def test_submit_and_process(self, seeded_db):
        """Submit a claim and verify it reaches ASSESSED status."""
        db, claim_id = seeded_db
        worker = BackgroundWorker(db, max_workers=1, process_function=_make_mock_process())

        ok = worker.submit_claim(claim_id)
        assert ok is True

        # Wait for the task to complete
        worker.shutdown(wait=True)

        with db.get_session() as session:
            repo = ClaimRepository(session)
            claim = repo.get_by_id(claim_id)
            assert claim.status == ClaimStatus.ASSESSED
            assert claim.progress_percent == 100.0

            ar = AssessmentRepository(session)
            assessment = ar.get_by_claim_id(claim_id)
            assert assessment is not None
            assert assessment.severity == "MEDIUM"

    def test_duplicate_submission_rejected(self, seeded_db):
        """Submitting the same claim_id twice returns False the second time."""
        db, claim_id = seeded_db
        # Use a slow process function so the first task is still active
        def slow_process(cid, update_progress):
            time.sleep(1.0)
            return _make_mock_process()(cid, update_progress)

        worker = BackgroundWorker(db, max_workers=1, process_function=slow_process)
        assert worker.submit_claim(claim_id) is True
        assert worker.submit_claim(claim_id) is False
        worker.shutdown(wait=True)

    def test_processing_failure_sets_failed_status(self, seeded_db):
        """When the process function raises, claim status becomes FAILED."""
        db, claim_id = seeded_db
        worker = BackgroundWorker(
            db, max_workers=1,
            process_function=_make_mock_process(error=RuntimeError("boom")),
        )

        worker.submit_claim(claim_id)
        worker.shutdown(wait=True)

        with db.get_session() as session:
            repo = ClaimRepository(session)
            claim = repo.get_by_id(claim_id)
            assert claim.status == ClaimStatus.FAILED
            assert "boom" in (claim.error_message or "")

    def test_progress_updates(self, seeded_db):
        """Verify progress callback writes to the database."""
        db, claim_id = seeded_db
        progress_values = []

        def tracking_process(cid, update_progress):
            for pct in [10.0, 50.0, 90.0]:
                update_progress(cid, pct)
                progress_values.append(pct)
            return _make_mock_process()(cid, update_progress)

        worker = BackgroundWorker(db, max_workers=1, process_function=tracking_process)
        worker.submit_claim(claim_id)
        worker.shutdown(wait=True)

        assert progress_values == [10.0, 50.0, 90.0]

    def test_get_status_processing(self, seeded_db):
        """get_status returns 'processing' while task is running."""
        db, claim_id = seeded_db

        def slow_process(cid, update_progress):
            time.sleep(1.0)
            return _make_mock_process()(cid, update_progress)

        worker = BackgroundWorker(db, max_workers=1, process_function=slow_process)
        worker.submit_claim(claim_id)

        # Immediately after submit, future should not be done yet
        status = worker.get_status(claim_id)
        assert status == "processing"
        worker.shutdown(wait=True)

    def test_get_status_unknown(self, db):
        """get_status returns None for unknown claim ID."""
        worker = BackgroundWorker(db, max_workers=1)
        assert worker.get_status("nonexistent") is None
        worker.shutdown(wait=False)


class TestWorkerLifecycle:

    def test_is_active_no_tasks(self, db):
        """Worker with no active tasks is not considered active."""
        worker = BackgroundWorker(db, max_workers=1)
        assert worker.is_active() is False
        worker.shutdown(wait=False)

    def test_shutdown_sets_flag(self, db):
        """Shutdown sets internal _shutdown flag."""
        worker = BackgroundWorker(db, max_workers=1)
        assert worker._shutdown is False
        worker.shutdown(wait=False)
        assert worker._shutdown is True

    def test_cleanup_task(self, db):
        """_cleanup_task removes claim from active_tasks dict."""
        worker = BackgroundWorker(db, max_workers=1)
        worker.active_tasks["fake"] = MagicMock()
        worker._cleanup_task("fake")
        assert "fake" not in worker.active_tasks
        worker.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Module-level singleton functions
# ---------------------------------------------------------------------------


class TestModuleLevelFunctions:

    def setup_method(self):
        """Reset global worker before each test."""
        bg_module._global_worker = None

    def teardown_method(self):
        """Clean up global worker after each test."""
        if bg_module._global_worker is not None:
            bg_module._global_worker.shutdown(wait=False)
            bg_module._global_worker = None

    def test_get_worker_before_init_raises(self):
        with pytest.raises(RuntimeError, match="not initialized"):
            get_worker()

    def test_initialize_and_get_worker(self, db):
        worker = initialize_worker(db, max_workers=1)
        assert worker is get_worker()
        shutdown_worker(wait=False)

    def test_initialize_worker_idempotent(self, db):
        w1 = initialize_worker(db, max_workers=1)
        w2 = initialize_worker(db, max_workers=2)
        assert w1 is w2  # second call returns existing
        shutdown_worker(wait=False)

    def test_shutdown_worker_clears_global(self, db):
        initialize_worker(db, max_workers=1)
        shutdown_worker(wait=False)
        assert bg_module._global_worker is None


# ---------------------------------------------------------------------------
# PipelineProcessor._assessment_to_dict
# ---------------------------------------------------------------------------


class TestPipelineProcessorAssessmentToDict:

    def test_conversion(self):
        """_assessment_to_dict converts domain objects to dict."""
        assessment = MagicMock()
        assessment.severity = "HIGH"
        assessment.confidence = 0.92
        assessment.prediction_set = {"HIGH", "MEDIUM"}
        assessment.review_priority = "URGENT"
        assessment.causal_reasoning = "Test reasoning"
        assessment.recommended_action = "REVIEW"
        assessment.hazards = []
        assessment.evidence = []

        fault = MagicMock()
        fault.fault_ratio = 80.0
        fault.reasoning = "Test fault"
        fault.applicable_rules = ["Rule 1"]
        fault.scenario_type = "rear_end"
        fault.traffic_signal = "red"
        fault.right_of_way = "lead vehicle"
        assessment.fault_assessment = fault

        fraud = MagicMock()
        fraud.risk_score = 0.1
        fraud.indicators = []
        fraud.reasoning = "No fraud"
        assessment.fraud_risk = fraud

        result = MagicMock()
        result.processing_time_sec = 5.0

        d = PipelineProcessor._assessment_to_dict(assessment, result)

        assert d["severity"] == "HIGH"
        assert d["confidence"] == 0.92
        assert isinstance(d["prediction_set"], list)
        assert d["review_priority"] == "URGENT"
        assert d["fault_assessment"]["fault_ratio"] == 80.0
        assert d["fault_assessment"]["traffic_signal"] == "red"
        assert d["fraud_risk"]["risk_score"] == 0.1
        assert d["processing_time_sec"] == 5.0
        assert d["model_version"] == "insurance-pipeline-v1.0"
