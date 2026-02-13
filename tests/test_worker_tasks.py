"""Tests for sopilot.worker_tasks — singleton service, job delegation, shutdown."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# We must reset the module-level global before each test
@pytest.fixture(autouse=True)
def _reset_worker_global():
    """Reset the worker_tasks module global state before each test."""
    import sopilot.worker_tasks as wt
    wt._service = None
    yield
    wt._service = None


def _patch_deps():
    """Create patches for all _get_service dependencies."""
    mock_settings = MagicMock()
    mock_settings.db_path = ":memory:"
    mock_db = MagicMock()
    mock_service = MagicMock()

    patches = {
        "get_settings": patch("sopilot.worker_tasks.get_settings", return_value=mock_settings),
        "ensure_directories": patch("sopilot.worker_tasks.ensure_directories"),
        "Database": patch("sopilot.worker_tasks.Database", return_value=mock_db),
        "SopilotService": patch("sopilot.worker_tasks.SopilotService", return_value=mock_service),
    }
    return patches, mock_settings, mock_db, mock_service


class TestGetService:
    def test_initializes_service_on_first_call(self):
        patches, mock_settings, mock_db, mock_service = _patch_deps()
        with patches["get_settings"], patches["ensure_directories"] as mock_ensure, \
             patches["Database"] as mock_db_cls, patches["SopilotService"] as mock_svc_cls:
            from sopilot.worker_tasks import _get_service
            svc = _get_service()
            assert svc is mock_service
            mock_ensure.assert_called_once_with(mock_settings)
            mock_db_cls.assert_called_once_with(mock_settings.db_path)
            mock_svc_cls.assert_called_once()
            # Verify runtime_mode="worker"
            _, kwargs = mock_svc_cls.call_args
            assert kwargs.get("runtime_mode") == "worker"

    def test_returns_cached_on_second_call(self):
        patches, _, _, mock_service = _patch_deps()
        with patches["get_settings"], patches["ensure_directories"], \
             patches["Database"], patches["SopilotService"] as mock_svc_cls:
            from sopilot.worker_tasks import _get_service
            svc1 = _get_service()
            svc2 = _get_service()
            assert svc1 is svc2
            # SopilotService should only be constructed once
            assert mock_svc_cls.call_count == 1

    def test_init_error_propagates(self):
        patches, _, _, _ = _patch_deps()
        with patches["get_settings"] as mock_gs:
            mock_gs.side_effect = RuntimeError("settings broken")
            from sopilot.worker_tasks import _get_service
            with pytest.raises(RuntimeError, match="settings broken"):
                _get_service()


class TestRunJobs:
    def test_run_ingest_job_delegates(self):
        patches, _, _, mock_service = _patch_deps()
        with patches["get_settings"], patches["ensure_directories"], \
             patches["Database"], patches["SopilotService"]:
            from sopilot.worker_tasks import run_ingest_job
            run_ingest_job("job-123")
            mock_service.run_ingest_job.assert_called_once_with("job-123")

    def test_run_score_job_delegates(self):
        patches, _, _, mock_service = _patch_deps()
        with patches["get_settings"], patches["ensure_directories"], \
             patches["Database"], patches["SopilotService"]:
            from sopilot.worker_tasks import run_score_job
            run_score_job("score-456")
            mock_service.run_score_job.assert_called_once_with("score-456")

    def test_run_training_job_delegates(self):
        patches, _, _, mock_service = _patch_deps()
        with patches["get_settings"], patches["ensure_directories"], \
             patches["Database"], patches["SopilotService"]:
            from sopilot.worker_tasks import run_training_job
            run_training_job("train-789")
            mock_service.run_training_job.assert_called_once_with("train-789")


class TestShutdown:
    def test_shutdown_calls_service_shutdown(self):
        patches, _, _, mock_service = _patch_deps()
        with patches["get_settings"], patches["ensure_directories"], \
             patches["Database"], patches["SopilotService"]:
            from sopilot.worker_tasks import _get_service, shutdown_worker_service
            _get_service()  # initialize
            shutdown_worker_service()
            mock_service.shutdown.assert_called_once()

    def test_shutdown_before_init_is_noop(self):
        from sopilot.worker_tasks import shutdown_worker_service
        # Should not raise
        shutdown_worker_service()

    def test_double_shutdown_is_safe(self):
        patches, _, _, mock_service = _patch_deps()
        with patches["get_settings"], patches["ensure_directories"], \
             patches["Database"], patches["SopilotService"]:
            from sopilot.worker_tasks import _get_service, shutdown_worker_service
            _get_service()
            shutdown_worker_service()
            shutdown_worker_service()
            # shutdown() called only once (second call sees _service=None)
            mock_service.shutdown.assert_called_once()

    def test_shutdown_clears_global(self):
        patches, _, _, mock_service = _patch_deps()
        with patches["get_settings"], patches["ensure_directories"], \
             patches["Database"], patches["SopilotService"]:
            import sopilot.worker_tasks as wt
            wt._get_service()
            assert wt._service is not None
            wt.shutdown_worker_service()
            assert wt._service is None
