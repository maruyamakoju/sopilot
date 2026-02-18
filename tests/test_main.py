"""Tests for main module (P2-5).

Covers app creation and entry point wiring.
"""

from sopilot.main import app


class TestApp:
    """Test application creation."""

    def test_app_exists(self):
        """FastAPI app is created."""
        assert app is not None

    def test_app_has_routes(self):
        """App has registered routes."""
        assert len(app.routes) > 0

    def test_run_function_exists(self):
        """run() function is importable."""
        from sopilot.main import run

        assert callable(run)
