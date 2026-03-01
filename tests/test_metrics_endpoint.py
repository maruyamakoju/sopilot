"""Tests for the /metrics Prometheus endpoint."""
import os
import tempfile
import unittest
from pathlib import Path


class MetricsEndpointTests(unittest.TestCase):
    def _make_client(self, tmp_dir: str):
        os.environ["SOPILOT_DATA_DIR"] = str(Path(tmp_dir) / "data")
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-metrics"
        from fastapi.testclient import TestClient

        from sopilot.main import create_app
        app = create_app()
        return TestClient(app)

    def test_metrics_returns_200(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self._make_client(tmp)
            with client:
                resp = client.get("/metrics")
            self.assertEqual(resp.status_code, 200)

    def test_metrics_content_type_is_text_plain(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self._make_client(tmp)
            with client:
                resp = client.get("/metrics")
            self.assertIn("text/plain", resp.headers["content-type"])

    def test_metrics_contains_required_metric_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self._make_client(tmp)
            with client:
                resp = client.get("/metrics")
            body = resp.text
            for metric in [
                "sopilot_queue_depth",
                "sopilot_embed_requests_total",
                "sopilot_embed_fallback_uses_total",
                "sopilot_embed_failed_over",
                "sopilot_embed_permanently_failed",
                "sopilot_info",
            ]:
                self.assertIn(metric, body, f"Metric '{metric}' missing from /metrics output")

    def test_metrics_info_label_contains_task_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self._make_client(tmp)
            with client:
                resp = client.get("/metrics")
            self.assertIn("task-metrics", resp.text)

    def test_metrics_help_and_type_lines_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self._make_client(tmp)
            with client:
                resp = client.get("/metrics")
            body = resp.text
            self.assertIn("# HELP sopilot_queue_depth", body)
            self.assertIn("# TYPE sopilot_queue_depth gauge", body)
            self.assertIn("# HELP sopilot_embed_requests_total", body)
            self.assertIn("# TYPE sopilot_embed_requests_total counter", body)

    def test_metrics_queue_depth_is_numeric(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self._make_client(tmp)
            with client:
                resp = client.get("/metrics")
            for line in resp.text.splitlines():
                if line.startswith("sopilot_queue_depth "):
                    value = line.split(" ", 1)[1].strip()
                    self.assertTrue(value.lstrip("-").isdigit() or "." in value,
                                    f"queue depth value not numeric: {value!r}")
                    break
            else:
                self.fail("sopilot_queue_depth value line not found")

    def test_metrics_info_contains_real_version(self):
        """Verify version is the actual semver string, not the literal '__version__'."""
        from sopilot import __version__
        with tempfile.TemporaryDirectory() as tmp:
            client = self._make_client(tmp)
            with client:
                resp = client.get("/metrics")
            body = resp.text
            self.assertIn(f'version="{__version__}"', body)
            self.assertNotIn("version=__version__", body)

    def tearDown(self):
        for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
            os.environ.pop(key, None)


if __name__ == "__main__":
    unittest.main()
