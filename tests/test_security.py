"""Tests for API key auth, CORS restrictions, and upload size limits."""
import io
import os
import tempfile
import unittest
from pathlib import Path

from sopilot.exceptions import FileTooLargeError
from sopilot.services.storage import FileStorage


class ApiKeyMiddlewareTests(unittest.TestCase):
    """API key authentication via X-API-Key header."""

    def _make_client(self, tmp_dir: str, *, api_key: str = ""):
        os.environ["SOPILOT_DATA_DIR"] = str(Path(tmp_dir) / "data")
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-auth"
        if api_key:
            os.environ["SOPILOT_API_KEY"] = api_key
        else:
            os.environ.pop("SOPILOT_API_KEY", None)
        from fastapi.testclient import TestClient

        from sopilot.main import create_app
        app = create_app()
        return TestClient(app)

    def test_no_key_configured_allows_all(self):
        """When SOPILOT_API_KEY is unset, all endpoints are accessible."""
        with tempfile.TemporaryDirectory() as tmp:
            client = self._make_client(tmp, api_key="")
            with client:
                resp = client.get("/dataset/summary")
            self.assertEqual(resp.status_code, 200)

    def test_key_configured_rejects_without_header(self):
        """When SOPILOT_API_KEY is set, requests without X-API-Key get 401."""
        with tempfile.TemporaryDirectory() as tmp:
            client = self._make_client(tmp, api_key="secret-123")
            with client:
                resp = client.get("/dataset/summary")
            self.assertEqual(resp.status_code, 401)
            self.assertIn("API key", resp.json()["detail"])

    def test_key_configured_rejects_wrong_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self._make_client(tmp, api_key="secret-123")
            with client:
                resp = client.get("/dataset/summary", headers={"X-API-Key": "wrong"})
            self.assertEqual(resp.status_code, 401)

    def test_key_configured_accepts_correct_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self._make_client(tmp, api_key="secret-123")
            with client:
                resp = client.get("/dataset/summary", headers={"X-API-Key": "secret-123"})
            self.assertEqual(resp.status_code, 200)

    def test_public_paths_exempt_from_auth(self):
        """Health, metrics, status, and root are accessible without key."""
        with tempfile.TemporaryDirectory() as tmp:
            client = self._make_client(tmp, api_key="secret-123")
            with client:
                for path in ["/health", "/readiness", "/metrics", "/status", "/"]:
                    resp = client.get(path)
                    self.assertIn(resp.status_code, {200, 307},
                                  f"{path} should be accessible without API key, got {resp.status_code}")

    def tearDown(self):
        for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND",
                     "SOPILOT_PRIMARY_TASK_ID", "SOPILOT_API_KEY"):
            os.environ.pop(key, None)


class UploadSizeLimitTests(unittest.TestCase):
    """FileStorage enforces max upload size."""

    def test_within_limit_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = FileStorage(Path(tmp), max_upload_bytes=1024 * 1024)
            data = io.BytesIO(b"\x00" * 1000)
            path = storage.save_upload(1, "small.mp4", data)
            self.assertTrue(Path(path).exists())

    def test_exceeds_limit_raises_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = FileStorage(Path(tmp), max_upload_bytes=100)
            data = io.BytesIO(b"\x00" * 200)
            with self.assertRaises(FileTooLargeError) as ctx:
                storage.save_upload(2, "big.mp4", data)
            self.assertIn("limit", str(ctx.exception).lower())
            # Partial file should be cleaned up
            target = Path(tmp) / "00000002.mp4"
            self.assertFalse(target.exists(), "Partial file should be deleted after exceeding limit")

    def test_zero_limit_means_unlimited(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = FileStorage(Path(tmp), max_upload_bytes=0)
            data = io.BytesIO(b"\x00" * 10_000)
            path = storage.save_upload(3, "large.mp4", data)
            self.assertTrue(Path(path).exists())

    def test_413_via_http(self):
        """Upload exceeding limit returns HTTP 413."""
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["SOPILOT_DATA_DIR"] = str(Path(tmp) / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-size"
            os.environ["SOPILOT_MAX_UPLOAD_MB"] = "1"  # 1 MB
            os.environ.pop("SOPILOT_API_KEY", None)
            try:
                from fastapi.testclient import TestClient

                from sopilot.main import create_app
                app = create_app()
                with TestClient(app) as client:
                    # Create a ~2MB payload (just random bytes, not a real video)
                    big_data = b"\x00" * (2 * 1024 * 1024)
                    resp = client.post(
                        "/videos",
                        files={"file": ("big.mp4", io.BytesIO(big_data), "video/mp4")},
                        data={"task_id": "task-size"},
                    )
                self.assertEqual(resp.status_code, 413)
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND",
                             "SOPILOT_PRIMARY_TASK_ID", "SOPILOT_MAX_UPLOAD_MB"):
                    os.environ.pop(key, None)


class SecurityHeaderTests(unittest.TestCase):
    """Security headers should be present on all responses."""

    def test_security_headers_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["SOPILOT_DATA_DIR"] = str(Path(tmp) / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-sec"
            os.environ.pop("SOPILOT_API_KEY", None)
            try:
                from fastapi.testclient import TestClient

                from sopilot.main import create_app
                app = create_app()
                with TestClient(app) as client:
                    resp = client.get("/health")
                self.assertEqual(resp.headers.get("X-Content-Type-Options"), "nosniff")
                self.assertEqual(resp.headers.get("X-Frame-Options"), "DENY")
                self.assertEqual(resp.headers.get("Referrer-Policy"), "strict-origin-when-cross-origin")
                self.assertIn("camera=()", resp.headers.get("Permissions-Policy", ""))
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)

    def test_request_id_header_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["SOPILOT_DATA_DIR"] = str(Path(tmp) / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-sec"
            os.environ.pop("SOPILOT_API_KEY", None)
            try:
                from fastapi.testclient import TestClient

                from sopilot.main import create_app
                app = create_app()
                with TestClient(app) as client:
                    resp = client.get("/health")
                self.assertIn("X-Request-ID", resp.headers)
                self.assertTrue(len(resp.headers["X-Request-ID"]) > 0)
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)


class CorsConfigTests(unittest.TestCase):
    """CORS origins should be configurable via env var."""

    def test_default_cors_origins(self):
        from sopilot.config import Settings
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["SOPILOT_DATA_DIR"] = tmp
            os.environ.pop("SOPILOT_CORS_ORIGINS", None)
            try:
                settings = Settings.from_env()
                self.assertIn("http://localhost:8000", settings.cors_origins)
                self.assertIn("http://127.0.0.1:8000", settings.cors_origins)
            finally:
                os.environ.pop("SOPILOT_DATA_DIR", None)

    def test_custom_cors_origins(self):
        from sopilot.config import Settings
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["SOPILOT_DATA_DIR"] = tmp
            os.environ["SOPILOT_CORS_ORIGINS"] = "https://app.example.com,https://admin.example.com"
            try:
                settings = Settings.from_env()
                self.assertEqual(settings.cors_origins, [
                    "https://app.example.com",
                    "https://admin.example.com",
                ])
            finally:
                os.environ.pop("SOPILOT_DATA_DIR", None)
                os.environ.pop("SOPILOT_CORS_ORIGINS", None)

    def test_wildcard_cors_origins(self):
        from sopilot.config import Settings
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["SOPILOT_DATA_DIR"] = tmp
            os.environ["SOPILOT_CORS_ORIGINS"] = "*"
            try:
                settings = Settings.from_env()
                self.assertEqual(settings.cors_origins, ["*"])
            finally:
                os.environ.pop("SOPILOT_DATA_DIR", None)
                os.environ.pop("SOPILOT_CORS_ORIGINS", None)


if __name__ == "__main__":
    unittest.main()
