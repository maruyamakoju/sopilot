"""Unit tests for the @service_errors decorator."""

import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sopilot.api.error_handling import service_errors
from sopilot.exceptions import (
    AlgorithmError,
    ConfigurationError,
    InvalidStateError,
    NotFoundError,
    ServiceError,
    TransientError,
    ValidationError,
)


def _build_app() -> FastAPI:
    app = FastAPI()

    @app.get("/ok")
    @service_errors
    async def ok():
        return {"status": "ok"}

    @app.get("/not-found")
    @service_errors
    async def not_found():
        raise NotFoundError("item missing")

    @app.get("/not-found-rich")
    @service_errors
    async def not_found_rich():
        raise NotFoundError(
            "video abc123 does not exist",
            error_code="VIDEO_NOT_FOUND",
            context={"video_id": "abc123"},
        )

    @app.get("/invalid-state")
    @service_errors
    async def invalid_state():
        raise InvalidStateError("bad state")

    @app.get("/service-error")
    @service_errors
    async def service_error():
        raise ServiceError("generic problem")

    @app.get("/validation-error")
    @service_errors
    async def validation_error():
        raise ValidationError(
            "fps must be positive",
            error_code="INVALID_FPS",
            context={"field": "fps", "value": -1},
        )

    @app.get("/transient-error")
    @service_errors
    async def transient_error():
        raise TransientError(
            "database busy",
            error_code="DB_BUSY",
            retry_after=10,
        )

    @app.get("/config-error")
    @service_errors
    async def config_error():
        raise ConfigurationError("embedder not configured")

    @app.get("/algorithm-error")
    @service_errors
    async def algorithm_error():
        raise AlgorithmError(
            "DTW diverged",
            error_code="DTW_DIVERGENCE",
            context={"video_id": "v1", "gold_id": "g1"},
        )

    @app.get("/unhandled")
    @service_errors
    async def unhandled():
        raise ValueError("unexpected")

    return app


class ServiceErrorsDecoratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(_build_app())

    def test_success_passes_through(self) -> None:
        resp = self.client.get("/ok")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})

    def test_not_found_returns_404(self) -> None:
        resp = self.client.get("/not-found")
        self.assertEqual(resp.status_code, 404)
        body = resp.json()["error"]
        self.assertEqual(body["code"], "NOT_FOUND")
        self.assertIn("item missing", body["message"])
        self.assertEqual(body["details"], {})

    def test_not_found_rich_returns_404_with_context(self) -> None:
        resp = self.client.get("/not-found-rich")
        self.assertEqual(resp.status_code, 404)
        body = resp.json()["error"]
        self.assertEqual(body["code"], "VIDEO_NOT_FOUND")
        self.assertIn("abc123", body["message"])
        self.assertEqual(body["details"], {"video_id": "abc123"})

    def test_invalid_state_returns_409(self) -> None:
        resp = self.client.get("/invalid-state")
        self.assertEqual(resp.status_code, 409)
        body = resp.json()["error"]
        self.assertEqual(body["code"], "INVALID_STATE")
        self.assertIn("bad state", body["message"])

    def test_service_error_returns_400(self) -> None:
        resp = self.client.get("/service-error")
        self.assertEqual(resp.status_code, 400)
        body = resp.json()["error"]
        self.assertEqual(body["code"], "SERVICE_ERROR")
        self.assertIn("generic problem", body["message"])

    def test_validation_error_returns_422(self) -> None:
        resp = self.client.get("/validation-error")
        self.assertEqual(resp.status_code, 422)
        body = resp.json()["error"]
        self.assertEqual(body["code"], "INVALID_FPS")
        self.assertIn("fps must be positive", body["message"])
        self.assertEqual(body["details"]["field"], "fps")
        self.assertEqual(body["details"]["value"], -1)

    def test_transient_error_returns_503_with_retry_after(self) -> None:
        resp = self.client.get("/transient-error")
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.headers["Retry-After"], "10")
        body = resp.json()["error"]
        self.assertEqual(body["code"], "DB_BUSY")
        self.assertIn("database busy", body["message"])

    def test_configuration_error_returns_500(self) -> None:
        resp = self.client.get("/config-error")
        self.assertEqual(resp.status_code, 500)
        body = resp.json()["error"]
        self.assertEqual(body["code"], "CONFIGURATION_ERROR")
        self.assertIn("embedder not configured", body["message"])

    def test_algorithm_error_returns_500(self) -> None:
        resp = self.client.get("/algorithm-error")
        self.assertEqual(resp.status_code, 500)
        body = resp.json()["error"]
        self.assertEqual(body["code"], "DTW_DIVERGENCE")
        self.assertIn("DTW diverged", body["message"])
        self.assertEqual(body["details"]["video_id"], "v1")

    def test_unhandled_error_returns_500_catchall(self) -> None:
        resp = self.client.get("/unhandled")
        self.assertEqual(resp.status_code, 500)
        body = resp.json()["error"]
        self.assertEqual(body["code"], "INTERNAL_ERROR")
        # The original ValueError message "unexpected" must NOT leak verbatim.
        # Our generic message says "An unexpected error occurred..." which is
        # intentional -- but "ValueError" or the raw traceback must not appear.
        self.assertNotIn("ValueError", body["message"])
        self.assertNotIn("Traceback", body["message"])

    def test_error_response_envelope_structure(self) -> None:
        """Every error response must have the canonical envelope."""
        for path in ["/not-found", "/invalid-state", "/service-error",
                     "/validation-error", "/transient-error",
                     "/config-error", "/algorithm-error", "/unhandled"]:
            resp = self.client.get(path)
            body = resp.json()
            self.assertIn("error", body, f"Missing 'error' key for {path}")
            err = body["error"]
            self.assertIn("code", err, f"Missing 'code' for {path}")
            self.assertIn("message", err, f"Missing 'message' for {path}")
            self.assertIn("details", err, f"Missing 'details' for {path}")
            self.assertIsInstance(err["code"], str)
            self.assertIsInstance(err["message"], str)
            self.assertIsInstance(err["details"], dict)


if __name__ == "__main__":
    unittest.main()
