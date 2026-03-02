"""Tests for Perception Router API endpoints (/vigil/perception/*).

Coverage:
  - POST /vigil/perception/narration — narration generation (schema, 404 no frames)
  - POST /vigil/perception/query    — NL context query (schema, Japanese question)
  - GET  /vigil/perception/summary  — session summary (schema, zero-state)
  - GET  /vigil/perception/entities/{id} — entity lookup (404 for missing)
  - GET  /vigil/perception/activities   — activity classification list
  - GET  /vigil/perception/predictions  — predictions list
  - GET  /vigil/perception/causality    — causal links list
  - POST /vigil/perception/timeline     — timeline with filters
  - GET  /vigil/perception/state        — perception state snapshot
  - 400 error when perception backend is not active
  - Request validation (style, language, question constraints)
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from fastapi.testclient import TestClient

from sopilot.database import Database
from sopilot.main import create_app


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _init_db(db_path: Path) -> str:
    """Initialise a fresh SQLite database with all tables + migrations."""
    db = Database(db_path)
    db.close()
    return str(db_path)


def _build_mock_perception_engine():
    """Build a mock PerceptionEngine with all components as mocks.

    Returns a mock that has the same attribute interface as the real
    PerceptionEngine so that the perception_router helper functions
    (_get_perception_engine, _require_component) work correctly.
    """
    engine = MagicMock()

    # Simulate zero-state: no frames processed
    engine.frames_processed = 0
    engine.average_processing_ms = 0.0
    engine.get_world_state.return_value = None

    # Components present (all non-None so _require_component passes)
    engine._detector = MagicMock()
    engine._tracker = MagicMock()
    engine._scene_builder = MagicMock()
    engine._world_model = MagicMock()
    engine._trajectory_predictor = MagicMock()
    engine._activity_classifier = MagicMock()
    engine._attention_scorer = MagicMock()
    engine._causal_reasoner = MagicMock()
    engine._context_memory = MagicMock()
    engine._narrator = MagicMock()
    engine._zones = []

    # CausalReasoner._links is an empty dict in fresh state
    engine._causal_reasoner._links = {}

    # ContextMemory.query returns a Japanese answer
    engine._context_memory.query.return_value = (
        "セッション中に 0 人を検出しました。現在 0 人がシーン内にいます。"
    )

    # ContextMemory.get_session_summary returns a dataclass-like mock
    summary = MagicMock()
    summary.start_time = 0.0
    summary.current_time = 0.0
    summary.duration_seconds = 0.0
    summary.total_frames_processed = 0
    summary.unique_entities_seen = 0
    summary.current_entity_count = 0
    summary.total_violations = 0
    summary.violations_by_severity = {}
    summary.violations_by_rule = {}
    summary.notable_events = []
    engine._context_memory.get_session_summary.return_value = summary

    # ContextMemory.get_entity_summary returns None for unknown entities
    engine._context_memory.get_entity_summary.return_value = None

    # ContextMemory.get_timeline returns empty list
    engine._context_memory.get_timeline.return_value = []

    return engine


def _build_mock_vlm_with_engine(engine):
    """Build a mock VLM client that has an _engine attribute.

    This mimics PerceptionVLMClient which has `_engine` that the
    perception_router inspects via hasattr(vlm, '_engine').
    """
    vlm = MagicMock()
    vlm._engine = engine
    return vlm


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Perception endpoints WITH perception backend active
# ──────────────────────────────────────────────────────────────────────────────


class TestPerceptionAPIWithBackend(unittest.TestCase):
    """E2E HTTP tests for /vigil/perception/* with perception backend active.

    Uses a mock PerceptionEngine attached to the pipeline's VLM client so
    that _get_perception_engine() in the router finds it via vlm._engine.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        os.environ["SOPILOT_DATA_DIR"] = str(self.root / "data")
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_PRIMARY_TASK_ID"] = "perception-test"
        os.environ["SOPILOT_RATE_LIMIT_RPM"] = "0"

        self.app = create_app()
        self.client = TestClient(self.app)

        # Inject mock perception engine via the VLM client
        self.engine = _build_mock_perception_engine()
        self.mock_vlm = _build_mock_vlm_with_engine(self.engine)
        self.app.state.vigil_pipeline._vlm = self.mock_vlm

    def tearDown(self) -> None:
        self._tmp.cleanup()
        for k in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND",
                   "SOPILOT_PRIMARY_TASK_ID", "SOPILOT_RATE_LIMIT_RPM"):
            os.environ.pop(k, None)

    # ── POST /vigil/perception/narration ──────────────────────────────────

    def test_narration_404_no_frames(self) -> None:
        """Narration returns 404 when no frames have been processed."""
        r = self.client.post(
            "/vigil/perception/narration",
            json={"style": "standard", "language": "ja"},
        )
        self.assertEqual(r.status_code, 404)
        self.assertIn("No frames processed", r.json()["detail"])

    def test_narration_success_with_world_state(self) -> None:
        """Narration returns 200 with valid schema when world state exists."""
        # Create a mock world state
        world_state = MagicMock()
        self.engine.get_world_state.return_value = world_state

        # Create a mock narration result
        narration = MagicMock()
        narration.text_ja = "現在、2名の作業員がいます。"
        narration.text_en = "Currently, 2 workers are present."
        narration.style = MagicMock()
        narration.style.value = "standard"
        narration.key_facts = ["2 workers present"]
        narration.entity_mentions = [1, 2]
        narration.timestamp = 10.5
        narration.frame_number = 10
        self.engine._narrator.narrate.return_value = narration

        r = self.client.post(
            "/vigil/perception/narration",
            json={"style": "standard", "language": "ja"},
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()

        # Verify response schema matches NarrationResponse
        self.assertIn("text_ja", data)
        self.assertIn("text_en", data)
        self.assertIn("style", data)
        self.assertIn("key_facts", data)
        self.assertIn("entity_mentions", data)
        self.assertIn("timestamp", data)
        self.assertIn("frame_number", data)

        # Verify content
        self.assertEqual(data["text_ja"], "現在、2名の作業員がいます。")
        self.assertEqual(data["text_en"], "Currently, 2 workers are present.")
        self.assertEqual(data["style"], "standard")
        self.assertIsInstance(data["key_facts"], list)
        self.assertIsInstance(data["entity_mentions"], list)
        self.assertIsInstance(data["timestamp"], float)
        self.assertIsInstance(data["frame_number"], int)

    def test_narration_brief_style(self) -> None:
        """Narration accepts brief style parameter."""
        world_state = MagicMock()
        self.engine.get_world_state.return_value = world_state

        narration = MagicMock()
        narration.text_ja = "作業員2名。"
        narration.text_en = "2 workers."
        narration.style = MagicMock()
        narration.style.value = "brief"
        narration.key_facts = []
        narration.entity_mentions = []
        narration.timestamp = 1.0
        narration.frame_number = 1
        self.engine._narrator.narrate.return_value = narration

        r = self.client.post(
            "/vigil/perception/narration",
            json={"style": "brief", "language": "en"},
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["style"], "brief")

    def test_narration_detailed_style(self) -> None:
        """Narration accepts detailed style parameter."""
        world_state = MagicMock()
        self.engine.get_world_state.return_value = world_state

        narration = MagicMock()
        narration.text_ja = "詳細ナレーション"
        narration.text_en = "Detailed narration"
        narration.style = MagicMock()
        narration.style.value = "detailed"
        narration.key_facts = ["fact1", "fact2"]
        narration.entity_mentions = [1]
        narration.timestamp = 5.0
        narration.frame_number = 5
        self.engine._narrator.narrate.return_value = narration

        r = self.client.post(
            "/vigil/perception/narration",
            json={"style": "detailed", "language": "ja"},
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["style"], "detailed")

    def test_narration_invalid_style_422(self) -> None:
        """Narration rejects invalid style with 422."""
        r = self.client.post(
            "/vigil/perception/narration",
            json={"style": "verbose", "language": "ja"},
        )
        self.assertEqual(r.status_code, 422)

    def test_narration_invalid_language_422(self) -> None:
        """Narration rejects invalid language with 422."""
        r = self.client.post(
            "/vigil/perception/narration",
            json={"style": "standard", "language": "fr"},
        )
        self.assertEqual(r.status_code, 422)

    def test_narration_default_params(self) -> None:
        """Narration uses defaults (standard, ja) when no body fields given."""
        world_state = MagicMock()
        self.engine.get_world_state.return_value = world_state

        narration = MagicMock()
        narration.text_ja = "テスト"
        narration.text_en = "Test"
        narration.style = MagicMock()
        narration.style.value = "standard"
        narration.key_facts = []
        narration.entity_mentions = []
        narration.timestamp = 0.0
        narration.frame_number = 0
        self.engine._narrator.narrate.return_value = narration

        r = self.client.post("/vigil/perception/narration", json={})
        self.assertEqual(r.status_code, 200)

    # ── POST /vigil/perception/query ──────────────────────────────────────

    def test_query_japanese_question(self) -> None:
        """Context query with Japanese question returns valid schema."""
        r = self.client.post(
            "/vigil/perception/query",
            json={"question": "何人いる？"},
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()

        # Verify response schema matches ContextQueryResponse
        self.assertIn("question", data)
        self.assertIn("answer", data)
        self.assertIn("session_id", data)

        # Verify echo of question
        self.assertEqual(data["question"], "何人いる？")
        # Verify answer is a string
        self.assertIsInstance(data["answer"], str)
        # session_id is always 0 for perception engine
        self.assertEqual(data["session_id"], 0)

    def test_query_english_question(self) -> None:
        """Context query with English question works."""
        self.engine._context_memory.query.return_value = "No persons detected."
        r = self.client.post(
            "/vigil/perception/query",
            json={"question": "How many people are there?"},
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["question"], "How many people are there?")
        self.assertEqual(data["answer"], "No persons detected.")

    def test_query_empty_question_422(self) -> None:
        """Context query rejects empty question with 422."""
        r = self.client.post(
            "/vigil/perception/query",
            json={"question": ""},
        )
        self.assertEqual(r.status_code, 422)

    def test_query_restricted_area_question(self) -> None:
        """Context query for restricted area returns answer."""
        self.engine._context_memory.query.return_value = (
            "制限エリアへの訪問者はいません。"
        )
        r = self.client.post(
            "/vigil/perception/query",
            json={"question": "制限エリアに何人入った？"},
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn("制限エリア", r.json()["answer"])

    def test_query_violation_count_question(self) -> None:
        """Context query for violation count returns answer."""
        self.engine._context_memory.query.return_value = (
            "違反は記録されていません。"
        )
        r = self.client.post(
            "/vigil/perception/query",
            json={"question": "違反は何件？"},
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn("違反", r.json()["answer"])

    # ── GET /vigil/perception/summary ─────────────────────────────────────

    def test_summary_zero_state(self) -> None:
        """Summary returns valid schema in zero-state (no frames processed)."""
        r = self.client.get("/vigil/perception/summary")
        self.assertEqual(r.status_code, 200)
        data = r.json()

        # Verify all SessionSummaryResponse fields
        expected_fields = [
            "start_time", "current_time", "duration_seconds",
            "total_frames_processed", "unique_entities_seen",
            "current_entity_count", "total_violations",
            "violations_by_severity", "violations_by_rule",
            "notable_events",
        ]
        for field in expected_fields:
            self.assertIn(field, data, f"Missing field: {field}")

        # Verify types
        self.assertIsInstance(data["start_time"], float)
        self.assertIsInstance(data["current_time"], float)
        self.assertIsInstance(data["duration_seconds"], float)
        self.assertIsInstance(data["total_frames_processed"], int)
        self.assertIsInstance(data["unique_entities_seen"], int)
        self.assertIsInstance(data["current_entity_count"], int)
        self.assertIsInstance(data["total_violations"], int)
        self.assertIsInstance(data["violations_by_severity"], dict)
        self.assertIsInstance(data["violations_by_rule"], dict)
        self.assertIsInstance(data["notable_events"], list)

        # Zero-state values
        self.assertEqual(data["total_frames_processed"], 0)
        self.assertEqual(data["unique_entities_seen"], 0)
        self.assertEqual(data["total_violations"], 0)

    def test_summary_with_data(self) -> None:
        """Summary returns populated data when session has activity."""
        summary = MagicMock()
        summary.start_time = 100.0
        summary.current_time = 200.0
        summary.duration_seconds = 100.0
        summary.total_frames_processed = 50
        summary.unique_entities_seen = 3
        summary.current_entity_count = 2
        summary.total_violations = 5
        summary.violations_by_severity = {"warning": 3, "critical": 2}
        summary.violations_by_rule = {"helmet_rule": 3, "zone_rule": 2}
        summary.notable_events = ["[100.0] Entry detected"]
        self.engine._context_memory.get_session_summary.return_value = summary

        r = self.client.get("/vigil/perception/summary")
        self.assertEqual(r.status_code, 200)
        data = r.json()

        self.assertEqual(data["total_frames_processed"], 50)
        self.assertEqual(data["unique_entities_seen"], 3)
        self.assertEqual(data["current_entity_count"], 2)
        self.assertEqual(data["total_violations"], 5)
        self.assertEqual(data["violations_by_severity"]["warning"], 3)
        self.assertEqual(data["violations_by_severity"]["critical"], 2)
        self.assertEqual(len(data["notable_events"]), 1)

    # ── GET /vigil/perception/entities/{id} ───────────────────────────────

    def test_entity_not_found_404(self) -> None:
        """Entity lookup returns 404 for nonexistent entity."""
        r = self.client.get("/vigil/perception/entities/9999")
        self.assertEqual(r.status_code, 404)
        self.assertIn("9999", r.json()["detail"])

    def test_entity_found(self) -> None:
        """Entity lookup returns valid schema when entity exists."""
        entity_summary = MagicMock()
        entity_summary.entity_id = 1
        entity_summary.label = "person"
        entity_summary.first_seen = 10.0
        entity_summary.last_seen = 50.0
        entity_summary.total_frames = 40
        entity_summary.zones_visited = ["work_area", "restricted_1"]
        entity_summary.activities = ["walking", "stationary"]
        entity_summary.current_activity = "stationary"
        entity_summary.current_zone = "work_area"
        entity_summary.total_distance = 12.345
        self.engine._context_memory.get_entity_summary.return_value = entity_summary

        r = self.client.get("/vigil/perception/entities/1")
        self.assertEqual(r.status_code, 200)
        data = r.json()

        # Verify all EntitySummaryResponse fields
        self.assertEqual(data["entity_id"], 1)
        self.assertEqual(data["label"], "person")
        self.assertEqual(data["first_seen"], 10.0)
        self.assertEqual(data["last_seen"], 50.0)
        self.assertEqual(data["total_frames"], 40)
        self.assertEqual(data["zones_visited"], ["work_area", "restricted_1"])
        self.assertEqual(data["activities"], ["walking", "stationary"])
        self.assertEqual(data["current_activity"], "stationary")
        self.assertEqual(data["current_zone"], "work_area")
        self.assertAlmostEqual(data["total_distance"], 12.345, places=2)

    def test_entity_with_null_zone(self) -> None:
        """Entity can have current_zone as null."""
        entity_summary = MagicMock()
        entity_summary.entity_id = 2
        entity_summary.label = "helmet"
        entity_summary.first_seen = 5.0
        entity_summary.last_seen = 5.0
        entity_summary.total_frames = 1
        entity_summary.zones_visited = []
        entity_summary.activities = []
        entity_summary.current_activity = "unknown"
        entity_summary.current_zone = None
        entity_summary.total_distance = 0.0
        self.engine._context_memory.get_entity_summary.return_value = entity_summary

        r = self.client.get("/vigil/perception/entities/2")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIsNone(data["current_zone"])
        self.assertEqual(data["zones_visited"], [])

    # ── GET /vigil/perception/activities ───────────────────────────────────

    def test_activities_empty_no_world_state(self) -> None:
        """Activities returns empty list when no world state exists."""
        r = self.client.get("/vigil/perception/activities")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 0)

    def test_activities_with_world_state(self) -> None:
        """Activities returns classification results when world state exists."""
        world_state = MagicMock()
        world_state.active_tracks = {1: MagicMock(), 2: MagicMock()}
        self.engine.get_world_state.return_value = world_state

        # Mock classifier returns
        from unittest.mock import MagicMock as MM
        cls1 = MM()
        cls1.activity = MM()
        cls1.activity.value = "walking"
        cls1.confidence = 0.85
        cls1.secondary_activity = None
        cls1.secondary_confidence = 0.0
        cls1.features = MM()
        cls1.features.mean_speed = 0.02
        cls1.features.max_speed = 0.05
        cls1.features.speed_variance = 0.001
        cls1.features.direction_change_rate = 0.1
        cls1.features.displacement_ratio = 0.8
        cls1.features.bounding_area = 0.005
        cls1.features.duration_frames = 30

        cls2 = MM()
        cls2.activity = MM()
        cls2.activity.value = "stationary"
        cls2.confidence = 0.95
        cls2.secondary_activity = MM()
        cls2.secondary_activity.value = "loitering"
        cls2.secondary_confidence = 0.3
        cls2.features = MM()
        cls2.features.mean_speed = 0.0
        cls2.features.max_speed = 0.001
        cls2.features.speed_variance = 0.0
        cls2.features.direction_change_rate = 0.0
        cls2.features.displacement_ratio = 0.0
        cls2.features.bounding_area = 0.004
        cls2.features.duration_frames = 50

        self.engine._activity_classifier.classify_batch.return_value = {
            1: cls1, 2: cls2,
        }

        r = self.client.get("/vigil/perception/activities")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)

        # Verify schema of each ActivityResponse
        for item in data:
            self.assertIn("entity_id", item)
            self.assertIn("activity", item)
            self.assertIn("confidence", item)
            self.assertIn("features", item)
            self.assertIsInstance(item["features"], dict)
            self.assertIn("mean_speed", item["features"])
            self.assertIn("max_speed", item["features"])

        # Check specific values
        activities_by_id = {item["entity_id"]: item for item in data}
        self.assertEqual(activities_by_id[1]["activity"], "walking")
        self.assertAlmostEqual(activities_by_id[1]["confidence"], 0.85, places=2)
        self.assertEqual(activities_by_id[2]["activity"], "stationary")
        self.assertEqual(activities_by_id[2]["secondary_activity"], "loitering")

    # ── GET /vigil/perception/predictions ─────────────────────────────────

    def test_predictions_empty_no_world_state(self) -> None:
        """Predictions returns empty list when no world state exists."""
        r = self.client.get("/vigil/perception/predictions")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 0)

    def test_predictions_empty_no_active_tracks(self) -> None:
        """Predictions returns empty list when world state has no tracks."""
        world_state = MagicMock()
        world_state.active_tracks = {}
        self.engine.get_world_state.return_value = world_state

        self.engine._trajectory_predictor.predict_zone_entry.return_value = []

        r = self.client.get("/vigil/perception/predictions")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 0)

    def test_predictions_with_zone_entry(self) -> None:
        """Predictions returns zone entry prediction when predicted."""
        world_state = MagicMock()
        track1 = MagicMock()
        track1.state = MagicMock()
        track1.state.value = "active"
        world_state.active_tracks = {1: track1}
        self.engine.get_world_state.return_value = world_state

        # Zone entry prediction
        zp = MagicMock()
        zp.zone_id = "restricted_1"
        zp.zone_name = "Restricted Area 1"
        zp.predicted_entry_point = (0.5, 0.3)
        zp.confidence = 0.8
        zp.estimated_seconds = 5.2
        self.engine._trajectory_predictor.predict_zone_entry.return_value = [zp]
        self.engine._trajectory_predictor.predict_collision.return_value = None

        r = self.client.get("/vigil/perception/predictions")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 1)

        pred = data[0]
        self.assertEqual(pred["prediction_type"], "zone_entry")
        self.assertEqual(pred["entity_id"], 1)
        self.assertIn("zone_id", pred["details"])
        self.assertEqual(pred["details"]["zone_id"], "restricted_1")
        self.assertAlmostEqual(pred["confidence"], 0.8, places=1)
        self.assertAlmostEqual(pred["estimated_seconds"], 5.2, places=1)

    # ── GET /vigil/perception/causality ───────────────────────────────────

    def test_causality_empty(self) -> None:
        """Causality returns empty list when no causal links exist."""
        r = self.client.get("/vigil/perception/causality")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 0)

    # ── POST /vigil/perception/timeline ───────────────────────────────────

    def test_timeline_empty(self) -> None:
        """Timeline returns empty list when no events exist."""
        r = self.client.post(
            "/vigil/perception/timeline",
            json={},
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 0)

    def test_timeline_with_entity_filter(self) -> None:
        """Timeline accepts entity_id filter."""
        r = self.client.post(
            "/vigil/perception/timeline",
            json={"entity_id": 1},
        )
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json(), list)

    def test_timeline_with_zone_filter(self) -> None:
        """Timeline accepts zone_id filter."""
        r = self.client.post(
            "/vigil/perception/timeline",
            json={"zone_id": "restricted_1"},
        )
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json(), list)

    def test_timeline_with_event_type_filter(self) -> None:
        """Timeline accepts event_types filter."""
        r = self.client.post(
            "/vigil/perception/timeline",
            json={"event_types": ["zone_entered", "zone_exited"]},
        )
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json(), list)

    def test_timeline_with_time_filter(self) -> None:
        """Timeline accepts last_n_minutes filter."""
        r = self.client.post(
            "/vigil/perception/timeline",
            json={"last_n_minutes": 5.0},
        )
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json(), list)

    def test_timeline_with_all_filters(self) -> None:
        """Timeline accepts all filters combined."""
        r = self.client.post(
            "/vigil/perception/timeline",
            json={
                "entity_id": 1,
                "zone_id": "work_area",
                "event_types": ["entered"],
                "last_n_minutes": 10.0,
            },
        )
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json(), list)

    def test_timeline_with_events_returned(self) -> None:
        """Timeline returns events with correct schema."""
        self.engine._context_memory.get_timeline.return_value = [
            {
                "event_type": "zone_entered",
                "entity_id": 1,
                "timestamp": 15.5,
                "frame_number": 15,
                "details": {"zone_id": "restricted_1"},
            },
            {
                "event_type": "zone_exited",
                "entity_id": 1,
                "timestamp": 25.0,
                "frame_number": 25,
                "details": {"zone_id": "restricted_1", "duration_seconds": 9.5},
            },
        ]

        r = self.client.post(
            "/vigil/perception/timeline",
            json={"entity_id": 1},
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(len(data), 2)

        # Verify TimelineEvent schema
        for event in data:
            self.assertIn("event_type", event)
            self.assertIn("entity_id", event)
            self.assertIn("timestamp", event)
            self.assertIn("frame_number", event)
            self.assertIn("details", event)
            self.assertIsInstance(event["details"], dict)

        self.assertEqual(data[0]["event_type"], "zone_entered")
        self.assertEqual(data[0]["entity_id"], 1)
        self.assertEqual(data[1]["event_type"], "zone_exited")

    def test_timeline_invalid_event_types_ignored(self) -> None:
        """Timeline ignores invalid event type strings (no crash)."""
        r = self.client.post(
            "/vigil/perception/timeline",
            json={"event_types": ["nonexistent_event", "zone_entered"]},
        )
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json(), list)

    # ── GET /vigil/perception/state ───────────────────────────────────────

    def test_state_zero_frames(self) -> None:
        """State returns valid schema in zero-state."""
        r = self.client.get("/vigil/perception/state")
        self.assertEqual(r.status_code, 200)
        data = r.json()

        # Verify all PerceptionStateResponse fields
        expected_fields = [
            "frames_processed", "average_processing_ms",
            "active_tracks", "total_entities_seen",
            "total_violations", "components",
            "latest_narration_ja", "latest_narration_en",
        ]
        for field in expected_fields:
            self.assertIn(field, data, f"Missing field: {field}")

        # Zero-state values
        self.assertEqual(data["frames_processed"], 0)
        self.assertEqual(data["average_processing_ms"], 0.0)
        self.assertEqual(data["active_tracks"], 0)

        # Components dict should contain booleans
        self.assertIsInstance(data["components"], dict)
        component_keys = [
            "detector", "tracker", "scene_builder", "world_model",
            "trajectory_predictor", "activity_classifier",
            "attention_scorer", "causal_reasoner",
            "context_memory", "narrator",
        ]
        for key in component_keys:
            self.assertIn(key, data["components"], f"Missing component: {key}")
            self.assertIsInstance(data["components"][key], bool)

        # All components should be True (we set mocks for all)
        for key in component_keys:
            self.assertTrue(
                data["components"][key],
                f"Component {key} should be True but is {data['components'][key]}",
            )

    def test_state_with_world_state(self) -> None:
        """State includes active track count when world state exists."""
        world_state = MagicMock()
        world_state.active_tracks = {1: MagicMock(), 2: MagicMock(), 3: MagicMock()}
        self.engine.get_world_state.return_value = world_state

        self.engine.frames_processed = 100
        self.engine.average_processing_ms = 45.5

        # Context memory reports entities and violations
        summary = MagicMock()
        summary.unique_entities_seen = 5
        summary.total_violations = 3
        self.engine._context_memory.get_session_summary.return_value = summary

        # Narrator generates narration
        narration = MagicMock()
        narration.text_ja = "作業員3名がいます。"
        narration.text_en = "3 workers present."
        self.engine._narrator.narrate.return_value = narration

        r = self.client.get("/vigil/perception/state")
        self.assertEqual(r.status_code, 200)
        data = r.json()

        self.assertEqual(data["frames_processed"], 100)
        self.assertAlmostEqual(data["average_processing_ms"], 45.5, places=1)
        self.assertEqual(data["active_tracks"], 3)
        self.assertEqual(data["total_entities_seen"], 5)
        self.assertEqual(data["total_violations"], 3)
        self.assertEqual(data["latest_narration_ja"], "作業員3名がいます。")
        self.assertEqual(data["latest_narration_en"], "3 workers present.")

    def test_state_narration_null_when_no_world_state(self) -> None:
        """State narration fields are null when no world state."""
        r = self.client.get("/vigil/perception/state")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIsNone(data["latest_narration_ja"])
        self.assertIsNone(data["latest_narration_en"])

    def test_state_narration_null_when_narrator_fails(self) -> None:
        """State still returns 200 with null narration when narrator raises."""
        world_state = MagicMock()
        world_state.active_tracks = {}
        self.engine.get_world_state.return_value = world_state
        self.engine._narrator.narrate.side_effect = RuntimeError("Narrator error")

        r = self.client.get("/vigil/perception/state")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        # Should be null due to exception handling
        self.assertIsNone(data["latest_narration_ja"])
        self.assertIsNone(data["latest_narration_en"])


# ──────────────────────────────────────────────────────────────────────────────
# Tests: 400 error when perception backend is NOT active
# ──────────────────────────────────────────────────────────────────────────────


class TestPerceptionAPIWithoutBackend(unittest.TestCase):
    """Tests that perception endpoints return 400 when backend is not perception.

    Uses the default VLM client (mock without _engine attribute) so that
    _get_perception_engine() raises HTTPException(400).
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        os.environ["SOPILOT_DATA_DIR"] = str(self.root / "data")
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_PRIMARY_TASK_ID"] = "no-perception-test"
        os.environ["SOPILOT_RATE_LIMIT_RPM"] = "0"

        self.app = create_app()
        self.client = TestClient(self.app)

        # Inject a mock VLM that does NOT have _engine (simulates non-perception backend)
        mock_vlm = MagicMock(spec=[])  # empty spec: no attributes at all
        # Remove _engine if it exists via spec — MagicMock with empty spec
        # has no attributes by default
        self.app.state.vigil_pipeline._vlm = mock_vlm

    def tearDown(self) -> None:
        self._tmp.cleanup()
        for k in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND",
                   "SOPILOT_PRIMARY_TASK_ID", "SOPILOT_RATE_LIMIT_RPM"):
            os.environ.pop(k, None)

    def _assert_400_perception_not_active(self, response) -> None:
        """Helper to verify 400 response with perception not active message."""
        self.assertEqual(response.status_code, 400)
        detail = response.json().get("detail", "")
        self.assertIn("Perception engine not active", detail)

    def test_narration_400_no_perception(self) -> None:
        """POST /narration returns 400 without perception backend."""
        r = self.client.post(
            "/vigil/perception/narration",
            json={"style": "standard", "language": "ja"},
        )
        self._assert_400_perception_not_active(r)

    def test_query_400_no_perception(self) -> None:
        """POST /query returns 400 without perception backend."""
        r = self.client.post(
            "/vigil/perception/query",
            json={"question": "何人いる？"},
        )
        self._assert_400_perception_not_active(r)

    def test_summary_400_no_perception(self) -> None:
        """GET /summary returns 400 without perception backend."""
        r = self.client.get("/vigil/perception/summary")
        self._assert_400_perception_not_active(r)

    def test_entities_400_no_perception(self) -> None:
        """GET /entities/{id} returns 400 without perception backend."""
        r = self.client.get("/vigil/perception/entities/1")
        self._assert_400_perception_not_active(r)

    def test_activities_400_no_perception(self) -> None:
        """GET /activities returns 400 without perception backend."""
        r = self.client.get("/vigil/perception/activities")
        self._assert_400_perception_not_active(r)

    def test_predictions_400_no_perception(self) -> None:
        """GET /predictions returns 400 without perception backend."""
        r = self.client.get("/vigil/perception/predictions")
        self._assert_400_perception_not_active(r)

    def test_causality_400_no_perception(self) -> None:
        """GET /causality returns 400 without perception backend."""
        r = self.client.get("/vigil/perception/causality")
        self._assert_400_perception_not_active(r)

    def test_timeline_400_no_perception(self) -> None:
        """POST /timeline returns 400 without perception backend."""
        r = self.client.post(
            "/vigil/perception/timeline",
            json={},
        )
        self._assert_400_perception_not_active(r)

    def test_state_400_no_perception(self) -> None:
        """GET /state returns 400 without perception backend."""
        r = self.client.get("/vigil/perception/state")
        self._assert_400_perception_not_active(r)


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Missing component returns 404
# ──────────────────────────────────────────────────────────────────────────────


class TestPerceptionAPIMissingComponents(unittest.TestCase):
    """Tests that endpoints return 404 when required components are None.

    Simulates a PerceptionEngine where specific subcomponents are not
    initialized (set to None), verifying _require_component behavior.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        os.environ["SOPILOT_DATA_DIR"] = str(self.root / "data")
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_PRIMARY_TASK_ID"] = "missing-comp-test"
        os.environ["SOPILOT_RATE_LIMIT_RPM"] = "0"

        self.app = create_app()
        self.client = TestClient(self.app)

        # Set up perception engine with all components as None
        self.engine = MagicMock()
        self.engine.frames_processed = 0
        self.engine.average_processing_ms = 0.0
        self.engine.get_world_state.return_value = None
        self.engine._detector = None
        self.engine._tracker = None
        self.engine._scene_builder = None
        self.engine._world_model = None
        self.engine._trajectory_predictor = None
        self.engine._activity_classifier = None
        self.engine._attention_scorer = None
        self.engine._causal_reasoner = None
        self.engine._context_memory = None
        self.engine._narrator = None
        self.engine._zones = []

        mock_vlm = _build_mock_vlm_with_engine(self.engine)
        self.app.state.vigil_pipeline._vlm = mock_vlm

    def tearDown(self) -> None:
        self._tmp.cleanup()
        for k in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND",
                   "SOPILOT_PRIMARY_TASK_ID", "SOPILOT_RATE_LIMIT_RPM"):
            os.environ.pop(k, None)

    def test_narration_404_no_narrator(self) -> None:
        """Narration returns 404 when narrator component is None."""
        r = self.client.post(
            "/vigil/perception/narration",
            json={"style": "standard", "language": "ja"},
        )
        self.assertEqual(r.status_code, 404)
        self.assertIn("SceneNarrator", r.json()["detail"])

    def test_query_404_no_context_memory(self) -> None:
        """Query returns 404 when context memory is None."""
        r = self.client.post(
            "/vigil/perception/query",
            json={"question": "何人いる？"},
        )
        self.assertEqual(r.status_code, 404)
        self.assertIn("ContextMemory", r.json()["detail"])

    def test_summary_404_no_context_memory(self) -> None:
        """Summary returns 404 when context memory is None."""
        r = self.client.get("/vigil/perception/summary")
        self.assertEqual(r.status_code, 404)
        self.assertIn("ContextMemory", r.json()["detail"])

    def test_entity_404_no_context_memory(self) -> None:
        """Entity lookup returns 404 when context memory is None."""
        r = self.client.get("/vigil/perception/entities/1")
        self.assertEqual(r.status_code, 404)
        self.assertIn("ContextMemory", r.json()["detail"])

    def test_activities_404_no_classifier(self) -> None:
        """Activities returns 404 when activity classifier is None."""
        r = self.client.get("/vigil/perception/activities")
        self.assertEqual(r.status_code, 404)
        self.assertIn("ActivityClassifier", r.json()["detail"])

    def test_predictions_404_no_predictor(self) -> None:
        """Predictions returns 404 when trajectory predictor is None."""
        r = self.client.get("/vigil/perception/predictions")
        self.assertEqual(r.status_code, 404)
        self.assertIn("TrajectoryPredictor", r.json()["detail"])

    def test_causality_404_no_reasoner(self) -> None:
        """Causality returns 404 when causal reasoner is None."""
        r = self.client.get("/vigil/perception/causality")
        self.assertEqual(r.status_code, 404)
        self.assertIn("CausalReasoner", r.json()["detail"])

    def test_timeline_404_no_context_memory(self) -> None:
        """Timeline returns 404 when context memory is None."""
        r = self.client.post(
            "/vigil/perception/timeline",
            json={},
        )
        self.assertEqual(r.status_code, 404)
        self.assertIn("ContextMemory", r.json()["detail"])

    def test_state_works_without_components(self) -> None:
        """State endpoint works even when all components are None.

        The state endpoint does not use _require_component; it reads
        engine attributes directly and handles None gracefully.
        """
        r = self.client.get("/vigil/perception/state")
        self.assertEqual(r.status_code, 200)
        data = r.json()

        # All components should be False (None)
        for key in ["detector", "tracker", "scene_builder", "world_model",
                     "trajectory_predictor", "activity_classifier",
                     "attention_scorer", "causal_reasoner",
                     "context_memory", "narrator"]:
            self.assertFalse(
                data["components"][key],
                f"Component {key} should be False when set to None",
            )

        # Narration should be null
        self.assertIsNone(data["latest_narration_ja"])
        self.assertIsNone(data["latest_narration_en"])
        self.assertEqual(data["total_entities_seen"], 0)
        self.assertEqual(data["total_violations"], 0)


if __name__ == "__main__":
    unittest.main()
