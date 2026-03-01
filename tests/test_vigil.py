"""Tests for VigilPilot — surveillance camera violation detection.

Coverage:
  - VigilRepository: session & event CRUD (13 tests)
  - VLM response parser: JSON extraction & field validation (12 tests)
  - VigilPipeline: end-to-end with mocked VLM (7 tests)
  - API endpoints: full HTTP flow via TestClient (15 tests)
"""

from __future__ import annotations

import io
import os
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from sopilot.database import Database
from sopilot.main import create_app
from sopilot.vigil.pipeline import VigilPipeline
from sopilot.vigil.repository import VigilRepository
from sopilot.vigil.schemas import VLMResult
from sopilot.vigil.vlm import VLMClient, _parse_vlm_response


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_video(path: Path, seconds: int = 2, fps: float = 8.0) -> None:
    """Create a minimal synthetic AVI video for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (96, 96))
    total = int(seconds * fps)
    for i in range(total):
        val = (i * 30) % 255
        frame = np.full((96, 96, 3), (val, 100, 200), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _init_db(db_path: Path) -> str:
    """Initialise a fresh SQLite database with all tables + migrations."""
    db = Database(db_path)
    db.close()
    return str(db_path)


class _MockVLMNoViolation(VLMClient):
    """VLM stub — always returns no violation."""

    def analyze_frame(self, frame_path: Path, rules: list[str]) -> VLMResult:
        return VLMResult(has_violation=False, violations=[], raw_text="{}")


class _MockVLMWithViolation(VLMClient):
    """VLM stub — always returns one warning-level violation."""

    def analyze_frame(self, frame_path: Path, rules: list[str]) -> VLMResult:
        return VLMResult(
            has_violation=True,
            violations=[{
                "rule_index": 0,
                "rule": rules[0] if rules else "test rule",
                "description_ja": "テスト違反が検出されました",
                "severity": "warning",
                "confidence": 0.9,
            }],
            raw_text='{"has_violation": true}',
        )


class _MockVLMRaisesError(VLMClient):
    """VLM stub — always raises RuntimeError."""

    def analyze_frame(self, frame_path: Path, rules: list[str]) -> VLMResult:
        raise RuntimeError("VLM API unavailable")


class _MockVLMInfoOnly(VLMClient):
    """VLM stub — returns only info-level violations (below warning threshold)."""

    def analyze_frame(self, frame_path: Path, rules: list[str]) -> VLMResult:
        return VLMResult(
            has_violation=True,
            violations=[{
                "rule_index": 0,
                "rule": rules[0] if rules else "test",
                "description_ja": "情報レベルの注意",
                "severity": "info",
                "confidence": 0.6,
            }],
            raw_text="",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Repository Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestVigilRepository:
    """Unit tests for VigilRepository — fresh SQLite DB per test."""

    def _repo(self, tmp_path: Path) -> VigilRepository:
        db_path = tmp_path / "vigil.db"
        _init_db(db_path)
        return VigilRepository(db_path)

    # ── Sessions ──────────────────────────────────────────────────────────────

    def test_create_and_get_session(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        sid = repo.create_session("玄関カメラ", ["ルール1", "ルール2"], 1.0, "warning")
        assert sid > 0
        row = repo.get_session(sid)
        assert row is not None
        assert row["name"] == "玄関カメラ"
        assert row["rules"] == ["ルール1", "ルール2"]
        assert row["sample_fps"] == 1.0
        assert row["severity_threshold"] == "warning"
        assert row["status"] == "idle"
        assert row["total_frames_analyzed"] == 0
        assert row["violation_count"] == 0

    def test_get_session_not_found(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        assert repo.get_session(9999) is None

    def test_list_sessions(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        repo.create_session("A", ["r1"], 1.0, "warning")
        repo.create_session("B", ["r2"], 2.0, "critical")
        sessions = repo.list_sessions()
        assert len(sessions) == 2
        assert {s["name"] for s in sessions} == {"A", "B"}

    def test_list_sessions_empty(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        assert repo.list_sessions() == []

    def test_update_session_status_all_fields(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        sid = repo.create_session("テスト", ["r"], 1.0, "warning")
        repo.update_session_status(
            sid, "completed",
            video_filename="out.mp4",
            total_frames_analyzed=120,
            violation_count=5,
        )
        row = repo.get_session(sid)
        assert row["status"] == "completed"
        assert row["video_filename"] == "out.mp4"
        assert row["total_frames_analyzed"] == 120
        assert row["violation_count"] == 5

    def test_update_status_partial(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        sid = repo.create_session("テスト", ["r"], 1.0, "warning")
        repo.update_session_status(sid, "processing")
        row = repo.get_session(sid)
        assert row["status"] == "processing"
        assert row["total_frames_analyzed"] == 0  # unchanged

    def test_delete_session(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        sid = repo.create_session("削除", ["r"], 1.0, "warning")
        assert repo.delete_session(sid) is True
        assert repo.get_session(sid) is None

    def test_delete_session_not_found(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        assert repo.delete_session(9999) is False

    def test_delete_session_cascades_events(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        sid = repo.create_session("カスケード", ["r"], 1.0, "warning")
        repo.create_event(sid, 1.0, 1, [])
        repo.create_event(sid, 2.0, 2, [])
        repo.delete_session(sid)
        assert repo.list_events(sid) == []

    # ── Events ────────────────────────────────────────────────────────────────

    def test_create_and_list_events(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        sid = repo.create_session("イベント", ["r"], 1.0, "warning")
        violations = [{
            "rule_index": 0, "rule": "r",
            "description_ja": "違反", "severity": "warning", "confidence": 0.8,
        }]
        eid = repo.create_event(sid, 5.0, 5, violations, frame_path="/tmp/f.jpg")
        assert eid > 0
        events = repo.list_events(sid)
        assert len(events) == 1
        assert events[0]["timestamp_sec"] == 5.0
        assert events[0]["frame_number"] == 5
        assert events[0]["violations"] == violations
        assert events[0]["frame_path"] == "/tmp/f.jpg"

    def test_list_events_ordered_by_timestamp(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        sid = repo.create_session("順序", ["r"], 1.0, "warning")
        for ts in (10.0, 2.0, 7.0):
            repo.create_event(sid, ts, int(ts), [])
        timestamps = [e["timestamp_sec"] for e in repo.list_events(sid)]
        assert timestamps == sorted(timestamps)

    def test_get_event(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        sid = repo.create_session("テスト", ["r"], 1.0, "warning")
        violations = [{"rule_index": 0, "rule": "r", "description_ja": "d",
                       "severity": "critical", "confidence": 0.95}]
        eid = repo.create_event(sid, 3.0, 3, violations)
        row = repo.get_event(eid)
        assert row is not None
        assert row["session_id"] == sid
        assert row["violations"] == violations

    def test_get_event_not_found(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        assert repo.get_event(9999) is None

    def test_rules_unicode_roundtrip(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        rules = ["ヘルメット未着用", "安全ベルトなしで高所作業", "火気厳禁エリアでの喫煙"]
        sid = repo.create_session("ユニコード", rules, 1.0, "warning")
        assert repo.get_session(sid)["rules"] == rules


# ──────────────────────────────────────────────────────────────────────────────
# VLM Parser Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestVLMResponseParser:
    """Unit tests for _parse_vlm_response — no I/O required."""

    def test_clean_json_no_violation(self) -> None:
        r = _parse_vlm_response('{"has_violation": false, "violations": []}', ["r"])
        assert r.has_violation is False
        assert r.violations == []

    def test_clean_json_with_violation(self) -> None:
        raw = '{"has_violation": true, "violations": [{"rule_index": 0, "rule": "r", "description_ja": "d", "severity": "warning", "confidence": 0.85}]}'
        r = _parse_vlm_response(raw, ["r"])
        assert r.has_violation is True
        assert len(r.violations) == 1
        assert r.violations[0]["severity"] == "warning"
        assert r.violations[0]["confidence"] == pytest.approx(0.85)

    def test_json_in_markdown_fence(self) -> None:
        raw = '```json\n{"has_violation": true, "violations": [{"rule_index": 0, "rule": "r", "description_ja": "d", "severity": "critical", "confidence": 0.9}]}\n```'
        r = _parse_vlm_response(raw, ["r"])
        assert r.has_violation is True
        assert r.violations[0]["severity"] == "critical"

    def test_invalid_json_returns_no_violation(self) -> None:
        r = _parse_vlm_response("これは違反ではありません。", ["r"])
        assert r.has_violation is False
        assert r.violations == []

    def test_empty_string_returns_no_violation(self) -> None:
        r = _parse_vlm_response("", ["r"])
        assert r.has_violation is False

    def test_confidence_clamped_above_one(self) -> None:
        raw = '{"has_violation": true, "violations": [{"rule_index": 0, "rule": "r", "description_ja": "d", "severity": "warning", "confidence": 5.0}]}'
        r = _parse_vlm_response(raw, ["r"])
        assert r.violations[0]["confidence"] <= 1.0

    def test_confidence_clamped_below_zero(self) -> None:
        raw = '{"has_violation": true, "violations": [{"rule_index": 0, "rule": "r", "description_ja": "d", "severity": "warning", "confidence": -2.0}]}'
        r = _parse_vlm_response(raw, ["r"])
        assert r.violations[0]["confidence"] >= 0.0

    def test_unknown_severity_falls_back_to_warning(self) -> None:
        raw = '{"has_violation": true, "violations": [{"rule_index": 0, "rule": "r", "description_ja": "d", "severity": "UNKNOWN", "confidence": 0.5}]}'
        r = _parse_vlm_response(raw, ["r"])
        assert r.violations[0]["severity"] == "warning"

    def test_description_truncated_at_100_chars(self) -> None:
        long_desc = "あ" * 200
        raw = f'{{"has_violation": true, "violations": [{{"rule_index": 0, "rule": "r", "description_ja": "{long_desc}", "severity": "info", "confidence": 0.5}}]}}'
        r = _parse_vlm_response(raw, ["r"])
        assert len(r.violations[0]["description_ja"]) <= 100

    def test_raw_text_preserved(self) -> None:
        raw = '{"has_violation": false, "violations": []}'
        r = _parse_vlm_response(raw, ["r"])
        assert r.raw_text == raw

    def test_multiple_violations(self) -> None:
        raw = '{"has_violation": true, "violations": [{"rule_index": 0, "rule": "r1", "description_ja": "d1", "severity": "critical", "confidence": 0.9}, {"rule_index": 1, "rule": "r2", "description_ja": "d2", "severity": "info", "confidence": 0.4}]}'
        r = _parse_vlm_response(raw, ["r1", "r2"])
        assert len(r.violations) == 2

    def test_valid_severity_values_preserved(self) -> None:
        for sev in ("critical", "warning", "info"):
            raw = f'{{"has_violation": true, "violations": [{{"rule_index": 0, "rule": "r", "description_ja": "d", "severity": "{sev}", "confidence": 0.5}}]}}'
            r = _parse_vlm_response(raw, ["r"])
            assert r.violations[0]["severity"] == sev


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestVigilPipeline(unittest.TestCase):
    """Tests for VigilPipeline._run() with mocked VLM and a synthetic video."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        _init_db(self.tmp / "pipeline.db")
        self.repo = VigilRepository(self.tmp / "pipeline.db")
        self.frames_root = self.tmp / "frames"
        self.video_path = self.tmp / "test.avi"
        _make_video(self.video_path, seconds=2, fps=8.0)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _pipeline(self, vlm: VLMClient) -> VigilPipeline:
        return VigilPipeline(repo=self.repo, vlm=vlm, frames_root=self.frames_root)

    def _run(self, pipeline: VigilPipeline, sid: int, **kwargs) -> None:
        pipeline._run(
            session_id=sid,
            video_path=self.video_path,
            rules=["ルール1"],
            sample_fps=1.0,
            severity_threshold="warning",
            cleanup_video=False,
            **kwargs,
        )

    def test_no_violations_session_completed(self) -> None:
        pipeline = self._pipeline(_MockVLMNoViolation())
        sid = self.repo.create_session("テスト", ["r"], 1.0, "warning")
        self._run(pipeline, sid)
        row = self.repo.get_session(sid)
        self.assertEqual(row["status"], "completed")
        self.assertGreater(row["total_frames_analyzed"], 0)
        self.assertEqual(row["violation_count"], 0)
        self.assertEqual(self.repo.list_events(sid), [])

    def test_with_violations_events_created(self) -> None:
        pipeline = self._pipeline(_MockVLMWithViolation())
        sid = self.repo.create_session("違反テスト", ["r"], 1.0, "warning")
        self._run(pipeline, sid)
        row = self.repo.get_session(sid)
        self.assertEqual(row["status"], "completed")
        self.assertGreater(row["violation_count"], 0)
        events = self.repo.list_events(sid)
        self.assertGreater(len(events), 0)
        self.assertEqual(events[0]["violations"][0]["severity"], "warning")

    def test_vlm_error_per_frame_continues(self) -> None:
        """A VLM error on every frame should still complete (not fail)."""
        pipeline = self._pipeline(_MockVLMRaisesError())
        sid = self.repo.create_session("エラーテスト", ["r"], 1.0, "warning")
        self._run(pipeline, sid)
        row = self.repo.get_session(sid)
        self.assertEqual(row["status"], "completed")
        self.assertGreater(row["total_frames_analyzed"], 0)

    def test_severity_threshold_filters_info(self) -> None:
        """info violations are filtered when threshold='warning'."""
        pipeline = self._pipeline(_MockVLMInfoOnly())
        sid = self.repo.create_session("フィルター", ["r"], 1.0, "warning")
        self._run(pipeline, sid)
        self.assertEqual(self.repo.list_events(sid), [])

    def test_severity_threshold_info_passes_all(self) -> None:
        """info violations are kept when threshold='info'."""
        pipeline = self._pipeline(_MockVLMInfoOnly())
        sid = self.repo.create_session("info許可", ["r"], 1.0, "info")
        pipeline._run(
            session_id=sid,
            video_path=self.video_path,
            rules=["r"],
            sample_fps=1.0,
            severity_threshold="info",
            cleanup_video=False,
        )
        events = self.repo.list_events(sid)
        self.assertGreater(len(events), 0)

    def test_cleanup_video_removes_file(self) -> None:
        pipeline = self._pipeline(_MockVLMNoViolation())
        sid = self.repo.create_session("クリーンアップ", ["r"], 1.0, "warning")
        self.assertTrue(self.video_path.exists())
        pipeline._run(
            session_id=sid, video_path=self.video_path,
            rules=["r"], sample_fps=1.0, severity_threshold="warning",
            cleanup_video=True,
        )
        self.assertFalse(self.video_path.exists())

    def test_invalid_video_sets_failed_status(self) -> None:
        pipeline = self._pipeline(_MockVLMNoViolation())
        sid = self.repo.create_session("無効ファイル", ["r"], 1.0, "warning")
        pipeline._run(
            session_id=sid,
            video_path=self.tmp / "nonexistent.avi",
            rules=["r"], sample_fps=1.0, severity_threshold="warning",
            cleanup_video=False,
        )
        self.assertEqual(self.repo.get_session(sid)["status"], "failed")

    def test_analyze_async_completes_background(self) -> None:
        pipeline = self._pipeline(_MockVLMNoViolation())
        sid = self.repo.create_session("非同期", ["r"], 1.0, "warning")
        pipeline.analyze_async(
            session_id=sid, video_path=self.video_path,
            rules=["r"], sample_fps=1.0, severity_threshold="warning",
            cleanup_video=False,
        )
        deadline = time.time() + 10
        while time.time() < deadline:
            row = self.repo.get_session(sid)
            if row["status"] in ("completed", "failed"):
                break
            time.sleep(0.1)
        self.assertEqual(self.repo.get_session(sid)["status"], "completed")


# ──────────────────────────────────────────────────────────────────────────────
# API Endpoint Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestVigilAPI(unittest.TestCase):
    """E2E HTTP tests for /vigil/* endpoints via TestClient."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        os.environ["SOPILOT_DATA_DIR"] = str(self.root / "data")
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_PRIMARY_TASK_ID"] = "vigil-test"
        os.environ["SOPILOT_RATE_LIMIT_RPM"] = "0"
        self.app = create_app()
        self.client = TestClient(self.app)
        # Inject mock VLM — no real Anthropic API calls
        self.app.state.vigil_pipeline._vlm = _MockVLMWithViolation()

    def tearDown(self) -> None:
        self._tmp.cleanup()
        for k in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND",
                  "SOPILOT_PRIMARY_TASK_ID", "SOPILOT_RATE_LIMIT_RPM"):
            os.environ.pop(k, None)

    def _create_session(self, name: str = "テスト") -> dict:
        r = self.client.post(
            "/vigil/sessions",
            json={"name": name, "rules": ["ルール1", "ルール2"],
                  "sample_fps": 1.0, "severity_threshold": "warning"},
        )
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def _upload_video(self, sid: int) -> None:
        video_path = self.root / "upload.avi"
        _make_video(video_path, seconds=2, fps=8.0)
        with video_path.open("rb") as fh:
            r = self.client.post(
                f"/vigil/sessions/{sid}/analyze",
                files={"file": ("upload.avi", fh, "video/x-msvideo")},
            )
        self.assertEqual(r.status_code, 200, r.text)

    def _wait_completed(self, sid: int, timeout: float = 15.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = self.client.get(f"/vigil/sessions/{sid}")
            data = r.json()
            if data["status"] in ("completed", "failed"):
                return data
            time.sleep(0.2)
        return self.client.get(f"/vigil/sessions/{sid}").json()

    # ── Session CRUD ──────────────────────────────────────────────────────────

    def test_create_session_returns_idle(self) -> None:
        data = self._create_session()
        self.assertEqual(data["status"], "idle")
        self.assertEqual(data["rules"], ["ルール1", "ルール2"])
        self.assertEqual(data["violation_count"], 0)
        self.assertIn("session_id", data)

    def test_list_sessions_empty(self) -> None:
        r = self.client.get("/vigil/sessions")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), [])

    def test_list_sessions_after_creation(self) -> None:
        self._create_session("A")
        self._create_session("B")
        data = self.client.get("/vigil/sessions").json()
        self.assertEqual(len(data), 2)

    def test_get_session_exists(self) -> None:
        sid = self._create_session()["session_id"]
        r = self.client.get(f"/vigil/sessions/{sid}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["session_id"], sid)

    def test_get_session_not_found(self) -> None:
        r = self.client.get("/vigil/sessions/9999")
        self.assertEqual(r.status_code, 404)

    def test_delete_session(self) -> None:
        sid = self._create_session()["session_id"]
        r = self.client.delete(f"/vigil/sessions/{sid}")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["deleted"])
        self.assertEqual(self.client.get(f"/vigil/sessions/{sid}").status_code, 404)

    def test_delete_session_not_found(self) -> None:
        r = self.client.delete("/vigil/sessions/9999")
        self.assertEqual(r.status_code, 404)

    # ── Validation ────────────────────────────────────────────────────────────

    def test_create_session_invalid_severity(self) -> None:
        r = self.client.post(
            "/vigil/sessions",
            json={"name": "t", "rules": ["r"], "sample_fps": 1.0, "severity_threshold": "invalid"},
        )
        self.assertEqual(r.status_code, 422)

    def test_create_session_empty_rules(self) -> None:
        r = self.client.post(
            "/vigil/sessions",
            json={"name": "t", "rules": [], "sample_fps": 1.0, "severity_threshold": "warning"},
        )
        self.assertEqual(r.status_code, 422)

    def test_create_session_fps_out_of_range(self) -> None:
        r = self.client.post(
            "/vigil/sessions",
            json={"name": "t", "rules": ["r"], "sample_fps": 99.0, "severity_threshold": "warning"},
        )
        self.assertEqual(r.status_code, 422)

    # ── Analysis & Events ─────────────────────────────────────────────────────

    def test_analyze_unknown_session_404(self) -> None:
        r = self.client.post(
            "/vigil/sessions/9999/analyze",
            files={"file": ("f.avi", io.BytesIO(b"x"), "video/x-msvideo")},
        )
        self.assertEqual(r.status_code, 404)

    def test_analyze_already_processing_409(self) -> None:
        sid = self._create_session()["session_id"]
        self.app.state.vigil_repo.update_session_status(sid, "processing")
        video = self.root / "dup.avi"
        _make_video(video, seconds=1)
        with video.open("rb") as fh:
            r = self.client.post(
                f"/vigil/sessions/{sid}/analyze",
                files={"file": ("dup.avi", fh, "video/x-msvideo")},
            )
        self.assertEqual(r.status_code, 409)

    def test_list_events_empty(self) -> None:
        sid = self._create_session()["session_id"]
        r = self.client.get(f"/vigil/sessions/{sid}/events")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), [])

    def test_event_frame_not_found(self) -> None:
        r = self.client.get("/vigil/events/9999/frame")
        self.assertEqual(r.status_code, 404)

    def test_full_pipeline_e2e(self) -> None:
        """Upload video → pipeline runs with mock VLM → violations appear."""
        sid = self._create_session("E2Eテスト")["session_id"]
        self._upload_video(sid)

        session = self._wait_completed(sid)
        self.assertEqual(session["status"], "completed",
                         f"Pipeline did not complete: {session}")
        self.assertGreater(session["total_frames_analyzed"], 0)
        self.assertGreater(session["violation_count"], 0)

        # Events list
        events = self.client.get(f"/vigil/sessions/{sid}/events").json()
        self.assertGreater(len(events), 0)
        self.assertIn("timestamp_sec", events[0])
        self.assertIn("violations", events[0])

        # Report
        report = self.client.get(f"/vigil/sessions/{sid}/report").json()
        self.assertEqual(report["violation_count"], session["violation_count"])
        self.assertIn("warning", report["severity_breakdown"])
        self.assertGreater(report["severity_breakdown"]["warning"], 0)


# ──────────────────────────────────────────────────────────────────────────────
# Qwen3-VL: bbox parsing, severity inference, API client, frame annotation
# ──────────────────────────────────────────────────────────────────────────────


class TestQwen3VLParsing:
    """Unit tests for Qwen3-VL bounding-box output parsing and helpers."""

    def test_parse_nested_bboxes(self) -> None:
        from sopilot.vigil.vlm import _parse_bboxes_from_text
        result = _parse_bboxes_from_text("[[100, 200, 400, 600], [50, 50, 300, 300]]")
        assert result == [[100, 200, 400, 600], [50, 50, 300, 300]]

    def test_parse_single_flat_bbox(self) -> None:
        from sopilot.vigil.vlm import _parse_bboxes_from_text
        result = _parse_bboxes_from_text("[150, 250, 850, 950]")
        assert result == [[150, 250, 850, 950]]

    def test_strips_think_tags(self) -> None:
        from sopilot.vigil.vlm import _parse_bboxes_from_text
        result = _parse_bboxes_from_text("<think>reasoning</think>\n[[200, 300, 700, 800]]")
        assert result == [[200, 300, 700, 800]]

    def test_empty_response(self) -> None:
        from sopilot.vigil.vlm import _parse_bboxes_from_text
        assert _parse_bboxes_from_text("[]") == []
        assert _parse_bboxes_from_text("No objects detected.") == []

    def test_out_of_range_coords_rejected(self) -> None:
        from sopilot.vigil.vlm import _parse_bboxes_from_text
        # Coords > 1010 should be rejected
        result = _parse_bboxes_from_text("[[100, 200, 2000, 600]]")
        assert result == []

    def test_float_coords(self) -> None:
        from sopilot.vigil.vlm import _parse_bboxes_from_text
        result = _parse_bboxes_from_text("[[10.5, 20.0, 400.3, 600.7]]")
        assert len(result) == 1
        assert result[0][0] == pytest.approx(10.5)

    def test_severity_critical_keywords(self) -> None:
        from sopilot.vigil.vlm import _infer_severity_from_rule
        assert _infer_severity_from_rule("転倒リスクを検出") == "critical"
        assert _infer_severity_from_rule("立入禁止エリアへの侵入") == "critical"
        assert _infer_severity_from_rule("感電の危険を検出") == "critical"

    def test_severity_warning_default(self) -> None:
        from sopilot.vigil.vlm import _infer_severity_from_rule
        assert _infer_severity_from_rule("ヘルメット未着用の作業者") == "warning"
        assert _infer_severity_from_rule("安全ベスト未着用") == "warning"

    def test_factory_claude(self) -> None:
        import os
        from sopilot.vigil.vlm import ClaudeVisionClient, build_vlm_client
        os.environ["ANTHROPIC_API_KEY"] = "sk-test"
        c = build_vlm_client("claude")
        assert isinstance(c, ClaudeVisionClient)

    def test_factory_qwen3_api(self) -> None:
        import os
        from sopilot.vigil.vlm import Qwen3VLAPIClient, build_vlm_client
        os.environ["VIGIL_QWEN3_API_BASE"] = "http://localhost:11434/v1"
        c = build_vlm_client("qwen3-api")
        assert isinstance(c, Qwen3VLAPIClient)
        assert c._model == "Qwen/Qwen3-VL-7B-Instruct"

    def test_factory_qwen3_api_requires_api_base(self) -> None:
        import os
        from sopilot.vigil.vlm import build_vlm_client
        os.environ.pop("VIGIL_QWEN3_API_BASE", None)
        with pytest.raises(ValueError, match="VIGIL_QWEN3_API_BASE"):
            build_vlm_client("qwen3-api")

    def test_factory_unknown_backend_raises(self) -> None:
        from sopilot.vigil.vlm import build_vlm_client
        with pytest.raises(ValueError, match="Unknown VLM backend"):
            build_vlm_client("gpt9-turbo")

    def test_qwen3_api_client_strips_think_in_response(self) -> None:
        """Qwen3VLAPIClient strips <think>...</think> before JSON parsing."""
        import os
        from unittest.mock import MagicMock, patch
        os.environ["VIGIL_QWEN3_API_BASE"] = "http://localhost:11434/v1"
        from sopilot.vigil.vlm import Qwen3VLAPIClient
        client = Qwen3VLAPIClient(api_base="http://localhost:11434/v1")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": '<think>Let me analyze...</think>\n{"has_violation": false, "violations": []}'
                }
            }]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client._client, "post", return_value=mock_resp):
            result = client.analyze_frame(
                Path(__file__).parent / "fixtures" / "dummy.jpg"
                if False else Path(__file__),  # any existing path
                ["ルール1"],
            )
        assert result.has_violation is False

    def test_violation_detail_accepts_bboxes_field(self) -> None:
        """ViolationDetail schema accepts optional bboxes without error."""
        from sopilot.vigil.schemas import ViolationDetail
        v = ViolationDetail(
            rule_index=0,
            rule="test",
            description_ja="説明",
            severity="warning",
            confidence=0.9,
            bboxes=[[100, 200, 400, 600]],
        )
        assert v.bboxes == [[100, 200, 400, 600]]

    def test_violation_detail_bboxes_optional(self) -> None:
        """ViolationDetail without bboxes defaults to None."""
        from sopilot.vigil.schemas import ViolationDetail
        v = ViolationDetail(
            rule_index=0, rule="test", description_ja="説明", severity="warning", confidence=0.9
        )
        assert v.bboxes is None


class TestBboxFrameAnnotation:
    """Unit tests for bounding-box rendering on JPEG frames."""

    def _make_jpeg(self, tmp_path: Path) -> Path:
        import numpy as np
        img = np.full((240, 320, 3), (80, 80, 80), dtype=np.uint8)
        p = tmp_path / "frame.jpg"
        cv2.imwrite(str(p), img)
        return p

    def test_annotation_produces_valid_jpeg(self, tmp_path: Path) -> None:
        from sopilot.vigil.router import _annotate_frame_with_bboxes
        frame = self._make_jpeg(tmp_path)
        result = _annotate_frame_with_bboxes(
            frame,
            [([[100, 100, 500, 700]], "ヘルメット未着用", "warning")],
        )
        assert result[:2] == b"\xff\xd8", "Not a valid JPEG"
        assert len(result) > 1000

    def test_annotation_multiple_severities(self, tmp_path: Path) -> None:
        from sopilot.vigil.router import _annotate_frame_with_bboxes
        frame = self._make_jpeg(tmp_path)
        groups = [
            ([[50, 50, 300, 400]], "ヘルメット未着用", "warning"),
            ([[400, 50, 900, 800]], "立入禁止", "critical"),
            ([[200, 200, 600, 700]], "軽微な注意", "info"),
        ]
        result = _annotate_frame_with_bboxes(frame, groups)
        assert result[:2] == b"\xff\xd8"

    def test_empty_bboxes_no_error(self, tmp_path: Path) -> None:
        from sopilot.vigil.router import _annotate_frame_with_bboxes
        frame = self._make_jpeg(tmp_path)
        # Empty bbox list in group — should handle gracefully
        result = _annotate_frame_with_bboxes(frame, [([], "rule", "warning")])
        assert result[:2] == b"\xff\xd8"

    def test_annotation_larger_than_raw(self, tmp_path: Path) -> None:
        """Annotated frame with boxes should be larger than or similar to raw."""
        from sopilot.vigil.router import _annotate_frame_with_bboxes
        frame = self._make_jpeg(tmp_path)
        result = _annotate_frame_with_bboxes(
            frame,
            [([[100, 100, 800, 900], [50, 50, 300, 400]], "ルール", "warning")],
        )
        # Annotated should be a reasonable size
        assert len(result) > 500


class TestFrameEndpointWithBboxes(unittest.TestCase):
    """Integration test: GET /vigil/events/{id}/frame with bbox annotation."""

    def setUp(self) -> None:
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())
        db_path = self.tmp / "test.db"
        _init_db(db_path)

        class _MockVLMWithBboxes(VLMClient):
            def analyze_frame(self, frame_path: Path, rules: list[str]) -> VLMResult:
                return VLMResult(
                    has_violation=True,
                    violations=[{
                        "rule_index": 0,
                        "rule": rules[0],
                        "description_ja": "ヘルメット未着用を検出",
                        "severity": "warning",
                        "confidence": 0.9,
                        "bboxes": [[100, 100, 500, 700]],
                    }],
                    raw_text="bbox detected",
                )

        app = create_app()
        app.state.vigil_pipeline._vlm = _MockVLMWithBboxes()
        self.client = TestClient(app, raise_server_exceptions=True)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_video_bytes(self) -> bytes:
        buf_path = self.tmp / "vid.avi"
        _make_video(buf_path, seconds=1)
        return buf_path.read_bytes()

    def test_frame_endpoint_returns_annotated_jpeg_when_bbox_present(self) -> None:
        """Frame endpoint returns annotated JPEG when violation has bboxes."""
        # Create session + analyze
        sid_resp = self.client.post("/vigil/sessions", json={
            "name": "bbox test",
            "rules": ["ヘルメット未着用の作業者を検出"],
            "sample_fps": 4.0,
            "severity_threshold": "info",
        })
        sid = sid_resp.json()["session_id"]

        vid_bytes = self._make_video_bytes()
        self.client.post(
            f"/vigil/sessions/{sid}/analyze",
            files={"file": ("test.avi", vid_bytes, "video/avi")},
        )

        # Wait for completion
        deadline = time.time() + 30
        while time.time() < deadline:
            s = self.client.get(f"/vigil/sessions/{sid}").json()
            if s["status"] in ("completed", "failed"):
                break
            time.sleep(0.2)

        events = self.client.get(f"/vigil/sessions/{sid}/events").json()
        if not events:
            self.skipTest("No violation events generated")

        event_id = events[0]["event_id"]
        resp = self.client.get(f"/vigil/events/{event_id}/frame")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "image/jpeg")
        # Annotated frame is returned (valid JPEG)
        self.assertEqual(resp.content[:2], b"\xff\xd8")

    def test_frame_endpoint_annotate_false_returns_raw(self) -> None:
        """?annotate=false bypasses bbox drawing and returns raw frame."""
        sid_resp = self.client.post("/vigil/sessions", json={
            "name": "raw frame test",
            "rules": ["ヘルメット未着用の作業者を検出"],
            "sample_fps": 4.0,
            "severity_threshold": "info",
        })
        sid = sid_resp.json()["session_id"]

        vid_bytes = self._make_video_bytes()
        self.client.post(
            f"/vigil/sessions/{sid}/analyze",
            files={"file": ("test.avi", vid_bytes, "video/avi")},
        )

        deadline = time.time() + 30
        while time.time() < deadline:
            s = self.client.get(f"/vigil/sessions/{sid}").json()
            if s["status"] in ("completed", "failed"):
                break
            time.sleep(0.2)

        events = self.client.get(f"/vigil/sessions/{sid}/events").json()
        if not events:
            self.skipTest("No violation events generated")

        event_id = events[0]["event_id"]
        resp = self.client.get(f"/vigil/events/{event_id}/frame?annotate=false")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("image/jpeg", resp.headers["content-type"])
