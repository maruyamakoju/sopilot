"""Tests for Phase 40 — IncidentReporter + WebhookClient.

Covers:
    - IncidentReport dataclass
    - IncidentReporter: report_violation, deduplication, severity filter
    - IncidentReporter: get_recent_incidents, get_stats, clear_dedup_cache
    - IncidentReporter: template-based descriptions
    - IncidentReporter: LLaVA integration (mock)
    - IncidentReporter: persistence (save/load)
    - WebhookConfig: validation
    - WebhookClient: build_slack_payload, build_teams_payload, build_generic_payload
    - WebhookClient: test_webhook (network mocked)
    - WebhookClient: get_stats
    - build_webhook_client factory: auto-detect type
    - Thread safety
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sopilot.perception.incident_reporter import (
    IncidentReport,
    IncidentReporter,
    _get_template,
)
from sopilot.perception.webhook_client import (
    DeliveryResult,
    WebhookClient,
    WebhookConfig,
    build_webhook_client,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_violation(
    rule: str = "ヘルメット未着用",
    severity: str = "warning",
    entity_ids: list[int] | None = None,
):
    """Return a minimal Violation-like mock object."""
    from sopilot.perception.types import ViolationSeverity

    severity_map = {
        "info": ViolationSeverity.INFO,
        "warning": ViolationSeverity.WARNING,
        "critical": ViolationSeverity.CRITICAL,
    }
    v = MagicMock()
    v.rule = rule
    v.description_ja = ""
    v.severity = severity_map.get(severity, ViolationSeverity.WARNING)
    v.entity_ids = entity_ids or []
    return v


def make_frame(h: int = 80, w: int = 120) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# IncidentReport
# ---------------------------------------------------------------------------


class TestIncidentReport:
    def test_is_frozen(self):
        r = IncidentReport(
            incident_id="abc",
            timestamp=0.0,
            camera_id="cam01",
            violation_type="test",
            severity="warning",
            description_ja="テスト",
            description_en="test",
        )
        with pytest.raises((AttributeError, TypeError)):
            r.incident_id = "xyz"  # type: ignore[misc]

    def test_to_dict_keys(self):
        r = IncidentReport(
            incident_id="abc",
            timestamp=1.0,
            camera_id="cam01",
            violation_type="ヘルメット未着用",
            severity="warning",
            description_ja="テスト",
            description_en="test",
            entity_ids=(1, 2),
            recommendations_ja=("対応A",),
        )
        d = r.to_dict()
        assert d["incident_id"] == "abc"
        assert d["entity_ids"] == [1, 2]
        assert d["recommendations_ja"] == ["対応A"]

    def test_to_dict_lists_not_tuples(self):
        r = IncidentReport(
            incident_id="x",
            timestamp=0.0,
            camera_id="c",
            violation_type="v",
            severity="info",
            description_ja="",
            description_en="",
            entity_ids=(3,),
        )
        d = r.to_dict()
        assert isinstance(d["entity_ids"], list)


# ---------------------------------------------------------------------------
# _get_template
# ---------------------------------------------------------------------------


class TestGetTemplate:
    def test_helmet_keyword(self):
        t = _get_template("ヘルメット未着用")
        assert "ja" in t
        assert "recommendations" in t

    def test_restricted_area(self):
        t = _get_template("制限エリア侵入")
        assert "制限" in t["ja"] or len(t["ja"]) > 0

    def test_unknown_falls_back_to_default(self):
        t = _get_template("未知の違反XYZ")
        assert "ja" in t
        assert len(t["ja"]) > 0

    def test_fall_keyword(self):
        t = _get_template("転倒検出")
        assert "転倒" in t["ja"]

    def test_loitering_keyword(self):
        t = _get_template("徘徊行動検出")
        assert len(t["ja"]) > 0

    def test_vest_keyword(self):
        t = _get_template("ベスト未着用")
        assert "ベスト" in t["ja"]


# ---------------------------------------------------------------------------
# IncidentReporter — basic reporting
# ---------------------------------------------------------------------------


class TestIncidentReporterBasic:
    def test_creates_report(self):
        reporter = IncidentReporter(cooldown_seconds=0.0)
        v = make_violation("ヘルメット未着用", "warning")
        frame = make_frame()
        report = reporter.report_violation(v, frame, None, "cam01")
        assert report is not None
        assert isinstance(report, IncidentReport)

    def test_report_has_uuid(self):
        reporter = IncidentReporter(cooldown_seconds=0.0)
        report = reporter.report_violation(make_violation(), make_frame(), None, "cam01")
        assert len(report.incident_id) == 36  # UUID4

    def test_report_has_description_ja(self):
        reporter = IncidentReporter(cooldown_seconds=0.0)
        report = reporter.report_violation(make_violation("ヘルメット"), make_frame(), None, "cam01")
        assert len(report.description_ja) > 0

    def test_report_has_description_en(self):
        reporter = IncidentReporter(cooldown_seconds=0.0)
        report = reporter.report_violation(make_violation(), make_frame(), None, "cam01")
        assert len(report.description_en) > 0

    def test_report_camera_id(self):
        reporter = IncidentReporter(cooldown_seconds=0.0)
        report = reporter.report_violation(make_violation(), make_frame(), None, "camera_99")
        assert report.camera_id == "camera_99"

    def test_report_severity(self):
        reporter = IncidentReporter(cooldown_seconds=0.0, min_severity="info")
        report = reporter.report_violation(make_violation("test", "critical"), make_frame(), None, "c")
        assert report.severity == "critical"

    def test_report_violation_type(self):
        reporter = IncidentReporter(cooldown_seconds=0.0)
        report = reporter.report_violation(make_violation("制限エリア侵入"), make_frame(), None, "c")
        assert report.violation_type == "制限エリア侵入"

    def test_report_has_recommendations(self):
        reporter = IncidentReporter(cooldown_seconds=0.0)
        report = reporter.report_violation(make_violation("ヘルメット"), make_frame(), None, "c")
        assert len(report.recommendations_ja) > 0

    def test_report_stored_in_log(self):
        reporter = IncidentReporter(cooldown_seconds=0.0)
        reporter.report_violation(make_violation(), make_frame(), None, "c")
        reporter.report_violation(make_violation("ベスト"), make_frame(), None, "c")
        assert reporter.get_stats()["total_reports"] == 2

    def test_entity_ids_in_report(self):
        reporter = IncidentReporter(cooldown_seconds=0.0)
        v = make_violation("test", entity_ids=[1, 2, 3])
        report = reporter.report_violation(v, make_frame(), None, "c")
        assert set(report.entity_ids) == {1, 2, 3}

    def test_created_at_set(self):
        reporter = IncidentReporter(cooldown_seconds=0.0)
        report = reporter.report_violation(make_violation(), make_frame(), None, "c")
        assert len(report.created_at) > 0


# ---------------------------------------------------------------------------
# IncidentReporter — deduplication
# ---------------------------------------------------------------------------


class TestIncidentReporterDedup:
    def test_dedup_suppresses_duplicate(self):
        reporter = IncidentReporter(cooldown_seconds=60.0)
        v = make_violation("ヘルメット未着用")
        frame = make_frame()
        r1 = reporter.report_violation(v, frame, None, "cam01")
        r2 = reporter.report_violation(v, frame, None, "cam01")
        assert r1 is not None
        assert r2 is None  # suppressed

    def test_dedup_allows_different_type(self):
        reporter = IncidentReporter(cooldown_seconds=60.0)
        frame = make_frame()
        r1 = reporter.report_violation(make_violation("ヘルメット"), frame, None, "cam01")
        r2 = reporter.report_violation(make_violation("ベスト"), frame, None, "cam01")
        assert r1 is not None
        assert r2 is not None

    def test_dedup_allows_different_camera(self):
        reporter = IncidentReporter(cooldown_seconds=60.0)
        v = make_violation("ヘルメット")
        frame = make_frame()
        r1 = reporter.report_violation(v, frame, None, "cam01")
        r2 = reporter.report_violation(v, frame, None, "cam02")
        assert r1 is not None
        assert r2 is not None

    def test_clear_dedup_allows_retry(self):
        reporter = IncidentReporter(cooldown_seconds=60.0)
        v = make_violation("ヘルメット")
        frame = make_frame()
        reporter.report_violation(v, frame, None, "cam01")
        reporter.clear_dedup_cache()
        r2 = reporter.report_violation(v, frame, None, "cam01")
        assert r2 is not None

    def test_cooldown_zero_allows_repeat(self):
        reporter = IncidentReporter(cooldown_seconds=0.0)
        v = make_violation("ヘルメット")
        frame = make_frame()
        r1 = reporter.report_violation(v, frame, None, "cam01")
        r2 = reporter.report_violation(v, frame, None, "cam01")
        assert r1 is not None
        assert r2 is not None


# ---------------------------------------------------------------------------
# IncidentReporter — severity filter
# ---------------------------------------------------------------------------


class TestIncidentReporterSeverity:
    def test_info_filtered_when_min_warning(self):
        reporter = IncidentReporter(cooldown_seconds=0.0, min_severity="warning")
        r = reporter.report_violation(make_violation("test", "info"), make_frame(), None, "c")
        assert r is None

    def test_warning_passes_when_min_warning(self):
        reporter = IncidentReporter(cooldown_seconds=0.0, min_severity="warning")
        r = reporter.report_violation(make_violation("test", "warning"), make_frame(), None, "c")
        assert r is not None

    def test_critical_passes_when_min_warning(self):
        reporter = IncidentReporter(cooldown_seconds=0.0, min_severity="warning")
        r = reporter.report_violation(make_violation("test", "critical"), make_frame(), None, "c")
        assert r is not None

    def test_all_pass_when_min_info(self):
        reporter = IncidentReporter(cooldown_seconds=0.0, min_severity="info")
        for sev in ["info", "warning", "critical"]:
            r = reporter.report_violation(make_violation("test", sev), make_frame(), None, "c")
            assert r is not None

    def test_only_critical_when_min_critical(self):
        reporter = IncidentReporter(cooldown_seconds=0.0, min_severity="critical")
        ri = reporter.report_violation(make_violation("t1", "info"), make_frame(), None, "c")
        rw = reporter.report_violation(make_violation("t2", "warning"), make_frame(), None, "c")
        rc = reporter.report_violation(make_violation("t3", "critical"), make_frame(), None, "c")
        assert ri is None
        assert rw is None
        assert rc is not None


# ---------------------------------------------------------------------------
# IncidentReporter — query / stats
# ---------------------------------------------------------------------------


class TestIncidentReporterQuery:
    def test_get_recent_incidents_empty(self):
        reporter = IncidentReporter()
        assert reporter.get_recent_incidents() == []

    def test_get_recent_incidents_newest_first(self):
        reporter = IncidentReporter(cooldown_seconds=0.0)
        frame = make_frame()
        for i in range(5):
            reporter.report_violation(make_violation(f"violation_{i}"), frame, None, "c")
        incidents = reporter.get_recent_incidents(limit=5)
        # Most recent first
        times = [inc["timestamp"] for inc in incidents]
        assert times == sorted(times, reverse=True)

    def test_get_recent_incidents_limit(self):
        reporter = IncidentReporter(cooldown_seconds=0.0)
        frame = make_frame()
        for i in range(10):
            reporter.report_violation(make_violation(f"v_{i}"), frame, None, "c")
        incidents = reporter.get_recent_incidents(limit=3)
        assert len(incidents) == 3

    def test_get_stats_structure(self):
        reporter = IncidentReporter()
        s = reporter.get_stats()
        assert "total_reports" in s
        assert "log_size" in s
        assert "cooldown_seconds" in s
        assert "webhook_enabled" in s
        assert "llava_enabled" in s

    def test_get_stats_counts(self):
        reporter = IncidentReporter(cooldown_seconds=0.0)
        frame = make_frame()
        reporter.report_violation(make_violation("A"), frame, None, "c")
        reporter.report_violation(make_violation("B"), frame, None, "c")
        s = reporter.get_stats()
        assert s["total_reports"] == 2
        assert s["log_size"] == 2


# ---------------------------------------------------------------------------
# IncidentReporter — LLaVA integration
# ---------------------------------------------------------------------------


class TestIncidentReporterLLaVA:
    def test_llava_description_used(self):
        mock_llava = MagicMock()
        mock_llava.answer_question.return_value = "LLaVA説明文です"
        reporter = IncidentReporter(llava_client=mock_llava, cooldown_seconds=0.0)
        frame = make_frame()
        report = reporter.report_violation(make_violation("ヘルメット"), frame, None, "c")
        assert "LLaVA説明文です" in report.description_ja

    def test_llava_failure_falls_back_to_template(self):
        mock_llava = MagicMock()
        mock_llava.answer_question.side_effect = Exception("LLaVA failed")
        reporter = IncidentReporter(llava_client=mock_llava, cooldown_seconds=0.0)
        report = reporter.report_violation(make_violation("ヘルメット"), make_frame(), None, "c")
        assert report is not None  # fallback works
        assert len(report.description_ja) > 0

    def test_no_llava_uses_template(self):
        reporter = IncidentReporter(llava_client=None, cooldown_seconds=0.0)
        report = reporter.report_violation(make_violation("転倒"), make_frame(), None, "c")
        assert "転倒" in report.description_ja


# ---------------------------------------------------------------------------
# IncidentReporter — Persistence
# ---------------------------------------------------------------------------


class TestIncidentReporterPersistence:
    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "incidents.json"
            reporter = IncidentReporter(cooldown_seconds=0.0, persist_path=path)
            reporter.report_violation(make_violation(), make_frame(), None, "c")
            reporter._save()
            assert path.exists()

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "incidents.json"
            reporter = IncidentReporter(cooldown_seconds=0.0, persist_path=path)
            reporter.report_violation(make_violation("ヘルメット"), make_frame(), None, "cam01")
            reporter._save()

            reporter2 = IncidentReporter(cooldown_seconds=0.0, persist_path=path)
            assert reporter2.get_stats()["log_size"] == 1

    def test_load_nonexistent_file_ok(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "missing.json"
            reporter = IncidentReporter(persist_path=path)
            assert reporter.get_stats()["log_size"] == 0


# ---------------------------------------------------------------------------
# IncidentReporter — Webhook integration
# ---------------------------------------------------------------------------


class TestIncidentReporterWebhook:
    def test_webhook_deliver_called(self):
        mock_webhook = MagicMock()
        mock_webhook.deliver.return_value = [
            MagicMock(success=True)
        ]
        reporter = IncidentReporter(
            webhook_client=mock_webhook,
            cooldown_seconds=0.0,
        )
        reporter.report_violation(make_violation("ヘルメット"), make_frame(), None, "c")
        mock_webhook.deliver.assert_called_once()

    def test_webhook_failure_tracked(self):
        mock_webhook = MagicMock()
        mock_webhook.deliver.return_value = [
            MagicMock(success=False)
        ]
        reporter = IncidentReporter(
            webhook_client=mock_webhook,
            cooldown_seconds=0.0,
        )
        reporter.report_violation(make_violation("test"), make_frame(), None, "c")
        assert reporter.get_stats()["webhook_failures"] > 0


# ---------------------------------------------------------------------------
# WebhookConfig
# ---------------------------------------------------------------------------


class TestWebhookConfig:
    def test_valid_types(self):
        for t in ("slack", "teams", "generic"):
            cfg = WebhookConfig(url="https://example.com", type=t)
            assert cfg.type == t

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            WebhookConfig(url="https://example.com", type="invalid")

    def test_default_type_generic(self):
        cfg = WebhookConfig(url="https://example.com")
        assert cfg.type == "generic"


# ---------------------------------------------------------------------------
# WebhookClient — payload builders
# ---------------------------------------------------------------------------


class TestWebhookClientPayloads:
    def make_report(self):
        return IncidentReport(
            incident_id="test-id-1234",
            timestamp=1700000000.0,
            camera_id="cam01",
            violation_type="ヘルメット未着用",
            severity="warning",
            description_ja="ヘルメット未着用を検出",
            description_en="Helmet not worn",
            entity_ids=(1,),
            recommendations_ja=("装備を確認してください",),
            created_at="2024-01-01 12:00:00",
        )

    def test_slack_payload_has_text(self):
        client = WebhookClient([WebhookConfig("https://example.com", "slack")])
        payload = client._build_slack_payload(self.make_report())
        assert "text" in payload
        assert "warning" in payload["text"].upper() or "⚠" in payload["text"]

    def test_slack_payload_has_attachments(self):
        client = WebhookClient([WebhookConfig("https://example.com", "slack")])
        payload = client._build_slack_payload(self.make_report())
        assert "attachments" in payload
        assert len(payload["attachments"]) > 0

    def test_teams_payload_type(self):
        client = WebhookClient([WebhookConfig("https://example.com", "teams")])
        payload = client._build_teams_payload(self.make_report())
        assert payload.get("@type") == "MessageCard"

    def test_teams_payload_has_sections(self):
        client = WebhookClient([WebhookConfig("https://example.com", "teams")])
        payload = client._build_teams_payload(self.make_report())
        assert "sections" in payload

    def test_generic_payload_keys(self):
        client = WebhookClient([WebhookConfig("https://example.com")])
        payload = client._build_generic_payload(self.make_report())
        for key in ("incident_id", "violation_type", "severity", "description_ja"):
            assert key in payload

    def test_generic_payload_values(self):
        client = WebhookClient([WebhookConfig("https://example.com")])
        report = self.make_report()
        payload = client._build_generic_payload(report)
        assert payload["severity"] == "warning"
        assert payload["camera_id"] == "cam01"


# ---------------------------------------------------------------------------
# WebhookClient — stats / factory
# ---------------------------------------------------------------------------


class TestWebhookClientStats:
    def test_initial_stats(self):
        client = WebhookClient([WebhookConfig("https://example.com")])
        s = client.get_stats()
        assert s["target_count"] == 1
        assert s["deliveries_sent"] == 0
        assert s["failures"] == 0

    def test_no_targets(self):
        client = WebhookClient([])
        s = client.get_stats()
        assert s["target_count"] == 0


class TestBuildWebhookClient:
    def test_slack_auto_detect(self):
        client = build_webhook_client(["https://hooks.slack.com/services/T/B/X"])
        assert client._configs[0].type == "slack"

    def test_teams_auto_detect(self):
        client = build_webhook_client(["https://webhook.office.com/webhookb2/X"])
        assert client._configs[0].type == "teams"

    def test_generic_fallback(self):
        client = build_webhook_client(["https://myserver.example.com/hook"])
        assert client._configs[0].type == "generic"

    def test_multiple_urls(self):
        client = build_webhook_client([
            "https://hooks.slack.com/x",
            "https://myserver.com/hook",
        ])
        assert len(client._configs) == 2


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestIncidentReporterThreadSafety:
    def test_concurrent_reports(self):
        reporter = IncidentReporter(cooldown_seconds=0.0)
        frame = make_frame()
        reports = []
        errors = []

        def worker(i):
            try:
                v = make_violation(f"violation_{i}")
                r = reporter.report_violation(v, frame, None, "c")
                reports.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert reporter.get_stats()["total_reports"] == 20
