"""Incident Intelligence — autonomous incident report generation.

Phase 40: Closes the loop between detection and communication.
When a violation is detected, the IncidentReporter:

    1. Deduplicates (suppresses duplicate reports within cooldown window)
    2. Generates a natural-language description via LLaVA-NeXT (if available)
       or falls back to rule-based Japanese templates
    3. Creates a structured IncidentReport with evidence
    4. Delivers via WebhookClient to Slack/Teams/generic endpoints
    5. Stores in a rolling incident log (JSON-persisted, 200 entries max)

Architecture:

    ViolationDetected → IncidentReporter.report_violation()
        ├── dedup check (cooldown per violation type + camera)
        ├── LLaVA describe_scene() → description_ja
        ├── IncidentReport(incident_id, timestamp, description_ja, ...)
        ├── WebhookClient.deliver(report) → async best-effort
        └── persist to JSON ring buffer

Integration in engine.py::

    # After Stage 6 violations
    if self._incident_reporter and violations:
        for v in violations:
            self._incident_reporter.report_violation(v, frame, world_state, "cam01")
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_MAX_LOG_ENTRIES = 200
_SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}

# ---------------------------------------------------------------------------
# IncidentReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IncidentReport:
    """A structured, self-contained safety incident report.

    Produced by :class:`IncidentReporter` for every unique violation.
    Delivered via :class:`WebhookClient` and stored in the incident log.
    """

    incident_id: str          # UUID4 unique identifier
    timestamp: float          # Unix epoch (seconds)
    camera_id: str            # originating camera/session ID
    violation_type: str       # e.g. "ヘルメット未着用"
    severity: str             # "info", "warning", "critical"
    description_ja: str       # Japanese natural-language description
    description_en: str       # English description
    entity_ids: tuple[int, ...] = ()       # involved track IDs
    recommendations_ja: tuple[str, ...] = ()  # Japanese action recommendations
    evidence_detections: tuple[str, ...] = ()  # detection labels involved
    created_at: str = ""      # ISO8601 creation time

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["entity_ids"] = list(self.entity_ids)
        d["recommendations_ja"] = list(self.recommendations_ja)
        d["evidence_detections"] = list(self.evidence_detections)
        return d


# ---------------------------------------------------------------------------
# Dedup key
# ---------------------------------------------------------------------------


def _dedup_key(violation_type: str, camera_id: str) -> str:
    return f"{camera_id}::{violation_type}"


# ---------------------------------------------------------------------------
# Rule-based description templates
# ---------------------------------------------------------------------------

_VIOLATION_TEMPLATES: dict[str, dict[str, Any]] = {
    # Japanese violation type keywords → template
    "ヘルメット": {
        "ja": "作業員がヘルメット未着用で作業エリアに存在しています。安全規則に違反しています。",
        "en": "Worker detected without helmet in the work area. Safety rule violation.",
        "recommendations": [
            "直ちにヘルメットを着用させてください。",
            "作業を一時停止し、安全装備を確認してください。",
        ],
    },
    "ベスト": {
        "ja": "安全ベストを着用していない作業員が確認されました。",
        "en": "Worker without safety vest detected.",
        "recommendations": [
            "安全ベストを着用するよう指示してください。",
        ],
    },
    "制限": {
        "ja": "権限のない人物が制限エリアへ侵入しました。",
        "en": "Unauthorized entry into restricted area detected.",
        "recommendations": [
            "直ちに退出を指示してください。",
            "アクセスログを確認し、管理者に報告してください。",
        ],
    },
    "転倒": {
        "ja": "人物の転倒が検出されました。負傷の可能性があります。",
        "en": "Person fall detected. Possible injury.",
        "recommendations": [
            "直ちに現場を確認し、医療支援を手配してください。",
            "緊急連絡先に通報してください。",
        ],
    },
    "徘徊": {
        "ja": "不審な徘徊行動が検出されました。",
        "en": "Loitering behavior detected.",
        "recommendations": [
            "当該人物に声がけを行い、目的を確認してください。",
        ],
    },
    "密集": {
        "ja": "危険エリアで人員の密集が検出されました。",
        "en": "Crowd density exceeded safe threshold in hazard zone.",
        "recommendations": [
            "エリアを分散させてください。",
            "安全管理者に報告してください。",
        ],
    },
}

_DEFAULT_TEMPLATE = {
    "ja": "安全違反が検出されました。詳細を確認してください。",
    "en": "Safety violation detected. Please review.",
    "recommendations": ["担当者に連絡してください。", "状況を記録してください。"],
}


def _get_template(violation_type: str) -> dict[str, Any]:
    """Return the best matching template for a violation type."""
    for keyword, template in _VIOLATION_TEMPLATES.items():
        if keyword in violation_type:
            return template
    return _DEFAULT_TEMPLATE


# ---------------------------------------------------------------------------
# IncidentReporter
# ---------------------------------------------------------------------------


class IncidentReporter:
    """Autonomous incident report generator.

    Receives violations from the perception engine, generates Japanese
    natural-language reports, and delivers them via webhooks.

    Parameters
    ----------
    llava_client:
        Optional :class:`VLMClient` for LLaVA-generated descriptions.
        Falls back to rule-based templates if None.
    webhook_client:
        Optional :class:`WebhookClient` for alert delivery.
    cooldown_seconds:
        Minimum seconds between reports for the same
        (violation_type × camera_id) pair (default: 300 = 5 min).
    min_severity:
        Minimum violation severity to report.
        ``"info"`` reports all; ``"critical"`` reports only critical.
    persist_path:
        Optional path for JSON persistence of incident log.
    max_log_entries:
        Maximum incidents kept in the rolling log.
    """

    def __init__(
        self,
        llava_client: Any = None,
        webhook_client: Any = None,
        cooldown_seconds: float = 300.0,
        min_severity: str = "warning",
        persist_path: Path | None = None,
        max_log_entries: int = _MAX_LOG_ENTRIES,
    ) -> None:
        self._llava = llava_client
        self._webhook = webhook_client
        self._cooldown = cooldown_seconds
        self._min_severity_order = _SEVERITY_ORDER.get(min_severity, 1)
        self._persist_path = persist_path
        self._max_log = max_log_entries

        # State (protected by lock for thread safety)
        self._lock = threading.Lock()
        self._log: deque[IncidentReport] = deque(maxlen=max_log_entries)
        self._last_reported: dict[str, float] = {}  # dedup_key → timestamp
        self._report_count = 0
        self._webhook_failures = 0

        # Load persisted log
        if persist_path:
            self._load()

        logger.info(
            "IncidentReporter created (cooldown=%.0fs, min_severity=%s, "
            "llava=%s, webhook=%s)",
            cooldown_seconds,
            min_severity,
            "yes" if llava_client else "no",
            "yes" if webhook_client else "no",
        )

    # ── Main entry point ───────────────────────────────────────────────

    def report_violation(
        self,
        violation: Any,
        frame: np.ndarray | None,
        world_state: Any = None,
        camera_id: str = "cam01",
    ) -> IncidentReport | None:
        """Generate and deliver an incident report for a violation.

        Parameters
        ----------
        violation:
            :class:`sopilot.perception.types.Violation` instance.
        frame:
            Current BGR frame (for LLaVA description + evidence).
        world_state:
            Optional :class:`WorldState` for entity context.
        camera_id:
            Camera or session identifier.

        Returns
        -------
        IncidentReport or None
            Report if generated; None if suppressed by dedup/severity filter.
        """
        severity = getattr(violation, "severity", None)
        severity_str = severity.value if hasattr(severity, "value") else str(severity)
        severity_order = _SEVERITY_ORDER.get(severity_str, 0)

        # Severity filter
        if severity_order < self._min_severity_order:
            return None

        violation_type = getattr(violation, "rule", "") or getattr(violation, "description_ja", "")

        # Dedup check
        key = _dedup_key(violation_type, camera_id)
        with self._lock:
            last = self._last_reported.get(key, 0.0)
            now = time.time()
            if now - last < self._cooldown:
                logger.debug(
                    "IncidentReporter: suppressing duplicate '%s' (cooldown %.0fs remaining)",
                    violation_type,
                    self._cooldown - (now - last),
                )
                return None
            self._last_reported[key] = now

        # Generate description
        description_ja, description_en = self._generate_description(
            violation, frame, world_state
        )

        # Build recommendations
        template = _get_template(violation_type)
        recommendations = list(template.get("recommendations", []))
        if hasattr(violation, "description_ja") and violation.description_ja:
            description_ja = violation.description_ja + " " + description_ja

        # Collect entity IDs and evidence
        entity_ids = tuple(getattr(violation, "entity_ids", []) or [])
        evidence_detections = tuple(
            str(eid) for eid in entity_ids
        )

        # Build report
        import datetime

        report = IncidentReport(
            incident_id=str(uuid.uuid4()),
            timestamp=now,
            camera_id=camera_id,
            violation_type=violation_type,
            severity=severity_str,
            description_ja=description_ja,
            description_en=description_en,
            entity_ids=entity_ids,
            recommendations_ja=tuple(recommendations),
            evidence_detections=evidence_detections,
            created_at=datetime.datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Store in log
        with self._lock:
            self._log.append(report)
            self._report_count += 1

        # Webhook delivery (best-effort, non-blocking log)
        if self._webhook is not None:
            try:
                results = self._webhook.deliver(report)
                failures = sum(1 for r in results if not r.success)
                if failures:
                    with self._lock:
                        self._webhook_failures += failures
                    logger.warning(
                        "IncidentReporter: %d webhook delivery failure(s) for incident %s",
                        failures,
                        report.incident_id[:8],
                    )
                else:
                    logger.info(
                        "IncidentReporter: incident %s delivered to %d webhook(s)",
                        report.incident_id[:8],
                        len(results),
                    )
            except Exception:
                logger.debug("Webhook delivery failed", exc_info=True)
                with self._lock:
                    self._webhook_failures += 1

        # Persist
        if self._persist_path:
            try:
                self._save()
            except Exception:
                logger.debug("Incident log persistence failed", exc_info=True)

        logger.info(
            "Incident reported: [%s] %s — %s (camera=%s)",
            severity_str.upper(),
            violation_type,
            report.incident_id[:8],
            camera_id,
        )
        return report

    # ── Query interface ────────────────────────────────────────────────

    def get_recent_incidents(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent incidents as dicts.

        Parameters
        ----------
        limit:
            Maximum number of incidents to return.

        Returns
        -------
        list[dict]
            Most recent incidents (newest first).
        """
        with self._lock:
            incidents = list(self._log)
        return [r.to_dict() for r in reversed(incidents[-limit:])]

    def get_stats(self) -> dict[str, Any]:
        """Return reporter statistics."""
        with self._lock:
            return {
                "total_reports": self._report_count,
                "log_size": len(self._log),
                "max_log": self._max_log,
                "cooldown_seconds": self._cooldown,
                "webhook_enabled": self._webhook is not None,
                "webhook_failures": self._webhook_failures,
                "llava_enabled": self._llava is not None,
            }

    def clear_dedup_cache(self) -> None:
        """Clear the deduplication cache (useful for testing)."""
        with self._lock:
            self._last_reported.clear()

    # ── Description generation ─────────────────────────────────────────

    def _generate_description(
        self,
        violation: Any,
        frame: np.ndarray | None,
        world_state: Any,
    ) -> tuple[str, str]:
        """Generate description_ja and description_en.

        Tries LLaVA first, falls back to rule-based template.
        """
        violation_type = getattr(violation, "rule", "") or ""
        template = _get_template(violation_type)

        # LLaVA description (optional)
        llava_ja = ""
        if self._llava is not None and frame is not None and frame.size > 0:
            try:
                prompt = (
                    f"カメラ映像で次の安全違反が検出されました: 「{violation_type}」。"
                    f"具体的に何が起きているかを1〜2文で説明してください。"
                )
                llava_ja = self._llava.answer_question(frame, prompt)
                if llava_ja and len(llava_ja) > 10:
                    logger.debug("LLaVA description: %s", llava_ja[:100])
            except Exception:
                logger.debug("LLaVA description failed", exc_info=True)

        description_ja = llava_ja if llava_ja else template.get("ja", _DEFAULT_TEMPLATE["ja"])
        description_en = template.get("en", _DEFAULT_TEMPLATE["en"])
        return description_ja, description_en

    # ── Persistence ────────────────────────────────────────────────────

    def _save(self) -> None:
        """Persist incident log to JSON (best-effort)."""
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = [r.to_dict() for r in self._log]
        with open(self._persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        """Load persisted incident log from JSON."""
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for d in data[-self._max_log:]:
                try:
                    report = IncidentReport(
                        incident_id=d["incident_id"],
                        timestamp=d["timestamp"],
                        camera_id=d["camera_id"],
                        violation_type=d["violation_type"],
                        severity=d["severity"],
                        description_ja=d["description_ja"],
                        description_en=d["description_en"],
                        entity_ids=tuple(d.get("entity_ids", [])),
                        recommendations_ja=tuple(d.get("recommendations_ja", [])),
                        evidence_detections=tuple(d.get("evidence_detections", [])),
                        created_at=d.get("created_at", ""),
                    )
                    self._log.append(report)
                except Exception:
                    continue
            logger.info(
                "IncidentReporter: loaded %d incidents from %s",
                len(self._log),
                self._persist_path,
            )
        except Exception:
            logger.debug("Failed to load incident log", exc_info=True)
