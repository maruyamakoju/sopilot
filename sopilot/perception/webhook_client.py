"""Webhook Client — multi-target alert delivery for incident reports.

Phase 40: Delivers structured incident reports to external systems
(Slack, Microsoft Teams, or any generic HTTP endpoint) with retry
logic and delivery tracking.

Usage::

    config = WebhookConfig(url="https://hooks.slack.com/...", type="slack")
    client = WebhookClient([config])
    results = client.deliver(report)  # returns list[DeliveryResult]

Supports:
    - Slack Incoming Webhooks (rich attachment format)
    - Microsoft Teams (Connector Card format)
    - Generic HTTP POST (JSON body)
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 5.0
_MAX_RETRIES = 2
_RETRY_DELAY = 0.5


# ---------------------------------------------------------------------------
# Config + Result
# ---------------------------------------------------------------------------


@dataclass
class WebhookConfig:
    """Configuration for a single webhook target.

    Parameters
    ----------
    url:
        Webhook endpoint URL.
    type:
        Delivery format: ``"slack"``, ``"teams"``, or ``"generic"``.
    headers:
        Extra HTTP headers (e.g., Authorization).
    timeout_seconds:
        Per-request timeout.
    """

    url: str
    type: str = "generic"  # "slack", "teams", "generic"
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = _DEFAULT_TIMEOUT

    def __post_init__(self) -> None:
        if self.type not in ("slack", "teams", "generic"):
            raise ValueError(f"Unknown webhook type: {self.type!r}")


@dataclass
class DeliveryResult:
    """Result of a single webhook delivery attempt."""

    url: str
    success: bool
    status_code: int = 0
    latency_ms: float = 0.0
    error: str = ""
    retries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "success": self.success,
            "status_code": self.status_code,
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error,
            "retries": self.retries,
        }


# ---------------------------------------------------------------------------
# WebhookClient
# ---------------------------------------------------------------------------


class WebhookClient:
    """Multi-target webhook delivery client.

    Delivers incident reports to one or more webhook endpoints.
    Thread-safe: state is immutable after construction.

    Parameters
    ----------
    configs:
        List of :class:`WebhookConfig` targets.
    max_retries:
        Maximum delivery attempts per target.
    retry_delay:
        Seconds between retries.
    """

    def __init__(
        self,
        configs: list[WebhookConfig],
        max_retries: int = _MAX_RETRIES,
        retry_delay: float = _RETRY_DELAY,
    ) -> None:
        self._configs = list(configs)
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._delivery_count = 0
        self._failure_count = 0
        logger.info(
            "WebhookClient created (%d target(s)): %s",
            len(configs),
            [c.type for c in configs],
        )

    # ── Public API ─────────────────────────────────────────────────────

    def deliver(self, report: Any) -> list[DeliveryResult]:
        """Deliver an incident report to all configured targets.

        Parameters
        ----------
        report:
            :class:`IncidentReport` instance.

        Returns
        -------
        list[DeliveryResult]
            One result per webhook target.
        """
        results: list[DeliveryResult] = []
        for config in self._configs:
            result = self._deliver_one(config, report)
            results.append(result)
            if result.success:
                self._delivery_count += 1
            else:
                self._failure_count += 1
        return results

    def test_webhook(self, url: str, webhook_type: str = "generic") -> dict[str, Any]:
        """Send a test ping to verify a webhook URL is reachable.

        Returns
        -------
        dict
            ``{"success": bool, "status_code": int, "latency_ms": float, "error": str}``
        """
        config = WebhookConfig(url=url, type=webhook_type)
        test_payload = {
            "text": "✅ SOPilot Webhook テスト成功 — 接続確認完了",
            "test": True,
            "source": "sopilot",
        }
        result = self._post(url, test_payload, config.headers, config.timeout_seconds)
        return result.to_dict()

    def get_stats(self) -> dict[str, Any]:
        return {
            "target_count": len(self._configs),
            "targets": [{"url": c.url, "type": c.type} for c in self._configs],
            "deliveries_sent": self._delivery_count,
            "failures": self._failure_count,
        }

    # ── Payload builders ───────────────────────────────────────────────

    def _build_slack_payload(self, report: Any) -> dict[str, Any]:
        """Build Slack Incoming Webhook payload."""
        color = {"critical": "#FF0000", "warning": "#FF8C00", "info": "#0078D7"}.get(
            report.severity, "#888888"
        )
        severity_emoji = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}.get(
            report.severity, "📷"
        )
        return {
            "text": f"{severity_emoji} *安全違反検出 [{report.severity.upper()}]*",
            "attachments": [
                {
                    "color": color,
                    "title": report.violation_type,
                    "text": report.description_ja,
                    "fields": [
                        {
                            "title": "カメラID",
                            "value": report.camera_id or "不明",
                            "short": True,
                        },
                        {
                            "title": "検出時刻",
                            "value": report.created_at,
                            "short": True,
                        },
                        {
                            "title": "対象エンティティ",
                            "value": (
                                ", ".join(f"ID:{e}" for e in report.entity_ids)
                                if report.entity_ids
                                else "なし"
                            ),
                            "short": True,
                        },
                        {
                            "title": "インシデントID",
                            "value": report.incident_id[:8],
                            "short": True,
                        },
                    ],
                    "footer": "SOPilot Safety AI",
                    "footer_icon": "https://cdn.jsdelivr.net/npm/@anthropic-ai/sdk@0.0.1/LICENSE",
                }
            ],
        }

    def _build_teams_payload(self, report: Any) -> dict[str, Any]:
        """Build Microsoft Teams Connector Card payload."""
        color = {"critical": "FF0000", "warning": "FF8C00", "info": "0078D7"}.get(
            report.severity, "888888"
        )
        return {
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "themeColor": color,
            "summary": f"安全違反: {report.violation_type}",
            "sections": [
                {
                    "activityTitle": f"🚨 {report.violation_type}",
                    "activitySubtitle": f"SOPilot — {report.created_at}",
                    "activityText": report.description_ja,
                    "facts": [
                        {"name": "カメラID", "value": report.camera_id or "不明"},
                        {"name": "重大度", "value": report.severity.upper()},
                        {"name": "インシデントID", "value": report.incident_id[:8]},
                        {
                            "name": "推奨対応",
                            "value": (
                                report.recommendations_ja[0]
                                if report.recommendations_ja
                                else "担当者に連絡してください"
                            ),
                        },
                    ],
                    "markdown": True,
                }
            ],
        }

    def _build_generic_payload(self, report: Any) -> dict[str, Any]:
        """Build generic JSON payload."""
        return {
            "incident_id": report.incident_id,
            "timestamp": report.timestamp,
            "camera_id": report.camera_id,
            "violation_type": report.violation_type,
            "severity": report.severity,
            "description_ja": report.description_ja,
            "description_en": report.description_en,
            "entity_ids": report.entity_ids,
            "recommendations_ja": report.recommendations_ja,
            "created_at": report.created_at,
            "source": "sopilot",
        }

    # ── Delivery internals ─────────────────────────────────────────────

    def _deliver_one(
        self, config: WebhookConfig, report: Any
    ) -> DeliveryResult:
        """Deliver to a single target with retry."""
        if config.type == "slack":
            payload = self._build_slack_payload(report)
        elif config.type == "teams":
            payload = self._build_teams_payload(report)
        else:
            payload = self._build_generic_payload(report)

        last_result = DeliveryResult(url=config.url, success=False)
        for attempt in range(self._max_retries + 1):
            result = self._post(config.url, payload, config.headers, config.timeout_seconds)
            result.retries = attempt
            if result.success:
                return result
            last_result = result
            if attempt < self._max_retries:
                time.sleep(self._retry_delay)

        logger.warning(
            "Webhook delivery failed after %d attempts: %s — %s",
            self._max_retries + 1,
            config.url,
            last_result.error,
        )
        return last_result

    def _post(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> DeliveryResult:
        """Send a single HTTP POST request."""
        t0 = time.monotonic()
        try:
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("User-Agent", "SOPilot/4.9.0")
            for k, v in headers.items():
                req.add_header(k, v)

            with urllib.request.urlopen(req, timeout=timeout) as resp:
                elapsed = (time.monotonic() - t0) * 1000
                return DeliveryResult(
                    url=url,
                    success=200 <= resp.status < 300,
                    status_code=resp.status,
                    latency_ms=elapsed,
                )
        except urllib.error.HTTPError as e:
            elapsed = (time.monotonic() - t0) * 1000
            return DeliveryResult(
                url=url,
                success=False,
                status_code=e.code,
                latency_ms=elapsed,
                error=f"HTTP {e.code}: {e.reason}",
            )
        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000
            return DeliveryResult(
                url=url,
                success=False,
                latency_ms=elapsed,
                error=str(exc),
            )


def build_webhook_client(
    webhook_urls: list[str],
    webhook_type: str = "generic",
) -> WebhookClient:
    """Build a WebhookClient from a list of URLs.

    Parameters
    ----------
    webhook_urls:
        List of webhook endpoint URLs.
    webhook_type:
        Default format: ``"slack"``, ``"teams"``, or ``"generic"``.

    Returns
    -------
    WebhookClient
    """
    configs = []
    for url in webhook_urls:
        # Auto-detect type from URL pattern
        detected_type = webhook_type
        if "hooks.slack.com" in url:
            detected_type = "slack"
        elif "webhook.office.com" in url or "teams.microsoft.com" in url:
            detected_type = "teams"
        configs.append(WebhookConfig(url=url, type=detected_type))
    return WebhookClient(configs)
