"""Fire-and-forget webhook dispatcher with HMAC-SHA256 signing support.

When a VigilPilot violation is detected the pipeline calls
:meth:`WebhookDispatcher.dispatch_violation` with the violation payload and
the full list of registered webhooks.  Each eligible webhook (enabled, meets
severity threshold) is delivered in its own daemon thread so the analysis
pipeline is never blocked waiting for a remote HTTP call.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading

import httpx

logger = logging.getLogger(__name__)

SEVERITY_ORDER: dict[str, int] = {"info": 0, "warning": 1, "critical": 2}

_USER_AGENT = "SOPilot-Webhook/1.4"


class WebhookDispatcher:
    """Fire-and-forget webhook dispatcher with HMAC signing support."""

    # ── Public API ─────────────────────────────────────────────────────────

    def dispatch_violation(
        self,
        violation_data: dict,
        webhooks: list[dict],
        repo: "WebhookRepository",  # noqa: F821 – forward ref for typing only
    ) -> None:
        """Dispatch *violation_data* to all eligible webhooks in background threads.

        Parameters
        ----------
        violation_data:
            Serialisable dict describing the violation event.  Must include at
            least a ``"severity"`` key with one of ``"info"``, ``"warning"``,
            or ``"critical"``.
        webhooks:
            All registered webhook rows from
            :meth:`~WebhookRepository.list_all`.  Disabled webhooks and those
            whose ``min_severity`` is above the event severity are silently
            skipped.
        repo:
            Repository instance used to call
            :meth:`~WebhookRepository.update_triggered` on success.
        """
        ev_sev = SEVERITY_ORDER.get(violation_data.get("severity", "critical"), 0)

        for wh in webhooks:
            if not wh.get("enabled", True):
                continue
            wh_sev = SEVERITY_ORDER.get(wh.get("min_severity", "critical"), 2)
            if ev_sev < wh_sev:
                continue  # event severity is below this webhook's threshold
            threading.Thread(
                target=self._send,
                args=(wh, violation_data, repo),
                daemon=True,
                name=f"vigil-wh-{wh.get('id', '?')}",
            ).start()

    def test_webhook(self, wh: dict) -> dict:
        """Send a test payload to *wh* and return a result dict.

        Returns a :class:`dict` compatible with :class:`WebhookTestResult`::

            {"ok": bool, "status_code": int | None, "error": str | None}
        """
        test_payload = {
            "event": "test",
            "message": "SOPilot webhook test",
            "source": "sopilot-vigil",
        }
        body = json.dumps(test_payload, ensure_ascii=False)
        headers = {
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
        }
        if wh.get("secret"):
            sig = hmac.new(
                wh["secret"].encode(),
                body.encode(),
                hashlib.sha256,
            ).hexdigest()
            headers["X-SOPilot-Signature"] = f"sha256={sig}"
        try:
            r = httpx.post(wh["url"], content=body, headers=headers, timeout=5.0)
            return {"ok": r.status_code < 400, "status_code": r.status_code, "error": None}
        except Exception as exc:
            return {"ok": False, "status_code": None, "error": str(exc)}

    # ── Internal ───────────────────────────────────────────────────────────

    def _send(self, wh: dict, payload: dict, repo) -> None:  # type: ignore[type-arg]
        """HTTP POST *payload* to *wh["url"]* and record the result."""
        body = json.dumps(payload, ensure_ascii=False)
        headers = {
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
        }
        if wh.get("secret"):
            sig = hmac.new(
                wh["secret"].encode(),
                body.encode(),
                hashlib.sha256,
            ).hexdigest()
            headers["X-SOPilot-Signature"] = f"sha256={sig}"
        try:
            r = httpx.post(wh["url"], content=body, headers=headers, timeout=10.0)
            logger.info(
                "Webhook %s → %s: HTTP %d", wh.get("id", "?"), wh["url"], r.status_code
            )
            repo.update_triggered(wh["id"])
        except Exception as exc:
            logger.warning("Webhook %s failed: %s", wh.get("id", "?"), exc)
