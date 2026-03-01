"""VLM client for frame-level violation detection.

Supports Claude Vision (Anthropic API) as the primary backend.
Designed for pluggable backends (GPT-4V, Gemini, local LLaVA, etc.).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path

import httpx

from sopilot.vigil.schemas import VLMResult

logger = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
あなたは工場・建設現場・施設の監視カメラ映像を分析する安全監視AIです。
ユーザーが定義したルールに基づき、映像フレームを正確に分析してください。

必ず以下のJSON形式のみで返答してください。説明文や前置きは不要です：
{
  "has_violation": true または false,
  "violations": [
    {
      "rule_index": 0,
      "rule": "ルールのテキスト",
      "description_ja": "違反の具体的な説明（日本語、50字以内）",
      "severity": "critical" または "warning" または "info",
      "confidence": 0.0〜1.0
    }
  ]
}

severity の基準:
- critical: 即時対応が必要な重大な危険（転倒、重大な安全違反）
- warning: 是正が必要な安全上の問題（保護具未着用など）
- info: 軽微な注意事項

違反がない場合は has_violation を false にして violations を空配列にしてください。
"""

_USER_PROMPT_TEMPLATE = """\
以下のルールに従って、この監視カメラ映像フレームを分析してください：

{rules_text}

上記のルールのいずれかに違反していると判断できる場合は report してください。
"""


def _format_rules(rules: list[str]) -> str:
    return "\n".join(f"ルール{i + 1}: {r}" for i, r in enumerate(rules))


# ── VLM backend base ───────────────────────────────────────────────────────


class VLMClient:
    """Base class for VLM violation detection backends."""

    def analyze_frame(self, frame_path: Path, rules: list[str]) -> VLMResult:
        raise NotImplementedError


# ── Claude Vision backend ──────────────────────────────────────────────────


class ClaudeVisionClient(VLMClient):
    """Analyzes frames using Claude claude-sonnet-4-6 via Anthropic Messages API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6-20251022",
        max_tokens: int = 512,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._client = httpx.Client(timeout=timeout)

    def analyze_frame(self, frame_path: Path, rules: list[str]) -> VLMResult:
        image_data = base64.standard_b64encode(frame_path.read_bytes()).decode()
        media_type = "image/jpeg"

        user_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data,
                },
            },
            {
                "type": "text",
                "text": _USER_PROMPT_TEMPLATE.format(rules_text=_format_rules(rules)),
            },
        ]

        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": _SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_content}],
        }

        resp = self._client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()

        raw_text = resp.json()["content"][0]["text"].strip()
        return _parse_vlm_response(raw_text, rules)

    def close(self) -> None:
        self._client.close()


# ── Parsing ────────────────────────────────────────────────────────────────


def _parse_vlm_response(raw_text: str, rules: list[str]) -> VLMResult:
    """Parse VLM JSON response, with graceful fallback on malformed output."""
    # Extract JSON block (handle markdown code fences)
    text = raw_text
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        text = m.group(0)

    try:
        data = json.loads(text)
        has_violation = bool(data.get("has_violation", False))
        violations = data.get("violations", [])

        # Validate and clamp fields
        cleaned: list[dict] = []
        for v in violations:
            idx = int(v.get("rule_index", 0))
            rule_text = v.get("rule", rules[idx] if idx < len(rules) else "unknown")
            cleaned.append({
                "rule_index": idx,
                "rule": rule_text,
                "description_ja": str(v.get("description_ja", ""))[:100],
                "severity": v.get("severity", "warning") if v.get("severity") in ("critical", "warning", "info") else "warning",
                "confidence": max(0.0, min(1.0, float(v.get("confidence", 0.5)))),
            })

        return VLMResult(has_violation=has_violation, violations=cleaned, raw_text=raw_text)

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("VLM response parse error: %s — raw: %.200s", e, raw_text)
        return VLMResult(has_violation=False, violations=[], raw_text=raw_text)


# ── Factory ────────────────────────────────────────────────────────────────


def build_vlm_client(backend: str | None = None, api_key: str | None = None) -> VLMClient:
    """Build the configured VLM client from environment or explicit args."""
    backend = backend or os.environ.get("VIGIL_VLM_BACKEND", "claude")
    key = (
        api_key
        or os.environ.get("VIGIL_VLM_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY", "")
    )
    model = os.environ.get("VIGIL_VLM_MODEL", "claude-sonnet-4-6-20251022")

    if backend == "claude":
        return ClaudeVisionClient(api_key=key, model=model)

    raise ValueError(f"Unknown VLM backend: {backend!r}. Supported: 'claude'")
