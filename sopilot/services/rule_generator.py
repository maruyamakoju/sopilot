"""Automatic monitoring rule generation from SOP gold videos.

Phase 28A — SOP×Vigil Auto-Bridge.

``RuleGenerator`` samples frames from a gold video, sends them to a VLM
with a rule-extraction prompt, and returns a structured list of monitoring
rules suitable for use in VigilPilot streaming sessions.

The generator is VLM-agnostic: it accepts any callable that takes
``(frame_path: Path, prompt: str) -> str``.  In production this is backed by
``ClaudeVisionClient``; in tests it is mocked.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
あなたは製造業・建設業・物流業の安全管理専門家AIです。
SOP（標準作業手順）の研修動画フレームを分析し、
現場カメラで継続的に監視すべき安全ルールを抽出してください。

以下のJSON形式のみで返答してください（説明不要）：
{
  "rules": [
    {"rule": "ルールのテキスト（日本語、20字以内）", "confidence": 0.0〜1.0, "category": "ppe|zone|behavior|equipment"},
    ...
  ]
}

ルール例:
- "ヘルメット着用必須"
- "安全ベスト着用必須"
- "立入禁止エリア侵入禁止"
- "2人作業必須"
- "手袋着用必須"
- "機械操作前確認必須"

観察できないルールは含めないでください。最大10件。
"""

_USER_PROMPT = """\
この作業訓練動画のフレームを分析し、現場カメラで監視すべき安全・コンプライアンスルールを
抽出してください。実際にフレームから観察・推定できるルールのみリストアップしてください。
"""

# ── Fallback rules (when VLM is unavailable or returns nothing) ────────────────

_DEFAULT_RULES = [
    "ヘルメット着用必須",
    "安全ベスト着用必須",
    "立入禁止エリア侵入禁止",
    "手袋着用必須",
]


class RuleGenerator:
    """Extracts VigilPilot monitoring rules from a SOP gold video.

    Parameters
    ----------
    vlm_callable:
        A function ``(jpeg_bytes: bytes, system_prompt: str, user_prompt: str) -> str``
        that calls the VLM and returns raw text.  If *None*, falls back to
        heuristic rules extracted from video properties.
    n_frames:
        Number of frames to sample for analysis (default: 5).
    """

    def __init__(
        self,
        vlm_callable: Callable[[bytes, str, str], str] | None = None,
        n_frames: int = 5,
    ) -> None:
        self._vlm = vlm_callable
        self._n_frames = n_frames

    def generate(self, video_path: str) -> list[dict[str, Any]]:
        """Sample *video_path* and return a list of rule dicts.

        Each dict has keys: ``rule`` (str), ``confidence`` (float),
        ``category`` (str).

        Falls back to ``_DEFAULT_RULES`` if the video cannot be read or the
        VLM is unavailable.
        """
        frames = self._sample_frames(video_path)
        if not frames:
            logger.warning("No frames sampled from %s — using default rules", video_path)
            return _make_default_rules()

        if self._vlm is None:
            logger.info("No VLM configured — using default rules")
            return _make_default_rules()

        rules: list[dict[str, Any]] = []
        for i, frame_jpeg in enumerate(frames):
            try:
                raw_text = self._vlm(frame_jpeg, _SYSTEM_PROMPT, _USER_PROMPT)
                parsed = _parse_vlm_response(raw_text)
                rules.extend(parsed)
            except Exception:
                logger.exception("VLM call failed for frame %d of %s", i, video_path)

        if not rules:
            logger.warning("VLM returned no rules for %s — using defaults", video_path)
            return _make_default_rules()

        return _deduplicate(rules)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _sample_frames(self, video_path: str) -> list[bytes]:
        """Sample N evenly-spaced JPEG frames from *video_path*."""
        try:
            import cv2  # type: ignore[import-untyped]
        except ImportError:
            logger.error("cv2 not available — cannot sample frames")
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Cannot open video: %s", video_path)
            return []

        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                return []

            positions = np.linspace(0, max(total - 1, 0), min(self._n_frames, total), dtype=int)
            frames: list[bytes] = []

            for pos in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
                ret, frame = cap.read()
                if not ret:
                    continue
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frames.append(bytes(buf))

            return frames
        finally:
            cap.release()


# ── Parsing helpers ────────────────────────────────────────────────────────────


def _parse_vlm_response(text: str) -> list[dict[str, Any]]:
    """Extract rule list from VLM JSON response, tolerating partial JSON."""
    # Try to find JSON block
    match = re.search(r'\{.*"rules"\s*:\s*\[.*?\]\s*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            rules = data.get("rules", [])
            return [
                {
                    "rule": str(r.get("rule", "")).strip(),
                    "confidence": float(r.get("confidence", 0.5)),
                    "category": str(r.get("category", "behavior")),
                }
                for r in rules
                if isinstance(r, dict) and r.get("rule")
            ]
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: extract lines that look like rules
    rules = []
    for line in text.splitlines():
        line = line.strip().strip('"').strip("'").strip("-").strip()
        if 3 < len(line) < 50 and any(
            kw in line for kw in ["必須", "禁止", "着用", "確認", "制限", "required", "prohibited"]
        ):
            rules.append({"rule": line, "confidence": 0.4, "category": "behavior"})
    return rules[:10]


def _deduplicate(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate rules (case-insensitive, keep highest confidence)."""
    seen: dict[str, dict[str, Any]] = {}
    for r in rules:
        key = r["rule"].lower().strip()
        if key not in seen or r["confidence"] > seen[key]["confidence"]:
            seen[key] = r
    return list(seen.values())[:15]


def _make_default_rules() -> list[dict[str, Any]]:
    return [
        {"rule": r, "confidence": 0.5, "category": "ppe"}
        for r in _DEFAULT_RULES
    ]


# ── VLM callable factory (for ClaudeVisionClient) ─────────────────────────────


def make_claude_vlm_callable(api_key: str, model: str = "claude-haiku-4-5-20251001") -> Callable:
    """Return a VLM callable backed by Claude Vision API.

    Uses Haiku for speed/cost in rule extraction.
    """
    import base64

    import httpx

    client = httpx.Client(timeout=30.0)

    def _call(jpeg_bytes: bytes, system_prompt: str, user_prompt: str) -> str:
        image_data = base64.standard_b64encode(jpeg_bytes).decode()
        payload = {
            "model": model,
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ],
        }
        r = client.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        r.raise_for_status()
        return r.json()["content"][0]["text"]

    return _call
