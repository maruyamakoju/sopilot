"""VLM client for frame-level violation detection.

Supports Claude Vision (Anthropic API) as the primary backend.
Designed for pluggable backends: Qwen3-VL (local GPU or OpenAI-compat API), GPT-4V, etc.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
from pathlib import Path

import httpx

from sopilot.vigil.schemas import VLMResult

logger = logging.getLogger(__name__)

# ── System prompt (JSON violation format — Claude & Qwen3-VL API) ────────────

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

# ── Qwen3-VL bounding-box system prompt (0-1000 normalized coords) ───────────

_QWEN3_BBOX_SYSTEM = """\
You are a helpful assistant to detect objects in images. \
When asked to detect elements based on a description you return bounding boxes \
for all elements in the form of [xmin, ymin, xmax, ymax] with the values being \
scaled between 0 and 1000. \
When there are more than one result, answer with a list of bounding boxes in the \
form of [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...].\
"""


def _format_rules(rules: list[str]) -> str:
    return "\n".join(f"ルール{i + 1}: {r}" for i, r in enumerate(rules))


# ── VLM backend base ───────────────────────────────────────────────────────


class VLMClient:
    """Base class for VLM violation detection backends."""

    def analyze_frame(self, frame_path: Path, rules: list[str]) -> VLMResult:
        raise NotImplementedError

    def reset_session(self) -> None:
        """Reset any internal state between analysis sessions.

        Stateless backends (Claude, Qwen3) are no-ops.  Stateful backends
        (PerceptionVLMClient) must clear tracker / world-model state so that
        tracking identities from one video do not bleed into the next.
        """

    def close(self) -> None:
        pass


# ── Claude Vision backend ──────────────────────────────────────────────────


class ClaudeVisionClient(VLMClient):
    """Analyzes frames using Claude claude-sonnet-4-6 via Anthropic Messages API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
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


# ── Qwen3-VL local transformers backend ────────────────────────────────────


class Qwen3VLTransformersClient(VLMClient):
    """Analyzes frames using Qwen3-VL via local transformers inference (GPU required).

    Detects violations by running per-rule bounding-box detection.
    Violation dicts include a ``bboxes`` key (list of [x1,y1,x2,y2] in 0-1000 scale)
    which the frame endpoint uses to render annotated thumbnails.

    Default model: prithivMLmods/Qwen3-VL-4B-Instruct-Unredacted-MAX
    (same weights used by the Qwen3-VL-Video-Grounding Gradio app).
    """

    def __init__(
        self,
        model_id: str = "prithivMLmods/Qwen3-VL-4B-Instruct-Unredacted-MAX",
        device: str = "auto",
        max_new_tokens: int = 256,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        logger.info("Loading Qwen3-VL: model=%s device=%s", model_id, device)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device
        ).eval()
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._max_new_tokens = max_new_tokens
        logger.info("Qwen3-VL loaded")

    def analyze_frame(self, frame_path: Path, rules: list[str]) -> VLMResult:
        from PIL import Image

        frame = Image.open(frame_path).convert("RGB")
        violations: list[dict] = []
        raw_parts: list[str] = []

        for i, rule in enumerate(rules):
            bboxes = self._detect_bboxes(frame, rule)
            raw_parts.append(f"rule={rule!r}: bboxes={bboxes}")

            if bboxes:
                severity = _infer_severity_from_rule(rule)
                violations.append({
                    "rule_index": i,
                    "rule": rule,
                    "description_ja": f"{rule}を検出（{len(bboxes)}件）",
                    "severity": severity,
                    "confidence": min(0.95, 0.75 + len(bboxes) * 0.05),
                    "bboxes": bboxes,  # [[x1,y1,x2,y2], ...] in 0-1000 scale
                })

        return VLMResult(
            has_violation=bool(violations),
            violations=violations,
            raw_text=" | ".join(raw_parts),
        )

    def _detect_bboxes(self, frame, prompt: str) -> list[list[float]]:
        import torch

        messages = [
            {"role": "system", "content": [{"type": "text", "text": _QWEN3_BBOX_SYSTEM}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame},
                    {"type": "text", "text": f"Detect all instances of: {prompt}"},
                ],
            },
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=[frame], padding=True, return_tensors="pt"
        ).to(self._model.device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=self._max_new_tokens, do_sample=False
            )
        generated = out[:, inputs.input_ids.shape[1]:]
        txt = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        return _parse_bboxes_from_text(txt)


# ── Qwen3-VL OpenAI-compatible API backend ──────────────────────────────────


class Qwen3VLAPIClient(VLMClient):
    """Analyzes frames via any OpenAI-compatible API serving a Qwen3-VL model.

    Compatible with Together.ai, Hyperbolic, vLLM self-hosted, etc.
    Returns the same JSON violation format as ClaudeVisionClient.
    Reasoning model <think> blocks are stripped before parsing.

    Configuration (env vars):
        VIGIL_QWEN3_API_BASE  — e.g. "https://api.together.xyz/v1"
        VIGIL_QWEN3_API_KEY   — bearer token
        VIGIL_QWEN3_MODEL     — model ID (default: Qwen/Qwen3-VL-7B-Instruct)
    """

    def __init__(
        self,
        api_base: str,
        api_key: str = "",
        model: str = "Qwen/Qwen3-VL-7B-Instruct",
        max_tokens: int = 512,
        timeout: float = 60.0,
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._client = httpx.Client(timeout=timeout)

    def analyze_frame(self, frame_path: Path, rules: list[str]) -> VLMResult:
        image_data = base64.standard_b64encode(frame_path.read_bytes()).decode()

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                    {
                        "type": "text",
                        "text": _USER_PROMPT_TEMPLATE.format(rules_text=_format_rules(rules)),
                    },
                ],
            },
        ]

        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
        }
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        resp = self._client.post(
            f"{self._api_base}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()

        raw_text = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip <think>...</think> blocks emitted by reasoning models
        raw_text = re.sub(r"<think>[\s\S]*?</think>", "", raw_text).strip()
        return _parse_vlm_response(raw_text, rules)

    def close(self) -> None:
        self._client.close()


# ── Bbox helpers (Qwen3-VL grounding output) ─────────────────────────────────


def _parse_bboxes_from_text(text: str) -> list[list[float]]:
    """Parse Qwen3-VL bounding box output (0-1000 normalized scale).

    Handles formats:
        [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]   ← nested list
        [x1, y1, x2, y2]                         ← single flat list
    """
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

    # Nested list: [[...], [...]]
    nested = re.findall(
        r"\[(\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*)\]",
        text,
    )
    if nested:
        result = []
        for item in nested:
            nums = [float(x.strip()) for x in item.split(",")]
            if len(nums) == 4 and all(0 <= n <= 1010 for n in nums):
                result.append(nums)
        if result:
            return result

    # Single flat list: [x1, y1, x2, y2]
    flat = re.findall(
        r"\[(\s*\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?){3})\]", text
    )
    if flat and len(flat) == 1:
        nums = [float(x.strip()) for x in flat[0].split(",")]
        if len(nums) == 4 and all(0 <= n <= 1010 for n in nums):
            return [nums]

    return []


def _infer_severity_from_rule(rule: str) -> str:
    """Infer violation severity from Japanese rule text keywords."""
    critical_kws = ("転倒", "立入禁止", "危険", "重大", "緊急", "落下", "感電", "爆発", "火災")
    for kw in critical_kws:
        if kw in rule:
            return "critical"
    return "warning"


# ── Perception Engine backend ─────────────────────────────────────────────


class PerceptionVLMClient(VLMClient):
    """VLM backend powered by the local Perception Engine.

    Instead of sending frames to an external API, uses local object detection,
    tracking, scene graph construction, and hybrid reasoning.

    Falls back to VLM API for complex cases where local reasoning is uncertain.

    **Session isolation**: Call :meth:`reset_session` before each new video /
    stream analysis so that tracker identities and world-model state from one
    session do not contaminate the next.  The VigilPipeline calls this
    automatically at the start of ``_run()`` and ``_stream_worker()``.
    """

    def __init__(self, config: object = None, vlm_fallback: VLMClient | None = None) -> None:
        from sopilot.perception.engine import build_perception_engine

        self._vlm_fallback = vlm_fallback
        self._config = config
        self._engine = build_perception_engine(
            config=config, vlm_client=vlm_fallback,
        )
        self._frame_number = 0
        # Guard concurrent access to the stateful engine from multiple
        # pipeline threads (video + webcam can overlap).
        self._lock = threading.Lock()
        logger.info(
            "PerceptionVLMClient initialized: fallback=%s",
            type(vlm_fallback).__name__ if vlm_fallback else "None",
        )

    def reset_session(self) -> None:
        """Reset perception engine state between analysis sessions.

        Clears the tracker identity pool, world model temporal state, and the
        internal frame counter so that a new video analysis starts with a clean
        slate.
        """
        with self._lock:
            self._engine.reset()
            self._frame_number = 0
            logger.info("PerceptionVLMClient session reset")

    def analyze_frame(self, frame_path: Path, rules: list[str]) -> VLMResult:
        try:
            import cv2
        except ImportError:
            logger.error(
                "OpenCV (cv2) is required for PerceptionVLMClient. "
                "Falling back to VLM API."
            )
            if self._vlm_fallback is not None:
                return self._vlm_fallback.analyze_frame(frame_path, rules)
            return VLMResult(has_violation=False, violations=[], raw_text="cv2 unavailable")

        frame = cv2.imread(str(frame_path))
        if frame is None:
            logger.warning("Failed to read frame: %s", frame_path)
            return VLMResult(has_violation=False, violations=[], raw_text=f"failed to read {frame_path}")

        with self._lock:
            self._frame_number += 1
            timestamp = self._frame_number / 1.0  # synthetic timestamp

            result = self._engine.process_frame(frame, timestamp, self._frame_number, rules)

        # Convert FrameResult.violations → VLMResult violation dicts
        violations: list[dict] = []
        raw_parts: list[str] = []
        for v in result.violations:
            vdict: dict = {
                "rule_index": v.rule_index if v.rule_index >= 0 else 0,
                "rule": v.rule,
                "description_ja": v.description_ja,
                "severity": v.severity.value,
                "confidence": v.confidence,
            }
            # Convert BBox (normalized 0-1) to 0-1000 scale like Qwen3
            if v.bbox is not None:
                vdict["bboxes"] = [[
                    round(v.bbox.x1 * 1000, 1),
                    round(v.bbox.y1 * 1000, 1),
                    round(v.bbox.x2 * 1000, 1),
                    round(v.bbox.y2 * 1000, 1),
                ]]
            violations.append(vdict)
            raw_parts.append(
                f"rule={v.rule!r} sev={v.severity.value} conf={v.confidence:.2f} "
                f"src={v.source}"
            )

        raw_text = (
            f"perception_engine: {len(violations)} violations, "
            f"{result.detections_count} detections, {result.tracks_count} tracks, "
            f"vlm_called={result.vlm_called} | "
            + " | ".join(raw_parts)
        )

        return VLMResult(
            has_violation=bool(violations),
            violations=violations,
            raw_text=raw_text,
        )

    def close(self) -> None:
        self._engine.close()
        if self._vlm_fallback is not None:
            self._vlm_fallback.close()
        logger.info("PerceptionVLMClient closed")


# ── JSON violation response parser (Claude / Qwen3-VL API) ─────────────────


def _parse_vlm_response(raw_text: str, rules: list[str]) -> VLMResult:
    """Parse VLM JSON response, with graceful fallback on malformed output."""
    text = raw_text
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        text = m.group(0)

    try:
        data = json.loads(text)
        has_violation = bool(data.get("has_violation", False))
        violations = data.get("violations", [])

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
    """Build the configured VLM client from environment or explicit args.

    Backends:
        claude      — Claude Vision via Anthropic API (default; no GPU required)
        qwen3       — Qwen3-VL via local transformers inference (GPU required)
        qwen3-api   — Qwen3-VL via OpenAI-compatible API endpoint (no GPU required)
        perception  — Local Perception Engine with optional Claude VLM fallback
    """
    backend = backend or os.environ.get("VIGIL_VLM_BACKEND", "claude")
    key = (
        api_key
        or os.environ.get("VIGIL_VLM_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY", "")
    )
    model = os.environ.get("VIGIL_VLM_MODEL", "claude-sonnet-4-6")

    if backend == "claude":
        return ClaudeVisionClient(api_key=key, model=model)

    if backend == "qwen3":
        model_id = os.environ.get(
            "VIGIL_QWEN3_MODEL_ID",
            "prithivMLmods/Qwen3-VL-4B-Instruct-Unredacted-MAX",
        )
        device = os.environ.get("VIGIL_QWEN3_DEVICE", "auto")
        return Qwen3VLTransformersClient(model_id=model_id, device=device)

    if backend == "qwen3-api":
        api_base = os.environ.get("VIGIL_QWEN3_API_BASE", "")
        if not api_base:
            raise ValueError(
                "VIGIL_QWEN3_API_BASE must be set when VIGIL_VLM_BACKEND=qwen3-api"
            )
        qwen_key = os.environ.get("VIGIL_QWEN3_API_KEY") or key
        qwen_model = os.environ.get("VIGIL_QWEN3_MODEL", "Qwen/Qwen3-VL-7B-Instruct")
        return Qwen3VLAPIClient(api_base=api_base, api_key=qwen_key, model=qwen_model)

    if backend == "perception":
        try:
            from sopilot.perception.engine import build_perception_engine  # noqa: F401
            from sopilot.perception.types import PerceptionConfig
        except ImportError:
            logger.warning(
                "Perception engine modules not available. "
                "Falling back to claude backend."
            )
            return ClaudeVisionClient(api_key=key, model=model)

        config = PerceptionConfig(
            detector_backend=os.environ.get("PERCEPTION_DETECTOR", "mock"),
            device=os.environ.get("PERCEPTION_DEVICE", "auto"),
        )
        vlm_fallback: VLMClient | None = None
        if key:  # If API key available, create Claude fallback
            vlm_fallback = ClaudeVisionClient(api_key=key, model=model)
            logger.info("Perception engine: Claude VLM fallback enabled")

        try:
            return PerceptionVLMClient(config=config, vlm_fallback=vlm_fallback)
        except Exception:
            logger.exception(
                "Failed to initialize Perception engine. "
                "Falling back to claude backend."
            )
            if vlm_fallback is not None:
                vlm_fallback.close()
            return ClaudeVisionClient(api_key=key, model=model)

    raise ValueError(
        f"Unknown VLM backend: {backend!r}. "
        f"Supported: 'claude', 'qwen3', 'qwen3-api', 'perception'"
    )
