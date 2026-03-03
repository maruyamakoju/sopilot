"""Anomaly → VLM escalation for natural-language explanations.

When an ANOMALY event fires, this module optionally calls a VLM
(Claude / Qwen3) to explain *why* it is anomalous given the current
frame.  The explanation is attached to the event's details dict so
downstream consumers (UI, narration, webhook) can display it.

Design:
    - Detector-specific prompt templates (behavioral / spatial / temporal / interaction)
    - Independent 120s cooldown (VLM cost control, separate from the ensemble's 60s)
    - Graceful degradation: returns None when VLM is unavailable
    - Frame encoding: temp JPEG → base64 (same pattern as reasoning.py VLMEscalation)
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from sopilot.perception.types import EntityEvent, WorldState

logger = logging.getLogger(__name__)


# ── Detector-specific prompt templates ────────────────────────────────────


_PROMPT_TEMPLATES: dict[str, str] = {
    "behavioral": (
        "この映像フレームで行動異常が検出されました。\n"
        "詳細: {description_ja}\n\n"
        "フレーム内の人物や物体の動きを観察し、"
        "なぜこの行動パターンが通常と異なるのか、"
        "考えられる原因を日本語で簡潔に説明してください。"
    ),
    "spatial": (
        "この映像フレームで空間異常が検出されました。\n"
        "詳細: {description_ja}\n\n"
        "通常は人が立ち入らないエリアにエンティティが存在しています。"
        "フレーム内の状況を観察し、なぜこの位置が異常なのか、"
        "安全上のリスクがあるか日本語で簡潔に説明してください。"
    ),
    "temporal": (
        "この映像フレームで時間帯異常が検出されました。\n"
        "詳細: {description_ja}\n\n"
        "この時間帯に通常と異なる人数が検出されています。"
        "フレーム内の状況を観察し、なぜ人数が異常なのか、"
        "考えられる理由を日本語で簡潔に説明してください。"
    ),
    "interaction": (
        "この映像フレームで関係性異常が検出されました。\n"
        "詳細: {description_ja}\n\n"
        "通常は観察されないエンティティ間の関係が発生しています。"
        "フレーム内のオブジェクト同士の位置関係を観察し、"
        "なぜこの組み合わせが異常なのか日本語で簡潔に説明してください。"
    ),
}

_DEFAULT_PROMPT = (
    "この映像フレームで異常が検出されました。\n"
    "詳細: {description_ja}\n\n"
    "フレーム内の状況を観察し、なぜ異常と判定されたのか"
    "日本語で簡潔に説明してください。"
)


class AnomalyExplainer:
    """Anomaly → VLM エスカレーションで自然言語説明を生成。

    Args:
        vlm_client: VLM client instance (ClaudeVisionClient, etc.) with
            ``analyze_frame(frame_path, prompt)`` method.
        cooldown_seconds: Minimum interval between VLM calls for the
            same (detector, metric, entity_id) triple.  Default 120s.
    """

    def __init__(
        self,
        vlm_client: Any = None,
        cooldown_seconds: float = 120.0,
    ) -> None:
        self._vlm_client = vlm_client
        self._cooldown_seconds = cooldown_seconds
        # (detector, metric, entity_id) → last_call_timestamp
        self._cooldown_map: dict[tuple[str, str, int], float] = {}
        self._total_calls: int = 0

    def explain(
        self,
        event: EntityEvent,
        frame: np.ndarray | None,
        world_state: WorldState,
    ) -> str | None:
        """Generate a VLM explanation for an anomaly event.

        Returns None if:
            - No VLM client is configured
            - The event is within the cooldown period
            - Frame encoding fails
            - VLM call fails

        Side effect: stores the explanation in event.details["vlm_explanation"].
        """
        if self._vlm_client is None:
            return None

        if frame is None:
            return None

        # Cooldown check
        detector = event.details.get("detector", "unknown")
        metric = event.details.get("metric", "unknown")
        entity_id = event.entity_id
        cooldown_key = (detector, metric, entity_id)

        now = time.time()
        last_call = self._cooldown_map.get(cooldown_key, float("-inf"))
        if (now - last_call) < self._cooldown_seconds:
            logger.debug(
                "AnomalyExplainer cooldown active for %s (%.0fs remaining)",
                cooldown_key,
                self._cooldown_seconds - (now - last_call),
            )
            return None

        # Build prompt
        prompt = self.build_prompt(event, world_state)

        # Encode frame to temp JPEG
        tmp_path: Path | None = None
        try:
            try:
                import cv2
            except ImportError:
                logger.debug("cv2 not available; cannot encode frame for VLM.")
                return None

            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False
            ) as tmp:
                tmp_path = Path(tmp.name)
                cv2.imwrite(str(tmp_path), frame)

            # Call VLM
            result = self._vlm_client.analyze_frame(str(tmp_path), prompt)
            self._cooldown_map[cooldown_key] = time.time()
            self._total_calls += 1

            # Extract explanation text
            explanation = self._extract_explanation(result)
            if explanation:
                logger.info(
                    "AnomalyExplainer VLM explanation for %s/%s: %s",
                    detector, metric, explanation[:80],
                )
            return explanation

        except Exception:
            logger.exception("AnomalyExplainer VLM call failed")
            return None
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def build_prompt(
        self, event: EntityEvent, world_state: WorldState
    ) -> str:
        """Build a detector-specific VLM prompt for the anomaly event."""
        detector = event.details.get("detector", "unknown")
        description_ja = event.details.get("description_ja", "異常検出")

        template = _PROMPT_TEMPLATES.get(detector, _DEFAULT_PROMPT)
        prompt = template.format(description_ja=description_ja)

        # Append scene context
        entity_count = world_state.entity_count
        person_count = world_state.person_count
        prompt += f"\n\n[コンテキスト] エンティティ数={entity_count}, 人数={person_count}"

        # Add z-score info
        z_score = event.details.get("z_score")
        if z_score is not None:
            prompt += f", 偏差スコア={z_score}"

        return prompt

    def _extract_explanation(self, vlm_result: Any) -> str | None:
        """Extract explanation text from VLM result.

        Handles both string results and structured dict results.
        """
        if vlm_result is None:
            return None
        if isinstance(vlm_result, str):
            return vlm_result.strip() if vlm_result.strip() else None
        if isinstance(vlm_result, dict):
            # Try common keys
            for key in ("explanation", "text", "description", "content", "answer"):
                val = vlm_result.get(key)
                if val and isinstance(val, str):
                    return val.strip()
            # Try violations list
            violations = vlm_result.get("violations", [])
            if violations and isinstance(violations, list):
                descs = [v.get("description", "") for v in violations if isinstance(v, dict)]
                return "; ".join(d for d in descs if d) or None
        return str(vlm_result).strip() if vlm_result else None

    @property
    def total_calls(self) -> int:
        """Total number of VLM calls made."""
        return self._total_calls
