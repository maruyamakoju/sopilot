"""Per-frame inference for live streaming sessions.

``FrameInferrer`` decodes a JPEG frame, runs an optional object detector,
matches detected labels against session rules, and annotates the frame with
bounding-box overlays.

It is intentionally lightweight — no VLM call, no tracker state.  For deeper
analysis, upgrade to a full VigilPipeline session.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Keyword sets for built-in rule matching
_PRESENCE_KEYWORDS = {"侵入", "入室", "presence", "intrusion"}
_ABSENCE_KEYWORDS = {"未着用", "不着用", "missing", "without", "no "}

# Equipment keyword → detector label mapping (lower-case)
_EQUIP_MAP: dict[str, list[str]] = {
    "ヘルメット": ["helmet", "hard hat"],
    "ベスト": ["vest", "safety vest"],
    "手袋": ["gloves"],
    "マスク": ["mask"],
    "ゴーグル": ["goggles"],
    "ハーネス": ["safety harness", "harness"],
    "helmet": ["helmet", "hard hat"],
    "vest": ["vest", "safety vest"],
    "gloves": ["gloves"],
    "mask": ["mask"],
}


class FrameInferrer:
    """Runs per-frame detection and rule-violation matching.

    Parameters
    ----------
    detector:
        Optional object with a ``detect(frame: np.ndarray) -> list[Detection]``
        method.  If *None*, the inferrer returns empty detections.
    """

    def __init__(self, detector: Any = None) -> None:
        self._detector = detector

    def infer(self, frame_jpeg: bytes, rules: list[str]) -> dict[str, Any]:
        """Decode JPEG, run detector, evaluate rules, annotate frame.

        Returns
        -------
        dict
            ``detections``      list[dict] — label, confidence, bbox
            ``violations``      list[str]  — triggered rules
            ``frame_ms``        float      — wall-clock inference time in ms
            ``annotated_jpeg``  bytes      — JPEG with bbox overlays
        """
        t0 = time.perf_counter()

        frame = self._decode(frame_jpeg)
        if frame is None:
            return {
                "detections": [],
                "violations": [],
                "frame_ms": 0.0,
                "annotated_jpeg": frame_jpeg,
            }

        detections = self._detect(frame)
        violations = self._match_rules(detections, rules)
        annotated_jpeg = self._draw_boxes(frame, detections, violations, frame_jpeg)
        frame_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "detections": [
                {
                    "label": d.label,
                    "confidence": round(d.confidence, 3),
                    "bbox": {
                        "x1": round(d.bbox.x1, 4),
                        "y1": round(d.bbox.y1, 4),
                        "x2": round(d.bbox.x2, 4),
                        "y2": round(d.bbox.y2, 4),
                    },
                }
                for d in detections
            ],
            "violations": violations,
            "frame_ms": round(frame_ms, 1),
            "annotated_jpeg": annotated_jpeg,
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _decode(self, jpeg_bytes: bytes) -> np.ndarray | None:
        try:
            import cv2  # type: ignore[import-untyped]

            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame if frame is not None else None
        except Exception:
            logger.exception("Frame decode failed")
            return None

    def _detect(self, frame: np.ndarray) -> list:
        if self._detector is None:
            return []
        try:
            return self._detector.detect(frame)
        except Exception:
            logger.exception("Detector.detect() raised")
            return []

    def _match_rules(self, detections: list, rules: list[str]) -> list[str]:
        """Return the subset of rules that are violated.

        Rule semantics
        --------------
        * **Absence rules** (contain e.g. "未着用", "missing"):
          Triggered when the required PPE equipment is *not* detected.
        * **Presence rules** (contain e.g. "侵入", "presence"):
          Triggered when a matching label *is* detected.
        * **Default** (no keyword matched):
          Triggered when no detected label appears in the rule string.
        """
        if not rules:
            return []

        detected_labels = {d.label.lower() for d in detections}
        violated: list[str] = []

        for rule in rules:
            rule_s = rule.strip()
            if not rule_s:
                continue
            rule_l = rule_s.lower()

            if any(kw in rule_l for kw in _ABSENCE_KEYWORDS):
                # Safety-absence rule: violation if required equipment missing
                triggered = False
                for equip_jp, equip_labels in _EQUIP_MAP.items():
                    if equip_jp.lower() in rule_l or any(e in rule_l for e in equip_labels):
                        if not any(e in detected_labels for e in equip_labels):
                            triggered = True
                            break
                if triggered:
                    violated.append(rule_s)

            elif any(kw in rule_l for kw in _PRESENCE_KEYWORDS):
                # Presence rule (e.g. "立入禁止エリア侵入"): violation if anyone is detected
                if detected_labels:
                    violated.append(rule_s)

            else:
                # Generic rule: violation if rule tokens don't match any detection
                rule_tokens = set(rule_l.split())
                if not (rule_tokens & detected_labels):
                    violated.append(rule_s)

        return violated

    def _draw_boxes(
        self, frame: np.ndarray, detections: list, violations: list[str], original_jpeg: bytes = b""
    ) -> bytes:
        try:
            import cv2  # type: ignore[import-untyped]

            annotated = frame.copy()
            h, w = annotated.shape[:2]
            has_violation = bool(violations)

            for d in detections:
                color = (50, 50, 220) if has_violation else (50, 200, 80)
                x1 = int(d.bbox.x1 * w)
                y1 = int(d.bbox.y1 * h)
                x2 = int(d.bbox.x2 * w)
                y2 = int(d.bbox.y2 * h)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label_text = f"{d.label} {d.confidence:.2f}"
                cv2.putText(
                    annotated,
                    label_text,
                    (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            if has_violation:
                # Overlay violation banner
                banner = f"VIOLATION: {violations[0]}"
                cv2.rectangle(annotated, (0, 0), (w, 22), (0, 0, 180), -1)
                cv2.putText(
                    annotated,
                    banner,
                    (6, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
            return bytes(buf)
        except Exception:
            logger.exception("bbox annotation failed")
            return original_jpeg
