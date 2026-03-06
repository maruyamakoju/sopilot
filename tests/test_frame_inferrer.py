"""Tests for FrameInferrer (vigil/frame_inferrer.py)."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sopilot.vigil.frame_inferrer import FrameInferrer


# ── Minimal Detection stub ────────────────────────────────────────────────────


@dataclass
class _BBox:
    x1: float = 0.1
    y1: float = 0.1
    x2: float = 0.4
    y2: float = 0.4


@dataclass
class _Det:
    label: str
    confidence: float
    bbox: _BBox = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.bbox is None:
            self.bbox = _BBox()


def _person(confidence: float = 0.9) -> _Det:
    return _Det(label="person", confidence=confidence)


def _helmet(confidence: float = 0.85) -> _Det:
    return _Det(label="helmet", confidence=confidence)


def _vest(confidence: float = 0.80) -> _Det:
    return _Det(label="safety vest", confidence=confidence)


def _make_jpeg(h: int = 64, w: int = 64) -> bytes:
    """Create a minimal valid JPEG bytes using OpenCV."""
    import cv2

    frame = np.zeros((h, w, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame)
    return bytes(buf)


# ── Constructor ───────────────────────────────────────────────────────────────


def test_inferrer_no_detector():
    fi = FrameInferrer()
    assert fi._detector is None


def test_inferrer_with_detector():
    det = MagicMock()
    fi = FrameInferrer(detector=det)
    assert fi._detector is det


# ── infer() ───────────────────────────────────────────────────────────────────


def test_infer_no_detector_returns_empty():
    fi = FrameInferrer()
    result = fi.infer(_make_jpeg(), rules=[])
    assert result["detections"] == []
    assert result["violations"] == []
    assert isinstance(result["frame_ms"], float)
    assert isinstance(result["annotated_jpeg"], bytes)
    assert len(result["annotated_jpeg"]) > 0


def test_infer_returns_required_keys():
    fi = FrameInferrer()
    result = fi.infer(_make_jpeg(), rules=[])
    assert set(result.keys()) >= {"detections", "violations", "frame_ms", "annotated_jpeg"}


def test_infer_with_detector_calls_detect():
    det = MagicMock()
    det.detect.return_value = [_person()]
    fi = FrameInferrer(detector=det)
    result = fi.infer(_make_jpeg(), rules=[])
    det.detect.assert_called_once()
    assert len(result["detections"]) == 1
    assert result["detections"][0]["label"] == "person"
    assert result["detections"][0]["confidence"] == 0.9


def test_infer_detection_bbox_structure():
    det = MagicMock()
    det.detect.return_value = [_Det(label="person", confidence=0.7, bbox=_BBox(0.1, 0.2, 0.5, 0.6))]
    fi = FrameInferrer(detector=det)
    result = fi.infer(_make_jpeg(), rules=[])
    bbox = result["detections"][0]["bbox"]
    assert "x1" in bbox and "y1" in bbox and "x2" in bbox and "y2" in bbox


def test_infer_bad_jpeg_returns_gracefully():
    fi = FrameInferrer()
    result = fi.infer(b"not-a-jpeg", rules=[])
    assert result["detections"] == []
    assert result["violations"] == []
    assert result["annotated_jpeg"] == b"not-a-jpeg"


def test_infer_empty_bytes_returns_gracefully():
    fi = FrameInferrer()
    result = fi.infer(b"", rules=[])
    assert result["detections"] == []


def test_infer_detector_exception_handled():
    det = MagicMock()
    det.detect.side_effect = RuntimeError("model error")
    fi = FrameInferrer(detector=det)  # noqa: F841
    result = fi.infer(_make_jpeg(), rules=[])
    assert result["detections"] == []


def test_infer_frame_ms_positive():
    fi = FrameInferrer()
    result = fi.infer(_make_jpeg(), rules=[])
    assert result["frame_ms"] >= 0.0


# ── _match_rules() ────────────────────────────────────────────────────────────


def test_match_rules_empty_rules_no_violation():
    fi = FrameInferrer()
    assert fi._match_rules([_person()], []) == []


def test_match_rules_absence_rule_triggered_when_no_helmet():
    fi = FrameInferrer()
    # No helmet detected → violation
    violated = fi._match_rules([_person()], ["ヘルメット未着用"])
    assert "ヘルメット未着用" in violated


def test_match_rules_absence_rule_not_triggered_when_helmet_present():
    fi = FrameInferrer()
    violated = fi._match_rules([_person(), _helmet()], ["ヘルメット未着用"])
    assert violated == []


def test_match_rules_absence_rule_vest():
    fi = FrameInferrer()
    violated = fi._match_rules([_person()], ["ベスト未着用"])
    assert "ベスト未着用" in violated


def test_match_rules_absence_rule_vest_present():
    fi = FrameInferrer()
    violated = fi._match_rules([_person(), _vest()], ["ベスト未着用"])
    assert violated == []


def test_match_rules_presence_rule_triggered():
    fi = FrameInferrer()
    violated = fi._match_rules([_person()], ["立入禁止エリア侵入"])
    assert "立入禁止エリア侵入" in violated


def test_match_rules_presence_rule_not_triggered_no_person():
    fi = FrameInferrer()
    # No person detected — presence rule not triggered
    violated = fi._match_rules([], ["立入禁止エリア侵入"])
    assert violated == []


def test_match_rules_multiple_rules():
    fi = FrameInferrer()
    dets = [_person()]
    violated = fi._match_rules(dets, ["ヘルメット未着用", "立入禁止エリア侵入"])
    assert "ヘルメット未着用" in violated


def test_match_rules_blank_rule_ignored():
    fi = FrameInferrer()
    violated = fi._match_rules([], ["", "  ", "\t"])
    assert violated == []


def test_match_rules_no_detections_no_violations():
    fi = FrameInferrer()
    assert fi._match_rules([], []) == []


# ── _draw_boxes() ─────────────────────────────────────────────────────────────


def test_draw_boxes_returns_bytes():
    import cv2

    fi = FrameInferrer()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    result = fi._draw_boxes(frame, [_person()], [])
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_draw_boxes_violation_banner():
    import cv2

    fi = FrameInferrer()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = fi._draw_boxes(frame, [_person()], ["ヘルメット未着用"])
    # Decode and check top rows are modified (banner)
    arr = np.frombuffer(result, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert decoded is not None
    # Banner occupies rows 0-22; at least some blue should be present
    assert decoded[:22, :, 2].max() > 0  # blue channel (BGR)


def test_draw_boxes_no_detector_no_crash():
    fi = FrameInferrer()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    result = fi._draw_boxes(frame, [], [])
    assert isinstance(result, bytes)
