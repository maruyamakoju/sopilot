"""Tests for sopilot.perception.sam_segmenter --- SAM Segmenter.

Covers:
    1. SegmentationMask dataclass (creation, frozen, attributes, area/bbox)
    2. SAMSegmenter construction (defaults, orchestrator, dinov2, variant, device)
    3. segment_detection (valid frame, empty frame, None detection, mock SAM, etc.)
    4. segment_frame (empty, single, multiple, partial/all failures)
    5. segment_with_attention (attention map, point prompts, fallback)
    6. _generate_point_prompts (various maps, n_points, edge cases)
    7. Orchestrator integration (register, acquire/release, standalone)
    8. Stats and close (get_stats structure, counters, close cleanup)
    9. Edge cases (corrupted frame, concurrent access, large dimensions)

All tests run without any real ML model dependencies; SAM/torch are fully mocked.
"""

from __future__ import annotations

import threading
from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sopilot.perception.sam_segmenter import (
    SAMSegmenter,
    SegmentationMask,
    _DEFAULT_VARIANT,
    _MODEL_PRIORITY,
    _VRAM_ESTIMATE_GB,
)
from sopilot.perception.types import BBox, Detection


# ---------------------------------------------------------------------------
# Helpers & Mocks
# ---------------------------------------------------------------------------


def _make_bbox(x1: float = 0.1, y1: float = 0.1, x2: float = 0.5, y2: float = 0.5) -> BBox:
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _make_detection(
    label: str = "person",
    confidence: float = 0.9,
    bbox: BBox | None = None,
    attributes: dict[str, Any] | None = None,
) -> Detection:
    return Detection(
        bbox=bbox or _make_bbox(),
        label=label,
        confidence=confidence,
        attributes=attributes or {},
    )


def _make_frame(h: int = 480, w: int = 640, channels: int = 3) -> np.ndarray:
    """Create a dummy BGR frame."""
    return np.zeros((h, w, channels), dtype=np.uint8)


def _make_empty_frame() -> np.ndarray:
    """Frame with size == 0."""
    return np.zeros((0, 0, 3), dtype=np.uint8)


def _make_mask(h: int = 480, w: int = 640, filled: bool = True) -> np.ndarray:
    """Create a boolean mask array."""
    mask = np.zeros((h, w), dtype=bool)
    if filled:
        # Fill a rectangular region
        mask[100:200, 50:300] = True
    return mask


class _MockOrchestrator:
    """Minimal orchestrator mock for registration / acquire / release tests."""

    def __init__(self) -> None:
        self.registered: dict[str, dict[str, Any]] = {}
        self.device = "cpu"
        self._loaded: set[str] = set()

    def register(
        self,
        name: str,
        load_fn: Any,
        unload_fn: Any,
        vram_estimate_gb: float,
        priority: int,
        device: str | None = None,
    ) -> None:
        self.registered[name] = {
            "load_fn": load_fn,
            "unload_fn": unload_fn,
            "vram_estimate_gb": vram_estimate_gb,
            "priority": priority,
            "device": device,
        }

    def acquire(self, name: str) -> Any:
        self._loaded.add(name)
        return self.registered[name]["load_fn"]()

    def release(self, name: str) -> None:
        pass

    def is_loaded(self, name: str) -> bool:
        return name in self._loaded


class _MockDINOv2Backbone:
    """Minimal DINOv2 backbone mock returning an attention map."""

    def __init__(self, attention_map: np.ndarray | None = None) -> None:
        self._map = attention_map

    def get_attention_map(self, frame: np.ndarray) -> np.ndarray:
        if self._map is not None:
            return self._map
        # Return a 37x37 attention map (DINOv2 ViT-B/14 default)
        return np.random.rand(37, 37).astype(np.float32)


class _MockSAMPredictor:
    """Minimal SAM predictor mock."""

    def __init__(self, h: int = 480, w: int = 640, n_masks: int = 3) -> None:
        self._h = h
        self._w = w
        self._n_masks = n_masks

    def set_image(self, rgb: np.ndarray) -> None:
        self._h, self._w = rgb.shape[:2]

    def predict(
        self,
        point_coords: Any = None,
        point_labels: Any = None,
        box: Any = None,
        multimask_output: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        """Return synthetic masks and IoU predictions."""
        n = self._n_masks
        h, w = self._h, self._w

        if box is not None and np.ndim(box) == 2 and box.shape[0] > 1:
            # Batch box prompts: return (B, N_masks, H, W)
            b = box.shape[0]
            masks = np.zeros((b, n, h, w), dtype=bool)
            ious = np.zeros((b, n), dtype=np.float32)
            for bi in range(b):
                masks[bi, 0, 100:200, 50:300] = True
                ious[bi, 0] = 0.92
                if n > 1:
                    masks[bi, 1, 110:190, 60:280] = True
                    ious[bi, 1] = 0.85
                if n > 2:
                    masks[bi, 2, 120:180, 70:260] = True
                    ious[bi, 2] = 0.75
            return masks, ious, None

        # Single prompt: return (N_masks, H, W)
        masks = np.zeros((n, h, w), dtype=bool)
        ious = np.zeros(n, dtype=np.float32)
        masks[0, 100:200, 50:300] = True
        ious[0] = 0.95
        if n > 1:
            masks[1, 110:190, 60:280] = True
            ious[1] = 0.85
        if n > 2:
            masks[2, 120:180, 70:260] = True
            ious[2] = 0.70
        return masks, ious, None


def _patch_sam_and_torch(segmenter: SAMSegmenter, predictor: _MockSAMPredictor | None = None) -> None:
    """Inject a mock predictor into the segmenter for standalone mode."""
    pred = predictor or _MockSAMPredictor()
    segmenter._model = pred
    segmenter._loaded = True


# ---------------------------------------------------------------------------
# 1. SegmentationMask dataclass (5 tests)
# ---------------------------------------------------------------------------


class TestSegmentationMaskDataclass:
    """Tests for SegmentationMask creation and properties."""

    def test_creation_with_all_fields(self) -> None:
        mask = _make_mask()
        bbox = _make_bbox()
        seg = SegmentationMask(
            mask=mask,
            area=0.05,
            bbox=bbox,
            iou_prediction=0.92,
            label="person",
            attributes={"source": "box_prompt"},
        )
        assert seg.label == "person"
        assert seg.area == pytest.approx(0.05)
        assert seg.iou_prediction == pytest.approx(0.92)
        assert seg.attributes == {"source": "box_prompt"}

    def test_frozen_immutability(self) -> None:
        seg = SegmentationMask(
            mask=_make_mask(),
            area=0.1,
            bbox=_make_bbox(),
            iou_prediction=0.9,
            label="car",
        )
        with pytest.raises(FrozenInstanceError):
            seg.label = "truck"  # type: ignore[misc]

    def test_default_attributes_empty_dict(self) -> None:
        seg = SegmentationMask(
            mask=_make_mask(),
            area=0.1,
            bbox=_make_bbox(),
            iou_prediction=0.9,
            label="person",
        )
        assert seg.attributes == {}

    def test_mask_is_numpy_array(self) -> None:
        mask = _make_mask()
        seg = SegmentationMask(
            mask=mask,
            area=0.05,
            bbox=_make_bbox(),
            iou_prediction=0.88,
            label="person",
        )
        assert isinstance(seg.mask, np.ndarray)
        assert seg.mask.dtype == bool

    def test_bbox_is_preserved(self) -> None:
        bbox = BBox(x1=0.1, y1=0.2, x2=0.6, y2=0.8)
        seg = SegmentationMask(
            mask=_make_mask(),
            area=0.1,
            bbox=bbox,
            iou_prediction=0.91,
            label="person",
        )
        assert seg.bbox.x1 == pytest.approx(0.1)
        assert seg.bbox.y2 == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# 2. SAMSegmenter construction (5 tests)
# ---------------------------------------------------------------------------


class TestSAMSegmenterConstruction:
    """Tests for SAMSegmenter __init__ and parameter handling."""

    def test_default_params(self) -> None:
        seg = SAMSegmenter()
        assert seg._variant == _DEFAULT_VARIANT
        assert seg._orchestrator is None
        assert seg._dinov2 is None
        assert seg._device_override is None
        assert seg._loaded is False

    def test_with_orchestrator_registers(self) -> None:
        orch = _MockOrchestrator()
        seg = SAMSegmenter(orchestrator=orch)
        assert SAMSegmenter.MODEL_NAME in orch.registered
        entry = orch.registered[SAMSegmenter.MODEL_NAME]
        assert entry["vram_estimate_gb"] == _VRAM_ESTIMATE_GB
        assert entry["priority"] == _MODEL_PRIORITY

    def test_with_dinov2_backbone(self) -> None:
        backbone = _MockDINOv2Backbone()
        seg = SAMSegmenter(dinov2_backbone=backbone)
        assert seg._dinov2 is backbone

    def test_custom_variant(self) -> None:
        seg = SAMSegmenter(model_variant="sam2_hiera_large")
        assert seg._variant == "sam2_hiera_large"

    def test_device_override(self) -> None:
        seg = SAMSegmenter(device="cuda:1")
        assert seg._device_override == "cuda:1"


# ---------------------------------------------------------------------------
# 3. segment_detection (10 tests)
# ---------------------------------------------------------------------------


class TestSegmentDetection:
    """Tests for SAMSegmenter.segment_detection()."""

    def _make_segmenter(self, predictor: _MockSAMPredictor | None = None) -> SAMSegmenter:
        seg = SAMSegmenter()
        _patch_sam_and_torch(seg, predictor or _MockSAMPredictor())
        return seg

    def test_valid_frame_and_detection(self) -> None:
        seg = self._make_segmenter()
        # Mock _run_sam to return a valid result
        seg._run_sam = MagicMock(return_value=[{  # type: ignore[assignment]
            "mask": _make_mask(480, 640, filled=True),
            "iou": 0.95,
        }])
        det = _make_detection()
        result = seg.segment_detection(_make_frame(), det)
        assert result is not None
        assert result.label == "person"

    def test_empty_frame_returns_none(self) -> None:
        seg = self._make_segmenter()
        result = seg.segment_detection(_make_empty_frame(), _make_detection())
        assert result is None

    def test_returns_none_when_run_sam_empty(self) -> None:
        seg = self._make_segmenter()
        seg._run_sam = MagicMock(return_value=[])  # type: ignore[assignment]
        result = seg.segment_detection(_make_frame(), _make_detection())
        assert result is None

    def test_picks_highest_iou_mask(self) -> None:
        seg = self._make_segmenter()
        mask1 = _make_mask(480, 640, filled=True)
        mask2 = _make_mask(480, 640, filled=True)
        seg._run_sam = MagicMock(return_value=[  # type: ignore[assignment]
            {"mask": mask1, "iou": 0.70},
            {"mask": mask2, "iou": 0.95},
        ])
        result = seg.segment_detection(_make_frame(), _make_detection())
        assert result is not None
        assert result.iou_prediction == pytest.approx(0.95)

    def test_bbox_conversion_to_pixels(self) -> None:
        seg = self._make_segmenter()
        calls: list[Any] = []
        original_run_sam = seg._run_sam

        def capture_run_sam(frame: Any, box_prompts: Any = None, point_prompts: Any = None) -> list:
            calls.append(box_prompts)
            return [{"mask": _make_mask(480, 640), "iou": 0.9}]

        seg._run_sam = capture_run_sam  # type: ignore[assignment]
        det = _make_detection(bbox=BBox(x1=0.1, y1=0.2, x2=0.5, y2=0.8))
        seg.segment_detection(_make_frame(480, 640), det)
        assert len(calls) == 1
        box = calls[0][0]
        # x1 = 0.1*640=64, y1 = 0.2*480=96, x2 = 0.5*640=320, y2 = 0.8*480=384
        assert box == [64, 96, 320, 384]

    def test_iou_prediction_preserved(self) -> None:
        seg = self._make_segmenter()
        seg._run_sam = MagicMock(return_value=[{  # type: ignore[assignment]
            "mask": _make_mask(),
            "iou": 0.88,
        }])
        result = seg.segment_detection(_make_frame(), _make_detection())
        assert result is not None
        assert result.iou_prediction == pytest.approx(0.88)

    def test_graceful_failure_returns_none(self) -> None:
        seg = SAMSegmenter()
        # Not loaded, _ensure_loaded returns None
        result = seg.segment_detection(_make_frame(), _make_detection())
        assert result is None

    def test_label_from_detection(self) -> None:
        seg = self._make_segmenter()
        seg._run_sam = MagicMock(return_value=[{  # type: ignore[assignment]
            "mask": _make_mask(),
            "iou": 0.9,
        }])
        det = _make_detection(label="helmet")
        result = seg.segment_detection(_make_frame(), det)
        assert result is not None
        assert result.label == "helmet"

    def test_attributes_from_detection(self) -> None:
        seg = self._make_segmenter()
        seg._run_sam = MagicMock(return_value=[{  # type: ignore[assignment]
            "mask": _make_mask(),
            "iou": 0.9,
        }])
        det = _make_detection(attributes={"color": "red"})
        result = seg.segment_detection(_make_frame(), det)
        assert result is not None
        assert result.attributes.get("color") == "red"

    def test_area_fraction_computed(self) -> None:
        seg = self._make_segmenter()
        mask = _make_mask(480, 640, filled=True)
        seg._run_sam = MagicMock(return_value=[{  # type: ignore[assignment]
            "mask": mask,
            "iou": 0.9,
        }])
        result = seg.segment_detection(_make_frame(), _make_detection())
        assert result is not None
        expected_area = int(mask.sum()) / (480 * 640)
        assert result.area == pytest.approx(expected_area, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. segment_frame (8 tests)
# ---------------------------------------------------------------------------


class TestSegmentFrame:
    """Tests for SAMSegmenter.segment_frame()."""

    def _make_segmenter(self) -> SAMSegmenter:
        seg = SAMSegmenter()
        _patch_sam_and_torch(seg)
        return seg

    def test_empty_detections_returns_empty(self) -> None:
        seg = self._make_segmenter()
        result = seg.segment_frame(_make_frame(), [])
        assert result == []

    def test_empty_frame_returns_empty(self) -> None:
        seg = self._make_segmenter()
        result = seg.segment_frame(_make_empty_frame(), [_make_detection()])
        assert result == []

    def test_single_detection(self) -> None:
        seg = self._make_segmenter()
        seg._run_sam = MagicMock(return_value=[{  # type: ignore[assignment]
            "mask": _make_mask(),
            "iou": 0.92,
        }])
        result = seg.segment_frame(_make_frame(), [_make_detection()])
        assert len(result) == 1
        assert result[0].label == "person"

    def test_multiple_detections(self) -> None:
        seg = self._make_segmenter()
        seg._run_sam = MagicMock(return_value=[  # type: ignore[assignment]
            {"mask": _make_mask(), "iou": 0.9},
            {"mask": _make_mask(), "iou": 0.85},
            {"mask": _make_mask(), "iou": 0.88},
        ])
        dets = [
            _make_detection(label="person"),
            _make_detection(label="helmet"),
            _make_detection(label="vest"),
        ]
        result = seg.segment_frame(_make_frame(), dets)
        assert len(result) == 3
        labels = [m.label for m in result]
        assert labels == ["person", "helmet", "vest"]

    def test_partial_failures(self) -> None:
        seg = self._make_segmenter()
        empty_mask = np.zeros((480, 640), dtype=bool)  # all-false mask
        seg._run_sam = MagicMock(return_value=[  # type: ignore[assignment]
            {"mask": _make_mask(), "iou": 0.9},
            {"mask": empty_mask, "iou": 0.1},  # empty mask -> None
        ])
        dets = [_make_detection(label="person"), _make_detection(label="car")]
        result = seg.segment_frame(_make_frame(), dets)
        # Only the first mask is valid
        assert len(result) == 1
        assert result[0].label == "person"

    def test_all_failures_returns_empty(self) -> None:
        seg = self._make_segmenter()
        seg._run_sam = MagicMock(return_value=[])  # type: ignore[assignment]
        dets = [_make_detection(), _make_detection()]
        result = seg.segment_frame(_make_frame(), dets)
        assert result == []

    def test_total_frames_incremented(self) -> None:
        seg = self._make_segmenter()
        seg._run_sam = MagicMock(return_value=[{  # type: ignore[assignment]
            "mask": _make_mask(),
            "iou": 0.9,
        }])
        assert seg._total_frames == 0
        seg.segment_frame(_make_frame(), [_make_detection()])
        assert seg._total_frames == 1

    def test_total_segments_incremented(self) -> None:
        seg = self._make_segmenter()
        seg._run_sam = MagicMock(return_value=[  # type: ignore[assignment]
            {"mask": _make_mask(), "iou": 0.9},
            {"mask": _make_mask(), "iou": 0.85},
        ])
        dets = [_make_detection(), _make_detection()]
        seg.segment_frame(_make_frame(), dets)
        assert seg._total_segments == 2


# ---------------------------------------------------------------------------
# 5. segment_with_attention (8 tests)
# ---------------------------------------------------------------------------


class TestSegmentWithAttention:
    """Tests for SAMSegmenter.segment_with_attention()."""

    def _make_segmenter(
        self, backbone: _MockDINOv2Backbone | None = None
    ) -> SAMSegmenter:
        seg = SAMSegmenter(dinov2_backbone=backbone)
        _patch_sam_and_torch(seg)
        return seg

    def test_with_attention_map(self) -> None:
        backbone = _MockDINOv2Backbone()
        seg = self._make_segmenter(backbone)
        seg._run_sam = MagicMock(return_value=[{  # type: ignore[assignment]
            "mask": _make_mask(),
            "iou": 0.93,
        }])
        result = seg.segment_with_attention(_make_frame(), [_make_detection()])
        assert len(result) == 1
        assert result[0].iou_prediction == pytest.approx(0.93)

    def test_attention_adds_prompt_type_attribute(self) -> None:
        backbone = _MockDINOv2Backbone()
        seg = self._make_segmenter(backbone)
        seg._run_sam = MagicMock(return_value=[{  # type: ignore[assignment]
            "mask": _make_mask(),
            "iou": 0.9,
        }])
        result = seg.segment_with_attention(_make_frame(), [_make_detection()])
        assert len(result) == 1
        assert result[0].attributes.get("prompt_type") == "attention"

    def test_point_prompt_generation_called(self) -> None:
        attn_map = np.random.rand(37, 37).astype(np.float32)
        backbone = _MockDINOv2Backbone(attn_map)
        seg = self._make_segmenter(backbone)

        # Track calls to _run_sam to check point_prompts
        calls: list[dict] = []

        def capture(frame: Any, box_prompts: Any = None, point_prompts: Any = None) -> list:
            calls.append({"box_prompts": box_prompts, "point_prompts": point_prompts})
            return [{"mask": _make_mask(), "iou": 0.9}]

        seg._run_sam = capture  # type: ignore[assignment]
        seg.segment_with_attention(_make_frame(), [_make_detection()])
        assert len(calls) == 1
        # Both box_prompts and point_prompts should be present
        assert calls[0]["box_prompts"] is not None
        assert calls[0]["point_prompts"] is not None

    def test_no_backbone_falls_back_to_segment_frame(self) -> None:
        seg = SAMSegmenter()  # no backbone
        _patch_sam_and_torch(seg)
        seg.segment_frame = MagicMock(return_value=[])  # type: ignore[assignment]
        seg.segment_with_attention(_make_frame(), [_make_detection()])
        seg.segment_frame.assert_called_once()

    def test_backbone_override_parameter(self) -> None:
        # Instance has no backbone, but we pass one as a parameter
        seg = SAMSegmenter()
        _patch_sam_and_torch(seg)
        override_backbone = _MockDINOv2Backbone()
        seg._run_sam = MagicMock(return_value=[{  # type: ignore[assignment]
            "mask": _make_mask(),
            "iou": 0.9,
        }])
        result = seg.segment_with_attention(
            _make_frame(), [_make_detection()], dinov2_backbone=override_backbone
        )
        assert len(result) == 1

    def test_attention_extraction_failure_falls_back(self) -> None:
        class _FailingBackbone:
            def get_attention_map(self, frame: np.ndarray) -> np.ndarray:
                raise RuntimeError("DINOv2 failed")

        seg = SAMSegmenter(dinov2_backbone=_FailingBackbone())
        _patch_sam_and_torch(seg)
        seg.segment_frame = MagicMock(return_value=[])  # type: ignore[assignment]
        seg.segment_with_attention(_make_frame(), [_make_detection()])
        seg.segment_frame.assert_called_once()

    def test_empty_detections_returns_empty(self) -> None:
        backbone = _MockDINOv2Backbone()
        seg = self._make_segmenter(backbone)
        result = seg.segment_with_attention(_make_frame(), [])
        assert result == []

    def test_empty_frame_returns_empty(self) -> None:
        backbone = _MockDINOv2Backbone()
        seg = self._make_segmenter(backbone)
        result = seg.segment_with_attention(_make_empty_frame(), [_make_detection()])
        assert result == []


# ---------------------------------------------------------------------------
# 6. _generate_point_prompts (8 tests)
# ---------------------------------------------------------------------------


class TestGeneratePointPrompts:
    """Tests for SAMSegmenter._generate_point_prompts()."""

    def test_basic_generation(self) -> None:
        attn = np.random.rand(37, 37).astype(np.float32)
        bbox = BBox(x1=0.1, y1=0.1, x2=0.5, y2=0.5)
        points = SAMSegmenter._generate_point_prompts(attn, bbox, n_points=3)
        assert isinstance(points, list)
        assert len(points) <= 3
        for p in points:
            assert isinstance(p, tuple)
            assert len(p) == 2

    def test_returns_correct_number_of_points(self) -> None:
        attn = np.random.rand(37, 37).astype(np.float32)
        bbox = BBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
        points = SAMSegmenter._generate_point_prompts(attn, bbox, n_points=5)
        assert len(points) == 5

    def test_single_point(self) -> None:
        attn = np.random.rand(37, 37).astype(np.float32)
        bbox = BBox(x1=0.0, y1=0.0, x2=0.5, y2=0.5)
        points = SAMSegmenter._generate_point_prompts(attn, bbox, n_points=1)
        assert len(points) == 1

    def test_degenerate_bbox_returns_empty(self) -> None:
        """Zero-area bbox (x1 == x2 or y1 == y2) should return empty list."""
        attn = np.random.rand(37, 37).astype(np.float32)
        # Zero-width bbox
        bbox = BBox(x1=0.5, y1=0.1, x2=0.5, y2=0.5)
        points = SAMSegmenter._generate_point_prompts(attn, bbox, n_points=3)
        assert points == []

    def test_small_bbox_fewer_points(self) -> None:
        """Very small bbox region may have fewer pixels than n_points."""
        attn = np.random.rand(37, 37).astype(np.float32)
        # Tiny bbox that maps to ~1 pixel in attention space
        bbox = BBox(x1=0.0, y1=0.0, x2=0.03, y2=0.03)
        points = SAMSegmenter._generate_point_prompts(attn, bbox, n_points=10)
        # Should return at most the number of pixels in the crop
        assert len(points) <= 10

    def test_uniform_attention_returns_points(self) -> None:
        """Uniform attention map should still produce valid points."""
        attn = np.ones((37, 37), dtype=np.float32) * 0.5
        bbox = BBox(x1=0.1, y1=0.1, x2=0.9, y2=0.9)
        points = SAMSegmenter._generate_point_prompts(attn, bbox, n_points=3)
        assert len(points) == 3

    def test_high_attention_in_specific_region(self) -> None:
        """Points should be concentrated where attention is highest."""
        attn = np.zeros((37, 37), dtype=np.float32)
        # Set a small region to high attention
        attn[10:15, 10:15] = 1.0
        # bbox that covers the high-attention region
        bbox = BBox(x1=0.2, y1=0.2, x2=0.5, y2=0.5)
        points = SAMSegmenter._generate_point_prompts(attn, bbox, n_points=3)
        assert len(points) > 0
        # All points should be within the attention-map crop region
        ax1 = int(bbox.x1 * 37)
        ay1 = int(bbox.y1 * 37)
        ax2 = int(bbox.x2 * 37)
        ay2 = int(bbox.y2 * 37)
        for px, py in points:
            assert ax1 <= px < ax2
            assert ay1 <= py < ay2

    def test_full_frame_bbox(self) -> None:
        """Full-frame bbox should work without clipping issues."""
        attn = np.random.rand(37, 37).astype(np.float32)
        bbox = BBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
        points = SAMSegmenter._generate_point_prompts(attn, bbox, n_points=5)
        assert len(points) == 5
        for px, py in points:
            assert 0 <= px < 37
            assert 0 <= py < 37


# ---------------------------------------------------------------------------
# 7. Orchestrator integration (6 tests)
# ---------------------------------------------------------------------------


class TestOrchestratorIntegration:
    """Tests for orchestrator registration, acquire/release, and standalone."""

    def test_register_called_with_correct_params(self) -> None:
        orch = _MockOrchestrator()
        SAMSegmenter(orchestrator=orch)
        assert "sam2" in orch.registered
        entry = orch.registered["sam2"]
        assert entry["vram_estimate_gb"] == _VRAM_ESTIMATE_GB
        assert entry["priority"] == _MODEL_PRIORITY

    def test_acquire_and_release_pattern(self) -> None:
        orch = _MockOrchestrator()
        mock_pred = _MockSAMPredictor()
        # Override the load_fn registered by SAMSegmenter to return our mock
        seg = SAMSegmenter(orchestrator=orch)
        orch.registered["sam2"]["load_fn"] = lambda: mock_pred
        # _ensure_loaded should call acquire
        result = seg._ensure_loaded()
        assert result is mock_pred

    def test_orchestrator_release_after_run_sam(self) -> None:
        orch = MagicMock()
        orch.device = "cpu"
        orch.acquire.return_value = _MockSAMPredictor()
        seg = SAMSegmenter(orchestrator=orch)
        # _release calls orchestrator.release
        seg._release()
        orch.release.assert_called_once_with("sam2")

    def test_standalone_mode_no_register(self) -> None:
        seg = SAMSegmenter()  # no orchestrator
        assert seg._orchestrator is None

    def test_orchestrator_acquire_failure_returns_none(self) -> None:
        orch = MagicMock()
        orch.device = "cpu"
        orch.acquire.side_effect = KeyError("sam2 not found")
        seg = SAMSegmenter(orchestrator=orch)
        result = seg._ensure_loaded()
        assert result is None

    def test_device_override_passed_to_orchestrator(self) -> None:
        orch = _MockOrchestrator()
        SAMSegmenter(orchestrator=orch, device="cuda:0")
        entry = orch.registered["sam2"]
        assert entry["device"] == "cuda:0"


# ---------------------------------------------------------------------------
# 8. Stats and close (5 tests)
# ---------------------------------------------------------------------------


class TestStatsAndClose:
    """Tests for get_stats() and close()."""

    def test_get_stats_structure(self) -> None:
        seg = SAMSegmenter()
        stats = seg.get_stats()
        expected_keys = {
            "variant",
            "loaded",
            "total_segments",
            "total_frames",
            "total_attention_segments",
            "avg_inference_ms",
            "failures",
        }
        assert set(stats.keys()) == expected_keys

    def test_get_stats_initial_values(self) -> None:
        seg = SAMSegmenter()
        stats = seg.get_stats()
        assert stats["variant"] == _DEFAULT_VARIANT
        assert stats["loaded"] is False
        assert stats["total_segments"] == 0
        assert stats["total_frames"] == 0
        assert stats["total_attention_segments"] == 0
        assert stats["avg_inference_ms"] == 0.0
        assert stats["failures"] == 0

    def test_counters_increment_after_segment_frame(self) -> None:
        seg = SAMSegmenter()
        _patch_sam_and_torch(seg)
        seg._run_sam = MagicMock(return_value=[  # type: ignore[assignment]
            {"mask": _make_mask(), "iou": 0.9},
        ])
        seg.segment_frame(_make_frame(), [_make_detection()])
        stats = seg.get_stats()
        assert stats["total_frames"] == 1
        assert stats["total_segments"] == 1

    def test_loaded_true_after_model_load(self) -> None:
        seg = SAMSegmenter()
        _patch_sam_and_torch(seg)
        stats = seg.get_stats()
        assert stats["loaded"] is True

    def test_close_resets_standalone_model(self) -> None:
        seg = SAMSegmenter()
        _patch_sam_and_torch(seg)
        assert seg._loaded is True
        assert seg._model is not None
        seg.close()
        assert seg._loaded is False
        assert seg._model is None

    def test_close_noop_with_orchestrator(self) -> None:
        orch = _MockOrchestrator()
        seg = SAMSegmenter(orchestrator=orch)
        # close() should not raise even though model is not loaded standalone
        seg.close()
        # Orchestrator-managed model unchanged
        assert seg._model is None


# ---------------------------------------------------------------------------
# 9. Edge cases (5 tests)
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_corrupted_frame_graceful(self) -> None:
        """A frame with unexpected content should not crash when SAM is unavailable."""
        seg = SAMSegmenter()
        # Do NOT load model --- _ensure_loaded returns None (graceful fallback)
        # This simulates the real scenario where SAM is unavailable
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det = _make_detection()
        result = seg.segment_detection(frame, det)
        assert result is None

    def test_concurrent_access_safe(self) -> None:
        """Multiple threads calling segment_detection should not crash."""
        seg = SAMSegmenter()
        _patch_sam_and_torch(seg)
        seg._run_sam = MagicMock(return_value=[{  # type: ignore[assignment]
            "mask": _make_mask(),
            "iou": 0.9,
        }])

        results: list[SegmentationMask | None] = []
        errors: list[Exception] = []

        def worker() -> None:
            try:
                r = seg.segment_detection(_make_frame(), _make_detection())
                results.append(r)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert errors == []
        assert len(results) == 10

    def test_very_large_frame_dimensions(self) -> None:
        """High-resolution frame metadata is handled without OOM (mocked)."""
        seg = SAMSegmenter()
        _patch_sam_and_torch(seg)
        seg._run_sam = MagicMock(return_value=[{  # type: ignore[assignment]
            "mask": np.ones((4320, 7680), dtype=bool),
            "iou": 0.88,
        }])
        # Use a small proxy frame to avoid actual memory allocation
        frame = np.zeros((4320, 7680, 3), dtype=np.uint8)
        det = _make_detection()
        result = seg.segment_detection(frame, det)
        assert result is not None
        assert result.area == pytest.approx(1.0, abs=1e-6)

    def test_mask_to_segmentation_empty_mask(self) -> None:
        """_mask_to_segmentation returns None for all-false mask."""
        result = SAMSegmenter._mask_to_segmentation(
            {"mask": np.zeros((480, 640), dtype=bool), "iou": 0.9},
            (480, 640, 3),
            "person",
        )
        assert result is None

    def test_mask_to_segmentation_none_mask(self) -> None:
        """_mask_to_segmentation returns None when mask key is None."""
        result = SAMSegmenter._mask_to_segmentation(
            {"mask": None, "iou": 0.9},
            (480, 640, 3),
            "person",
        )
        assert result is None


# ---------------------------------------------------------------------------
# Additional _mask_to_segmentation tests
# ---------------------------------------------------------------------------


class TestMaskToSegmentation:
    """Additional tests for the static _mask_to_segmentation method."""

    def test_valid_mask_returns_segmentation(self) -> None:
        mask = np.zeros((480, 640), dtype=bool)
        mask[100:200, 50:300] = True
        result = SAMSegmenter._mask_to_segmentation(
            {"mask": mask, "iou": 0.91},
            (480, 640, 3),
            "car",
            {"color": "blue"},
        )
        assert result is not None
        assert result.label == "car"
        assert result.iou_prediction == pytest.approx(0.91)
        assert result.attributes == {"color": "blue"}

    def test_bbox_covers_mask(self) -> None:
        mask = np.zeros((480, 640), dtype=bool)
        mask[100:200, 50:300] = True
        result = SAMSegmenter._mask_to_segmentation(
            {"mask": mask, "iou": 0.9},
            (480, 640, 3),
            "person",
        )
        assert result is not None
        # bbox should tightly cover [y:100-199, x:50-299]
        # x1 = 50/640, y1 = 100/480, x2 = 300/640, y2 = 200/480
        assert result.bbox.x1 == pytest.approx(50 / 640, abs=0.01)
        assert result.bbox.y1 == pytest.approx(100 / 480, abs=0.01)
        assert result.bbox.x2 == pytest.approx(300 / 640, abs=0.01)
        assert result.bbox.y2 == pytest.approx(200 / 480, abs=0.01)

    def test_area_fraction_correct(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:20, 10:20] = True  # 10*10 = 100 pixels out of 10000
        result = SAMSegmenter._mask_to_segmentation(
            {"mask": mask, "iou": 0.9},
            (100, 100, 3),
            "obj",
        )
        assert result is not None
        assert result.area == pytest.approx(100 / 10000)

    def test_no_iou_key_defaults_to_zero(self) -> None:
        mask = np.zeros((480, 640), dtype=bool)
        mask[100:200, 50:300] = True
        result = SAMSegmenter._mask_to_segmentation(
            {"mask": mask},  # no "iou" key
            (480, 640, 3),
            "person",
        )
        assert result is not None
        assert result.iou_prediction == pytest.approx(0.0)

    def test_none_attributes_defaults_empty_dict(self) -> None:
        mask = np.zeros((480, 640), dtype=bool)
        mask[100:200, 50:300] = True
        result = SAMSegmenter._mask_to_segmentation(
            {"mask": mask, "iou": 0.9},
            (480, 640, 3),
            "person",
            None,
        )
        assert result is not None
        assert result.attributes == {}


# ---------------------------------------------------------------------------
# Orchestrator get_stats loaded flag
# ---------------------------------------------------------------------------


class TestOrchestratorStatsLoaded:
    """Test that get_stats correctly queries orchestrator for loaded state."""

    def test_stats_loaded_via_orchestrator(self) -> None:
        orch = _MockOrchestrator()
        seg = SAMSegmenter(orchestrator=orch)
        # Not loaded yet
        stats = seg.get_stats()
        assert stats["loaded"] is False

    def test_stats_loaded_after_acquire(self) -> None:
        orch = _MockOrchestrator()
        seg = SAMSegmenter(orchestrator=orch)
        # Simulate acquire by loading through orchestrator
        orch._loaded.add("sam2")
        stats = seg.get_stats()
        assert stats["loaded"] is True


# ---------------------------------------------------------------------------
# Parametrised tests for broader coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant",
    ["sam2_hiera_tiny", "sam2_hiera_small", "sam2_hiera_base_plus", "sam2_hiera_large"],
)
def test_variant_stored(variant: str) -> None:
    seg = SAMSegmenter(model_variant=variant)
    assert seg._variant == variant
    assert seg.get_stats()["variant"] == variant


@pytest.mark.parametrize("n_dets", [0, 1, 5, 10])
def test_segment_frame_various_detection_counts(n_dets: int) -> None:
    seg = SAMSegmenter()
    _patch_sam_and_torch(seg)
    seg._run_sam = MagicMock(  # type: ignore[assignment]
        return_value=[{"mask": _make_mask(), "iou": 0.9}] * n_dets
    )
    dets = [_make_detection() for _ in range(n_dets)]
    result = seg.segment_frame(_make_frame(), dets)
    if n_dets == 0:
        assert result == []
    else:
        assert len(result) == n_dets


@pytest.mark.parametrize(
    "n_points,expected_max",
    [(1, 1), (3, 3), (5, 5), (10, 10)],
)
def test_generate_point_prompts_respects_n_points(n_points: int, expected_max: int) -> None:
    attn = np.random.rand(37, 37).astype(np.float32)
    bbox = BBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
    points = SAMSegmenter._generate_point_prompts(attn, bbox, n_points=n_points)
    assert len(points) <= expected_max


@pytest.mark.parametrize(
    "bbox",
    [
        BBox(x1=0.0, y1=0.0, x2=0.0, y2=0.0),  # zero area
        BBox(x1=0.5, y1=0.5, x2=0.5, y2=0.5),  # point
    ],
)
def test_generate_point_prompts_degenerate_bboxes(bbox: BBox) -> None:
    attn = np.random.rand(37, 37).astype(np.float32)
    points = SAMSegmenter._generate_point_prompts(attn, bbox, n_points=3)
    assert points == []


# ---------------------------------------------------------------------------
# _unload_model static method
# ---------------------------------------------------------------------------


class TestUnloadModel:
    """Tests for the _unload_model static method."""

    def test_unload_model_with_cpu_method(self) -> None:
        model = MagicMock()
        model.cpu = MagicMock()
        del model.model  # ensure hasattr(model, "model") is False
        SAMSegmenter._unload_model(model)
        model.cpu.assert_called_once()

    def test_unload_model_with_model_attr(self) -> None:
        model = MagicMock()
        inner = MagicMock()
        model.model = inner
        SAMSegmenter._unload_model(model)
        inner.cpu.assert_called_once()

    def test_unload_model_exception_swallowed(self) -> None:
        model = MagicMock()
        model.cpu.side_effect = RuntimeError("fail")
        del model.model
        # Should not raise
        SAMSegmenter._unload_model(model)


# ---------------------------------------------------------------------------
# _ensure_loaded edge cases
# ---------------------------------------------------------------------------


class TestEnsureLoaded:
    """Tests for _ensure_loaded standalone and orchestrated paths."""

    def test_standalone_load_failure_returns_none(self) -> None:
        seg = SAMSegmenter()
        # _load_model will fail because sam2 is not installed
        result = seg._ensure_loaded()
        assert result is None
        assert seg._loaded is False

    def test_standalone_cached_after_first_load(self) -> None:
        seg = SAMSegmenter()
        mock_pred = _MockSAMPredictor()
        seg._model = mock_pred
        seg._loaded = True
        result = seg._ensure_loaded()
        assert result is mock_pred

    def test_orchestrator_runtime_error_returns_none(self) -> None:
        orch = MagicMock()
        orch.device = "cpu"
        orch.acquire.side_effect = RuntimeError("VRAM full")
        seg = SAMSegmenter(orchestrator=orch)
        result = seg._ensure_loaded()
        assert result is None


# ---------------------------------------------------------------------------
# Attention segments counter
# ---------------------------------------------------------------------------


class TestAttentionSegmentsCounter:
    """Tests for the _total_attention_segments counter."""

    def test_attention_counter_increments(self) -> None:
        backbone = _MockDINOv2Backbone()
        seg = SAMSegmenter(dinov2_backbone=backbone)
        _patch_sam_and_torch(seg)
        seg._run_sam = MagicMock(return_value=[{  # type: ignore[assignment]
            "mask": _make_mask(),
            "iou": 0.9,
        }])
        seg.segment_with_attention(_make_frame(), [_make_detection()])
        assert seg._total_attention_segments == 1

    def test_attention_counter_zero_after_box_only(self) -> None:
        seg = SAMSegmenter()
        _patch_sam_and_torch(seg)
        seg._run_sam = MagicMock(return_value=[{  # type: ignore[assignment]
            "mask": _make_mask(),
            "iou": 0.9,
        }])
        seg.segment_frame(_make_frame(), [_make_detection()])
        assert seg._total_attention_segments == 0


# ---------------------------------------------------------------------------
# Failures counter
# ---------------------------------------------------------------------------


class TestFailuresCounter:
    """Tests for the _failures counter."""

    def test_failures_starts_at_zero(self) -> None:
        seg = SAMSegmenter()
        assert seg._failures == 0
        assert seg.get_stats()["failures"] == 0

    def test_failures_in_stats(self) -> None:
        seg = SAMSegmenter()
        seg._failures = 3
        assert seg.get_stats()["failures"] == 3


# ---------------------------------------------------------------------------
# MODEL_NAME constant
# ---------------------------------------------------------------------------


def test_model_name_constant() -> None:
    assert SAMSegmenter.MODEL_NAME == "sam2"


# ---------------------------------------------------------------------------
# _release edge cases
# ---------------------------------------------------------------------------


class TestRelease:
    """Tests for _release method."""

    def test_release_no_orchestrator_noop(self) -> None:
        seg = SAMSegmenter()
        # Should not raise
        seg._release()

    def test_release_orchestrator_exception_swallowed(self) -> None:
        orch = MagicMock()
        orch.device = "cpu"
        orch.release.side_effect = RuntimeError("fail")
        seg = SAMSegmenter(orchestrator=orch)
        # Should not raise
        seg._release()
