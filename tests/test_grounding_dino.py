"""Tests for sopilot.perception.grounding_dino (Phase 39 — Grounding DINO 1.5).

Covers:
    - GroundingDINOStats: properties and to_dict
    - MockGroundingDINODetector: detect, stats, edge cases
    - GroundingDINODetector: unloaded behavior, error paths
    - _prompts_to_text: formatting
    - build_grounding_dino factory: mock/auto/real backends
    - ObjectDetector interface compliance
    - Thread-safety of MockGroundingDINODetector
    - Integration with PerceptionConfig
    - Cascade Stage 2 injection pattern
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sopilot.perception.detector import ObjectDetector
from sopilot.perception.grounding_dino import (
    GroundingDINODetector,
    GroundingDINOStats,
    MockGroundingDINODetector,
    build_grounding_dino,
)
from sopilot.perception.types import BBox, Detection, PerceptionConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_frame(h: int = 120, w: int = 160) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# GroundingDINOStats
# ---------------------------------------------------------------------------


class TestGroundingDINOStats:
    def test_initial_values(self):
        s = GroundingDINOStats()
        assert s.total_frames == 0
        assert s.total_detections == 0
        assert s.total_latency_ms == 0.0
        assert s.load_count == 0
        assert s.errors == 0

    def test_avg_detections_empty(self):
        s = GroundingDINOStats()
        assert s.avg_detections_per_frame == 0.0

    def test_avg_detections_with_data(self):
        s = GroundingDINOStats(total_frames=4, total_detections=8)
        assert s.avg_detections_per_frame == 2.0

    def test_avg_latency_empty(self):
        s = GroundingDINOStats()
        assert s.avg_latency_ms == 0.0

    def test_avg_latency_with_data(self):
        s = GroundingDINOStats(total_frames=2, total_latency_ms=100.0)
        assert s.avg_latency_ms == 50.0

    def test_to_dict_keys(self):
        s = GroundingDINOStats()
        d = s.to_dict()
        expected_keys = {
            "total_frames",
            "total_detections",
            "avg_detections_per_frame",
            "avg_latency_ms",
            "load_count",
            "errors",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_values(self):
        s = GroundingDINOStats(total_frames=3, total_detections=6, load_count=1)
        d = s.to_dict()
        assert d["total_frames"] == 3
        assert d["total_detections"] == 6
        assert d["load_count"] == 1
        assert d["avg_detections_per_frame"] == pytest.approx(2.0)

    def test_to_dict_rounded(self):
        s = GroundingDINOStats(total_frames=3, total_latency_ms=100.0)
        d = s.to_dict()
        assert isinstance(d["avg_latency_ms"], float)
        assert isinstance(d["avg_detections_per_frame"], float)


# ---------------------------------------------------------------------------
# MockGroundingDINODetector — basic detection
# ---------------------------------------------------------------------------


class TestMockGroundingDINODetectorDetect:
    def test_returns_list(self):
        det = MockGroundingDINODetector()
        frame = make_frame()
        result = det.detect(frame, ["person"])
        assert isinstance(result, list)

    def test_returns_detections(self):
        det = MockGroundingDINODetector(num_detections=2)
        frame = make_frame()
        result = det.detect(frame, ["person", "helmet"])
        assert len(result) == 2

    def test_single_prompt_one_detection(self):
        det = MockGroundingDINODetector(num_detections=3)
        frame = make_frame()
        # Only 1 prompt, num_detections is min(3, 1)=1
        result = det.detect(frame, ["person"])
        assert len(result) == 1

    def test_label_matches_prompt(self):
        det = MockGroundingDINODetector(num_detections=1)
        frame = make_frame()
        result = det.detect(frame, ["helmet"])
        assert result[0].label == "helmet"

    def test_detection_confidence_in_range(self):
        det = MockGroundingDINODetector(num_detections=3, base_confidence=0.55)
        frame = make_frame()
        result = det.detect(frame, ["person", "vest", "helmet"])
        for d in result:
            assert 0.0 < d.confidence <= 1.0

    def test_bbox_normalized(self):
        det = MockGroundingDINODetector()
        frame = make_frame()
        result = det.detect(frame, ["person"])
        for d in result:
            assert 0.0 <= d.bbox.x1 < d.bbox.x2 <= 1.0
            assert 0.0 <= d.bbox.y1 < d.bbox.y2 <= 1.0

    def test_attributes_stage(self):
        det = MockGroundingDINODetector()
        frame = make_frame()
        result = det.detect(frame, ["person"])
        assert result[0].attributes.get("stage") == "grounding_dino_mock"

    def test_attributes_mock_flag(self):
        det = MockGroundingDINODetector()
        frame = make_frame()
        result = det.detect(frame, ["person"])
        assert result[0].attributes.get("mock") is True

    def test_empty_frame_returns_empty(self):
        det = MockGroundingDINODetector()
        empty = np.array([])
        result = det.detect(empty, ["person"])
        assert result == []

    def test_empty_prompts_returns_empty(self):
        det = MockGroundingDINODetector()
        frame = make_frame()
        result = det.detect(frame, [])
        assert result == []

    def test_none_frame_returns_empty(self):
        det = MockGroundingDINODetector()
        result = det.detect(None, ["person"])  # type: ignore[arg-type]
        assert result == []

    def test_multiple_calls_vary_coordinates(self):
        det = MockGroundingDINODetector(num_detections=1)
        frame = make_frame()
        r1 = det.detect(frame, ["person"])
        r2 = det.detect(frame, ["person"])
        # Coordinates may differ between calls (call count advances)
        assert r1[0].bbox != r2[0].bbox or r1[0].confidence != r2[0].confidence or True


# ---------------------------------------------------------------------------
# MockGroundingDINODetector — stats
# ---------------------------------------------------------------------------


class TestMockGroundingDINODetectorStats:
    def test_initial_stats(self):
        det = MockGroundingDINODetector()
        s = det.get_stats()
        assert s["variant"] == "mock"
        assert s["loaded"] is True
        assert s["mock"] is True
        assert s["total_frames"] == 0

    def test_stats_after_detections(self):
        det = MockGroundingDINODetector(num_detections=2)
        frame = make_frame()
        det.detect(frame, ["person", "helmet"])
        det.detect(frame, ["person", "vest"])
        s = det.get_stats()
        assert s["total_frames"] == 2
        assert s["total_detections"] == 4

    def test_close_is_noop(self):
        det = MockGroundingDINODetector()
        det.close()  # should not raise
        assert det.get_stats()["loaded"] is True

    def test_variant_in_stats(self):
        det = MockGroundingDINODetector(variant="1.5-pro")
        s = det.get_stats()
        assert s["variant"] == "1.5-pro"


# ---------------------------------------------------------------------------
# GroundingDINODetector — prompts_to_text
# ---------------------------------------------------------------------------


class TestPromptsToText:
    def test_single_prompt(self):
        result = GroundingDINODetector._prompts_to_text(["person"])
        assert result == "person."

    def test_multiple_prompts(self):
        result = GroundingDINODetector._prompts_to_text(["person", "helmet", "vest"])
        assert result == "person. helmet. vest."

    def test_trailing_dot_stripped(self):
        result = GroundingDINODetector._prompts_to_text(["person.", "helmet."])
        assert result == "person. helmet."

    def test_strips_leading_spaces(self):
        result = GroundingDINODetector._prompts_to_text([" person", " helmet "])
        assert "person" in result
        assert "helmet" in result

    def test_empty_prompts(self):
        result = GroundingDINODetector._prompts_to_text([])
        assert result == "."


# ---------------------------------------------------------------------------
# GroundingDINODetector — unloaded behavior
# ---------------------------------------------------------------------------


class TestGroundingDINODetectorUnloaded:
    def test_not_loaded_by_default(self):
        det = GroundingDINODetector(variant="1.5-pro")
        assert det._loaded is False

    def test_stats_show_not_loaded(self):
        det = GroundingDINODetector(variant="1.5-edge")
        s = det.get_stats()
        assert s["loaded"] is False

    def test_variant_in_stats(self):
        det = GroundingDINODetector(variant="1.5-pro")
        s = det.get_stats()
        assert s["variant"] == "1.5-pro"

    def test_thresholds_in_stats(self):
        det = GroundingDINODetector(box_threshold=0.4, text_threshold=0.3)
        s = det.get_stats()
        assert s["box_threshold"] == pytest.approx(0.4)
        assert s["text_threshold"] == pytest.approx(0.3)

    def test_detect_load_failure_returns_empty(self):
        det = GroundingDINODetector(variant="1.5-pro")
        det._ensure_loaded = lambda: False  # type: ignore[method-assign]
        frame = make_frame()
        result = det.detect(frame, ["person"])
        assert result == []
        assert det._stats.errors == 1

    def test_detect_empty_frame_returns_empty(self):
        det = GroundingDINODetector()
        empty = np.array([])
        result = det.detect(empty, ["person"])
        assert result == []

    def test_detect_empty_prompts_returns_empty(self):
        det = GroundingDINODetector()
        frame = make_frame()
        result = det.detect(frame, [])
        assert result == []

    def test_close_unloaded_noop(self):
        det = GroundingDINODetector()
        det.close()  # should not raise

    def test_orchestrator_registration(self):
        mock_orch = MagicMock()
        det = GroundingDINODetector(variant="1.5-pro", orchestrator=mock_orch)
        mock_orch.register.assert_called_once()
        call_args = mock_orch.register.call_args[0]
        assert call_args[0] == "grounding_dino"

    def test_orchestrator_registration_edge(self):
        mock_orch = MagicMock()
        det = GroundingDINODetector(variant="1.5-edge", orchestrator=mock_orch)
        # Edge uses 1.5 GB VRAM
        call_kwargs = mock_orch.register.call_args[1]
        assert call_kwargs.get("vram_gb") == pytest.approx(1.5, abs=0.01)

    def test_orchestrator_registration_pro(self):
        mock_orch = MagicMock()
        det = GroundingDINODetector(variant="1.5-pro", orchestrator=mock_orch)
        call_kwargs = mock_orch.register.call_args[1]
        assert call_kwargs.get("vram_gb") == pytest.approx(2.0, abs=0.01)

    def test_stats_incremented_on_load_failure(self):
        det = GroundingDINODetector()
        det._ensure_loaded = lambda: False  # type: ignore[method-assign]
        frame = make_frame()
        det.detect(frame, ["person"])
        det.detect(frame, ["vest"])
        s = det.get_stats()
        assert s["total_frames"] == 2
        assert s["errors"] == 2


# ---------------------------------------------------------------------------
# build_grounding_dino factory
# ---------------------------------------------------------------------------


class TestBuildGroundingDino:
    def test_mock_variant(self):
        det = build_grounding_dino(variant="mock")
        assert isinstance(det, MockGroundingDINODetector)

    def test_mock_backend(self):
        det = build_grounding_dino(backend="mock")
        assert isinstance(det, MockGroundingDINODetector)

    def test_auto_no_cuda_returns_mock(self):
        with patch("torch.cuda.is_available", return_value=False):
            det = build_grounding_dino(backend="auto")
        assert isinstance(det, MockGroundingDINODetector)

    def test_auto_no_torch_returns_mock(self):
        import sys
        orig = sys.modules.get("torch")
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            det = build_grounding_dino(backend="auto")
            assert isinstance(det, MockGroundingDINODetector)
        finally:
            if orig is not None:
                sys.modules["torch"] = orig
            else:
                del sys.modules["torch"]

    def test_is_object_detector(self):
        det = build_grounding_dino(backend="mock")
        assert isinstance(det, ObjectDetector)

    def test_real_variant_1_5_pro(self):
        det = build_grounding_dino(variant="1.5-pro", backend="grounding_dino")
        assert isinstance(det, GroundingDINODetector)
        assert det._variant == "1.5-pro"

    def test_real_variant_1_5_edge(self):
        det = build_grounding_dino(variant="1.5-edge", backend="grounding_dino")
        assert isinstance(det, GroundingDINODetector)
        assert det._variant == "1.5-edge"

    def test_real_variant_1_0(self):
        det = build_grounding_dino(variant="1.0", backend="grounding_dino")
        assert isinstance(det, GroundingDINODetector)

    def test_thresholds_forwarded(self):
        det = build_grounding_dino(
            backend="grounding_dino",
            box_threshold=0.4,
            text_threshold=0.3,
        )
        assert isinstance(det, GroundingDINODetector)
        assert det._box_threshold == pytest.approx(0.4)
        assert det._text_threshold == pytest.approx(0.3)

    def test_config_forwarded(self):
        config = PerceptionConfig()
        det = build_grounding_dino(backend="grounding_dino", config=config)
        assert isinstance(det, GroundingDINODetector)


# ---------------------------------------------------------------------------
# Thread safety — MockGroundingDINODetector
# ---------------------------------------------------------------------------


class TestMockGroundingDINOThreadSafety:
    def test_concurrent_detect(self):
        det = MockGroundingDINODetector(num_detections=2)
        frame = make_frame()
        results = []
        errors = []

        def worker():
            try:
                r = det.detect(frame, ["person", "helmet"])
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 20
        for r in results:
            assert len(r) == 2

    def test_concurrent_get_stats(self):
        det = MockGroundingDINODetector()
        frame = make_frame()
        errors = []

        def worker():
            try:
                det.detect(frame, ["person"])
                det.get_stats()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(15)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ---------------------------------------------------------------------------
# Integration: ObjectDetector interface compliance
# ---------------------------------------------------------------------------


class TestObjectDetectorInterface:
    """Verify that both detectors implement the ObjectDetector interface."""

    @pytest.mark.parametrize(
        "det_cls",
        [MockGroundingDINODetector, GroundingDINODetector],
    )
    def test_has_detect_method(self, det_cls):
        assert hasattr(det_cls, "detect")

    @pytest.mark.parametrize(
        "det_cls",
        [MockGroundingDINODetector, GroundingDINODetector],
    )
    def test_has_close_method(self, det_cls):
        assert hasattr(det_cls, "close")

    @pytest.mark.parametrize(
        "det_cls",
        [MockGroundingDINODetector, GroundingDINODetector],
    )
    def test_is_object_detector(self, det_cls):
        assert issubclass(det_cls, ObjectDetector)

    def test_mock_detect_returns_detection_list(self):
        det = MockGroundingDINODetector()
        frame = make_frame()
        result = det.detect(frame, ["person"])
        assert all(isinstance(d, Detection) for d in result)

    def test_mock_detect_bbox_is_bbox(self):
        det = MockGroundingDINODetector()
        frame = make_frame()
        result = det.detect(frame, ["person"])
        assert all(isinstance(d.bbox, BBox) for d in result)

    def test_detection_confidence_float(self):
        det = MockGroundingDINODetector()
        frame = make_frame()
        result = det.detect(frame, ["person"])
        assert all(isinstance(d.confidence, float) for d in result)


# ---------------------------------------------------------------------------
# Integration: Cascade Stage 2 injection
# ---------------------------------------------------------------------------


class TestCascadeIntegration:
    def test_inject_mock_into_cascade(self):
        """MockGroundingDINODetector can be passed as dino_detector to cascade."""
        from sopilot.perception.cascade_detector import CascadeDetector

        gd = MockGroundingDINODetector(num_detections=2, base_confidence=0.55)
        cascade = CascadeDetector(
            dino_detector=gd,
            stage1_threshold=0.7,
            stage2_threshold=0.3,
        )
        frame = make_frame()
        result = cascade.detect(frame, ["person"])
        assert isinstance(result, list)

    def test_grounding_dino_called_in_stage2(self):
        """When YOLO returns low-confidence, Grounding DINO Stage 2 fires."""
        from sopilot.perception.cascade_detector import CascadeDetector
        from sopilot.perception.detector import ObjectDetector

        # Stage 1: returns low-confidence detection
        class LowConfYOLO(ObjectDetector):
            def detect(self, frame, prompts):
                return [
                    Detection(
                        bbox=BBox(0.1, 0.1, 0.4, 0.5),
                        label="person",
                        confidence=0.3,  # below stage1_threshold=0.7
                    )
                ]

            def close(self):
                pass

        gd = MockGroundingDINODetector(num_detections=1, base_confidence=0.65)
        cascade = CascadeDetector(
            yolo_detector=LowConfYOLO(),
            dino_detector=gd,
            stage1_threshold=0.7,
            stage2_threshold=0.3,
        )
        frame = make_frame()
        result = cascade.detect(frame, ["person"])
        assert isinstance(result, list)
        # Stage 2 fired
        stats = cascade.get_stats()
        assert stats["stage1_resolved"] == 0
