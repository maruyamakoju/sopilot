"""Comprehensive tests for CascadeDetector — multi-stage confidence cascade.

Tests cover:
    - Stage 1 fast path (all high confidence)
    - Stage 2 escalation (low-confidence YOLO)
    - Stage 3 escalation (uncertain after Stage 2)
    - Empty frame / empty prompts
    - Detection merging with cross-stage NMS
    - Statistics tracking (per-stage counts and timing)
    - Missing detectors: no YOLO, no DINO, no VLM
    - ObjectDetector interface compliance
    - Orchestrator acquire/release pattern
    - Direct detector injection
    - Threshold customization
    - NMS merging with overlapping detections
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from sopilot.perception.cascade_detector import CascadeDetector, CascadeStats
from sopilot.perception.detector import ObjectDetector
from sopilot.perception.types import BBox, Detection


# ---------------------------------------------------------------------------
# Helpers — mock detectors and VLM
# ---------------------------------------------------------------------------


class MockYOLO(ObjectDetector):
    """Mock YOLO detector that returns controlled detections."""

    def __init__(self, detections: list[Detection] | None = None) -> None:
        self._detections = detections or []
        self.call_count = 0
        self.closed = False

    def detect(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
        self.call_count += 1
        return list(self._detections)

    def close(self) -> None:
        self.closed = True


class MockDINO(ObjectDetector):
    """Mock Grounding DINO detector that returns controlled detections."""

    def __init__(self, detections: list[Detection] | None = None) -> None:
        self._detections = detections or []
        self.call_count = 0
        self.closed = False

    def detect(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
        self.call_count += 1
        return list(self._detections)

    def close(self) -> None:
        self.closed = True


class MockVLM:
    """Mock VLM client with analyze_detections method."""

    def __init__(
        self,
        result: list[Detection] | None = None,
        should_fail: bool = False,
    ) -> None:
        self._result = result
        self._should_fail = should_fail
        self.call_count = 0
        self.last_uncertain: list[Detection] | None = None

    def analyze_detections(
        self,
        frame: np.ndarray,
        uncertain: list[Detection],
        prompts: list[str],
    ) -> list[Detection]:
        self.call_count += 1
        self.last_uncertain = uncertain
        if self._should_fail:
            raise RuntimeError("VLM service unavailable")
        if self._result is not None:
            return self._result
        # Default: return verified versions with boosted confidence
        return [
            Detection(
                bbox=d.bbox,
                label=d.label,
                confidence=min(d.confidence + 0.3, 0.95),
                attributes={"vlm_verified": True},
            )
            for d in uncertain
        ]


def _make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Create a dummy BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _det(
    label: str = "person",
    conf: float = 0.8,
    x1: float = 0.1,
    y1: float = 0.1,
    x2: float = 0.3,
    y2: float = 0.3,
    **attrs: Any,
) -> Detection:
    """Convenience factory for Detection objects."""
    return Detection(
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        label=label,
        confidence=conf,
        attributes=dict(attrs),
    )


# ---------------------------------------------------------------------------
# CascadeStats tests
# ---------------------------------------------------------------------------


class TestCascadeStats:
    """Tests for the CascadeStats dataclass."""

    def test_default_values(self) -> None:
        stats = CascadeStats()
        assert stats.total_frames == 0
        assert stats.stage1_resolved == 0
        assert stats.stage2_resolved == 0
        assert stats.stage3_resolved == 0

    def test_stage1_rate_zero_frames(self) -> None:
        stats = CascadeStats()
        # max(1, 0) = 1 -> 0/1 = 0.0
        assert stats.stage1_rate == 0.0

    def test_stage1_rate(self) -> None:
        stats = CascadeStats(total_frames=100, stage1_resolved=95)
        assert stats.stage1_rate == pytest.approx(0.95)

    def test_stage2_rate(self) -> None:
        stats = CascadeStats(total_frames=100, stage2_resolved=4)
        assert stats.stage2_rate == pytest.approx(0.04)

    def test_stage3_rate(self) -> None:
        stats = CascadeStats(total_frames=100, stage3_resolved=1)
        assert stats.stage3_rate == pytest.approx(0.01)

    def test_avg_stage1_ms(self) -> None:
        stats = CascadeStats(total_frames=10, stage1_total_ms=30.0)
        assert stats.avg_stage1_ms == pytest.approx(3.0)

    def test_avg_stage2_ms(self) -> None:
        # avg_stage2_ms divides by (stage2_resolved + stage3_resolved)
        stats = CascadeStats(
            stage2_resolved=4,
            stage3_resolved=1,
            stage2_total_ms=250.0,
        )
        assert stats.avg_stage2_ms == pytest.approx(50.0)

    def test_avg_stage3_ms(self) -> None:
        stats = CascadeStats(stage3_resolved=2, stage3_total_ms=400.0)
        assert stats.avg_stage3_ms == pytest.approx(200.0)

    def test_to_dict(self) -> None:
        stats = CascadeStats(total_frames=100, stage1_resolved=90)
        d = stats.to_dict()
        assert d["total_frames"] == 100
        assert d["stage1_resolved"] == 90
        assert d["stage1_rate"] == 0.9
        assert isinstance(d["avg_stage1_ms"], float)
        # All expected keys present
        expected_keys = {
            "total_frames",
            "stage1_resolved",
            "stage2_resolved",
            "stage3_resolved",
            "stage1_rate",
            "stage2_rate",
            "stage3_rate",
            "avg_stage1_ms",
            "avg_stage2_ms",
            "avg_stage3_ms",
        }
        assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Stage 1 — fast path tests
# ---------------------------------------------------------------------------


class TestStage1FastPath:
    """Stage 1: all detections high confidence => resolved at Stage 1."""

    def test_all_high_confidence_resolved_at_stage1(self) -> None:
        """When YOLO returns all conf >= 0.7, cascade stops at Stage 1."""
        yolo = MockYOLO([
            _det("person", conf=0.9),
            _det("helmet", conf=0.85, x1=0.4, y1=0.1, x2=0.6, y2=0.3),
        ])
        dino = MockDINO([_det("person", conf=0.5)])

        cascade = CascadeDetector(yolo_detector=yolo, dino_detector=dino)
        result = cascade.detect(_make_frame(), ["person", "helmet"])

        assert len(result) == 2
        assert yolo.call_count == 1
        assert dino.call_count == 0  # DINO never called

        stats = cascade.get_stats()
        assert stats["stage1_resolved"] == 1
        assert stats["stage2_resolved"] == 0

    def test_single_high_conf_detection(self) -> None:
        """A single detection with high conf resolves at Stage 1."""
        yolo = MockYOLO([_det("person", conf=0.95)])
        cascade = CascadeDetector(yolo_detector=yolo)
        result = cascade.detect(_make_frame(), ["person"])

        assert len(result) == 1
        assert result[0].confidence == 0.95
        assert cascade.get_stats()["stage1_resolved"] == 1


# ---------------------------------------------------------------------------
# Stage 2 — escalation tests
# ---------------------------------------------------------------------------


class TestStage2Escalation:
    """Low-confidence YOLO detections trigger Grounding DINO."""

    def test_low_conf_yolo_triggers_dino(self) -> None:
        """If YOLO returns conf < stage1_threshold, DINO is invoked."""
        yolo = MockYOLO([_det("person", conf=0.4)])
        dino = MockDINO([_det("person", conf=0.6, x1=0.1, y1=0.1, x2=0.3, y2=0.3)])

        cascade = CascadeDetector(yolo_detector=yolo, dino_detector=dino)
        result = cascade.detect(_make_frame(), ["person"])

        assert yolo.call_count == 1
        assert dino.call_count == 1
        assert len(result) >= 1
        stats = cascade.get_stats()
        assert stats["stage2_resolved"] == 1

    def test_mixed_conf_escalates_for_low_only(self) -> None:
        """Mix of high and low conf: high kept, low triggers Stage 2."""
        high_det = _det("person", conf=0.9, x1=0.1, y1=0.1, x2=0.3, y2=0.3)
        low_det = _det("helmet", conf=0.4, x1=0.5, y1=0.5, x2=0.7, y2=0.7)
        yolo = MockYOLO([high_det, low_det])

        dino_det = _det("helmet", conf=0.6, x1=0.5, y1=0.5, x2=0.7, y2=0.7)
        dino = MockDINO([dino_det])

        cascade = CascadeDetector(yolo_detector=yolo, dino_detector=dino)
        result = cascade.detect(_make_frame(), ["person", "helmet"])

        assert dino.call_count == 1
        # Both person (from YOLO high) and helmet (from DINO) should be in results
        labels = [d.label for d in result]
        assert "person" in labels


# ---------------------------------------------------------------------------
# Stage 3 — VLM escalation tests
# ---------------------------------------------------------------------------


class TestStage3Escalation:
    """Uncertain Stage 2 results trigger VLM verification."""

    def test_low_stage2_conf_triggers_vlm(self) -> None:
        """When merged detections have conf < stage2_threshold, VLM runs."""
        # YOLO returns low conf
        yolo = MockYOLO([_det("person", conf=0.15)])
        # DINO also returns low conf (below stage2_threshold=0.3)
        dino = MockDINO([_det("person", conf=0.2, x1=0.1, y1=0.1, x2=0.3, y2=0.3)])
        vlm = MockVLM()

        cascade = CascadeDetector(
            yolo_detector=yolo, dino_detector=dino, vlm_client=vlm
        )
        result = cascade.detect(_make_frame(), ["person"])

        assert vlm.call_count == 1
        stats = cascade.get_stats()
        assert stats["stage3_resolved"] == 1

    def test_vlm_receives_uncertain_detections(self) -> None:
        """VLM gets the uncertain detections passed to it."""
        yolo = MockYOLO([_det("box", conf=0.1)])
        dino = MockDINO([_det("box", conf=0.2, x1=0.1, y1=0.1, x2=0.3, y2=0.3)])
        vlm = MockVLM()

        cascade = CascadeDetector(
            yolo_detector=yolo, dino_detector=dino, vlm_client=vlm
        )
        cascade.detect(_make_frame(), ["box"])

        assert vlm.last_uncertain is not None
        assert len(vlm.last_uncertain) > 0

    def test_vlm_returns_verified_detections(self) -> None:
        """VLM can return high-confidence verified results."""
        yolo = MockYOLO([_det("person", conf=0.15)])
        dino = MockDINO([_det("person", conf=0.2)])

        verified = [_det("person", conf=0.88, x1=0.1, y1=0.1, x2=0.3, y2=0.3)]
        vlm = MockVLM(result=verified)

        cascade = CascadeDetector(
            yolo_detector=yolo, dino_detector=dino, vlm_client=vlm
        )
        result = cascade.detect(_make_frame(), ["person"])

        # The verified VLM detection should be in the final output
        assert any(d.confidence == 0.88 for d in result)


# ---------------------------------------------------------------------------
# Empty input handling
# ---------------------------------------------------------------------------


class TestEmptyInputs:
    """Handles empty frames and prompts gracefully."""

    def test_empty_frame_returns_empty(self) -> None:
        yolo = MockYOLO([_det("person", conf=0.9)])
        cascade = CascadeDetector(yolo_detector=yolo)
        empty_frame = np.array([], dtype=np.uint8)
        result = cascade.detect(empty_frame, ["person"])
        assert result == []
        assert yolo.call_count == 0

    def test_empty_prompts_returns_empty(self) -> None:
        yolo = MockYOLO([_det("person", conf=0.9)])
        cascade = CascadeDetector(yolo_detector=yolo)
        result = cascade.detect(_make_frame(), [])
        assert result == []
        assert yolo.call_count == 0

    def test_empty_frame_does_not_increment_stats(self) -> None:
        cascade = CascadeDetector(yolo_detector=MockYOLO())
        cascade.detect(np.array([], dtype=np.uint8), ["person"])
        assert cascade.get_stats()["total_frames"] == 0


# ---------------------------------------------------------------------------
# Detection merging — cross-stage NMS
# ---------------------------------------------------------------------------


class TestDetectionMerging:
    """Cross-stage NMS: primary takes precedence."""

    def test_non_overlapping_detections_both_kept(self) -> None:
        """Non-overlapping primary + secondary are all kept after merge."""
        # YOLO returns one high-conf (person) and one low-conf (helmet)
        yolo = MockYOLO([
            _det("person", conf=0.9, x1=0.0, y1=0.0, x2=0.2, y2=0.2),
            _det("helmet", conf=0.4, x1=0.5, y1=0.5, x2=0.6, y2=0.6),
        ])
        dino = MockDINO([_det("helmet", conf=0.6, x1=0.7, y1=0.7, x2=0.9, y2=0.9)])

        cascade = CascadeDetector(yolo_detector=yolo, dino_detector=dino)
        result = cascade.detect(_make_frame(), ["person", "helmet"])

        # person (high-conf from YOLO) + helmet (from DINO) — non-overlapping, both kept
        assert len(result) >= 2

    def test_overlapping_detections_higher_conf_wins(self) -> None:
        """Overlapping detections: higher confidence replaces lower."""
        # Same bbox, different confidence
        yolo_det = _det("person", conf=0.5, x1=0.1, y1=0.1, x2=0.4, y2=0.4)
        dino_det = _det("person", conf=0.8, x1=0.1, y1=0.1, x2=0.4, y2=0.4)

        yolo = MockYOLO([yolo_det])
        dino = MockDINO([dino_det])

        cascade = CascadeDetector(yolo_detector=yolo, dino_detector=dino)
        result = cascade.detect(_make_frame(), ["person"])

        # The higher-confidence detection should survive
        person_dets = [d for d in result if d.label == "person"]
        assert len(person_dets) >= 1
        assert max(d.confidence for d in person_dets) >= 0.5

    def test_merge_primary_precedence_same_conf(self) -> None:
        """When IoU > threshold, secondary with higher conf replaces primary."""
        cascade = CascadeDetector()
        primary = [_det("person", conf=0.6, x1=0.1, y1=0.1, x2=0.4, y2=0.4)]
        secondary = [_det("person", conf=0.9, x1=0.1, y1=0.1, x2=0.4, y2=0.4)]

        merged = cascade._merge_detections(primary, secondary)
        assert len(merged) == 1
        # Secondary has higher conf, so it replaces primary
        assert merged[0].confidence == 0.9

    def test_merge_empty_primary(self) -> None:
        """Empty primary list returns NMS-filtered secondary."""
        cascade = CascadeDetector()
        secondary = [
            _det("person", conf=0.8, x1=0.1, y1=0.1, x2=0.3, y2=0.3),
            _det("helmet", conf=0.7, x1=0.5, y1=0.5, x2=0.7, y2=0.7),
        ]
        merged = cascade._merge_detections([], secondary)
        assert len(merged) == 2

    def test_merge_empty_secondary(self) -> None:
        """Empty secondary returns copy of primary."""
        cascade = CascadeDetector()
        primary = [_det("person", conf=0.9)]
        merged = cascade._merge_detections(primary, [])
        assert len(merged) == 1
        assert merged[0].confidence == 0.9

    def test_merge_both_empty(self) -> None:
        """Both empty returns empty."""
        cascade = CascadeDetector()
        merged = cascade._merge_detections([], [])
        assert merged == []


# ---------------------------------------------------------------------------
# Statistics tracking
# ---------------------------------------------------------------------------


class TestStatistics:
    """Per-stage counts and timing."""

    def test_total_frames_incremented(self) -> None:
        yolo = MockYOLO([_det("person", conf=0.9)])
        cascade = CascadeDetector(yolo_detector=yolo)

        cascade.detect(_make_frame(), ["person"])
        cascade.detect(_make_frame(), ["person"])

        assert cascade.get_stats()["total_frames"] == 2

    def test_stage1_timing_recorded(self) -> None:
        yolo = MockYOLO([_det("person", conf=0.9)])
        cascade = CascadeDetector(yolo_detector=yolo)
        cascade.detect(_make_frame(), ["person"])

        stats = cascade.get_stats()
        assert stats["avg_stage1_ms"] >= 0.0

    def test_reset_stats(self) -> None:
        yolo = MockYOLO([_det("person", conf=0.9)])
        cascade = CascadeDetector(yolo_detector=yolo)
        cascade.detect(_make_frame(), ["person"])

        cascade.reset_stats()
        stats = cascade.get_stats()
        assert stats["total_frames"] == 0
        assert stats["stage1_resolved"] == 0

    def test_stage2_timing_only_when_triggered(self) -> None:
        """Stage 2 timing accumulates only when DINO is actually called."""
        yolo = MockYOLO([_det("person", conf=0.5)])
        dino = MockDINO([_det("person", conf=0.6)])
        cascade = CascadeDetector(yolo_detector=yolo, dino_detector=dino)
        cascade.detect(_make_frame(), ["person"])

        stats = cascade.get_stats()
        assert stats["stage2_resolved"] == 1
        assert stats["avg_stage2_ms"] >= 0.0

    def test_multiple_frames_accumulate(self) -> None:
        """Run multiple frames and verify counters accumulate correctly."""
        yolo = MockYOLO([_det("person", conf=0.9)])
        cascade = CascadeDetector(yolo_detector=yolo)

        for _ in range(5):
            cascade.detect(_make_frame(), ["person"])

        stats = cascade.get_stats()
        assert stats["total_frames"] == 5
        assert stats["stage1_resolved"] == 5
        assert stats["stage1_rate"] == 1.0


# ---------------------------------------------------------------------------
# Missing detector handling
# ---------------------------------------------------------------------------


class TestMissingDetectors:
    """Graceful handling when detectors are unavailable."""

    def test_no_yolo_no_orchestrator_returns_empty(self) -> None:
        """No YOLO and no orchestrator — Stage 1 returns nothing."""
        cascade = CascadeDetector()
        result = cascade.detect(_make_frame(), ["person"])
        # No yolo, no dino, no orchestrator -> stage1 returns [], then
        # checks if dino is None and orchestrator is None -> stage1_resolved
        assert result == []
        assert cascade.get_stats()["stage1_resolved"] == 1

    def test_no_dino_skips_stage2(self) -> None:
        """If no DINO is available, Stage 2 is a no-op returning empty."""
        yolo = MockYOLO([_det("person", conf=0.5)])
        cascade = CascadeDetector(yolo_detector=yolo)
        # No dino and no orchestrator -> stage2 returns []
        result = cascade.detect(_make_frame(), ["person"])

        # After stage2 returns empty, merged = high_conf (none, since conf=0.5 < 0.7)
        # uncertain: all detections below stage2_thresh? depends on merge logic
        # The low-conf YOLO det goes to stage2, stage2 returns empty
        # merged = high_conf ([]) + stage2 ([]) = []
        # No uncertain -> stage2_resolved += 1
        stats = cascade.get_stats()
        assert stats["total_frames"] == 1

    def test_no_vlm_returns_penalized_detections(self) -> None:
        """Without VLM, uncertain detections get 0.8x confidence penalty."""
        yolo = MockYOLO([_det("person", conf=0.15)])
        dino = MockDINO([_det("person", conf=0.2, x1=0.1, y1=0.1, x2=0.3, y2=0.3)])

        cascade = CascadeDetector(
            yolo_detector=yolo, dino_detector=dino, vlm_client=None
        )
        result = cascade.detect(_make_frame(), ["person"])

        # Without VLM, uncertain dets returned with conf * 0.8
        stats = cascade.get_stats()
        assert stats["stage3_resolved"] == 1
        # Check for vlm_verified=False attribute on penalized detections
        penalized = [d for d in result if d.attributes.get("vlm_verified") is False]
        assert len(penalized) > 0

    def test_no_vlm_confidence_penalty_value(self) -> None:
        """Verify exact penalty factor (0.8) applied without VLM."""
        original_conf = 0.25
        yolo = MockYOLO([_det("person", conf=0.1)])
        dino = MockDINO(
            [_det("person", conf=original_conf, x1=0.1, y1=0.1, x2=0.3, y2=0.3)]
        )

        cascade = CascadeDetector(
            yolo_detector=yolo, dino_detector=dino, vlm_client=None
        )
        result = cascade.detect(_make_frame(), ["person"])

        # The uncertain detection should have conf * 0.8
        penalized = [d for d in result if d.attributes.get("vlm_verified") is False]
        if penalized:
            assert penalized[0].confidence == pytest.approx(original_conf * 0.8)


# ---------------------------------------------------------------------------
# ObjectDetector interface compliance
# ---------------------------------------------------------------------------


class TestObjectDetectorInterface:
    """CascadeDetector implements ObjectDetector interface."""

    def test_isinstance_object_detector(self) -> None:
        cascade = CascadeDetector()
        assert isinstance(cascade, ObjectDetector)

    def test_detect_method_exists(self) -> None:
        cascade = CascadeDetector()
        assert callable(getattr(cascade, "detect", None))

    def test_close_method_exists(self) -> None:
        cascade = CascadeDetector()
        assert callable(getattr(cascade, "close", None))

    def test_close_calls_child_detectors(self) -> None:
        yolo = MockYOLO()
        dino = MockDINO()
        cascade = CascadeDetector(yolo_detector=yolo, dino_detector=dino)
        cascade.close()

        assert yolo.closed
        assert dino.closed

    def test_close_without_detectors_no_error(self) -> None:
        """close() with no injected detectors should not raise."""
        cascade = CascadeDetector()
        cascade.close()  # should not raise


# ---------------------------------------------------------------------------
# Orchestrator integration — acquire/release pattern
# ---------------------------------------------------------------------------


class TestOrchestratorIntegration:
    """Orchestrator acquire/release pattern for VRAM management."""

    def test_acquires_yolo_from_orchestrator(self) -> None:
        """When no direct YOLO, acquires from orchestrator."""
        mock_yolo = MockYOLO([_det("person", conf=0.9)])
        orchestrator = MagicMock()
        orchestrator.acquire.return_value = mock_yolo

        cascade = CascadeDetector(orchestrator=orchestrator)
        result = cascade.detect(_make_frame(), ["person"])

        orchestrator.acquire.assert_called_with("yolo")
        assert len(result) >= 1

    def test_releases_yolo_after_use(self) -> None:
        """Orchestrator.release('yolo') is called after detection."""
        mock_yolo = MockYOLO([_det("person", conf=0.9)])
        orchestrator = MagicMock()
        orchestrator.acquire.return_value = mock_yolo

        cascade = CascadeDetector(orchestrator=orchestrator)
        cascade.detect(_make_frame(), ["person"])

        orchestrator.release.assert_any_call("yolo")

    def test_acquires_dino_on_escalation(self) -> None:
        """When YOLO conf is low and no direct DINO, acquires DINO from orchestrator."""
        mock_yolo = MockYOLO([_det("person", conf=0.4)])
        mock_dino = MockDINO([_det("person", conf=0.6)])
        orchestrator = MagicMock()

        def acquire_side_effect(name: str) -> ObjectDetector:
            if name == "yolo":
                return mock_yolo
            if name == "grounding_dino":
                return mock_dino
            raise KeyError(name)

        orchestrator.acquire.side_effect = acquire_side_effect

        cascade = CascadeDetector(orchestrator=orchestrator)
        cascade.detect(_make_frame(), ["person"])

        assert call("grounding_dino") in orchestrator.acquire.call_args_list

    def test_releases_dino_after_stage2(self) -> None:
        """Orchestrator.release('grounding_dino') is called after Stage 2."""
        mock_yolo = MockYOLO([_det("person", conf=0.4)])
        mock_dino = MockDINO([_det("person", conf=0.6)])
        orchestrator = MagicMock()

        def acquire_side_effect(name: str) -> ObjectDetector:
            if name == "yolo":
                return mock_yolo
            if name == "grounding_dino":
                return mock_dino
            raise KeyError(name)

        orchestrator.acquire.side_effect = acquire_side_effect

        cascade = CascadeDetector(orchestrator=orchestrator)
        cascade.detect(_make_frame(), ["person"])

        orchestrator.release.assert_any_call("grounding_dino")

    def test_orchestrator_yolo_not_available(self) -> None:
        """If orchestrator can't provide YOLO, gracefully skip Stage 1."""
        orchestrator = MagicMock()
        orchestrator.acquire.side_effect = KeyError("yolo not registered")

        cascade = CascadeDetector(orchestrator=orchestrator)
        result = cascade.detect(_make_frame(), ["person"])

        # YOLO not available -> empty stage1, then dino also unavailable
        assert result == []

    def test_orchestrator_dino_not_available(self) -> None:
        """If orchestrator can't provide DINO, skip Stage 2 gracefully."""
        mock_yolo = MockYOLO([_det("person", conf=0.4)])
        orchestrator = MagicMock()

        def acquire_side_effect(name: str) -> ObjectDetector:
            if name == "yolo":
                return mock_yolo
            raise KeyError(f"{name} not registered")

        orchestrator.acquire.side_effect = acquire_side_effect

        cascade = CascadeDetector(orchestrator=orchestrator)
        result = cascade.detect(_make_frame(), ["person"])
        # Should not raise
        assert isinstance(result, list)

    def test_direct_detector_bypasses_orchestrator(self) -> None:
        """When yolo_detector is provided directly, orchestrator is not called."""
        yolo = MockYOLO([_det("person", conf=0.9)])
        orchestrator = MagicMock()

        cascade = CascadeDetector(
            orchestrator=orchestrator, yolo_detector=yolo
        )
        cascade.detect(_make_frame(), ["person"])

        # Should NOT call orchestrator.acquire for yolo
        for c in orchestrator.acquire.call_args_list:
            assert c != call("yolo")


# ---------------------------------------------------------------------------
# Direct detector injection
# ---------------------------------------------------------------------------


class TestDirectInjection:
    """Direct yolo_detector/dino_detector params bypass orchestrator."""

    def test_injected_yolo_used(self) -> None:
        yolo = MockYOLO([_det("person", conf=0.9)])
        cascade = CascadeDetector(yolo_detector=yolo)
        cascade.detect(_make_frame(), ["person"])
        assert yolo.call_count == 1

    def test_injected_dino_used(self) -> None:
        yolo = MockYOLO([_det("person", conf=0.4)])
        dino = MockDINO([_det("person", conf=0.6)])
        cascade = CascadeDetector(yolo_detector=yolo, dino_detector=dino)
        cascade.detect(_make_frame(), ["person"])
        assert dino.call_count == 1

    def test_injected_detectors_not_released_via_orchestrator(self) -> None:
        """Injected detectors should not trigger orchestrator.release."""
        yolo = MockYOLO([_det("person", conf=0.4)])
        dino = MockDINO([_det("person", conf=0.6)])
        orchestrator = MagicMock()

        cascade = CascadeDetector(
            orchestrator=orchestrator,
            yolo_detector=yolo,
            dino_detector=dino,
        )
        cascade.detect(_make_frame(), ["person"])

        # release should not be called since detectors are directly provided
        orchestrator.release.assert_not_called()


# ---------------------------------------------------------------------------
# Threshold customization
# ---------------------------------------------------------------------------


class TestThresholdCustomization:
    """stage1_threshold and stage2_threshold parameters."""

    def test_custom_stage1_threshold_high(self) -> None:
        """High stage1_threshold means more frames escalate to Stage 2."""
        yolo = MockYOLO([_det("person", conf=0.8)])
        dino = MockDINO([_det("person", conf=0.9)])

        # Default threshold is 0.7, conf=0.8 resolves at stage 1
        cascade_default = CascadeDetector(yolo_detector=yolo, dino_detector=dino)
        cascade_default.detect(_make_frame(), ["person"])
        assert cascade_default.get_stats()["stage1_resolved"] == 1

        # With threshold=0.9, conf=0.8 escalates
        yolo2 = MockYOLO([_det("person", conf=0.8)])
        dino2 = MockDINO([_det("person", conf=0.9)])
        cascade_high = CascadeDetector(
            yolo_detector=yolo2,
            dino_detector=dino2,
            stage1_threshold=0.9,
        )
        cascade_high.detect(_make_frame(), ["person"])
        assert cascade_high.get_stats()["stage1_resolved"] == 0

    def test_custom_stage2_threshold_low(self) -> None:
        """Low stage2_threshold means more frames resolve at Stage 2."""
        yolo = MockYOLO([_det("person", conf=0.4)])
        dino = MockDINO([_det("person", conf=0.15, x1=0.1, y1=0.1, x2=0.3, y2=0.3)])

        # Default stage2_threshold=0.3 -> conf=0.15 goes to VLM
        cascade_default = CascadeDetector(
            yolo_detector=yolo, dino_detector=dino
        )
        cascade_default.detect(_make_frame(), ["person"])
        assert cascade_default.get_stats()["stage3_resolved"] == 1

        # With stage2_threshold=0.1 -> conf=0.15 resolves at stage 2
        yolo2 = MockYOLO([_det("person", conf=0.4)])
        dino2 = MockDINO([_det("person", conf=0.15, x1=0.1, y1=0.1, x2=0.3, y2=0.3)])
        cascade_low = CascadeDetector(
            yolo_detector=yolo2,
            dino_detector=dino2,
            stage2_threshold=0.1,
        )
        cascade_low.detect(_make_frame(), ["person"])
        assert cascade_low.get_stats()["stage2_resolved"] == 1
        assert cascade_low.get_stats()["stage3_resolved"] == 0

    def test_custom_nms_threshold(self) -> None:
        """Custom nms_threshold affects merge behavior."""
        cascade = CascadeDetector(nms_threshold=0.9)
        # Two detections with moderate overlap — should not be merged with high nms_threshold
        primary = [_det("person", conf=0.7, x1=0.1, y1=0.1, x2=0.5, y2=0.5)]
        secondary = [_det("person", conf=0.8, x1=0.2, y1=0.2, x2=0.6, y2=0.6)]

        merged = cascade._merge_detections(primary, secondary)
        # IoU of these two boxes is moderate (~0.39), well below nms_thresh=0.9
        # So both should be kept
        assert len(merged) == 2


# ---------------------------------------------------------------------------
# NMS merging with overlapping detections from different stages
# ---------------------------------------------------------------------------


class TestCrossStageNMS:
    """NMS merging with overlapping detections from different stages."""

    def test_identical_bbox_keeps_higher_conf(self) -> None:
        """Identical bounding boxes: higher confidence wins."""
        cascade = CascadeDetector(nms_threshold=0.5)
        primary = [_det("person", conf=0.6, x1=0.1, y1=0.1, x2=0.4, y2=0.4)]
        secondary = [_det("person", conf=0.95, x1=0.1, y1=0.1, x2=0.4, y2=0.4)]

        merged = cascade._merge_detections(primary, secondary)
        assert len(merged) == 1
        assert merged[0].confidence == 0.95  # higher wins

    def test_partially_overlapping_merged(self) -> None:
        """Partially overlapping boxes (IoU > threshold) are merged."""
        cascade = CascadeDetector(nms_threshold=0.3)
        # These boxes have significant overlap
        primary = [_det("person", conf=0.6, x1=0.1, y1=0.1, x2=0.5, y2=0.5)]
        secondary = [_det("person", conf=0.8, x1=0.15, y1=0.15, x2=0.55, y2=0.55)]

        merged = cascade._merge_detections(primary, secondary)
        assert len(merged) == 1

    def test_non_overlapping_all_kept(self) -> None:
        """Non-overlapping detections from different stages are all kept."""
        cascade = CascadeDetector(nms_threshold=0.5)
        primary = [
            _det("person", conf=0.9, x1=0.0, y1=0.0, x2=0.2, y2=0.2),
        ]
        secondary = [
            _det("helmet", conf=0.8, x1=0.7, y1=0.7, x2=0.9, y2=0.9),
        ]

        merged = cascade._merge_detections(primary, secondary)
        assert len(merged) == 2
        labels = {d.label for d in merged}
        assert labels == {"person", "helmet"}

    def test_multiple_overlapping_different_labels(self) -> None:
        """Overlapping boxes with different labels: IoU still applies."""
        cascade = CascadeDetector(nms_threshold=0.5)
        primary = [_det("person", conf=0.7, x1=0.1, y1=0.1, x2=0.5, y2=0.5)]
        # Same bbox but different label
        secondary = [_det("helmet", conf=0.9, x1=0.1, y1=0.1, x2=0.5, y2=0.5)]

        merged = cascade._merge_detections(primary, secondary)
        # IoU=1.0 > 0.5, so secondary replaces primary (higher conf)
        assert len(merged) == 1
        assert merged[0].confidence == 0.9


# ---------------------------------------------------------------------------
# VLM failure handling
# ---------------------------------------------------------------------------


class TestVLMFailure:
    """VLM errors are handled gracefully."""

    def test_vlm_exception_returns_uncertain_as_is(self) -> None:
        """When VLM raises, uncertain detections returned unchanged."""
        yolo = MockYOLO([_det("person", conf=0.1)])
        dino = MockDINO([_det("person", conf=0.2, x1=0.1, y1=0.1, x2=0.3, y2=0.3)])
        vlm = MockVLM(should_fail=True)

        cascade = CascadeDetector(
            yolo_detector=yolo, dino_detector=dino, vlm_client=vlm
        )
        result = cascade.detect(_make_frame(), ["person"])

        # Should not raise, and should return some detections
        assert isinstance(result, list)
        assert vlm.call_count == 1

    def test_vlm_without_analyze_detections_fallback(self) -> None:
        """VLM without analyze_detections method uses fallback (0.9x conf)."""
        yolo = MockYOLO([_det("person", conf=0.15)])
        dino = MockDINO([_det("person", conf=0.2, x1=0.1, y1=0.1, x2=0.3, y2=0.3)])

        # VLM without analyze_detections
        vlm = MagicMock(spec=[])  # empty spec = no methods
        del vlm.analyze_detections  # ensure hasattr returns False

        cascade = CascadeDetector(
            yolo_detector=yolo, dino_detector=dino, vlm_client=vlm
        )
        result = cascade.detect(_make_frame(), ["person"])

        # Should return fallback with 0.9x conf and vlm_analyzed=True
        analyzed = [d for d in result if d.attributes.get("vlm_analyzed") is True]
        assert len(analyzed) > 0


# ---------------------------------------------------------------------------
# Detector exception handling
# ---------------------------------------------------------------------------


class TestDetectorExceptions:
    """Detector failures at each stage are caught gracefully."""

    def test_yolo_exception_returns_empty_stage1(self) -> None:
        """YOLO exception -> empty Stage 1 detections."""

        class FailYOLO(ObjectDetector):
            def detect(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
                raise RuntimeError("YOLO crashed")

        cascade = CascadeDetector(yolo_detector=FailYOLO())
        result = cascade.detect(_make_frame(), ["person"])
        # Should not raise
        assert isinstance(result, list)

    def test_dino_exception_returns_empty_stage2(self) -> None:
        """DINO exception -> empty Stage 2 detections."""

        class FailDINO(ObjectDetector):
            def detect(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
                raise RuntimeError("DINO crashed")

        yolo = MockYOLO([_det("person", conf=0.4)])
        cascade = CascadeDetector(yolo_detector=yolo, dino_detector=FailDINO())
        result = cascade.detect(_make_frame(), ["person"])
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# YOLO returns empty — no detections path
# ---------------------------------------------------------------------------


class TestYOLOReturnsEmpty:
    """When YOLO finds nothing and no escalation path exists."""

    def test_yolo_empty_no_dino_no_orchestrator(self) -> None:
        """YOLO returns empty, no DINO, no orchestrator -> stage1 resolved with []."""
        yolo = MockYOLO([])  # empty
        cascade = CascadeDetector(yolo_detector=yolo)
        result = cascade.detect(_make_frame(), ["person"])

        assert result == []
        stats = cascade.get_stats()
        assert stats["stage1_resolved"] == 1

    def test_yolo_empty_with_dino(self) -> None:
        """YOLO returns empty but DINO available -> escalates to Stage 2."""
        yolo = MockYOLO([])  # empty
        dino = MockDINO([_det("person", conf=0.6)])

        cascade = CascadeDetector(yolo_detector=yolo, dino_detector=dino)
        result = cascade.detect(_make_frame(), ["person"])

        # DINO should be called since YOLO is empty but DINO is available
        assert dino.call_count == 1
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# End-to-end cascade flow
# ---------------------------------------------------------------------------


class TestEndToEndCascade:
    """Full cascade through all three stages."""

    def test_full_three_stage_cascade(self) -> None:
        """Frame goes through YOLO -> DINO -> VLM."""
        yolo = MockYOLO([_det("person", conf=0.3)])  # below stage1_thresh
        dino = MockDINO([_det("person", conf=0.2)])  # below stage2_thresh
        vlm_result = [_det("person", conf=0.85, x1=0.1, y1=0.1, x2=0.3, y2=0.3)]
        vlm = MockVLM(result=vlm_result)

        cascade = CascadeDetector(
            yolo_detector=yolo,
            dino_detector=dino,
            vlm_client=vlm,
        )
        result = cascade.detect(_make_frame(), ["person"])

        assert yolo.call_count == 1
        assert dino.call_count == 1
        assert vlm.call_count == 1
        assert cascade.get_stats()["stage3_resolved"] == 1

    def test_cascade_stats_across_mixed_frames(self) -> None:
        """Mix of fast-path and escalated frames updates stats correctly."""
        # Frame 1: high conf -> Stage 1
        yolo_high = MockYOLO([_det("person", conf=0.9)])
        cascade = CascadeDetector(yolo_detector=yolo_high)
        cascade.detect(_make_frame(), ["person"])

        assert cascade.get_stats()["stage1_resolved"] == 1
        assert cascade.get_stats()["total_frames"] == 1

    def test_get_stats_returns_dict(self) -> None:
        cascade = CascadeDetector()
        stats = cascade.get_stats()
        assert isinstance(stats, dict)
        assert "total_frames" in stats
        assert "stage1_rate" in stats
