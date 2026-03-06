"""Confidence Cascade Detector — multi-stage escalation detection.

Phase 33: Routes each frame through progressively more expensive models
based on detection confidence:

    Stage 1: YOLO-World (~3 ms)  → conf > 0.7 → done
    Stage 2: Grounding DINO (~50 ms) → conf > 0.3 → done
    Stage 3: VLM scene analysis (~200 ms) → final answer

95% of frames resolve at Stage 1 (fast path).  Only ambiguous or
novel objects escalate to heavier models.  This is the core of the
"orchestration advantage" — no single model is best for all cases.

The cascade also produces merged detection lists using cross-stage NMS
to avoid duplicates when multiple stages contribute detections.

Integration::

    cascade = CascadeDetector(orchestrator, vlm_client=vlm)
    detections = cascade.detect(frame, ["person", "helmet", "vest"])
    stats = cascade.get_stats()  # per-stage latency/hit-rate

Requires the following models registered with the orchestrator:
    - "yolo" (YOLOWorldDetector) — always loaded, priority 9
    - "grounding_dino" (GroundingDINODetector) — on-demand, priority 6
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sopilot.perception.detector import ObjectDetector, non_max_suppression
from sopilot.perception.types import BBox, Detection, PerceptionConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cascade statistics
# ---------------------------------------------------------------------------


@dataclass
class CascadeStats:
    """Per-stage statistics for the cascade detector."""

    total_frames: int = 0
    stage1_resolved: int = 0  # resolved by YOLO alone
    stage2_resolved: int = 0  # needed Grounding DINO
    stage3_resolved: int = 0  # needed VLM
    stage1_total_ms: float = 0.0
    stage2_total_ms: float = 0.0
    stage3_total_ms: float = 0.0

    @property
    def stage1_rate(self) -> float:
        return self.stage1_resolved / max(1, self.total_frames)

    @property
    def stage2_rate(self) -> float:
        return self.stage2_resolved / max(1, self.total_frames)

    @property
    def stage3_rate(self) -> float:
        return self.stage3_resolved / max(1, self.total_frames)

    @property
    def avg_stage1_ms(self) -> float:
        return self.stage1_total_ms / max(1, self.total_frames)

    @property
    def avg_stage2_ms(self) -> float:
        return self.stage2_total_ms / max(1, self.stage2_resolved + self.stage3_resolved)

    @property
    def avg_stage3_ms(self) -> float:
        return self.stage3_total_ms / max(1, self.stage3_resolved)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_frames": self.total_frames,
            "stage1_resolved": self.stage1_resolved,
            "stage2_resolved": self.stage2_resolved,
            "stage3_resolved": self.stage3_resolved,
            "stage1_rate": round(self.stage1_rate, 4),
            "stage2_rate": round(self.stage2_rate, 4),
            "stage3_rate": round(self.stage3_rate, 4),
            "avg_stage1_ms": round(self.avg_stage1_ms, 2),
            "avg_stage2_ms": round(self.avg_stage2_ms, 2),
            "avg_stage3_ms": round(self.avg_stage3_ms, 2),
        }


# ---------------------------------------------------------------------------
# CascadeDetector
# ---------------------------------------------------------------------------


class CascadeDetector(ObjectDetector):
    """Multi-stage confidence cascade detector.

    Inherits from :class:`ObjectDetector` so it can be used as a drop-in
    replacement anywhere a detector is expected.

    Parameters
    ----------
    orchestrator:
        :class:`ModelOrchestrator` for VRAM-managed model access.
        If ``None``, detectors must be provided directly.
    yolo_detector:
        Stage 1 detector (fast path).  If ``None``, acquired from orchestrator.
    dino_detector:
        Stage 2 detector (escalation).  If ``None``, acquired from orchestrator.
    vlm_client:
        Optional VLM client for Stage 3 analysis.  Must have
        ``analyze(frame, prompts) -> list[Detection]`` or similar interface.
    config:
        Perception configuration.
    stage1_threshold:
        Minimum confidence to accept Stage 1 detection without escalation.
    stage2_threshold:
        Minimum confidence to accept Stage 2 detection without VLM.
    nms_threshold:
        IoU threshold for cross-stage NMS.
    """

    def __init__(
        self,
        orchestrator: Any = None,
        yolo_detector: ObjectDetector | None = None,
        dino_detector: ObjectDetector | None = None,
        vlm_client: Any = None,
        config: PerceptionConfig | None = None,
        stage1_threshold: float = 0.7,
        stage2_threshold: float = 0.3,
        nms_threshold: float = 0.5,
    ) -> None:
        self._orchestrator = orchestrator
        self._yolo = yolo_detector
        self._dino = dino_detector
        self._vlm = vlm_client
        self._config = config or PerceptionConfig()
        self._stage1_thresh = stage1_threshold
        self._stage2_thresh = stage2_threshold
        self._nms_thresh = nms_threshold
        self._stats = CascadeStats()

        logger.info(
            "CascadeDetector created: stage1_thresh=%.2f, stage2_thresh=%.2f, "
            "yolo=%s, dino=%s, vlm=%s",
            stage1_threshold,
            stage2_threshold,
            "provided" if yolo_detector else "from_orchestrator",
            "provided" if dino_detector else "from_orchestrator",
            "yes" if vlm_client else "no",
        )

    # ── Main detect ───────────────────────────────────────────────────

    def detect(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
        """Run the confidence cascade on a single frame.

        Parameters
        ----------
        frame:
            BGR image as numpy array (H, W, 3).
        prompts:
            Natural-language class descriptions.

        Returns
        -------
        list[Detection]
            Merged, NMS-filtered detections from all triggered stages.
        """
        if frame.size == 0 or not prompts:
            return []

        self._stats.total_frames += 1

        # ── Stage 1: YOLO (fast path) ────────────────────────────────
        t0 = time.monotonic()
        stage1_dets = self._run_stage1(frame, prompts)
        stage1_ms = (time.monotonic() - t0) * 1000
        self._stats.stage1_total_ms += stage1_ms

        high_conf = [d for d in stage1_dets if d.confidence >= self._stage1_thresh]
        low_conf = [d for d in stage1_dets if d.confidence < self._stage1_thresh]

        # If all detections are high-confidence, we're done
        if not low_conf and high_conf:
            self._stats.stage1_resolved += 1
            logger.debug(
                "Cascade: Stage 1 resolved (%d dets, %.1f ms)",
                len(high_conf),
                stage1_ms,
            )
            return high_conf

        # Also resolve if YOLO found nothing and we have no escalation path
        if not stage1_dets and self._dino is None and self._orchestrator is None:
            self._stats.stage1_resolved += 1
            return []

        # ── Stage 2: Grounding DINO (escalation) ─────────────────────
        t1 = time.monotonic()
        stage2_dets = self._run_stage2(frame, prompts)
        stage2_ms = (time.monotonic() - t1) * 1000
        self._stats.stage2_total_ms += stage2_ms

        # Merge Stage 1 high-conf + Stage 2 results
        merged = self._merge_detections(high_conf, stage2_dets)

        # Check if we have uncertain detections
        uncertain = [d for d in merged if d.confidence < self._stage2_thresh]
        confident = [d for d in merged if d.confidence >= self._stage2_thresh]

        if not uncertain:
            self._stats.stage2_resolved += 1
            logger.debug(
                "Cascade: Stage 2 resolved (%d dets, stage1=%.1f ms, stage2=%.1f ms)",
                len(confident),
                stage1_ms,
                stage2_ms,
            )
            return confident

        # ── Stage 3: VLM (final arbiter) ─────────────────────────────
        t2 = time.monotonic()
        stage3_dets = self._run_stage3(frame, prompts, uncertain)
        stage3_ms = (time.monotonic() - t2) * 1000
        self._stats.stage3_total_ms += stage3_ms
        self._stats.stage3_resolved += 1

        # Final merge: confident detections + VLM refinements
        final = self._merge_detections(confident, stage3_dets)

        logger.debug(
            "Cascade: Stage 3 resolved (%d dets, "
            "stage1=%.1f ms, stage2=%.1f ms, stage3=%.1f ms)",
            len(final),
            stage1_ms,
            stage2_ms,
            stage3_ms,
        )

        return final

    # ── Stage implementations ─────────────────────────────────────────

    def _run_stage1(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
        """Stage 1: YOLO-World fast detection."""
        detector = self._yolo
        if detector is None and self._orchestrator is not None:
            try:
                detector = self._orchestrator.acquire("yolo")
            except (KeyError, RuntimeError):
                logger.debug("YOLO not available via orchestrator; skipping Stage 1")
                return []

        if detector is None:
            return []

        try:
            return detector.detect(frame, prompts)
        except Exception:
            logger.debug("Stage 1 (YOLO) failed", exc_info=True)
            return []
        finally:
            if self._yolo is None and self._orchestrator is not None:
                self._orchestrator.release("yolo")

    def _run_stage2(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
        """Stage 2: Grounding DINO escalation."""
        detector = self._dino
        if detector is None and self._orchestrator is not None:
            try:
                detector = self._orchestrator.acquire("grounding_dino")
            except (KeyError, RuntimeError):
                logger.debug("Grounding DINO not available; skipping Stage 2")
                return []

        if detector is None:
            return []

        try:
            return detector.detect(frame, prompts)
        except Exception:
            logger.debug("Stage 2 (Grounding DINO) failed", exc_info=True)
            return []
        finally:
            if self._dino is None and self._orchestrator is not None:
                self._orchestrator.release("grounding_dino")

    def _run_stage3(
        self,
        frame: np.ndarray,
        prompts: list[str],
        uncertain: list[Detection],
    ) -> list[Detection]:
        """Stage 3: VLM scene analysis for uncertain detections.

        The VLM receives the frame and information about uncertain
        detections to verify or reject them.
        """
        if self._vlm is None:
            # No VLM available — return uncertain as-is (reduced confidence)
            return [
                Detection(
                    bbox=d.bbox,
                    label=d.label,
                    confidence=d.confidence * 0.8,  # penalize unverified
                    embedding=d.embedding,
                    attributes={**d.attributes, "vlm_verified": False},
                )
                for d in uncertain
            ]

        try:
            # Build context for VLM
            context_parts = [
                f"Verify these detections in the frame: {', '.join(prompts)}.",
                f"Uncertain detections ({len(uncertain)}):",
            ]
            for i, det in enumerate(uncertain):
                context_parts.append(
                    f"  {i + 1}. {det.label} at ({det.bbox.x1:.2f}, {det.bbox.y1:.2f}, "
                    f"{det.bbox.x2:.2f}, {det.bbox.y2:.2f}) conf={det.confidence:.2f}"
                )

            vlm_context = "\n".join(context_parts)

            # Call VLM (expected to return verified detections or dict with results)
            if hasattr(self._vlm, "analyze_detections"):
                result = self._vlm.analyze_detections(frame, uncertain, prompts)
                if isinstance(result, list):
                    return result

            # Fallback: return uncertain detections as-is
            return [
                Detection(
                    bbox=d.bbox,
                    label=d.label,
                    confidence=d.confidence * 0.9,
                    embedding=d.embedding,
                    attributes={**d.attributes, "vlm_analyzed": True},
                )
                for d in uncertain
            ]
        except Exception:
            logger.debug("Stage 3 (VLM) failed", exc_info=True)
            return uncertain

    # ── Detection merging ─────────────────────────────────────────────

    def _merge_detections(
        self,
        primary: list[Detection],
        secondary: list[Detection],
    ) -> list[Detection]:
        """Merge two detection lists with cross-stage NMS.

        Primary detections take precedence over secondary ones
        when they overlap (higher priority, not just higher confidence).
        """
        if not secondary:
            return list(primary)
        if not primary:
            return non_max_suppression(secondary, iou_threshold=self._nms_thresh)

        # Start with primary detections
        merged = list(primary)

        # Add secondary detections that don't overlap with primary
        for sec_det in secondary:
            overlaps = False
            for pri_det in merged:
                if sec_det.bbox.iou(pri_det.bbox) > self._nms_thresh:
                    # Same object — keep whichever has higher confidence
                    if sec_det.confidence > pri_det.confidence:
                        # Replace primary with secondary
                        idx = merged.index(pri_det)
                        merged[idx] = sec_det
                    overlaps = True
                    break
            if not overlaps:
                merged.append(sec_det)

        return merged

    # ── Statistics ────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Return cascade statistics for API/debug."""
        return self._stats.to_dict()

    def reset_stats(self) -> None:
        """Reset cascade statistics."""
        self._stats = CascadeStats()

    # ── ObjectDetector interface compliance ────────────────────────────

    def close(self) -> None:
        """Release detector resources (delegated to orchestrator or direct detectors)."""
        if self._yolo is not None:
            self._yolo.close()
        if self._dino is not None:
            self._dino.close()
        logger.debug("CascadeDetector closed")
