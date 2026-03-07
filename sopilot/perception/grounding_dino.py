"""Grounding DINO 1.5 Detector — open-vocabulary object detection.

Phase 39: Production-grade Stage 2 detector for the confidence cascade.
Grounding DINO 1.5 Pro/Edge from IDEA-Research substantially improves
upon the original 1.0:

    - 2× better zero-shot accuracy on COCO/LVIS
    - Improved text-image alignment
    - Better small-object detection
    - Lower false positive rate

Architecture:
    Text encoder: BERT-like (SBERT)
    Vision encoder: SwinB (1.5-edge) or SwinL (1.5-pro)
    Decoder: multi-scale deformable cross-attention

Integrates with ModelOrchestrator (priority=6, vram=1.5–2.0 GB):

    detector = build_grounding_dino(variant="1.5-pro", orchestrator=orchestrator)
    cascade = CascadeDetector(dino_detector=detector, ...)

Standalone usage::

    det = GroundingDINODetector(variant="1.5-edge")
    results = det.detect(frame, ["person", "helmet", "safety vest"])

Model variants:
    "1.5-pro"   — SwinL backbone, highest accuracy (~2.0 GB VRAM)
    "1.5-edge"  — SwinB backbone, faster inference (~1.5 GB VRAM)
    "1.0"       — Original Grounding DINO (compatibility / 0.8 GB VRAM)
    "mock"      — Deterministic mock detector for testing
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from sopilot.perception.detector import ObjectDetector
from sopilot.perception.types import BBox, Detection, PerceptionConfig

logger = logging.getLogger(__name__)

# Model registry (HuggingFace IDs)
_MODEL_IDS: dict[str, str] = {
    "1.5-pro": "IDEA-Research/grounding-dino-1.5-pro",
    "1.5-edge": "IDEA-Research/grounding-dino-1.5-edge",
    "1.0": "IDEA-Research/grounding-dino-tiny",
}
_DEFAULT_VARIANT = "1.5-pro"
_MODEL_NAME = "grounding_dino"
_VRAM_GB: dict[str, float] = {
    "1.5-pro": 2.0,
    "1.5-edge": 1.5,
    "1.0": 0.8,
}
_PRIORITY = 6  # on-demand, before YOLO (9) but before LLaVA (3)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class GroundingDINOStats:
    """Per-session statistics for the Grounding DINO detector."""

    total_frames: int = 0
    total_detections: int = 0
    total_latency_ms: float = 0.0
    load_count: int = 0
    errors: int = 0

    @property
    def avg_detections_per_frame(self) -> float:
        return self.total_detections / max(1, self.total_frames)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(1, self.total_frames)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "avg_detections_per_frame": round(self.avg_detections_per_frame, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "load_count": self.load_count,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# GroundingDINODetector
# ---------------------------------------------------------------------------


class GroundingDINODetector(ObjectDetector):
    """Grounding DINO 1.5 open-vocabulary detector.

    Implements :class:`ObjectDetector` for drop-in use in the confidence
    cascade pipeline as Stage 2 (escalation path).

    Parameters
    ----------
    variant:
        Model variant: ``"1.5-pro"``, ``"1.5-edge"``, or ``"1.0"``.
    orchestrator:
        Optional :class:`ModelOrchestrator` for VRAM lifecycle.
    config:
        :class:`PerceptionConfig` for detection thresholds.
    box_threshold:
        Minimum box objectness score.
    text_threshold:
        Minimum text–image alignment score.
    device:
        Torch device (``"auto"``, ``"cuda"``, ``"cpu"``).
    """

    def __init__(
        self,
        variant: str = _DEFAULT_VARIANT,
        orchestrator: Any = None,
        config: PerceptionConfig | None = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: str = "auto",
    ) -> None:
        self._variant = variant
        self._orchestrator = orchestrator
        self._config = config or PerceptionConfig()
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._device = device
        self._stats = GroundingDINOStats()

        # Lazy-loaded model
        self._processor: Any = None
        self._model: Any = None
        self._loaded = False

        vram = _VRAM_GB.get(variant, 2.0)

        # Register with orchestrator for VRAM lifecycle management
        if orchestrator is not None:
            try:
                orchestrator.register(
                    _MODEL_NAME,
                    self._load_model,
                    vram_gb=vram,
                    priority=_PRIORITY,
                    description=f"Grounding DINO {variant}",
                )
                logger.info(
                    "GroundingDINODetector registered with orchestrator "
                    "(variant=%s, vram=%.1f GB, priority=%d)",
                    variant,
                    vram,
                    _PRIORITY,
                )
            except Exception:
                logger.debug(
                    "Failed to register Grounding DINO with orchestrator", exc_info=True
                )

        logger.info(
            "GroundingDINODetector created "
            "(variant=%s, box_thresh=%.2f, text_thresh=%.2f)",
            variant,
            box_threshold,
            text_threshold,
        )

    # ── Model lifecycle ────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load model and processor (lazy on first detect call)."""
        if self._loaded:
            return

        import torch

        model_id = _MODEL_IDS.get(self._variant, _MODEL_IDS[_DEFAULT_VARIANT])
        logger.info("Loading Grounding DINO %s from %s ...", self._variant, model_id)

        try:
            # Primary loader: transformers AutoModel (works for 1.0 and 1.5)
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = AutoProcessor.from_pretrained(model_id)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
            self._model.eval()
            self._device = device
            logger.info(
                "Grounding DINO %s loaded via transformers on %s",
                self._variant,
                device,
            )
        except Exception:
            logger.debug(
                "transformers loader failed for %s; trying groundingdino package",
                self._variant,
                exc_info=True,
            )
            # Fallback: original groundingdino package (1.0 only)
            from groundingdino.util.inference import load_model as _gd_load

            self._model = _gd_load(
                "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                "weights/groundingdino_swint_ogc.pth",
            )
            self._processor = None

        self._stats.load_count += 1
        self._loaded = True
        logger.info("Grounding DINO model loaded successfully")

    def _ensure_loaded(self) -> bool:
        if not self._loaded:
            try:
                self._load_model()
            except Exception:
                logger.debug("Grounding DINO load failed", exc_info=True)
                return False
        return self._loaded

    def _frame_to_pil(self, frame: np.ndarray):
        import cv2
        from PIL import Image as PILImage

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(rgb)

    @staticmethod
    def _prompts_to_text(prompts: list[str]) -> str:
        """Format prompts for Grounding DINO text encoder.

        Grounding DINO expects prompts joined with '. ' and a trailing '.'.
        """
        return ". ".join(p.strip().rstrip(".") for p in prompts) + "."

    # ── ObjectDetector interface ───────────────────────────────────────

    def detect(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
        """Run Grounding DINO 1.5 detection on a single frame.

        Parameters
        ----------
        frame:
            BGR image (H, W, 3).
        prompts:
            Natural-language class descriptions.

        Returns
        -------
        list[Detection]
            Detected objects with normalized BBox and confidence score.
        """
        if frame is None or frame.size == 0 or not prompts:
            return []

        self._stats.total_frames += 1
        t0 = time.monotonic()

        if not self._ensure_loaded():
            self._stats.errors += 1
            return []

        try:
            import torch

            image = self._frame_to_pil(frame)
            text = self._prompts_to_text(prompts)
            h, w = frame.shape[:2]

            inputs = self._processor(images=image, text=text, return_tensors="pt")
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Post-process to pixel boxes
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=self._box_threshold,
                text_threshold=self._text_threshold,
                target_sizes=[(h, w)],
            )

            detections: list[Detection] = []
            if results:
                result = results[0]
                boxes = result.get("boxes", [])
                scores = result.get("scores", [])
                labels = result.get("labels", [])

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = (
                        box.tolist() if hasattr(box, "tolist") else map(float, box)
                    )
                    detections.append(
                        Detection(
                            bbox=BBox(
                                x1=max(0.0, x1 / w),
                                y1=max(0.0, y1 / h),
                                x2=min(1.0, x2 / w),
                                y2=min(1.0, y2 / h),
                            ),
                            label=str(label),
                            confidence=float(score),
                            attributes={
                                "stage": "grounding_dino_1.5",
                                "variant": self._variant,
                            },
                        )
                    )

            elapsed = (time.monotonic() - t0) * 1000
            self._stats.total_latency_ms += elapsed
            self._stats.total_detections += len(detections)
            logger.debug(
                "Grounding DINO %s: %d detections (%.1f ms)",
                self._variant,
                len(detections),
                elapsed,
            )
            return detections

        except Exception:
            self._stats.errors += 1
            logger.debug("Grounding DINO detect failed", exc_info=True)
            return []

    def get_stats(self) -> dict[str, Any]:
        return {
            "variant": self._variant,
            "loaded": self._loaded,
            "box_threshold": self._box_threshold,
            "text_threshold": self._text_threshold,
            **self._stats.to_dict(),
        }

    def close(self) -> None:
        """Release model resources and free VRAM."""
        if self._loaded:
            try:
                import torch

                del self._model
                del self._processor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._model = None
                self._processor = None
                self._loaded = False
                logger.info("Grounding DINO model released")
            except Exception:
                logger.debug("Grounding DINO close failed", exc_info=True)


# ---------------------------------------------------------------------------
# Mock detector (for testing / no-GPU environments)
# ---------------------------------------------------------------------------


class MockGroundingDINODetector(ObjectDetector):
    """Mock Grounding DINO detector for testing.

    Returns synthetic detections with configurable properties.
    Deterministic and thread-safe.

    Parameters
    ----------
    num_detections:
        Number of synthetic detections to return per frame.
    base_confidence:
        Base detection confidence (varied slightly per detection).
    variant:
        Reported variant string in detection attributes.
    """

    def __init__(
        self,
        num_detections: int = 2,
        base_confidence: float = 0.55,
        variant: str = "mock",
    ) -> None:
        self._num_detections = num_detections
        self._base_confidence = base_confidence
        self._variant = variant
        self._stats = GroundingDINOStats()
        self._call_count = 0

    def detect(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
        if frame is None or frame.size == 0 or not prompts:
            return []

        self._stats.total_frames += 1
        t0 = time.monotonic()

        num = min(self._num_detections, len(prompts))
        detections: list[Detection] = []

        for i in range(num):
            # Deterministic coordinates that vary between calls
            x1 = max(0.05, min(0.7, 0.1 + (self._call_count % 5) * 0.08 + i * 0.18))
            y1 = max(0.05, min(0.7, 0.1 + i * 0.15))
            x2 = min(0.95, x1 + 0.15)
            y2 = min(0.95, y1 + 0.30)
            conf_delta = (i % 3 - 1) * 0.04
            confidence = max(0.05, min(0.95, self._base_confidence + conf_delta))

            detections.append(
                Detection(
                    bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    label=prompts[i % len(prompts)],
                    confidence=confidence,
                    attributes={
                        "stage": "grounding_dino_mock",
                        "variant": self._variant,
                        "mock": True,
                    },
                )
            )

        self._call_count += 1
        elapsed = (time.monotonic() - t0) * 1000
        self._stats.total_latency_ms += elapsed
        self._stats.total_detections += len(detections)
        return detections

    def get_stats(self) -> dict[str, Any]:
        return {
            "variant": self._variant,
            "loaded": True,
            "mock": True,
            **self._stats.to_dict(),
        }

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_grounding_dino(
    variant: str = _DEFAULT_VARIANT,
    orchestrator: Any = None,
    config: PerceptionConfig | None = None,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: str = "auto",
    backend: str = "auto",
) -> ObjectDetector:
    """Build a Grounding DINO detector.

    Parameters
    ----------
    variant:
        ``"1.5-pro"``, ``"1.5-edge"``, ``"1.0"``, or ``"mock"``.
    orchestrator:
        Optional :class:`ModelOrchestrator`.
    config:
        Optional :class:`PerceptionConfig`.
    box_threshold:
        Box confidence threshold.
    text_threshold:
        Text alignment threshold.
    device:
        Torch device.
    backend:
        ``"grounding_dino"`` — real model.
        ``"mock"`` — MockGroundingDINODetector.
        ``"auto"`` — tries real model, falls back to mock if CUDA unavailable.

    Returns
    -------
    ObjectDetector
        :class:`GroundingDINODetector` or :class:`MockGroundingDINODetector`.
    """
    if variant == "mock" or backend == "mock":
        logger.info("build_grounding_dino: using MockGroundingDINODetector")
        return MockGroundingDINODetector()

    if backend == "auto":
        try:
            import torch

            if not torch.cuda.is_available():
                logger.info(
                    "build_grounding_dino: CUDA not available → MockGroundingDINODetector"
                )
                return MockGroundingDINODetector()
        except ImportError:
            logger.info(
                "build_grounding_dino: torch not available → MockGroundingDINODetector"
            )
            return MockGroundingDINODetector()

    return GroundingDINODetector(
        variant=variant,
        orchestrator=orchestrator,
        config=config,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )
