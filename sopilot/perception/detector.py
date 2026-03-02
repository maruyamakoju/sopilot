"""Open-vocabulary object detection for the Perception Engine.

Provides a detection layer that accepts natural-language prompts and returns
bounding boxes with confidence scores.  Two concrete backends:

    GroundingDINODetector  — HuggingFace Grounding-DINO (GPU or CPU)
    MockDetector           — Preset / heuristic detections for testing

Every detector produces :class:`Detection` objects from ``types.py`` with
normalized [0, 1] bounding boxes so downstream components are resolution-
independent.

Example usage::

    detector = GroundingDINODetector()              # lazy-loads model
    dets = detector.detect(frame, ["person", "helmet", "gloves"])
    detector.close()                                # free VRAM
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

import numpy as np

from sopilot.perception.types import BBox, Detection, PerceptionConfig

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def non_max_suppression(
    detections: list[Detection],
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """Filter overlapping detections using greedy Non-Maximum Suppression.

    Detections are sorted by descending confidence.  For each detection we
    suppress any remaining detection whose IoU with the current one exceeds
    *iou_threshold*.  This is a pure Python/numpy implementation with no
    external dependencies.

    Parameters
    ----------
    detections:
        List of :class:`Detection` objects, potentially with overlapping boxes.
    iou_threshold:
        IoU above which a lower-confidence detection is suppressed.

    Returns
    -------
    list[Detection]
        Filtered detections with duplicates removed.
    """
    if len(detections) <= 1:
        return list(detections)

    # Sort by confidence descending.
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep: list[Detection] = []

    for det in sorted_dets:
        # Check if this detection overlaps too much with any kept detection.
        suppressed = False
        for kept in keep:
            if det.bbox.iou(kept.bbox) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(det)

    if len(keep) < len(detections):
        logger.debug(
            "NMS: %d → %d detections (iou_threshold=%.2f)",
            len(detections),
            len(keep),
            iou_threshold,
        )

    return keep


def _best_prompt_match(raw_label: str, prompts: list[str]) -> str:
    """Map a raw detection label back to the closest matching prompt string.

    Uses :func:`difflib.SequenceMatcher` for fuzzy matching.  This is needed
    because Grounding-DINO may return labels that differ slightly from the
    input prompts (e.g. "person." vs "person", capitalization differences,
    or partial matches when the prompt is a phrase).

    Parameters
    ----------
    raw_label:
        The label string returned by the model.
    prompts:
        The original list of natural-language prompts.

    Returns
    -------
    str
        The closest matching prompt, or the cleaned raw_label if no prompt
        is a reasonable match (ratio < 0.3).
    """
    if not prompts:
        return raw_label.strip().rstrip(".")

    cleaned = raw_label.strip().lower().rstrip(".")
    best_score = 0.0
    best_prompt = prompts[0]

    for prompt in prompts:
        prompt_lower = prompt.lower().strip()
        # Exact containment check first — handles "person" in "a person" etc.
        if cleaned == prompt_lower or cleaned in prompt_lower or prompt_lower in cleaned:
            return prompt
        score = SequenceMatcher(None, cleaned, prompt_lower).ratio()
        if score > best_score:
            best_score = score
            best_prompt = prompt

    # Only accept the match if the similarity is reasonable.
    if best_score >= 0.3:
        return best_prompt
    return raw_label.strip().rstrip(".")


# ── Abstract base class ──────────────────────────────────────────────────────


class ObjectDetector(ABC):
    """Abstract interface for open-vocabulary object detectors.

    All detectors accept an image frame and a list of natural-language prompts
    describing what to look for, and return a list of :class:`Detection` objects.
    """

    @abstractmethod
    def detect(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
        """Detect objects in *frame* matching the given *prompts*.

        Parameters
        ----------
        frame:
            BGR or RGB image as a numpy array of shape ``(H, W, 3)``.
        prompts:
            Natural-language class descriptions, e.g. ``["person", "hard hat"]``.

        Returns
        -------
        list[Detection]
            Detected objects with normalized bounding boxes and confidence scores.
        """

    def close(self) -> None:
        """Release model resources (VRAM, memory-mapped files, etc.).

        The default implementation is a no-op.  Subclasses that hold GPU
        resources **must** override this.
        """


# ── Grounding-DINO backend ───────────────────────────────────────────────────


class GroundingDINODetector(ObjectDetector):
    """Open-vocabulary object detector based on Grounding-DINO.

    Uses HuggingFace ``transformers`` (``AutoProcessor`` +
    ``AutoModelForZeroShotObjectDetection``).  The model is loaded lazily
    on the first call to :meth:`detect` so that construction is cheap and
    import-time side effects are avoided.

    Parameters
    ----------
    config:
        :class:`PerceptionConfig` controlling model ID, device, thresholds.
        If *None*, sensible defaults are used.
    """

    def __init__(self, config: PerceptionConfig | None = None) -> None:
        self._config = config or PerceptionConfig()
        self._model: Any = None
        self._processor: Any = None
        self._device: str | None = None
        self._loaded = False
        logger.info(
            "GroundingDINODetector created (model=%s, lazy load)",
            self._config.detector_model_id,
        )

    # ── Lazy loading ──────────────────────────────────────────────────

    def _resolve_device(self) -> str:
        """Determine the best available device."""
        requested = self._config.device
        if requested != "auto":
            return requested
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _load_model(self) -> None:
        """Load model and processor on first use."""
        if self._loaded:
            return

        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "GroundingDINODetector requires the `transformers` package.  "
                "Install it with:  pip install transformers torch\n"
                "For GPU support also install the appropriate CUDA-enabled PyTorch."
            ) from exc

        self._device = self._resolve_device()
        model_id = self._config.detector_model_id

        logger.info("Loading Grounding-DINO model '%s' on %s ...", model_id, self._device)
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self._model.to(self._device)  # type: ignore[union-attr]
        self._model.eval()  # type: ignore[union-attr]
        self._loaded = True
        logger.info("Model loaded successfully on %s", self._device)

    # ── Detection ─────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
        """Run open-vocabulary detection on *frame*.

        The *prompts* are joined into a single period-separated text string
        that Grounding-DINO uses as its text query.  Post-processing includes
        confidence thresholding, NMS, and label remapping to the closest
        original prompt.
        """
        if frame.size == 0 or not prompts:
            return []

        self._load_model()

        import torch

        # Grounding-DINO expects a PIL image and a period-separated text prompt.
        try:
            from PIL import Image as PILImage
        except ImportError as exc:
            raise ImportError(
                "GroundingDINODetector requires Pillow.  "
                "Install with:  pip install Pillow"
            ) from exc

        # Convert numpy frame (H, W, 3) to PIL.  Assume BGR if from OpenCV.
        if frame.ndim == 3 and frame.shape[2] == 3:
            # Heuristic: treat as BGR (OpenCV convention) and convert to RGB.
            rgb = frame[:, :, ::-1]
        else:
            rgb = frame
        pil_image = PILImage.fromarray(rgb.astype(np.uint8))

        h, w = frame.shape[:2]
        text_prompt = ". ".join(prompts) + "."

        # Tokenize and run inference.
        inputs = self._processor(images=pil_image, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)  # type: ignore[misc]

        # Post-process: get boxes + scores + labels above threshold.
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=self._config.detection_confidence_threshold,
            text_threshold=self._config.detection_confidence_threshold,
            target_sizes=[(h, w)],
        )

        if not results:
            return []

        result = results[0]  # single image
        boxes = result["boxes"]  # Tensor of shape (N, 4) in pixel coords
        scores = result["scores"]  # Tensor of shape (N,)
        labels = result["labels"]  # list[str] of length N

        if hasattr(boxes, "cpu"):
            boxes = boxes.cpu().numpy()
            scores = scores.cpu().numpy()

        detections: list[Detection] = []
        max_dets = self._config.max_detections_per_frame

        for i in range(min(len(scores), max_dets)):
            conf = float(scores[i])
            if conf < self._config.detection_confidence_threshold:
                continue

            # Pixel coords → normalized [0, 1].
            x1_px, y1_px, x2_px, y2_px = boxes[i]
            bbox = BBox(
                x1=float(np.clip(x1_px / w, 0.0, 1.0)),
                y1=float(np.clip(y1_px / h, 0.0, 1.0)),
                x2=float(np.clip(x2_px / w, 0.0, 1.0)),
                y2=float(np.clip(y2_px / h, 0.0, 1.0)),
            )

            # Skip degenerate boxes.
            if bbox.area < 1e-6:
                continue

            raw_label = labels[i] if i < len(labels) else prompts[0]
            matched_label = _best_prompt_match(raw_label, prompts)

            detections.append(
                Detection(bbox=bbox, label=matched_label, confidence=conf)
            )

        # Apply NMS to remove duplicates.
        detections = non_max_suppression(
            detections, iou_threshold=self._config.detection_nms_threshold
        )

        logger.debug(
            "GroundingDINO detected %d objects (prompts=%s)", len(detections), prompts
        )
        return detections

    # ── Cleanup ───────────────────────────────────────────────────────

    def close(self) -> None:
        """Free GPU memory and model references."""
        if self._model is not None:
            # Move model to CPU before deleting to free VRAM immediately.
            try:
                self._model.cpu()
            except Exception:
                pass
            self._model = None
        self._processor = None
        self._loaded = False
        logger.info("GroundingDINODetector resources released")


# ── Mock backend ──────────────────────────────────────────────────────────────


@dataclass
class MockRule:
    """A rule that tells :class:`MockDetector` what to generate.

    Each rule specifies a label to detect, a bounding box to return, a
    confidence score, and an optional *trigger* predicate that examines the
    frame to decide whether to fire.

    Parameters
    ----------
    label:
        Object label (e.g. ``"person"``).
    bbox:
        Fixed normalized bounding box to return.
    confidence:
        Detection confidence score in [0, 1].
    trigger:
        Optional callable ``(frame: np.ndarray) -> bool``.  If provided, the
        rule only fires when the trigger returns ``True``.  The default
        trigger always fires.
    jitter:
        Random jitter magnitude applied to bbox coordinates each call, making
        the detections slightly different per frame (more realistic).  0.0
        means no jitter.
    """

    label: str = "object"
    bbox: BBox = field(default_factory=lambda: BBox(0.1, 0.1, 0.3, 0.3))
    confidence: float = 0.85
    trigger: Any = None  # Callable[[np.ndarray], bool] | None
    jitter: float = 0.0


class MockDetector(ObjectDetector):
    """Deterministic or heuristic detector for testing without a GPU.

    Operates in two modes depending on how it is configured:

    1. **Rules mode** (``rules`` provided) — Returns detections defined by
       :class:`MockRule` objects.  Each rule optionally checks a trigger
       predicate on the frame.

    2. **Heuristic mode** (``use_heuristics=True``) — Analyses the frame
       content (mean brightness, color channel ratios) to decide what to
       return.  Useful for integration tests where detection results should
       vary with frame content.

    In either mode the returned :class:`Detection` objects are valid and
    compatible with the rest of the perception pipeline.

    Parameters
    ----------
    rules:
        Explicit detection rules.  If empty and ``use_heuristics`` is False,
        every call returns an empty list.
    use_heuristics:
        When True and no matching rules exist, fall back to simple frame-
        content heuristics.
    config:
        Perception configuration (used for threshold info).
    """

    def __init__(
        self,
        rules: list[MockRule] | None = None,
        use_heuristics: bool = False,
        config: PerceptionConfig | None = None,
    ) -> None:
        self._rules = rules or []
        self._use_heuristics = use_heuristics
        self._config = config or PerceptionConfig()
        self._rng = np.random.default_rng(42)
        self._call_count = 0
        logger.info(
            "MockDetector created (%d rules, heuristics=%s)",
            len(self._rules),
            use_heuristics,
        )

    def detect(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
        """Return mock detections for testing purposes.

        Parameters
        ----------
        frame:
            Image array (content is used only in heuristic mode).
        prompts:
            Prompt list.  In rules mode, only rules whose labels appear in
            *prompts* will fire (case-insensitive substring match).  In
            heuristic mode, the prompts control which heuristic branches run.
        """
        self._call_count += 1
        detections: list[Detection] = []

        # ── Rules mode ────────────────────────────────────────────────
        prompt_lower = {p.lower() for p in prompts}

        for rule in self._rules:
            # Only fire if the rule's label matches a prompt.
            label_lower = rule.label.lower()
            if not any(label_lower in p or p in label_lower for p in prompt_lower):
                continue

            # Check trigger predicate.
            if rule.trigger is not None:
                try:
                    if not rule.trigger(frame):
                        continue
                except Exception:
                    continue  # skip broken triggers silently

            # Apply optional jitter to bbox coordinates.
            bbox = rule.bbox
            if rule.jitter > 0:
                j = rule.jitter
                bbox = BBox(
                    x1=float(np.clip(bbox.x1 + self._rng.uniform(-j, j), 0.0, 1.0)),
                    y1=float(np.clip(bbox.y1 + self._rng.uniform(-j, j), 0.0, 1.0)),
                    x2=float(np.clip(bbox.x2 + self._rng.uniform(-j, j), 0.0, 1.0)),
                    y2=float(np.clip(bbox.y2 + self._rng.uniform(-j, j), 0.0, 1.0)),
                )
                # Ensure x2 > x1 and y2 > y1.
                if bbox.x2 <= bbox.x1 or bbox.y2 <= bbox.y1:
                    bbox = rule.bbox  # fall back to un-jittered

            detections.append(
                Detection(
                    bbox=bbox,
                    label=rule.label,
                    confidence=rule.confidence,
                )
            )

        # ── Heuristic mode (fallback) ────────────────────────────────
        if not detections and self._use_heuristics and frame.size > 0:
            detections = self._heuristic_detect(frame, prompts)

        logger.debug(
            "MockDetector returned %d detections (call #%d)",
            len(detections),
            self._call_count,
        )
        return detections

    def _heuristic_detect(
        self, frame: np.ndarray, prompts: list[str]
    ) -> list[Detection]:
        """Simple image-content heuristics for mock detection.

        This is intentionally simplistic — it analyses mean brightness and
        color-channel ratios to produce plausible-looking detections.  The
        purpose is to make integration tests data-dependent without needing
        a real model.
        """
        detections: list[Detection] = []
        h, w = frame.shape[:2]
        prompt_set = {p.lower() for p in prompts}

        # Overall frame statistics.
        mean_brightness = float(np.mean(frame))
        is_bright = mean_brightness > 127

        # Analyze quadrants for spatial variation.
        mid_h, mid_w = h // 2, w // 2
        quadrants = [
            frame[:mid_h, :mid_w],      # top-left
            frame[:mid_h, mid_w:],       # top-right
            frame[mid_h:, :mid_w],       # bottom-left
            frame[mid_h:, mid_w:],       # bottom-right
        ]
        quadrant_boxes = [
            BBox(0.05, 0.05, 0.45, 0.45),
            BBox(0.55, 0.05, 0.95, 0.45),
            BBox(0.05, 0.55, 0.45, 0.95),
            BBox(0.55, 0.55, 0.95, 0.95),
        ]

        for idx, (quad, qbox) in enumerate(zip(quadrants, quadrant_boxes)):
            quad_mean = float(np.mean(quad))

            # "person" heuristic: brighter-than-average quadrant.
            if "person" in prompt_set and quad_mean > mean_brightness * 1.1:
                # Tighter bbox centered in the quadrant.
                cx, cy = qbox.center
                person_box = BBox(
                    x1=max(0.0, cx - 0.08),
                    y1=max(0.0, cy - 0.18),
                    x2=min(1.0, cx + 0.08),
                    y2=min(1.0, cy + 0.18),
                )
                conf = 0.6 + 0.3 * (quad_mean / 255.0)
                detections.append(
                    Detection(bbox=person_box, label="person", confidence=min(conf, 0.95))
                )

            # Color-based heuristics (require 3-channel image).
            if quad.ndim == 3 and quad.shape[2] >= 3:
                channel_means = np.mean(quad, axis=(0, 1))
                r_ratio = channel_means[2] / (np.sum(channel_means) + 1e-8)
                g_ratio = channel_means[1] / (np.sum(channel_means) + 1e-8)
                b_ratio = channel_means[0] / (np.sum(channel_means) + 1e-8)

                # "helmet" / "hard hat" heuristic: high yellow/green ratio.
                hat_prompts = {"helmet", "hard hat", "safety helmet"}
                if hat_prompts & prompt_set and (r_ratio > 0.38 and g_ratio > 0.32):
                    cx, cy = qbox.center
                    hat_box = BBox(
                        x1=max(0.0, cx - 0.05),
                        y1=max(0.0, cy - 0.15),
                        x2=min(1.0, cx + 0.05),
                        y2=min(1.0, cy - 0.05),
                    )
                    if hat_box.area > 1e-6:
                        matched = next(iter(hat_prompts & prompt_set))
                        detections.append(
                            Detection(bbox=hat_box, label=matched, confidence=0.65)
                        )

                # "vest" / "safety vest" heuristic: bright region.
                vest_prompts = {"vest", "safety vest", "hi-vis vest"}
                if vest_prompts & prompt_set and is_bright and g_ratio > 0.35:
                    cx, cy = qbox.center
                    vest_box = BBox(
                        x1=max(0.0, cx - 0.07),
                        y1=max(0.0, cy - 0.05),
                        x2=min(1.0, cx + 0.07),
                        y2=min(1.0, cy + 0.05),
                    )
                    if vest_box.area > 1e-6:
                        matched = next(iter(vest_prompts & prompt_set))
                        detections.append(
                            Detection(bbox=vest_box, label=matched, confidence=0.55)
                        )

        # Cap total detections.
        detections = detections[: self._config.max_detections_per_frame]
        return detections

    def close(self) -> None:
        """No-op for mock detector."""
        logger.debug("MockDetector.close() — nothing to release")


# ── YOLO-World backend ────────────────────────────────────────────────────


class YOLOWorldDetector(ObjectDetector):
    """Open-vocabulary object detector using YOLO-World (ultralytics).

    Runs on CPU without a GPU. The model file (~88 MB) is downloaded
    automatically on the first call to :meth:`detect`.

    Supports dynamic vocabulary updates via ``set_classes()`` — each unique
    set of prompts triggers a single ``set_classes()`` call so repeated
    inference with the same prompts avoids redundant updates.

    Parameters
    ----------
    config:
        :class:`PerceptionConfig` controlling confidence threshold, NMS
        threshold, max detections, and device.  If *None* sensible defaults
        are used.
    model_name:
        YOLO-World model variant to load.  Defaults to
        ``"yolov8s-worldv2.pt"`` (small, ~88 MB, CPU-friendly).
    """

    DEFAULT_MODEL: str = "yolov8s-worldv2.pt"
    MEDIUM_MODEL: str = "yolov8m-worldv2.pt"
    DEFAULT_CLASSES: list[str] = [
        "person",
        "worker",
        "hard hat",
        "helmet",
        "safety vest",
        "vest",
        "gloves",
        "forklift",
        "machine",
        "vehicle",
        "pallet",
        "box",
        "equipment",
        "tool",
        "safety cone",
        "conveyor belt",
    ]

    def __init__(
        self,
        config: PerceptionConfig | None = None,
        model_name: str | None = None,
    ) -> None:
        self._config = config or PerceptionConfig()
        self._model_name = model_name or self.DEFAULT_MODEL
        self._model: Any = None
        self._current_classes: list[str] = []
        self._loaded = False
        logger.info(
            "YOLOWorldDetector created (model=%s, lazy load)",
            self._model_name,
        )

    # ── Lazy loading ──────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load YOLO-World model on first use."""
        if self._loaded:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "YOLOWorldDetector requires the `ultralytics` package. "
                "Install with:  pip install ultralytics"
            ) from exc

        logger.info("Loading YOLO-World model '%s' ...", self._model_name)
        self._model = YOLO(self._model_name)
        self._model.set_classes(self.DEFAULT_CLASSES)
        self._current_classes = list(self.DEFAULT_CLASSES)
        self._loaded = True
        logger.info("YOLO-World model loaded (classes=%s)", self._current_classes)

    def _maybe_update_classes(self, prompts: list[str]) -> None:
        """Update classes only if prompts differ from current set."""
        if not prompts or set(prompts) == set(self._current_classes):
            return
        self._model.set_classes(prompts)
        self._current_classes = list(prompts)
        logger.debug("YOLO-World classes updated: %s", prompts)

    # ── Detection ─────────────────────────────────────────────────────

    def _detect_raw(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
        """Run YOLO-World inference on a single frame or tile.

        Returns detections with normalized coordinates relative to *frame*.
        Capped at ``max_detections_per_frame`` to bound per-tile overhead.
        """
        h, w = frame.shape[:2]
        conf_threshold = self._config.yolo_confidence_threshold

        try:
            results = self._model.predict(
                frame,
                conf=conf_threshold,
                iou=self._config.detection_nms_threshold,
                verbose=False,
                device="cpu",
            )
        except Exception as exc:
            logger.warning("YOLO-World inference failed: %s", exc)
            return []

        detections: list[Detection] = []
        max_dets = self._config.max_detections_per_frame

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            names = r.names  # dict[int, str]
            for box in boxes:
                if len(detections) >= max_dets:
                    break
                conf = float(box.conf[0])
                if conf < conf_threshold:
                    continue
                cls_id = int(box.cls[0])
                label = names.get(cls_id, prompts[0] if prompts else "object")

                xyxy = box.xyxy[0]
                if hasattr(xyxy, "cpu"):
                    xyxy = xyxy.cpu().numpy()
                x1_px, y1_px, x2_px, y2_px = xyxy

                bbox = BBox(
                    x1=float(np.clip(x1_px / w, 0.0, 1.0)),
                    y1=float(np.clip(y1_px / h, 0.0, 1.0)),
                    x2=float(np.clip(x2_px / w, 0.0, 1.0)),
                    y2=float(np.clip(y2_px / h, 0.0, 1.0)),
                )
                if bbox.area < 1e-6:
                    continue
                detections.append(Detection(bbox=bbox, label=label, confidence=conf))

        return detections

    def _sahi_detect(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
        """SAHI: Slicing Aided Hyper Inference for small-object detection.

        Slices *frame* into overlapping tiles of size
        (``sahi_slice_height`` × ``sahi_slice_width``) with ``sahi_overlap_ratio``
        overlap, runs :meth:`_detect_raw` on each tile, remaps coordinates back
        to original image space, and applies a final global NMS.  A full-frame
        pass is also included to catch large objects that span multiple tiles.

        For frames smaller than a single tile, falls back to a single
        :meth:`_detect_raw` call (no extra overhead).
        """
        h, w = frame.shape[:2]
        slice_h = self._config.sahi_slice_height
        slice_w = self._config.sahi_slice_width
        overlap = self._config.sahi_overlap_ratio

        # Fast path: frame fits in one tile — no slicing needed.
        if h <= slice_h and w <= slice_w:
            dets = self._detect_raw(frame, prompts)
            dets = non_max_suppression(dets, iou_threshold=self._config.detection_nms_threshold)
            return dets[: self._config.max_detections_per_frame]

        stride_h = max(1, int(slice_h * (1.0 - overlap)))
        stride_w = max(1, int(slice_w * (1.0 - overlap)))

        # Generate tile top-left y positions.
        y_starts: list[int] = list(range(0, h - slice_h + 1, stride_h))
        if not y_starts or y_starts[-1] + slice_h < h:
            y_starts.append(max(0, h - slice_h))

        # Generate tile top-left x positions.
        x_starts: list[int] = list(range(0, w - slice_w + 1, stride_w))
        if not x_starts or x_starts[-1] + slice_w < w:
            x_starts.append(max(0, w - slice_w))

        all_dets: list[Detection] = []

        for y1_px in y_starts:
            for x1_px in x_starts:
                y2_px = min(y1_px + slice_h, h)
                x2_px = min(x1_px + slice_w, w)
                tile = frame[y1_px:y2_px, x1_px:x2_px]
                if tile.size == 0:
                    continue

                tile_dets = self._detect_raw(tile, prompts)

                # Remap normalised tile coords → original image coords.
                th, tw = tile.shape[:2]
                for det in tile_dets:
                    orig_bbox = BBox(
                        x1=float(np.clip((x1_px + det.bbox.x1 * tw) / w, 0.0, 1.0)),
                        y1=float(np.clip((y1_px + det.bbox.y1 * th) / h, 0.0, 1.0)),
                        x2=float(np.clip((x1_px + det.bbox.x2 * tw) / w, 0.0, 1.0)),
                        y2=float(np.clip((y1_px + det.bbox.y2 * th) / h, 0.0, 1.0)),
                    )
                    if orig_bbox.area < 1e-6:
                        continue
                    all_dets.append(Detection(bbox=orig_bbox, label=det.label, confidence=det.confidence))

        # Full-frame pass catches large objects that span multiple tiles.
        all_dets.extend(self._detect_raw(frame, prompts))

        logger.debug(
            "SAHI: %d tiles, prompts=%s → %d raw detections before NMS",
            len(y_starts) * len(x_starts),
            prompts,
            len(all_dets),
        )

        final = non_max_suppression(all_dets, iou_threshold=self._config.detection_nms_threshold)
        return final[: self._config.max_detections_per_frame]

    def detect(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
        """Run YOLO-World detection on *frame*.

        When ``config.sahi_enabled`` is True (default), uses
        :meth:`_sahi_detect` which slices the frame into overlapping tiles for
        better small-object recall in high-resolution footage.  Set
        ``sahi_enabled=False`` for real-time RTSP streams where latency matters.

        Parameters
        ----------
        frame:
            BGR image as numpy array ``(H, W, 3)``.
        prompts:
            Natural-language class names.  If empty, :attr:`DEFAULT_CLASSES`
            are used.
        """
        if frame.size == 0:
            return []

        effective_prompts = prompts if prompts else self.DEFAULT_CLASSES
        self._load_model()
        self._maybe_update_classes(effective_prompts)

        if self._config.sahi_enabled:
            detections = self._sahi_detect(frame, effective_prompts)
        else:
            detections = self._detect_raw(frame, effective_prompts)
            detections = non_max_suppression(
                detections, iou_threshold=self._config.detection_nms_threshold
            )
            detections = detections[: self._config.max_detections_per_frame]

        logger.debug(
            "YOLO-World detected %d objects (sahi=%s, prompts=%s)",
            len(detections),
            self._config.sahi_enabled,
            effective_prompts,
        )
        return detections

    # ── Cleanup ───────────────────────────────────────────────────────

    def close(self) -> None:
        """Release model resources."""
        self._model = None
        self._current_classes = []
        self._loaded = False
        logger.info("YOLOWorldDetector resources released")
