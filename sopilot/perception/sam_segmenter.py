"""SAM Segmenter --- precision segmentation via Segment Anything Model.

Phase 34: Integrates SAM (Segment Anything Model) with the perception
orchestration stack to produce pixel-precise segmentation masks from
detection bounding boxes.

The segmenter accepts detections from the cascade detector and produces
:class:`SegmentationMask` objects containing boolean masks, area ratios,
and SAM's predicted IoU quality scores.  Two prompting strategies are
supported:

    1. **Box prompts** --- each detection bbox becomes a SAM box prompt
       (fast, works for well-framed objects).
    2. **Attention-guided point prompts** --- DINOv2 self-attention maps
       are sampled within the bbox to generate high-confidence point
       prompts for SAM, improving segmentation of occluded or oddly
       shaped objects.

Integration::

    segmenter = SAMSegmenter(orchestrator, dinov2_backbone)
    masks = segmenter.segment_frame(frame, detections)
    mask  = segmenter.segment_detection(frame, detection)

    # With DINOv2 attention guidance:
    masks = segmenter.segment_with_attention(frame, detections, dinov2_backbone)

    stats = segmenter.get_stats()

Requires: ``sam2`` package (``pip install segment-anything-2``).
Falls back gracefully when SAM is not installed.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sopilot.perception.types import BBox, Detection

logger = logging.getLogger(__name__)

# Default SAM configuration
_DEFAULT_VARIANT = "sam2_hiera_small"
_VRAM_ESTIMATE_GB = 2.5
_MODEL_PRIORITY = 5  # On-demand; lower than DINOv2 (9) and YOLO (9)


# ---------------------------------------------------------------------------
# SegmentationMask --- immutable result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentationMask:
    """Pixel-precise segmentation mask for a single detected object.

    Attributes
    ----------
    mask:
        Boolean array of shape ``(H, W)`` where ``True`` pixels belong
        to the segmented object.
    area:
        Fraction of the frame covered by the mask (0.0 -- 1.0).
    bbox:
        Tight bounding box around the mask in normalized [0, 1] coords.
    iou_prediction:
        SAM's self-predicted IoU quality score for this mask.
    label:
        Semantic label inherited from the source detection.
    attributes:
        Extra metadata (e.g. ``{"source": "box_prompt"}``).
    """

    mask: np.ndarray  # (H, W) bool
    area: float
    bbox: BBox
    iou_prediction: float
    label: str
    attributes: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SAMSegmenter
# ---------------------------------------------------------------------------


class SAMSegmenter:
    """Precision segmenter backed by SAM 2 (Segment Anything Model).

    All expensive operations (model load, inference) go through the
    :class:`ModelOrchestrator` for VRAM-aware lifecycle management.
    When the orchestrator is ``None`` the segmenter manages its own
    model lifecycle (standalone mode).

    Parameters
    ----------
    orchestrator:
        :class:`ModelOrchestrator` instance managing GPU resources.
        If ``None``, the segmenter operates in standalone mode.
    dinov2_backbone:
        Optional :class:`DINOv2Backbone` for attention-guided prompts.
    model_variant:
        SAM 2 model variant name.  Default ``"sam2_hiera_small"``.
    device:
        Override device (default: use orchestrator's device or ``"cpu"``).
    """

    MODEL_NAME = "sam2"

    def __init__(
        self,
        orchestrator: Any = None,
        dinov2_backbone: Any = None,
        model_variant: str = _DEFAULT_VARIANT,
        device: str | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._dinov2 = dinov2_backbone
        self._variant = model_variant
        self._device_override = device
        self._lock = threading.RLock()

        # Standalone mode state (used when no orchestrator)
        self._model: Any = None
        self._loaded = False

        # Statistics
        self._total_segments: int = 0
        self._total_frames: int = 0
        self._total_attention_segments: int = 0
        self._total_inference_ms: float = 0.0
        self._failures: int = 0

        if orchestrator is not None:
            self._register_with_orchestrator()

        logger.info(
            "SAMSegmenter created: variant=%s, orchestrator=%s, dinov2=%s",
            model_variant,
            "yes" if orchestrator else "standalone",
            "yes" if dinov2_backbone else "no",
        )

    # ── Orchestrator registration ─────────────────────────────────────

    def _register_with_orchestrator(self) -> None:
        """Register SAM with the model orchestrator."""
        self._orchestrator.register(
            name=self.MODEL_NAME,
            load_fn=self._load_model,
            unload_fn=self._unload_model,
            vram_estimate_gb=_VRAM_ESTIMATE_GB,
            priority=_MODEL_PRIORITY,
            device=self._device_override,
        )

    def _load_model(self) -> Any:
        """Load SAM 2 model and build an image predictor.

        Attempts to import ``sam2`` and build the model from a
        checkpoint.  The exact loading pattern is kept flexible to
        accommodate different SAM versions and checkpoint locations.

        Returns
        -------
        SAM2ImagePredictor
            Ready-to-use predictor instance.
        """
        try:
            from sam2.build_sam import build_sam2  # type: ignore[import-untyped]
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "sam2 package not installed.  SAMSegmenter will be unavailable.  "
                "Install with: pip install segment-anything-2"
            )
            raise

        device_str = self._device_override or (
            self._orchestrator.device if self._orchestrator else "cpu"
        )

        logger.info("Loading SAM 2 model '%s' on %s ...", self._variant, device_str)

        # build_sam2 returns the model; we wrap it in SAM2ImagePredictor
        model = build_sam2(self._variant, device=device_str)
        predictor = SAM2ImagePredictor(model)

        logger.info("SAM 2 loaded: variant=%s, device=%s", self._variant, device_str)
        return predictor

    @staticmethod
    def _unload_model(model: Any) -> None:
        """Move model to CPU and release references."""
        try:
            if hasattr(model, "model"):
                model.model.cpu()
            elif hasattr(model, "cpu"):
                model.cpu()
        except Exception:
            pass

    # ── Standalone mode lifecycle ─────────────────────────────────────

    def _ensure_loaded(self) -> Any:
        """Get the loaded predictor, loading if necessary.

        Returns ``None`` when SAM cannot be loaded (graceful fallback).
        """
        if self._orchestrator is not None:
            try:
                return self._orchestrator.acquire(self.MODEL_NAME)
            except (KeyError, RuntimeError):
                logger.debug("SAM not available via orchestrator")
                return None

        with self._lock:
            if not self._loaded:
                try:
                    self._model = self._load_model()
                    self._loaded = True
                except Exception:
                    logger.debug("SAM load failed; segmenter unavailable")
                    return None
            return self._model

    def _release(self) -> None:
        """Release the model after use."""
        if self._orchestrator is not None:
            try:
                self._orchestrator.release(self.MODEL_NAME)
            except Exception:
                pass

    # ── Public API: single detection ──────────────────────────────────

    def segment_detection(
        self, frame: np.ndarray, detection: Detection
    ) -> SegmentationMask | None:
        """Segment a single detection using its bbox as a box prompt.

        Parameters
        ----------
        frame:
            BGR image as numpy array ``(H, W, 3)``.
        detection:
            A :class:`Detection` whose bbox defines the box prompt.

        Returns
        -------
        SegmentationMask or None
            The best-quality mask, or ``None`` if SAM is unavailable or
            inference fails.
        """
        if frame.size == 0:
            return None

        h, w = frame.shape[:2]
        px1, py1, px2, py2 = detection.bbox.to_pixels(w, h)
        box_prompt = [px1, py1, px2, py2]

        results = self._run_sam(frame, box_prompts=[box_prompt])
        if not results:
            return None

        # Pick the mask with highest predicted IoU
        best = max(results, key=lambda r: r.get("iou", 0.0))
        return self._mask_to_segmentation(
            best, frame.shape, detection.label, detection.attributes
        )

    # ── Public API: batch segmentation ────────────────────────────────

    def segment_frame(
        self, frame: np.ndarray, detections: list[Detection]
    ) -> list[SegmentationMask | None]:
        """Segment all detections in a frame using box prompts.

        Parameters
        ----------
        frame:
            BGR image as numpy array ``(H, W, 3)``.
        detections:
            List of detections to segment.

        Returns
        -------
        list[SegmentationMask | None]
            One mask per detection (in the same order).  Entries are
            None for detections that fail to segment.
        """
        if frame.size == 0 or not detections:
            return []

        self._total_frames += 1
        h, w = frame.shape[:2]

        box_prompts = []
        for det in detections:
            px1, py1, px2, py2 = det.bbox.to_pixels(w, h)
            box_prompts.append([px1, py1, px2, py2])

        results = self._run_sam(frame, box_prompts=box_prompts)
        if not results:
            return [None] * len(detections)

        masks: list[SegmentationMask | None] = []
        valid_count = 0
        for raw, det in zip(results, detections):
            seg = self._mask_to_segmentation(
                raw, frame.shape, det.label, det.attributes
            )
            masks.append(seg)
            if seg is not None:
                valid_count += 1

        self._total_segments += valid_count
        return masks

    # ── Public API: attention-guided segmentation ─────────────────────

    def segment_with_attention(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        dinov2_backbone: Any = None,
    ) -> list[SegmentationMask]:
        """Segment detections using DINOv2 attention maps as point prompts.

        High-attention regions within each detection bbox are sampled as
        positive point prompts for SAM, improving segmentation quality
        for occluded or irregularly shaped objects.

        Parameters
        ----------
        frame:
            BGR image as numpy array ``(H, W, 3)``.
        detections:
            List of detections to segment.
        dinov2_backbone:
            Optional :class:`DINOv2Backbone` override.  Falls back to
            the instance-level backbone, then to plain box prompts.

        Returns
        -------
        list[SegmentationMask]
            One mask per detection where segmentation succeeds.
        """
        if frame.size == 0 or not detections:
            return []

        backbone = dinov2_backbone or self._dinov2
        if backbone is None:
            # Fall back to box-only prompts
            logger.debug("No DINOv2 backbone; falling back to box prompts")
            return self.segment_frame(frame, detections)

        self._total_frames += 1
        h, w = frame.shape[:2]

        # Get attention map (H_patches, W_patches)
        try:
            attention_map = backbone.get_attention_map(frame)
        except Exception:
            logger.debug("DINOv2 attention extraction failed; falling back to box prompts")
            return self.segment_frame(frame, detections)

        masks: list[SegmentationMask] = []
        for det in detections:
            # Generate point prompts from attention within the bbox
            point_prompts = self._generate_point_prompts(
                attention_map, det.bbox, n_points=3
            )

            # Also pass the box as a box prompt for stability
            px1, py1, px2, py2 = det.bbox.to_pixels(w, h)
            box_prompt = [px1, py1, px2, py2]

            results = self._run_sam(
                frame,
                box_prompts=[box_prompt],
                point_prompts=point_prompts,
            )
            if results:
                best = max(results, key=lambda r: r.get("iou", 0.0))
                seg = self._mask_to_segmentation(
                    best,
                    frame.shape,
                    det.label,
                    {**det.attributes, "prompt_type": "attention"},
                )
                if seg is not None:
                    masks.append(seg)

        self._total_attention_segments += len(masks)
        self._total_segments += len(masks)
        return masks

    # ── Point prompt generation ───────────────────────────────────────

    @staticmethod
    def _generate_point_prompts(
        attention_map: np.ndarray,
        detection_bbox: BBox,
        n_points: int = 3,
    ) -> list[tuple[int, int]]:
        """Extract high-attention points within a bbox as SAM point prompts.

        The attention map (from DINOv2) is cropped to the detection bbox
        region and the top-*n_points* highest-attention locations are
        returned as ``(x_pixel, y_pixel)`` tuples in the *attention map*
        coordinate space, which callers must scale to frame pixels.

        Parameters
        ----------
        attention_map:
            Attention map of shape ``(H_patches, W_patches)`` with values
            in ``[0, 1]``.
        detection_bbox:
            The detection bounding box in normalized [0, 1] coordinates.
        n_points:
            Number of point prompts to generate.

        Returns
        -------
        list[tuple[int, int]]
            Up to *n_points* ``(x, y)`` coordinates in patch-grid space,
            scaled to approximate pixel coordinates assuming the attention
            map covers the full frame.
        """
        ah, aw = attention_map.shape[:2]

        # Map normalized bbox to attention-map coordinates
        ax1 = max(0, int(detection_bbox.x1 * aw))
        ay1 = max(0, int(detection_bbox.y1 * ah))
        ax2 = min(aw, int(detection_bbox.x2 * aw))
        ay2 = min(ah, int(detection_bbox.y2 * ah))

        if ax2 <= ax1 or ay2 <= ay1:
            return []

        # Crop attention to bbox region
        region = attention_map[ay1:ay2, ax1:ax2]
        if region.size == 0:
            return []

        # Flatten and pick top-N indices
        flat = region.flatten()
        k = min(n_points, flat.size)
        top_indices = np.argpartition(flat, -k)[-k:]
        # Sort by descending attention value
        top_indices = top_indices[np.argsort(flat[top_indices])[::-1]]

        points: list[tuple[int, int]] = []
        region_w = ax2 - ax1
        for idx in top_indices:
            ry = int(idx // region_w)
            rx = int(idx % region_w)
            # Convert back to full attention-map coords, then to pixel coords
            # Attention map covers the full frame, so scale proportionally
            px = int((ax1 + rx + 0.5) / aw * 1.0)  # Stays normalized for now
            py = int((ay1 + ry + 0.5) / ah * 1.0)
            # We store as approximate pixel coords by scaling by attention dims
            # (the caller in _run_sam will use actual frame dims)
            points.append((ax1 + rx, ay1 + ry))

        return points

    # ── Core SAM inference ────────────────────────────────────────────

    def _run_sam(
        self,
        frame: np.ndarray,
        box_prompts: list[list[int]] | None = None,
        point_prompts: list[tuple[int, int]] | None = None,
    ) -> list[dict[str, Any]]:
        """Run SAM inference with the given prompts.

        Parameters
        ----------
        frame:
            BGR image as numpy array ``(H, W, 3)``.
        box_prompts:
            List of ``[x1, y1, x2, y2]`` pixel-coordinate boxes.
        point_prompts:
            List of ``(x, y)`` positive point prompts in patch-grid coords.
            Scaled to pixel coordinates using the frame dimensions.

        Returns
        -------
        list[dict]
            Raw results, each with keys ``"mask"`` (H, W bool),
            ``"iou"`` (float), and ``"bbox_pixels"`` (x1, y1, x2, y2).
        """
        predictor = self._ensure_loaded()
        if predictor is None:
            return []

        t0 = time.monotonic()
        try:
            import torch

            h, w = frame.shape[:2]

            # SAM expects RGB
            if frame.ndim == 3 and frame.shape[2] == 3:
                rgb = frame[:, :, ::-1].copy()
            else:
                rgb = frame

            predictor.set_image(rgb)

            # Build prompt tensors
            input_boxes = None
            input_points = None
            input_labels = None

            if box_prompts:
                input_boxes = torch.tensor(box_prompts, dtype=torch.float32)

            if point_prompts:
                # Scale attention-grid coords to pixel coords
                ah = max(1, int(np.sqrt(len(point_prompts) * 3)))  # heuristic
                # Use frame dimensions directly --- point_prompts are in
                # attention-map patch-grid space; scale to pixels
                attn_h, attn_w = 37, 37  # DINOv2 ViT-B/14 default patch grid
                scaled = [
                    (int((px + 0.5) / attn_w * w), int((py + 0.5) / attn_h * h))
                    for px, py in point_prompts
                ]
                input_points = torch.tensor(
                    [[x, y] for x, y in scaled], dtype=torch.float32
                ).unsqueeze(0)  # (1, N, 2)
                # All points are foreground (label = 1)
                input_labels = torch.ones(
                    1, len(scaled), dtype=torch.int64
                )  # (1, N)

            # Run prediction
            masks_np, iou_preds, _ = predictor.predict(
                point_coords=input_points.numpy() if input_points is not None else None,
                point_labels=input_labels.numpy() if input_labels is not None else None,
                box=input_boxes.numpy() if input_boxes is not None else None,
                multimask_output=True,
            )

            # Build result list
            results: list[dict[str, Any]] = []
            if masks_np.ndim == 3:
                # Single prompt: (N_masks, H, W)
                for i in range(masks_np.shape[0]):
                    mask_bool = masks_np[i].astype(bool)
                    iou_val = float(iou_preds[i]) if i < len(iou_preds) else 0.0
                    results.append({
                        "mask": mask_bool,
                        "iou": iou_val,
                    })
            elif masks_np.ndim == 4:
                # Batch prompts: (B, N_masks, H, W)
                for b in range(masks_np.shape[0]):
                    # Pick the best mask for each prompt
                    best_idx = int(np.argmax(iou_preds[b]))
                    mask_bool = masks_np[b, best_idx].astype(bool)
                    iou_val = float(iou_preds[b, best_idx])
                    results.append({
                        "mask": mask_bool,
                        "iou": iou_val,
                    })

            elapsed_ms = (time.monotonic() - t0) * 1000
            self._total_inference_ms += elapsed_ms
            logger.debug(
                "SAM inference: %d prompts -> %d masks in %.1f ms",
                len(box_prompts or []),
                len(results),
                elapsed_ms,
            )
            return results

        except Exception:
            self._failures += 1
            logger.debug("SAM inference failed", exc_info=True)
            return []
        finally:
            self._release()

    # ── Mask conversion ───────────────────────────────────────────────

    @staticmethod
    def _mask_to_segmentation(
        mask_data: dict[str, Any],
        frame_shape: tuple[int, ...],
        label: str,
        attributes: dict[str, Any] | None = None,
    ) -> SegmentationMask | None:
        """Convert raw SAM output to a :class:`SegmentationMask`.

        Parameters
        ----------
        mask_data:
            Dict with ``"mask"`` (H, W bool) and ``"iou"`` (float).
        frame_shape:
            Shape of the original frame ``(H, W, C)`` or ``(H, W)``.
        label:
            Semantic label for the mask.
        attributes:
            Extra metadata to attach.

        Returns
        -------
        SegmentationMask or None
            Converted mask, or ``None`` if the mask is empty.
        """
        mask = mask_data.get("mask")
        if mask is None or not mask.any():
            return None

        h, w = frame_shape[:2]
        iou_pred = mask_data.get("iou", 0.0)

        # Compute area as fraction of frame
        pixel_area = int(mask.sum())
        area_fraction = pixel_area / max(1, h * w)

        # Compute tight bounding box from mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return None

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        bbox = BBox(
            x1=float(cmin) / w,
            y1=float(rmin) / h,
            x2=float(cmax + 1) / w,
            y2=float(rmax + 1) / h,
        )

        return SegmentationMask(
            mask=mask,
            area=area_fraction,
            bbox=bbox,
            iou_prediction=iou_pred,
            label=label,
            attributes=dict(attributes) if attributes else {},
        )

    # ── Statistics ────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Return segmentation statistics for API/debug.

        Returns
        -------
        dict
            Keys: ``total_segments``, ``total_frames``,
            ``total_attention_segments``, ``avg_inference_ms``,
            ``failures``, ``variant``, ``loaded``.
        """
        loaded = (
            self._orchestrator.is_loaded(self.MODEL_NAME)
            if self._orchestrator
            else self._loaded
        )
        avg_ms = (
            self._total_inference_ms / max(1, self._total_frames)
        )
        return {
            "variant": self._variant,
            "loaded": loaded,
            "total_segments": self._total_segments,
            "total_frames": self._total_frames,
            "total_attention_segments": self._total_attention_segments,
            "avg_inference_ms": round(avg_ms, 2),
            "failures": self._failures,
        }

    # ── Cleanup ───────────────────────────────────────────────────────

    def close(self) -> None:
        """Release resources in standalone mode.

        When using the orchestrator, model lifecycle is managed
        externally and this method is a no-op.
        """
        if self._orchestrator is None and self._model is not None:
            self._unload_model(self._model)
            self._model = None
            self._loaded = False
            logger.info("SAMSegmenter closed (standalone)")
