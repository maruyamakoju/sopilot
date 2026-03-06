"""ONNX-based YOLO-World object detector.

Drop-in replacement for ``YOLOWorldDetector`` that uses ``onnxruntime``
instead of ultralytics.  Controlled via ``PerceptionConfig.use_onnx``.

Interface is identical to ``YOLOWorldDetector``:
- ``detect(frame) → List[Detection]``
- ``set_classes(class_names)``

Requires: ``pip install sopilot[onnx]`` (onnxruntime>=1.18.0)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from sopilot.perception.types import BBox, Detection

logger = logging.getLogger(__name__)

# Default COCO/YOLO-World class names (can be overridden via set_classes)
_DEFAULT_CLASSES = [
    "person", "helmet", "hard hat", "safety vest", "gloves",
    "goggles", "safety harness", "mask", "fire extinguisher",
    "bicycle", "car", "motorcycle", "bus", "truck",
    "forklift", "pallet", "box", "barrel",
]


class OnnxYOLOWorldDetector:
    """ONNX-backed YOLO-World detector.

    Loads an ``.onnx`` file exported via ``scripts/export_onnx.py`` and runs
    inference with ``onnxruntime``.  Applies NMS post-processing and returns
    ``List[Detection]`` with normalised bounding boxes.
    """

    def __init__(
        self,
        model_path: str,
        *,
        confidence_threshold: float = 0.05,
        nms_threshold: float = 0.5,
        class_names: list[str] | None = None,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._nms_threshold = nms_threshold
        self._class_names: list[str] = class_names or list(_DEFAULT_CLASSES)
        self._session: Any = None
        self._input_name: str = "images"
        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for ONNX inference. "
                "Install with: pip install sopilot[onnx]"
            ) from exc

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(model_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        logger.info("OnnxYOLOWorldDetector loaded: %s", model_path)

    def set_classes(self, class_names: list[str]) -> None:
        """Override the class name list (for open-vocabulary detection)."""
        self._class_names = list(class_names)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference and return detections with normalised coords."""
        if self._session is None:
            return []

        h, w = frame.shape[:2]
        blob = self._preprocess(frame)

        try:
            outputs = self._session.run(None, {self._input_name: blob})
        except Exception:
            logger.exception("ONNX inference failed")
            return []

        return self._postprocess(outputs, img_h=h, img_w=w)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray, target_size: int = 640) -> np.ndarray:
        """Resize + normalise frame to CHW float32 blob."""
        import cv2  # type: ignore[import-untyped]

        resized = cv2.resize(frame, (target_size, target_size))
        blob = resized[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, normalise
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # HWC → 1CHW
        return blob

    def _postprocess(
        self,
        outputs: list[np.ndarray],
        *,
        img_h: int,
        img_w: int,
        target_size: int = 640,
    ) -> list[Detection]:
        """Convert raw ONNX output to List[Detection] with NMS."""
        if not outputs or outputs[0] is None:
            return []

        raw = outputs[0]  # shape: (1, num_detections, 4+1+num_classes) or (1, 4+num_classes, N)

        if raw.ndim == 2:
            raw = raw[np.newaxis]  # add batch dim if missing

        raw = raw[0]  # (attrs, N) or (N, attrs)

        # Normalise to (N, attrs): transpose if attrs dimension looks smaller than N dimension.
        # Standard ONNX YOLOv8/YOLOWorld output is (attrs, N) where attrs = 4 + num_classes.
        # attrs is typically 5–85 and N is 8400+, so raw.shape[0] < raw.shape[1] means (attrs, N).
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T  # now (N, attrs)

        # raw is now (N, attrs)
        boxes_xywh = raw[:, :4]     # (N, 4) in pixel space relative to target_size
        class_scores = raw[:, 4:]   # (N, num_classes)

        if class_scores.size == 0:
            return []

        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores.max(axis=1)

        mask = confidences >= self._confidence_threshold
        if not mask.any():
            return []

        boxes_xywh = boxes_xywh[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        # Convert xywh (center) → xyxy (corners), normalised
        cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = np.clip((cx - bw / 2) / target_size, 0.0, 1.0)
        y1 = np.clip((cy - bh / 2) / target_size, 0.0, 1.0)
        x2 = np.clip((cx + bw / 2) / target_size, 0.0, 1.0)
        y2 = np.clip((cy + bh / 2) / target_size, 0.0, 1.0)

        # NMS (per-class)
        import cv2  # type: ignore[import-untyped]

        pixel_boxes = np.stack([x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h], axis=1).tolist()
        keep_indices = cv2.dnn.NMSBoxes(
            pixel_boxes,
            confidences.tolist(),
            self._confidence_threshold,
            self._nms_threshold,
        )
        if len(keep_indices) == 0:
            return []

        keep = keep_indices.flatten() if hasattr(keep_indices, "flatten") else keep_indices
        detections: list[Detection] = []
        for i in keep:
            label_idx = int(class_ids[i])
            label = (
                self._class_names[label_idx]
                if label_idx < len(self._class_names)
                else f"class_{label_idx}"
            )
            detections.append(
                Detection(
                    label=label,
                    confidence=float(confidences[i]),
                    bbox=BBox(
                        x1=float(x1[i]),
                        y1=float(y1[i]),
                        x2=float(x2[i]),
                        y2=float(y2[i]),
                    ),
                )
            )
        return detections
