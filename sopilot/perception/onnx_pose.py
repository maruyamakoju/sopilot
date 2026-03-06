"""ONNX-based YOLOv8-pose estimator.

Drop-in replacement for ``PoseEstimator`` that uses ``onnxruntime``
instead of ultralytics.  Returns ``List[PoseResult]`` with 17 COCO keypoints.

Requires: ``pip install sopilot[onnx]``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sopilot.perception.types import BBox

logger = logging.getLogger(__name__)

# 17 COCO keypoint names (index → name)
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Indices relevant for PPE / safety heuristics
_HELMET_KP = [0, 1, 2, 3, 4]    # nose + eyes + ears → head region
_VEST_KP   = [5, 6, 11, 12]     # shoulders + hips → torso region


@dataclass
class OnnxPoseResult:
    """Single person's pose result from ONNX inference."""
    bbox: BBox
    confidence: float
    keypoints: list[tuple[float, float, float]]  # (x_norm, y_norm, conf) per keypoint
    has_helmet: bool = False
    has_vest: bool = False

    def get_keypoint(self, name: str) -> tuple[float, float, float] | None:
        try:
            idx = COCO_KEYPOINTS.index(name)
            return self.keypoints[idx]
        except (ValueError, IndexError):
            return None


class OnnxPoseEstimator:
    """ONNX-backed YOLOv8-pose estimator.

    Loads a ``yolov8s-pose.onnx`` file and returns a list of
    ``OnnxPoseResult`` objects for each detected person.
    """

    def __init__(
        self,
        model_path: str,
        *,
        confidence_threshold: float = 0.4,
        keypoint_confidence: float = 0.3,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._keypoint_confidence = keypoint_confidence
        self._session: Any = None
        self._input_name: str = "images"
        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required. Install with: pip install sopilot[onnx]"
            ) from exc

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(model_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        logger.info("OnnxPoseEstimator loaded: %s", model_path)

    def estimate(self, frame: np.ndarray) -> list[OnnxPoseResult]:
        """Run pose inference; return one OnnxPoseResult per detected person."""
        if self._session is None:
            return []

        h, w = frame.shape[:2]
        blob = self._preprocess(frame)

        try:
            outputs = self._session.run(None, {self._input_name: blob})
        except Exception:
            logger.exception("ONNX pose inference failed")
            return []

        return self._postprocess(outputs, img_h=h, img_w=w)

    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray, target_size: int = 640) -> np.ndarray:
        import cv2  # type: ignore[import-untyped]

        resized = cv2.resize(frame, (target_size, target_size))
        blob = resized[:, :, ::-1].astype(np.float32) / 255.0
        return blob.transpose(2, 0, 1)[np.newaxis]

    def _postprocess(
        self,
        outputs: list[np.ndarray],
        *,
        img_h: int,
        img_w: int,
        target_size: int = 640,
    ) -> list[OnnxPoseResult]:
        """YOLOv8-pose output: (1, 56, N) where 56 = 4(box)+1(conf)+17*3(kp)."""
        if not outputs:
            return []

        raw = outputs[0]  # (1, 56, N) or (1, N, 56)
        if raw.ndim == 2:
            raw = raw[np.newaxis]  # add batch dim if missing
        raw = raw[0]  # (56, N) or (N, 56)

        # Normalise to (N, 56): transpose if attrs (56) is the first dimension.
        # raw.shape[0] < raw.shape[1] means (attrs, N) where attrs < N → transpose.
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T  # now (N, 56)

        # At this point raw may be either (N, 56) or still (56, N) when N > 56.
        # Determine orientation: column 4 should be person confidence in range [0,1].
        # We check both orientations and pick the one that makes axis 1 have 56 columns.
        if raw.ndim == 2 and raw.shape[1] != 56 and raw.shape[0] == 56:
            raw = raw.T  # force to (N, 56)

        if raw.ndim != 2 or raw.shape[1] < 5:
            return []

        person_conf = raw[:, 4]
        mask = person_conf >= self._confidence_threshold
        raw = raw[mask]

        if len(raw) == 0:
            return []

        results: list[OnnxPoseResult] = []
        for row in raw:
            # Bounding box (xywh centre format)
            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            conf = float(row[4])
            x1 = float(np.clip((cx - bw / 2) / target_size, 0.0, 1.0))
            y1 = float(np.clip((cy - bh / 2) / target_size, 0.0, 1.0))
            x2 = float(np.clip((cx + bw / 2) / target_size, 0.0, 1.0))
            y2 = float(np.clip((cy + bh / 2) / target_size, 0.0, 1.0))
            bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)

            # Keypoints: row[5:] in groups of 3 (x, y, visibility)
            kp_flat = row[5:]
            keypoints: list[tuple[float, float, float]] = []
            for k in range(17):
                base = k * 3
                if base + 2 < len(kp_flat):
                    kx = float(np.clip(kp_flat[base] / target_size, 0.0, 1.0))
                    ky = float(np.clip(kp_flat[base + 1] / target_size, 0.0, 1.0))
                    kv = float(kp_flat[base + 2])
                    keypoints.append((kx, ky, kv))
                else:
                    keypoints.append((0.0, 0.0, 0.0))

            # PPE heuristics based on keypoint positions
            has_helmet, has_vest = self._ppe_heuristics(keypoints, bbox)

            results.append(
                OnnxPoseResult(
                    bbox=bbox,
                    confidence=conf,
                    keypoints=keypoints,
                    has_helmet=has_helmet,
                    has_vest=has_vest,
                )
            )

        return results

    def _ppe_heuristics(
        self,
        keypoints: list[tuple[float, float, float]],
        bbox: BBox,
    ) -> tuple[bool, bool]:
        """Rough PPE detection based on head/torso keypoint visibility."""
        kp_conf_threshold = self._keypoint_confidence

        # Helmet: high confidence on head keypoints (nose/eyes/ears)
        head_visible = sum(
            1 for i in _HELMET_KP
            if i < len(keypoints) and keypoints[i][2] >= kp_conf_threshold
        )
        has_helmet = head_visible >= 3  # at least 3 head kp visible = head region clear

        # Safety vest: both shoulder + hip keypoints visible → torso covered
        torso_visible = sum(
            1 for i in _VEST_KP
            if i < len(keypoints) and keypoints[i][2] >= kp_conf_threshold
        )
        has_vest = torso_visible >= 3

        return has_helmet, has_vest
