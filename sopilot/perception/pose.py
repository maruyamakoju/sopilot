"""Pose estimation and PPE (Personal Protective Equipment) safety detection.

Uses YOLOv8s-pose to detect 17 COCO keypoints per person and infers
helmet / safety-vest compliance from color analysis of head and torso
sub-images.

All processing is CPU-only.  The ultralytics library is lazy-loaded on
first use so that importing this module has no heavy side effects.

Example usage::

    estimator = PoseEstimator()
    results = estimator.estimate(frame)
    for r in results:
        if not r.ppe.has_helmet:
            print("Missing helmet, conf=", r.ppe.helmet_confidence)
    estimator.close()
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from sopilot.perception.types import BBox, PPEStatus, PoseKeypoint, PoseResult

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────

# Minimum pixel dimension of a sub-region to attempt color analysis on.
_MIN_REGION_PX = 4


class PoseEstimator:
    """Pose-based PPE safety detector using YOLOv8s-pose (CPU-only).

    Detects 17 COCO keypoints per person, derives head and torso bounding
    boxes from visible keypoints, and runs HSV color analysis to infer
    whether each person is wearing a helmet and/or a safety vest.

    The ultralytics YOLO model is lazy-loaded on the first call to
    :meth:`estimate` so construction is lightweight.

    Parameters
    ----------
    model_name:
        Path or model name passed to ``ultralytics.YOLO()``.
        Defaults to ``"yolov8s-pose.pt"``.
    confidence_threshold:
        Minimum person detection confidence (0–1).  Persons below this
        threshold are discarded.
    keypoint_confidence:
        Minimum keypoint visibility confidence for a keypoint to be used
        when computing head/torso regions.  Keypoints are still stored in
        the output regardless of visibility.
    """

    DEFAULT_MODEL: str = "yolov8s-pose.pt"

    COCO_KEYPOINTS: list[str] = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    # Indices of COCO keypoints used for each body region
    HEAD_KP_INDICES = [0, 1, 2, 3, 4]    # nose, eyes, ears
    SHOULDER_KP_INDICES = [5, 6]           # left_shoulder, right_shoulder
    HIP_KP_INDICES = [11, 12]             # left_hip, right_hip

    def __init__(
        self,
        model_name: str | None = None,
        confidence_threshold: float = 0.4,
        keypoint_confidence: float = 0.3,
    ) -> None:
        self._model_name = model_name or self.DEFAULT_MODEL
        self._confidence_threshold = confidence_threshold
        self._keypoint_confidence = keypoint_confidence
        self._model: Any = None
        self._loaded = False
        logger.info(
            "PoseEstimator created (model=%s, lazy load)", self._model_name
        )

    # ── Lazy loading ──────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load YOLOv8-pose model on first use (CPU-only)."""
        if self._loaded:
            return

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "PoseEstimator requires the `ultralytics` package. "
                "Install with:  pip install ultralytics"
            ) from exc

        logger.info("Loading YOLOv8-pose model '%s' on CPU ...", self._model_name)
        self._model = YOLO(self._model_name)
        self._loaded = True
        logger.info("YOLOv8-pose model loaded")

    # ── Main public method ────────────────────────────────────────────

    def estimate(self, frame: np.ndarray) -> list[PoseResult]:
        """Run pose estimation on a single frame.

        Parameters
        ----------
        frame:
            BGR image as a numpy array of shape ``(H, W, 3)``.

        Returns
        -------
        list[PoseResult]
            One entry per detected person with keypoints and PPE status.
            Returns an empty list if the frame is empty or no persons are found.
        """
        if frame is None or frame.size == 0:
            return []

        self._load_model()

        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return []

        try:
            results = self._model.predict(frame, verbose=False, device="cpu")
        except Exception as exc:
            logger.warning("PoseEstimator inference failed: %s", exc)
            return []

        pose_results: list[PoseResult] = []
        total_persons_detected = 0

        for r in results:
            if r.keypoints is None or r.boxes is None:
                continue

            # keypoints data: shape (N, 17, 3) — (x_px, y_px, conf)
            kp_data = r.keypoints.data
            if hasattr(kp_data, "cpu"):
                kp_data = kp_data.cpu().numpy()
            else:
                kp_data = np.asarray(kp_data)

            boxes = r.boxes
            n_persons = len(kp_data)
            total_persons_detected += n_persons

            for i in range(n_persons):
                # Person detection confidence
                box = boxes[i]
                person_conf = float(box.conf[0])
                if person_conf < self._confidence_threshold:
                    continue

                # Person bounding box (pixel → normalized)
                xyxy = box.xyxy[0]
                if hasattr(xyxy, "cpu"):
                    xyxy = xyxy.cpu().numpy()
                x1_px, y1_px, x2_px, y2_px = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                person_bbox = BBox(
                    x1=float(np.clip(x1_px / w, 0.0, 1.0)),
                    y1=float(np.clip(y1_px / h, 0.0, 1.0)),
                    x2=float(np.clip(x2_px / w, 0.0, 1.0)),
                    y2=float(np.clip(y2_px / h, 0.0, 1.0)),
                )
                if person_bbox.area < 1e-6:
                    continue

                # Extract 17 keypoints for this person
                kps_raw = kp_data[i]  # shape (17, 3)
                keypoints: list[PoseKeypoint] = []
                for kx_px, ky_px, kconf in kps_raw:
                    keypoints.append(
                        PoseKeypoint(
                            x=float(np.clip(float(kx_px) / w, 0.0, 1.0)),
                            y=float(np.clip(float(ky_px) / h, 0.0, 1.0)),
                            confidence=float(kconf),
                        )
                    )

                # Pad to 17 if missing (shouldn't happen with YOLO-pose but be safe)
                while len(keypoints) < 17:
                    keypoints.append(PoseKeypoint(x=0.0, y=0.0, confidence=0.0))

                # Derive head bbox from visible head keypoints
                head_bbox = self._compute_head_bbox(keypoints, w, h)

                # Derive torso bbox from shoulder + hip keypoints
                torso_bbox = self._compute_torso_bbox(keypoints, w, h)

                # Color-based PPE inference
                has_helmet, helmet_conf = self._infer_helmet(frame, head_bbox)
                has_vest, vest_conf = self._infer_vest(frame, torso_bbox)

                ppe = PPEStatus(
                    has_helmet=has_helmet,
                    helmet_confidence=helmet_conf,
                    has_vest=has_vest,
                    vest_confidence=vest_conf,
                )

                pose_results.append(
                    PoseResult(
                        person_bbox=person_bbox,
                        keypoints=keypoints,
                        ppe=ppe,
                        pose_confidence=person_conf,
                    )
                )

        logger.debug(
            "PoseEstimator: %d persons detected, %d results returned",
            total_persons_detected,
            len(pose_results),
        )
        return pose_results

    # ── Region computation ────────────────────────────────────────────

    def _compute_head_bbox(
        self,
        keypoints: list[PoseKeypoint],
        frame_w: int,
        frame_h: int,
    ) -> BBox | None:
        """Compute head bounding box from visible head keypoints (indices 0-4).

        Returns None if no head keypoints are visible above the confidence threshold.
        """
        visible: list[PoseKeypoint] = []
        for idx in self.HEAD_KP_INDICES:
            if idx < len(keypoints):
                kp = keypoints[idx]
                if kp.confidence >= self._keypoint_confidence:
                    visible.append(kp)

        if not visible:
            return None

        xs = [kp.x for kp in visible]
        ys = [kp.y for kp in visible]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        # Expand head region slightly upward (helmet sits above the keypoints)
        head_h = y2 - y1
        expansion_y = max(head_h * 0.5, 0.02)
        expansion_x = max((x2 - x1) * 0.3, 0.02)

        bbox = BBox(
            x1=float(np.clip(x1 - expansion_x, 0.0, 1.0)),
            y1=float(np.clip(y1 - expansion_y, 0.0, 1.0)),
            x2=float(np.clip(x2 + expansion_x, 0.0, 1.0)),
            y2=float(np.clip(y2 + expansion_y * 0.2, 0.0, 1.0)),
        )
        if bbox.area < 1e-6:
            return None
        return bbox

    def _compute_torso_bbox(
        self,
        keypoints: list[PoseKeypoint],
        frame_w: int,
        frame_h: int,
    ) -> BBox | None:
        """Compute torso bounding box from shoulder + hip keypoints.

        Returns None if fewer than 2 anchor keypoints are visible.
        """
        torso_indices = self.SHOULDER_KP_INDICES + self.HIP_KP_INDICES
        visible: list[PoseKeypoint] = []
        for idx in torso_indices:
            if idx < len(keypoints):
                kp = keypoints[idx]
                if kp.confidence >= self._keypoint_confidence:
                    visible.append(kp)

        if len(visible) < 2:
            return None

        xs = [kp.x for kp in visible]
        ys = [kp.y for kp in visible]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        # Expand torso region slightly for better coverage
        expansion = 0.02
        bbox = BBox(
            x1=float(np.clip(x1 - expansion, 0.0, 1.0)),
            y1=float(np.clip(y1 - expansion, 0.0, 1.0)),
            x2=float(np.clip(x2 + expansion, 0.0, 1.0)),
            y2=float(np.clip(y2 + expansion, 0.0, 1.0)),
        )
        if bbox.area < 1e-6:
            return None
        return bbox

    # ── Color analysis helpers ────────────────────────────────────────

    def _extract_region(
        self, frame: np.ndarray, bbox: BBox | None
    ) -> np.ndarray | None:
        """Crop a normalized bbox from the frame.  Returns None on failure."""
        if bbox is None:
            return None
        h, w = frame.shape[:2]
        x1 = int(bbox.x1 * w)
        y1 = int(bbox.y1 * h)
        x2 = int(bbox.x2 * w)
        y2 = int(bbox.y2 * h)

        # Clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < _MIN_REGION_PX or y2 - y1 < _MIN_REGION_PX:
            return None
        return frame[y1:y2, x1:x2]

    def _bgr_to_hsv(self, region: np.ndarray) -> np.ndarray:
        """Convert a BGR uint8 region to HSV using a pure-numpy approach.

        Avoids depending on cv2 to keep the module lightweight.

        Returns
        -------
        np.ndarray
            Array of shape (H, W, 3) with H=[0,179], S=[0,255], V=[0,255]
            matching OpenCV convention.
        """
        region_f = region.astype(np.float32) / 255.0
        b, g, r = region_f[..., 0], region_f[..., 1], region_f[..., 2]

        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin

        # Hue [0, 360) → scaled to [0, 179] (OpenCV convention)
        hue = np.zeros_like(cmax)
        mask_r = (cmax == r) & (delta > 0)
        mask_g = (cmax == g) & (delta > 0)
        mask_b = (cmax == b) & (delta > 0)
        hue[mask_r] = (60.0 * ((g[mask_r] - b[mask_r]) / delta[mask_r])) % 360.0
        hue[mask_g] = 60.0 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120.0
        hue[mask_b] = 60.0 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240.0
        hue_ocv = (hue / 2.0).astype(np.float32)  # [0, 179]

        # Saturation [0, 255]
        sat = np.where(cmax > 0, (delta / cmax) * 255.0, 0.0).astype(np.float32)

        # Value [0, 255]
        val = (cmax * 255.0).astype(np.float32)

        return np.stack([hue_ocv, sat, val], axis=-1)

    def _infer_helmet(
        self, frame: np.ndarray, head_bbox: BBox | None
    ) -> tuple[bool, float]:
        """Infer helmet presence from head region color analysis.

        A helmet is inferred if the head sub-image contains a substantial
        proportion of:
        - Low-saturation (near-white) pixels: S < 60  (white helmet)
        - Yellow-orange hue pixels: Hue in [15, 45] with S > 60 (yellow/orange helmet)

        The combined ratio of qualifying pixels is used as confidence.
        Threshold: >20% qualifying pixels → helmet detected.

        Parameters
        ----------
        frame:
            BGR frame (H, W, 3).
        head_bbox:
            Normalized bbox of the head region, or None.

        Returns
        -------
        tuple[bool, float]
            (has_helmet, confidence)
        """
        region = self._extract_region(frame, head_bbox)
        if region is None:
            return False, 0.0

        hsv = self._bgr_to_hsv(region)
        h_ch = hsv[..., 0]   # hue [0, 179]
        s_ch = hsv[..., 1]   # saturation [0, 255]
        v_ch = hsv[..., 2]   # value [0, 255]

        total_pixels = region.shape[0] * region.shape[1]
        if total_pixels == 0:
            return False, 0.0

        # White / near-white pixels: low saturation, high brightness
        white_mask = (s_ch < 60) & (v_ch > 100)

        # Yellow-orange pixels: hue 15–45 (≈ 30–90 in [0,360]), decent saturation
        yellow_mask = (h_ch >= 15) & (h_ch <= 45) & (s_ch > 60)

        # Hard-hat orange: slightly wider range
        orange_mask = (h_ch >= 5) & (h_ch < 15) & (s_ch > 80)

        qualifying = np.sum(white_mask | yellow_mask | orange_mask)
        ratio = float(qualifying) / float(total_pixels)

        threshold = 0.20
        has_helmet = ratio > threshold
        confidence = float(np.clip(ratio / threshold, 0.0, 1.0))
        return has_helmet, round(confidence, 4)

    def _infer_vest(
        self, frame: np.ndarray, torso_bbox: BBox | None
    ) -> tuple[bool, float]:
        """Infer safety vest presence from torso region color analysis.

        A safety vest is inferred if the torso sub-image contains a substantial
        proportion of high-saturation yellow / orange / lime-green pixels:
        - Hue range 20–85 (yellow through green), Saturation > 100

        Threshold: >15% qualifying pixels → vest detected.

        Parameters
        ----------
        frame:
            BGR frame (H, W, 3).
        torso_bbox:
            Normalized bbox of the torso region, or None.

        Returns
        -------
        tuple[bool, float]
            (has_vest, confidence)
        """
        region = self._extract_region(frame, torso_bbox)
        if region is None:
            return False, 0.0

        hsv = self._bgr_to_hsv(region)
        h_ch = hsv[..., 0]   # hue [0, 179]
        s_ch = hsv[..., 1]   # saturation [0, 255]

        total_pixels = region.shape[0] * region.shape[1]
        if total_pixels == 0:
            return False, 0.0

        # High-saturation yellow / orange / green — safety vest colors
        vest_mask = (h_ch >= 20) & (h_ch <= 85) & (s_ch > 100)

        qualifying = np.sum(vest_mask)
        ratio = float(qualifying) / float(total_pixels)

        threshold = 0.15
        has_vest = ratio > threshold
        confidence = float(np.clip(ratio / threshold, 0.0, 1.0))
        return has_vest, round(confidence, 4)

    # ── Cleanup ───────────────────────────────────────────────────────

    def close(self) -> None:
        """Release model resources."""
        self._model = None
        self._loaded = False
        logger.info("PoseEstimator resources released")
