"""Monocular depth estimation from detection bbox geometry.

Heuristic depth cues (no ML model required):
1. Bbox area: larger area → closer → lower depth value
2. Y-position: lower center in frame → closer (perspective projection)
3. Known reference heights: pinhole model for metric depth estimates

All estimates are relative [0.0=very near, 1.0=very far] unless
camera_height_m and focal_length_px are provided for metric estimates.
"""
from __future__ import annotations
import math
import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class DepthEstimate:
    entity_id: int
    label: str
    bbox: list[float]  # [x, y, w, h] normalized
    depth_relative: float  # 0.0=very near, 1.0=very far
    depth_metric_m: float | None  # metric if camera params known
    confidence: float  # estimation confidence [0,1]

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "bbox": self.bbox,
            "depth_relative": round(self.depth_relative, 4),
            "depth_metric_m": round(self.depth_metric_m, 2) if self.depth_metric_m is not None else None,
            "confidence": round(self.confidence, 4),
        }


class MonocularDepthEstimator:
    """Heuristic monocular depth from bbox geometry.

    Parameters
    ----------
    camera_height_m : float
        Camera mounting height above floor (used for metric depth).
    tilt_angle_deg : float
        Camera downward tilt angle in degrees (0 = horizontal).
    focal_length_px : float | None
        Camera focal length in pixels. If None, metric depth is unavailable.
    frame_width_px : int
        Frame width in pixels (for focal length estimation).
    frame_height_px : int
        Frame height in pixels.
    area_weight : float
        Weight of bbox-area cue in [0,1] blend.
    y_weight : float
        Weight of y-position cue (1 - area_weight).
    """

    # Default known object heights in meters (for metric depth estimation)
    KNOWN_HEIGHTS_M: dict[str, float] = {
        "person": 1.7,
        "car": 1.5,
        "truck": 2.5,
        "bicycle": 1.0,
        "motorcycle": 1.1,
        "worker": 1.7,
    }

    def __init__(
        self,
        camera_height_m: float = 2.5,
        tilt_angle_deg: float = 15.0,
        focal_length_px: float | None = None,
        frame_width_px: int = 1280,
        frame_height_px: int = 720,
        area_weight: float = 0.6,
        y_weight: float = 0.4,
    ) -> None:
        self._camera_height_m = camera_height_m
        self._tilt_rad = math.radians(tilt_angle_deg)
        self._focal_px = focal_length_px
        self._frame_w = frame_width_px
        self._frame_h = frame_height_px
        self._area_weight = max(0.0, min(1.0, area_weight))
        self._y_weight = 1.0 - self._area_weight
        self._lock = threading.Lock()

    def estimate(self, detections: list) -> list[DepthEstimate]:
        """Estimate depth for each detection.

        Each item must have .entity_id, .label, .bbox (BBox with x,y,w,h or list).
        """
        results = []
        with self._lock:
            for det in detections:
                bbox = self._get_bbox(det)
                d_area = self._depth_from_area(bbox)
                d_y = self._depth_from_y(bbox)
                depth_rel = self._area_weight * d_area + self._y_weight * d_y
                depth_rel = max(0.0, min(1.0, depth_rel))

                label = getattr(det, "label", "")
                depth_metric = self._metric_depth(label, bbox)

                # confidence: higher for larger objects (more signal)
                area = bbox[2] * bbox[3]
                confidence = min(1.0, math.sqrt(area) * 3.0)

                results.append(DepthEstimate(
                    entity_id=getattr(det, "entity_id", 0),
                    label=label,
                    bbox=list(bbox),
                    depth_relative=round(depth_rel, 4),
                    depth_metric_m=depth_metric,
                    confidence=round(confidence, 4),
                ))
        return results

    def _get_bbox(self, det) -> list[float]:
        """Extract [x,y,w,h] from detection."""
        bbox = getattr(det, "bbox", None)
        if bbox is None:
            return [0.5, 0.5, 0.1, 0.1]
        if hasattr(bbox, "x"):
            return [bbox.x, bbox.y, bbox.w, bbox.h]
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return list(bbox)
        return [0.5, 0.5, 0.1, 0.1]

    def _depth_from_area(self, bbox: list[float]) -> float:
        """Larger bbox area → object is closer → lower depth."""
        area = bbox[2] * bbox[3]
        area = max(1e-6, min(1.0, area))
        return max(0.0, 1.0 - math.sqrt(area) * 2.5)

    def _depth_from_y(self, bbox: list[float]) -> float:
        """Lower center in frame → closer → lower depth (perspective)."""
        cy = bbox[1] + bbox[3] / 2.0
        return max(0.0, min(1.0, 1.0 - cy))

    def _metric_depth(self, label: str, bbox: list[float]) -> float | None:
        """Pinhole model: known_height / (bbox_h_px / focal_px) = depth."""
        if self._focal_px is None:
            return None
        known_h = self.KNOWN_HEIGHTS_M.get(label.lower())
        if known_h is None:
            return None
        bbox_h_px = bbox[3] * self._frame_h
        if bbox_h_px < 1.0:
            return None
        depth_m = (known_h * self._focal_px) / bbox_h_px
        return round(max(0.1, depth_m), 2)

    def get_config(self) -> dict:
        return {
            "camera_height_m": self._camera_height_m,
            "tilt_angle_deg": math.degrees(self._tilt_rad),
            "focal_length_px": self._focal_px,
            "frame_width_px": self._frame_w,
            "frame_height_px": self._frame_h,
            "area_weight": self._area_weight,
            "y_weight": self._y_weight,
        }
