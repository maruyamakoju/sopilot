"""Object Detection and Proximity Analysis for Insurance Video Review

Detects nearby objects (vehicles, pedestrians, cyclists) and computes danger scores:
- Proximity score: Large bounding boxes = close objects = dangerous
- Center distance: Objects in center of frame = front of vehicle = dangerous
- Object type weighting: Pedestrians/cyclists more important than cars

Uses YOLOv8n for fast object detection.

Design:
- Process every N frames (not every frame - too slow)
- Return normalized danger scores [0.0, 1.0]
- Handle GPU/CPU automatically
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProximityConfig:
    """Configuration for proximity analysis"""

    # YOLOv8 model
    model_name: str = "yolov8n.pt"  # Nano model (fastest)
    confidence_threshold: float = 0.25  # Detection confidence threshold
    device: str = "auto"  # "auto", "cuda", "cpu"

    # Object classes to detect (COCO dataset indices)
    # person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7
    target_classes: list[int] = None

    # Object type weighting (higher weight = more dangerous)
    class_weights: dict[int, float] = None

    # Frame sampling
    frame_skip: int = 5  # Process every 5 frames

    # Danger computation
    proximity_percentile: float = 95.0  # Percentile for normalization

    def __post_init__(self):
        if self.target_classes is None:
            # Default: person, bicycle, car, motorcycle, bus, truck
            self.target_classes = [0, 1, 2, 3, 5, 7]

        if self.class_weights is None:
            # Default weights: pedestrians/cyclists more dangerous than vehicles
            self.class_weights = {
                0: 1.5,  # person (high priority)
                1: 1.3,  # bicycle (high priority)
                2: 1.0,  # car
                3: 1.2,  # motorcycle
                5: 1.0,  # bus
                7: 1.0,  # truck
            }


class ProximityAnalyzer:
    """
    Analyze object proximity for danger signals.

    Extracts:
    1. Proximity score - bbox area (large = close = dangerous)
    2. Center distance - distance from center (center = front = dangerous)
    3. Object type weighting - pedestrians/cyclists prioritized
    """

    def __init__(self, config: Optional[ProximityConfig] = None):
        self.config = config or ProximityConfig()
        self.model = None

    def _load_model(self):
        """Lazy-load YOLOv8 model"""
        if self.model is not None:
            return

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "ultralytics not installed. Install with: pip install ultralytics"
            ) from e

        logger.info(f"Loading YOLOv8 model: {self.config.model_name}")

        # Determine device
        if self.config.device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device

        logger.info(f"Using device: {device}")

        self.model = YOLO(self.config.model_name)
        self.model.to(device)

        logger.info("YOLOv8 model loaded successfully")

    def analyze(self, video_path: Path | str) -> np.ndarray:
        """
        Analyze object proximity and return danger scores per second.

        Args:
            video_path: Path to video file

        Returns:
            np.ndarray: Danger scores per second [0.0, 1.0], shape (n_seconds,)

        Raises:
            RuntimeError: If video file cannot be read or model fails to load
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise RuntimeError(f"Video file not found: {video_path}")

        logger.info(f"Analyzing proximity: {video_path.name}")

        # Load model
        self._load_model()

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            cap.release()
            raise RuntimeError(f"Invalid FPS: {fps}")

        duration_sec = frame_count / fps
        n_seconds = int(np.ceil(duration_sec))

        logger.info(f"Video: {frame_count} frames, {fps:.1f} fps, {duration_sec:.1f}s, {width}x{height}")

        # Process video
        proximity_per_frame = []
        frame_timestamps = []

        frame_idx = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Skip frames for performance
            if frame_idx % self.config.frame_skip != 0:
                frame_idx += 1
                continue

            # Run object detection
            try:
                results = self.model.predict(
                    frame,
                    conf=self.config.confidence_threshold,
                    classes=self.config.target_classes,
                    verbose=False,
                )

                # Compute proximity danger score for this frame
                danger_score = self._compute_frame_danger(results[0], width, height)
                proximity_per_frame.append(danger_score)

                timestamp_sec = frame_idx / fps
                frame_timestamps.append(timestamp_sec)

            except Exception as e:
                logger.warning(f"Detection failed at frame {frame_idx}: {e}")
                proximity_per_frame.append(0.0)
                timestamp_sec = frame_idx / fps
                frame_timestamps.append(timestamp_sec)

            frame_idx += 1

            # Log progress every 100 frames
            if frame_idx % 100 == 0:
                logger.debug(f"Processed {frame_idx}/{frame_count} frames")

        cap.release()

        if len(proximity_per_frame) == 0:
            logger.warning(f"No detections computed for {video_path.name}")
            return np.zeros(n_seconds, dtype=np.float32)

        # Convert to numpy arrays
        proximity_arr = np.array(proximity_per_frame, dtype=np.float32)
        timestamps_arr = np.array(frame_timestamps, dtype=np.float32)

        # Aggregate to per-second scores
        proximity_per_sec = self._aggregate_to_seconds(timestamps_arr, proximity_arr, n_seconds)

        # Normalize
        danger_scores = self._normalize(proximity_per_sec, self.config.proximity_percentile)

        logger.info(
            f"Proximity analysis complete: {n_seconds}s, "
            f"mean_danger={danger_scores.mean():.3f}, "
            f"max_danger={danger_scores.max():.3f}, "
            f"peaks={np.sum(danger_scores > 0.5)}"
        )

        return danger_scores

    def _compute_frame_danger(self, result, frame_width: int, frame_height: int) -> float:
        """
        Compute danger score for a single frame based on detected objects.

        Args:
            result: YOLOv8 detection result
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels

        Returns:
            Danger score [0.0, infinity)
        """
        if result.boxes is None or len(result.boxes) == 0:
            return 0.0

        frame_area = frame_width * frame_height
        center_x = frame_width / 2.0
        center_y = frame_height / 2.0

        total_danger = 0.0

        for box in result.boxes:
            # Extract bbox coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())

            # Compute bbox area (normalized)
            bbox_area = (x2 - x1) * (y2 - y1)
            area_ratio = bbox_area / frame_area

            # Compute distance from center (normalized)
            bbox_center_x = (x1 + x2) / 2.0
            bbox_center_y = (y1 + y2) / 2.0
            dist_x = abs(bbox_center_x - center_x) / frame_width
            dist_y = abs(bbox_center_y - center_y) / frame_height
            center_dist = np.sqrt(dist_x**2 + dist_y**2)

            # Proximity score: large bbox = close = dangerous
            proximity_score = area_ratio

            # Center score: center of frame = front = dangerous
            # Invert distance: closer to center = higher score
            center_score = 1.0 - center_dist

            # Combine proximity and center
            object_danger = proximity_score * 0.7 + center_score * 0.3

            # Apply class weighting
            class_weight = self.config.class_weights.get(cls, 1.0)
            object_danger *= class_weight

            # Apply confidence weighting
            object_danger *= conf

            total_danger += object_danger

        return total_danger

    def _aggregate_to_seconds(self, timestamps: np.ndarray, values: np.ndarray, n_seconds: int) -> np.ndarray:
        """
        Aggregate frame-level values to per-second values using max pooling.

        Args:
            timestamps: Frame timestamps (seconds)
            values: Frame-level values
            n_seconds: Total number of seconds

        Returns:
            Per-second aggregated values
        """
        per_sec = np.zeros(n_seconds, dtype=np.float32)

        for ts, val in zip(timestamps, values):
            sec_idx = int(np.floor(ts))
            if 0 <= sec_idx < n_seconds:
                # Use max pooling (keep highest danger within each second)
                per_sec[sec_idx] = max(per_sec[sec_idx], val)

        return per_sec

    def _normalize(self, values: np.ndarray, percentile: float) -> np.ndarray:
        """
        Normalize values to [0.0, 1.0] using percentile-based scaling.

        Args:
            values: Raw values
            percentile: Percentile for normalization (avoid outlier saturation)

        Returns:
            Normalized values
        """
        if len(values) == 0 or values.max() == 0:
            return values

        # Use percentile to avoid outlier saturation
        percentile_val = np.percentile(values, percentile)

        if percentile_val > 0:
            normalized = values / percentile_val
            normalized = np.clip(normalized, 0.0, 1.0)
            return normalized
        else:
            return values


def visualize_detections(
    video_path: Path | str,
    output_path: Optional[Path | str] = None,
    config: Optional[ProximityConfig] = None,
) -> Optional[Path]:
    """
    Generate object detection visualization video for debugging.

    Args:
        video_path: Input video path
        output_path: Output video path (optional, auto-generated if None)
        config: ProximityConfig (optional)

    Returns:
        Path to output video, or None if failed
    """
    video_path = Path(video_path)

    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_detections.mp4"
    else:
        output_path = Path(output_path)

    logger.info(f"Generating detection visualization: {video_path.name} -> {output_path.name}")

    analyzer = ProximityAnalyzer(config)
    analyzer._load_model()

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps / analyzer.config.frame_skip, (width, height))

    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % analyzer.config.frame_skip != 0:
            frame_idx += 1
            continue

        # Run detection
        try:
            results = analyzer.model.predict(
                frame,
                conf=analyzer.config.confidence_threshold,
                classes=analyzer.config.target_classes,
                verbose=False,
            )

            # Draw bounding boxes
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        except Exception as e:
            logger.warning(f"Detection failed at frame {frame_idx}: {e}")
            out.write(frame)

        frame_idx += 1

    cap.release()
    out.release()

    logger.info(f"Detection visualization saved: {output_path}")
    return output_path
