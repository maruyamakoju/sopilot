"""Generate a simple test video for VIGIL-RAG E2E testing.

Creates a 10-second video with different colored scenes for testing
multi-scale chunking and retrieval.

Usage:
    python scripts/generate_test_video.py --output test_video.mp4 --duration 10
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def generate_test_video(
    output_path: Path,
    duration_sec: float = 10.0,
    fps: float = 30.0,
    width: int = 640,
    height: int = 480,
) -> None:
    """Generate a test video with different colored scenes.

    Args:
        output_path: Path to save video
        duration_sec: Video duration in seconds
        fps: Frame rate
        width: Video width
        height: Video height
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    total_frames = int(duration_sec * fps)

    # Define scenes with different colors and text
    scenes = [
        {"name": "Red Scene", "color": (0, 0, 255), "duration": 0.2},  # 20%
        {"name": "Green Scene", "color": (0, 255, 0), "duration": 0.3},  # 30%
        {"name": "Blue Scene", "color": (255, 0, 0), "duration": 0.2},  # 20%
        {"name": "Yellow Scene", "color": (0, 255, 255), "duration": 0.15},  # 15%
        {"name": "Purple Scene", "color": (255, 0, 255), "duration": 0.15},  # 15%
    ]

    logger.info("Generating %d frames at %.1f fps...", total_frames, fps)

    for frame_idx in range(total_frames):
        # Determine current scene
        progress = frame_idx / total_frames
        cumulative_duration = 0.0
        current_scene = scenes[0]

        for scene in scenes:
            cumulative_duration += scene["duration"]
            if progress < cumulative_duration:
                current_scene = scene
                break

        # Create frame with solid color
        frame = np.full((height, width, 3), current_scene["color"], dtype=np.uint8)

        # Add text overlay
        text = current_scene["name"]
        time_text = f"t={frame_idx / fps:.2f}s"

        cv2.putText(
            frame,
            text,
            (width // 4, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3,
        )
        cv2.putText(
            frame,
            time_text,
            (width // 4, height // 2 + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        writer.write(frame)

        if (frame_idx + 1) % int(fps) == 0:
            logger.info(
                "Progress: %d/%d frames (%.1f%%)", frame_idx + 1, total_frames, 100 * (frame_idx + 1) / total_frames
            )

    writer.release()
    logger.info("✅ Test video saved: %s", output_path)
    logger.info("   Duration: %.1f sec, %d frames at %.1f fps", duration_sec, total_frames, fps)


def main():
    parser = argparse.ArgumentParser(description="Generate test video for VIGIL-RAG")
    parser.add_argument(
        "--output",
        type=str,
        default="test_video.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Video duration in seconds",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frame rate",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Video width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_test_video(
        output_path,
        duration_sec=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
