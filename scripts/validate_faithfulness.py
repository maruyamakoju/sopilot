"""Validate RAG faithfulness - ensure Video-LLM only sees retrieved clips.

This script tests that the Video-LLM answers based on the retrieved clip ONLY,
not the full video. This is critical for RAG integrity.

Test method:
1. Create a video with distinct scenes (Red, Blue, Green, Yellow)
2. Ask "What color is shown?" twice:
   - Once with clip containing Blue scene
   - Once with clip containing Red scene
3. Verify answers are different and match the actual clip content

Expected behavior:
- Clip [2-4 sec] (Red scene) → Answer mentions "red"
- Clip [6-8 sec] (Blue scene) → Answer mentions "blue"

Failure mode (if faithfulness broken):
- Both clips → Same answer (model sees full video)

Usage:
    python scripts/validate_faithfulness.py --device cuda --llm-model qwen2.5-vl-7b
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def create_multicolor_video(output_path: Path, duration: int = 10, fps: int = 30) -> None:
    """Create a video with distinct colored scenes for testing.

    Args:
        output_path: Path to save video
        duration: Duration in seconds
        fps: Frame rate

    Scenes:
        0-2 sec: Black (intro)
        2-4 sec: Red scene
        4-6 sec: Green scene
        6-8 sec: Blue scene
        8-10 sec: Yellow scene
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (640, 480))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    total_frames = duration * fps

    scenes = [
        (0.0, 0.2, (0, 0, 0), "Black (Intro)"),  # 0-2 sec
        (0.2, 0.4, (0, 0, 255), "Red Scene"),  # 2-4 sec
        (0.4, 0.6, (0, 255, 0), "Green Scene"),  # 4-6 sec
        (0.6, 0.8, (255, 0, 0), "Blue Scene"),  # 6-8 sec
        (0.8, 1.0, (0, 255, 255), "Yellow Scene"),  # 8-10 sec
    ]

    logger.info("Creating multicolor test video...")

    for frame_idx in range(total_frames):
        progress = frame_idx / total_frames

        # Determine current scene
        current_scene = scenes[0]
        for scene in scenes:
            if scene[0] <= progress < scene[1]:
                current_scene = scene
                break

        color = current_scene[2]
        name = current_scene[3]

        # Create frame
        frame = np.full((480, 640, 3), color, dtype=np.uint8)

        # Add text overlay
        time_text = f"t={frame_idx / fps:.2f}s"
        cv2.putText(frame, name, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(frame, time_text, (200, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        writer.write(frame)

    writer.release()
    logger.info("✅ Test video created: %s", output_path)


def test_faithfulness(video_path: Path, llm_service, device: str) -> dict:
    """Test RAG faithfulness with two different clips.

    Args:
        video_path: Path to test video
        llm_service: VideoLLMService instance
        device: Device for inference

    Returns:
        Dict with test results
    """
    question = "What color is shown in this video clip?"

    # Test 1: Red scene (2-4 sec)
    logger.info("")
    logger.info("=" * 60)
    logger.info("Test 1: Red scene [2.0-4.0 sec]")
    logger.info("=" * 60)

    result_red = llm_service.answer_question(
        video_path,
        question,
        start_sec=2.0,
        end_sec=4.0,
        enable_cot=False,
    )

    logger.info("Question: %s", question)
    logger.info("Answer: %s", result_red.answer)

    # Test 2: Blue scene (6-8 sec)
    logger.info("")
    logger.info("=" * 60)
    logger.info("Test 2: Blue scene [6.0-8.0 sec]")
    logger.info("=" * 60)

    result_blue = llm_service.answer_question(
        video_path,
        question,
        start_sec=6.0,
        end_sec=8.0,
        enable_cot=False,
    )

    logger.info("Question: %s", question)
    logger.info("Answer: %s", result_blue.answer)

    # Validate faithfulness
    logger.info("")
    logger.info("=" * 60)
    logger.info("Faithfulness Validation")
    logger.info("=" * 60)

    # Check if answers are different
    answers_different = result_red.answer != result_blue.answer

    # Check if answers mention expected colors (case-insensitive)
    red_mentions_red = "red" in result_red.answer.lower()
    blue_mentions_blue = "blue" in result_blue.answer.lower()

    # Check if there's cross-contamination (wrong color mentioned)
    red_mentions_blue = "blue" in result_red.answer.lower()
    blue_mentions_red = "red" in result_blue.answer.lower()

    results = {
        "answers_different": answers_different,
        "red_correct": red_mentions_red and not red_mentions_blue,
        "blue_correct": blue_mentions_blue and not blue_mentions_red,
        "red_answer": result_red.answer,
        "blue_answer": result_blue.answer,
    }

    logger.info("✅ Answers are different: %s", answers_different)
    logger.info("✅ Red clip mentions 'red': %s", red_mentions_red)
    logger.info("✅ Blue clip mentions 'blue': %s", blue_mentions_blue)

    if red_mentions_blue:
        logger.warning("⚠️  Red clip answer mentions 'blue' (cross-contamination)")

    if blue_mentions_red:
        logger.warning("⚠️  Blue clip answer mentions 'red' (cross-contamination)")

    # Overall faithfulness score
    faithfulness_ok = (
        answers_different
        and red_mentions_red
        and blue_mentions_blue
        and not red_mentions_blue
        and not blue_mentions_red
    )

    logger.info("")
    if faithfulness_ok:
        logger.info("🎉 FAITHFULNESS TEST PASSED")
        logger.info("   Model answers are based on retrieved clips, not full video")
    else:
        logger.error("❌ FAITHFULNESS TEST FAILED")
        logger.error("   Model may be seeing full video instead of clips")
        if not answers_different:
            logger.error("   → Answers are identical (should be different)")
        if not red_mentions_red:
            logger.error("   → Red clip doesn't mention 'red'")
        if not blue_mentions_blue:
            logger.error("   → Blue clip doesn't mention 'blue'")
        if red_mentions_blue or blue_mentions_red:
            logger.error("   → Cross-contamination detected")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate RAG faithfulness")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="mock",
        choices=["mock", "qwen2.5-vl-7b"],
        help="Video-LLM model",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="faithfulness_test_video.mp4",
        help="Path to test video (will create if doesn't exist)",
    )
    args = parser.parse_args()

    video_path = Path(args.video)

    # Create test video if it doesn't exist
    if not video_path.exists():
        logger.info("Test video not found, creating...")
        create_multicolor_video(video_path, duration=10, fps=30)

    # Initialize Video-LLM service
    from sopilot.video_llm_service import VideoLLMService, get_default_config

    llm_config = get_default_config(args.llm_model)
    llm_config.device = args.device

    logger.info("=" * 60)
    logger.info("Initializing Video-LLM: %s", args.llm_model)
    logger.info("=" * 60)

    llm_service = VideoLLMService(llm_config)

    if args.llm_model != "mock" and llm_service._model is None:
        logger.error("❌ Model failed to load!")
        logger.error("   Install dependencies: pip install -e '.[vigil]'")
        return 1

    if llm_service._model is not None:
        logger.info("✅ Model loaded: %s", type(llm_service._model).__name__)

    # Run faithfulness test
    results = test_faithfulness(video_path, llm_service, args.device)

    # Return exit code
    faithfulness_ok = results["red_correct"] and results["blue_correct"]
    return 0 if faithfulness_ok else 1


if __name__ == "__main__":
    sys.exit(main())
