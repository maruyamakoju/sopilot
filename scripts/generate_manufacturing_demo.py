#!/usr/bin/env python3
"""Generate realistic manufacturing SOP demo videos for SOPilot.

Creates synthetic but realistic-looking oil change procedure videos:
- Gold standard: Correct 10-step procedure
- Trainee deviations: Missing steps, wrong tool, no safety gear

Each step uses distinct visual patterns + on-screen text labels to simulate
real-world procedural videos with industrial settings.

Output:
    demo_videos/manufacturing/oil_change_gold.mp4
    demo_videos/manufacturing/oil_change_trainee.mp4

Usage:
    python scripts/generate_manufacturing_demo.py
    python scripts/generate_manufacturing_demo.py --steps 8 --fps 30 --duration 120
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# Oil change SOP steps (industry standard)
OIL_CHANGE_STEPS = [
    ("PARK", "Park vehicle on level surface", (255, 100, 100)),  # Red
    ("SAFETY", "Put on safety glasses and gloves", (100, 255, 100)),  # Green
    ("LIFT", "Lift vehicle with jack and stands", (100, 100, 255)),  # Blue
    ("LOCATE", "Locate oil drain plug under engine", (255, 255, 100)),  # Yellow
    ("DRAIN", "Place drain pan and remove plug", (255, 100, 255)),  # Magenta
    ("FILTER", "Remove old oil filter with wrench", (100, 255, 255)),  # Cyan
    ("INSTALL_FILTER", "Install new filter (hand-tight)", (255, 150, 100)),  # Orange
    ("REINSTALL_PLUG", "Reinstall drain plug with torque wrench", (200, 200, 200)),  # White
    ("FILL", "Add new oil through filler cap", (150, 100, 255)),  # Purple
    ("CHECK", "Check oil level with dipstick", (100, 200, 150)),  # Teal
]


def draw_tool_icon(frame: np.ndarray, tool_name: str, x: int, y: int, size: int = 60) -> None:
    """Draw a simple tool icon representation.

    Args:
        frame: Image to draw on
        tool_name: Tool identifier (wrench, jack, pan, filter, dipstick)
        x, y: Position
        size: Icon size in pixels
    """
    color = (200, 200, 200)  # Gray for tools

    if tool_name == "wrench":
        # L-shaped wrench
        cv2.rectangle(frame, (x, y), (x + size // 3, y + size), color, -1)
        cv2.rectangle(frame, (x, y), (x + size, y + size // 3), color, -1)
    elif tool_name == "jack":
        # Triangle (jack stands)
        pts = np.array([[x + size // 2, y], [x, y + size], [x + size, y + size]], np.int32)
        cv2.fillPoly(frame, [pts], color)
    elif tool_name == "pan":
        # Rectangle (drain pan)
        cv2.rectangle(frame, (x, y + size // 2), (x + size, y + size), color, -1)
    elif tool_name == "filter":
        # Cylinder (oil filter)
        cv2.rectangle(frame, (x + size // 4, y), (x + 3 * size // 4, y + size), color, -1)
        cv2.ellipse(frame, (x + size // 2, y), (size // 4, size // 8), 0, 0, 180, color, -1)
    elif tool_name == "dipstick":
        # Thin rod (dipstick)
        cv2.rectangle(frame, (x + size // 3, y), (x + 2 * size // 3, y + size), color, -1)
        cv2.circle(frame, (x + size // 2, y - 10), 10, color, -1)
    elif tool_name == "gloves":
        # Two circles (gloves)
        cv2.circle(frame, (x + size // 3, y + size // 2), size // 3, (100, 255, 100), -1)
        cv2.circle(frame, (x + 2 * size // 3, y + size // 2), size // 3, (100, 255, 100), -1)
    elif tool_name == "glasses":
        # Safety glasses
        cv2.rectangle(frame, (x, y + size // 3), (x + size, y + 2 * size // 3), (100, 200, 255), -1)
        cv2.circle(frame, (x + size // 4, y + size // 2), size // 6, (50, 100, 150), 2)
        cv2.circle(frame, (x + 3 * size // 4, y + size // 2), size // 6, (50, 100, 150), 2)


def draw_step_frame(
    width: int,
    height: int,
    step_name: str,
    step_desc: str,
    bg_color: tuple[int, int, int],
    tool: str | None = None,
    frame_idx: int = 0,
    total_frames: int = 24,
) -> np.ndarray:
    """Draw a single frame for a procedural step.

    Args:
        width, height: Frame dimensions
        step_name: Step identifier (PARK, SAFETY, etc.)
        step_desc: Step description text
        bg_color: Background color (BGR)
        tool: Optional tool icon to display
        frame_idx: Current frame index (for animation)
        total_frames: Total frames in step

    Returns:
        Frame as numpy array (H, W, 3) uint8
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Animated background gradient based on step color
    alpha = frame_idx / max(total_frames - 1, 1)
    for i in range(height):
        blend = int(255 * (1 - alpha * i / height))
        color = tuple(int(c * blend / 255) for c in bg_color)
        frame[i, :] = color

    # Step name (large title)
    title_y = 50
    cv2.putText(
        frame,
        f"STEP: {step_name}",
        (20, title_y),
        cv2.FONT_HERSHEY_DUPLEX,
        1.2,
        (255, 255, 255),
        3,
    )

    # Step description (subtitle)
    desc_y = 100
    cv2.putText(
        frame,
        step_desc,
        (20, desc_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
    )

    # Tool icon (if specified)
    if tool:
        tool_x = width - 100
        tool_y = height // 2 - 30
        draw_tool_icon(frame, tool, tool_x, tool_y, size=60)

        # Tool label
        cv2.putText(
            frame,
            tool.upper(),
            (tool_x - 20, height // 2 + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

    # Progress bar (bottom)
    bar_y = height - 20
    bar_width = int(width * 0.9)
    bar_x = (width - bar_width) // 2
    progress = frame_idx / max(total_frames - 1, 1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 10), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + 10), (0, 255, 0), -1)

    return frame


def generate_oil_change_video(
    output_path: Path,
    variant: str = "gold",
    width: int = 640,
    height: int = 480,
    fps: int = 24,
    step_duration_sec: float = 10.0,
) -> None:
    """Generate oil change SOP video.

    Args:
        output_path: Output video path
        variant: 'gold' (correct) or 'trainee' (with deviations)
        width, height: Video dimensions
        fps: Frames per second
        step_duration_sec: Duration per step in seconds
    """
    frames_per_step = int(fps * step_duration_sec)

    # Define deviations for trainee version
    if variant == "trainee":
        # Trainee mistakes:
        # 1. Skip SAFETY (no glasses/gloves)
        # 2. Use wrong tool for FILTER (jack instead of wrench)
        # 3. Skip CHECK (forget dipstick verification)
        steps_to_use = [
            ("PARK", "Park vehicle on level surface", (255, 100, 100), None),
            # SKIP SAFETY
            ("LIFT", "Lift vehicle with jack and stands", (100, 100, 255), "jack"),
            ("LOCATE", "Locate oil drain plug under engine", (255, 255, 100), None),
            ("DRAIN", "Place drain pan and remove plug", (255, 100, 255), "pan"),
            ("FILTER", "Remove old oil filter with JACK (WRONG TOOL!)", (100, 255, 255), "jack"),  # DEVIATION
            ("INSTALL_FILTER", "Install new filter (hand-tight)", (255, 150, 100), "filter"),
            ("REINSTALL_PLUG", "Reinstall drain plug with torque wrench", (200, 200, 200), "wrench"),
            ("FILL", "Add new oil through filler cap", (150, 100, 255), None),
            # SKIP CHECK
        ]
    else:
        # Gold standard: all steps, correct tools
        steps_to_use = [
            ("PARK", "Park vehicle on level surface", (255, 100, 100), None),
            ("SAFETY", "Put on safety glasses and gloves", (100, 255, 100), "gloves"),
            ("LIFT", "Lift vehicle with jack and stands", (100, 100, 255), "jack"),
            ("LOCATE", "Locate oil drain plug under engine", (255, 255, 100), None),
            ("DRAIN", "Place drain pan and remove plug", (255, 100, 255), "pan"),
            ("FILTER", "Remove old oil filter with wrench", (100, 255, 255), "wrench"),
            ("INSTALL_FILTER", "Install new filter (hand-tight)", (255, 150, 100), "filter"),
            ("REINSTALL_PLUG", "Reinstall drain plug with torque wrench", (200, 200, 200), "wrench"),
            ("FILL", "Add new oil through filler cap", (150, 100, 255), None),
            ("CHECK", "Check oil level with dipstick", (100, 200, 150), "dipstick"),
        ]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer for {output_path}")

    # Generate frames
    total_steps = len(steps_to_use)
    for step_idx, (step_name, step_desc, bg_color, tool) in enumerate(steps_to_use):
        for frame_idx in range(frames_per_step):
            frame = draw_step_frame(
                width,
                height,
                step_name,
                step_desc,
                bg_color,
                tool,
                frame_idx,
                frames_per_step,
            )
            writer.write(frame)

        print(f"  [{variant}] Generated step {step_idx + 1}/{total_steps}: {step_name}")

    writer.release()

    total_duration = total_steps * step_duration_sec
    print(f"  [{variant}] Wrote {output_path} ({total_steps} steps, {total_duration:.1f}s @ {fps}fps)")


def main():
    parser = argparse.ArgumentParser(description="Generate manufacturing SOP demo videos")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("demo_videos/manufacturing"),
        help="Output directory",
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
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second",
    )
    parser.add_argument(
        "--step-duration",
        type=float,
        default=10.0,
        help="Duration per step (seconds)",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Manufacturing SOP Demo Video Generator")
    print("=" * 70)
    print(f"Output directory: {out_dir}")
    print(f"Resolution: {args.width}x{args.height} @ {args.fps}fps")
    print(f"Step duration: {args.step_duration}s")
    print()

    # Generate gold standard
    gold_path = out_dir / "oil_change_gold.mp4"
    print(f"[1/2] Generating GOLD standard video...")
    generate_oil_change_video(
        gold_path,
        variant="gold",
        width=args.width,
        height=args.height,
        fps=args.fps,
        step_duration_sec=args.step_duration,
    )
    print()

    # Generate trainee with deviations
    trainee_path = out_dir / "oil_change_trainee.mp4"
    print(f"[2/2] Generating TRAINEE video (with deviations)...")
    generate_oil_change_video(
        trainee_path,
        variant="trainee",
        width=args.width,
        height=args.height,
        fps=args.fps,
        step_duration_sec=args.step_duration,
    )
    print()

    print("=" * 70)
    print("✅ Success!")
    print("=" * 70)
    print()
    print("Generated videos:")
    print(f"  Gold:    {gold_path}")
    print(f"  Trainee: {trainee_path}")
    print()
    print("Trainee deviations:")
    print("  1. SKIPPED: Safety step (no glasses/gloves)")
    print("  2. WRONG TOOL: Used jack instead of wrench for filter removal")
    print("  3. SKIPPED: Check step (no dipstick verification)")
    print()
    print("Next steps:")
    print("  • Evaluate with SOPilot: python scripts/evaluate_manufacturing.py")
    print("  • Use in VIGIL-RAG: python scripts/vigil_smoke_e2e.py --video <path>")
    print()


if __name__ == "__main__":
    main()
