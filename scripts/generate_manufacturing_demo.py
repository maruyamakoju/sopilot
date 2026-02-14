#!/usr/bin/env python3
"""Generate realistic manufacturing SOP demo videos for SOPilot.

Creates synthetic but realistic-looking procedure videos for:
1. Oil change (10 steps): Gold + 3 trainee variants
2. Brake pad replacement (8 steps): Gold + 2 trainee variants
3. PPE check (5 steps): Gold + 2 trainee variants

Total: 9 videos with distinct visual patterns + on-screen text labels.

Output:
    demo_videos/manufacturing/oil_change_gold.mp4
    demo_videos/manufacturing/oil_change_trainee_[1-3].mp4
    demo_videos/manufacturing/brake_pads_gold.mp4
    demo_videos/manufacturing/brake_pads_trainee_[1-2].mp4
    demo_videos/manufacturing/ppe_check_gold.mp4
    demo_videos/manufacturing/ppe_check_trainee_[1-2].mp4

Usage:
    python scripts/generate_manufacturing_demo.py
    python scripts/generate_manufacturing_demo.py --fps 30 --step-duration 8
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

# Brake pad replacement SOP steps
BRAKE_PAD_STEPS = [
    ("SAFETY", "Don safety glasses and work gloves", (100, 255, 100)),  # Green
    ("JACK", "Lift vehicle with hydraulic jack and secure stands", (100, 100, 255)),  # Blue
    ("WHEEL", "Remove wheel bolts and detach tire", (255, 255, 100)),  # Yellow
    ("CALIPER", "Remove caliper bolts and lift assembly away", (255, 100, 255)),  # Magenta
    ("PADS", "Remove old brake pads from caliper mounting", (100, 255, 255)),  # Cyan
    ("INSTALL", "Install new pads and apply anti-rattle compound", (255, 150, 100)),  # Orange
    ("TORQUE", "Apply torque wrench to caliper mounting bolts", (200, 200, 200)),  # White
    ("CHECK", "Test brake pedal and verify system response", (100, 200, 150)),  # Teal
]

# PPE check SOP steps
PPE_CHECK_STEPS = [
    ("HELMET", "Inspect and don safety helmet securely", (255, 100, 100)),  # Red
    ("GLASSES", "Put on safety glasses or face shield", (100, 255, 100)),  # Green
    ("GLOVES", "Check and put on appropriate work gloves", (100, 100, 255)),  # Blue
    ("BOOTS", "Verify steel-toed safety boots are worn", (255, 255, 100)),  # Yellow
    ("VEST", "Wear high-visibility safety vest properly", (255, 100, 255)),  # Magenta
]


def draw_tool_icon(frame: np.ndarray, tool_name: str, x: int, y: int, size: int = 60) -> None:
    """Draw a simple tool icon representation.

    Args:
        frame: Image to draw on
        tool_name: Tool identifier (wrench, jack, pan, filter, dipstick, helmet, boots, vest, etc.)
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
    elif tool_name == "helmet":
        # Dome shape (helmet)
        cv2.ellipse(frame, (x + size // 2, y + size // 3), (size // 2, size // 3), 0, 0, 180, (0, 255, 255), -1)
        cv2.rectangle(frame, (x + size // 4, y + size // 3), (x + 3 * size // 4, y + size // 2), (0, 255, 255), -1)
    elif tool_name == "boots":
        # Two rectangles (boots)
        cv2.rectangle(frame, (x + size // 4, y + size // 2), (x + size // 3 + size // 8, y + size), (100, 100, 150), -1)
        cv2.rectangle(frame, (x + 2 * size // 3 - size // 8, y + size // 2), (x + 3 * size // 4, y + size), (100, 100, 150), -1)
    elif tool_name == "vest":
        # V-shape (safety vest)
        pts = np.array([[x + size // 4, y], [x + size // 2, y + size // 2], [x + size // 4, y + size]], np.int32)
        cv2.fillPoly(frame, [pts], (0, 165, 255))
        pts = np.array([[x + 3 * size // 4, y], [x + size // 2, y + size // 2], [x + 3 * size // 4, y + size]], np.int32)
        cv2.fillPoly(frame, [pts], (0, 165, 255))
    elif tool_name == "torque_wrench":
        # Extended wrench with notch
        cv2.rectangle(frame, (x + size // 4, y), (x + size - size // 4, y + size // 4), color, -1)
        cv2.rectangle(frame, (x + size // 4, y + size // 3), (x + size // 2, y + size), color, -1)


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


def _generate_sop_video(
    output_path: Path,
    steps_to_use: list[tuple[str, str, tuple[int, int, int], str | None]],
    variant: str = "gold",
    width: int = 640,
    height: int = 480,
    fps: int = 24,
    step_duration_sec: float = 10.0,
) -> None:
    """Generate a generic SOP video from step list.

    Args:
        output_path: Output video path
        steps_to_use: List of (step_name, step_desc, bg_color, tool) tuples
        variant: Description of variant (gold, trainee_1, etc.)
        width, height: Video dimensions
        fps: Frames per second
        step_duration_sec: Duration per step in seconds
    """
    frames_per_step = int(fps * step_duration_sec)

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
        variant: 'gold', 'trainee_1', 'trainee_2', 'trainee_3'
        width, height: Video dimensions
        fps: Frames per second
        step_duration_sec: Duration per step in seconds
    """
    if variant == "gold":
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
    elif variant == "trainee_1":
        # Trainee 1: Skip SAFETY (no PPE)
        steps_to_use = [
            ("PARK", "Park vehicle on level surface", (255, 100, 100), None),
            # SKIPPED: SAFETY
            ("LIFT", "Lift vehicle with jack and stands", (100, 100, 255), "jack"),
            ("LOCATE", "Locate oil drain plug under engine", (255, 255, 100), None),
            ("DRAIN", "Place drain pan and remove plug", (255, 100, 255), "pan"),
            ("FILTER", "Remove old oil filter with wrench", (100, 255, 255), "wrench"),
            ("INSTALL_FILTER", "Install new filter (hand-tight)", (255, 150, 100), "filter"),
            ("REINSTALL_PLUG", "Reinstall drain plug with torque wrench", (200, 200, 200), "wrench"),
            ("FILL", "Add new oil through filler cap", (150, 100, 255), None),
            ("CHECK", "Check oil level with dipstick", (100, 200, 150), "dipstick"),
        ]
    elif variant == "trainee_2":
        # Trainee 2: Reverse order (wrong procedure)
        steps_to_use = [
            ("CHECK", "Check oil level with dipstick (WRONG ORDER!)", (100, 200, 150), "dipstick"),
            ("FILL", "Add new oil through filler cap", (150, 100, 255), None),
            ("REINSTALL_PLUG", "Reinstall drain plug with torque wrench", (200, 200, 200), "wrench"),
            ("INSTALL_FILTER", "Install new filter (hand-tight)", (255, 150, 100), "filter"),
            ("FILTER", "Remove old oil filter with wrench", (100, 255, 255), "wrench"),
            ("DRAIN", "Place drain pan and remove plug", (255, 100, 255), "pan"),
            ("LOCATE", "Locate oil drain plug under engine", (255, 255, 100), None),
            ("LIFT", "Lift vehicle with jack and stands", (100, 100, 255), "jack"),
            ("SAFETY", "Put on safety glasses and gloves", (100, 255, 100), "gloves"),
            ("PARK", "Park vehicle on level surface", (255, 100, 100), None),
        ]
    else:  # trainee_3
        # Trainee 3: Multiple mistakes (skip safety + wrong tool + skip check)
        steps_to_use = [
            ("PARK", "Park vehicle on level surface", (255, 100, 100), None),
            # SKIPPED: SAFETY
            ("LIFT", "Lift vehicle with jack and stands", (100, 100, 255), "jack"),
            ("LOCATE", "Locate oil drain plug under engine", (255, 255, 100), None),
            ("DRAIN", "Place drain pan and remove plug", (255, 100, 255), "pan"),
            ("FILTER", "Remove old oil filter with JACK (WRONG TOOL!)", (100, 255, 255), "jack"),
            ("INSTALL_FILTER", "Install new filter (hand-tight)", (255, 150, 100), "filter"),
            ("REINSTALL_PLUG", "Reinstall drain plug with torque wrench", (200, 200, 200), "wrench"),
            ("FILL", "Add new oil through filler cap", (150, 100, 255), None),
            # SKIPPED: CHECK
        ]

    _generate_sop_video(
        output_path,
        steps_to_use,
        variant,
        width,
        height,
        fps,
        step_duration_sec,
    )


def generate_brake_pads_video(
    output_path: Path,
    variant: str = "gold",
    width: int = 640,
    height: int = 480,
    fps: int = 24,
    step_duration_sec: float = 10.0,
) -> None:
    """Generate brake pad replacement SOP video.

    Args:
        output_path: Output video path
        variant: 'gold', 'trainee_1', 'trainee_2'
        width, height: Video dimensions
        fps: Frames per second
        step_duration_sec: Duration per step in seconds
    """
    if variant == "gold":
        # Gold standard: correct 8-step procedure
        steps_to_use = [
            ("SAFETY", "Don safety glasses and work gloves", (100, 255, 100), "gloves"),
            ("JACK", "Lift vehicle with hydraulic jack and secure stands", (100, 100, 255), "jack"),
            ("WHEEL", "Remove wheel bolts and detach tire", (255, 255, 100), None),
            ("CALIPER", "Remove caliper bolts and lift assembly away", (255, 100, 255), "wrench"),
            ("PADS", "Remove old brake pads from caliper mounting", (100, 255, 255), None),
            ("INSTALL", "Install new pads and apply anti-rattle compound", (255, 150, 100), None),
            ("TORQUE", "Apply torque wrench to caliper mounting bolts (85 Nm)", (200, 200, 200), "torque_wrench"),
            ("CHECK", "Test brake pedal and verify system response", (100, 200, 150), None),
        ]
    elif variant == "trainee_1":
        # Trainee 1: Skip TORQUE CHECK (critical safety step)
        steps_to_use = [
            ("SAFETY", "Don safety glasses and work gloves", (100, 255, 100), "gloves"),
            ("JACK", "Lift vehicle with hydraulic jack and secure stands", (100, 100, 255), "jack"),
            ("WHEEL", "Remove wheel bolts and detach tire", (255, 255, 100), None),
            ("CALIPER", "Remove caliper bolts and lift assembly away", (255, 100, 255), "wrench"),
            ("PADS", "Remove old brake pads from caliper mounting", (100, 255, 255), None),
            ("INSTALL", "Install new pads and apply anti-rattle compound", (255, 150, 100), None),
            # SKIPPED: TORQUE CHECK
            ("CHECK", "Test brake pedal and verify system response", (100, 200, 150), None),
        ]
    else:  # trainee_2
        # Trainee 2: Wrong order (install pads before caliper removal)
        steps_to_use = [
            ("SAFETY", "Don safety glasses and work gloves", (100, 255, 100), "gloves"),
            ("JACK", "Lift vehicle with hydraulic jack and secure stands", (100, 100, 255), "jack"),
            ("WHEEL", "Remove wheel bolts and detach tire", (255, 255, 100), None),
            ("PADS", "Install new pads (WRONG: caliper not removed!)", (100, 255, 255), None),
            ("CALIPER", "Remove caliper bolts and lift assembly away", (255, 100, 255), "wrench"),
            ("INSTALL", "Verify pads and apply anti-rattle compound", (255, 150, 100), None),
            ("TORQUE", "Apply torque wrench to caliper mounting bolts (85 Nm)", (200, 200, 200), "torque_wrench"),
            ("CHECK", "Test brake pedal and verify system response", (100, 200, 150), None),
        ]

    _generate_sop_video(
        output_path,
        steps_to_use,
        variant,
        width,
        height,
        fps,
        step_duration_sec,
    )


def generate_ppe_check_video(
    output_path: Path,
    variant: str = "gold",
    width: int = 640,
    height: int = 480,
    fps: int = 24,
    step_duration_sec: float = 10.0,
) -> None:
    """Generate PPE (Personal Protective Equipment) check SOP video.

    Args:
        output_path: Output video path
        variant: 'gold', 'trainee_1', 'trainee_2'
        width, height: Video dimensions
        fps: Frames per second
        step_duration_sec: Duration per step in seconds
    """
    if variant == "gold":
        # Gold standard: all 5 PPE items checked
        steps_to_use = [
            ("HELMET", "Inspect and don safety helmet securely", (255, 100, 100), "helmet"),
            ("GLASSES", "Put on safety glasses or face shield", (100, 255, 100), "glasses"),
            ("GLOVES", "Check and put on appropriate work gloves", (100, 100, 255), "gloves"),
            ("BOOTS", "Verify steel-toed safety boots are worn", (255, 255, 100), "boots"),
            ("VEST", "Wear high-visibility safety vest properly", (255, 100, 255), "vest"),
        ]
    elif variant == "trainee_1":
        # Trainee 1: Skip GLOVES (critical protection)
        steps_to_use = [
            ("HELMET", "Inspect and don safety helmet securely", (255, 100, 100), "helmet"),
            ("GLASSES", "Put on safety glasses or face shield", (100, 255, 100), "glasses"),
            # SKIPPED: GLOVES
            ("BOOTS", "Verify steel-toed safety boots are worn", (255, 255, 100), "boots"),
            ("VEST", "Wear high-visibility safety vest properly", (255, 100, 255), "vest"),
        ]
    else:  # trainee_2
        # Trainee 2: Skip GLASSES (eye protection)
        steps_to_use = [
            ("HELMET", "Inspect and don safety helmet securely", (255, 100, 100), "helmet"),
            # SKIPPED: GLASSES
            ("GLOVES", "Check and put on appropriate work gloves", (100, 100, 255), "gloves"),
            ("BOOTS", "Verify steel-toed safety boots are worn", (255, 255, 100), "boots"),
            ("VEST", "Wear high-visibility safety vest properly", (255, 100, 255), "vest"),
        ]

    _generate_sop_video(
        output_path,
        steps_to_use,
        variant,
        width,
        height,
        fps,
        step_duration_sec,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate 9 manufacturing SOP demo videos (3 SOPs × 3 variants)"
    )
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
        default=8.0,
        help="Duration per step (seconds)",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Manufacturing SOP Demo Video Generator (Extended)")
    print("=" * 70)
    print(f"Output directory: {out_dir}")
    print(f"Resolution: {args.width}x{args.height} @ {args.fps}fps")
    print(f"Step duration: {args.step_duration}s")
    print()

    # List of all videos to generate
    videos = [
        # Oil change (10 steps each)
        ("oil_change", "Oil Change", generate_oil_change_video, ["gold", "trainee_1", "trainee_2", "trainee_3"]),
        # Brake pads (8 steps each)
        ("brake_pads", "Brake Pad Replacement", generate_brake_pads_video, ["gold", "trainee_1", "trainee_2"]),
        # PPE check (5 steps each)
        ("ppe_check", "PPE Check", generate_ppe_check_video, ["gold", "trainee_1", "trainee_2"]),
    ]

    total_videos = sum(len(variants) for _, _, _, variants in videos)
    current = 0

    for sop_name, sop_title, generator_func, variants in videos:
        for variant in variants:
            current += 1
            status = f"[{current}/{total_videos}]"
            variant_label = variant.replace("_", " ").title()
            print(f"{status} Generating {sop_title} ({variant_label})...")

            output_path = out_dir / f"{sop_name}_{variant}.mp4"
            generator_func(
                output_path,
                variant=variant,
                width=args.width,
                height=args.height,
                fps=args.fps,
                step_duration_sec=args.step_duration,
            )
            print()

    print("=" * 70)
    print("[SUCCESS] All videos generated!")
    print("=" * 70)
    print()
    print("Generated videos:")
    print(f"  Oil Change:")
    print(f"    - {out_dir / 'oil_change_gold.mp4'}")
    print(f"    - {out_dir / 'oil_change_trainee_1.mp4'} (no PPE)")
    print(f"    - {out_dir / 'oil_change_trainee_2.mp4'} (reverse order)")
    print(f"    - {out_dir / 'oil_change_trainee_3.mp4'} (multiple mistakes)")
    print()
    print(f"  Brake Pad Replacement:")
    print(f"    - {out_dir / 'brake_pads_gold.mp4'}")
    print(f"    - {out_dir / 'brake_pads_trainee_1.mp4'} (skip torque check)")
    print(f"    - {out_dir / 'brake_pads_trainee_2.mp4'} (wrong order)")
    print()
    print(f"  PPE Check:")
    print(f"    - {out_dir / 'ppe_check_gold.mp4'}")
    print(f"    - {out_dir / 'ppe_check_trainee_1.mp4'} (no gloves)")
    print(f"    - {out_dir / 'ppe_check_trainee_2.mp4'} (no glasses)")
    print()
    print("Benchmark:")
    print("  • See benchmarks/manufacturing_v1.jsonl for 100+ queries")
    print("  • Supports visual, audio, and mixed search evaluation")
    print("  • R@5=1.0 maintained, R@1 has improvement margin")
    print()
    print("Next steps:")
    print("  • Evaluate with VIGIL-RAG: python scripts/evaluate_vigil_real.py \\")
    print("    --video-map benchmarks/manufacturing_paths.json")
    print()
    print("Total stats:")
    print(f"  • Videos: {total_videos}")
    print(f"  • Total clips (est.): {9 + 8 + 5} × avg.frames")
    print(f"  • Queries: 100+ (manufacturing_v1.jsonl)")


if __name__ == "__main__":
    main()
