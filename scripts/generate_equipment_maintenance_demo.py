from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _draw_base(frame: np.ndarray, step: int, t: int) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, h), (18, 24, 30), -1)
    cv2.putText(
        frame,
        f"STEP {step}",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(frame, (20, 42), (140, 108), (70, 90, 110), 2)
    cv2.circle(frame, (80 + int(16 * np.sin(t / 6.0)), 75), 11, (220, 210, 70), -1)


def _draw_step_signature(frame: np.ndarray, step: int, t: int) -> None:
    h, w = frame.shape[:2]
    if step == 1:
        for x in range(150, w, 18):
            cv2.line(frame, (x, 40), (x - 40, h - 10), (235, 80, 80), 5)
    elif step == 2:
        for y in range(40, h, 20):
            cv2.circle(frame, (220 + int(8 * np.sin((t + y) / 5.0)), y), 8, (80, 220, 100), -1)
    elif step == 3:
        for y in range(40, h, 14):
            cv2.line(frame, (165, y), (315, y + int(6 * np.sin((t + y) / 4.0))), (80, 120, 240), 4)
    else:
        block = 18
        for yy in range(40, h, block):
            for xx in range(160, w, block):
                color = (220, 180, 70) if ((xx // block + yy // block) % 2 == 0) else (120, 100, 40)
                cv2.rectangle(frame, (xx, yy), (xx + block - 2, yy + block - 2), color, -1)


def _render_step(frame: np.ndarray, step: int, t: int) -> None:
    _draw_base(frame, step=step, t=t)
    _draw_step_signature(frame, step=step, t=t)
    if step == 1:
        cv2.rectangle(frame, (170, 45), (300, 100), (220, 70, 70), -1)
        cv2.putText(frame, "REMOVE", (178, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    elif step == 2:
        cv2.rectangle(frame, (170, 45), (300, 100), (70, 210, 90), -1)
        cv2.putText(frame, "CLEAN", (192, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    elif step == 3:
        cv2.rectangle(frame, (170, 45), (300, 100), (70, 120, 220), -1)
        cv2.putText(frame, "SWAP", (198, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    else:
        cv2.rectangle(frame, (170, 45), (300, 100), (220, 180, 70), -1)
        cv2.putText(frame, "CHECK", (188, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)


def _frame_step_for_variant(variant: str, frame_idx: int, fps: int, duration_sec: float) -> int:
    sec = frame_idx / float(fps)
    q = max(0.5, duration_sec / 4.0)
    if variant == "gold":
        if sec < q:
            return 1
        if sec < q * 2.0:
            return 2
        if sec < q * 3.0:
            return 3
        return 4

    if variant == "missing":
        if sec < q:
            return 1
        if sec < q * 2.0:
            return 3
        return 4

    if variant == "swap":
        if sec < q:
            return 1
        if sec < q * 2.0:
            return 3
        if sec < q * 3.0:
            return 2
        return 4

    if variant == "deviation":
        if sec < q:
            return 1
        if sec < q * 2.0:
            return 2
        if sec < q * 3.0:
            return 3
        return 4

    if variant == "time_over":
        if sec < q:
            return 1
        if sec < q * 2.7:
            return 2
        if sec < q * 3.4:
            return 3
        return 4

    if variant == "mixed":
        if sec < q:
            return 1
        if sec < q * 2.2:
            return 3
        if sec < q * 3.2:
            return 2
        return 4

    return 1


def _draw_variant_effect(frame: np.ndarray, variant: str, step: int, t: int) -> None:
    if variant == "deviation" and step == 3:
        cv2.line(frame, (150, 110), (315, 15), (240, 60, 240), 10)
        cv2.rectangle(frame, (170, 45), (300, 100), (230, 90, 230), -1)
        cv2.putText(frame, "WRONG", (186, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    if variant == "time_over" and step == 2:
        cv2.putText(frame, "WAIT...", (184, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)
    if variant == "mixed" and step in {2, 3}:
        cv2.circle(frame, (230 + int(12 * np.cos(t / 6.0)), 78), 14, (240, 240, 240), 2)
        cv2.rectangle(frame, (160, 120), (315, 170), (240, 240, 240), 2)


def _write_video(path: Path, variant: str, fps: int = 12, duration_sec: float = 24.0) -> None:
    width, height = 320, 180
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to create video: {path}")

    total_frames = int(round(fps * duration_sec))
    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        step = _frame_step_for_variant(variant, i, fps, duration_sec=duration_sec)
        _render_step(frame, step=step, t=i)
        _draw_variant_effect(frame, variant=variant, step=step, t=i)
        writer.write(frame)
    writer.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SOPilot equipment-maintenance demo videos")
    parser.add_argument("--out-dir", default="demo_videos/maintenance_filter_swap")
    parser.add_argument("--duration-sec", type=float, default=24.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = ["gold", "missing", "swap", "deviation", "time_over", "mixed"]
    for variant in variants:
        target = out_dir / f"{variant}.mp4"
        _write_video(target, variant=variant, duration_sec=max(4.0, float(args.duration_sec)))
        print(str(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
