#!/usr/bin/env python3
"""Generate a 96-second benchmark video with 10 distinct visual+audio steps.

Produces a video with:
- 10 steps, each ~9.6s (96s total), 640x360 at 24fps
- Distinct visual patterns per step (stripes, circles, lines, etc.)
- Audio sine tones at unique frequencies per step (440-880Hz)
- Optional TTS step names via pyttsx3 (graceful fallback to tone-only)
- ffmpeg muxes audio+video into final MP4

Expected chunking output: ~30 micro, 5-8 meso, 2-3 macro chunks.

Usage:
    python scripts/generate_vigil_benchmark_v2.py --out-dir demo_videos/benchmark_v2
    python scripts/generate_vigil_benchmark_v2.py --out-dir demo_videos/benchmark_v2 --tts
"""

from __future__ import annotations

import argparse
import logging
import math
import struct
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

STEPS = [
    {"name": "REMOVE",     "freq": 440,  "color": (0, 0, 200),     "pattern": "red_stripes"},
    {"name": "CLEAN",      "freq": 494,  "color": (0, 180, 0),     "pattern": "green_circles"},
    {"name": "SWAP",       "freq": 523,  "color": (200, 0, 0),     "pattern": "blue_lines"},
    {"name": "CHECK",      "freq": 554,  "color": (0, 200, 200),   "pattern": "yellow_checker"},
    {"name": "RINSE",      "freq": 587,  "color": (200, 200, 0),   "pattern": "cyan_rings"},
    {"name": "INSTALL",    "freq": 622,  "color": (200, 0, 200),   "pattern": "magenta_stars"},
    {"name": "ALIGN",      "freq": 659,  "color": (0, 128, 255),   "pattern": "orange_zigzag"},
    {"name": "TIGHTEN",    "freq": 698,  "color": (255, 255, 255), "pattern": "white_dots"},
    {"name": "CALIBRATE",  "freq": 740,  "color": (128, 0, 200),   "pattern": "purple_gradient"},
    {"name": "VERIFY",     "freq": 784,  "color": (200, 128, 0),   "pattern": "teal_crosshatch"},
]

DURATION_SEC = 96.0
FPS = 24
WIDTH = 640
HEIGHT = 360
SAMPLE_RATE = 44100
NUM_STEPS = len(STEPS)
STEP_DURATION = DURATION_SEC / NUM_STEPS  # 9.6s per step


# ---------------------------------------------------------------------------
# Pattern renderers
# ---------------------------------------------------------------------------


def _draw_red_stripes(frame: np.ndarray, t: float) -> None:
    """Red diagonal stripes, animated."""
    h, w = frame.shape[:2]
    offset = int(t * 30) % 40
    for y in range(h):
        for x in range(0, w, 2):
            if ((x + y + offset) // 20) % 2 == 0:
                frame[y, x] = (0, 0, 220)
                if x + 1 < w:
                    frame[y, x + 1] = (0, 0, 220)


def _draw_red_stripes_fast(frame: np.ndarray, t: float) -> None:
    """Red diagonal stripes using vectorised operations."""
    h, w = frame.shape[:2]
    offset = int(t * 30) % 40
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (((xx + yy + offset) // 20) % 2 == 0)
    frame[mask] = (0, 0, 220)


def _draw_green_circles(frame: np.ndarray, t: float) -> None:
    """Green circles on dark background."""
    frame[:] = (20, 20, 20)
    radius = 20 + int(5 * math.sin(t * 2))
    for cx in range(50, WIDTH, 100):
        for cy in range(40, HEIGHT, 80):
            cv2.circle(frame, (cx, cy), radius, (0, 200, 0), -1)


def _draw_blue_lines(frame: np.ndarray, t: float) -> None:
    """Blue horizontal lines, with vertical drift."""
    frame[:] = (10, 10, 10)
    offset = int(t * 10) % 30
    for y in range(offset, HEIGHT, 30):
        cv2.line(frame, (0, y), (WIDTH, y), (200, 0, 0), 3)


def _draw_yellow_checker(frame: np.ndarray, t: float) -> None:
    """Yellow/black checkerboard."""
    frame[:] = (0, 0, 0)
    size = 40
    phase = int(t * 2) % 2
    for row in range(0, HEIGHT, size):
        for col in range(0, WIDTH, size):
            if ((row // size + col // size + phase) % 2) == 0:
                frame[row : row + size, col : col + size] = (0, 220, 220)


def _draw_cyan_rings(frame: np.ndarray, t: float) -> None:
    """Concentric cyan rings, pulsing."""
    frame[:] = (15, 15, 15)
    cx, cy = WIDTH // 2, HEIGHT // 2
    max_r = int(math.hypot(cx, cy))
    pulse = int(t * 15) % 30
    for r in range(pulse, max_r, 30):
        cv2.circle(frame, (cx, cy), r, (220, 220, 0), 2)


def _draw_magenta_stars(frame: np.ndarray, t: float) -> None:
    """Magenta star shapes on dark background."""
    frame[:] = (10, 10, 10)
    seed = int(t * 3) % 5
    rng = np.random.RandomState(seed)
    for _ in range(12):
        cx = rng.randint(40, WIDTH - 40)
        cy = rng.randint(30, HEIGHT - 30)
        size = rng.randint(15, 35)
        _draw_star(frame, cx, cy, size, (220, 0, 220))


def _draw_star(frame: np.ndarray, cx: int, cy: int, size: int, color: tuple) -> None:
    """Draw a 5-pointed star."""
    pts = []
    for i in range(5):
        angle = math.radians(i * 72 - 90)
        pts.append((int(cx + size * math.cos(angle)), int(cy + size * math.sin(angle))))
    inner_r = size * 0.4
    inner_pts = []
    for i in range(5):
        angle = math.radians(i * 72 - 90 + 36)
        inner_pts.append((int(cx + inner_r * math.cos(angle)), int(cy + inner_r * math.sin(angle))))
    all_pts = []
    for i in range(5):
        all_pts.append(pts[i])
        all_pts.append(inner_pts[i])
    pts_arr = np.array(all_pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(frame, [pts_arr], color)


def _draw_orange_zigzag(frame: np.ndarray, t: float) -> None:
    """Orange zigzag lines."""
    frame[:] = (15, 15, 15)
    offset = int(t * 20) % 60
    for base_y in range(offset - 60, HEIGHT + 60, 60):
        pts = []
        for x in range(0, WIDTH + 30, 30):
            y = base_y + (20 if (x // 30) % 2 == 0 else -20)
            pts.append((x, y))
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], (0, 140, 255), 3)


def _draw_white_dots(frame: np.ndarray, t: float) -> None:
    """White dots grid, blinking."""
    frame[:] = (10, 10, 10)
    phase = int(t * 4) % 3
    for y in range(20, HEIGHT, 40):
        for x in range(20, WIDTH, 40):
            if ((x // 40 + y // 40 + phase) % 3) != 0:
                cv2.circle(frame, (x, y), 8, (255, 255, 255), -1)


def _draw_purple_gradient(frame: np.ndarray, t: float) -> None:
    """Vertical purple gradient, shifting over time."""
    shift = int(t * 30) % 256
    for y in range(HEIGHT):
        val = (y * 256 // HEIGHT + shift) % 256
        frame[y, :] = (val, 0, min(255, val + 80))


def _draw_teal_crosshatch(frame: np.ndarray, t: float) -> None:
    """Teal crosshatch pattern."""
    frame[:] = (15, 15, 15)
    spacing = 25
    offset = int(t * 10) % spacing
    for x in range(offset - WIDTH, WIDTH + HEIGHT, spacing):
        cv2.line(frame, (x, 0), (x + HEIGHT, HEIGHT), (180, 128, 0), 2)
    for x in range(offset - WIDTH, WIDTH + HEIGHT, spacing):
        cv2.line(frame, (x, HEIGHT), (x + HEIGHT, 0), (180, 128, 0), 2)


PATTERN_RENDERERS = {
    "red_stripes": _draw_red_stripes_fast,
    "green_circles": _draw_green_circles,
    "blue_lines": _draw_blue_lines,
    "yellow_checker": _draw_yellow_checker,
    "cyan_rings": _draw_cyan_rings,
    "magenta_stars": _draw_magenta_stars,
    "orange_zigzag": _draw_orange_zigzag,
    "white_dots": _draw_white_dots,
    "purple_gradient": _draw_purple_gradient,
    "teal_crosshatch": _draw_teal_crosshatch,
}


# ---------------------------------------------------------------------------
# Audio generation
# ---------------------------------------------------------------------------


def generate_audio_wav(out_path: Path) -> None:
    """Generate a WAV with distinct sine tones per step.

    Each step gets a unique frequency from STEPS[i]["freq"].
    """
    total_samples = int(DURATION_SEC * SAMPLE_RATE)
    samples_per_step = total_samples // NUM_STEPS

    data = []
    for _step_idx, step in enumerate(STEPS):
        freq = step["freq"]
        for s in range(samples_per_step):
            t = s / SAMPLE_RATE
            # Sine tone with gentle amplitude envelope
            envelope = 0.7 * min(1.0, t / 0.1, (samples_per_step / SAMPLE_RATE - t) / 0.1)
            value = envelope * math.sin(2 * math.pi * freq * t)
            data.append(int(value * 32767))

    # Pad to exact duration
    while len(data) < total_samples:
        data.append(0)
    data = data[:total_samples]

    with wave.open(str(out_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(struct.pack(f"<{len(data)}h", *data))

    logger.info("Audio WAV saved: %s (%.1fs, %d samples)", out_path, DURATION_SEC, len(data))


def generate_tts_wav(out_path: Path) -> bool:
    """Generate TTS audio with step names spoken at appropriate times.

    Returns True if successful, False if pyttsx3 not available.
    """
    try:
        import pyttsx3
    except ImportError:
        logger.warning("pyttsx3 not installed, skipping TTS (pip install pyttsx3)")
        return False

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 130)

        # Generate each step name individually then concatenate
        step_wavs = []
        for i, step in enumerate(STEPS):
            step_wav = out_path.parent / f"_tts_step_{i}.wav"
            engine.save_to_file(f"Step {i+1}: {step['name']}", str(step_wav))
            step_wavs.append(step_wav)

        engine.runAndWait()

        # Mix TTS into a timeline WAV
        total_samples = int(DURATION_SEC * SAMPLE_RATE)
        mixed = np.zeros(total_samples, dtype=np.float32)

        for i, step_wav in enumerate(step_wavs):
            if step_wav.exists():
                try:
                    with wave.open(str(step_wav), "r") as wf:
                        n_frames = wf.getnframes()
                        raw = wf.readframes(n_frames)
                        if wf.getsampwidth() == 2:
                            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                        else:
                            samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                        # Place at step start + 0.5s offset
                        start_sample = int((i * STEP_DURATION + 0.5) * SAMPLE_RATE)
                        end_sample = min(start_sample + len(samples), total_samples)
                        n = end_sample - start_sample
                        if n > 0:
                            mixed[start_sample:end_sample] = samples[:n]
                except Exception as exc:
                    logger.warning("Failed to read TTS wav for step %d: %s", i, exc)
                step_wav.unlink(missing_ok=True)

        # Write mixed WAV
        int_data = (mixed * 32767).astype(np.int16)
        with wave.open(str(out_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(int_data.tobytes())

        logger.info("TTS WAV saved: %s", out_path)
        return True

    except Exception as exc:
        logger.warning("TTS generation failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------


def generate_video_only(out_path: Path) -> None:
    """Generate raw video (no audio) with 10 distinct visual patterns."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {out_path}")

    total_frames = int(DURATION_SEC * FPS)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    for f_idx in range(total_frames):
        t = f_idx / FPS
        step_idx = min(int(t / STEP_DURATION), NUM_STEPS - 1)
        step = STEPS[step_idx]
        local_t = t - step_idx * STEP_DURATION

        # Draw pattern
        frame[:] = 0
        renderer = PATTERN_RENDERERS[step["pattern"]]
        renderer(frame, local_t)

        # Overlay step label
        label = f"Step {step_idx+1}: {step['name']}"
        time_label = f"t={t:.1f}s"
        cv2.putText(frame, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, time_label, (20, HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        writer.write(frame)

        if (f_idx + 1) % (FPS * 10) == 0:
            logger.info("  Video: %d/%d frames (%.0f%%)", f_idx + 1, total_frames, 100 * (f_idx + 1) / total_frames)

    writer.release()
    logger.info("Video-only saved: %s (%d frames, %.1fs)", out_path, total_frames, DURATION_SEC)


# ---------------------------------------------------------------------------
# Mux with ffmpeg
# ---------------------------------------------------------------------------


def mux_audio_video(video_path: Path, audio_path: Path, output_path: Path) -> None:
    """Combine video and audio using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        str(output_path),
    ]
    logger.info("Muxing: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("ffmpeg failed: %s", result.stderr[-500:] if result.stderr else "unknown error")
        raise RuntimeError(f"ffmpeg exited with code {result.returncode}")
    logger.info("Muxed output: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate VIGIL benchmark v2 video (96s, 10 steps)")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="demo_videos/benchmark_v2",
        help="Output directory (default: demo_videos/benchmark_v2)",
    )
    parser.add_argument(
        "--tts",
        action="store_true",
        help="Use pyttsx3 TTS for spoken step names (optional)",
    )
    parser.add_argument(
        "--tone-only",
        action="store_true",
        help="Skip TTS, use sine tones only (default behavior when --tts not set)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_only_path = out_dir / "_video_raw.mp4"
    audio_path = out_dir / "_audio.wav"
    tts_path = out_dir / "_tts.wav"
    final_path = out_dir / "gold.mp4"

    # Step 1: Generate video
    logger.info("=== Step 1/3: Generating video (%d steps, %.0fs) ===", NUM_STEPS, DURATION_SEC)
    generate_video_only(video_only_path)

    # Step 2: Generate audio
    logger.info("=== Step 2/3: Generating audio ===")
    use_tts = False
    if args.tts:
        use_tts = generate_tts_wav(tts_path)

    if use_tts:
        # Mix TTS + tones
        logger.info("Mixing TTS + sine tones")
        generate_audio_wav(audio_path)
        _mix_wav_files(audio_path, tts_path, audio_path)
    else:
        generate_audio_wav(audio_path)

    # Step 3: Mux
    logger.info("=== Step 3/3: Muxing audio + video ===")
    try:
        mux_audio_video(video_only_path, audio_path, final_path)
    except RuntimeError:
        logger.warning("ffmpeg mux failed; copying video-only as fallback")
        import shutil
        shutil.copy2(video_only_path, final_path)

    # Clean up intermediate files
    for p in [video_only_path, audio_path, tts_path]:
        p.unlink(missing_ok=True)

    logger.info("=== Done! ===")
    logger.info("Output: %s", final_path)
    logger.info("Steps: %s", ", ".join(s["name"] for s in STEPS))
    logger.info("Expected: ~30 micro, 5-8 meso, 2-3 macro chunks")

    return 0


def _mix_wav_files(base_path: Path, overlay_path: Path, output_path: Path) -> None:
    """Mix two mono WAV files (add samples, clip)."""
    with wave.open(str(base_path), "r") as wf:
        base_raw = wf.readframes(wf.getnframes())
        sr = wf.getframerate()
        base_samples = np.frombuffer(base_raw, dtype=np.int16).astype(np.float32)

    with wave.open(str(overlay_path), "r") as wf:
        overlay_raw = wf.readframes(wf.getnframes())
        overlay_samples = np.frombuffer(overlay_raw, dtype=np.int16).astype(np.float32)

    # Pad shorter to match longer
    n = max(len(base_samples), len(overlay_samples))
    base_padded = np.zeros(n, dtype=np.float32)
    base_padded[: len(base_samples)] = base_samples
    overlay_padded = np.zeros(n, dtype=np.float32)
    overlay_padded[: len(overlay_samples)] = overlay_samples

    mixed = np.clip(base_padded * 0.5 + overlay_padded * 0.7, -32768, 32767).astype(np.int16)

    with wave.open(str(output_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(mixed.tobytes())


if __name__ == "__main__":
    sys.exit(main())
