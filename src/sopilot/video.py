from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re

import cv2
import numpy as np


@dataclass
class ClipWindow:
    clip_idx: int
    start_sec: float
    end_sec: float
    frames: np.ndarray
    quality_flags: list[str]


def _resize_keep_ratio(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame
    scale = max_side / float(longest)
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def _clip_quality_flags(frames: np.ndarray) -> list[str]:
    if frames.size == 0:
        return ["empty"]

    grayscale = np.mean(frames, axis=3).astype(np.uint8)
    brightness = float(np.mean(grayscale))

    lap_vars = []
    for frame in grayscale:
        lap_vars.append(float(cv2.Laplacian(frame, cv2.CV_64F).var()))
    sharpness = float(np.mean(lap_vars)) if lap_vars else 0.0

    flags: list[str] = []
    if brightness < 40.0:
        flags.append("low_light")
    if brightness > 215.0:
        flags.append("over_exposed")
    if sharpness < 40.0:
        flags.append("blurry")
    return flags


def parse_mask_rects(spec: str) -> list[tuple[float, float, float, float]]:
    """
    Parse normalized rectangles from env spec.
    Format:
      "x1:y1:x2:y2;x1:y1:x2:y2"
    Values are clamped to [0, 1].
    """
    raw = spec.strip()
    if not raw:
        return []
    out: list[tuple[float, float, float, float]] = []
    for token in re.split(r"[;,]+", raw):
        item = token.strip()
        if not item:
            continue
        parts = [x.strip() for x in item.split(":")]
        if len(parts) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(x) for x in parts]
        except ValueError:
            continue
        x1 = min(1.0, max(0.0, x1))
        y1 = min(1.0, max(0.0, y1))
        x2 = min(1.0, max(0.0, x2))
        y2 = min(1.0, max(0.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((x1, y1, x2, y2))
    return out


def _blur_patch(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return
    h = max(3, (region.shape[0] // 8) * 2 + 1)
    w = max(3, (region.shape[1] // 8) * 2 + 1)
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, (w, h), 0)


def apply_privacy_mask(
    frame: np.ndarray,
    *,
    rects: list[tuple[float, float, float, float]],
    mode: str = "black",
    face_blur: bool = False,
    face_cascade=None,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    mode_norm = mode.strip().lower()

    for x1n, y1n, x2n, y2n in rects:
        x1 = int(round(x1n * w))
        y1 = int(round(y1n * h))
        x2 = int(round(x2n * w))
        y2 = int(round(y2n * h))
        x1 = min(max(0, x1), w)
        x2 = min(max(0, x2), w)
        y1 = min(max(0, y1), h)
        y2 = min(max(0, y2), h)
        if x2 <= x1 or y2 <= y1:
            continue
        if mode_norm == "blur":
            _blur_patch(out, x1, y1, x2, y2)
        else:
            out[y1:y2, x1:x2] = 0

    if face_blur and face_cascade is not None:
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
        for x, y, fw, fh in faces:
            _blur_patch(out, int(x), int(y), int(x + fw), int(y + fh))

    return out


class ClipWindowStream:
    """
    Streaming clip extractor to keep memory bounded on long videos.

    Usage:
        stream = ClipWindowStream(...)
        for clip in stream:
            ...
        stats = stream.stats
    """

    def __init__(
        self,
        *,
        video_path: Path,
        target_fps: int,
        clip_seconds: float,
        max_side: int,
        min_clip_coverage: float,
        privacy_mask_enabled: bool = False,
        privacy_mask_mode: str = "black",
        privacy_mask_rects: str = "",
        privacy_face_blur: bool = False,
    ) -> None:
        self.video_path = video_path
        self.target_fps = target_fps
        self.clip_seconds = clip_seconds
        self.max_side = max_side
        self.min_clip_coverage = min_clip_coverage
        self.privacy_mask_enabled = bool(privacy_mask_enabled)
        self.privacy_mask_mode = privacy_mask_mode.strip().lower() or "black"
        self.privacy_mask_rects = parse_mask_rects(privacy_mask_rects)
        self.privacy_face_blur = bool(privacy_face_blur)
        self._face_cascade = None
        if self.privacy_mask_enabled and self.privacy_face_blur:
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                candidate = cv2.CascadeClassifier(cascade_path)
                if not candidate.empty():
                    self._face_cascade = candidate
            except Exception:
                self._face_cascade = None
        self.stats: dict = {}

    def __iter__(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"unable to open video: {self.video_path}")

        src_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(src_fps) or src_fps <= 0.0:
            src_fps = 30.0

        sample_every = max(1, int(round(src_fps / max(self.target_fps, 1))))
        sampled_fps = src_fps / sample_every
        frames_per_clip = max(1, int(round(sampled_fps * self.clip_seconds)))

        frame_idx = 0
        sampled_count = 0
        clip_idx = 0
        clip_frames: list[np.ndarray] = []
        clip_sampled_start = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_idx % sample_every == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb = _resize_keep_ratio(rgb, self.max_side)
                    if self.privacy_mask_enabled:
                        rgb = apply_privacy_mask(
                            rgb,
                            rects=self.privacy_mask_rects,
                            mode=self.privacy_mask_mode,
                            face_blur=self.privacy_face_blur,
                            face_cascade=self._face_cascade,
                        )
                    if not clip_frames:
                        clip_sampled_start = sampled_count
                    clip_frames.append(rgb)
                    sampled_count += 1

                    if len(clip_frames) >= frames_per_clip:
                        window = np.stack(clip_frames, axis=0)
                        start_sec = clip_sampled_start / sampled_fps
                        end_sec = sampled_count / sampled_fps
                        yield ClipWindow(
                            clip_idx=clip_idx,
                            start_sec=float(start_sec),
                            end_sec=float(end_sec),
                            frames=window,
                            quality_flags=_clip_quality_flags(window),
                        )
                        clip_idx += 1
                        clip_frames = []

                frame_idx += 1

            remainder = len(clip_frames)
            if remainder > 0:
                coverage = remainder / float(max(frames_per_clip, 1))
                if coverage >= self.min_clip_coverage:
                    window = np.stack(clip_frames, axis=0)
                    pad_len = frames_per_clip - int(window.shape[0])
                    if pad_len > 0:
                        pad = np.repeat(window[-1:], repeats=pad_len, axis=0)
                        padded = np.concatenate([window, pad], axis=0)
                    else:
                        padded = window

                    start_sec = clip_sampled_start / sampled_fps
                    end_sec = sampled_count / sampled_fps
                    yield ClipWindow(
                        clip_idx=clip_idx,
                        start_sec=float(start_sec),
                        end_sec=float(end_sec),
                        frames=padded,
                        quality_flags=_clip_quality_flags(window),
                    )
                    clip_idx += 1
                elif clip_idx == 0:
                    # Very short input still yields one clip.
                    window = np.stack(clip_frames, axis=0)
                    yield ClipWindow(
                        clip_idx=0,
                        start_sec=0.0,
                        end_sec=float(sampled_count / sampled_fps) if sampled_fps > 0 else 0.0,
                        frames=window,
                        quality_flags=_clip_quality_flags(window),
                    )
                    clip_idx = 1

            self.stats = {
                "source_fps": float(src_fps),
                "sampled_fps": float(sampled_fps),
                "sample_every": int(sample_every),
                "frames_total": int(sampled_count),
                "frames_per_clip": int(frames_per_clip),
                "clip_count": int(clip_idx),
            }
        finally:
            cap.release()


def extract_clip_windows(
    video_path: Path,
    target_fps: int,
    clip_seconds: float,
    max_side: int,
    min_clip_coverage: float,
    privacy_mask_enabled: bool = False,
    privacy_mask_mode: str = "black",
    privacy_mask_rects: str = "",
    privacy_face_blur: bool = False,
) -> tuple[list[ClipWindow], dict]:
    stream = ClipWindowStream(
        video_path=video_path,
        target_fps=target_fps,
        clip_seconds=clip_seconds,
        max_side=max_side,
        min_clip_coverage=min_clip_coverage,
        privacy_mask_enabled=privacy_mask_enabled,
        privacy_mask_mode=privacy_mask_mode,
        privacy_mask_rects=privacy_mask_rects,
        privacy_face_blur=privacy_face_blur,
    )
    clips = list(stream)
    return clips, stream.stats
