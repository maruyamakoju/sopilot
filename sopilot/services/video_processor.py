import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sopilot.constants import BLUR_SHARPNESS_THRESHOLD, BRIGHTNESS_HIGH, BRIGHTNESS_LOW
from sopilot.services.embedder import VideoEmbedder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClipFeature:
    clip_index: int
    start_sec: float
    end_sec: float
    embedding: list[float]
    quality_flag: str | None = None


@dataclass
class _RawClipBuffer:
    """Holds raw decoded frames and timing before embedding."""
    frames: list[np.ndarray]
    start_sec: float
    end_sec: float


class VideoProcessor:
    def __init__(
        self,
        sample_fps: int,
        clip_seconds: int,
        frame_size: int,
        embedder: VideoEmbedder,
    ) -> None:
        self.sample_fps = max(sample_fps, 1)
        self.clip_seconds = max(clip_seconds, 1)
        self.frame_size = max(frame_size, 64)
        self.embedder = embedder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, video_path: str) -> list[ClipFeature]:
        """
        Decode *video_path*, split into clips, embed all clips, and return
        a list of ClipFeature objects.

        When the embedder exposes ``embed_batch()``, all clips are embedded
        in a single forward pass (important for GPU throughput with V-JEPA2).
        The single-clip ``embed()`` path is kept as a fallback.
        """
        logger.info("processing video %s", video_path)
        raw_clips = self._collect_raw_clips(video_path)

        embed_batch_fn = getattr(self.embedder, "embed_batch", None)
        if callable(embed_batch_fn) and len(raw_clips) > 1:
            embeddings = self._batch_embed(
                [rc.frames for rc in raw_clips], embed_batch_fn, video_path
            )
        else:
            embeddings = np.stack([
                self.embedder.embed(rc.frames) for rc in raw_clips
            ])

        clips = [
            ClipFeature(
                clip_index=i,
                start_sec=round(rc.start_sec, 4),
                end_sec=round(rc.end_sec, 4),
                embedding=embeddings[i].tolist(),
                quality_flag=self._quality_flag(rc.frames),
            )
            for i, rc in enumerate(raw_clips)
        ]
        logger.info("extracted %d clips from video", len(clips))
        return clips

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_raw_clips(self, video_path: str) -> list[_RawClipBuffer]:
        """Decode all frames and split into fixed-length clip buffers."""
        frames_per_clip = max(self.sample_fps * self.clip_seconds, 1)
        raw_clips: list[_RawClipBuffer] = []

        cap = cv2.VideoCapture(str(Path(video_path)))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        try:
            native_fps = float(cap.get(cv2.CAP_PROP_FPS))
            if native_fps <= 0:
                native_fps = 30.0
            sample_interval = max(int(round(native_fps / self.sample_fps)), 1)

            frame_idx = 0
            clip_frames: list[np.ndarray] = []
            clip_start_sec: float | None = None
            clip_end_sec: float | None = None

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % sample_interval == 0:
                    frame = cv2.resize(
                        frame,
                        (self.frame_size, self.frame_size),
                        interpolation=cv2.INTER_AREA,
                    )
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    timestamp = frame_idx / native_fps

                    if clip_start_sec is None:
                        clip_start_sec = timestamp
                    clip_end_sec = timestamp + (1.0 / self.sample_fps)
                    clip_frames.append(frame)

                    if len(clip_frames) >= frames_per_clip:
                        raw_clips.append(
                            _RawClipBuffer(
                                frames=clip_frames,
                                start_sec=clip_start_sec,
                                end_sec=clip_end_sec,
                            )
                        )
                        clip_frames = []
                        clip_start_sec = None
                        clip_end_sec = None
                frame_idx += 1

            if clip_frames and clip_start_sec is not None and clip_end_sec is not None:
                raw_clips.append(
                    _RawClipBuffer(
                        frames=clip_frames,
                        start_sec=clip_start_sec,
                        end_sec=clip_end_sec,
                    )
                )
        finally:
            cap.release()

        if not raw_clips:
            raise ValueError("No clips extracted from video")
        return raw_clips

    def _batch_embed(
        self,
        all_frames: list[list[np.ndarray]],
        embed_batch_fn: Any,
        video_path: str,
    ) -> np.ndarray:
        """Call embed_batch, setting runtime context for each clip beforehand."""
        set_context = getattr(self.embedder, "set_runtime_context", None)
        if callable(set_context):
            # Set a video-level context (clip-level context not available in batch mode).
            set_context(
                video_path=str(Path(video_path).resolve()),
                clip_index=None,
                clip_start_sec=None,
                clip_end_sec=None,
                frame_count=int(len(all_frames[0])) if all_frames else None,
            )
        return np.asarray(embed_batch_fn(all_frames))

    def _quality_flag(self, frames: list[np.ndarray]) -> str | None:
        arr = np.asarray(frames, dtype=np.float32)
        gray = (
            0.2989 * arr[..., 0]
            + 0.5870 * arr[..., 1]
            + 0.1140 * arr[..., 2]
        )
        brightness = float(gray.mean())
        if brightness < BRIGHTNESS_LOW or brightness > BRIGHTNESS_HIGH:
            return "extreme_exposure"

        blur_samples: list[float] = []
        for frame in frames[:: max(len(frames) // 3, 1)]:
            g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            blur_samples.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
        if blur_samples and float(np.mean(blur_samples)) < BLUR_SHARPNESS_THRESHOLD:
            return "low_sharpness"
        return None
