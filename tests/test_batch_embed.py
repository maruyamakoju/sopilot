"""Tests for embed_batch across all embedder types and VideoProcessor batch path."""
import contextlib
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sopilot.services.embedder import ColorMotionEmbedder, FallbackEmbedder


def _random_frames(n: int = 8, h: int = 32, w: int = 32) -> list[np.ndarray]:
    return [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]


class ColorMotionEmbedBatchTests(unittest.TestCase):
    def setUp(self):
        self.embedder = ColorMotionEmbedder()

    def test_embed_batch_returns_2d_array(self):
        clips = [_random_frames() for _ in range(4)]
        result = self.embedder.embed_batch(clips)
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape[0], 4)
        self.assertEqual(result.dtype, np.float32)

    def test_embed_batch_dim_matches_single_embed(self):
        frames = _random_frames()
        single = self.embedder.embed(frames)
        batch = self.embedder.embed_batch([frames, frames])
        self.assertEqual(batch.shape[1], single.shape[0])

    def test_embed_batch_single_clip(self):
        clips = [_random_frames()]
        result = self.embedder.embed_batch(clips)
        self.assertEqual(result.shape[0], 1)

    def test_embed_batch_deterministic(self):
        np.random.seed(42)
        clips = [_random_frames() for _ in range(3)]
        r1 = self.embedder.embed_batch(clips)
        r2 = self.embedder.embed_batch(clips)
        np.testing.assert_array_equal(r1, r2)


class FallbackEmbedderBatchTests(unittest.TestCase):
    """FallbackEmbedder.embed_batch delegates to primary when healthy."""

    def _make_fallback(self):
        primary = ColorMotionEmbedder()
        fallback = ColorMotionEmbedder()
        return FallbackEmbedder(primary=primary, fallback=fallback)

    def test_embed_batch_returns_correct_shape(self):
        embedder = self._make_fallback()
        clips = [_random_frames() for _ in range(5)]
        result = embedder.embed_batch(clips)
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape[0], 5)

    def test_embed_batch_uses_primary_embed_batch_when_healthy(self):
        primary = ColorMotionEmbedder()
        fallback = ColorMotionEmbedder()
        embedder = FallbackEmbedder(primary=primary, fallback=fallback)

        original_batch = primary.embed_batch
        called = []

        def _spy(clips):
            called.append(len(clips))
            return original_batch(clips)

        primary.embed_batch = _spy
        clips = [_random_frames() for _ in range(3)]
        embedder.embed_batch(clips)
        self.assertEqual(called, [3], "primary.embed_batch should have been called once with 3 clips")

    def test_embed_batch_falls_back_to_per_clip_when_failed_over(self):
        """When primary is failed over, embed_batch falls back to per-clip embed()."""

        class _AlwaysFailPrimary:
            name = "fail"

            def embed(self, frames):
                raise RuntimeError("primary down")

            def embed_batch(self, clips):
                raise RuntimeError("primary down batch")

        fallback = ColorMotionEmbedder()
        embedder = FallbackEmbedder(primary=_AlwaysFailPrimary(), fallback=fallback)
        clips = [_random_frames() for _ in range(2)]
        # Force failover by calling embed() until _failed_over is set.
        # FallbackEmbedder fails over after MAX_CONSECUTIVE_FAILURES; use enough calls.
        for _ in range(embedder.MAX_CONSECUTIVE_FAILURES + 1):
            with contextlib.suppress(Exception):
                embedder.embed(clips[0])
        # Now embed_batch should route through per-clip fallback path.
        result = embedder.embed_batch(clips)
        self.assertEqual(result.shape[0], 2)


class VideoProcessorBatchPathTests(unittest.TestCase):
    """VideoProcessor uses embed_batch when processing a real video file."""

    def _make_video(self, path: Path, n_colors: int = 3) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(path), fourcc, 8.0, (64, 64))
        for i in range(n_colors):
            color = [0, 0, 0]
            color[i % 3] = 200
            frame = np.full((64, 64, 3), color, dtype=np.uint8)
            for _ in range(16):
                writer.write(frame)
        writer.release()

    def test_process_calls_embed_batch(self):
        from sopilot.services.video_processor import VideoProcessor

        embedder = ColorMotionEmbedder()
        batch_calls: list[int] = []
        original_batch = embedder.embed_batch

        def _spy(clips):
            batch_calls.append(len(clips))
            return original_batch(clips)

        embedder.embed_batch = _spy

        processor = VideoProcessor(
            sample_fps=4,
            clip_seconds=2,
            frame_size=64,
            embedder=embedder,
        )

        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "test.avi"
            self._make_video(video_path)
            clips = processor.process(str(video_path))

        # embed_batch should have been called at least once
        self.assertGreater(sum(batch_calls), 0, "embed_batch was never called")
        # The total clips embedded should match the returned clip list
        total_embedded = sum(batch_calls)
        self.assertEqual(total_embedded, len(clips))

    def test_process_returns_clips_with_embeddings(self):
        from sopilot.services.video_processor import VideoProcessor

        processor = VideoProcessor(
            sample_fps=4,
            clip_seconds=2,
            frame_size=64,
            embedder=ColorMotionEmbedder(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "test.avi"
            self._make_video(video_path, n_colors=4)
            clips = processor.process(str(video_path))

        self.assertGreater(len(clips), 0)
        for clip in clips:
            self.assertTrue(hasattr(clip, "embedding"), "ClipFeature should have 'embedding' attribute")
            emb = np.asarray(clip.embedding)
            self.assertEqual(emb.ndim, 1)
            self.assertGreater(emb.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
