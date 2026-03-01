import importlib
import unittest

import numpy as np

from sopilot.services.embedder import FallbackEmbedder, VJEPA2Config, VJEPA2HubEmbedder

# Skip all VJEPA2HubEmbedder tests in environments that have no torch installed
# (e.g. CPU-only CI). The mock-based tests import torch inside each method.
_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class _FailingEmbedder:
    name = "primary"

    def embed(self, frames):
        raise RuntimeError("boom")


class _WorkingEmbedder:
    name = "fallback"

    def __init__(self):
        self.calls = 0

    def embed(self, frames):
        self.calls += 1
        return np.ones(3, dtype=np.float32)


class _SequencePrimary:
    name = "primary"

    def __init__(self):
        self.calls = 0

    def embed(self, frames):
        self.calls += 1
        if self.calls == 1:
            return np.ones(5, dtype=np.float32)
        raise RuntimeError("primary down")


class _ShortFallback:
    name = "fallback"

    def __init__(self):
        self.calls = 0

    def embed(self, frames):
        self.calls += 1
        return np.asarray([1.0, 2.0, 3.0], dtype=np.float32)


class _AlwaysPrimary:
    name = "primary"

    def __init__(self):
        self.calls = 0

    def embed(self, frames):
        self.calls += 1
        return np.asarray([1.0, 2.0, 3.0], dtype=np.float32)


class FallbackEmbedderTests(unittest.TestCase):
    def test_falls_back_and_keeps_using_fallback(self) -> None:
        fallback = _WorkingEmbedder()
        embedder = FallbackEmbedder(_FailingEmbedder(), fallback)
        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]

        first = embedder.embed(frames)
        second = embedder.embed(frames)

        self.assertEqual(fallback.calls, 2)
        self.assertEqual(first.tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(second.tolist(), [1.0, 1.0, 1.0])

    def test_fallback_embedding_is_coerced_to_expected_dimension(self) -> None:
        primary = _SequencePrimary()
        fallback = _ShortFallback()
        embedder = FallbackEmbedder(primary, fallback)
        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]

        first = embedder.embed(frames)
        second = embedder.embed(frames)

        self.assertEqual(primary.calls, 2)
        self.assertEqual(fallback.calls, 1)
        self.assertEqual(first.shape[0], 5)
        self.assertEqual(second.shape[0], 5)
        # The first elements retain fallback signal after coercion.
        self.assertGreater(float(second[0]), 0.0)
        self.assertGreater(float(second[1]), 0.0)

    def test_primary_success_is_called_once_per_request(self) -> None:
        primary = _AlwaysPrimary()
        fallback = _WorkingEmbedder()
        embedder = FallbackEmbedder(primary, fallback)
        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]

        embedder.embed(frames)
        embedder.embed(frames)

        stats = embedder.get_stats()
        self.assertEqual(primary.calls, 2)
        self.assertEqual(fallback.calls, 0)
        self.assertEqual(int(stats["primary_attempts"]), 2)
        self.assertEqual(int(stats["primary_successes"]), 2)
        self.assertEqual(int(stats["fallback_uses"]), 0)


@unittest.skipUnless(_TORCH_AVAILABLE, "torch not installed — skipped in CPU-only CI")
class VJEPA2HubEmbedderTests(unittest.TestCase):
    def test_single_frame_clip_is_duplicated_for_temporal_depth(self) -> None:
        import torch

        class _SpyProcessor:
            def __init__(self):
                self.last_clip_shape = None

            def __call__(self, clip):
                self.last_clip_shape = tuple(int(v) for v in clip.shape)
                # Return C,T,H,W tensor expected by embedder.
                return torch.zeros((3, int(clip.shape[0]), 16, 16), dtype=torch.float32)

        class _DummyEncoder:
            def __call__(self, model_input):
                # Return [B, tokens, dim]
                bsz = int(model_input.shape[0])
                return torch.ones((bsz, 2, 4), dtype=torch.float32)

        embedder = VJEPA2HubEmbedder(
            VJEPA2Config(
                variant="vjepa2_vit_large",
                pretrained=False,
                crop_size=256,
                device="cpu",
                use_amp=False,
            )
        )
        spy = _SpyProcessor()
        embedder._ready = True
        embedder._torch = torch
        embedder._encoder = _DummyEncoder()
        embedder._processor = spy
        embedder._device = torch.device("cpu")

        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        vector = embedder.embed([frame])

        self.assertEqual(spy.last_clip_shape[0], 2)
        self.assertEqual(vector.shape[0], 4)

    def test_pooling_mode_first_token_uses_first_token_vector(self) -> None:
        import torch

        class _SpyProcessor:
            def __call__(self, clip):
                return torch.zeros((3, int(clip.shape[0]), 16, 16), dtype=torch.float32)

        class _DummyEncoder:
            def __call__(self, model_input):
                # [B, tokens, dim] with distinct first token.
                return torch.tensor([[[3.0, 4.0], [0.0, 1.0]]], dtype=torch.float32)

        embedder = VJEPA2HubEmbedder(
            VJEPA2Config(
                variant="vjepa2_vit_large",
                pretrained=False,
                crop_size=256,
                device="cpu",
                use_amp=False,
                pooling="first_token",
            )
        )
        embedder._ready = True
        embedder._torch = torch
        embedder._encoder = _DummyEncoder()
        embedder._processor = _SpyProcessor()
        embedder._device = torch.device("cpu")

        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        vector = embedder.embed([frame, frame])

        self.assertEqual(vector.shape[0], 2)
        # L2-normalized [3,4] -> [0.6, 0.8]
        self.assertAlmostEqual(float(vector[0]), 0.6, places=4)
        self.assertAlmostEqual(float(vector[1]), 0.8, places=4)

    def test_preprocessor_receives_tchw_torch_tensor(self) -> None:
        """embed() must convert (H,W,C) numpy frames to (T,C,H,W) torch tensor for preprocessor."""
        import torch

        received: list = []

        class _RecordingProcessor:
            def __call__(self, clip):
                received.append(clip)
                # Return C,T,H,W tensor as the real preprocessor does (in a list).
                t = int(clip.shape[0])
                return [torch.zeros((3, t, 16, 16), dtype=torch.float32)]

        class _DummyEncoder:
            def __call__(self, model_input):
                bsz = int(model_input.shape[0])
                return torch.ones((bsz, 8, 4), dtype=torch.float32)

        embedder = VJEPA2HubEmbedder(
            VJEPA2Config(
                variant="vjepa2_vit_large",
                pretrained=False,
                crop_size=256,
                device="cpu",
                use_amp=False,
            )
        )
        embedder._ready = True
        embedder._torch = torch
        embedder._encoder = _DummyEncoder()
        embedder._processor = _RecordingProcessor()
        embedder._device = torch.device("cpu")

        frames = [np.zeros((16, 16, 3), dtype=np.uint8)] * 4
        embedder.embed(frames)

        self.assertEqual(len(received), 1)
        clip = received[0]
        # Must be a torch.Tensor, not a numpy array.
        self.assertIsInstance(clip, torch.Tensor)
        # Shape must be (T, C, H, W) = (4, 3, 16, 16)
        self.assertEqual(clip.ndim, 4)
        self.assertEqual(int(clip.shape[0]), 4)   # T
        self.assertEqual(int(clip.shape[1]), 3)   # C
        self.assertEqual(int(clip.shape[2]), 16)  # H
        self.assertEqual(int(clip.shape[3]), 16)  # W
        # dtype must be uint8 (raw pixel values 0-255)
        self.assertEqual(clip.dtype, torch.uint8)

    def test_pooling_mode_flatten_flattens_token_tensor(self) -> None:
        import torch

        class _SpyProcessor:
            def __call__(self, clip):
                return torch.zeros((3, int(clip.shape[0]), 8, 8), dtype=torch.float32)

        class _DummyEncoder:
            def __call__(self, model_input):
                return torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)

        embedder = VJEPA2HubEmbedder(
            VJEPA2Config(
                variant="vjepa2_vit_large",
                pretrained=False,
                crop_size=256,
                device="cpu",
                use_amp=False,
                pooling="flatten",
            )
        )
        embedder._ready = True
        embedder._torch = torch
        embedder._encoder = _DummyEncoder()
        embedder._processor = _SpyProcessor()
        embedder._device = torch.device("cpu")

        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        vector = embedder.embed([frame, frame])
        self.assertEqual(vector.shape[0], 8)


if __name__ == "__main__":
    unittest.main()
