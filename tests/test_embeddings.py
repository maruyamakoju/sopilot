"""Tests for sopilot.embeddings — HeuristicClipEmbedder, VJepa2Embedder, AutoEmbedder."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sopilot.embeddings import (
    AutoEmbedder,
    HeuristicClipEmbedder,
    VJepa2Embedder,
    _l2_normalize,
    build_embedder,
)
from sopilot.video import ClipWindow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_clip(n_frames: int = 8, h: int = 32, w: int = 32) -> ClipWindow:
    """Create a synthetic ClipWindow with random uint8 frames."""
    frames = np.random.randint(0, 256, (n_frames, h, w, 3), dtype=np.uint8)
    return ClipWindow(clip_idx=0, start_sec=0.0, end_sec=1.0, frames=frames, quality_flags=[])


def _make_clip_const(n_frames: int = 4, value: int = 128, h: int = 32, w: int = 32) -> ClipWindow:
    """Create a constant-value clip (all pixels same color)."""
    frames = np.full((n_frames, h, w, 3), value, dtype=np.uint8)
    return ClipWindow(clip_idx=0, start_sec=0.0, end_sec=1.0, frames=frames, quality_flags=[])


# ---------------------------------------------------------------------------
# _l2_normalize
# ---------------------------------------------------------------------------


class TestL2Normalize:
    def test_normalizes_to_unit_length(self):
        v = np.array([3.0, 4.0], dtype=np.float64)
        result = _l2_normalize(v)
        assert result.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-6)

    def test_zero_vector_unchanged(self):
        v = np.zeros(5, dtype=np.float32)
        result = _l2_normalize(v)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, np.zeros(5, dtype=np.float32))

    def test_near_zero_vector_unchanged(self):
        v = np.full(3, 1e-14, dtype=np.float64)
        result = _l2_normalize(v)
        assert result.dtype == np.float32
        # norm ≈ 1.7e-14 < 1e-12 → skip normalization
        np.testing.assert_allclose(result, v.astype(np.float32))


# ---------------------------------------------------------------------------
# HeuristicClipEmbedder
# ---------------------------------------------------------------------------


class TestHeuristicClipEmbedder:
    def test_embed_clip_returns_float32(self):
        emb = HeuristicClipEmbedder()
        clip = _make_clip()
        vec = emb.embed_clip(clip.frames)
        assert vec.dtype == np.float32

    def test_embed_clip_output_is_normalized(self):
        emb = HeuristicClipEmbedder()
        clip = _make_clip()
        vec = emb.embed_clip(clip.frames)
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_embed_clip_single_frame(self):
        """Single frame → motion_stats should be zeros."""
        emb = HeuristicClipEmbedder()
        clip = _make_clip(n_frames=1)
        vec = emb.embed_clip(clip.frames)
        assert vec.dtype == np.float32
        assert vec.ndim == 1
        assert len(vec) > 0

    def test_embed_clip_all_black(self):
        """All-black frames → should not crash (zero norm edge case)."""
        emb = HeuristicClipEmbedder()
        clip = _make_clip_const(value=0)
        vec = emb.embed_clip(clip.frames)
        assert vec.dtype == np.float32
        assert vec.ndim == 1

    def test_embed_clip_all_white(self):
        emb = HeuristicClipEmbedder()
        clip = _make_clip_const(value=255)
        vec = emb.embed_clip(clip.frames)
        assert vec.dtype == np.float32

    def test_embed_clips_batch(self):
        emb = HeuristicClipEmbedder()
        clips = [_make_clip() for _ in range(3)]
        mat = emb.embed_clips(clips)
        assert mat.shape[0] == 3
        assert mat.dtype == np.float32
        # Each row should be unit-normalized
        norms = np.linalg.norm(mat, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_embed_clips_deterministic(self):
        """Same input → same output."""
        emb = HeuristicClipEmbedder()
        frames = np.random.randint(0, 256, (4, 32, 32, 3), dtype=np.uint8)
        clip = ClipWindow(clip_idx=0, start_sec=0.0, end_sec=1.0, frames=frames, quality_flags=[])
        v1 = emb.embed_clip(frames.copy())
        v2 = emb.embed_clip(frames.copy())
        np.testing.assert_array_equal(v1, v2)

    def test_different_clips_different_embeddings(self):
        emb = HeuristicClipEmbedder()
        clip_a = _make_clip_const(value=0)
        clip_b = _make_clip_const(value=255)
        va = emb.embed_clip(clip_a.frames)
        vb = emb.embed_clip(clip_b.frames)
        assert not np.allclose(va, vb), "Black and white clips should differ"

    def test_custom_hist_bins(self):
        emb = HeuristicClipEmbedder(hist_bins=4)
        clip = _make_clip()
        vec = emb.embed_clip(clip.frames)
        assert vec.dtype == np.float32
        # With 4 bins instead of 8, feature vector is shorter
        emb8 = HeuristicClipEmbedder(hist_bins=8)
        vec8 = emb8.embed_clip(clip.frames)
        assert len(vec) < len(vec8)

    def test_active_name(self):
        emb = HeuristicClipEmbedder()
        assert emb.active_name() == "heuristic-v1"

    def test_feature_vector_dimension(self):
        """Feature vector: 3+3+3+2+3+2+24+288+288 = 616...
        Actually: mean_rgb(3)+std_rgb(3)+temporal_delta(3)+gray_mean(2)+motion(3)+edge(2)+hist(24)+low_mean(432)+low_delta(432)
        low_size=(16,9) => 16*9*3=432 each
        Total = 3+3+3+2+3+2+24+432+432 = 904... let me check.
        Actually the feature concatenation is:
        mean_rgb(3), std_rgb(3), temporal_delta(3), gray_mean(2), motion_stats(3),
        edge_stats(2), color_hist(8*3=24), low_mean(16*9*3=432), low_delta(16*9*3=432)
        Total = 3+3+3+2+3+2+24+432+432 = 904
        """
        emb = HeuristicClipEmbedder()
        clip = _make_clip()
        vec = emb.embed_clip(clip.frames)
        expected_dim = 3 + 3 + 3 + 2 + 3 + 2 + (8 * 3) + (16 * 9 * 3) + (16 * 9 * 3)
        assert len(vec) == expected_dim


# ---------------------------------------------------------------------------
# VJepa2Embedder
# ---------------------------------------------------------------------------


def _make_vjepa2(**overrides) -> VJepa2Embedder:
    defaults = dict(
        repo="facebookresearch/vjepa2",
        source="hub",
        local_repo="",
        local_checkpoint="",
        variant="vjepa2_vit_large",
        pretrained=True,
        device="cpu",
        num_frames=16,
        image_size=224,
        batch_size=4,
    )
    defaults.update(overrides)
    return VJepa2Embedder(**defaults)


class TestVJepa2Embedder:
    def test_empty_clips_returns_0x0(self):
        emb = _make_vjepa2()
        result = emb.embed_clips([])
        assert result.shape == (0, 0)
        assert result.dtype == np.float32

    def test_active_name_pretrained(self):
        emb = _make_vjepa2(variant="vjepa2_vit_large", pretrained=True)
        assert emb.active_name() == "vjepa2:vjepa2_vit_large:pt"

    def test_active_name_scratch(self):
        emb = _make_vjepa2(variant="vjepa2_vit_large", pretrained=False)
        assert emb.active_name() == "vjepa2:vjepa2_vit_large:scratch"

    def test_batch_size_clamped(self):
        emb = _make_vjepa2(batch_size=0)
        assert emb.batch_size == 1

    def test_batch_size_negative_clamped(self):
        emb = _make_vjepa2(batch_size=-5)
        assert emb.batch_size == 1

    def test_source_stripped_and_lowered(self):
        emb = _make_vjepa2(source="  HUB  ")
        assert emb.source == "hub"

    def test_unsupported_source_raises(self):
        emb = _make_vjepa2(source="ftp")
        with pytest.raises(RuntimeError, match="unsupported V-JEPA2 source"):
            emb._load_model_if_needed()

    def test_local_source_requires_local_repo(self):
        emb = _make_vjepa2(source="local", local_repo="")
        with pytest.raises(RuntimeError, match="SOPILOT_VJEPA2_LOCAL_REPO is required"):
            emb._load_model_if_needed()

    def test_temporal_sampling_exact(self):
        emb = _make_vjepa2(num_frames=8)
        frames = np.random.randint(0, 256, (8, 32, 32, 3), dtype=np.uint8)
        result = emb._sample_temporal(frames)
        assert result.shape[0] == 8
        np.testing.assert_array_equal(result, frames)

    def test_temporal_sampling_undersample(self):
        emb = _make_vjepa2(num_frames=4)
        frames = np.random.randint(0, 256, (16, 32, 32, 3), dtype=np.uint8)
        result = emb._sample_temporal(frames)
        assert result.shape[0] == 4

    def test_temporal_sampling_oversample(self):
        emb = _make_vjepa2(num_frames=8)
        frames = np.random.randint(0, 256, (3, 32, 32, 3), dtype=np.uint8)
        result = emb._sample_temporal(frames)
        assert result.shape[0] == 8

    def test_temporal_sampling_single_frame(self):
        emb = _make_vjepa2(num_frames=4)
        frames = np.random.randint(0, 256, (1, 32, 32, 3), dtype=np.uint8)
        result = emb._sample_temporal(frames)
        assert result.shape[0] == 4
        # All frames should be the same (replicated)
        for i in range(4):
            np.testing.assert_array_equal(result[i], frames[0])

    def test_pool_encoder_output_3d(self):
        """3D tensor (B x Tokens x Dim) → mean over tokens → (B x Dim)."""
        import torch

        output = torch.randn(2, 10, 64)
        result = VJepa2Embedder._pool_encoder_output(output)
        assert result.shape == (2, 64)

    def test_pool_encoder_output_2d(self):
        import torch

        output = torch.randn(2, 64)
        result = VJepa2Embedder._pool_encoder_output(output)
        assert result.shape == (2, 64)

    def test_pool_encoder_output_4d(self):
        import torch

        output = torch.randn(2, 8, 8, 64)
        result = VJepa2Embedder._pool_encoder_output(output)
        assert result.shape == (2, 64)

    def test_pool_encoder_output_tuple(self):
        import torch

        output = (torch.randn(2, 10, 64), torch.randn(2, 10, 32))
        result = VJepa2Embedder._pool_encoder_output(output)
        assert result.shape == (2, 64)

    def test_pool_encoder_output_unsupported_shape(self):
        import torch

        output = torch.randn(2, 3, 4, 5, 6)  # 5D
        with pytest.raises(ValueError, match="unsupported"):
            VJepa2Embedder._pool_encoder_output(output)

    def test_pool_encoder_output_non_tensor_raises(self):
        with pytest.raises(ValueError, match="unexpected encoder output type"):
            VJepa2Embedder._pool_encoder_output("not a tensor")

    def test_clean_backbone_key(self):
        state_dict = {
            "module.backbone.layer1.weight": 1,
            "module.layer2.bias": 2,
            "backbone.layer3.weight": 3,
            "layer4.bias": 4,
        }
        cleaned = VJepa2Embedder._clean_backbone_key(state_dict)
        assert set(cleaned.keys()) == {"layer1.weight", "layer2.bias", "layer3.weight", "layer4.bias"}

    def test_resolve_device_cpu(self):
        import torch

        emb = _make_vjepa2(device="cpu")
        dev = emb._resolve_device(torch)
        assert dev == torch.device("cpu")

    def test_resolve_device_auto_no_cuda(self):
        import torch

        emb = _make_vjepa2(device="auto")
        with patch.object(torch.cuda, "is_available", return_value=False):
            dev = emb._resolve_device(torch)
        assert dev == torch.device("cpu")

    def test_load_error_retry(self):
        """After a load error, next embed_clips call should retry."""
        emb = _make_vjepa2(source="ftp")  # will fail
        with pytest.raises(RuntimeError):
            emb.embed_clips([_make_clip()])
        assert emb._load_error is not None
        # Second attempt should also try (not cached error)
        emb.source = "ftp"
        with pytest.raises(RuntimeError):
            emb.embed_clips([_make_clip()])


# ---------------------------------------------------------------------------
# AutoEmbedder
# ---------------------------------------------------------------------------


class TestAutoEmbedder:
    def _make_mock_embedder(self, name: str, fail: bool = False):
        emb = MagicMock()
        emb.name = name
        emb.active_name.return_value = name
        if fail:
            emb.embed_clips.side_effect = RuntimeError("mock failure")
        else:
            emb.embed_clips.return_value = np.ones((1, 4), dtype=np.float32)
        return emb

    def test_primary_success(self):
        primary = self._make_mock_embedder("primary")
        fallback = self._make_mock_embedder("fallback")
        auto = AutoEmbedder(primary, fallback)
        result = auto.embed_clips([_make_clip()])
        primary.embed_clips.assert_called_once()
        fallback.embed_clips.assert_not_called()
        assert auto.active_name() == "primary"

    def test_primary_failure_switches_to_fallback(self):
        primary = self._make_mock_embedder("primary", fail=True)
        fallback = self._make_mock_embedder("fallback")
        auto = AutoEmbedder(primary, fallback)
        result = auto.embed_clips([_make_clip()])
        assert auto._using_fallback
        assert auto.active_name() == "fallback"
        fallback.embed_clips.assert_called_once()

    def test_fallback_stays_until_retry_interval(self):
        primary = self._make_mock_embedder("primary", fail=True)
        fallback = self._make_mock_embedder("fallback")
        auto = AutoEmbedder(primary, fallback)
        auto.embed_clips([_make_clip()])  # triggers fallback

        # Second call should stay on fallback (interval not elapsed)
        primary.embed_clips.reset_mock()
        fallback.embed_clips.reset_mock()
        auto.embed_clips([_make_clip()])
        fallback.embed_clips.assert_called_once()
        # Primary should NOT be retried yet
        primary.embed_clips.assert_not_called()

    def test_recovery_after_retry_interval(self):
        primary = self._make_mock_embedder("primary", fail=True)
        fallback = self._make_mock_embedder("fallback")
        auto = AutoEmbedder(primary, fallback)
        auto.embed_clips([_make_clip()])  # triggers fallback

        # Fix primary
        primary.embed_clips.side_effect = None
        primary.embed_clips.return_value = np.ones((1, 4), dtype=np.float32)

        # Simulate time elapsed past retry interval
        auto._fallback_since = time.monotonic() - auto._RETRY_INTERVAL_SEC - 1
        result = auto.embed_clips([_make_clip()])
        assert not auto._using_fallback
        assert auto.active_name() == "primary"

    def test_recovery_attempt_fails_stays_on_fallback(self):
        primary = self._make_mock_embedder("primary", fail=True)
        fallback = self._make_mock_embedder("fallback")
        auto = AutoEmbedder(primary, fallback)
        auto.embed_clips([_make_clip()])  # triggers fallback

        # Simulate time elapsed
        auto._fallback_since = time.monotonic() - auto._RETRY_INTERVAL_SEC - 1
        result = auto.embed_clips([_make_clip()])
        assert auto._using_fallback
        assert auto.active_name() == "fallback"

    def test_name_includes_both(self):
        primary = self._make_mock_embedder("vjepa2")
        fallback = self._make_mock_embedder("heuristic-v1")
        auto = AutoEmbedder(primary, fallback)
        assert "vjepa2" in auto.name
        assert "heuristic-v1" in auto.name


# ---------------------------------------------------------------------------
# build_embedder
# ---------------------------------------------------------------------------


class TestBuildEmbedder:
    def test_heuristic_backend(self):
        from conftest import make_test_settings

        settings = make_test_settings(embedder_backend="heuristic")
        emb = build_embedder(settings)
        assert isinstance(emb, HeuristicClipEmbedder)

    def test_vjepa2_backend(self):
        from conftest import make_test_settings

        settings = make_test_settings(
            embedder_backend="vjepa2",
            vjepa2_repo="facebookresearch/vjepa2",
            vjepa2_variant="vjepa2_vit_large",
        )
        emb = build_embedder(settings)
        assert isinstance(emb, VJepa2Embedder)

    def test_auto_with_fallback(self):
        from conftest import make_test_settings

        settings = make_test_settings(
            embedder_backend="auto",
            embedder_fallback_enabled=True,
            vjepa2_repo="facebookresearch/vjepa2",
            vjepa2_variant="vjepa2_vit_large",
        )
        emb = build_embedder(settings)
        assert isinstance(emb, AutoEmbedder)

    def test_auto_without_fallback(self):
        from conftest import make_test_settings

        settings = make_test_settings(
            embedder_backend="auto",
            embedder_fallback_enabled=False,
            vjepa2_repo="facebookresearch/vjepa2",
            vjepa2_variant="vjepa2_vit_large",
        )
        emb = build_embedder(settings)
        assert isinstance(emb, VJepa2Embedder)

    def test_unknown_backend_raises(self):
        from conftest import make_test_settings

        settings = make_test_settings(embedder_backend="unknown_backend")
        with pytest.raises(ValueError, match="unknown embedder backend"):
            build_embedder(settings)
