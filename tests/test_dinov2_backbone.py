"""Tests for DINOv2Backbone and NeuralReIDExtractor.

Covers:
  DINOv2Backbone (sopilot/perception/dinov2_backbone.py):
  - __init__ standalone and with orchestrator
  - Registration with orchestrator (_register_with_orchestrator)
  - extract_features() returns {"cls": (768,), "patches": (N, 768)}
  - extract_cls() returns (768,)
  - extract_reid_embedding() returns L2-normalized (768,)
  - extract_reid_embedding() with None/empty input returns zero vector
  - classify_scene() without prototypes returns uniform distribution
  - classify_scene() with prototypes returns softmax scores
  - update_scene_prototype() first set and EMA update
  - update_feature_stats() running statistics
  - anomaly_score() returns 0 without baseline
  - anomaly_score() returns Mahalanobis distance with baseline
  - get_attention_map() returns 2D array
  - get_attention_map() fallback on failure
  - close() releases standalone model
  - close() with orchestrator is no-op
  - get_status() standalone
  - get_status() with orchestrator
  - embedding_dim property
  - feature_count property
  - _recompute_cov_inv with insufficient data

  NeuralReIDExtractor (sopilot/perception/neural_reid.py):
  - embed() with valid crop returns (768,) L2-normalized
  - embed() with None crop returns None
  - embed() with empty crop returns None
  - embed() with too-small crop returns None and increments fallback
  - embed() with backbone failure returns None
  - embed() with zero-vector backbone returns None
  - fallback count tracking
  - embed_count tracking
  - get_stats() structure and values
  - embedding_dim property
  - min_crop_size filtering at boundary
  - Integration: NeuralReIDExtractor as embedder for ReIDExtractor
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from typing import Any

import numpy as np
import pytest

from sopilot.perception.dinov2_backbone import (
    DINOv2Backbone,
    _EMBEDDING_DIM,
    _INPUT_SIZE,
    _PATCH_SIZE,
    _SCENE_LABELS,
)
from sopilot.perception.neural_reid import NeuralReIDExtractor


# ---------------------------------------------------------------------------
# Helpers / Fake objects
# ---------------------------------------------------------------------------

_N_PATCHES = (_INPUT_SIZE // _PATCH_SIZE) ** 2  # 37 * 37 = 1369


class _FakeTensor:
    """Minimal tensor-like object supporting .cpu().numpy(), .to(), .unsqueeze(),
    indexing, .device, .parameters(), etc."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._arr

    def astype(self, dtype: Any) -> np.ndarray:
        return self._arr.astype(dtype)

    def to(self, device: Any) -> "_FakeTensor":
        return self

    def unsqueeze(self, dim: int) -> "_FakeTensor":
        return _FakeTensor(np.expand_dims(self._arr, axis=dim))

    def __getitem__(self, key: Any) -> "_FakeTensor":
        result = self._arr[key]
        if isinstance(result, np.ndarray):
            return _FakeTensor(result)
        return result

    @property
    def shape(self) -> tuple:
        return self._arr.shape

    @property
    def ndim(self) -> int:
        return self._arr.ndim

    @property
    def device(self) -> str:
        return "cpu"

    def mean(self, dim: int = 0) -> "_FakeTensor":
        return _FakeTensor(self._arr.mean(axis=dim))


class _FakeModel:
    """Fake DINOv2 model returning predictable features."""

    def __init__(self) -> None:
        rng = np.random.RandomState(42)
        # Pre-generate a consistent CLS token and patch tokens
        self._cls = rng.randn(_EMBEDDING_DIM).astype(np.float32)
        self._patches = rng.randn(_N_PATCHES, _EMBEDDING_DIM).astype(np.float32)

    def parameters(self):
        """Yield a fake parameter so next(model.parameters()).device works."""
        fake_param = MagicMock()
        fake_param.device = "cpu"
        yield fake_param

    def forward_features(self, batch: Any) -> dict:
        """Return a dict matching DINOv2 output format."""
        cls = _FakeTensor(self._cls[np.newaxis, :])   # (1, 768)
        patches = _FakeTensor(self._patches[np.newaxis, :, :])  # (1, N, 768)
        return {
            "x_norm_clstoken": cls,
            "x_norm_patchtokens": patches,
        }

    def __call__(self, batch: Any) -> _FakeTensor:
        return _FakeTensor(np.zeros((1, _EMBEDDING_DIM)))

    def to(self, device: Any) -> "_FakeModel":
        return self

    def eval(self) -> "_FakeModel":
        return self

    def cpu(self) -> "_FakeModel":
        return self


def _make_backbone(orchestrator: Any = None) -> DINOv2Backbone:
    """Create a DINOv2Backbone wired to a fake model, skipping torch imports."""
    bb = DINOv2Backbone.__new__(DINOv2Backbone)
    import threading
    bb._orchestrator = orchestrator
    bb._variant = "dinov2_vitb14"
    bb._device_override = None
    bb._lock = threading.RLock()
    bb._model = _FakeModel()
    bb._loaded = True
    bb._transform = _fake_transform
    bb._feature_mean = None
    bb._feature_cov_inv = None
    bb._feature_count = 0
    bb._feature_sum = np.zeros(_EMBEDDING_DIM, dtype=np.float64)
    bb._feature_sq_sum = np.zeros((_EMBEDDING_DIM, _EMBEDDING_DIM), dtype=np.float64)
    bb._scene_prototypes = None
    return bb


def _fake_transform(image: np.ndarray) -> _FakeTensor:
    """Dummy transform returning a (3, 518, 518) tensor."""
    return _FakeTensor(np.zeros((3, _INPUT_SIZE, _INPUT_SIZE), dtype=np.float32))


def _dummy_image(h: int = 480, w: int = 640) -> np.ndarray:
    """Create a dummy BGR image."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# DINOv2Backbone — Initialization
# ---------------------------------------------------------------------------

class TestDINOv2BackboneInit:
    """Tests for DINOv2Backbone construction and configuration."""

    def test_standalone_init(self) -> None:
        """Standalone mode: no orchestrator, model not loaded until needed."""
        bb = _make_backbone(orchestrator=None)
        assert bb._orchestrator is None
        assert bb._variant == "dinov2_vitb14"

    def test_orchestrator_init_calls_register(self) -> None:
        """When orchestrator is provided, _register_with_orchestrator is called."""
        orch = MagicMock()
        with patch.object(DINOv2Backbone, "_load_model"), \
             patch.object(DINOv2Backbone, "_build_transform"):
            bb = DINOv2Backbone(orchestrator=orch)
        orch.register.assert_called_once()
        call_kwargs = orch.register.call_args
        assert call_kwargs[1]["name"] == "dinov2" or call_kwargs.kwargs["name"] == "dinov2"

    def test_register_passes_correct_params(self) -> None:
        """Registration includes vram_estimate_gb and priority."""
        orch = MagicMock()
        with patch.object(DINOv2Backbone, "_load_model"), \
             patch.object(DINOv2Backbone, "_build_transform"):
            DINOv2Backbone(orchestrator=orch)
        kwargs = orch.register.call_args.kwargs
        assert kwargs["vram_estimate_gb"] == 0.35
        assert kwargs["priority"] == 9

    def test_embedding_dim_property(self) -> None:
        bb = _make_backbone()
        assert bb.embedding_dim == 768

    def test_initial_feature_count_zero(self) -> None:
        bb = _make_backbone()
        assert bb.feature_count == 0


# ---------------------------------------------------------------------------
# DINOv2Backbone — Feature Extraction
# ---------------------------------------------------------------------------

class TestDINOv2FeatureExtraction:
    """Tests for extract_features, extract_cls, extract_reid_embedding."""

    def test_extract_features_returns_dict(self) -> None:
        bb = _make_backbone()
        result = bb.extract_features(_dummy_image())
        assert isinstance(result, dict)
        assert "cls" in result
        assert "patches" in result

    def test_extract_features_cls_shape(self) -> None:
        bb = _make_backbone()
        result = bb.extract_features(_dummy_image())
        assert result["cls"].shape == (_EMBEDDING_DIM,)

    def test_extract_features_patches_shape(self) -> None:
        bb = _make_backbone()
        result = bb.extract_features(_dummy_image())
        assert result["patches"].shape == (_N_PATCHES, _EMBEDDING_DIM)

    def test_extract_features_dtype(self) -> None:
        bb = _make_backbone()
        result = bb.extract_features(_dummy_image())
        assert result["cls"].dtype == np.float32
        assert result["patches"].dtype == np.float32

    def test_extract_cls_returns_1d(self) -> None:
        bb = _make_backbone()
        cls = bb.extract_cls(_dummy_image())
        assert cls.ndim == 1
        assert cls.shape == (_EMBEDDING_DIM,)

    def test_extract_cls_equals_features_cls(self) -> None:
        bb = _make_backbone()
        img = _dummy_image()
        cls = bb.extract_cls(img)
        features = bb.extract_features(img)
        np.testing.assert_array_equal(cls, features["cls"])

    def test_extract_features_tensor_fallback(self) -> None:
        """When model returns a plain tensor, first token is CLS, rest are patches."""
        bb = _make_backbone()
        rng = np.random.RandomState(99)
        total_tokens = 1 + _N_PATCHES
        all_tokens = rng.randn(1, total_tokens, _EMBEDDING_DIM).astype(np.float32)

        bb._model.forward_features = lambda batch: _FakeTensor(all_tokens)
        result = bb.extract_features(_dummy_image())
        np.testing.assert_array_almost_equal(result["cls"], all_tokens[0, 0])
        np.testing.assert_array_almost_equal(result["patches"], all_tokens[0, 1:])


# ---------------------------------------------------------------------------
# DINOv2Backbone — Re-ID Embedding
# ---------------------------------------------------------------------------

class TestDINOv2ReIDEmbedding:
    """Tests for extract_reid_embedding."""

    def test_reid_embedding_shape(self) -> None:
        bb = _make_backbone()
        emb = bb.extract_reid_embedding(_dummy_image(100, 80))
        assert emb.shape == (_EMBEDDING_DIM,)

    def test_reid_embedding_l2_normalized(self) -> None:
        bb = _make_backbone()
        emb = bb.extract_reid_embedding(_dummy_image(100, 80))
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-5, f"Expected L2 norm ~1.0, got {norm}"

    def test_reid_embedding_none_input(self) -> None:
        bb = _make_backbone()
        emb = bb.extract_reid_embedding(None)
        np.testing.assert_array_equal(emb, np.zeros(_EMBEDDING_DIM, dtype=np.float32))

    def test_reid_embedding_empty_input(self) -> None:
        bb = _make_backbone()
        emb = bb.extract_reid_embedding(np.array([]))
        np.testing.assert_array_equal(emb, np.zeros(_EMBEDDING_DIM, dtype=np.float32))

    def test_reid_embedding_exception_returns_zero(self) -> None:
        """If internal extraction fails, return zero vector."""
        bb = _make_backbone()
        bb._model.forward_features = MagicMock(side_effect=RuntimeError("boom"))
        emb = bb.extract_reid_embedding(_dummy_image(100, 80))
        np.testing.assert_array_equal(emb, np.zeros(_EMBEDDING_DIM, dtype=np.float32))


# ---------------------------------------------------------------------------
# DINOv2Backbone — Scene Classification
# ---------------------------------------------------------------------------

class TestDINOv2SceneClassification:
    """Tests for classify_scene and update_scene_prototype."""

    def test_classify_scene_no_prototypes_uniform(self) -> None:
        """Without prototypes, all labels get 1/N."""
        bb = _make_backbone()
        result = bb.classify_scene(_dummy_image())
        assert set(result.keys()) == set(_SCENE_LABELS)
        expected = 1.0 / len(_SCENE_LABELS)
        for score in result.values():
            assert abs(score - expected) < 1e-6

    def test_classify_scene_with_prototypes(self) -> None:
        """With prototypes, returns softmax scores summing to ~1."""
        bb = _make_backbone()
        # Set prototypes: factory_floor aligned with CLS, others random
        rng = np.random.RandomState(7)
        for label in _SCENE_LABELS:
            bb.update_scene_prototype(label, rng.randn(_EMBEDDING_DIM).astype(np.float32))
        result = bb.classify_scene(_dummy_image())
        assert set(result.keys()) == set(_SCENE_LABELS)
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-5, f"Softmax sum should be ~1, got {total}"

    def test_classify_scene_scores_positive(self) -> None:
        """All softmax scores must be > 0."""
        bb = _make_backbone()
        rng = np.random.RandomState(8)
        for label in _SCENE_LABELS:
            bb.update_scene_prototype(label, rng.randn(_EMBEDDING_DIM).astype(np.float32))
        result = bb.classify_scene(_dummy_image())
        for label, score in result.items():
            assert score > 0, f"{label} score should be > 0, got {score}"

    def test_update_scene_prototype_first_set(self) -> None:
        """First update sets the prototype directly (copy)."""
        bb = _make_backbone()
        feat = np.ones(_EMBEDDING_DIM, dtype=np.float32) * 3.0
        bb.update_scene_prototype("factory_floor", feat)
        assert bb._scene_prototypes is not None
        np.testing.assert_array_almost_equal(bb._scene_prototypes["factory_floor"], feat)

    def test_update_scene_prototype_ema(self) -> None:
        """Subsequent updates use EMA with alpha=0.1."""
        bb = _make_backbone()
        v1 = np.ones(_EMBEDDING_DIM, dtype=np.float32) * 10.0
        v2 = np.ones(_EMBEDDING_DIM, dtype=np.float32) * 20.0
        bb.update_scene_prototype("corridor", v1)
        bb.update_scene_prototype("corridor", v2)
        # Expected: (1 - 0.1) * 10.0 + 0.1 * 20.0 = 11.0
        expected = 0.9 * 10.0 + 0.1 * 20.0
        np.testing.assert_array_almost_equal(
            bb._scene_prototypes["corridor"],
            np.full(_EMBEDDING_DIM, expected, dtype=np.float32),
            decimal=5,
        )

    def test_update_scene_prototype_independent_labels(self) -> None:
        """Updating one label does not affect another."""
        bb = _make_backbone()
        v1 = np.ones(_EMBEDDING_DIM, dtype=np.float32)
        v2 = np.ones(_EMBEDDING_DIM, dtype=np.float32) * 5.0
        bb.update_scene_prototype("office", v1)
        bb.update_scene_prototype("warehouse", v2)
        np.testing.assert_array_almost_equal(bb._scene_prototypes["office"], v1)
        np.testing.assert_array_almost_equal(bb._scene_prototypes["warehouse"], v2)


# ---------------------------------------------------------------------------
# DINOv2Backbone — Anomaly Detection
# ---------------------------------------------------------------------------

class TestDINOv2AnomalyDetection:
    """Tests for update_feature_stats and anomaly_score."""

    def test_anomaly_score_no_baseline(self) -> None:
        """Without any statistics, anomaly_score returns 0.0."""
        bb = _make_backbone()
        score = bb.anomaly_score(_dummy_image())
        assert score == 0.0

    def test_update_feature_stats_increments_count(self) -> None:
        bb = _make_backbone()
        feat = np.random.randn(_EMBEDDING_DIM).astype(np.float32)
        bb.update_feature_stats(feat)
        assert bb.feature_count == 1
        bb.update_feature_stats(feat)
        assert bb.feature_count == 2

    def test_update_feature_stats_accumulates_sum(self) -> None:
        bb = _make_backbone()
        feat = np.ones(_EMBEDDING_DIM, dtype=np.float32) * 2.0
        bb.update_feature_stats(feat)
        np.testing.assert_array_almost_equal(bb._feature_sum, feat.astype(np.float64))

    def test_recompute_cov_inv_at_100(self) -> None:
        """Inverse covariance is computed at count=100."""
        bb = _make_backbone()
        rng = np.random.RandomState(0)
        for _ in range(100):
            bb.update_feature_stats(rng.randn(_EMBEDDING_DIM).astype(np.float32))
        assert bb._feature_mean is not None
        assert bb._feature_cov_inv is not None
        assert bb._feature_cov_inv.shape == (_EMBEDDING_DIM, _EMBEDDING_DIM)

    def test_anomaly_score_with_baseline(self) -> None:
        """With baseline, anomaly_score returns a positive float."""
        bb = _make_backbone()
        rng = np.random.RandomState(1)
        for _ in range(100):
            bb.update_feature_stats(rng.randn(_EMBEDDING_DIM).astype(np.float32))
        score = bb.anomaly_score(_dummy_image())
        assert isinstance(score, float)
        assert score >= 0.0

    def test_recompute_cov_inv_too_few_samples(self) -> None:
        """_recompute_cov_inv does nothing if count < 2."""
        bb = _make_backbone()
        bb._feature_count = 1
        bb._recompute_cov_inv()
        assert bb._feature_mean is None
        assert bb._feature_cov_inv is None


# ---------------------------------------------------------------------------
# DINOv2Backbone — Attention Map
# ---------------------------------------------------------------------------

class TestDINOv2AttentionMap:
    """Tests for get_attention_map."""

    def test_attention_map_fallback_shape(self) -> None:
        """When attention extraction yields None, returns uniform map."""
        bb = _make_backbone()
        # Ensure _extract_attention returns None (model has no .blocks)
        with patch.object(DINOv2Backbone, "_extract_attention", return_value=None):
            attn = bb.get_attention_map(_dummy_image())
        n_patches_side = _INPUT_SIZE // _PATCH_SIZE
        assert attn.shape == (n_patches_side, n_patches_side)
        assert attn.dtype == np.float32

    def test_attention_map_fallback_uniform(self) -> None:
        """Fallback map should be uniform (each cell = 1/total)."""
        bb = _make_backbone()
        with patch.object(DINOv2Backbone, "_extract_attention", return_value=None):
            attn = bb.get_attention_map(_dummy_image())
        n_side = _INPUT_SIZE // _PATCH_SIZE
        expected_val = 1.0 / (n_side * n_side)
        np.testing.assert_array_almost_equal(attn, expected_val)

    def test_attention_map_with_attention(self) -> None:
        """When attention is available, returns normalized 2D map."""
        bb = _make_backbone()
        n_side = _INPUT_SIZE // _PATCH_SIZE
        n_patches = n_side * n_side
        # Simulate attention output: (1, num_heads, N+1, N+1)
        rng = np.random.RandomState(5)
        attn_data = rng.rand(1, 6, n_patches + 1, n_patches + 1).astype(np.float32)
        fake_attn = _FakeTensor(attn_data)

        with patch.object(DINOv2Backbone, "_extract_attention", return_value=fake_attn):
            attn = bb.get_attention_map(_dummy_image())
        assert attn.ndim == 2
        assert attn.shape == (n_side, n_side)
        # Should be normalized to [0, 1]
        assert attn.min() >= 0.0 - 1e-6
        assert attn.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# DINOv2Backbone — Lifecycle / Status
# ---------------------------------------------------------------------------

class TestDINOv2Lifecycle:
    """Tests for close() and get_status()."""

    def test_close_standalone_releases_model(self) -> None:
        bb = _make_backbone()
        assert bb._loaded is True
        bb.close()
        assert bb._loaded is False
        assert bb._model is None

    def test_close_with_orchestrator_is_noop(self) -> None:
        """close() with orchestrator should not touch _model."""
        orch = MagicMock()
        bb = _make_backbone(orchestrator=orch)
        bb.close()
        # Model still there — orchestrator manages lifecycle
        assert bb._model is not None

    def test_get_status_standalone(self) -> None:
        bb = _make_backbone()
        status = bb.get_status()
        assert status["variant"] == "dinov2_vitb14"
        assert status["embedding_dim"] == _EMBEDDING_DIM
        assert status["loaded"] is True
        assert status["feature_count"] == 0
        assert status["has_anomaly_baseline"] is False
        assert status["scene_prototypes"] == []

    def test_get_status_with_orchestrator(self) -> None:
        orch = MagicMock()
        orch.is_loaded.return_value = False
        bb = _make_backbone(orchestrator=orch)
        status = bb.get_status()
        assert status["loaded"] is False
        orch.is_loaded.assert_called_with("dinov2")

    def test_get_status_with_scene_prototypes(self) -> None:
        bb = _make_backbone()
        bb.update_scene_prototype("office", np.ones(_EMBEDDING_DIM))
        bb.update_scene_prototype("corridor", np.ones(_EMBEDDING_DIM))
        status = bb.get_status()
        assert sorted(status["scene_prototypes"]) == ["corridor", "office"]

    def test_get_status_has_anomaly_baseline_after_stats(self) -> None:
        bb = _make_backbone()
        rng = np.random.RandomState(2)
        for _ in range(100):
            bb.update_feature_stats(rng.randn(_EMBEDDING_DIM).astype(np.float32))
        status = bb.get_status()
        assert status["has_anomaly_baseline"] is True
        assert status["feature_count"] == 100

    def test_ensure_loaded_standalone(self) -> None:
        """Standalone mode: _ensure_loaded returns the model directly."""
        bb = _make_backbone()
        model = bb._ensure_loaded()
        assert model is bb._model

    def test_ensure_loaded_with_orchestrator_acquires(self) -> None:
        orch = MagicMock()
        orch.acquire.return_value = _FakeModel()
        bb = _make_backbone(orchestrator=orch)
        model = bb._ensure_loaded()
        orch.acquire.assert_called_once_with("dinov2")

    def test_release_with_orchestrator(self) -> None:
        orch = MagicMock()
        bb = _make_backbone(orchestrator=orch)
        bb._release()
        orch.release.assert_called_once_with("dinov2")

    def test_release_standalone_noop(self) -> None:
        """Standalone release does nothing (no exception)."""
        bb = _make_backbone()
        bb._release()  # should not raise


# ---------------------------------------------------------------------------
# NeuralReIDExtractor
# ---------------------------------------------------------------------------

class TestNeuralReIDExtractor:
    """Tests for NeuralReIDExtractor."""

    def _make_extractor(
        self, min_crop_size: int = 32
    ) -> tuple[NeuralReIDExtractor, DINOv2Backbone]:
        bb = _make_backbone()
        ext = NeuralReIDExtractor(backbone=bb, min_crop_size=min_crop_size)
        return ext, bb

    def test_embed_valid_crop(self) -> None:
        ext, _ = self._make_extractor()
        crop = _dummy_image(100, 80)
        emb = ext.embed(crop)
        assert emb is not None
        assert emb.shape == (_EMBEDDING_DIM,)

    def test_embed_result_l2_normalized(self) -> None:
        ext, _ = self._make_extractor()
        emb = ext.embed(_dummy_image(100, 80))
        assert emb is not None
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-5

    def test_embed_none_crop(self) -> None:
        ext, _ = self._make_extractor()
        assert ext.embed(None) is None

    def test_embed_empty_crop(self) -> None:
        ext, _ = self._make_extractor()
        assert ext.embed(np.array([])) is None

    def test_embed_too_small_height(self) -> None:
        ext, _ = self._make_extractor(min_crop_size=32)
        crop = _dummy_image(20, 100)  # height < 32
        assert ext.embed(crop) is None

    def test_embed_too_small_width(self) -> None:
        ext, _ = self._make_extractor(min_crop_size=32)
        crop = _dummy_image(100, 20)  # width < 32
        assert ext.embed(crop) is None

    def test_embed_exactly_min_size(self) -> None:
        """Crop exactly at min_crop_size should be accepted."""
        ext, _ = self._make_extractor(min_crop_size=32)
        crop = _dummy_image(32, 32)
        emb = ext.embed(crop)
        assert emb is not None
        assert emb.shape == (_EMBEDDING_DIM,)

    def test_embed_backbone_exception_returns_none(self) -> None:
        ext, bb = self._make_extractor()
        bb._model.forward_features = MagicMock(side_effect=RuntimeError("GPU OOM"))
        result = ext.embed(_dummy_image(100, 80))
        assert result is None

    def test_embed_backbone_zero_vector_returns_none(self) -> None:
        """If backbone returns zero vector, embed returns None."""
        ext, bb = self._make_extractor()
        # Make extract_reid_embedding return zeros
        bb.extract_reid_embedding = MagicMock(
            return_value=np.zeros(_EMBEDDING_DIM, dtype=np.float32)
        )
        result = ext.embed(_dummy_image(100, 80))
        assert result is None

    def test_fallback_count_on_small_crop(self) -> None:
        ext, _ = self._make_extractor(min_crop_size=50)
        ext.embed(_dummy_image(30, 30))
        ext.embed(_dummy_image(40, 40))
        assert ext._fallback_count == 2

    def test_embed_count_on_success(self) -> None:
        ext, _ = self._make_extractor()
        ext.embed(_dummy_image(100, 80))
        ext.embed(_dummy_image(200, 150))
        assert ext._embed_count == 2

    def test_fallback_count_on_exception(self) -> None:
        ext, bb = self._make_extractor()
        bb._model.forward_features = MagicMock(side_effect=RuntimeError("fail"))
        ext.embed(_dummy_image(100, 80))
        assert ext._fallback_count == 1
        assert ext._embed_count == 0

    def test_embedding_dim_property(self) -> None:
        ext, _ = self._make_extractor()
        assert ext.embedding_dim == 768

    def test_get_stats_initial(self) -> None:
        ext, _ = self._make_extractor()
        stats = ext.get_stats()
        assert stats["embedding_dim"] == 768
        assert stats["total_extractions"] == 0
        assert stats["neural_embeddings"] == 0
        assert stats["fallbacks"] == 0
        assert stats["neural_rate"] == 0.0

    def test_get_stats_after_mixed(self) -> None:
        ext, bb = self._make_extractor(min_crop_size=32)
        # 2 successful
        ext.embed(_dummy_image(100, 100))
        ext.embed(_dummy_image(80, 80))
        # 1 fallback (too small)
        ext.embed(_dummy_image(20, 20))
        stats = ext.get_stats()
        assert stats["total_extractions"] == 3
        assert stats["neural_embeddings"] == 2
        assert stats["fallbacks"] == 1
        assert abs(stats["neural_rate"] - 2 / 3) < 1e-3


# ---------------------------------------------------------------------------
# Integration: NeuralReIDExtractor as embedder for ReIDExtractor
# ---------------------------------------------------------------------------

class TestNeuralReIDIntegration:
    """Test NeuralReIDExtractor plugged into ReIDExtractor as embedder."""

    def test_reid_extractor_uses_neural_embedder(self) -> None:
        """ReIDExtractor.extract() should delegate to NeuralReIDExtractor."""
        from sopilot.vigil.reid_extractor import ReIDExtractor

        bb = _make_backbone()
        neural = NeuralReIDExtractor(backbone=bb, min_crop_size=10)
        extractor = ReIDExtractor(embedder=neural)

        # Create a frame and a fake bbox
        frame = _dummy_image(480, 640)
        bbox = MagicMock()
        bbox.x1, bbox.y1, bbox.x2, bbox.y2 = 0.1, 0.1, 0.5, 0.5

        emb = extractor.extract(frame, bbox)
        assert emb.ndim == 1
        assert emb.shape[0] > 0
        # Should have used neural embedder (768-dim, L2-normalized)
        assert emb.shape[0] == _EMBEDDING_DIM
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-5

    def test_reid_extractor_fallback_to_histogram(self) -> None:
        """When neural embedder returns None, ReIDExtractor falls back to histogram."""
        from sopilot.vigil.reid_extractor import ReIDExtractor

        bb = _make_backbone()
        neural = NeuralReIDExtractor(backbone=bb, min_crop_size=9999)  # always too small
        extractor = ReIDExtractor(embedder=neural)

        frame = _dummy_image(480, 640)
        bbox = MagicMock()
        bbox.x1, bbox.y1, bbox.x2, bbox.y2 = 0.1, 0.1, 0.5, 0.5

        emb = extractor.extract(frame, bbox)
        assert emb.ndim == 1
        # Histogram fallback yields 512-dim
        assert emb.shape[0] == 512


# ---------------------------------------------------------------------------
# DINOv2Backbone — Constants
# ---------------------------------------------------------------------------

class TestDINOv2Constants:
    """Verify module-level constants are correct."""

    def test_embedding_dim(self) -> None:
        assert _EMBEDDING_DIM == 768

    def test_input_size(self) -> None:
        assert _INPUT_SIZE == 518

    def test_patch_size(self) -> None:
        assert _PATCH_SIZE == 14

    def test_input_size_divisible_by_patch_size(self) -> None:
        assert _INPUT_SIZE % _PATCH_SIZE == 0

    def test_scene_labels(self) -> None:
        expected = [
            "factory_floor", "warehouse", "corridor", "office",
            "outdoor", "parking_lot", "loading_dock", "construction_site",
        ]
        assert _SCENE_LABELS == expected

    def test_scene_labels_count(self) -> None:
        assert len(_SCENE_LABELS) == 8
