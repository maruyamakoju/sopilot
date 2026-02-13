"""Tests for dtw_gpu.py (P0-3).

Covers:
- DtwAlignment dataclass
- CPU fallback when CuPy unavailable
- dtw_align_auto routing (prefer_gpu=True/False)
- is_gpu_available()
- get_gpu_info()
- _dtw_align_gpu raises RuntimeError without CuPy
"""

from __future__ import annotations

import numpy as np
import pytest

from sopilot.dtw_gpu import (
    CUPY_AVAILABLE,
    DtwAlignment,
    dtw_align_auto,
    get_gpu_info,
    is_gpu_available,
)


class TestDtwAlignment:
    """Test the DtwAlignment dataclass."""

    def test_creation(self):
        alignment = DtwAlignment(
            cost=10.5,
            mean_cost=2.1,
            path=[(0, 0, 0.9), (1, 1, 0.85)],
        )
        assert alignment.cost == 10.5
        assert alignment.mean_cost == 2.1
        assert len(alignment.path) == 2

    def test_empty_path(self):
        alignment = DtwAlignment(cost=0.0, mean_cost=0.0, path=[])
        assert alignment.path == []


class TestDtwAlignAuto:
    """Test dtw_align_auto with CPU fallback."""

    def _make_embeddings(self, m=5, n=8, d=16, seed=42):
        rng = np.random.RandomState(seed)
        gold = rng.randn(m, d).astype(np.float32)
        trainee = rng.randn(n, d).astype(np.float32)
        # L2-normalize
        gold = gold / (np.linalg.norm(gold, axis=1, keepdims=True) + 1e-9)
        trainee = trainee / (np.linalg.norm(trainee, axis=1, keepdims=True) + 1e-9)
        return gold, trainee

    def test_cpu_fallback(self):
        """dtw_align_auto with prefer_gpu=False always uses CPU path."""
        gold, trainee = self._make_embeddings()
        result = dtw_align_auto(gold, trainee, prefer_gpu=False)
        # CPU path returns AlignmentResult (step_engine), not DtwAlignment
        assert hasattr(result, "path")
        assert hasattr(result, "mean_cost")
        assert len(result.path) > 0

    def test_auto_with_prefer_gpu_true(self):
        """dtw_align_auto with prefer_gpu=True falls back to CPU if no CuPy/CUDA."""
        gold, trainee = self._make_embeddings()
        result = dtw_align_auto(gold, trainee, prefer_gpu=True)
        assert hasattr(result, "path")
        assert hasattr(result, "mean_cost")

    def test_path_boundaries(self):
        """Path should start near (0,0) and end near (m-1, n-1)."""
        gold, trainee = self._make_embeddings(m=4, n=6)
        result = dtw_align_auto(gold, trainee, prefer_gpu=False)
        if result.path:
            first = result.path[0]
            last = result.path[-1]
            assert first[0] == 0  # starts at gold[0]
            assert last[0] == 3  # ends at gold[m-1]
            assert last[1] == 5  # ends at trainee[n-1]

    def test_identical_sequences(self):
        """Identical sequences should have low cost."""
        gold, _ = self._make_embeddings(m=5, n=5)
        result = dtw_align_auto(gold, gold, prefer_gpu=False)
        # Cost should be near 0 for identical normalized sequences
        assert result.mean_cost < 0.1

    def test_similarity_in_path(self):
        """Path entries should include cosine similarity."""
        gold, trainee = self._make_embeddings(m=3, n=3)
        result = dtw_align_auto(gold, trainee, prefer_gpu=False)
        for i, j, sim in result.path:
            assert isinstance(sim, float)
            # Cosine similarity range: [-1, 1]
            assert -1.1 <= sim <= 1.1

    def test_deterministic(self):
        """Same inputs should give same outputs."""
        gold, trainee = self._make_embeddings()
        r1 = dtw_align_auto(gold, trainee, prefer_gpu=False)
        r2 = dtw_align_auto(gold, trainee, prefer_gpu=False)
        assert r1.mean_cost == r2.mean_cost
        assert r1.path == r2.path


class TestIsGpuAvailable:
    """Test GPU availability check."""

    def test_returns_bool(self):
        result = is_gpu_available()
        assert isinstance(result, bool)

    def test_consistent_with_cupy_flag(self):
        if not CUPY_AVAILABLE:
            assert is_gpu_available() is False


class TestGetGpuInfo:
    """Test GPU info retrieval."""

    def test_returns_dict(self):
        info = get_gpu_info()
        assert isinstance(info, dict)
        assert "available" in info

    def test_no_cupy_info(self):
        if not CUPY_AVAILABLE:
            info = get_gpu_info()
            assert info["available"] is False
            assert "reason" in info


class TestGpuDtwDirect:
    """Test _dtw_align_gpu directly."""

    def test_raises_without_cupy(self):
        if not CUPY_AVAILABLE:
            from sopilot.dtw_gpu import _dtw_align_gpu

            gold = np.random.randn(3, 8).astype(np.float32)
            trainee = np.random.randn(4, 8).astype(np.float32)
            with pytest.raises(RuntimeError, match="CuPy not available"):
                _dtw_align_gpu(gold, trainee)
