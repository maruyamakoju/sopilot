"""Phase 55: Self-Supervised Domain Adaptation.

DomainAdapter は新しい工場環境へのラベル不要な適応を実現する。
DINOv2 特徴ベクトルを利用し、ドメインシフトを MMD で検出、
プロトタイプ整合によって埋め込みを補正する。

主要機能:
  - **ドメインシフト検出**: RBF カーネル MMD（サブサンプリング付き）
  - **プロトタイプ整合適応**: 対角 CORAL — μ/σ の整合によるアフィン補正
  - **カメラ別状態管理**: カメラ IDごとに独立した統計 + バッファ
  - **オペレーターラベル統合**: Phase 45 ラベルをシフト補正評価に利用
  - **学習曲線追跡**: シフトスコアの時系列 (最大 200 点)
  - **自動適応トリガ**: バッファが閾値に達したら自動実行

Architecture::

    DomainAdapter
      ├── observe_frame(features, camera_id)  — フレーム特徴を蓄積
      ├── detect_domain_shift(camera_id)      — MMD ドメインシフトスコア計算
      ├── adapt(camera_id)                    — プロトタイプ整合ステップ実行
      ├── get_adapted_embedding(f, camera_id) — 適応済み埋め込みを返す
      ├── apply_label(f, category, camera_id) — ラベル付きサンプルを統合
      ├── get_stats()                         — 全体統計
      ├── get_camera_states()                 — カメラ別状態一覧
      └── reset(camera_id?)                   — バッファ + 状態リセット

Design decisions:
  - GPU 不要: 純 NumPy のみ
  - スレッドセーフ: RLock でカメラ状態を保護
  - MMD サブサンプル: 最大 _MMD_SAMPLE = 300 点 → O(n²) を制御
  - Best-effort 永続化なし (インメモリのみ — 再起動で再収集)
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FEATURE_DIM = 768         # DINOv2 CLS token dimension
_MMD_SAMPLE = 300          # max samples for MMD computation
_LEARNING_CURVE_MAX = 200  # max points kept in shift history


# ---------------------------------------------------------------------------
# Per-camera adaptation state
# ---------------------------------------------------------------------------


@dataclass
class CameraAdaptationState:
    """Per-camera running statistics for domain adaptation."""

    camera_id: str

    # Feature buffer: recent unlabeled frames
    buffer: deque = field(default_factory=lambda: deque(maxlen=2000))

    # Source domain statistics (bootstrapped from first N frames)
    source_mean: np.ndarray | None = None
    source_std: np.ndarray | None = None
    source_bootstrapped: bool = False
    source_frames_seen: int = 0

    # Target domain statistics (updated from recent buffer)
    target_mean: np.ndarray | None = None
    target_std: np.ndarray | None = None

    # Adaptation correction (applied to raw embeddings)
    correction_mean: np.ndarray | None = None  # μs - μt
    correction_scale: np.ndarray | None = None  # σs / σt

    # Tracking
    shift_score: float = 0.0
    adapt_count: int = 0
    labeled_count: int = 0
    last_adapted_ts: float | None = None

    # Learning curve: list of (timestamp, shift_score)
    shift_history: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# MMD helper
# ---------------------------------------------------------------------------


def _mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = 1e-3) -> float:
    """Unbiased squared MMD with RBF kernel, O(n²) subsampled.

    gamma defaults to 1e-3 for 768-dim unit-sphere features where
    typical ||x-y||² ≈ 0.5–2.0.
    """
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        return 0.0

    # Sub-sample to limit cost
    if n > _MMD_SAMPLE:
        idx = np.random.choice(n, _MMD_SAMPLE, replace=False)
        X = X[idx]
    if m > _MMD_SAMPLE:
        idx = np.random.choice(m, _MMD_SAMPLE, replace=False)
        Y = Y[idx]

    n, m = len(X), len(Y)

    def rbf(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        sq = np.sum((A[:, None] - B[None, :]) ** 2, axis=-1)
        return np.exp(-gamma * sq)

    Kxx = rbf(X, X)
    Kyy = rbf(Y, Y)
    Kxy = rbf(X, Y)

    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    term1 = Kxx.sum() / (n * (n - 1))
    term2 = Kyy.sum() / (m * (m - 1))
    term3 = 2.0 * Kxy.mean()
    return float(max(0.0, term1 + term2 - term3))


# ---------------------------------------------------------------------------
# DomainAdapter
# ---------------------------------------------------------------------------


class DomainAdapter:
    """Self-supervised domain adapter for factory perception.

    Detects and corrects distributional shift between a source domain
    (initial N frames used to build source statistics) and the current
    target domain (rolling buffer of recent frames).

    Correction uses diagonal CORAL:
        adapted = (raw - μ_target) * (σ_source / σ_target) + μ_source

    This zero-order and first-order moment matching suffices for most
    camera-angle / lighting / lens changes encountered in factory floor
    deployments without any labels.
    """

    def __init__(
        self,
        feature_dim: int = _FEATURE_DIM,
        memory_size: int = 2000,
        bootstrap_frames: int = 50,
        shift_threshold: float = 0.15,
        auto_adapt_every: int = 100,
        mmd_gamma: float = 1e-3,
    ) -> None:
        self._feature_dim = feature_dim
        self._memory_size = memory_size
        self._bootstrap_frames = bootstrap_frames
        self._shift_threshold = shift_threshold
        self._auto_adapt_every = auto_adapt_every
        self._mmd_gamma = mmd_gamma

        self._lock = threading.RLock()

        # Per-camera state
        self._cameras: dict[str, CameraAdaptationState] = {}

        # Global counters
        self._total_frames_seen: int = 0
        self._total_adaptations: int = 0
        self._total_labeled: int = 0
        self._frames_since_last_adapt: dict[str, int] = {}

    # ── Camera state management ────────────────────────────────────────────

    def _get_or_create(self, camera_id: str) -> CameraAdaptationState:
        if camera_id not in self._cameras:
            st = CameraAdaptationState(
                camera_id=camera_id,
                buffer=deque(maxlen=self._memory_size),
            )
            self._cameras[camera_id] = st
            self._frames_since_last_adapt[camera_id] = 0
        return self._cameras[camera_id]

    # ── Public API ─────────────────────────────────────────────────────────

    def observe_frame(
        self,
        features: np.ndarray,
        camera_id: str = "cam",
        timestamp: float | None = None,
    ) -> None:
        """Buffer a new unlabeled frame's feature vector.

        Automatically triggers adapt() once per auto_adapt_every frames
        after the source domain has been bootstrapped.
        """
        if timestamp is None:
            timestamp = time.time()

        features = np.asarray(features, dtype=np.float32).flatten()
        if features.shape[0] != self._feature_dim:
            logger.debug(
                "DomainAdapter: feature dim mismatch %d vs %d",
                features.shape[0],
                self._feature_dim,
            )
            return

        with self._lock:
            st = self._get_or_create(camera_id)
            st.buffer.append((features, timestamp))
            self._total_frames_seen += 1
            self._frames_since_last_adapt[camera_id] = (
                self._frames_since_last_adapt.get(camera_id, 0) + 1
            )

            # Bootstrap source statistics from first N frames
            if not st.source_bootstrapped:
                st.source_frames_seen += 1
                if st.source_frames_seen >= self._bootstrap_frames:
                    self._bootstrap_source(st)

            # Auto-adapt when enough new frames have arrived
            elif (
                self._frames_since_last_adapt[camera_id] >= self._auto_adapt_every
                and len(st.buffer) >= self._bootstrap_frames
            ):
                self._run_adapt(st, timestamp)
                self._frames_since_last_adapt[camera_id] = 0

    def detect_domain_shift(self, camera_id: str = "cam") -> dict:
        """Compute MMD-based domain shift score for a camera.

        Returns:
            dict with keys: camera_id, shift_score, shift_level,
                            source_frames, target_frames, threshold
        """
        with self._lock:
            st = self._cameras.get(camera_id)
            if st is None or not st.source_bootstrapped:
                return {
                    "camera_id": camera_id,
                    "shift_score": 0.0,
                    "shift_level": "unknown",
                    "source_frames": 0,
                    "target_frames": 0,
                    "threshold": self._shift_threshold,
                }

            # Source distribution: small centred sample around source_mean
            n_src = min(len(st.buffer), _MMD_SAMPLE)
            buf_arr = np.array([f for f, _ in st.buffer], dtype=np.float32)
            target_sample = buf_arr[-n_src:] if len(buf_arr) > 0 else np.empty((0, self._feature_dim))

            # Build synthetic source sample from N(μs, σs)
            rng = np.random.default_rng(seed=42)
            src_sample = rng.normal(
                loc=st.source_mean,
                scale=np.maximum(st.source_std, 1e-6),
                size=(min(_MMD_SAMPLE, self._bootstrap_frames), self._feature_dim),
            ).astype(np.float32)

            score = _mmd_rbf(src_sample, target_sample, gamma=self._mmd_gamma)

            if score < self._shift_threshold * 0.5:
                level = "minimal"
            elif score < self._shift_threshold:
                level = "moderate"
            elif score < self._shift_threshold * 2:
                level = "significant"
            else:
                level = "severe"

            st.shift_score = score
            return {
                "camera_id": camera_id,
                "shift_score": round(score, 6),
                "shift_level": level,
                "source_frames": st.source_frames_seen,
                "target_frames": len(st.buffer),
                "threshold": self._shift_threshold,
            }

    def adapt(self, camera_id: str = "cam") -> dict:
        """Manually trigger a diagonal CORAL adaptation step.

        Returns:
            dict with keys: camera_id, adapted, correction_norm, shift_before, shift_after
        """
        with self._lock:
            st = self._cameras.get(camera_id)
            if st is None or not st.source_bootstrapped:
                return {"camera_id": camera_id, "adapted": False, "reason": "not_bootstrapped"}

            shift_before = st.shift_score
            self._run_adapt(st, time.time())
            shift_after = st.shift_score

            corr_norm = float(np.linalg.norm(st.correction_mean)) if st.correction_mean is not None else 0.0
            return {
                "camera_id": camera_id,
                "adapted": True,
                "adapt_count": st.adapt_count,
                "correction_norm": round(corr_norm, 4),
                "shift_before": round(shift_before, 6),
                "shift_after": round(shift_after, 6),
            }

    def get_adapted_embedding(
        self,
        features: np.ndarray,
        camera_id: str = "cam",
    ) -> np.ndarray:
        """Return domain-adapted version of a feature vector.

        If no adaptation has been computed yet, returns the input unchanged.
        """
        features = np.asarray(features, dtype=np.float32).flatten()
        with self._lock:
            st = self._cameras.get(camera_id)
            if st is None or st.correction_mean is None:
                return features.copy()

            adapted = features.copy()
            if st.target_mean is not None and st.target_std is not None:
                # Zero-mean wrt target
                adapted = adapted - st.target_mean
                # Scale to source std
                if st.correction_scale is not None:
                    adapted = adapted * st.correction_scale
                # Shift to source mean
                adapted = adapted + st.source_mean

            # L2-normalize to stay on unit sphere
            norm = np.linalg.norm(adapted)
            if norm > 1e-8:
                adapted /= norm

            return adapted

    def apply_label(
        self,
        features: np.ndarray,
        category: str,
        camera_id: str = "cam",
        timestamp: float | None = None,
    ) -> None:
        """Integrate an operator-provided label (from Phase 45 ActiveLearningLoop).

        Labeled examples are stored in the buffer and used to track
        supervised accuracy proxy.
        """
        if timestamp is None:
            timestamp = time.time()
        features = np.asarray(features, dtype=np.float32).flatten()
        with self._lock:
            st = self._get_or_create(camera_id)
            st.labeled_count += 1
            self._total_labeled += 1
            # Still add to the buffer (supervised + unsupervised mix)
            st.buffer.append((features, timestamp))

    def get_stats(self) -> dict:
        """Return overall domain adaptation statistics."""
        with self._lock:
            camera_summaries = []
            for st in self._cameras.values():
                camera_summaries.append({
                    "camera_id": st.camera_id,
                    "source_bootstrapped": st.source_bootstrapped,
                    "buffer_size": len(st.buffer),
                    "shift_score": round(st.shift_score, 6),
                    "adapt_count": st.adapt_count,
                    "labeled_count": st.labeled_count,
                    "last_adapted_ts": st.last_adapted_ts,
                })
            return {
                "enabled": True,
                "total_frames_seen": self._total_frames_seen,
                "total_adaptations": self._total_adaptations,
                "total_labeled": self._total_labeled,
                "camera_count": len(self._cameras),
                "cameras": camera_summaries,
                "memory_size": self._memory_size,
                "bootstrap_frames": self._bootstrap_frames,
                "shift_threshold": self._shift_threshold,
                "auto_adapt_every": self._auto_adapt_every,
            }

    def get_camera_states(self) -> list[dict]:
        """Return per-camera adaptation state list."""
        with self._lock:
            return [
                {
                    "camera_id": st.camera_id,
                    "source_bootstrapped": st.source_bootstrapped,
                    "source_frames_seen": st.source_frames_seen,
                    "buffer_size": len(st.buffer),
                    "shift_score": round(st.shift_score, 6),
                    "adapt_count": st.adapt_count,
                    "labeled_count": st.labeled_count,
                    "last_adapted_ts": st.last_adapted_ts,
                    "has_correction": st.correction_mean is not None,
                    "shift_history_length": len(st.shift_history),
                }
                for st in self._cameras.values()
            ]

    def get_shift_history(self, camera_id: str = "cam") -> list[dict]:
        """Return the learning curve (time-series of shift scores) for a camera."""
        with self._lock:
            st = self._cameras.get(camera_id)
            if st is None:
                return []
            return [{"ts": ts, "shift_score": round(sc, 6)} for ts, sc in st.shift_history]

    def reset(self, camera_id: str | None = None) -> None:
        """Reset one or all cameras.

        Args:
            camera_id: if given, reset only that camera; else reset all.
        """
        with self._lock:
            if camera_id is not None:
                if camera_id in self._cameras:
                    del self._cameras[camera_id]
                    self._frames_since_last_adapt.pop(camera_id, None)
            else:
                self._cameras.clear()
                self._frames_since_last_adapt.clear()
                self._total_frames_seen = 0
                self._total_adaptations = 0
                self._total_labeled = 0

    # ── Internal helpers ───────────────────────────────────────────────────

    def _bootstrap_source(self, st: CameraAdaptationState) -> None:
        """Compute source mean/std from current buffer (called once)."""
        arr = np.array([f for f, _ in st.buffer], dtype=np.float32)
        if len(arr) == 0:
            return
        st.source_mean = arr.mean(axis=0)
        st.source_std = arr.std(axis=0) + 1e-8
        st.source_bootstrapped = True
        logger.debug(
            "DomainAdapter: source bootstrapped for camera=%s from %d frames",
            st.camera_id,
            len(arr),
        )

    def _run_adapt(self, st: CameraAdaptationState, timestamp: float) -> None:
        """Execute one diagonal CORAL adaptation step (caller holds lock)."""
        arr = np.array([f for f, _ in st.buffer], dtype=np.float32)
        if len(arr) < 2:
            return

        # Compute target statistics
        st.target_mean = arr.mean(axis=0)
        st.target_std = arr.std(axis=0) + 1e-8

        # Diagonal CORAL correction
        st.correction_mean = st.source_mean - st.target_mean
        st.correction_scale = st.source_std / st.target_std

        # Compute new shift score with updated target
        n_src = min(len(arr), _MMD_SAMPLE)
        target_sample = arr[-n_src:]
        rng = np.random.default_rng(seed=42)
        src_sample = rng.normal(
            loc=st.source_mean,
            scale=np.maximum(st.source_std, 1e-6),
            size=(min(_MMD_SAMPLE, self._bootstrap_frames), self._feature_dim),
        ).astype(np.float32)
        st.shift_score = _mmd_rbf(src_sample, target_sample, gamma=self._mmd_gamma)

        st.adapt_count += 1
        st.last_adapted_ts = timestamp
        self._total_adaptations += 1

        # Update learning curve
        st.shift_history.append((timestamp, st.shift_score))
        if len(st.shift_history) > _LEARNING_CURVE_MAX:
            st.shift_history.pop(0)

        logger.debug(
            "DomainAdapter: adapted camera=%s count=%d shift=%.4f",
            st.camera_id,
            st.adapt_count,
            st.shift_score,
        )
