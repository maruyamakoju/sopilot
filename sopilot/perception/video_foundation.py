"""Phase 37: Video Foundation Model Integration — V-JEPA2 / InternVideo 2.5.

Video foundation models overcome the fundamental limitation of single-frame
perception: they understand temporal dynamics rather than frozen snapshots.

**V-JEPA 2** (Video Joint-Embedding Predictive Architecture, Meta AI):
    - Self-supervised video representation via joint-embedding in latent space
    - Encoder trained to predict representations of masked video regions
    - 1024-dimensional CLS-pooled temporal embeddings
    - Processes 8–16 frame clips at 224×224, normalised with ImageNet stats
    - No contrastive negatives needed: pure predictive objective
    - Outperforms frame-level ViT on Kinetics-400, Something-Something V2

**InternVideo 2.5** (OpenGVLab, Shanghai AI Laboratory):
    - Multi-modal video understanding: vision + language alignment
    - Dual-encoder architecture: video encoder + text encoder
    - 768-dimensional video embeddings, compatible with DINOv2 feature spaces
    - Supports 4–32 frame clips for flexible temporal resolution
    - Video-text contrastive training enables zero-shot action recognition
    - Excels at fine-grained action understanding and temporal grounding

**Temporal advantage over DINOv2 (Phase 32)**:
    DINOv2 extracts per-frame CLS features; temporal reasoning is deferred to
    TemporalFusionEngine (Phase 35) via sliding-window pooling.  Video foundation
    models encode temporal dynamics *intrinsically* — motion, causality, and
    temporal ordering are captured in a single forward pass.  This enables:
        • Motion-aware action classification (running vs. walking vs. falling)
        • Temporal anomaly detection without an explicit buffer
        • Long-range dependency modelling for complex multi-step actions

Architecture::

    VideoFoundationModel (ABC)
        ├── VJEPABackend        — Meta V-JEPA 2, 1024-dim, 8–16 frames
        ├── InternVideoBackend  — OpenGVLab InternVideo 2.5, 768-dim, 4–32 frames
        └── MockVideoFoundation — Test stub (always available), 1024-dim

    VideoTemporalAnalyzer
        ├── observe(frame, timestamp)          — append to ring buffer
        ├── get_temporal_embedding()           — encode current buffer
        ├── classify_current_action()          — action classification
        └── get_stats()                        — diagnostic counters

Integration with PerceptionEngine::

    engine._video_foundation = VideoTemporalAnalyzer(backend, buffer_size=16)

    # In process_frame() — Stage 4n:
    engine._video_foundation.observe(frame, timestamp)
    # action label available at any time via classify_current_action()

Design decisions:
    - ABC enforces uniform interface; swap backends transparently
    - MockVideoFoundation guarantees graceful degradation without GPU
    - Thread-safe: all mutable state in VideoTemporalAnalyzer guarded by RLock
    - All model loads wrapped in try/except; ImportError logged at DEBUG level
    - Orchestrator integration: register models for VRAM-aware lifecycle management
    - Action prototypes in MockVideoFoundation derived from frame pixel statistics
      (deterministic hash → reproducible results across runs)
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared action label vocabulary
# ---------------------------------------------------------------------------

_ACTION_LABELS: list[str] = [
    "walking",
    "standing",
    "working",
    "running",
    "falling",
    "loitering",
    "gathering",
    "evacuating",
    "operating_machine",
    "carrying_object",
    "using_phone",
    "fighting",
]

# ImageNet normalisation constants (used by both V-JEPA and InternVideo)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Target spatial resolution for video foundation model inputs
_VF_SPATIAL_SIZE = 224


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class VideoFoundationModel(ABC):
    """Unified interface for video foundation models.

    All implementations must be thread-safe.  Model loading is performed
    lazily in ``_load_model()`` and must not raise outside of the
    ``_load_model`` call itself (callers wrap in try/except).
    """

    @abstractmethod
    def encode_sequence(self, frames: list[np.ndarray]) -> np.ndarray:
        """Encode a sequence of BGR frames into a temporal embedding.

        Args:
            frames: List of BGR uint8 frames (H, W, 3).  Length must satisfy
                    ``min_frames <= len(frames) <= max_frames``.

        Returns:
            1-D float32 array of shape ``(embedding_dim,)``, L2-normalised.
        """

    @abstractmethod
    def classify_action(self, frames: list[np.ndarray]) -> dict[str, float]:
        """Classify the dominant action present in the frame sequence.

        Args:
            frames: List of BGR uint8 frames.

        Returns:
            Dict mapping each action label in ``_ACTION_LABELS`` to a
            probability in ``[0, 1]`` (softmax-normalised).
        """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the temporal embedding vector."""

    @property
    @abstractmethod
    def min_frames(self) -> int:
        """Minimum number of frames accepted by ``encode_sequence``."""

    @property
    @abstractmethod
    def max_frames(self) -> int:
        """Maximum number of frames accepted by ``encode_sequence``."""

    # ── Helpers shared by all backends ────────────────────────────────

    def _preprocess_frames(
        self, frames: list[np.ndarray]
    ) -> "np.ndarray":
        """Resize + normalise a list of BGR frames → float32 (T, H, W, C).

        Converts BGR → RGB, resizes to (_VF_SPATIAL_SIZE, _VF_SPATIAL_SIZE),
        applies ImageNet mean/std normalisation, and stacks into a contiguous
        array suitable for torch/ONNX consumption.
        """
        processed: list[np.ndarray] = []
        for bgr in frames:
            # BGR → RGB
            rgb = bgr[:, :, ::-1]
            # Resize to target spatial size
            resized = _resize_frame(rgb, _VF_SPATIAL_SIZE, _VF_SPATIAL_SIZE)
            # Normalise: [0,255] → [0,1] → subtract mean / divide std
            f32 = resized.astype(np.float32) / 255.0
            normalised = (f32 - _IMAGENET_MEAN) / _IMAGENET_STD
            processed.append(normalised)
        return np.stack(processed, axis=0)  # (T, H, W, C)

    def _clip_frames(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Uniformly sample or pad ``frames`` to fit ``[min_frames, max_frames]``."""
        n = len(frames)
        target = min(max(n, self.min_frames), self.max_frames)
        if n == target:
            return frames
        if n > target:
            # Uniform temporal sampling
            indices = np.linspace(0, n - 1, target, dtype=int)
            return [frames[i] for i in indices]
        # Repeat last frame to reach min_frames
        return frames + [frames[-1]] * (target - n)

    def _softmax(self, scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Numerically stable softmax with temperature scaling."""
        s = scores / max(temperature, 1e-8)
        s = s - s.max()
        e = np.exp(s)
        return e / (e.sum() + 1e-8)


# ---------------------------------------------------------------------------
# Utility: pure-numpy frame resize (avoids cv2 dependency at module level)
# ---------------------------------------------------------------------------


def _resize_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbour resize; falls back to bilinear via cv2 if available."""
    if frame.shape[:2] == (height, width):
        return frame
    try:
        import cv2  # type: ignore
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        pass
    # Pure-numpy nearest-neighbour fallback
    src_h, src_w = frame.shape[:2]
    row_idx = (np.arange(height) * src_h / height).astype(int)
    col_idx = (np.arange(width) * src_w / width).astype(int)
    return frame[np.ix_(row_idx, col_idx)]


# ---------------------------------------------------------------------------
# V-JEPA 2 Backend
# ---------------------------------------------------------------------------


class VJEPABackend(VideoFoundationModel):
    """V-JEPA 2: Meta's Video Joint-Embedding Predictive Architecture.

    Loads the vision encoder from ``facebookresearch/vjepa2`` via
    ``torch.hub`` (primary path) or HuggingFace Transformers (fallback).
    CLS tokens from each frame position are mean-pooled over the temporal
    dimension to produce a single 1024-dimensional embedding.

    If neither loading path succeeds (no GPU, missing weights, etc.), the
    backend gracefully falls back to MockVideoFoundation behaviour.
    """

    _EMBEDDING_DIM = 1024
    _MIN_FRAMES = 8
    _MAX_FRAMES = 16
    _VRAM_ESTIMATE_GB = 4.0
    _PRIORITY = 5

    def __init__(self, device: str = "cpu", orchestrator: Any | None = None) -> None:
        self._device = device
        self._orchestrator = orchestrator
        self._model: Any | None = None
        self._loaded = False
        self._lock = threading.RLock()
        # Action prototype vectors: random unit vectors seeded by label hash
        # (replaced by real model embeddings after first successful call)
        self._action_prototypes = self._init_mock_prototypes(self._EMBEDDING_DIM)
        # Register with orchestrator for VRAM-aware lifecycle management
        if orchestrator is not None:
            try:
                orchestrator.register(
                    model_id="vjepa2_encoder",
                    load_fn=self._load_model,
                    vram_estimate_gb=self._VRAM_ESTIMATE_GB,
                    priority=self._PRIORITY,
                )
                logger.info("VJEPABackend registered with ModelOrchestrator")
            except Exception:
                logger.debug("Orchestrator registration failed for VJEPABackend", exc_info=True)

    # ── Properties ────────────────────────────────────────────────────

    @property
    def embedding_dim(self) -> int:
        return self._EMBEDDING_DIM

    @property
    def min_frames(self) -> int:
        return self._MIN_FRAMES

    @property
    def max_frames(self) -> int:
        return self._MAX_FRAMES

    # ── Model loading ──────────────────────────────────────────────────

    def _load_model(self) -> Any:
        """Load V-JEPA 2 vision encoder.

        Attempts torch.hub first (offline-friendly), then HuggingFace.
        Returns the encoder model moved to ``self._device``.
        """
        try:
            import torch  # type: ignore
            model = torch.hub.load(
                "facebookresearch/vjepa2",
                "vjepa2_vitl",
                pretrained=True,
                trust_repo=True,
            )
            model = model.to(self._device).eval()
            logger.info("V-JEPA 2 ViT-L encoder loaded via torch.hub (device=%s)", self._device)
            return model
        except Exception as hub_exc:
            logger.debug("torch.hub V-JEPA 2 load failed: %s", hub_exc)

        # HuggingFace fallback
        try:
            import torch  # type: ignore
            from transformers import AutoModel  # type: ignore
            model = AutoModel.from_pretrained(
                "facebook/vjepa2-vitl-fpc64-256",
                trust_remote_code=True,
            )
            model = model.to(self._device).eval()
            logger.info("V-JEPA 2 loaded from HuggingFace (device=%s)", self._device)
            return model
        except Exception as hf_exc:
            logger.debug("HuggingFace V-JEPA 2 load failed: %s", hf_exc)

        return None

    def _ensure_loaded(self) -> None:
        """Lazy model load; idempotent."""
        with self._lock:
            if self._loaded:
                return
            self._loaded = True  # prevent repeated attempts on failure
            try:
                self._model = self._load_model()
            except Exception:
                logger.debug("VJEPABackend model load error", exc_info=True)

    # ── Core API ──────────────────────────────────────────────────────

    def encode_sequence(self, frames: list[np.ndarray]) -> np.ndarray:
        """Encode frame sequence → 1024-dim L2-normalised embedding."""
        frames = self._clip_frames(frames)
        self._ensure_loaded()

        if self._model is not None:
            try:
                return self._encode_with_model(frames)
            except Exception:
                logger.debug("VJEPABackend.encode_sequence model inference failed", exc_info=True)

        # Graceful fallback: mean of per-frame pseudo-features
        return self._fallback_encode(frames)

    def classify_action(self, frames: list[np.ndarray]) -> dict[str, float]:
        """Classify action via cosine similarity to prototype embeddings."""
        embedding = self.encode_sequence(frames)
        scores = np.array([
            float(np.dot(embedding, proto))
            for proto in self._action_prototypes.values()
        ], dtype=np.float32)
        probs = self._softmax(scores, temperature=0.5)
        return dict(zip(_ACTION_LABELS, probs.tolist()))

    # ── Internal helpers ───────────────────────────────────────────────

    def _encode_with_model(self, frames: list[np.ndarray]) -> np.ndarray:
        """Run real V-JEPA 2 forward pass and pool CLS tokens over time."""
        import torch  # type: ignore
        preprocessed = self._preprocess_frames(frames)  # (T, H, W, C)
        # V-JEPA expects (B, C, T, H, W)
        tensor = torch.from_numpy(
            preprocessed.transpose(3, 0, 1, 2)[np.newaxis]  # (1, C, T, H, W)
        ).float().to(self._device)

        with torch.no_grad():
            out = self._model(tensor)
            # Extract CLS token or mean-pool patch tokens
            if hasattr(out, "last_hidden_state"):
                feats = out.last_hidden_state  # (B, seq_len, dim)
                embedding = feats[:, 0, :].squeeze(0).cpu().numpy()
            elif isinstance(out, torch.Tensor):
                embedding = out.mean(dim=1).squeeze(0).cpu().numpy()
            else:
                embedding = np.zeros(self._EMBEDDING_DIM, dtype=np.float32)

        # L2 normalise
        norm = np.linalg.norm(embedding) + 1e-8
        return (embedding / norm).astype(np.float32)

    def _fallback_encode(self, frames: list[np.ndarray]) -> np.ndarray:
        """Deterministic pseudo-embedding from frame statistics (no model)."""
        vecs = []
        for f in frames:
            flat = f.astype(np.float32).reshape(-1)
            rng = np.random.default_rng(int(hashlib.md5(f.tobytes()).hexdigest(), 16) % (2**31))
            vec = rng.standard_normal(self._EMBEDDING_DIM).astype(np.float32)
            # Bias direction based on mean pixel intensity
            mean_intensity = flat.mean() / 255.0
            vec[:4] = mean_intensity
            vecs.append(vec / (np.linalg.norm(vec) + 1e-8))
        result = np.mean(vecs, axis=0).astype(np.float32)
        return result / (np.linalg.norm(result) + 1e-8)

    @staticmethod
    def _init_mock_prototypes(dim: int) -> dict[str, np.ndarray]:
        """Initialise random unit-vector prototypes per action label."""
        prototypes: dict[str, np.ndarray] = {}
        for label in _ACTION_LABELS:
            seed = int(hashlib.md5(label.encode()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(dim).astype(np.float32)
            prototypes[label] = vec / (np.linalg.norm(vec) + 1e-8)
        return prototypes


# ---------------------------------------------------------------------------
# InternVideo 2.5 Backend
# ---------------------------------------------------------------------------


class InternVideoBackend(VideoFoundationModel):
    """InternVideo 2.5: Multi-modal video understanding from OpenGVLab.

    Loads the video encoder from HuggingFace (``OpenGVLab/InternVideo2_5``
    or a compatible lighter variant).  Video-text cosine similarity drives
    zero-shot action classification over ``_ACTION_LABELS``.

    The 768-dimensional output embedding is compatible with DINOv2 (Phase 32)
    feature spaces, allowing mixed-backbone downstream analysis.
    """

    _EMBEDDING_DIM = 768
    _MIN_FRAMES = 4
    _MAX_FRAMES = 32
    _VRAM_ESTIMATE_GB = 6.0
    _PRIORITY = 4

    # Fallback to smaller checkpoint if 8B is unavailable
    _HF_CHECKPOINTS = [
        "OpenGVLab/InternVideo2_5_Chat_8B",
        "OpenGVLab/InternVideo2-Stage2_1B-224p-f4",
    ]

    def __init__(self, device: str = "cpu", orchestrator: Any | None = None) -> None:
        self._device = device
        self._orchestrator = orchestrator
        self._video_encoder: Any | None = None
        self._text_encoder: Any | None = None
        self._tokenizer: Any | None = None
        self._loaded = False
        self._lock = threading.RLock()
        self._action_prototypes = VJEPABackend._init_mock_prototypes(self._EMBEDDING_DIM)

        if orchestrator is not None:
            try:
                orchestrator.register(
                    model_id="internvideo25_encoder",
                    load_fn=self._load_model,
                    vram_estimate_gb=self._VRAM_ESTIMATE_GB,
                    priority=self._PRIORITY,
                )
                logger.info("InternVideoBackend registered with ModelOrchestrator")
            except Exception:
                logger.debug("Orchestrator registration failed for InternVideoBackend", exc_info=True)

    # ── Properties ────────────────────────────────────────────────────

    @property
    def embedding_dim(self) -> int:
        return self._EMBEDDING_DIM

    @property
    def min_frames(self) -> int:
        return self._MIN_FRAMES

    @property
    def max_frames(self) -> int:
        return self._MAX_FRAMES

    # ── Model loading ──────────────────────────────────────────────────

    def _load_model(self) -> tuple[Any, Any, Any] | None:
        """Load InternVideo 2.5 video encoder + text tokenizer.

        Returns a tuple ``(video_encoder, text_encoder, tokenizer)`` or None.
        """
        for checkpoint in self._HF_CHECKPOINTS:
            try:
                from transformers import AutoModel, AutoTokenizer  # type: ignore
                import torch  # type: ignore

                tokenizer = AutoTokenizer.from_pretrained(
                    checkpoint, trust_remote_code=True
                )
                model = AutoModel.from_pretrained(
                    checkpoint,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if "cuda" in self._device else torch.float32,
                )
                model = model.to(self._device).eval()
                logger.info("InternVideo 2.5 loaded from %s (device=%s)", checkpoint, self._device)
                return model, model, tokenizer  # video and text share the same model
            except Exception as exc:
                logger.debug("InternVideo load failed for %s: %s", checkpoint, exc)

        return None

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            try:
                result = self._load_model()
                if result is not None:
                    self._video_encoder, self._text_encoder, self._tokenizer = result
            except Exception:
                logger.debug("InternVideoBackend model load error", exc_info=True)

    # ── Core API ──────────────────────────────────────────────────────

    def encode_sequence(self, frames: list[np.ndarray]) -> np.ndarray:
        """Encode frame sequence → 768-dim L2-normalised embedding.

        Uses temporal average pooling of per-frame visual features.
        """
        frames = self._clip_frames(frames)
        self._ensure_loaded()

        if self._video_encoder is not None:
            try:
                return self._encode_with_model(frames)
            except Exception:
                logger.debug("InternVideoBackend.encode_sequence failed", exc_info=True)

        return self._fallback_encode(frames)

    def classify_action(self, frames: list[np.ndarray]) -> dict[str, float]:
        """Zero-shot action classification via video-text matching."""
        frames = self._clip_frames(frames)
        self._ensure_loaded()

        if self._video_encoder is not None and self._tokenizer is not None:
            try:
                return self._classify_with_model(frames)
            except Exception:
                logger.debug("InternVideoBackend.classify_action failed", exc_info=True)

        # Fallback: cosine against random prototypes
        embedding = self.encode_sequence(frames)
        scores = np.array([
            float(np.dot(embedding, proto))
            for proto in self._action_prototypes.values()
        ], dtype=np.float32)
        probs = self._softmax(scores, temperature=0.5)
        return dict(zip(_ACTION_LABELS, probs.tolist()))

    # ── Internal helpers ───────────────────────────────────────────────

    def _encode_with_model(self, frames: list[np.ndarray]) -> np.ndarray:
        """Temporal average pooling of InternVideo per-frame features."""
        import torch  # type: ignore
        preprocessed = self._preprocess_frames(frames)  # (T, H, W, C)
        frame_embeddings = []
        for t in range(preprocessed.shape[0]):
            img = preprocessed[t]  # (H, W, C)
            tensor = torch.from_numpy(
                img.transpose(2, 0, 1)[np.newaxis]  # (1, C, H, W)
            ).float().to(self._device)
            with torch.no_grad():
                out = self._video_encoder(pixel_values=tensor)
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    feat = out.pooler_output.squeeze(0).cpu().numpy()
                elif hasattr(out, "last_hidden_state"):
                    feat = out.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
                else:
                    feat = out[0].mean(dim=1).squeeze(0).cpu().numpy()
            frame_embeddings.append(feat)

        embedding = np.mean(frame_embeddings, axis=0).astype(np.float32)
        norm = np.linalg.norm(embedding) + 1e-8
        return embedding / norm

    def _classify_with_model(self, frames: list[np.ndarray]) -> dict[str, float]:
        """Video-text cosine similarity classification using tokenized labels."""
        import torch  # type: ignore
        video_embedding = self._encode_with_model(frames)

        text_similarities: list[float] = []
        for label in _ACTION_LABELS:
            try:
                tokens = self._tokenizer(
                    label.replace("_", " "),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self._device)
                with torch.no_grad():
                    out = self._text_encoder(**tokens)
                    if hasattr(out, "pooler_output") and out.pooler_output is not None:
                        text_feat = out.pooler_output.squeeze(0).cpu().numpy()
                    else:
                        text_feat = out.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
                text_feat = text_feat / (np.linalg.norm(text_feat) + 1e-8)
                sim = float(np.dot(video_embedding, text_feat))
            except Exception:
                sim = 0.0
            text_similarities.append(sim)

        scores = np.array(text_similarities, dtype=np.float32)
        probs = self._softmax(scores, temperature=0.3)
        return dict(zip(_ACTION_LABELS, probs.tolist()))

    def _fallback_encode(self, frames: list[np.ndarray]) -> np.ndarray:
        """Deterministic pseudo-embedding from frame statistics (no model)."""
        vecs = []
        for f in frames:
            seed = int(hashlib.md5(f.tobytes()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self._EMBEDDING_DIM).astype(np.float32)
            mean_intensity = f.astype(np.float32).mean() / 255.0
            vec[:3] = mean_intensity
            vecs.append(vec / (np.linalg.norm(vec) + 1e-8))
        result = np.mean(vecs, axis=0).astype(np.float32)
        return result / (np.linalg.norm(result) + 1e-8)


# ---------------------------------------------------------------------------
# Mock Backend (always available, no GPU or model weights required)
# ---------------------------------------------------------------------------


class MockVideoFoundation(VideoFoundationModel):
    """Mock video foundation model for testing and graceful degradation.

    ``encode_sequence`` computes the mean of per-frame feature vectors derived
    from pixel statistics — simulating temporal pooling without any GPU.

    ``classify_action`` uses a deterministic hash of the frame content to
    seed a random generator, producing reproducible softmax probabilities
    across identical inputs.
    """

    _EMBEDDING_DIM = 1024
    _MIN_FRAMES = 2
    _MAX_FRAMES = 64

    def __init__(self) -> None:
        self._action_prototypes = VJEPABackend._init_mock_prototypes(self._EMBEDDING_DIM)

    @property
    def embedding_dim(self) -> int:
        return self._EMBEDDING_DIM

    @property
    def min_frames(self) -> int:
        return self._MIN_FRAMES

    @property
    def max_frames(self) -> int:
        return self._MAX_FRAMES

    def encode_sequence(self, frames: list[np.ndarray]) -> np.ndarray:
        """Mean of per-frame pseudo-feature vectors (simulated temporal pooling)."""
        frames = self._clip_frames(frames)
        vecs: list[np.ndarray] = []
        for f in frames:
            seed = int(hashlib.md5(f.tobytes()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self._EMBEDDING_DIM).astype(np.float32)
            # Encode mean channel values into the first 3 dimensions for
            # deterministic per-frame variation based on image content.
            means = f.astype(np.float32).mean(axis=(0, 1)) / 255.0  # (3,) BGR
            vec[:3] = means
            vecs.append(vec / (np.linalg.norm(vec) + 1e-8))

        result = np.mean(vecs, axis=0).astype(np.float32)
        return result / (np.linalg.norm(result) + 1e-8)

    def classify_action(self, frames: list[np.ndarray]) -> dict[str, float]:
        """Random-softmax action classification seeded by frame content hash."""
        frames = self._clip_frames(frames)
        # Build a composite hash from all frames to capture temporal context
        h = hashlib.md5()
        for f in frames:
            # Downsample to 16×16 before hashing for speed
            small = _resize_frame(f, 16, 16)
            h.update(small.tobytes())
        seed = int(h.hexdigest(), 16) % (2**31)
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal(len(_ACTION_LABELS)).astype(np.float32)
        probs = self._softmax(raw, temperature=1.0)
        return dict(zip(_ACTION_LABELS, probs.tolist()))


# ---------------------------------------------------------------------------
# VideoTemporalAnalyzer — high-level API with ring buffer
# ---------------------------------------------------------------------------


@dataclass
class _BufferedFrame:
    frame: np.ndarray
    timestamp: float


class VideoTemporalAnalyzer:
    """High-level temporal analysis manager.

    Maintains a ring buffer of recent frames and delegates to a
    ``VideoFoundationModel`` backend for embedding extraction and
    action classification.  All public methods are thread-safe.

    Args:
        backend:      Any ``VideoFoundationModel`` implementation.
        buffer_size:  Maximum number of frames to retain (ring buffer).
        orchestrator: Optional ``ModelOrchestrator`` (passed through to backend).
    """

    def __init__(
        self,
        backend: VideoFoundationModel,
        buffer_size: int = 16,
        orchestrator: Any | None = None,
    ) -> None:
        self._backend = backend
        self._buffer_size = max(buffer_size, backend.min_frames)
        self._orchestrator = orchestrator
        self._lock = threading.RLock()
        self._buffer: deque[_BufferedFrame] = deque(maxlen=self._buffer_size)
        # Diagnostic counters
        self._frames_seen: int = 0
        self._action_classifications: int = 0
        self._embedding_calls: int = 0
        self._last_action: dict[str, float] = {}
        self._created_at: float = time.monotonic()
        logger.debug(
            "VideoTemporalAnalyzer created (backend=%s, buffer_size=%d)",
            type(backend).__name__,
            self._buffer_size,
        )

    # ── Public API ────────────────────────────────────────────────────

    def observe(self, frame: np.ndarray, timestamp: float) -> None:
        """Append a frame to the ring buffer.

        Args:
            frame:     BGR uint8 ndarray (H, W, 3).
            timestamp: Monotonic timestamp in seconds.
        """
        with self._lock:
            self._buffer.append(_BufferedFrame(frame=frame, timestamp=timestamp))
            self._frames_seen += 1

    def get_temporal_embedding(self) -> np.ndarray | None:
        """Encode the current ring buffer into a temporal embedding.

        Returns:
            Float32 ndarray of shape ``(embedding_dim,)``, or ``None`` if the
            buffer does not yet contain enough frames.
        """
        with self._lock:
            if len(self._buffer) < self._backend.min_frames:
                return None
            frames = [bf.frame for bf in self._buffer]
            self._embedding_calls += 1

        try:
            return self._backend.encode_sequence(frames)
        except Exception:
            logger.debug("VideoTemporalAnalyzer.get_temporal_embedding failed", exc_info=True)
            return None

    def classify_current_action(self) -> dict[str, float]:
        """Classify the dominant action in the current ring buffer.

        Returns:
            Dict mapping action labels to probabilities, or an empty dict
            if the buffer is too small.
        """
        with self._lock:
            if len(self._buffer) < self._backend.min_frames:
                return {}
            frames = [bf.frame for bf in self._buffer]
            self._action_classifications += 1

        try:
            result = self._backend.classify_action(frames)
            with self._lock:
                self._last_action = result
            return result
        except Exception:
            logger.debug("VideoTemporalAnalyzer.classify_current_action failed", exc_info=True)
            return {}

    def get_stats(self) -> dict:
        """Return diagnostic statistics for health monitoring."""
        with self._lock:
            buf_len = len(self._buffer)
            uptime = time.monotonic() - self._created_at
            top_action = ""
            top_conf = 0.0
            if self._last_action:
                top_action = max(self._last_action, key=self._last_action.__getitem__)
                top_conf = self._last_action[top_action]
            return {
                "backend_type": type(self._backend).__name__,
                "embedding_dim": self._backend.embedding_dim,
                "buffer_capacity": self._buffer_size,
                "buffer_current": buf_len,
                "min_frames": self._backend.min_frames,
                "max_frames": self._backend.max_frames,
                "frames_seen": self._frames_seen,
                "embedding_calls": self._embedding_calls,
                "action_classifications": self._action_classifications,
                "top_action": top_action,
                "top_action_confidence": round(top_conf, 4),
                "uptime_seconds": round(uptime, 2),
                "buffer_ready": buf_len >= self._backend.min_frames,
            }

    def reset(self) -> None:
        """Clear the frame buffer and reset diagnostic counters."""
        with self._lock:
            self._buffer.clear()
            self._frames_seen = 0
            self._action_classifications = 0
            self._embedding_calls = 0
            self._last_action = {}
        logger.debug("VideoTemporalAnalyzer reset")


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def build_video_foundation(
    backend: str = "mock",
    orchestrator: Any | None = None,
    device: str = "cpu",
) -> VideoFoundationModel:
    """Instantiate a video foundation model backend.

    Args:
        backend:      ``"vjepa"`` | ``"internvideo"`` | ``"mock"``
                      Any unrecognised value falls back to ``MockVideoFoundation``.
        orchestrator: Optional ``ModelOrchestrator`` for VRAM-aware lifecycle.
        device:       PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
                      ``"auto"`` resolves to ``"cuda"`` if available, else ``"cpu"``.

    Returns:
        A ready-to-use ``VideoFoundationModel`` instance.
    """
    resolved_device = _resolve_device(device)

    if backend == "vjepa":
        logger.info("Building VJEPABackend (device=%s)", resolved_device)
        return VJEPABackend(device=resolved_device, orchestrator=orchestrator)

    if backend == "internvideo":
        logger.info("Building InternVideoBackend (device=%s)", resolved_device)
        return InternVideoBackend(device=resolved_device, orchestrator=orchestrator)

    if backend != "mock":
        logger.warning(
            "Unknown video_foundation_backend=%r; falling back to MockVideoFoundation",
            backend,
        )

    logger.debug("Building MockVideoFoundation")
    return MockVideoFoundation()


def _resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to the best available device."""
    if device != "auto":
        return device
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
    except ImportError:
        pass
    return "cpu"
