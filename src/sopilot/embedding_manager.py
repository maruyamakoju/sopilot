from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np

from .utils import normalize_rows
from .video import ClipWindow

if __import__("typing").TYPE_CHECKING:
    from .config import Settings
    from .embeddings import ClipEmbedder


logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embedder, feature adapter loading/caching, and adapter application.

    Supports two adapter modes:
    - Z-score (legacy): mean/std normalization from .npz files
    - Neural (new): learned ProjectionHead from .pt files
    """

    def __init__(self, settings: Settings, embedder: ClipEmbedder) -> None:
        self.settings = settings
        self.embedder = embedder
        self._adapter_pointer_path = settings.models_dir / "current_adapter.json"
        self._adapter_lock = threading.Lock()
        self._adapter_cache_path: Path | None = None
        self._adapter_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._neural_adapter: object | None = None  # ProjectionHead when loaded
        self._neural_adapter_path: Path | None = None

    def active_model_name(self) -> str:
        try:
            return self.embedder.active_name()
        except Exception:
            return self.embedder.name

    def embed_batch(self, clips: list[ClipWindow]) -> tuple[np.ndarray, np.ndarray]:
        raw = self.embedder.embed_clips(clips).astype(np.float32)
        effective = self.apply_adapter(raw)
        return raw, effective

    def load_adapter(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.settings.enable_feature_adapter:
            return None
        if not self._adapter_pointer_path.exists():
            return None
        try:
            pointer = json.loads(self._adapter_pointer_path.read_text(encoding="utf-8"))
            adapter_path = Path(pointer["adapter_path"])
        except Exception:
            return None
        if not adapter_path.exists():
            return None

        with self._adapter_lock:
            if self._adapter_cache is not None and self._adapter_cache_path == adapter_path:
                return self._adapter_cache

            blob = np.load(adapter_path)
            mean = blob["mean"].astype(np.float32)
            std = np.maximum(blob["std"].astype(np.float32), 1e-6)
            self._adapter_cache = (mean, std)
            self._adapter_cache_path = adapter_path
            return self._adapter_cache

    def apply_adapter(self, embeddings: np.ndarray) -> np.ndarray:
        out = embeddings.astype(np.float32)

        # Try neural adapter first when neural mode is enabled
        if self.settings.neural_mode and self.settings.neural_projection_enabled:
            neural_out = self._apply_neural_adapter(out)
            if neural_out is not None:
                return neural_out

        # Fallback to Z-score adapter
        adapter = self.load_adapter()
        if adapter is None:
            return out
        mean, std = adapter
        if out.ndim != 2 or mean.ndim != 1 or out.shape[1] != mean.shape[0]:
            return out
        out = (out - mean[None, :]) / std[None, :]
        return normalize_rows(out)

    def _apply_neural_adapter(self, embeddings: np.ndarray) -> np.ndarray | None:
        """Apply learned ProjectionHead if available.

        Returns None if neural adapter cannot be loaded (falls back to Z-score).
        """
        model = self._load_neural_adapter()
        if model is None:
            return None
        try:
            import torch

            with torch.no_grad():
                x = torch.from_numpy(embeddings)
                device = next(model.parameters()).device
                x = x.to(device)
                out = model(x).cpu().numpy()
            return out.astype(np.float32)
        except Exception as e:
            logger.warning("Neural adapter forward pass failed, falling back: %s", e)
            return None

    def _load_neural_adapter(self) -> object | None:
        """Load ProjectionHead model, with caching."""
        model_path = self.settings.neural_model_dir / "projection_head.pt"
        if not model_path.exists():
            return None

        with self._adapter_lock:
            if self._neural_adapter is not None and self._neural_adapter_path == model_path:
                return self._neural_adapter
            try:
                from .nn.projection_head import load_projection_head

                device = self._resolve_neural_device()
                model = load_projection_head(model_path, device=device)
                self._neural_adapter = model
                self._neural_adapter_path = model_path
                return model
            except Exception as e:
                logger.warning("Failed to load neural adapter: %s", e)
                return None

    def _resolve_neural_device(self) -> str:
        """Resolve neural device setting to actual device string."""
        device = self.settings.neural_device
        if device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    def invalidate_cache(self) -> None:
        with self._adapter_lock:
            self._adapter_cache = None
            self._adapter_cache_path = None
            self._neural_adapter = None
            self._neural_adapter_path = None
