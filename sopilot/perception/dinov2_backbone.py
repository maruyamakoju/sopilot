"""DINOv2 Feature Backbone — universal visual feature extractor.

Phase 32: Uses DINOv2 ViT as a shared feature backbone providing:

    1. **Re-ID embeddings** — CLS token for person re-identification
       (replaces HSV color histogram with 768-dim neural features)
    2. **Scene classification** — CLS token for scene-type discrimination
    3. **Anomaly detection** — Mahalanobis distance in feature space
    4. **Attention maps** — Self-attention for SAM prompt generation

All features come from a single forward pass, making DINOv2 the most
cost-effective model in the orchestration stack.

Integration::

    backbone = DINOv2Backbone(orchestrator)
    features = backbone.extract_features(frame)       # full patch + CLS
    reid_emb = backbone.extract_reid_embedding(crop)  # 768-dim for Re-ID
    scene    = backbone.classify_scene(frame)          # scene label
    anomaly  = backbone.anomaly_score(frame)           # Mahalanobis dist
    attn_map = backbone.get_attention_map(frame)       # H×W float32

Requires: torch, torchvision (auto-downloads ViT-B/14 weights on first use).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default model configuration
_DEFAULT_MODEL = "dinov2_vitb14"
_EMBEDDING_DIM = 768       # ViT-B output dimension
_PATCH_SIZE = 14           # DINOv2 ViT-B patch size
_INPUT_SIZE = 518          # 37 patches × 14 = 518 (divisible by 14)
_SCENE_LABELS = [
    "factory_floor",
    "warehouse",
    "corridor",
    "office",
    "outdoor",
    "parking_lot",
    "loading_dock",
    "construction_site",
]


class DINOv2Backbone:
    """Universal visual feature extractor using DINOv2 ViT-B/14.

    All expensive operations (model load, inference) go through the
    :class:`ModelOrchestrator` for VRAM-aware lifecycle management.

    Parameters
    ----------
    orchestrator:
        The :class:`ModelOrchestrator` instance managing GPU resources.
        If ``None``, the backbone operates in standalone mode (self-manages).
    variant:
        PyTorch Hub model name.  Default ``"dinov2_vitb14"``.
    device:
        Override device (default: use orchestrator's device).
    """

    MODEL_NAME = "dinov2"

    def __init__(
        self,
        orchestrator: Any = None,
        variant: str = _DEFAULT_MODEL,
        device: str | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._variant = variant
        self._device_override = device
        self._lock = threading.RLock()

        # Standalone mode state (used when no orchestrator)
        self._model: Any = None
        self._loaded = False
        self._transform: Any = None

        # Online scene statistics for anomaly scoring
        self._feature_mean: np.ndarray | None = None
        self._feature_cov_inv: np.ndarray | None = None
        self._feature_count: int = 0
        self._feature_sum: np.ndarray = np.zeros(_EMBEDDING_DIM, dtype=np.float64)
        self._feature_sq_sum: np.ndarray = np.zeros(
            (_EMBEDDING_DIM, _EMBEDDING_DIM), dtype=np.float64
        )

        # Scene classifier prototypes (lazy-initialized)
        self._scene_prototypes: dict[str, np.ndarray] | None = None

        if orchestrator is not None:
            self._register_with_orchestrator()

        logger.info(
            "DINOv2Backbone created: variant=%s, orchestrator=%s",
            variant,
            "yes" if orchestrator else "standalone",
        )

    # ── Orchestrator registration ─────────────────────────────────────

    def _register_with_orchestrator(self) -> None:
        """Register DINOv2 with the model orchestrator."""
        self._orchestrator.register(
            name=self.MODEL_NAME,
            load_fn=self._load_model,
            unload_fn=self._unload_model,
            vram_estimate_gb=0.35,  # ViT-B/14 ~ 350 MB
            priority=9,  # High priority: shared backbone
            device=self._device_override,
        )

    def _load_model(self) -> Any:
        """Load DINOv2 from PyTorch Hub."""
        import torch

        logger.info("Loading DINOv2 model '%s' ...", self._variant)
        model = torch.hub.load("facebookresearch/dinov2", self._variant)

        device_str = self._device_override or (
            self._orchestrator.device if self._orchestrator else "cpu"
        )
        device = torch.device(device_str)
        model = model.to(device)
        model.eval()

        # Build transform
        self._transform = self._build_transform()

        logger.info("DINOv2 loaded on %s", device_str)
        return model

    @staticmethod
    def _unload_model(model: Any) -> None:
        """Move model to CPU and release."""
        try:
            model.cpu()
        except Exception:
            pass

    # ── Standalone mode ───────────────────────────────────────────────

    def _ensure_loaded(self) -> Any:
        """Get the loaded model, loading if necessary."""
        if self._orchestrator is not None:
            return self._orchestrator.acquire(self.MODEL_NAME)

        with self._lock:
            if not self._loaded:
                self._model = self._load_model()
                self._loaded = True
            return self._model

    def _release(self) -> None:
        """Release the model after use."""
        if self._orchestrator is not None:
            self._orchestrator.release(self.MODEL_NAME)

    # ── Transform ─────────────────────────────────────────────────────

    @staticmethod
    def _build_transform() -> Any:
        """Build the image preprocessing transform."""
        from torchvision import transforms

        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(_INPUT_SIZE, interpolation=3),  # BICUBIC
                transforms.CenterCrop(_INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _preprocess(self, image: np.ndarray) -> Any:
        """Convert BGR numpy image to a batched tensor."""
        import torch

        if self._transform is None:
            self._transform = self._build_transform()

        # BGR → RGB
        if image.ndim == 3 and image.shape[2] == 3:
            rgb = image[:, :, ::-1].copy()
        else:
            rgb = image

        tensor = self._transform(rgb)
        return tensor.unsqueeze(0)  # (1, 3, H, W)

    # ── Feature extraction ────────────────────────────────────────────

    def extract_features(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Extract DINOv2 features from an image.

        Parameters
        ----------
        image:
            BGR image as numpy array (H, W, 3).

        Returns
        -------
        dict with keys:
            ``"cls"`` — CLS token embedding, shape (768,)
            ``"patches"`` — Patch token embeddings, shape (N_patches, 768)
        """
        import torch

        model = self._ensure_loaded()
        try:
            device = next(model.parameters()).device
            batch = self._preprocess(image).to(device)

            with torch.no_grad():
                # DINOv2 forward_features returns dict with different keys
                # depending on version; we use the direct forward
                features = model.forward_features(batch)

                # features is a dict: {
                #   "x_norm_clstoken": (B, D),
                #   "x_norm_patchtokens": (B, N, D),
                #   ...
                # }
                if isinstance(features, dict):
                    cls_token = features.get(
                        "x_norm_clstoken",
                        features.get("cls_token", None),
                    )
                    patch_tokens = features.get(
                        "x_norm_patchtokens",
                        features.get("patch_tokens", None),
                    )

                    if cls_token is None:
                        # Fallback: treat first token as CLS
                        all_tokens = next(iter(features.values()))
                        if all_tokens.ndim == 3:
                            cls_token = all_tokens[:, 0]
                            patch_tokens = all_tokens[:, 1:]
                        else:
                            cls_token = all_tokens
                            patch_tokens = all_tokens.unsqueeze(1)
                else:
                    # Tensor output: first token is CLS
                    cls_token = features[:, 0]
                    patch_tokens = features[:, 1:]

            cls_np = cls_token[0].cpu().numpy().astype(np.float32)
            patches_np = patch_tokens[0].cpu().numpy().astype(np.float32)

            return {"cls": cls_np, "patches": patches_np}
        finally:
            self._release()

    def extract_cls(self, image: np.ndarray) -> np.ndarray:
        """Extract only the CLS token (768-dim).  Faster for Re-ID / classification."""
        return self.extract_features(image)["cls"]

    # ── Re-ID embedding ───────────────────────────────────────────────

    def extract_reid_embedding(self, crop: np.ndarray) -> np.ndarray:
        """Extract a person Re-ID embedding from a bounding box crop.

        Parameters
        ----------
        crop:
            BGR person crop (H, W, 3).

        Returns
        -------
        np.ndarray
            L2-normalized 768-dim embedding.  Zero vector on failure.
        """
        if crop is None or crop.size == 0:
            return np.zeros(_EMBEDDING_DIM, dtype=np.float32)

        try:
            cls = self.extract_cls(crop)
            # L2 normalize
            norm = np.linalg.norm(cls)
            if norm > 1e-8:
                cls = cls / norm
            return cls
        except Exception:
            logger.debug("DINOv2 Re-ID extraction failed; returning zero vector")
            return np.zeros(_EMBEDDING_DIM, dtype=np.float32)

    @property
    def embedding_dim(self) -> int:
        return _EMBEDDING_DIM

    # ── Scene classification ──────────────────────────────────────────

    def classify_scene(self, frame: np.ndarray) -> dict[str, float]:
        """Classify the scene type using CLS token similarity.

        Returns a dict of scene labels → confidence scores (softmax).
        This uses pre-computed text-image alignment (no CLIP needed).
        Without prototypes, returns uniform distribution.
        """
        cls = self.extract_cls(frame)

        if self._scene_prototypes is None:
            # No prototypes yet — return uniform
            n = len(_SCENE_LABELS)
            return {label: 1.0 / n for label in _SCENE_LABELS}

        # Cosine similarity with each prototype
        scores: dict[str, float] = {}
        for label, proto in self._scene_prototypes.items():
            sim = float(np.dot(cls, proto) / (
                np.linalg.norm(cls) * np.linalg.norm(proto) + 1e-8
            ))
            scores[label] = sim

        # Softmax
        max_score = max(scores.values())
        exp_scores = {k: np.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        return {k: float(v / total) for k, v in exp_scores.items()}

    def update_scene_prototype(self, label: str, features: np.ndarray) -> None:
        """Update a scene prototype with EMA from new features."""
        if self._scene_prototypes is None:
            self._scene_prototypes = {}

        alpha = 0.1
        if label in self._scene_prototypes:
            self._scene_prototypes[label] = (
                (1 - alpha) * self._scene_prototypes[label] + alpha * features
            )
        else:
            self._scene_prototypes[label] = features.copy()

    # ── Anomaly scoring (Mahalanobis distance) ────────────────────────

    def update_feature_stats(self, features: np.ndarray) -> None:
        """Update running mean and covariance for anomaly detection.

        Call this on every frame to build a feature-space baseline.
        """
        self._feature_count += 1
        self._feature_sum += features.astype(np.float64)
        outer = np.outer(features, features).astype(np.float64)
        self._feature_sq_sum += outer

        # Recompute inverse covariance periodically (every 100 frames)
        if self._feature_count % 100 == 0 and self._feature_count >= 50:
            self._recompute_cov_inv()

    def _recompute_cov_inv(self) -> None:
        """Recompute the inverse covariance matrix."""
        n = self._feature_count
        if n < 2:
            return

        mean = self._feature_sum / n
        cov = self._feature_sq_sum / n - np.outer(mean, mean)

        # Regularize for numerical stability
        cov += np.eye(_EMBEDDING_DIM) * 1e-5

        try:
            self._feature_cov_inv = np.linalg.inv(cov).astype(np.float32)
            self._feature_mean = mean.astype(np.float32)
        except np.linalg.LinAlgError:
            logger.debug("Covariance inversion failed; skipping update")

    def anomaly_score(self, frame: np.ndarray) -> float:
        """Compute Mahalanobis distance anomaly score for a frame.

        Returns 0.0 if insufficient data for statistics.
        Higher values = more anomalous.
        """
        if self._feature_mean is None or self._feature_cov_inv is None:
            return 0.0

        cls = self.extract_cls(frame)
        diff = cls - self._feature_mean
        # Mahalanobis: sqrt(diff^T @ cov_inv @ diff)
        mahal_sq = float(diff @ self._feature_cov_inv @ diff)
        return float(np.sqrt(max(0.0, mahal_sq)))

    @property
    def feature_count(self) -> int:
        return self._feature_count

    # ── Attention map ─────────────────────────────────────────────────

    def get_attention_map(self, image: np.ndarray) -> np.ndarray:
        """Extract self-attention map from the last layer.

        Returns an attention map of shape (H_patches, W_patches) where
        each value represents how much attention the CLS token pays to
        that patch.  Useful for SAM point-prompt generation.

        Returns a uniform map on failure.
        """
        import torch

        model = self._ensure_loaded()
        try:
            device = next(model.parameters()).device
            batch = self._preprocess(image).to(device)

            # Number of patches along each dimension
            n_patches_side = _INPUT_SIZE // _PATCH_SIZE  # 37

            with torch.no_grad():
                # Get attention from last block
                # DINOv2 models store attention in the last block's attn module
                attentions = self._extract_attention(model, batch)

            if attentions is not None:
                # attentions shape: (B, num_heads, N+1, N+1)
                # We want CLS→patch attention, averaged over heads
                # CLS is token 0, patches are tokens 1:
                attn = attentions[0].mean(dim=0)  # (N+1, N+1) — avg over heads
                cls_attn = attn[0, 1:]  # (N,) — CLS → each patch
                cls_attn = cls_attn.cpu().numpy().astype(np.float32)

                # Reshape to spatial grid
                n_patches = cls_attn.shape[0]
                side = int(np.sqrt(n_patches))
                if side * side == n_patches:
                    attn_map = cls_attn.reshape(side, side)
                else:
                    attn_map = cls_attn.reshape(n_patches_side, -1)

                # Normalize to [0, 1]
                attn_min = attn_map.min()
                attn_max = attn_map.max()
                if attn_max - attn_min > 1e-8:
                    attn_map = (attn_map - attn_min) / (attn_max - attn_min)

                return attn_map

            # Fallback: uniform attention
            return np.ones((n_patches_side, n_patches_side), dtype=np.float32) / (
                n_patches_side * n_patches_side
            )
        finally:
            self._release()

    @staticmethod
    def _extract_attention(model: Any, batch: Any) -> Any:
        """Extract attention weights from DINOv2's last transformer block.

        DINOv2 models have a `get_intermediate_layers` method or we can
        hook into the last attention block.
        """
        import torch

        attentions = None

        def hook_fn(module: Any, input: Any, output: Any) -> None:
            nonlocal attentions
            # MultiheadAttention stores attn weights as second output
            if isinstance(output, tuple) and len(output) >= 2:
                attentions = output[1]

        # Find the last attention module
        last_block = None
        for block in getattr(model, "blocks", []):
            last_block = block

        if last_block is None:
            return None

        attn_module = getattr(last_block, "attn", None)
        if attn_module is None:
            return None

        # Register forward hook
        handle = attn_module.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                model(batch)
        finally:
            handle.remove()

        return attentions

    # ── Cleanup ───────────────────────────────────────────────────────

    def close(self) -> None:
        """Release resources in standalone mode."""
        if self._orchestrator is None and self._model is not None:
            self._unload_model(self._model)
            self._model = None
            self._loaded = False

    def get_status(self) -> dict[str, Any]:
        """Return backbone status for API/debug."""
        loaded = (
            self._orchestrator.is_loaded(self.MODEL_NAME)
            if self._orchestrator
            else self._loaded
        )
        return {
            "variant": self._variant,
            "embedding_dim": _EMBEDDING_DIM,
            "loaded": loaded,
            "feature_count": self._feature_count,
            "has_anomaly_baseline": self._feature_mean is not None,
            "scene_prototypes": (
                list(self._scene_prototypes.keys())
                if self._scene_prototypes
                else []
            ),
        }
