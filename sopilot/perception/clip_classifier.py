"""CLIP-based zero-shot entity classification.

Provides open-vocabulary entity labeling using CLIP cosine similarity.
Falls back gracefully to MockCLIPBackend when CLIP libraries are not installed.

Backend priority:
  1. "clip"         — OpenAI clip package (pip install clip)
  2. "transformers" — HuggingFace transformers CLIPModel
  3. "mock"         — Deterministic hash-based mock (always available, for testing)
  "auto" selects the first available real backend, or mock if none.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_LABELS = [
    "person", "worker", "car", "truck", "forklift", "box", "equipment",
    "tool", "bicycle", "motorcycle", "safety_cone", "conveyor_belt",
    "helmet", "vest",
]


@dataclass
class CLIPScore:
    entity_id: int
    original_label: str
    refined_label: str
    score: float                      # cosine similarity to best-matching class
    all_scores: dict[str, float]      # label -> score for all candidates
    backend: str

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "original_label": self.original_label,
            "refined_label": self.refined_label,
            "score": round(self.score, 4),
            "all_scores": {k: round(v, 4) for k, v in self.all_scores.items()},
            "backend": self.backend,
        }


@runtime_checkable
class CLIPBackend(Protocol):
    def encode_text(self, texts: list[str]) -> np.ndarray: ...
    def encode_image(self, image: np.ndarray) -> np.ndarray: ...
    @property
    def is_real(self) -> bool: ...


class MockCLIPBackend:
    """Deterministic mock using hash-based pseudo-scores. Always available."""

    is_real = False

    def encode_text(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            rng = np.random.RandomState(h % (2**31))
            v = rng.randn(512).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9
            vecs.append(v)
        return np.stack(vecs, axis=0)  # (N, 512)

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        # Hash of image shape + mean → deterministic
        key = f"{image.shape}_{float(image.mean()):.4f}"
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        rng = np.random.RandomState(h % (2**31))
        v = rng.randn(512).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-9
        return v.reshape(1, 512)  # (1, 512)


class CLIPZeroShotClassifier:
    """Zero-shot entity classification using CLIP similarity scoring.

    Thread-safe via RLock.
    Text embeddings are cached per label set (invalidated on set_labels()).
    """

    def __init__(
        self,
        candidate_labels: list[str] | None = None,
        confidence_threshold: float = 0.25,
        backend: str = "auto",
        model_name: str = "ViT-B/32",
    ) -> None:
        self._labels = list(candidate_labels or DEFAULT_CANDIDATE_LABELS)
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name
        self._lock = threading.RLock()
        self._text_cache: np.ndarray | None = None
        self._cached_labels: list[str] | None = None
        self._stats = {"calls_total": 0, "cache_hits": 0, "refinements_made": 0}
        self._backend: Any = self._load_backend(backend)

    # ── Backend ──────────────────────────────────────────────────────────────

    def _load_backend(self, backend: str) -> Any:
        if backend == "mock":
            logger.debug("CLIP: using mock backend")
            return MockCLIPBackend()
        if backend in ("auto", "clip"):
            try:
                import clip as _clip  # type: ignore
                import torch  # type: ignore
                model, preprocess = _clip.load(self.model_name, device="cpu")
                logger.info("CLIP: loaded openai/clip backend (%s)", self.model_name)
                return _OpenAICLIPWrapper(model, preprocess)
            except Exception:
                if backend == "clip":
                    logger.warning("CLIP: openai/clip not available, falling back to mock")
        if backend in ("auto", "transformers"):
            try:
                from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor  # type: ignore
                model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                logger.info("CLIP: loaded transformers CLIP backend")
                return _TransformersCLIPWrapper(model, processor)
            except Exception:
                if backend == "transformers":
                    logger.warning("CLIP: transformers CLIP not available, falling back to mock")
        logger.debug("CLIP: using mock backend (no real backend available)")
        return MockCLIPBackend()

    @property
    def backend_name(self) -> str:
        return type(self._backend).__name__

    @property
    def is_available(self) -> bool:
        return getattr(self._backend, "is_real", False)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_labels(self, labels: list[str]) -> None:
        """Update candidate labels and invalidate text embedding cache."""
        with self._lock:
            self._labels = list(labels)
            self._text_cache = None
            self._cached_labels = None

    def classify_entities(
        self,
        detections: list,
        frame: np.ndarray | None = None,
    ) -> list[CLIPScore]:
        """Re-classify detected entities using CLIP.

        detections: list of objects with .entity_id, .label, .bbox attributes
        frame: full frame as (H, W, 3) uint8 numpy array, or None
        """
        if not detections:
            return []
        with self._lock:
            self._stats["calls_total"] += 1
            text_embs = self._get_text_embeddings()
            results = []
            for det in detections:
                eid = int(getattr(det, "entity_id", 0))
                orig_label = str(getattr(det, "label", "unknown"))
                crop = self._extract_crop(det, frame)
                score_obj = self._score_entity(eid, orig_label, crop, text_embs)
                if score_obj.refined_label != orig_label:
                    self._stats["refinements_made"] += 1
                results.append(score_obj)
            return results

    def classify_crop(
        self,
        crop: np.ndarray,
        candidate_labels: list[str] | None = None,
    ) -> CLIPScore:
        """Classify a single image crop against candidate_labels (or default labels)."""
        with self._lock:
            labels = candidate_labels or self._labels
            text_embs = self._backend.encode_text(labels)
            img_emb = self._backend.encode_image(crop)
            scores = (img_emb @ text_embs.T).flatten()
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            all_scores = {lbl: float(s) for lbl, s in zip(labels, scores)}
            refined = labels[best_idx] if best_score >= self.confidence_threshold else "unknown"
            return CLIPScore(
                entity_id=-1,
                original_label="",
                refined_label=refined,
                score=best_score,
                all_scores=all_scores,
                backend=self.backend_name,
            )

    def get_state_dict(self) -> dict:
        with self._lock:
            return {
                "backend": self.backend_name,
                "is_available": self.is_available,
                "candidate_labels": list(self._labels),
                "label_count": len(self._labels),
                "confidence_threshold": self.confidence_threshold,
                **self._stats,
            }

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_text_embeddings(self) -> np.ndarray:
        if self._text_cache is not None and self._cached_labels == self._labels:
            self._stats["cache_hits"] += 1
            return self._text_cache
        self._text_cache = self._backend.encode_text(self._labels)
        self._cached_labels = list(self._labels)
        return self._text_cache

    def _extract_crop(self, det: Any, frame: np.ndarray | None) -> np.ndarray:
        if frame is None:
            return np.zeros((32, 32, 3), dtype=np.uint8)
        try:
            bbox = getattr(det, "bbox", None)
            if bbox is None or len(bbox) < 4:
                return frame
            H, W = frame.shape[:2]
            x1 = int(bbox[0] * W)
            y1 = int(bbox[1] * H)
            x2 = int(bbox[2] * W)
            y2 = int(bbox[3] * H)
            x1, x2 = max(0, x1), min(W, x2)
            y1, y2 = max(0, y1), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                return frame
            return frame[y1:y2, x1:x2]
        except Exception:
            return frame if frame is not None else np.zeros((32, 32, 3), dtype=np.uint8)

    def _score_entity(
        self,
        entity_id: int,
        original_label: str,
        crop: np.ndarray,
        text_embs: np.ndarray,
    ) -> CLIPScore:
        img_emb = self._backend.encode_image(crop)
        scores = (img_emb @ text_embs.T).flatten()
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        all_scores = {lbl: float(s) for lbl, s in zip(self._labels, scores)}
        refined = (
            self._labels[best_idx]
            if best_score >= self.confidence_threshold
            else original_label
        )
        return CLIPScore(
            entity_id=entity_id,
            original_label=original_label,
            refined_label=refined,
            score=best_score,
            all_scores=all_scores,
            backend=self.backend_name,
        )


# ── Real backend wrappers (only instantiated if libraries available) ──────────

class _OpenAICLIPWrapper:
    is_real = True

    def __init__(self, model: Any, preprocess: Any) -> None:
        self._model = model
        self._preprocess = preprocess

    def encode_text(self, texts: list[str]) -> np.ndarray:
        import clip  # type: ignore
        import torch  # type: ignore
        tokens = clip.tokenize(texts)
        with torch.no_grad():
            embs = self._model.encode_text(tokens).float()
            embs /= embs.norm(dim=-1, keepdim=True) + 1e-9
        return embs.numpy()

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        import torch  # type: ignore
        from PIL import Image  # type: ignore
        pil = Image.fromarray(image.astype(np.uint8))
        tensor = self._preprocess(pil).unsqueeze(0)
        with torch.no_grad():
            emb = self._model.encode_image(tensor).float()
            emb /= emb.norm(dim=-1, keepdim=True) + 1e-9
        return emb.numpy()


class _TransformersCLIPWrapper:
    is_real = True

    def __init__(self, model: Any, processor: Any) -> None:
        self._model = model
        self._processor = processor

    def encode_text(self, texts: list[str]) -> np.ndarray:
        import torch  # type: ignore
        inputs = self._processor(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            embs = self._model.get_text_features(**inputs).float()
            embs /= embs.norm(dim=-1, keepdim=True) + 1e-9
        return embs.numpy()

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        import torch  # type: ignore
        from PIL import Image  # type: ignore
        pil = Image.fromarray(image.astype(np.uint8))
        inputs = self._processor(images=pil, return_tensors="pt")
        with torch.no_grad():
            emb = self._model.get_image_features(**inputs).float()
            emb /= emb.norm(dim=-1, keepdim=True) + 1e-9
        return emb.numpy()


# ── Factory ───────────────────────────────────────────────────────────────────

def build_clip_classifier(
    labels: list[str] | None = None,
    backend: str = "auto",
    confidence_threshold: float = 0.25,
) -> CLIPZeroShotClassifier:
    return CLIPZeroShotClassifier(
        candidate_labels=labels,
        backend=backend,
        confidence_threshold=confidence_threshold,
    )
