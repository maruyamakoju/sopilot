"""Person appearance embedding extractor for cross-camera Re-ID.

Phase 28B — Multi-Camera Person Re-Identification.

Uses an HSV color histogram of the bounding box crop as the appearance
descriptor.  This is resolution-independent, does not require a trained model,
and is fast enough for real-time use (< 1ms on CPU).

The descriptor is a 512-dimensional float32 vector (8×8×8 HSV bins, L2-
normalised).  Cosine similarity is used for matching.

If a richer embedding (e.g. from a ReID neural network) becomes available it
can be plugged in as a replacement for ``_color_histogram`` without changing
the gallery interface.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Histogram configuration
_H_BINS = 8
_S_BINS = 8
_V_BINS = 8
_DIM = _H_BINS * _S_BINS * _V_BINS  # 512


class ReIDExtractor:
    """Extracts appearance embeddings from person crops.

    Parameters
    ----------
    embedder:
        Optional external embedder with ``embed(frame: np.ndarray) -> np.ndarray``
        interface.  If provided and the frame is a valid crop, its output is
        used; otherwise falls back to the color histogram.
    """

    def __init__(self, embedder: Any = None) -> None:
        self._embedder = embedder

    @property
    def embedding_dim(self) -> int:
        return _DIM

    def extract(self, frame: np.ndarray, bbox: Any) -> np.ndarray:
        """Extract a normalised appearance embedding for a detected person.

        Parameters
        ----------
        frame:
            Full BGR frame (H × W × 3).
        bbox:
            An object with attributes ``x1, y1, x2, y2`` in [0, 1] (normalised).

        Returns
        -------
        np.ndarray
            Shape (512,), dtype float32, L2-normalised.  Zero vector on failure.
        """
        crop = self._crop(frame, bbox)
        if crop is None or crop.size == 0:
            return np.zeros(_DIM, dtype=np.float32)

        if self._embedder is not None:
            try:
                emb = self._embedder.embed(crop)
                if emb is not None and emb.ndim == 1 and emb.shape[0] > 0:
                    return self._l2_norm(emb.astype(np.float32))
            except Exception:
                logger.debug("External embedder failed; falling back to histogram")

        return _color_histogram(crop)

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity in [0, 1] between two embedding vectors."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))

    # ── Private ────────────────────────────────────────────────────────────────

    def _crop(self, frame: np.ndarray, bbox: Any) -> np.ndarray | None:
        try:
            h, w = frame.shape[:2]
            x1 = max(0, int(bbox.x1 * w))
            y1 = max(0, int(bbox.y1 * h))
            x2 = min(w, int(bbox.x2 * w))
            y2 = min(h, int(bbox.y2 * h))
            if x2 <= x1 or y2 <= y1:
                return None
            return frame[y1:y2, x1:x2]
        except Exception:
            return None

    @staticmethod
    def _l2_norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else v


def _color_histogram(crop: np.ndarray) -> np.ndarray:
    """Compute a normalised HSV color histogram for *crop*."""
    try:
        import cv2  # type: ignore[import-untyped]

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [_H_BINS, _S_BINS, _V_BINS],
            [0, 180, 0, 256, 0, 256],
        )
        hist = hist.flatten().astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist
    except Exception:
        logger.debug("Color histogram failed")
        return np.zeros(_DIM, dtype=np.float32)
