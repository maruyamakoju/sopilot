"""Privacy-Preserving Federated Re-ID — cross-camera identity federation.

Phase 41: Enables cameras to share person re-identification embeddings
while preserving individual privacy via differential privacy (DP).

Research contribution:
    Uses the Gaussian mechanism (Dwork & Roth, 2014) to add calibrated
    noise to DINOv2 768-dim embeddings before export. Each camera can
    import other cameras' noisy aggregates to improve cross-camera
    matching without ever sharing raw biometric data.

Privacy guarantee:
    (ε, δ)-differential privacy where:
        ε = privacy_epsilon  (default 1.0 — reasonable privacy/utility tradeoff)
        δ = privacy_delta    (default 1e-5 — negligible failure probability)

    Gaussian noise scale: σ = Δf · √(2 ln(1.25/δ)) / ε
    where Δf = sensitivity = max embedding L2 norm (clipped to 1.0)

Architecture::

    Camera A:  DINOv2 embeddings → FederatedReIDStore.add_local_embedding()
               FederatedReIDStore.export_private() → DP-noisy JSON
                   ↓ (share via API)
    Camera B:  FederatedReIDStore.import_from_camera(camera_a_data)
               FederatedReIDStore.cross_camera_match(query) → sorted matches

Usage::

    store = FederatedReIDStore(privacy_epsilon=1.0)
    store.add_local_embedding("person_42", embedding_768d, "camera_01")
    export_data = store.export_private(epsilon_budget=0.5)
    # Share export_data with peer cameras via API

    store_b.import_from_camera("camera_01", export_data)
    matches = store_b.cross_camera_match(query_embedding, top_k=5)
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 768  # DINOv2 ViT-B CLS token dimension
_CLIP_NORM = 1.0       # L2 sensitivity bound for DP
_MAX_LOCAL = 1000      # max local embeddings to store
_MAX_IMPORTED = 5000   # max imported embeddings to store


# ---------------------------------------------------------------------------
# Privacy Budget
# ---------------------------------------------------------------------------


@dataclass
class PrivacyBudget:
    """Differential privacy budget tracker.

    Parameters
    ----------
    epsilon:
        Privacy budget (lower = more privacy, less utility).
    delta:
        Failure probability for approximate DP guarantee.
    max_queries:
        Maximum export queries before budget is exhausted.
    """

    epsilon: float = 1.0
    delta: float = 1e-5
    max_queries: int = 100
    queries_used: int = 0

    @property
    def epsilon_per_query(self) -> float:
        """Privacy cost per export query."""
        return self.epsilon / max(1, self.max_queries)

    @property
    def remaining_budget(self) -> float:
        """Remaining epsilon budget (basic composition)."""
        used = self.queries_used * self.epsilon_per_query
        return max(0.0, self.epsilon - used)

    @property
    def is_exhausted(self) -> bool:
        return self.queries_used >= self.max_queries

    def noise_scale(self, sensitivity: float = _CLIP_NORM) -> float:
        """Calibrated Gaussian noise scale for (ε, δ)-DP.

        σ = Δ₂ · √(2 ln(1.25/δ)) / ε_per_query
        """
        import math

        eps = self.epsilon_per_query
        if eps <= 0:
            return float("inf")
        return sensitivity * math.sqrt(2 * math.log(1.25 / max(self.delta, 1e-12))) / eps

    def to_dict(self) -> dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "max_queries": self.max_queries,
            "queries_used": self.queries_used,
            "remaining_budget": round(self.remaining_budget, 4),
            "is_exhausted": self.is_exhausted,
            "noise_scale": round(self.noise_scale(), 4),
        }


# ---------------------------------------------------------------------------
# Stored embeddings
# ---------------------------------------------------------------------------


@dataclass
class LocalEmbedding:
    """A single locally-observed person embedding."""

    entry_id: str
    person_id: str
    camera_id: str
    embedding: np.ndarray
    timestamp: float
    confidence: float = 1.0


@dataclass
class ImportedEmbedding:
    """A differentially private embedding imported from a peer camera."""

    entry_id: str
    person_id: str          # peer camera's person ID (may not match local)
    source_camera: str
    embedding: np.ndarray   # noisy (DP-protected) embedding
    timestamp: float
    dp_epsilon: float       # privacy cost used for this embedding


# ---------------------------------------------------------------------------
# FederatedReIDStore
# ---------------------------------------------------------------------------


class FederatedReIDStore:
    """Privacy-preserving federated re-identification store.

    Collects local DINOv2 embeddings, applies Gaussian DP noise before
    export, and allows importing from peer cameras for cross-camera matching.

    Parameters
    ----------
    privacy_epsilon:
        Total privacy budget ε (default 1.0).
    privacy_delta:
        DP failure probability δ (default 1e-5).
    max_export_queries:
        Maximum times export_private() can be called before budget exhausts.
    camera_id:
        This camera's identifier.
    persist_path:
        Optional path for JSON persistence.
    """

    def __init__(
        self,
        privacy_epsilon: float = 1.0,
        privacy_delta: float = 1e-5,
        max_export_queries: int = 100,
        camera_id: str = "local",
        persist_path: Path | None = None,
    ) -> None:
        self._camera_id = camera_id
        self._budget = PrivacyBudget(
            epsilon=privacy_epsilon,
            delta=privacy_delta,
            max_queries=max_export_queries,
        )
        self._persist_path = persist_path

        self._lock = threading.Lock()
        self._local: list[LocalEmbedding] = []
        self._imported: list[ImportedEmbedding] = []
        self._import_stats: dict[str, int] = {}  # source_camera → count
        self._match_count = 0
        self._created_at = time.time()

        if persist_path:
            self._load()

        logger.info(
            "FederatedReIDStore created (camera=%s, ε=%.2f, δ=%.0e, max_q=%d)",
            camera_id,
            privacy_epsilon,
            privacy_delta,
            max_export_queries,
        )

    # ── Local embeddings ───────────────────────────────────────────────

    def add_local_embedding(
        self,
        person_id: str,
        embedding: np.ndarray,
        camera_id: str | None = None,
        confidence: float = 1.0,
    ) -> str:
        """Store a locally-observed person embedding.

        Parameters
        ----------
        person_id:
            Local person track identifier.
        embedding:
            L2-normalized DINOv2 768-dim embedding.
        camera_id:
            Camera identifier (defaults to self._camera_id).
        confidence:
            Observation confidence [0, 1].

        Returns
        -------
        str
            Entry ID (UUID4).
        """
        cam = camera_id or self._camera_id
        entry_id = str(uuid.uuid4())

        # L2-normalize and clip sensitivity
        emb = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm

        entry = LocalEmbedding(
            entry_id=entry_id,
            person_id=person_id,
            camera_id=cam,
            embedding=emb,
            timestamp=time.time(),
            confidence=float(confidence),
        )

        with self._lock:
            self._local.append(entry)
            # Rolling window: keep only last _MAX_LOCAL entries
            if len(self._local) > _MAX_LOCAL:
                self._local = self._local[-_MAX_LOCAL:]

        logger.debug(
            "FederatedReIDStore: added local embedding for %s (total=%d)",
            person_id,
            len(self._local),
        )
        return entry_id

    def get_local_count(self) -> int:
        with self._lock:
            return len(self._local)

    # ── DP Export ──────────────────────────────────────────────────────

    def export_private(self, epsilon_budget: float | None = None) -> dict[str, Any]:
        """Export local embeddings with Gaussian DP noise.

        The exported embeddings are (ε_per_query, δ)-differentially private:
        no single individual's presence or absence changes the output
        distribution by more than e^ε_per_query × δ probability mass.

        Parameters
        ----------
        epsilon_budget:
            Override epsilon for this export. Uses per-query budget if None.

        Returns
        -------
        dict
            Serializable export packet::

                {
                    "source_camera": str,
                    "export_epsilon": float,
                    "noise_scale": float,
                    "timestamp": float,
                    "embeddings": [
                        {
                            "person_id": str,
                            "embedding_b64": str,  # base64 float32
                            "confidence": float,
                        }
                    ]
                }

        Raises
        ------
        RuntimeError
            If privacy budget is exhausted.
        """
        import base64

        with self._lock:
            if self._budget.is_exhausted:
                raise RuntimeError(
                    f"Privacy budget exhausted after {self._budget.queries_used} queries"
                )
            self._budget.queries_used += 1
            noise_sigma = self._budget.noise_scale()
            eps_used = self._budget.epsilon_per_query
            local_copy = list(self._local)

        if epsilon_budget is not None and epsilon_budget > 0:
            import math
            noise_sigma = _CLIP_NORM * math.sqrt(
                2 * math.log(1.25 / max(self._budget.delta, 1e-12))
            ) / epsilon_budget
            eps_used = epsilon_budget

        rng = np.random.default_rng()
        embeddings_out = []
        for entry in local_copy:
            # Add calibrated Gaussian noise
            noise = rng.normal(0, noise_sigma, entry.embedding.shape).astype(np.float32)
            noisy_emb = entry.embedding + noise
            # Re-normalize (noisy embedding may not be unit norm)
            n = np.linalg.norm(noisy_emb)
            if n > 1e-8:
                noisy_emb /= n

            embeddings_out.append({
                "person_id": entry.person_id,
                "embedding_b64": base64.b64encode(noisy_emb.tobytes()).decode("ascii"),
                "confidence": entry.confidence,
                "timestamp": entry.timestamp,
            })

        export_packet = {
            "source_camera": self._camera_id,
            "export_epsilon": round(eps_used, 6),
            "noise_scale": round(noise_sigma, 6),
            "timestamp": time.time(),
            "embedding_dim": _EMBEDDING_DIM,
            "count": len(embeddings_out),
            "embeddings": embeddings_out,
        }

        logger.info(
            "FederatedReIDStore: exported %d embeddings (σ=%.3f, ε_used=%.4f, "
            "budget_remaining=%.4f)",
            len(embeddings_out),
            noise_sigma,
            eps_used,
            self._budget.remaining_budget,
        )
        return export_packet

    # ── Import from peer cameras ───────────────────────────────────────

    def import_from_camera(
        self, source_camera: str, data: dict[str, Any]
    ) -> int:
        """Import DP-protected embeddings from a peer camera.

        Parameters
        ----------
        source_camera:
            Peer camera identifier.
        data:
            Export packet from :meth:`export_private`.

        Returns
        -------
        int
            Number of embeddings successfully imported.
        """
        import base64

        embeddings_raw = data.get("embeddings", [])
        dp_epsilon = data.get("export_epsilon", 1.0)
        dim = data.get("embedding_dim", _EMBEDDING_DIM)

        imported: list[ImportedEmbedding] = []
        for e in embeddings_raw:
            try:
                raw = base64.b64decode(e["embedding_b64"])
                emb = np.frombuffer(raw, dtype=np.float32).copy()
                if len(emb) != dim:
                    continue
                imported.append(
                    ImportedEmbedding(
                        entry_id=str(uuid.uuid4()),
                        person_id=e["person_id"],
                        source_camera=source_camera,
                        embedding=emb,
                        timestamp=e.get("timestamp", time.time()),
                        dp_epsilon=dp_epsilon,
                    )
                )
            except Exception:
                continue

        with self._lock:
            self._imported.extend(imported)
            # Rolling window
            if len(self._imported) > _MAX_IMPORTED:
                self._imported = self._imported[-_MAX_IMPORTED:]
            self._import_stats[source_camera] = (
                self._import_stats.get(source_camera, 0) + len(imported)
            )

        logger.info(
            "FederatedReIDStore: imported %d embeddings from camera %s (ε=%.4f)",
            len(imported),
            source_camera,
            dp_epsilon,
        )
        return len(imported)

    # ── Cross-camera matching ──────────────────────────────────────────

    def cross_camera_match(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_similarity: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Find the best matching identities across all cameras.

        Computes cosine similarity between query and all imported
        (DP-noised) embeddings. Returns top-k matches above threshold.

        Parameters
        ----------
        query_embedding:
            L2-normalized query embedding (768-dim).
        top_k:
            Maximum matches to return.
        min_similarity:
            Minimum cosine similarity to include a match.

        Returns
        -------
        list[dict]
            Sorted by similarity (descending)::

                [{"person_id": str, "source_camera": str, "similarity": float}]
        """
        query = np.asarray(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 1e-8:
            query = query / norm

        with self._lock:
            imported = list(self._imported)
            self._match_count += 1

        if not imported:
            return []

        # Vectorized cosine similarity
        emb_matrix = np.stack([e.embedding for e in imported], axis=0)  # (N, D)
        similarities = emb_matrix @ query  # cosine sim (embeddings are L2-normalized)

        # Get top-k above threshold
        top_indices = np.argsort(similarities)[::-1][:top_k]
        matches = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim < min_similarity:
                break
            entry = imported[idx]
            matches.append({
                "person_id": entry.person_id,
                "source_camera": entry.source_camera,
                "similarity": round(sim, 4),
                "dp_epsilon": entry.dp_epsilon,
                "timestamp": entry.timestamp,
            })

        return matches

    # ── Statistics ─────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Return comprehensive stats including privacy budget."""
        with self._lock:
            return {
                "camera_id": self._camera_id,
                "local_embeddings": len(self._local),
                "imported_embeddings": len(self._imported),
                "peer_cameras": list(self._import_stats.keys()),
                "import_stats": dict(self._import_stats),
                "match_count": self._match_count,
                "privacy_budget": self._budget.to_dict(),
                "uptime_seconds": round(time.time() - self._created_at, 1),
            }

    # ── Persistence ────────────────────────────────────────────────────

    def _save(self) -> None:
        """Persist local embeddings and budget to JSON (best-effort)."""
        import base64

        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            data = {
                "camera_id": self._camera_id,
                "budget_queries_used": self._budget.queries_used,
                "local": [
                    {
                        "entry_id": e.entry_id,
                        "person_id": e.person_id,
                        "camera_id": e.camera_id,
                        "embedding_b64": base64.b64encode(
                            e.embedding.astype(np.float32).tobytes()
                        ).decode("ascii"),
                        "timestamp": e.timestamp,
                        "confidence": e.confidence,
                    }
                    for e in self._local
                ],
            }

        with open(self._persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _load(self) -> None:
        """Load persisted local embeddings from JSON."""
        import base64

        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._budget.queries_used = data.get("budget_queries_used", 0)
            for e in data.get("local", []):
                raw = base64.b64decode(e["embedding_b64"])
                emb = np.frombuffer(raw, dtype=np.float32).copy()
                self._local.append(
                    LocalEmbedding(
                        entry_id=e["entry_id"],
                        person_id=e["person_id"],
                        camera_id=e["camera_id"],
                        embedding=emb,
                        timestamp=e["timestamp"],
                        confidence=e.get("confidence", 1.0),
                    )
                )
            logger.info(
                "FederatedReIDStore: loaded %d local embeddings from %s",
                len(self._local),
                self._persist_path,
            )
        except Exception:
            logger.debug("Failed to load federated Re-ID store", exc_info=True)
