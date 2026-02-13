"""VIGIL-RAG Benchmark Runner.

Loads benchmark queries from JSONL, runs retrieval in visual-only and hybrid modes
using controlled synthetic embeddings, and computes comparison metrics.

Synthetic embedding design:
    - Each micro clip gets a random unit vector (visual embedding).
    - Relevant clips for visual queries have their visual embedding nudged toward
      the query direction (high cosine similarity).
    - Relevant clips for audio queries have their *text* embedding nudged toward
      the query direction (only reachable through hybrid search).
    - Mixed queries have relevant clips nudged in *both* visual and text spaces.

This makes the benchmark deterministic and self-contained — no real video,
Whisper, or LLM required.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sopilot.evaluation.vigil_metrics import (
    evidence_recall_at_k,
    mrr,
    ndcg_at_k,
)
from sopilot.qdrant_service import QdrantConfig, QdrantService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 512
NUM_MICRO_CLIPS = 20  # micro-00 .. micro-19


@dataclass
class BenchmarkQuery:
    """A single benchmark query loaded from JSONL."""

    query_id: str
    video_id: str
    query_text: str
    query_type: str  # "visual" | "audio" | "mixed"
    relevant_clip_ids: list[str]
    relevant_time_ranges: list[dict] = field(default_factory=list)
    event_detection: bool = False
    event_type: str | None = None
    notes: str = ""


@dataclass
class QueryResult:
    """Per-query evaluation result."""

    query_id: str
    query_type: str
    retrieved_clip_ids: list[str]
    relevant_clip_ids: list[str]
    recall_at_5: float
    mrr_score: float
    hit: bool  # at least one relevant in top-5


@dataclass
class ModeResult:
    """Aggregate results for one retrieval mode."""

    mode: str  # "visual_only" | "hybrid"
    alpha: float  # 0.0 for visual_only
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr_score: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    num_queries: int = 0
    # Per-type breakdowns
    by_type: dict[str, dict] = field(default_factory=dict)
    per_query: list[QueryResult] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    """Full comparison report."""

    benchmark_file: str
    num_queries: int
    modes: list[ModeResult]
    deltas: dict[str, dict] = field(default_factory=dict)  # metric -> {query_type -> delta}


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------


class VIGILBenchmarkRunner:
    """Runs VIGIL-RAG retrieval benchmark with synthetic embeddings."""

    def __init__(
        self,
        *,
        seed: int = 42,
        embedding_dim: int = EMBEDDING_DIM,
        num_clips: int = NUM_MICRO_CLIPS,
        signal_strength: float = 0.85,
    ) -> None:
        """Initialize benchmark runner.

        Args:
            seed: Random seed for reproducibility.
            embedding_dim: Embedding dimension (matches OpenCLIP ViT-B-32).
            num_clips: Number of synthetic micro clips per video.
            signal_strength: Cosine similarity between query and relevant clip (0-1).
                Higher = easier benchmark, lower = harder.
        """
        self.seed = seed
        self.dim = embedding_dim
        self.num_clips = num_clips
        self.signal_strength = signal_strength
        self._rng = np.random.RandomState(seed)

    def load_benchmark(self, path: Path | str) -> list[BenchmarkQuery]:
        """Load benchmark queries from JSONL file.

        Args:
            path: Path to benchmark JSONL file.

        Returns:
            List of BenchmarkQuery objects.
        """
        path = Path(path)
        queries: list[BenchmarkQuery] = []
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping invalid JSON at line %d: %s", line_num, exc)
                    continue
                queries.append(
                    BenchmarkQuery(
                        query_id=data["query_id"],
                        video_id=data["video_id"],
                        query_text=data["query_text"],
                        query_type=data["query_type"],
                        relevant_clip_ids=data["relevant_clip_ids"],
                        relevant_time_ranges=data.get("relevant_time_ranges", []),
                        event_detection=data.get("event_detection", False),
                        event_type=data.get("event_type"),
                        notes=data.get("notes", ""),
                    )
                )
        logger.info("Loaded %d benchmark queries from %s", len(queries), path)
        return queries

    def run(
        self,
        queries: list[BenchmarkQuery],
        *,
        alphas: list[float] | None = None,
        top_k: int = 10,
    ) -> BenchmarkReport:
        """Run benchmark with visual-only and hybrid modes.

        Args:
            queries: Loaded benchmark queries.
            alphas: List of alpha values for hybrid mode (default [0.7]).
            top_k: Number of results to retrieve per query.

        Returns:
            BenchmarkReport with comparison across modes.
        """
        if alphas is None:
            alphas = [0.7]

        # Filter to retrieval queries only (skip event detection)
        retrieval_queries = [q for q in queries if not q.event_detection]
        if not retrieval_queries:
            logger.warning("No retrieval queries in benchmark")
            return BenchmarkReport(
                benchmark_file="",
                num_queries=0,
                modes=[],
            )

        # Collect unique video IDs
        video_ids = sorted({q.video_id for q in retrieval_queries})

        # Generate synthetic data for each video
        video_data: dict[str, dict] = {}
        for vid in video_ids:
            vid_queries = [q for q in retrieval_queries if q.video_id == vid]
            video_data[vid] = self._generate_synthetic_data(vid, vid_queries)

        # Run visual-only mode
        modes: list[ModeResult] = []

        visual_result = self._evaluate_mode(
            "visual_only",
            alpha=0.0,
            queries=retrieval_queries,
            video_data=video_data,
            top_k=top_k,
        )
        modes.append(visual_result)

        # Run hybrid modes
        for alpha in alphas:
            hybrid_result = self._evaluate_mode(
                "hybrid",
                alpha=alpha,
                queries=retrieval_queries,
                video_data=video_data,
                top_k=top_k,
            )
            modes.append(hybrid_result)

        # Compute deltas (hybrid vs visual_only)
        deltas = self._compute_deltas(modes)

        return BenchmarkReport(
            benchmark_file="",
            num_queries=len(retrieval_queries),
            modes=modes,
            deltas=deltas,
        )

    # ------------------------------------------------------------------
    # Internal: synthetic data generation
    # ------------------------------------------------------------------

    def _generate_synthetic_data(
        self,
        video_id: str,
        queries: list[BenchmarkQuery],
    ) -> dict:
        """Generate synthetic visual/text embeddings for a video.

        Design:
        - Base visual embeddings: random unit vectors per clip.
        - Base text embeddings: random unit vectors per clip (independent of visual).
        - For each query, nudge relevant clips toward the query direction in the
          appropriate modality (visual, text, or both).
        """
        n = self.num_clips

        # Base embeddings (random directions)
        visual_embs = self._random_unit_vectors(n)
        text_embs = self._random_unit_vectors(n)

        # Metadata per clip
        metadata = []
        for i in range(n):
            clip_id = f"micro-{i:02d}"
            metadata.append(
                {
                    "clip_id": clip_id,
                    "video_id": video_id,
                    "start_sec": float(i * 3),
                    "end_sec": float((i + 1) * 3),
                }
            )

        # Generate a query direction per query and nudge relevant clips
        query_embeddings: dict[str, np.ndarray] = {}
        for q in queries:
            q_dir = self._random_unit_vectors(1)[0]
            query_embeddings[q.query_id] = q_dir

            for clip_id in q.relevant_clip_ids:
                idx = self._clip_id_to_index(clip_id)
                if idx is None or idx >= n:
                    continue

                if q.query_type in ("visual", "mixed"):
                    visual_embs[idx] = self._nudge_toward(visual_embs[idx], q_dir, self.signal_strength)

                if q.query_type in ("audio", "mixed"):
                    text_embs[idx] = self._nudge_toward(text_embs[idx], q_dir, self.signal_strength)

        # Add transcript_text to clips that have text embeddings nudged
        text_clip_ids: set[str] = set()
        for q in queries:
            if q.query_type in ("audio", "mixed"):
                text_clip_ids.update(q.relevant_clip_ids)
        for meta in metadata:
            if meta["clip_id"] in text_clip_ids:
                meta["transcript_text"] = f"[synthetic transcript for {meta['clip_id']}]"

        return {
            "visual_embeddings": visual_embs,
            "text_embeddings": text_embs,
            "metadata": metadata,
            "query_embeddings": query_embeddings,
        }

    # ------------------------------------------------------------------
    # Internal: evaluation per mode
    # ------------------------------------------------------------------

    def _evaluate_mode(
        self,
        mode: str,
        *,
        alpha: float,
        queries: list[BenchmarkQuery],
        video_data: dict[str, dict],
        top_k: int,
    ) -> ModeResult:
        """Evaluate one retrieval mode (visual_only or hybrid).

        Creates a fresh FAISS-backed QdrantService, populates it, and runs queries.
        """
        qdrant = QdrantService(QdrantConfig(host="localhost", port=19999), use_faiss_fallback=True)

        # Populate vector DB per video
        for _vid, data in video_data.items():
            visual = data["visual_embeddings"].astype(np.float32)
            qdrant.add_embeddings("micro", visual, data["metadata"])

            if mode == "hybrid" and alpha > 0:
                text = data["text_embeddings"].astype(np.float32)
                # Only add text embeddings for clips that have transcript
                text_meta = [m for m in data["metadata"] if m.get("transcript_text")]
                if text_meta:
                    text_indices = [self._clip_id_to_index(m["clip_id"]) for m in text_meta]
                    text_vecs = np.array([text[i] for i in text_indices if i is not None], dtype=np.float32)
                    if len(text_vecs) > 0:
                        qdrant.add_embeddings("micro_text", text_vecs, text_meta)

        # Run each query
        all_retrieved: list[list[str]] = []
        all_relevant: list[list[str]] = []
        all_relevant_sets: list[set[str]] = []
        all_relevance_scores: list[dict[str, float]] = []
        per_query_results: list[QueryResult] = []

        for q in queries:
            vdata = video_data[q.video_id]
            q_emb = vdata["query_embeddings"][q.query_id].astype(np.float32)

            # Retrieve
            retrieved_ids = self._retrieve(qdrant, q_emb, mode=mode, alpha=alpha, video_id=q.video_id, top_k=top_k)

            all_retrieved.append(retrieved_ids)
            all_relevant.append(q.relevant_clip_ids)
            all_relevant_sets.append(set(q.relevant_clip_ids))

            # Binary relevance for nDCG
            rel_map = {cid: 1.0 for cid in q.relevant_clip_ids}
            all_relevance_scores.append(rel_map)

            # Per-query recall@5 and MRR
            r5 = len(set(retrieved_ids[:5]) & set(q.relevant_clip_ids)) / max(len(q.relevant_clip_ids), 1)
            first_rank = 0.0
            for rank, cid in enumerate(retrieved_ids, 1):
                if cid in set(q.relevant_clip_ids):
                    first_rank = 1.0 / rank
                    break

            per_query_results.append(
                QueryResult(
                    query_id=q.query_id,
                    query_type=q.query_type,
                    retrieved_clip_ids=retrieved_ids,
                    relevant_clip_ids=q.relevant_clip_ids,
                    recall_at_5=r5,
                    mrr_score=first_rank,
                    hit=r5 > 0,
                )
            )

        # Compute aggregate metrics
        recall_result = evidence_recall_at_k(all_retrieved, all_relevant)
        mrr_score = mrr(all_retrieved, all_relevant_sets)
        ndcg_5 = ndcg_at_k(all_retrieved, all_relevance_scores, k=5)
        ndcg_10 = ndcg_at_k(all_retrieved, all_relevance_scores, k=10)

        # Per query-type breakdown
        by_type: dict[str, dict] = {}
        for qt in ("visual", "audio", "mixed"):
            qt_queries = [pq for pq in per_query_results if pq.query_type == qt]
            if qt_queries:
                by_type[qt] = {
                    "num_queries": len(qt_queries),
                    "recall_at_5": sum(pq.recall_at_5 for pq in qt_queries) / len(qt_queries),
                    "mrr": sum(pq.mrr_score for pq in qt_queries) / len(qt_queries),
                    "hit_rate": sum(1 for pq in qt_queries if pq.hit) / len(qt_queries),
                }

        mode_label = f"hybrid(alpha={alpha})" if mode == "hybrid" else "visual_only"
        return ModeResult(
            mode=mode_label,
            alpha=alpha,
            recall_at_1=recall_result.recall_at_1,
            recall_at_3=recall_result.recall_at_3,
            recall_at_5=recall_result.recall_at_5,
            recall_at_10=recall_result.recall_at_10,
            mrr_score=mrr_score,
            ndcg_at_5=ndcg_5,
            ndcg_at_10=ndcg_10,
            num_queries=len(queries),
            by_type=by_type,
            per_query=per_query_results,
        )

    def _retrieve(
        self,
        qdrant: QdrantService,
        query_embedding: np.ndarray,
        *,
        mode: str,
        alpha: float,
        video_id: str | None,
        top_k: int,
    ) -> list[str]:
        """Run retrieval and return clip IDs (visual-only or hybrid fusion)."""
        visual_results = qdrant.search(
            level="micro",
            query_vector=query_embedding,
            top_k=top_k,
            video_id=video_id,
        )

        if mode == "visual_only" or alpha <= 0:
            return [r.clip_id for r in visual_results]

        # Hybrid: also search micro_text
        try:
            audio_results = qdrant.search(
                level="micro_text",
                query_vector=query_embedding,
                top_k=top_k,
                video_id=video_id,
            )
        except Exception:
            return [r.clip_id for r in visual_results]

        if not audio_results:
            return [r.clip_id for r in visual_results]

        # Fuse: max(visual_score, alpha * audio_score) — same as RAGService._hybrid_search
        visual_map = {r.clip_id: r.score for r in visual_results}
        audio_map = {r.clip_id: r.score for r in audio_results}

        all_clip_ids = set(visual_map) | set(audio_map)
        fused: list[tuple[str, float]] = []
        for cid in all_clip_ids:
            v_score = visual_map.get(cid, 0.0)
            a_score = audio_map.get(cid, 0.0) * alpha
            fused.append((cid, max(v_score, a_score)))

        fused.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in fused[:top_k]]

    # ------------------------------------------------------------------
    # Internal: comparison / deltas
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_deltas(modes: list[ModeResult]) -> dict[str, dict]:
        """Compute metric deltas between hybrid modes and visual_only baseline."""
        baseline = modes[0]  # visual_only is always first

        deltas: dict[str, dict] = {}
        for mode in modes[1:]:
            key = mode.mode
            deltas[key] = {
                "overall": {
                    "recall_at_5": mode.recall_at_5 - baseline.recall_at_5,
                    "mrr": mode.mrr_score - baseline.mrr_score,
                    "ndcg_at_5": mode.ndcg_at_5 - baseline.ndcg_at_5,
                    "ndcg_at_10": mode.ndcg_at_10 - baseline.ndcg_at_10,
                },
            }
            # Per-type deltas
            for qt in ("visual", "audio", "mixed"):
                base_bt = baseline.by_type.get(qt, {})
                mode_bt = mode.by_type.get(qt, {})
                if base_bt and mode_bt:
                    deltas[key][qt] = {
                        "recall_at_5": mode_bt["recall_at_5"] - base_bt["recall_at_5"],
                        "mrr": mode_bt["mrr"] - base_bt["mrr"],
                    }
        return deltas

    # ------------------------------------------------------------------
    # Internal: vector utilities
    # ------------------------------------------------------------------

    def _random_unit_vectors(self, n: int) -> np.ndarray:
        """Generate n random unit vectors of shape (n, dim)."""
        vecs = self._rng.randn(n, self.dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        return vecs / norms

    @staticmethod
    def _nudge_toward(vec: np.ndarray, target: np.ndarray, strength: float) -> np.ndarray:
        """Nudge vec toward target direction by strength (0-1).

        Result is normalized to unit length.
        """
        nudged = (1 - strength) * vec + strength * target
        return nudged / (np.linalg.norm(nudged) + 1e-9)

    @staticmethod
    def _clip_id_to_index(clip_id: str) -> int | None:
        """Convert 'micro-03' -> 3."""
        try:
            return int(clip_id.split("-")[1])
        except (IndexError, ValueError):
            return None


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_report(report: BenchmarkReport) -> str:
    """Format a BenchmarkReport as human-readable text."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("VIGIL-RAG Benchmark Report")
    lines.append(f"  Benchmark: {report.benchmark_file}")
    lines.append(f"  Queries: {report.num_queries} (retrieval only)")
    lines.append("=" * 72)

    for mode in report.modes:
        lines.append("")
        lines.append(f"--- {mode.mode} ---")
        lines.append(f"  Recall@1 = {mode.recall_at_1:.4f}")
        lines.append(f"  Recall@3 = {mode.recall_at_3:.4f}")
        lines.append(f"  Recall@5 = {mode.recall_at_5:.4f}")
        lines.append(f"  Recall@10= {mode.recall_at_10:.4f}")
        lines.append(f"  MRR      = {mode.mrr_score:.4f}")
        lines.append(f"  nDCG@5   = {mode.ndcg_at_5:.4f}")
        lines.append(f"  nDCG@10  = {mode.ndcg_at_10:.4f}")

        if mode.by_type:
            lines.append("")
            lines.append("  By query type:")
            for qt in ("visual", "audio", "mixed"):
                bt = mode.by_type.get(qt)
                if bt:
                    lines.append(
                        f"    {qt:8s}  n={bt['num_queries']:2d}  "
                        f"R@5={bt['recall_at_5']:.4f}  "
                        f"MRR={bt['mrr']:.4f}  "
                        f"hit={bt['hit_rate']:.0%}"
                    )

    # Deltas
    if report.deltas:
        lines.append("")
        lines.append("--- Delta (hybrid - visual_only) ---")
        for mode_key, delta_data in report.deltas.items():
            lines.append(f"  {mode_key}:")
            overall = delta_data.get("overall", {})
            lines.append(
                f"    overall  dR@5={overall.get('recall_at_5', 0):+.4f}  "
                f"dMRR={overall.get('mrr', 0):+.4f}  "
                f"dnDCG@5={overall.get('ndcg_at_5', 0):+.4f}"
            )
            for qt in ("visual", "audio", "mixed"):
                qt_delta = delta_data.get(qt)
                if qt_delta:
                    lines.append(f"    {qt:8s}  dR@5={qt_delta['recall_at_5']:+.4f}  dMRR={qt_delta['mrr']:+.4f}")

    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


def report_to_dict(report: BenchmarkReport) -> dict:
    """Convert BenchmarkReport to a JSON-serializable dict."""
    modes_out = []
    for m in report.modes:
        modes_out.append(
            {
                "mode": m.mode,
                "alpha": m.alpha,
                "recall_at_1": m.recall_at_1,
                "recall_at_3": m.recall_at_3,
                "recall_at_5": m.recall_at_5,
                "recall_at_10": m.recall_at_10,
                "mrr": m.mrr_score,
                "ndcg_at_5": m.ndcg_at_5,
                "ndcg_at_10": m.ndcg_at_10,
                "num_queries": m.num_queries,
                "by_type": m.by_type,
                "per_query": [
                    {
                        "query_id": pq.query_id,
                        "query_type": pq.query_type,
                        "retrieved": pq.retrieved_clip_ids[:5],
                        "relevant": pq.relevant_clip_ids,
                        "recall_at_5": pq.recall_at_5,
                        "mrr": pq.mrr_score,
                        "hit": pq.hit,
                    }
                    for pq in m.per_query
                ],
            }
        )

    return {
        "benchmark_file": report.benchmark_file,
        "num_queries": report.num_queries,
        "modes": modes_out,
        "deltas": report.deltas,
    }
