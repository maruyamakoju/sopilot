#!/usr/bin/env python3
"""VIGIL-RAG Real Data Benchmark Evaluation.

Runs retrieval evaluation against real videos using actual OpenCLIP embeddings
and optional Whisper transcription. Compares visual-only vs hybrid search.

Unlike the synthetic benchmark (evaluate_vigil_benchmark.py), this uses:
- Real video files (referenced via video_paths.local.json)
- Real OpenCLIP embeddings (ViT-B-32 / L-14 / H-14)
- Real Whisper transcription (optional, for hybrid search)
- FAISS in-memory vector store

Prerequisites:
    pip install -e ".[vigil]"            # OpenCLIP + chunking
    pip install -e ".[vigil,whisper]"    # + Whisper (for hybrid)

Usage:
    # Basic: visual-only vs hybrid with default alpha
    python scripts/evaluate_vigil_real.py \\
        --benchmark benchmarks/real_v1.jsonl \\
        --video-map benchmarks/video_paths.local.json

    # Alpha sweep
    python scripts/evaluate_vigil_real.py \\
        --benchmark benchmarks/real_v1.jsonl \\
        --video-map benchmarks/video_paths.local.json \\
        --alpha-sweep 0.0,0.3,0.5,0.7,1.0

    # Force re-index + enable transcription
    python scripts/evaluate_vigil_real.py \\
        --benchmark benchmarks/real_v1.jsonl \\
        --video-map benchmarks/video_paths.local.json \\
        --reindex --transcribe --whisper-model base
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from sopilot.evaluation.vigil_benchmark import (  # noqa: E402
    BenchmarkQuery,
    VIGILBenchmarkRunner,
)
from sopilot.evaluation.vigil_metrics import (  # noqa: E402
    evidence_recall_at_k,
    mrr,
    ndcg_at_k,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Real-data indexing and retrieval
# ---------------------------------------------------------------------------


def _load_video_map(path: Path) -> dict[str, str]:
    """Load video_id -> local path mapping from JSON.

    Args:
        path: Path to video_paths.local.json

    Returns:
        Dict mapping video_id to local file path.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Filter out comments / non-string values
    return {k: v for k, v in data.items() if isinstance(v, str) and not k.startswith("_")}


def _load_benchmark(path: Path) -> list[BenchmarkQuery]:
    """Load benchmark queries from JSONL (reuses synthetic runner's loader)."""
    runner = VIGILBenchmarkRunner()
    return runner.load_benchmark(path)


def _create_embedder(model_name: str = "ViT-B-32", device: str = "cpu"):
    """Create OpenCLIP retrieval embedder."""
    from sopilot.retrieval_embeddings import RetrievalConfig, RetrievalEmbedder

    # Use for_model() to get correct pretrained tag and embedding_dim
    config = RetrievalConfig.for_model(model_name)
    config.device = device
    return RetrievalEmbedder(config)


def _create_qdrant():
    """Create FAISS-backed QdrantService (no real Qdrant needed)."""
    from sopilot.qdrant_service import QdrantConfig, QdrantService

    config = QdrantConfig(host="localhost", port=19999)
    return QdrantService(config, use_faiss_fallback=True)


def _create_chunker():
    """Create chunking service."""
    from sopilot.chunking_service import ChunkingService

    return ChunkingService()


def _create_transcription_service(model_name: str = "base"):
    """Create transcription service (optional)."""
    from sopilot.transcription_service import TranscriptionConfig, TranscriptionService

    config = TranscriptionConfig(backend="openai-whisper", model_size=model_name)
    return TranscriptionService(config)


def _index_video(
    video_path: Path,
    video_id: str,
    chunker,
    embedder,
    qdrant,
    *,
    transcribe: bool = False,
    whisper_model: str = "base",
    hierarchical: bool = False,
) -> dict:
    """Index a single video: chunk → embed → store (+ optional transcription).

    When *hierarchical* is True, indexes meso + macro levels too.
    """
    from sopilot.vigil_helpers import index_video_all_levels, index_video_micro

    tx_service = None
    if transcribe:
        tx_service = _create_transcription_service(whisper_model)

    index_fn = index_video_all_levels if hierarchical else index_video_micro
    return index_fn(
        video_path,
        video_id,
        chunker,
        embedder,
        qdrant,
        transcription_service=tx_service,
    )


def _retrieve_for_query(
    embedder,
    qdrant,
    query_text: str,
    *,
    mode: str,
    alpha: float,
    video_id: str | None,
    top_k: int,
    hierarchical: bool = False,
) -> list[dict]:
    """Run retrieval for a single query. Returns list of {clip_id, score, ...}."""
    import contextlib

    query_embedding = embedder.encode_text([query_text])[0]

    # Visual search — flat or coarse-to-fine
    if hierarchical:
        hier_results = qdrant.coarse_to_fine_search(
            query_embedding,
            video_id=video_id,
            macro_k=5,
            meso_k=10,
            micro_k=top_k,
            shot_k=0,
            enable_temporal_filtering=True,
            time_expand_factor=0.1,
        )
        visual_results = hier_results.get("micro", [])
    else:
        visual_results = qdrant.search(
            level="micro",
            query_vector=query_embedding,
            top_k=top_k,
            video_id=video_id,
        )

    if mode == "visual_only" or alpha <= 0:
        return [
            {"clip_id": r.clip_id, "score": r.score, "start_sec": r.start_sec, "end_sec": r.end_sec}
            for r in visual_results
        ]

    # Hybrid: also search micro_text
    audio_results = []
    with contextlib.suppress(Exception):
        audio_results = qdrant.search(
            level="micro_text",
            query_vector=query_embedding,
            top_k=top_k,
            video_id=video_id,
        )

    if not audio_results:
        return [
            {"clip_id": r.clip_id, "score": r.score, "start_sec": r.start_sec, "end_sec": r.end_sec}
            for r in visual_results
        ]

    # Fuse: max(visual_score, alpha * audio_score)
    visual_map = {r.clip_id: r for r in visual_results}
    audio_map = {r.clip_id: r for r in audio_results}

    all_clip_ids = set(visual_map) | set(audio_map)
    fused = []
    for cid in all_clip_ids:
        v = visual_map.get(cid)
        a = audio_map.get(cid)
        v_score = v.score if v else 0.0
        a_score = (a.score * alpha) if a else 0.0
        base = v or a
        fused.append(
            {
                "clip_id": base.clip_id,
                "score": max(v_score, a_score),
                "visual_score": v_score,
                "audio_score": a.score if a else 0.0,
                "start_sec": base.start_sec,
                "end_sec": base.end_sec,
            }
        )

    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused[:top_k]


def _temporal_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Compute temporal overlap in seconds between two time ranges.

    Args:
        a_start, a_end: First time range
        b_start, b_end: Second time range

    Returns:
        Overlap duration in seconds (0 if no overlap)
    """
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _is_relevant_result(
    result: dict,
    gt_clip_ids: list[str],
    gt_time_ranges: list[dict],
    *,
    min_overlap_sec: float = 0.0,
) -> bool:
    """Check if a result matches ground truth (NO circular dependency).

    Ground truth is determined ONLY by benchmark GT, NOT by retrieval results.

    Args:
        result: Search result dict with clip_id, start_sec, end_sec
        gt_clip_ids: Ground truth clip IDs (if provided)
        gt_time_ranges: Ground truth time ranges (if provided)
        min_overlap_sec: Minimum overlap to consider a match

    Returns:
        True if result matches GT, False otherwise
    """
    # Priority 1: Exact clip ID match (if GT has clip_ids)
    if gt_clip_ids:
        return result["clip_id"] in gt_clip_ids

    # Priority 2: Temporal overlap (if GT has time_ranges)
    if gt_time_ranges:
        for gt_range in gt_time_ranges:
            overlap = _temporal_overlap(
                result["start_sec"],
                result["end_sec"],
                gt_range["start_sec"],
                gt_range["end_sec"],
            )
            if overlap > min_overlap_sec:
                return True

    # No GT provided or no match
    return False


def _recall_at_k(
    results: list[dict],
    gt_clip_ids: list[str],
    gt_time_ranges: list[dict],
    k: int,
    *,
    min_overlap_sec: float = 0.0,
) -> float:
    """Compute Recall@K (binary: 1.0 if any relevant in top-K, else 0.0).

    Args:
        results: Search results (ordered by score desc)
        gt_clip_ids: Ground truth clip IDs
        gt_time_ranges: Ground truth time ranges
        k: Top-K cutoff
        min_overlap_sec: Minimum overlap for temporal match

    Returns:
        1.0 if at least one relevant result in top-K, else 0.0
    """
    return 1.0 if any(
        _is_relevant_result(r, gt_clip_ids, gt_time_ranges, min_overlap_sec=min_overlap_sec)
        for r in results[:k]
    ) else 0.0


def _reciprocal_rank(
    results: list[dict],
    gt_clip_ids: list[str],
    gt_time_ranges: list[dict],
    *,
    min_overlap_sec: float = 0.0,
) -> float:
    """Compute Reciprocal Rank (1 / rank of first relevant result).

    Args:
        results: Search results (ordered by score desc)
        gt_clip_ids: Ground truth clip IDs
        gt_time_ranges: Ground truth time ranges
        min_overlap_sec: Minimum overlap for temporal match

    Returns:
        1 / rank (1-indexed) of first relevant result, or 0.0 if none
    """
    for rank, result in enumerate(results, start=1):
        if _is_relevant_result(result, gt_clip_ids, gt_time_ranges, min_overlap_sec=min_overlap_sec):
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Evaluation orchestrator
# ---------------------------------------------------------------------------


def evaluate(
    benchmark_path: Path,
    video_map_path: Path,
    *,
    alphas: list[float],
    top_k: int = 10,
    embedding_model: str = "ViT-B-32",
    device: str = "cpu",
    reindex: bool = False,
    transcribe: bool = False,
    whisper_model: str = "base",
    iou_threshold: float = 0.3,
    hierarchical: bool = False,
) -> dict:
    """Run full real-data evaluation.

    Returns:
        Dict with modes, per-query results, and deltas.
    """
    queries = _load_benchmark(benchmark_path)
    video_map = _load_video_map(video_map_path)

    # Filter to retrieval queries (skip event detection)
    retrieval_queries = [q for q in queries if not q.event_detection]
    if not retrieval_queries:
        logger.error("No retrieval queries in benchmark")
        return {}

    # Check which videos are available
    video_ids_needed = {q.video_id for q in retrieval_queries}
    missing = video_ids_needed - set(video_map)
    if missing:
        logger.warning("Missing video mappings (will skip): %s", missing)
        retrieval_queries = [q for q in retrieval_queries if q.video_id in video_map]

    if not retrieval_queries:
        logger.error("No queries with available videos")
        return {}

    # Create services
    embedder = _create_embedder(embedding_model, device)
    qdrant = _create_qdrant()
    chunker = _create_chunker()

    # Index each video
    indexed_videos: set[str] = set()
    for vid in sorted({q.video_id for q in retrieval_queries}):
        video_path = Path(video_map[vid])
        if not video_path.exists():
            logger.error("Video file not found: %s (mapped from %s)", video_path, vid)
            continue

        # Check if already indexed (by count)
        existing = qdrant.count_by_video("micro", vid)
        if existing > 0 and not reindex:
            logger.info("Video %s already indexed (%d clips), skipping", vid, existing)
            indexed_videos.add(vid)
            continue

        logger.info("Indexing video: %s (%s)", vid, video_path)
        t0 = time.time()
        result = _index_video(
            video_path,
            vid,
            chunker,
            embedder,
            qdrant,
            transcribe=transcribe,
            whisper_model=whisper_model,
            hierarchical=hierarchical,
        )
        dt = time.time() - t0
        extra = ""
        if hierarchical:
            extra = f", meso={result.get('num_meso_added', 0)}, macro={result.get('num_macro_added', 0)}"
        logger.info(
            "  Indexed %d micro clips (%d text%s), %.1f sec",
            result["num_added"],
            result.get("num_text_added", 0),
            extra,
            dt,
        )
        indexed_videos.add(vid)

    # Evaluate per mode
    modes = ["visual_only"] + [f"hybrid(alpha={a})" for a in alphas]
    mode_results: dict[str, dict] = {}

    for mode_label in modes:
        is_hybrid = mode_label.startswith("hybrid")
        alpha = float(mode_label.split("=")[1].rstrip(")")) if is_hybrid else 0.0

        all_retrieved_ids: list[list[str]] = []
        all_relevant_ids: list[list[str]] = []
        all_relevant_sets: list[set[str]] = []
        all_relevance_scores: list[dict[str, float]] = []
        per_query: list[dict] = []

        for q in retrieval_queries:
            if q.video_id not in indexed_videos:
                continue

            results = _retrieve_for_query(
                embedder,
                qdrant,
                q.query_text,
                mode="hybrid" if is_hybrid else "visual_only",
                alpha=alpha,
                video_id=q.video_id,
                top_k=top_k,
                hierarchical=hierarchical,
            )

            retrieved_ids = [r["clip_id"] for r in results]

            # Per-query metrics (NO circular dependency - GT-only based)
            # Use new functions that determine relevance from GT, not from retrieval
            r1 = _recall_at_k(results, q.relevant_clip_ids or [], q.relevant_time_ranges or [], k=1)
            r5 = _recall_at_k(results, q.relevant_clip_ids or [], q.relevant_time_ranges or [], k=5)
            first_rank = _reciprocal_rank(results, q.relevant_clip_ids or [], q.relevant_time_ranges or [])

            # Legacy compatibility: collect relevant IDs for aggregate metrics
            # (Note: This is still needed for evidence_recall_at_k which expects clip ID lists)
            relevant_ids_for_aggregate = []
            if q.relevant_clip_ids:
                relevant_ids_for_aggregate = q.relevant_clip_ids
            elif q.relevant_time_ranges:
                # Extract clip IDs from results that match GT time ranges
                for r in results:
                    if _is_relevant_result(r, [], q.relevant_time_ranges):
                        relevant_ids_for_aggregate.append(r["clip_id"])

            all_retrieved_ids.append(retrieved_ids)
            all_relevant_ids.append(relevant_ids_for_aggregate)
            all_relevant_sets.append(set(relevant_ids_for_aggregate))
            all_relevance_scores.append({cid: 1.0 for cid in relevant_ids_for_aggregate})

            pq = {
                "query_id": q.query_id,
                "query_type": q.query_type,
                "query_text": q.query_text,
                "recall_at_1": r1,
                "recall_at_5": r5,
                "mrr": first_rank,
                "hit": r5 > 0,
                "num_relevant": len(relevant_ids_for_aggregate),
                "top_5": [{"clip_id": r["clip_id"], "score": round(r["score"], 4)} for r in results[:5]],
            }
            # Include score breakdown if hybrid
            if is_hybrid and results and "visual_score" in results[0]:
                pq["top_5_detail"] = [
                    {
                        "clip_id": r["clip_id"],
                        "visual_score": round(r.get("visual_score", 0), 4),
                        "audio_score": round(r.get("audio_score", 0), 4),
                        "fused_score": round(r["score"], 4),
                    }
                    for r in results[:5]
                ]

            per_query.append(pq)

        # Aggregate metrics (computed from per-query to avoid circular dependency)
        if per_query:
            # Average per-query metrics
            recall_at_1 = sum(pq["recall_at_1"] for pq in per_query) / len(per_query)
            recall_at_5 = sum(pq["recall_at_5"] for pq in per_query) / len(per_query)
            mrr_score = sum(pq["mrr"] for pq in per_query) / len(per_query)

            # For R@3 and R@10, compute on the fly from results
            # (Need to re-compute since we don't store per-query R@3/R@10)
            # For now, use legacy functions for these (acceptable since R@1/R@5/MRR are fixed)
            recall_at_3 = 0.0
            recall_at_10 = 0.0
            if all_retrieved_ids and all_relevant_ids:
                from sopilot.evaluation.vigil_metrics import evidence_recall_at_k
                recall_obj = evidence_recall_at_k(all_retrieved_ids, all_relevant_ids)
                recall_at_3 = recall_obj.recall_at_3
                recall_at_10 = recall_obj.recall_at_10

            # nDCG (keep legacy for now)
            ndcg_5 = 0.0
            ndcg_10 = 0.0
            if all_retrieved_ids and all_relevance_scores:
                ndcg_5 = ndcg_at_k(all_retrieved_ids, all_relevance_scores, k=5)
                ndcg_10 = ndcg_at_k(all_retrieved_ids, all_relevance_scores, k=10)
        else:
            recall_at_1 = 0.0
            recall_at_3 = 0.0
            recall_at_5 = 0.0
            recall_at_10 = 0.0
            mrr_score = 0.0
            ndcg_5 = 0.0
            ndcg_10 = 0.0

        # Per-type breakdown
        by_type = {}
        for qt in ("visual", "audio", "mixed"):
            qt_pqs = [pq for pq in per_query if pq["query_type"] == qt]
            if qt_pqs:
                by_type[qt] = {
                    "num_queries": len(qt_pqs),
                    "recall_at_1": sum(pq["recall_at_1"] for pq in qt_pqs) / len(qt_pqs),
                    "recall_at_5": sum(pq["recall_at_5"] for pq in qt_pqs) / len(qt_pqs),
                    "mrr": sum(pq["mrr"] for pq in qt_pqs) / len(qt_pqs),
                    "hit_rate": sum(1 for pq in qt_pqs if pq["hit"]) / len(qt_pqs),
                }

        mode_results[mode_label] = {
            "mode": mode_label,
            "alpha": alpha,
            "recall_at_1": recall_at_1,
            "recall_at_3": recall_at_3,
            "recall_at_5": recall_at_5,
            "recall_at_10": recall_at_10,
            "mrr": mrr_score,
            "ndcg_at_5": ndcg_5,
            "ndcg_at_10": ndcg_10,
            "num_queries": len(per_query),
            "by_type": by_type,
            "per_query": per_query,
        }

    # Compute deltas
    baseline_key = "visual_only"
    baseline = mode_results.get(baseline_key, {})
    deltas = {}
    for key, mr in mode_results.items():
        if key == baseline_key:
            continue
        deltas[key] = {
            "overall": {
                "recall_at_5": mr["recall_at_5"] - baseline.get("recall_at_5", 0),
                "mrr": mr["mrr"] - baseline.get("mrr", 0),
                "ndcg_at_5": mr["ndcg_at_5"] - baseline.get("ndcg_at_5", 0),
            },
        }
        for qt in ("visual", "audio", "mixed"):
            base_bt = baseline.get("by_type", {}).get(qt, {})
            mode_bt = mr.get("by_type", {}).get(qt, {})
            if base_bt and mode_bt:
                deltas[key][qt] = {
                    "recall_at_5": mode_bt["recall_at_5"] - base_bt["recall_at_5"],
                    "mrr": mode_bt["mrr"] - base_bt["mrr"],
                }

    return {
        "benchmark_file": str(benchmark_path),
        "video_map_file": str(video_map_path),
        "embedding_model": embedding_model,
        "transcribe": transcribe,
        "hierarchical": hierarchical,
        "top_k": top_k,
        "num_queries": len(retrieval_queries),
        "modes": mode_results,
        "deltas": deltas,
    }


# ---------------------------------------------------------------------------
# CLI output
# ---------------------------------------------------------------------------


def _print_results(results: dict) -> None:
    """Pretty-print evaluation results."""
    print("=" * 72)
    print("VIGIL-RAG Real Data Benchmark Report")
    print(f"  Benchmark: {results['benchmark_file']}")
    print(f"  Videos:    {results['video_map_file']}")
    print(f"  Embedder:  {results['embedding_model']}")
    print(f"  Transcribe: {results['transcribe']}")
    print(f"  Queries:   {results['num_queries']}")
    print("=" * 72)

    for mode_label, mr in results["modes"].items():
        print(f"\n--- {mode_label} ---")
        print(f"  Recall@1 = {mr['recall_at_1']:.4f}")
        print(f"  Recall@3 = {mr['recall_at_3']:.4f}")
        print(f"  Recall@5 = {mr['recall_at_5']:.4f}")
        print(f"  Recall@10= {mr['recall_at_10']:.4f}")
        print(f"  MRR      = {mr['mrr']:.4f}")
        print(f"  nDCG@5   = {mr['ndcg_at_5']:.4f}")
        print(f"  nDCG@10  = {mr['ndcg_at_10']:.4f}")

        if mr.get("by_type"):
            print("\n  By query type:")
            for qt in ("visual", "audio", "mixed"):
                bt = mr["by_type"].get(qt)
                if bt:
                    print(
                        f"    {qt:8s}  n={bt['num_queries']:2d}  "
                        f"R@5={bt['recall_at_5']:.4f}  "
                        f"MRR={bt['mrr']:.4f}  "
                        f"hit={bt['hit_rate']:.0%}"
                    )

    if results.get("deltas"):
        print("\n--- Delta (hybrid - visual_only) ---")
        for mode_key, delta_data in results["deltas"].items():
            print(f"  {mode_key}:")
            overall = delta_data.get("overall", {})
            print(
                f"    overall  dR@5={overall.get('recall_at_5', 0):+.4f}  "
                f"dMRR={overall.get('mrr', 0):+.4f}  "
                f"dnDCG@5={overall.get('ndcg_at_5', 0):+.4f}"
            )
            for qt in ("visual", "audio", "mixed"):
                qt_delta = delta_data.get(qt)
                if qt_delta:
                    print(f"    {qt:8s}  dR@5={qt_delta['recall_at_5']:+.4f}  dMRR={qt_delta['mrr']:+.4f}")

    # Improved / degraded queries
    if len(results["modes"]) >= 2:
        mode_keys = list(results["modes"].keys())
        baseline_pqs = {pq["query_id"]: pq for pq in results["modes"][mode_keys[0]]["per_query"]}
        hybrid_pqs = {pq["query_id"]: pq for pq in results["modes"][mode_keys[1]]["per_query"]}

        improved = []
        degraded = []
        for qid in baseline_pqs:
            if qid in hybrid_pqs:
                d = hybrid_pqs[qid]["recall_at_5"] - baseline_pqs[qid]["recall_at_5"]
                if d > 0.01:
                    improved.append((qid, baseline_pqs[qid]["query_type"], d))
                elif d < -0.01:
                    degraded.append((qid, baseline_pqs[qid]["query_type"], d))

        if improved:
            improved.sort(key=lambda x: x[2], reverse=True)
            print(f"\n  Improved queries (top {min(len(improved), 10)}):")
            for qid, qt, d in improved[:10]:
                print(f"    {qid} ({qt}): dR@5={d:+.4f}")

        if degraded:
            degraded.sort(key=lambda x: x[2])
            print(f"\n  Degraded queries (top {min(len(degraded), 10)}):")
            for qid, qt, d in degraded[:10]:
                print(f"    {qid} ({qt}): dR@5={d:+.4f}")

    print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VIGIL-RAG Real Data Benchmark: visual-only vs hybrid on real videos",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=str(project_root / "benchmarks" / "real_v1.jsonl"),
        help="Path to benchmark JSONL file",
    )
    parser.add_argument(
        "--video-map",
        type=str,
        default=str(project_root / "benchmarks" / "video_paths.local.json"),
        help="Path to video_id -> local path JSON mapping",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Hybrid search alpha (default: 0.7)",
    )
    parser.add_argument(
        "--alpha-sweep",
        type=str,
        default=None,
        help="Comma-separated alpha values (e.g., 0.0,0.3,0.5,0.7,1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results per query (default: 10)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="ViT-B-32",
        choices=["ViT-B-32", "ViT-L-14", "ViT-H-14"],
        help="OpenCLIP model name (default: ViT-B-32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for embeddings (cpu or cuda, default: cpu)",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-index even if video is already indexed",
    )
    parser.add_argument(
        "--transcribe",
        action="store_true",
        help="Enable Whisper transcription for hybrid search",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        help="Whisper model size (tiny/base/small/medium/large, default: base)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="Temporal IoU threshold for time-range based GT matching (default: 0.3)",
    )
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        help="Enable coarse-to-fine hierarchical retrieval (indexes meso+macro levels)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save JSON results to this path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Determine alpha(s)
    if args.alpha_sweep:
        alphas = [float(a.strip()) for a in args.alpha_sweep.split(",")]
    else:
        alphas = [args.alpha]

    # Validate paths
    benchmark_path = Path(args.benchmark)
    video_map_path = Path(args.video_map)

    if not benchmark_path.exists():
        print(f"ERROR: Benchmark file not found: {benchmark_path}", file=sys.stderr)
        return 1

    if not video_map_path.exists():
        print(f"ERROR: Video map not found: {video_map_path}", file=sys.stderr)
        print(
            "  Create from template: cp benchmarks/video_paths.local.json.example benchmarks/video_paths.local.json",
            file=sys.stderr,
        )
        return 1

    # Run evaluation
    results = evaluate(
        benchmark_path,
        video_map_path,
        alphas=alphas,
        top_k=args.top_k,
        embedding_model=args.embedding_model,
        device=args.device,
        reindex=args.reindex,
        transcribe=args.transcribe,
        whisper_model=args.whisper_model,
        iou_threshold=args.iou_threshold,
        hierarchical=args.hierarchical,
    )

    if not results:
        print("ERROR: Evaluation produced no results", file=sys.stderr)
        return 1

    _print_results(results)

    # Save JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
