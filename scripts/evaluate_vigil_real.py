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

    config = RetrievalConfig(model_name=model_name, device=device)
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

    config = TranscriptionConfig(backend="openai-whisper", model_name=model_name)
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
) -> dict:
    """Index a single video: chunk → embed → store (+ optional transcription)."""
    from sopilot.vigil_helpers import index_video_micro

    tx_service = None
    if transcribe:
        tx_service = _create_transcription_service(whisper_model)

    return index_video_micro(
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
) -> list[dict]:
    """Run retrieval for a single query. Returns list of {clip_id, score, ...}."""
    import contextlib

    query_embedding = embedder.encode_text([query_text])[0]

    # Visual search
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


def _match_clip_by_time(
    retrieved: list[dict],
    gt_time_ranges: list[dict],
    *,
    iou_threshold: float = 0.3,
) -> list[str]:
    """Match retrieved clips to GT time ranges by temporal overlap.

    When ground truth uses time_ranges instead of clip_ids, we match any
    retrieved clip that overlaps a GT range by at least iou_threshold.

    Returns:
        List of clip_ids that match at least one GT range.
    """
    from sopilot.temporal import temporal_iou

    matched_ids = []
    for r in retrieved:
        for gt in gt_time_ranges:
            iou = temporal_iou(r["start_sec"], r["end_sec"], gt["start_sec"], gt["end_sec"])
            if iou >= iou_threshold:
                matched_ids.append(r["clip_id"])
                break
    return matched_ids


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
        )
        dt = time.time() - t0
        logger.info(
            "  Indexed %d micro clips (%d text), %.1f sec",
            result["num_added"],
            result.get("num_text_added", 0),
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
            )

            retrieved_ids = [r["clip_id"] for r in results]

            # Determine relevant clip IDs
            if q.relevant_clip_ids:
                relevant = q.relevant_clip_ids
            elif q.relevant_time_ranges:
                # Match by temporal overlap
                relevant = _match_clip_by_time(results, q.relevant_time_ranges, iou_threshold=iou_threshold)
            else:
                relevant = []

            all_retrieved_ids.append(retrieved_ids)
            all_relevant_ids.append(relevant)
            all_relevant_sets.append(set(relevant))
            all_relevance_scores.append({cid: 1.0 for cid in relevant})

            # Per-query metrics
            r5 = len(set(retrieved_ids[:5]) & set(relevant)) / max(len(relevant), 1) if relevant else 0.0
            first_rank = 0.0
            for rank, cid in enumerate(retrieved_ids, 1):
                if cid in set(relevant):
                    first_rank = 1.0 / rank
                    break

            pq = {
                "query_id": q.query_id,
                "query_type": q.query_type,
                "query_text": q.query_text,
                "recall_at_5": r5,
                "mrr": first_rank,
                "hit": r5 > 0,
                "num_relevant": len(relevant),
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

        # Aggregate metrics
        if all_retrieved_ids:
            recall = evidence_recall_at_k(all_retrieved_ids, all_relevant_ids)
            mrr_score = mrr(all_retrieved_ids, all_relevant_sets)
            ndcg_5 = ndcg_at_k(all_retrieved_ids, all_relevance_scores, k=5)
            ndcg_10 = ndcg_at_k(all_retrieved_ids, all_relevance_scores, k=10)
        else:
            from sopilot.evaluation.vigil_metrics import EvidenceRecallResult

            recall = EvidenceRecallResult(0, 0, 0, 0, 0)
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
                    "recall_at_5": sum(pq["recall_at_5"] for pq in qt_pqs) / len(qt_pqs),
                    "mrr": sum(pq["mrr"] for pq in qt_pqs) / len(qt_pqs),
                    "hit_rate": sum(1 for pq in qt_pqs if pq["hit"]) / len(qt_pqs),
                }

        mode_results[mode_label] = {
            "mode": mode_label,
            "alpha": alpha,
            "recall_at_1": recall.recall_at_1,
            "recall_at_3": recall.recall_at_3,
            "recall_at_5": recall.recall_at_5,
            "recall_at_10": recall.recall_at_10,
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
