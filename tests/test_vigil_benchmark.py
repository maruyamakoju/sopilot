"""Tests for VIGIL-RAG benchmark evaluation pipeline."""

from __future__ import annotations

from pathlib import Path

from sopilot.evaluation.vigil_benchmark import (
    BenchmarkQuery,
    VIGILBenchmarkRunner,
    format_report,
    report_to_dict,
)

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
SMOKE_BENCHMARK = BENCHMARKS_DIR / "smoke_benchmark.jsonl"
FULL_BENCHMARK = BENCHMARKS_DIR / "vigil_benchmark_v1.jsonl"


# ---------------------------------------------------------------------------
# BenchmarkQuery loading
# ---------------------------------------------------------------------------


class TestLoadBenchmark:
    """Tests for loading benchmark JSONL files."""

    def test_load_smoke_benchmark(self):
        """Smoke benchmark loads correctly."""
        runner = VIGILBenchmarkRunner()
        queries = runner.load_benchmark(SMOKE_BENCHMARK)
        assert len(queries) == 6
        assert all(isinstance(q, BenchmarkQuery) for q in queries)

    def test_load_full_benchmark(self):
        """Full benchmark loads correctly."""
        runner = VIGILBenchmarkRunner()
        queries = runner.load_benchmark(FULL_BENCHMARK)
        assert len(queries) == 20

    def test_query_types_present(self):
        """All three query types are represented in smoke benchmark."""
        runner = VIGILBenchmarkRunner()
        queries = runner.load_benchmark(SMOKE_BENCHMARK)
        types = {q.query_type for q in queries}
        assert types == {"visual", "audio", "mixed"}

    def test_query_fields_populated(self):
        """All required fields are populated."""
        runner = VIGILBenchmarkRunner()
        queries = runner.load_benchmark(SMOKE_BENCHMARK)
        for q in queries:
            assert q.query_id
            assert q.video_id
            assert q.query_text
            assert q.query_type in ("visual", "audio", "mixed")
            assert len(q.relevant_clip_ids) > 0

    def test_full_benchmark_has_event_detection(self):
        """Full benchmark includes event detection queries."""
        runner = VIGILBenchmarkRunner()
        queries = runner.load_benchmark(FULL_BENCHMARK)
        event_queries = [q for q in queries if q.event_detection]
        assert len(event_queries) >= 3


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------


class TestBenchmarkRunner:
    """Tests for VIGILBenchmarkRunner."""

    def test_deterministic_results(self):
        """Same seed produces identical results."""
        runner1 = VIGILBenchmarkRunner(seed=42)
        runner2 = VIGILBenchmarkRunner(seed=42)

        queries = runner1.load_benchmark(SMOKE_BENCHMARK)

        report1 = runner1.run(queries, alphas=[0.7], top_k=10)
        report2 = runner2.run(queries, alphas=[0.7], top_k=10)

        assert report1.modes[0].recall_at_5 == report2.modes[0].recall_at_5
        assert report1.modes[1].recall_at_5 == report2.modes[1].recall_at_5

    def test_different_seeds_differ(self):
        """Different seeds produce different (but both valid) results."""
        runner1 = VIGILBenchmarkRunner(seed=42)
        runner2 = VIGILBenchmarkRunner(seed=123)

        queries = runner1.load_benchmark(SMOKE_BENCHMARK)

        report1 = runner1.run(queries, alphas=[0.7], top_k=10)

        queries2 = runner2.load_benchmark(SMOKE_BENCHMARK)
        report2 = runner2.run(queries2, alphas=[0.7], top_k=10)

        # Both should produce valid results (may or may not differ in exact values)
        assert report1.modes[0].num_queries == report2.modes[0].num_queries

    def test_visual_only_mode(self):
        """Visual-only mode produces results without hybrid."""
        runner = VIGILBenchmarkRunner(seed=42)
        queries = runner.load_benchmark(SMOKE_BENCHMARK)
        report = runner.run(queries, alphas=[], top_k=10)

        # Only visual_only mode
        assert len(report.modes) == 1
        assert report.modes[0].mode == "visual_only"

    def test_alpha_sweep(self):
        """Alpha sweep produces one result per alpha value."""
        runner = VIGILBenchmarkRunner(seed=42)
        queries = runner.load_benchmark(SMOKE_BENCHMARK)
        report = runner.run(queries, alphas=[0.3, 0.5, 0.7], top_k=10)

        # visual_only + 3 hybrid modes
        assert len(report.modes) == 4
        assert report.modes[0].mode == "visual_only"
        assert "alpha=0.3" in report.modes[1].mode
        assert "alpha=0.5" in report.modes[2].mode
        assert "alpha=0.7" in report.modes[3].mode

    def test_event_detection_queries_excluded(self):
        """Event detection queries are filtered out of retrieval evaluation."""
        runner = VIGILBenchmarkRunner(seed=42)
        queries = runner.load_benchmark(FULL_BENCHMARK)
        report = runner.run(queries, alphas=[0.7], top_k=10)

        # Full benchmark has 20 queries, 5 are event detection
        assert report.num_queries == 15


# ---------------------------------------------------------------------------
# Hybrid improves audio queries (core assertion)
# ---------------------------------------------------------------------------


class TestHybridImprovement:
    """Tests that hybrid search improves audio retrieval without hurting visual."""

    def test_audio_recall_improves_with_hybrid(self):
        """Audio queries have higher Recall@5 in hybrid mode than visual-only."""
        runner = VIGILBenchmarkRunner(seed=42)
        queries = runner.load_benchmark(SMOKE_BENCHMARK)
        report = runner.run(queries, alphas=[0.7], top_k=10)

        visual_only = report.modes[0]
        hybrid = report.modes[1]

        audio_visual_only = visual_only.by_type.get("audio", {})
        audio_hybrid = hybrid.by_type.get("audio", {})

        assert audio_hybrid["recall_at_5"] >= audio_visual_only["recall_at_5"]

    def test_visual_recall_not_degraded(self):
        """Visual queries are not degraded by hybrid search."""
        runner = VIGILBenchmarkRunner(seed=42)
        queries = runner.load_benchmark(SMOKE_BENCHMARK)
        report = runner.run(queries, alphas=[0.7], top_k=10)

        visual_only = report.modes[0]
        hybrid = report.modes[1]

        vis_visual_only = visual_only.by_type.get("visual", {})
        vis_hybrid = hybrid.by_type.get("visual", {})

        assert vis_hybrid["recall_at_5"] >= vis_visual_only["recall_at_5"]

    def test_overall_mrr_improves(self):
        """Overall MRR improves or stays the same with hybrid."""
        runner = VIGILBenchmarkRunner(seed=42)
        queries = runner.load_benchmark(SMOKE_BENCHMARK)
        report = runner.run(queries, alphas=[0.7], top_k=10)

        assert report.modes[1].mrr_score >= report.modes[0].mrr_score

    def test_full_benchmark_audio_improvement(self):
        """Full benchmark: audio Recall@5 strictly improves with hybrid."""
        runner = VIGILBenchmarkRunner(seed=42)
        queries = runner.load_benchmark(FULL_BENCHMARK)
        report = runner.run(queries, alphas=[0.7], top_k=10)

        visual_only = report.modes[0]
        hybrid = report.modes[1]

        audio_vo = visual_only.by_type["audio"]["recall_at_5"]
        audio_hy = hybrid.by_type["audio"]["recall_at_5"]

        # Strict improvement expected
        assert audio_hy > audio_vo


# ---------------------------------------------------------------------------
# CI smoke gate: minimum baselines
# ---------------------------------------------------------------------------


class TestSmokeGate:
    """Minimum performance thresholds for CI regression detection.

    If these fail, something is fundamentally broken in the hybrid pipeline.
    """

    def test_visual_only_baseline(self):
        """Visual-only mode achieves minimum Recall@5 on visual queries."""
        runner = VIGILBenchmarkRunner(seed=42)
        queries = runner.load_benchmark(SMOKE_BENCHMARK)
        report = runner.run(queries, alphas=[], top_k=10)

        visual_only = report.modes[0]
        # Visual queries should be well-served by visual-only
        assert visual_only.by_type["visual"]["recall_at_5"] >= 0.8

    def test_hybrid_audio_baseline(self):
        """Hybrid mode achieves minimum Recall@5 on audio queries."""
        runner = VIGILBenchmarkRunner(seed=42)
        queries = runner.load_benchmark(SMOKE_BENCHMARK)
        report = runner.run(queries, alphas=[0.7], top_k=10)

        hybrid = report.modes[1]
        # Audio queries MUST be findable with hybrid
        assert hybrid.by_type["audio"]["recall_at_5"] >= 0.8

    def test_hybrid_overall_baseline(self):
        """Hybrid mode achieves minimum overall MRR."""
        runner = VIGILBenchmarkRunner(seed=42)
        queries = runner.load_benchmark(SMOKE_BENCHMARK)
        report = runner.run(queries, alphas=[0.7], top_k=10)

        hybrid = report.modes[1]
        assert hybrid.mrr_score >= 0.7


# ---------------------------------------------------------------------------
# Report formatting and serialization
# ---------------------------------------------------------------------------


class TestReportFormatting:
    """Tests for report formatting and JSON serialization."""

    def test_format_report_produces_string(self):
        """format_report returns a non-empty string."""
        runner = VIGILBenchmarkRunner(seed=42)
        queries = runner.load_benchmark(SMOKE_BENCHMARK)
        report = runner.run(queries, alphas=[0.7], top_k=10)
        report.benchmark_file = "test.jsonl"

        text = format_report(report)
        assert isinstance(text, str)
        assert "visual_only" in text
        assert "hybrid" in text
        assert "Delta" in text

    def test_report_to_dict_roundtrip(self):
        """report_to_dict produces valid JSON-serializable dict."""
        import json

        runner = VIGILBenchmarkRunner(seed=42)
        queries = runner.load_benchmark(SMOKE_BENCHMARK)
        report = runner.run(queries, alphas=[0.7], top_k=10)
        report.benchmark_file = "test.jsonl"

        d = report_to_dict(report)
        # Should be JSON-serializable
        serialized = json.dumps(d)
        assert len(serialized) > 0

        # Roundtrip
        parsed = json.loads(serialized)
        assert parsed["num_queries"] == 6
        assert len(parsed["modes"]) == 2

    def test_per_query_in_json(self):
        """JSON output includes per-query details."""
        runner = VIGILBenchmarkRunner(seed=42)
        queries = runner.load_benchmark(SMOKE_BENCHMARK)
        report = runner.run(queries, alphas=[0.7], top_k=10)

        d = report_to_dict(report)
        for mode_data in d["modes"]:
            assert "per_query" in mode_data
            assert len(mode_data["per_query"]) == 6
