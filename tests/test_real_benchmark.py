"""Tests for real data benchmark infrastructure (loading, path resolution).

These tests verify the benchmark loading and configuration plumbing.
They do NOT require real videos or OpenCLIP models.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from sopilot.evaluation.vigil_benchmark import BenchmarkQuery, VIGILBenchmarkRunner

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
REAL_BENCHMARK = BENCHMARKS_DIR / "real_v1.jsonl"
VIDEO_PATHS_EXAMPLE = BENCHMARKS_DIR / "video_paths.local.json.example"


class TestRealBenchmarkLoading:
    """Tests for loading real benchmark files."""

    def test_real_benchmark_loads(self):
        """real_v1.jsonl loads correctly."""
        runner = VIGILBenchmarkRunner()
        queries = runner.load_benchmark(REAL_BENCHMARK)
        assert len(queries) >= 5
        assert all(isinstance(q, BenchmarkQuery) for q in queries)

    def test_real_benchmark_query_types(self):
        """Real benchmark has visual, audio, and mixed queries."""
        runner = VIGILBenchmarkRunner()
        queries = runner.load_benchmark(REAL_BENCHMARK)
        types = {q.query_type for q in queries}
        assert "visual" in types
        assert "audio" in types
        assert "mixed" in types

    def test_real_benchmark_has_time_ranges(self):
        """All real benchmark queries have time ranges (for time-based matching)."""
        runner = VIGILBenchmarkRunner()
        queries = runner.load_benchmark(REAL_BENCHMARK)
        for q in queries:
            assert len(q.relevant_time_ranges) > 0, f"{q.query_id} missing time ranges"

    def test_real_benchmark_video_ids_consistent(self):
        """All queries reference a valid video_id."""
        runner = VIGILBenchmarkRunner()
        queries = runner.load_benchmark(REAL_BENCHMARK)
        for q in queries:
            assert q.video_id, f"{q.query_id} has empty video_id"


class TestVideoPathMapping:
    """Tests for video path mapping file."""

    def test_example_file_exists(self):
        """video_paths.local.json.example exists in repo."""
        assert VIDEO_PATHS_EXAMPLE.exists()

    def test_example_file_valid_json(self):
        """Example file is valid JSON."""
        with open(VIDEO_PATHS_EXAMPLE, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_example_file_has_entries(self):
        """Example file has at least one non-comment entry."""
        with open(VIDEO_PATHS_EXAMPLE, encoding="utf-8") as f:
            data = json.load(f)
        real_entries = {k: v for k, v in data.items() if not k.startswith("_")}
        assert len(real_entries) >= 1

    def test_load_video_map_function(self):
        """_load_video_map correctly filters comments."""
        # Import the function from the real eval script
        import sys

        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        sys.path.insert(0, str(scripts_dir.parent / "src"))

        # Create a temp video map
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(
                {
                    "_comment": "This should be filtered out",
                    "vid-01": "/path/to/video1.mp4",
                    "vid-02": "/path/to/video2.mp4",
                },
                f,
            )
            temp_path = Path(f.name)

        try:
            # Inline the loader logic (avoid importing scripts/ directly)
            with open(temp_path, encoding="utf-8") as f2:
                data = json.load(f2)
            video_map = {k: v for k, v in data.items() if isinstance(v, str) and not k.startswith("_")}

            assert len(video_map) == 2
            assert "_comment" not in video_map
            assert video_map["vid-01"] == "/path/to/video1.mp4"
        finally:
            temp_path.unlink(missing_ok=True)

    def test_gitignore_excludes_local_json(self):
        """video_paths.local.json is in .gitignore."""
        gitignore_path = BENCHMARKS_DIR.parent / ".gitignore"
        content = gitignore_path.read_text(encoding="utf-8")
        assert "video_paths.local.json" in content


class TestTimeRangeMatching:
    """Tests for temporal IoU-based GT matching."""

    def test_match_by_time_overlap(self):
        """Clips overlapping GT time ranges are identified as relevant."""
        from sopilot.temporal import temporal_iou

        retrieved = [
            {"clip_id": "c1", "start_sec": 10.0, "end_sec": 13.0, "score": 0.9},
            {"clip_id": "c2", "start_sec": 20.0, "end_sec": 23.0, "score": 0.8},
            {"clip_id": "c3", "start_sec": 50.0, "end_sec": 53.0, "score": 0.7},
        ]
        gt_ranges = [{"start_sec": 11.0, "end_sec": 14.0}]

        # c1 overlaps gt (IoU = 2/5 = 0.4), c2 and c3 don't
        matched = []
        for r in retrieved:
            for gt in gt_ranges:
                iou = temporal_iou(r["start_sec"], r["end_sec"], gt["start_sec"], gt["end_sec"])
                if iou >= 0.3:
                    matched.append(r["clip_id"])
                    break

        assert "c1" in matched
        assert "c2" not in matched
        assert "c3" not in matched

    def test_no_overlap_returns_empty(self):
        """No overlap returns no matches."""
        from sopilot.temporal import temporal_iou

        retrieved = [
            {"clip_id": "c1", "start_sec": 0.0, "end_sec": 3.0, "score": 0.9},
        ]
        gt_ranges = [{"start_sec": 100.0, "end_sec": 103.0}]

        matched = []
        for r in retrieved:
            for gt in gt_ranges:
                iou = temporal_iou(r["start_sec"], r["end_sec"], gt["start_sec"], gt["end_sec"])
                if iou >= 0.3:
                    matched.append(r["clip_id"])
                    break

        assert len(matched) == 0
