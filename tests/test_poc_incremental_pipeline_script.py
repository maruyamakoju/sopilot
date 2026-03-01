import json
import tempfile
import unittest
from pathlib import Path

from scripts.poc_incremental_pipeline import (
    build_labels_template,
    build_video_path_map,
    classify,
    file_fingerprint,
    load_existing_expected,
    load_manifest,
    save_manifest,
)


class PocIncrementalPipelineScriptTests(unittest.TestCase):
    def test_classify_auto(self) -> None:
        self.assertEqual(classify(Path("x/gold/a.webm"), "auto"), "gold")
        self.assertEqual(classify(Path("x/trainee/a.webm"), "auto"), "trainee")
        self.assertEqual(classify(Path("x/unknown/a.webm"), "auto"), "trainee")

    def test_file_fingerprint_contains_size_and_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "x.mp4"
            path.write_bytes(b"abc")
            fp = file_fingerprint(path)
            self.assertEqual(fp["size"], 3)
            self.assertIn("mtime_ns", fp)
            self.assertIsInstance(fp["mtime_ns"], int)

    def test_manifest_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "m.json"
            payload = {"version": 1, "files": {"a": {"size": 1, "mtime_ns": 2}}}
            save_manifest(path, payload)
            loaded = load_manifest(path)
            self.assertEqual(loaded["files"]["a"]["size"], 1)

    def test_manifest_invalid_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad.json"
            path.write_text("{bad", encoding="utf-8")
            loaded = load_manifest(path)
            self.assertEqual(loaded, {"version": 1, "files": {}})

    def test_build_labels_template(self) -> None:
        completed_jobs = [
            {"id": 1, "score": {"deviations": [{"type": "missing_step", "severity": "critical"}]}},
            {"id": 2, "score": {"deviations": []}},
        ]
        labels = build_labels_template("filter_change", completed_jobs)
        self.assertEqual(labels["task_id"], "filter_change")
        self.assertEqual(len(labels["jobs"]), 2)
        self.assertTrue(labels["jobs"][0]["predicted_critical"])
        self.assertFalse(labels["jobs"][1]["predicted_critical"])

    def test_build_video_path_map(self) -> None:
        manifest = {
            "version": 1,
            "files": {
                "a.mp4": {"video_id": 3},
                "b.mp4": {"video_id": "4"},
                "c.mp4": {"x": 1},
            },
        }
        out = build_video_path_map(manifest)
        self.assertEqual(out[3], "a.mp4")
        self.assertEqual(out[4], "b.mp4")

    def test_load_existing_expected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "labels.json"
            path.write_text(
                json.dumps(
                    {
                        "jobs": [
                            {"job_id": 1, "critical_expected": True},
                            {"job_id": 2, "critical_expected": False},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            expected = load_existing_expected(path)
            self.assertTrue(expected[1])
            self.assertFalse(expected[2])

    def test_build_labels_template_preserves_existing_expected(self) -> None:
        completed_jobs = [
            {"id": 1, "score": {"deviations": []}},
            {"id": 2, "score": {"deviations": []}},
        ]
        labels = build_labels_template("t", completed_jobs, existing_expected={2: True})
        self.assertIsNone(labels["jobs"][0]["critical_expected"])
        self.assertTrue(labels["jobs"][1]["critical_expected"])

    def test_load_existing_expected_keeps_null(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "labels.json"
            path.write_text(
                json.dumps(
                    {
                        "jobs": [
                            {"job_id": 1, "critical_expected": None},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            expected = load_existing_expected(path)
            self.assertIn(1, expected)
            self.assertIsNone(expected[1])


if __name__ == "__main__":
    unittest.main()
