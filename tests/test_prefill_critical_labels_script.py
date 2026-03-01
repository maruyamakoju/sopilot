import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.prefill_critical_labels import main


class PrefillCriticalLabelsScriptTests(unittest.TestCase):
    def test_prefill_marks_matching_paths_as_critical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary = {
                "uploads": [
                    {"video_id": 10, "path": "x/trainee_bad/sample_bad_freeze.mp4"},
                    {"video_id": 11, "path": "x/trainee/sample_ok.mp4"},
                ],
                "scores_completed": [
                    {"job_id": 1, "trainee_video_id": 10},
                    {"job_id": 2, "trainee_video_id": 11},
                ],
            }
            labels = {
                "task_id": "filter_change",
                "jobs": [
                    {"job_id": 1, "critical_expected": False, "predicted_critical": True},
                    {"job_id": 2, "critical_expected": False, "predicted_critical": False},
                ],
            }
            summary_path = root / "summary.json"
            labels_path = root / "labels.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            labels_path.write_text(json.dumps(labels), encoding="utf-8")

            with patch(
                "sys.argv",
                [
                    "prefill_critical_labels.py",
                    "--summary",
                    str(summary_path),
                    "--labels",
                    str(labels_path),
                    "--critical-pattern",
                    "_bad_freeze",
                ],
            ), patch("builtins.print"):
                main()

            updated = json.loads(labels_path.read_text(encoding="utf-8"))
            self.assertTrue(updated["jobs"][0]["critical_expected"])
            self.assertFalse(updated["jobs"][1]["critical_expected"])

    def test_prefill_uses_all_score_jobs_and_nulls_unmapped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary = {
                "video_path_by_id": {
                    "10": "x/trainee_bad/sample_bad_freeze.mp4",
                },
                "all_score_jobs": [
                    {"job_id": 1, "trainee_video_id": 10},
                ],
            }
            labels = {
                "task_id": "filter_change",
                "jobs": [
                    {"job_id": 1, "critical_expected": False, "predicted_critical": True},
                    {"job_id": 2, "critical_expected": True, "predicted_critical": False},
                ],
            }
            summary_path = root / "summary.json"
            labels_path = root / "labels.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            labels_path.write_text(json.dumps(labels), encoding="utf-8")

            with patch(
                "sys.argv",
                [
                    "prefill_critical_labels.py",
                    "--summary",
                    str(summary_path),
                    "--labels",
                    str(labels_path),
                    "--critical-pattern",
                    "_bad_freeze",
                ],
            ), patch("builtins.print"):
                main()

            updated = json.loads(labels_path.read_text(encoding="utf-8"))
            self.assertTrue(updated["jobs"][0]["critical_expected"])
            # unmapped job becomes unknown
            self.assertIsNone(updated["jobs"][1]["critical_expected"])

    def test_prefill_marks_internal_raw_paths_as_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary = {
                "video_path_by_id": {
                    "10": str(root / "raw" / "00000010.mp4"),
                },
                "all_score_jobs": [
                    {"job_id": 1, "trainee_video_id": 10},
                ],
            }
            labels = {
                "task_id": "filter_change",
                "jobs": [
                    {"job_id": 1, "critical_expected": False, "predicted_critical": True},
                ],
            }
            summary_path = root / "summary.json"
            labels_path = root / "labels.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            labels_path.write_text(json.dumps(labels), encoding="utf-8")

            with patch(
                "sys.argv",
                [
                    "prefill_critical_labels.py",
                    "--summary",
                    str(summary_path),
                    "--labels",
                    str(labels_path),
                    "--critical-pattern",
                    "_bad_freeze",
                ],
            ), patch("builtins.print"):
                main()

            updated = json.loads(labels_path.read_text(encoding="utf-8"))
            self.assertIsNone(updated["jobs"][0]["critical_expected"])


if __name__ == "__main__":
    unittest.main()
