import tempfile
import unittest
from pathlib import Path

from scripts.poc_local_pipeline import _build_labels_template, classify, iter_videos


class PocLocalPipelineScriptTests(unittest.TestCase):
    def test_classify_auto(self) -> None:
        self.assertEqual(classify(Path("x/gold/a.webm"), "auto"), "gold")
        self.assertEqual(classify(Path("x/trainee/a.webm"), "auto"), "trainee")
        self.assertEqual(classify(Path("x/misc/a.webm"), "auto"), "trainee")

    def test_iter_videos_includes_webm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "a.webm").write_bytes(b"x")
            (root / "b.mp4").write_bytes(b"x")
            (root / "c.txt").write_text("x", encoding="utf-8")
            items = iter_videos(root, recursive=False)
            self.assertEqual(sorted(p.name for p in items), ["a.webm", "b.mp4"])

    def test_build_labels_template(self) -> None:
        payload = _build_labels_template(
            "filter_change",
            [
                {
                    "job_id": 10,
                    "result": {
                        "deviations": [{"type": "missing_step", "severity": "critical"}],
                    },
                },
                {"job_id": 11, "result": {"deviations": []}},
            ],
        )
        self.assertEqual(payload["task_id"], "filter_change")
        self.assertEqual(len(payload["jobs"]), 2)
        self.assertTrue(payload["jobs"][0]["predicted_critical"])
        self.assertFalse(payload["jobs"][1]["predicted_critical"])


if __name__ == "__main__":
    unittest.main()
