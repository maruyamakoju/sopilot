import tempfile
import unittest
from pathlib import Path

from scripts.bulk_upload_folder import classify, iter_videos


class BulkUploadScriptTests(unittest.TestCase):
    def test_classify_auto(self) -> None:
        self.assertEqual(classify(Path("x/gold/video.mp4"), "auto"), "gold")
        self.assertEqual(classify(Path("x/trainee/video.mp4"), "auto"), "trainee")
        self.assertEqual(classify(Path("x/unknown.mp4"), "auto"), "trainee")

    def test_iter_videos_filters_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "a.mp4").write_bytes(b"x")
            (root / "b.avi").write_bytes(b"x")
            (root / "c.txt").write_text("x", encoding="utf-8")
            (root / "e.webm").write_bytes(b"x")
            nested = root / "nested"
            nested.mkdir(parents=True, exist_ok=True)
            (nested / "d.mov").write_bytes(b"x")

            flat = iter_videos(root, recursive=False)
            self.assertEqual(sorted(p.name for p in flat), ["a.mp4", "b.avi", "e.webm"])

            all_items = iter_videos(root, recursive=True)
            self.assertEqual(sorted(p.name for p in all_items), ["a.mp4", "b.avi", "d.mov", "e.webm"])


if __name__ == "__main__":
    unittest.main()
