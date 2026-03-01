import unittest
from pathlib import Path

from scripts.fetch_commons_videos import (
    _load_custom_queries,
    _load_keyword_list,
    _sanitize_filename,
    resolve_target_path,
)


class FetchCommonsScriptTests(unittest.TestCase):
    def test_sanitize_filename_normalizes_symbols(self) -> None:
        self.assertEqual(
            _sanitize_filename(" Cleaning the Office (TESDA)! "),
            "Cleaning_the_Office_TESDA_",
        )

    def test_load_custom_queries_supports_repeat_and_csv(self) -> None:
        queries = _load_custom_queries(["maintenance reel,cleaning TESDA", "inspection"])
        self.assertEqual(queries, ["maintenance reel", "cleaning TESDA", "inspection"])

    def test_load_keyword_list_falls_back_to_defaults(self) -> None:
        self.assertEqual(_load_keyword_list(None, ["a", "b"]), ["a", "b"])
        self.assertEqual(_load_keyword_list([""], ["x"]), ["x"])

    def test_resolve_target_path_split_layout(self) -> None:
        target, subset = resolve_target_path(
            idx=3,
            output_dir=Path("poc_videos"),
            layout="split",
            gold_count=2,
            file_name="Cleaning the Office (TESDA).webm",
            mime="video/webm",
        )
        self.assertEqual(subset, "trainee")
        self.assertTrue(
            str(target).replace("\\", "/").endswith(
                "poc_videos/trainee/trainee_003_Cleaning_the_Office_TESDA_.webm"
            )
        )

    def test_resolve_target_path_candidates_layout(self) -> None:
        target, subset = resolve_target_path(
            idx=1,
            output_dir=Path("poc_videos/candidates"),
            layout="candidates",
            gold_count=0,
            file_name="CH-47 Maintenance Reel (956208).webm",
            mime="video/webm",
        )
        self.assertEqual(subset, "candidate")
        self.assertTrue(
            str(target).replace("\\", "/").endswith(
                "poc_videos/candidates/001_CH-47_Maintenance_Reel_956208_.webm"
            )
        )


if __name__ == "__main__":
    unittest.main()
