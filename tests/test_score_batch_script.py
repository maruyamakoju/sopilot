import unittest

from scripts.score_batch import _pick_gold


class ScoreBatchScriptTests(unittest.TestCase):
    def test_pick_gold_prefers_explicit(self) -> None:
        golds = [{"video_id": 9}, {"video_id": 8}]
        self.assertEqual(_pick_gold(golds, 777), 777)

    def test_pick_gold_uses_latest_when_missing(self) -> None:
        golds = [{"video_id": 9}, {"video_id": 8}]
        self.assertEqual(_pick_gold(golds, None), 9)


if __name__ == "__main__":
    unittest.main()

