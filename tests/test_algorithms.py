import unittest

import numpy as np

from sopilot.core.dtw import dtw_align
from sopilot.core.scoring import ScoreWeights, score_alignment
from sopilot.core.segmentation import detect_step_boundaries


class SegmentationTests(unittest.TestCase):
    def test_detects_single_clear_boundary(self) -> None:
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.99, 0.01, 0.0],
                [0.98, 0.02, 0.0],
                [0.05, 0.95, 0.0],
                [0.04, 0.96, 0.0],
            ],
            dtype=np.float32,
        )
        boundaries = detect_step_boundaries(embeddings, min_gap=1, z_threshold=0.4)
        self.assertIn(3, boundaries)


class DTWTests(unittest.TestCase):
    def test_alignment_covers_start_and_end(self) -> None:
        gold = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        trainee = np.array(
            [
                [0.95, 0.05],
                [0.1, 0.9],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        result = dtw_align(gold, trainee)
        self.assertEqual(result.path[0], (0, 0))
        self.assertEqual(result.path[-1], (1, 2))
        self.assertLess(result.normalized_cost, 0.2)


class ScoringTests(unittest.TestCase):
    def test_perfect_trajectory_scores_high(self) -> None:
        gold = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        trainee = gold.copy()
        alignment = dtw_align(gold, trainee)
        result = score_alignment(
            alignment=alignment,
            gold_len=len(gold),
            trainee_len=len(trainee),
            gold_boundaries=[2],
            trainee_boundaries=[2],
            weights=ScoreWeights(),
            deviation_threshold=0.2,
        )
        self.assertGreaterEqual(result["score"], 95.0)

    def test_missing_step_reduces_score(self) -> None:
        gold = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        trainee = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        )
        alignment = dtw_align(gold, trainee)
        result = score_alignment(
            alignment=alignment,
            gold_len=len(gold),
            trainee_len=len(trainee),
            gold_boundaries=[2],
            trainee_boundaries=[],
            weights=ScoreWeights(),
            deviation_threshold=0.2,
        )
        self.assertLess(result["score"], 90.0)
        self.assertGreaterEqual(result["metrics"]["miss_steps"], 1)
        self.assertTrue(any(item.get("type") == "missing_step" for item in result["deviations"]))


if __name__ == "__main__":
    unittest.main()
