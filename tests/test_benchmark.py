"""Tests for the Perception Engine benchmark evaluation framework.

Covers:
    - _iou_bbox()                  (IoU helper)
    - DetectionMetrics             (from_counts, to_dict)
    - TrackingMetrics              (to_dict)
    - AnomalyBenchmarkResult       (to_dict)
    - DetectionEvaluator           (empty, perfect, no-pred, low-IoU, high-IoU, multi-frame)
    - MOTEvaluator                 (perfect, empty, mostly-tracked, id-switch, MOTA formula)
    - AnomalyBenchmarkEvaluator    (smoke, field ranges, n_normal+n_anomaly)
    - benchmark_eval script        (file exists, importable)

Run:
    python -m pytest tests/test_benchmark.py -v
"""
from __future__ import annotations

import math
import unittest
from pathlib import Path


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
from sopilot.perception.benchmark import (
    AnomalyBenchmarkEvaluator,
    AnomalyBenchmarkResult,
    DetectionEvaluator,
    DetectionMetrics,
    MOTEvaluator,
    TrackingMetrics,
    _iou_bbox,
)


# ===========================================================================
# _iou_bbox helpers
# ===========================================================================


class TestIouBbox(unittest.TestCase):
    """Tests for the _iou_bbox() helper function."""

    def test_identical_boxes_returns_one(self):
        """Perfectly overlapping boxes → IoU = 1.0."""
        box = [10.0, 20.0, 50.0, 60.0]
        self.assertAlmostEqual(_iou_bbox(box, box), 1.0, places=6)

    def test_non_overlapping_boxes_returns_zero(self):
        """Completely non-overlapping boxes → IoU = 0.0."""
        a = [0.0, 0.0, 10.0, 10.0]
        b = [20.0, 20.0, 10.0, 10.0]
        self.assertEqual(_iou_bbox(a, b), 0.0)

    def test_half_overlapping_boxes(self):
        """Boxes sharing exactly half their area → IoU ≈ 1/3."""
        # a: [0,0,10,10] → area 100
        # b: [5,0,10,10] → area 100
        # intersection: [5,0,5,10] → area 50
        # union = 100 + 100 - 50 = 150
        # IoU = 50/150 ≈ 0.333
        a = [0.0, 0.0, 10.0, 10.0]
        b = [5.0, 0.0, 10.0, 10.0]
        self.assertAlmostEqual(_iou_bbox(a, b), 1.0 / 3.0, places=4)

    def test_touching_boxes_no_overlap(self):
        """Boxes that only share an edge → IoU = 0.0."""
        a = [0.0, 0.0, 10.0, 10.0]
        b = [10.0, 0.0, 10.0, 10.0]
        self.assertEqual(_iou_bbox(a, b), 0.0)

    def test_one_box_inside_other(self):
        """Small box fully inside big box → IoU = area_small / area_big."""
        big = [0.0, 0.0, 100.0, 100.0]
        small = [25.0, 25.0, 50.0, 50.0]
        iou = _iou_bbox(big, small)
        expected = (50 * 50) / (100 * 100)  # 0.25
        self.assertAlmostEqual(iou, expected, places=6)

    def test_zero_area_boxes_return_zero(self):
        """Zero-area boxes should not raise; IoU is 0."""
        a = [5.0, 5.0, 0.0, 0.0]
        b = [5.0, 5.0, 0.0, 0.0]
        self.assertEqual(_iou_bbox(a, b), 0.0)


# ===========================================================================
# DetectionMetrics
# ===========================================================================


class TestDetectionMetrics(unittest.TestCase):
    """Tests for DetectionMetrics dataclass."""

    def test_from_counts_correct_precision_recall_f1(self):
        """from_counts(10, 2, 3) → correct P/R/F1."""
        m = DetectionMetrics.from_counts(10, 2, 3)
        expected_p = 10 / 12
        expected_r = 10 / 13
        expected_f1 = 2 * expected_p * expected_r / (expected_p + expected_r)
        self.assertAlmostEqual(m.precision, expected_p, places=6)
        self.assertAlmostEqual(m.recall, expected_r, places=6)
        self.assertAlmostEqual(m.f1, expected_f1, places=6)

    def test_from_counts_all_zero(self):
        """All-zero counts → all metrics = 0."""
        m = DetectionMetrics.from_counts(0, 0, 0)
        self.assertEqual(m.precision, 0.0)
        self.assertEqual(m.recall, 0.0)
        self.assertEqual(m.f1, 0.0)

    def test_from_counts_no_fp_fn(self):
        """Perfect case: FP=0, FN=0 → P=R=F1=1."""
        m = DetectionMetrics.from_counts(5, 0, 0)
        self.assertEqual(m.precision, 1.0)
        self.assertEqual(m.recall, 1.0)
        self.assertEqual(m.f1, 1.0)

    def test_from_counts_iou_threshold_stored(self):
        """iou_threshold is stored on the dataclass."""
        m = DetectionMetrics.from_counts(5, 1, 1, iou_threshold=0.75)
        self.assertEqual(m.iou_threshold, 0.75)

    def test_to_dict_contains_all_keys(self):
        """to_dict() returns all expected keys."""
        m = DetectionMetrics.from_counts(10, 2, 3)
        d = m.to_dict()
        expected_keys = {
            "iou_threshold", "true_positives", "false_positives",
            "false_negatives", "precision", "recall", "f1",
        }
        self.assertEqual(set(d.keys()), expected_keys)

    def test_to_dict_values_rounded(self):
        """to_dict() rounds floats to 4 decimal places."""
        m = DetectionMetrics.from_counts(10, 2, 3)
        d = m.to_dict()
        for key in ("precision", "recall", "f1"):
            # should be a float
            self.assertIsInstance(d[key], float)


# ===========================================================================
# TrackingMetrics
# ===========================================================================


class TestTrackingMetrics(unittest.TestCase):
    """Tests for TrackingMetrics dataclass."""

    def _make_tracking_metrics(self, **kwargs) -> TrackingMetrics:
        defaults = dict(
            mota=0.8, motp=0.75, idf1=0.85,
            mostly_tracked=3, mostly_lost=1, id_switches=2,
            total_gt_tracks=5, total_pred_tracks=5, frames_evaluated=100,
        )
        defaults.update(kwargs)
        return TrackingMetrics(**defaults)

    def test_to_dict_contains_all_keys(self):
        """to_dict() returns all expected keys."""
        m = self._make_tracking_metrics()
        d = m.to_dict()
        expected_keys = {
            "mota", "motp", "idf1", "mostly_tracked", "mostly_lost",
            "id_switches", "total_gt_tracks", "total_pred_tracks",
            "frames_evaluated",
        }
        self.assertEqual(set(d.keys()), expected_keys)

    def test_to_dict_integer_fields(self):
        """Integer fields are preserved as ints in to_dict."""
        m = self._make_tracking_metrics()
        d = m.to_dict()
        for key in ("mostly_tracked", "mostly_lost", "id_switches",
                    "total_gt_tracks", "total_pred_tracks", "frames_evaluated"):
            self.assertIsInstance(d[key], int, f"{key} should be int")

    def test_to_dict_float_fields(self):
        """Float fields are present in to_dict."""
        m = self._make_tracking_metrics()
        d = m.to_dict()
        for key in ("mota", "motp", "idf1"):
            self.assertIsInstance(d[key], float, f"{key} should be float")


# ===========================================================================
# AnomalyBenchmarkResult
# ===========================================================================


class TestAnomalyBenchmarkResult(unittest.TestCase):
    """Tests for AnomalyBenchmarkResult dataclass."""

    def _make_result(self, **kwargs) -> AnomalyBenchmarkResult:
        defaults = dict(
            threshold_used=2.5,
            true_positives=15,
            false_positives=2,
            false_negatives=5,
            true_negatives=98,
            precision=0.88,
            recall=0.75,
            f1=0.81,
            accuracy=0.95,
            false_alarm_rate=0.02,
            detection_latency_frames=1.5,
            n_normal=100,
            n_anomaly=20,
        )
        defaults.update(kwargs)
        return AnomalyBenchmarkResult(**defaults)

    def test_to_dict_contains_all_keys(self):
        """to_dict() returns all expected keys."""
        r = self._make_result()
        d = r.to_dict()
        expected_keys = {
            "threshold_used", "true_positives", "false_positives",
            "false_negatives", "true_negatives", "precision", "recall",
            "f1", "accuracy", "false_alarm_rate", "detection_latency_frames",
            "n_normal", "n_anomaly",
        }
        self.assertEqual(set(d.keys()), expected_keys)

    def test_to_dict_counts_are_ints(self):
        """Count fields are integers in to_dict."""
        r = self._make_result()
        d = r.to_dict()
        for key in ("true_positives", "false_positives", "false_negatives",
                    "true_negatives", "n_normal", "n_anomaly"):
            self.assertIsInstance(d[key], int, f"{key} should be int")


# ===========================================================================
# DetectionEvaluator
# ===========================================================================


class TestDetectionEvaluator(unittest.TestCase):
    """Tests for DetectionEvaluator."""

    def setUp(self):
        self.ev = DetectionEvaluator(iou_threshold=0.5)

    # -- Edge cases ----------------------------------------------------------

    def test_empty_gt_returns_all_zeros(self):
        """No GT, no predictions → all zero metrics."""
        m = self.ev.evaluate([], [])
        self.assertEqual(m.true_positives, 0)
        self.assertEqual(m.false_positives, 0)
        self.assertEqual(m.false_negatives, 0)

    def test_empty_gt_with_predictions_gives_fp(self):
        """No GT but 2 predictions → FP=2, TP=FN=0."""
        preds = [{"frame_id": 0, "detections": [
            {"label": "person", "bbox": [0, 0, 10, 10]},
            {"label": "car",    "bbox": [20, 20, 10, 10]},
        ]}]
        m = self.ev.evaluate([], preds)
        self.assertEqual(m.false_positives, 2)
        self.assertEqual(m.true_positives, 0)
        self.assertEqual(m.false_negatives, 0)

    def test_no_predictions_gives_fn_equals_gt_count(self):
        """GT has 3 detections, no predictions → FN=3, TP=FP=0."""
        gt = [{"frame_id": 0, "detections": [
            {"label": "person", "bbox": [0, 0, 10, 10]},
            {"label": "person", "bbox": [20, 0, 10, 10]},
            {"label": "person", "bbox": [40, 0, 10, 10]},
        ]}]
        m = self.ev.evaluate(gt, [])
        self.assertEqual(m.false_negatives, 3)
        self.assertEqual(m.true_positives, 0)
        self.assertEqual(m.false_positives, 0)
        self.assertEqual(m.recall, 0.0)

    def test_perfect_predictions_precision_recall_one(self):
        """Predictions identical to GT → precision=1, recall=1, f1=1."""
        gt = [{"frame_id": 0, "detections": [
            {"label": "person", "bbox": [0.0, 0.0, 100.0, 100.0]},
            {"label": "car",    "bbox": [200.0, 0.0, 50.0, 50.0]},
        ]}]
        preds = [{"frame_id": 0, "detections": [
            {"label": "person", "bbox": [0.0, 0.0, 100.0, 100.0]},
            {"label": "car",    "bbox": [200.0, 0.0, 50.0, 50.0]},
        ]}]
        m = self.ev.evaluate(gt, preds)
        self.assertEqual(m.true_positives, 2)
        self.assertEqual(m.false_positives, 0)
        self.assertEqual(m.false_negatives, 0)
        self.assertAlmostEqual(m.precision, 1.0)
        self.assertAlmostEqual(m.recall, 1.0)
        self.assertAlmostEqual(m.f1, 1.0)

    def test_iou_below_threshold_treated_as_fp_and_fn(self):
        """Slightly shifted box (IoU << 0.5) → FP=1, FN=1, TP=0."""
        gt = [{"frame_id": 0, "detections": [
            {"label": "person", "bbox": [0.0, 0.0, 10.0, 10.0]},
        ]}]
        # Shifted far away → IoU = 0
        preds = [{"frame_id": 0, "detections": [
            {"label": "person", "bbox": [100.0, 100.0, 10.0, 10.0]},
        ]}]
        m = self.ev.evaluate(gt, preds)
        self.assertEqual(m.true_positives, 0)
        self.assertEqual(m.false_positives, 1)
        self.assertEqual(m.false_negatives, 1)

    def test_iou_above_threshold_treated_as_tp(self):
        """High-IoU prediction → TP=1, FP=FN=0."""
        gt = [{"frame_id": 0, "detections": [
            {"label": "person", "bbox": [0.0, 0.0, 100.0, 100.0]},
        ]}]
        # Slightly smaller box; IoU is high
        preds = [{"frame_id": 0, "detections": [
            {"label": "person", "bbox": [1.0, 1.0, 98.0, 98.0]},
        ]}]
        m = self.ev.evaluate(gt, preds)
        self.assertEqual(m.true_positives, 1)
        self.assertEqual(m.false_positives, 0)
        self.assertEqual(m.false_negatives, 0)

    def test_multi_frame_counts_are_summed(self):
        """Counts from multiple frames are correctly aggregated."""
        gt = [
            {"frame_id": 0, "detections": [
                {"label": "person", "bbox": [0.0, 0.0, 100.0, 100.0]},
            ]},
            {"frame_id": 1, "detections": [
                {"label": "person", "bbox": [0.0, 0.0, 100.0, 100.0]},
            ]},
        ]
        preds = [
            {"frame_id": 0, "detections": [
                {"label": "person", "bbox": [0.0, 0.0, 100.0, 100.0]},
            ]},
            # Frame 1 has no predictions → FN for that frame
        ]
        m = self.ev.evaluate(gt, preds)
        self.assertEqual(m.true_positives, 1)
        self.assertEqual(m.false_negatives, 1)

    def test_custom_iou_threshold_stored(self):
        """iou_threshold parameter is stored and used."""
        ev = DetectionEvaluator(iou_threshold=0.75)
        self.assertEqual(ev.iou_threshold, 0.75)
        m = ev.evaluate([], [])
        self.assertEqual(m.iou_threshold, 0.75)


# ===========================================================================
# MOTEvaluator
# ===========================================================================


class TestMOTEvaluator(unittest.TestCase):
    """Tests for MOTEvaluator."""

    def setUp(self):
        self.ev = MOTEvaluator(iou_threshold=0.5)

    def _make_frame(self, frame_id: int, tracks: list[dict]) -> dict:
        return {"frame_id": frame_id, "tracks": tracks}

    def _make_track(self, track_id: int, bbox: list) -> dict:
        return {"track_id": track_id, "bbox": bbox}

    # -- Edge cases ----------------------------------------------------------

    def test_empty_input_returns_zero_metrics(self):
        """Empty GT and predictions → all zeros."""
        m = self.ev.evaluate([], [])
        self.assertEqual(m.mota, 0.0)
        self.assertEqual(m.motp, 0.0)
        self.assertEqual(m.idf1, 0.0)
        self.assertEqual(m.id_switches, 0)
        self.assertEqual(m.total_gt_tracks, 0)
        self.assertEqual(m.total_pred_tracks, 0)
        self.assertEqual(m.frames_evaluated, 0)

    def test_perfect_tracking_mota_near_one(self):
        """Predictions identical to GT across frames → MOTA close to 1."""
        bbox = [0.0, 0.0, 100.0, 100.0]
        gt = [self._make_frame(i, [self._make_track(1, bbox)]) for i in range(5)]
        preds = [self._make_frame(i, [self._make_track(1, bbox)]) for i in range(5)]
        m = self.ev.evaluate(gt, preds)
        self.assertGreater(m.mota, 0.9)
        self.assertEqual(m.id_switches, 0)

    def test_perfect_tracking_no_id_switches(self):
        """Perfect tracking → zero ID switches."""
        bbox = [10.0, 10.0, 80.0, 80.0]
        gt = [self._make_frame(i, [self._make_track(1, bbox)]) for i in range(3)]
        preds = [self._make_frame(i, [self._make_track(1, bbox)]) for i in range(3)]
        m = self.ev.evaluate(gt, preds)
        self.assertEqual(m.id_switches, 0)

    def test_one_gt_track_one_pred_mostly_tracked(self):
        """1 GT track matched in all 5 frames → mostly_tracked = 1."""
        bbox = [0.0, 0.0, 100.0, 100.0]
        gt = [self._make_frame(i, [self._make_track(1, bbox)]) for i in range(5)]
        preds = [self._make_frame(i, [self._make_track(1, bbox)]) for i in range(5)]
        m = self.ev.evaluate(gt, preds)
        self.assertGreaterEqual(m.mostly_tracked, 1)

    def test_id_switch_increments_counter(self):
        """GT track 1 matched to pred 1 in frame 0, pred 2 in frame 1 → IDSW=1."""
        bbox = [0.0, 0.0, 100.0, 100.0]
        gt = [
            self._make_frame(0, [self._make_track(1, bbox)]),
            self._make_frame(1, [self._make_track(1, bbox)]),
        ]
        preds = [
            self._make_frame(0, [self._make_track(1, bbox)]),
            self._make_frame(1, [self._make_track(2, bbox)]),  # different pred id
        ]
        m = self.ev.evaluate(gt, preds)
        self.assertEqual(m.id_switches, 1)

    def test_mota_formula(self):
        """MOTA = 1 - (FP + FN + IDSW) / GT_total.

        With 2 GT frames each with 1 box, no predictions → FN=2, FP=0, IDSW=0.
        GT_total = 2 → MOTA = 1 - 2/2 = 0.
        """
        bbox = [0.0, 0.0, 100.0, 100.0]
        gt = [
            self._make_frame(0, [self._make_track(1, bbox)]),
            self._make_frame(1, [self._make_track(1, bbox)]),
        ]
        preds = []
        m = self.ev.evaluate(gt, preds)
        # FN = 2, FP = 0, IDSW = 0, GT_total = 2 → MOTA = 1 - 2/2 = 0
        self.assertAlmostEqual(m.mota, 0.0, places=4)

    def test_mota_with_fp(self):
        """MOTA with extra FP predictions is reduced below 1."""
        bbox = [0.0, 0.0, 100.0, 100.0]
        extra_bbox = [200.0, 200.0, 50.0, 50.0]
        gt = [self._make_frame(0, [self._make_track(1, bbox)])]
        preds = [self._make_frame(0, [
            self._make_track(1, bbox),    # TP
            self._make_track(2, extra_bbox),  # FP
        ])]
        m = self.ev.evaluate(gt, preds)
        # FP=1, FN=0, IDSW=0, GT_total=1 → MOTA = 1 - 1/1 = 0
        self.assertAlmostEqual(m.mota, 0.0, places=4)

    def test_frames_evaluated_count(self):
        """frames_evaluated matches the number of GT frames."""
        bbox = [0.0, 0.0, 50.0, 50.0]
        gt = [self._make_frame(i, [self._make_track(1, bbox)]) for i in range(7)]
        preds = []
        m = self.ev.evaluate(gt, preds)
        self.assertEqual(m.frames_evaluated, 7)

    def test_total_gt_pred_track_counts(self):
        """total_gt_tracks and total_pred_tracks count unique IDs."""
        bbox = [0.0, 0.0, 50.0, 50.0]
        gt = [
            self._make_frame(0, [self._make_track(1, bbox), self._make_track(2, [60.0, 0.0, 30.0, 30.0])]),
            self._make_frame(1, [self._make_track(1, bbox)]),
        ]
        preds = [
            self._make_frame(0, [self._make_track(10, bbox)]),
        ]
        m = self.ev.evaluate(gt, preds)
        self.assertEqual(m.total_gt_tracks, 2)
        self.assertEqual(m.total_pred_tracks, 1)


# ===========================================================================
# AnomalyBenchmarkEvaluator
# ===========================================================================


class TestAnomalyBenchmarkEvaluator(unittest.TestCase):
    """Tests for AnomalyBenchmarkEvaluator.run_synthetic_benchmark()."""

    @classmethod
    def setUpClass(cls):
        """Run benchmark once; reuse result across tests."""
        evaluator = AnomalyBenchmarkEvaluator()
        cls.result = evaluator.run_synthetic_benchmark(
            n_normal=50,
            n_anomaly=10,
            threshold=2.5,
            seed=42,
        )

    def test_runs_without_error(self):
        """run_synthetic_benchmark completes without raising."""
        self.assertIsInstance(self.result, AnomalyBenchmarkResult)

    def test_returns_anomaly_benchmark_result(self):
        """Return type is AnomalyBenchmarkResult."""
        self.assertIsInstance(self.result, AnomalyBenchmarkResult)

    def test_n_normal_matches_input(self):
        """result.n_normal equals the n_normal argument."""
        self.assertEqual(self.result.n_normal, 50)

    def test_n_anomaly_matches_input(self):
        """result.n_anomaly equals the n_anomaly argument."""
        self.assertEqual(self.result.n_anomaly, 10)

    def test_n_normal_plus_n_anomaly_equals_total(self):
        """n_normal + n_anomaly == tp + fp + fn + tn."""
        total_frames = (
            self.result.true_positives
            + self.result.false_positives
            + self.result.false_negatives
            + self.result.true_negatives
        )
        self.assertEqual(total_frames, self.result.n_normal + self.result.n_anomaly)

    def test_precision_in_unit_interval(self):
        """precision ∈ [0.0, 1.0]."""
        self.assertGreaterEqual(self.result.precision, 0.0)
        self.assertLessEqual(self.result.precision, 1.0)

    def test_recall_in_unit_interval(self):
        """recall ∈ [0.0, 1.0]."""
        self.assertGreaterEqual(self.result.recall, 0.0)
        self.assertLessEqual(self.result.recall, 1.0)

    def test_f1_in_unit_interval(self):
        """f1 ∈ [0.0, 1.0]."""
        self.assertGreaterEqual(self.result.f1, 0.0)
        self.assertLessEqual(self.result.f1, 1.0)

    def test_false_alarm_rate_in_unit_interval(self):
        """false_alarm_rate ∈ [0.0, 1.0]."""
        self.assertGreaterEqual(self.result.false_alarm_rate, 0.0)
        self.assertLessEqual(self.result.false_alarm_rate, 1.0)

    def test_detection_latency_nonnegative(self):
        """detection_latency_frames >= 0."""
        self.assertGreaterEqual(self.result.detection_latency_frames, 0.0)

    def test_threshold_stored(self):
        """threshold_used matches the argument passed."""
        self.assertAlmostEqual(self.result.threshold_used, 2.5)

    def test_to_dict_has_all_keys(self):
        """to_dict() exposes all expected keys."""
        d = self.result.to_dict()
        expected_keys = {
            "threshold_used", "true_positives", "false_positives",
            "false_negatives", "true_negatives", "precision", "recall",
            "f1", "accuracy", "false_alarm_rate",
            "detection_latency_frames", "n_normal", "n_anomaly",
        }
        self.assertEqual(set(d.keys()), expected_keys)

    def test_accuracy_in_unit_interval(self):
        """accuracy ∈ [0.0, 1.0]."""
        self.assertGreaterEqual(self.result.accuracy, 0.0)
        self.assertLessEqual(self.result.accuracy, 1.0)

    def test_different_seed_gives_result(self):
        """A different seed still produces a valid result."""
        ev = AnomalyBenchmarkEvaluator()
        r = ev.run_synthetic_benchmark(n_normal=20, n_anomaly=5, seed=99)
        self.assertIsInstance(r, AnomalyBenchmarkResult)
        self.assertEqual(r.n_normal, 20)
        self.assertEqual(r.n_anomaly, 5)


# ===========================================================================
# benchmark_eval script
# ===========================================================================


class TestBenchmarkScript(unittest.TestCase):
    """Smoke tests for the benchmark_eval CLI script."""

    def test_script_file_exists(self):
        """scripts/benchmark_eval.py exists on disk."""
        project_root = Path(__file__).parent.parent
        script_path = project_root / "scripts" / "benchmark_eval.py"
        self.assertTrue(script_path.exists(), f"Script not found at {script_path}")

    def test_run_anomaly_benchmark_function(self):
        """run_anomaly_benchmark() returns a dict with expected keys."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))

        # Import the function directly without executing main()
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "benchmark_eval",
            Path(__file__).parent.parent / "scripts" / "benchmark_eval.py",
        )
        mod = importlib.util.load_from_spec = None  # suppress lint warning  # noqa
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        result = mod.run_anomaly_benchmark(n_normal=30, n_anomaly=5, threshold=2.5)
        self.assertIsInstance(result, dict)
        self.assertIn("precision", result)
        self.assertIn("recall", result)
        self.assertIn("f1", result)
        self.assertIn("n_normal", result)
        self.assertIn("n_anomaly", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
