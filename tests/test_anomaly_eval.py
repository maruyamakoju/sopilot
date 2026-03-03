"""Tests for the Anomaly Detection Evaluation Framework.

Tests cover:
    - LabeledAnomalyEvent creation
    - AnomalyEvalResult properties (precision, recall, F1, accuracy)
    - AnomalyEvalResult.to_dict()
    - AnomalyEvaluator.evaluate() — all-TP scenario
    - AnomalyEvaluator.evaluate() — all-FP scenario
    - AnomalyEvaluator.find_optimal_threshold()
    - AnomalyEvaluator.generate_report()
    - AnomalyFalsePositiveFilter.record() / get_fp_rate()
    - AnomalyFalsePositiveFilter.get_suppressed_pairs()
    - AnomalyFalsePositiveFilter.apply_to_ensemble()
    - AnomalyFalsePositiveFilter.get_stats()

All tests use mock WorldState objects (no GPU required).
Run:  python -m pytest tests/test_anomaly_eval.py -v
"""

from __future__ import annotations

import time
import unittest

from sopilot.perception.anomaly import AnomalyDetectorEnsemble
from sopilot.perception.anomaly_eval import (
    AnomalyEvalResult,
    AnomalyEvaluator,
    AnomalyFalsePositiveFilter,
    LabeledAnomalyEvent,
)
from sopilot.perception.types import (
    BBox,
    EntityEvent,
    EntityEventType,
    PerceptionConfig,
    SceneEntity,
    SceneGraph,
    Track,
    TrackState,
    WorldState,
)


# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_anomaly_detector.py conventions)
# ---------------------------------------------------------------------------


def _make_entity(
    entity_id: int = 1,
    label: str = "person",
    x1: float = 0.1,
    y1: float = 0.1,
    x2: float = 0.3,
    y2: float = 0.5,
) -> SceneEntity:
    return SceneEntity(
        entity_id=entity_id,
        label=label,
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        confidence=0.9,
    )


def _make_track(
    track_id: int = 1,
    label: str = "person",
    velocity: tuple[float, float] = (0.01, 0.0),
    activity: str = "walking",
) -> Track:
    return Track(
        track_id=track_id,
        label=label,
        state=TrackState.ACTIVE,
        bbox=BBox(0.1, 0.1, 0.3, 0.5),
        velocity=velocity,
        confidence=0.9,
        attributes={"activity": activity},
    )


def _make_world_state(
    frame_number: int = 1,
    timestamp: float = 1.0,
    entity_count: int = 3,
    velocity: tuple[float, float] = (0.01, 0.0),
    activity: str = "walking",
    entity_x: float = 0.2,
) -> WorldState:
    entities = [
        _make_entity(
            entity_id=i + 1,
            x1=entity_x,
            y1=0.1 + i * 0.05,
            x2=entity_x + 0.15,
            y2=0.35 + i * 0.05,
        )
        for i in range(max(1, entity_count))
    ]
    sg = SceneGraph(
        timestamp=timestamp,
        frame_number=frame_number,
        entities=entities,
        relations=[],
        frame_shape=(480, 640),
    )
    active_tracks = {
        e.entity_id: _make_track(
            track_id=e.entity_id,
            label=e.label,
            velocity=velocity,
            activity=activity,
        )
        for e in entities
    }
    return WorldState(
        timestamp=timestamp,
        frame_number=frame_number,
        scene_graph=sg,
        active_tracks=active_tracks,
        events=[],
        zone_occupancy={},
        entity_count=len(entities),
        person_count=len(entities),
    )


def _make_warmed_ensemble(
    sigma_threshold: float = 2.0,
    cooldown_seconds: float = 1.0,
    warmup_frames: int = 5,
) -> AnomalyDetectorEnsemble:
    """Build an ensemble with low warmup suitable for tests."""
    config = PerceptionConfig(
        anomaly_warmup_frames=warmup_frames,
        anomaly_sigma_threshold=sigma_threshold,
        anomaly_cooldown_seconds=cooldown_seconds,
    )
    return AnomalyDetectorEnsemble(config)


def _make_labeled_event(
    frame_number: int = 1,
    timestamp: float = 1.0,
    is_anomaly: bool = False,
    velocity: tuple[float, float] = (0.01, 0.0),
    note: str = "",
) -> LabeledAnomalyEvent:
    ws = _make_world_state(
        frame_number=frame_number,
        timestamp=timestamp,
        velocity=velocity,
    )
    return LabeledAnomalyEvent(
        world_state=ws,
        is_anomaly=is_anomaly,
        frame_number=frame_number,
        timestamp=timestamp,
        note=note,
    )


# ---------------------------------------------------------------------------
# LabeledAnomalyEvent tests
# ---------------------------------------------------------------------------


class TestLabeledAnomalyEvent(unittest.TestCase):
    """Tests for the LabeledAnomalyEvent dataclass."""

    def test_create_normal_event(self):
        ws = _make_world_state()
        event = LabeledAnomalyEvent(
            world_state=ws,
            is_anomaly=False,
            frame_number=1,
            timestamp=1.0,
        )
        self.assertFalse(event.is_anomaly)
        self.assertEqual(event.frame_number, 1)
        self.assertEqual(event.timestamp, 1.0)
        self.assertEqual(event.note, "")

    def test_create_anomaly_event(self):
        ws = _make_world_state()
        event = LabeledAnomalyEvent(
            world_state=ws,
            is_anomaly=True,
            frame_number=42,
            timestamp=100.0,
            note="speed_spike",
        )
        self.assertTrue(event.is_anomaly)
        self.assertEqual(event.frame_number, 42)
        self.assertEqual(event.note, "speed_spike")

    def test_world_state_stored(self):
        ws = _make_world_state(frame_number=7)
        event = LabeledAnomalyEvent(
            world_state=ws,
            is_anomaly=False,
            frame_number=7,
            timestamp=7.0,
        )
        self.assertIs(event.world_state, ws)


# ---------------------------------------------------------------------------
# AnomalyEvalResult tests
# ---------------------------------------------------------------------------


class TestAnomalyEvalResult(unittest.TestCase):
    """Tests for AnomalyEvalResult properties and serialization."""

    def _make_result(self, tp=4, fp=1, fn=1, tn=4, threshold=2.0) -> AnomalyEvalResult:
        return AnomalyEvalResult(
            threshold=threshold,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
        )

    def test_precision(self):
        result = self._make_result(tp=3, fp=1)
        self.assertAlmostEqual(result.precision, 0.75)

    def test_precision_zero_denominator(self):
        result = self._make_result(tp=0, fp=0)
        self.assertEqual(result.precision, 0.0)

    def test_recall(self):
        result = self._make_result(tp=3, fn=1)
        self.assertAlmostEqual(result.recall, 0.75)

    def test_recall_zero_denominator(self):
        result = self._make_result(tp=0, fn=0)
        self.assertEqual(result.recall, 0.0)

    def test_f1_balanced(self):
        result = self._make_result(tp=4, fp=0, fn=0, tn=4)
        self.assertAlmostEqual(result.f1, 1.0)

    def test_f1_zero_when_no_positives(self):
        result = self._make_result(tp=0, fp=0, fn=0, tn=10)
        self.assertEqual(result.f1, 0.0)

    def test_f1_formula(self):
        # precision = 0.75, recall = 0.75 → F1 = 0.75
        result = self._make_result(tp=3, fp=1, fn=1, tn=5)
        self.assertAlmostEqual(result.f1, 0.75, places=4)

    def test_accuracy_perfect(self):
        result = self._make_result(tp=5, fp=0, fn=0, tn=5)
        self.assertAlmostEqual(result.accuracy, 1.0)

    def test_accuracy_mixed(self):
        # 4 correct out of 10
        result = self._make_result(tp=2, fp=3, fn=3, tn=2)
        self.assertAlmostEqual(result.accuracy, 0.4)

    def test_accuracy_zero_samples(self):
        result = self._make_result(tp=0, fp=0, fn=0, tn=0)
        self.assertEqual(result.accuracy, 0.0)

    def test_to_dict_keys(self):
        result = self._make_result()
        d = result.to_dict()
        expected_keys = {
            "threshold", "true_positives", "false_positives",
            "false_negatives", "true_negatives",
            "precision", "recall", "f1", "accuracy",
        }
        self.assertEqual(set(d.keys()), expected_keys)

    def test_to_dict_values(self):
        result = self._make_result(tp=4, fp=1, fn=1, tn=4, threshold=2.5)
        d = result.to_dict()
        self.assertEqual(d["threshold"], 2.5)
        self.assertEqual(d["true_positives"], 4)
        self.assertIsInstance(d["f1"], float)

    def test_to_dict_precision_rounded(self):
        result = self._make_result(tp=1, fp=2, fn=0, tn=7)
        d = result.to_dict()
        # precision = 1/3 ≈ 0.3333
        self.assertAlmostEqual(d["precision"], round(1 / 3, 4), places=4)


# ---------------------------------------------------------------------------
# AnomalyEvaluator tests
# ---------------------------------------------------------------------------


class TestAnomalyEvaluator(unittest.TestCase):
    """Tests for AnomalyEvaluator."""

    def _build_normal_events(self, n: int = 20) -> list[LabeledAnomalyEvent]:
        """Build n identical 'normal' labeled events."""
        base_ts = time.time()
        return [
            _make_labeled_event(
                frame_number=i,
                timestamp=base_ts + i * 0.5,
                is_anomaly=False,
                velocity=(0.01, 0.0),
            )
            for i in range(n)
        ]

    def _build_anomalous_events(self, n: int = 20) -> list[LabeledAnomalyEvent]:
        """Build n 'anomalous' labeled events with extreme velocities."""
        base_ts = time.time() + 1000.0
        return [
            _make_labeled_event(
                frame_number=i,
                timestamp=base_ts + i * 0.5,
                is_anomaly=True,
                velocity=(0.5 + i * 0.01, 0.3),
            )
            for i in range(n)
        ]

    def test_evaluate_returns_eval_result(self):
        ensemble = _make_warmed_ensemble()
        evaluator = AnomalyEvaluator(ensemble)
        events = self._build_normal_events(10)
        result = evaluator.evaluate(events)
        self.assertIsInstance(result, AnomalyEvalResult)

    def test_evaluate_threshold_stored_in_result(self):
        ensemble = _make_warmed_ensemble(sigma_threshold=3.0)
        evaluator = AnomalyEvaluator(ensemble)
        events = self._build_normal_events(5)
        result = evaluator.evaluate(events, threshold=3.0)
        self.assertEqual(result.threshold, 3.0)

    def test_evaluate_counts_are_non_negative(self):
        ensemble = _make_warmed_ensemble()
        evaluator = AnomalyEvaluator(ensemble)
        events = self._build_normal_events(10) + self._build_anomalous_events(10)
        result = evaluator.evaluate(events)
        self.assertGreaterEqual(result.true_positives, 0)
        self.assertGreaterEqual(result.false_positives, 0)
        self.assertGreaterEqual(result.false_negatives, 0)
        self.assertGreaterEqual(result.true_negatives, 0)

    def test_evaluate_total_equals_event_count(self):
        ensemble = _make_warmed_ensemble()
        evaluator = AnomalyEvaluator(ensemble)
        events = self._build_normal_events(8) + self._build_anomalous_events(8)
        result = evaluator.evaluate(events)
        total = (
            result.true_positives
            + result.false_positives
            + result.false_negatives
            + result.true_negatives
        )
        self.assertEqual(total, 16)

    def test_evaluate_all_normal_no_tp(self):
        """All-normal dataset: TP must be 0."""
        ensemble = _make_warmed_ensemble(sigma_threshold=2.0)
        evaluator = AnomalyEvaluator(ensemble)
        events = self._build_normal_events(20)
        result = evaluator.evaluate(events)
        self.assertEqual(result.true_positives, 0)
        self.assertEqual(result.false_negatives, 0)

    def test_evaluate_resets_ensemble_before_run(self):
        """Calling evaluate twice should give consistent results (ensemble is reset)."""
        ensemble = _make_warmed_ensemble()
        evaluator = AnomalyEvaluator(ensemble)
        events = self._build_normal_events(10)
        result1 = evaluator.evaluate(events, threshold=2.0)
        result2 = evaluator.evaluate(events, threshold=2.0)
        self.assertEqual(result1.true_positives, result2.true_positives)
        self.assertEqual(result1.false_positives, result2.false_positives)

    def test_evaluate_threshold_overrides_ensemble(self):
        """Passing threshold=99.0 should suppress nearly all anomalies."""
        ensemble = _make_warmed_ensemble(sigma_threshold=2.0)
        evaluator = AnomalyEvaluator(ensemble)
        events = self._build_normal_events(15) + self._build_anomalous_events(15)
        result = evaluator.evaluate(events, threshold=99.0)
        # With an impossibly high threshold, there should be no detections
        self.assertEqual(result.true_positives, 0)
        self.assertEqual(result.false_positives, 0)

    def test_evaluate_threshold_restored_after_call(self):
        """The ensemble's threshold should be restored after evaluate()."""
        ensemble = _make_warmed_ensemble(sigma_threshold=2.5)
        evaluator = AnomalyEvaluator(ensemble)
        events = self._build_normal_events(5)
        evaluator.evaluate(events, threshold=99.0)
        self.assertAlmostEqual(ensemble._sigma_threshold, 2.5)

    def test_find_optimal_threshold_returns_tuple(self):
        ensemble = _make_warmed_ensemble()
        evaluator = AnomalyEvaluator(ensemble)
        events = self._build_normal_events(10) + self._build_anomalous_events(10)
        best_thresh, best_result = evaluator.find_optimal_threshold(events)
        self.assertIsInstance(best_thresh, float)
        self.assertIsInstance(best_result, AnomalyEvalResult)

    def test_find_optimal_threshold_in_search_space(self):
        ensemble = _make_warmed_ensemble()
        evaluator = AnomalyEvaluator(ensemble)
        events = self._build_normal_events(10) + self._build_anomalous_events(10)
        thresholds = [1.0, 2.0, 3.0]
        best_thresh, _ = evaluator.find_optimal_threshold(events, thresholds=thresholds)
        self.assertIn(best_thresh, thresholds)

    def test_find_optimal_threshold_custom_metric_precision(self):
        ensemble = _make_warmed_ensemble()
        evaluator = AnomalyEvaluator(ensemble)
        events = self._build_normal_events(10) + self._build_anomalous_events(10)
        best_thresh, best_result = evaluator.find_optimal_threshold(
            events, metric="precision"
        )
        self.assertIsInstance(best_thresh, float)

    def test_find_optimal_threshold_invalid_metric_raises(self):
        ensemble = _make_warmed_ensemble()
        evaluator = AnomalyEvaluator(ensemble)
        events = self._build_normal_events(5)
        with self.assertRaises(ValueError):
            evaluator.find_optimal_threshold(events, metric="invalid")

    def test_generate_report_empty(self):
        ensemble = _make_warmed_ensemble()
        evaluator = AnomalyEvaluator(ensemble)
        report = evaluator.generate_report()
        self.assertEqual(report["total_evaluations"], 0)
        self.assertEqual(report["results"], [])

    def test_generate_report_accumulates(self):
        ensemble = _make_warmed_ensemble()
        evaluator = AnomalyEvaluator(ensemble)
        events = self._build_normal_events(5)
        evaluator.evaluate(events)
        evaluator.evaluate(events)
        report = evaluator.generate_report()
        self.assertEqual(report["total_evaluations"], 2)
        self.assertEqual(len(report["results"]), 2)

    def test_generate_report_contains_dicts(self):
        ensemble = _make_warmed_ensemble()
        evaluator = AnomalyEvaluator(ensemble)
        events = self._build_normal_events(5)
        evaluator.evaluate(events)
        report = evaluator.generate_report()
        self.assertIn("threshold", report["results"][0])
        self.assertIn("f1", report["results"][0])


# ---------------------------------------------------------------------------
# AnomalyFalsePositiveFilter tests
# ---------------------------------------------------------------------------


class TestAnomalyFalsePositiveFilter(unittest.TestCase):
    """Tests for AnomalyFalsePositiveFilter."""

    def _make_filter(
        self, fp_threshold: float = 0.7, cooldown_multiplier: float = 3.0
    ) -> AnomalyFalsePositiveFilter:
        return AnomalyFalsePositiveFilter(
            fp_threshold=fp_threshold,
            cooldown_multiplier=cooldown_multiplier,
        )

    # -- record / get_fp_rate -------------------------------------------------

    def test_initial_fp_rate_is_zero(self):
        fp_filter = self._make_filter()
        self.assertEqual(fp_filter.get_fp_rate("behavioral", "speed_zscore"), 0.0)

    def test_record_fp_increases_rate(self):
        fp_filter = self._make_filter()
        fp_filter.record("behavioral", "speed_zscore", is_fp=True)
        self.assertAlmostEqual(fp_filter.get_fp_rate("behavioral", "speed_zscore"), 1.0)

    def test_record_tp_does_not_increase_fp_count(self):
        fp_filter = self._make_filter()
        fp_filter.record("behavioral", "speed_zscore", is_fp=False)
        self.assertEqual(fp_filter.get_fp_rate("behavioral", "speed_zscore"), 0.0)

    def test_fp_rate_mixed(self):
        fp_filter = self._make_filter()
        fp_filter.record("spatial", "rare_cell", is_fp=True)
        fp_filter.record("spatial", "rare_cell", is_fp=True)
        fp_filter.record("spatial", "rare_cell", is_fp=False)
        rate = fp_filter.get_fp_rate("spatial", "rare_cell")
        self.assertAlmostEqual(rate, 2 / 3)

    def test_fp_rate_independent_per_pair(self):
        fp_filter = self._make_filter()
        fp_filter.record("behavioral", "speed_zscore", is_fp=True)
        fp_filter.record("spatial", "rare_cell", is_fp=False)
        self.assertAlmostEqual(fp_filter.get_fp_rate("behavioral", "speed_zscore"), 1.0)
        self.assertEqual(fp_filter.get_fp_rate("spatial", "rare_cell"), 0.0)

    # -- get_suppressed_pairs -------------------------------------------------

    def test_no_suppressed_pairs_initially(self):
        fp_filter = self._make_filter()
        self.assertEqual(fp_filter.get_suppressed_pairs(), [])

    def test_suppressed_when_rate_exceeds_threshold(self):
        fp_filter = self._make_filter(fp_threshold=0.7)
        for _ in range(8):
            fp_filter.record("temporal", "hourly_density", is_fp=True)
        for _ in range(2):
            fp_filter.record("temporal", "hourly_density", is_fp=False)
        suppressed = fp_filter.get_suppressed_pairs()
        self.assertIn(("temporal", "hourly_density"), suppressed)

    def test_not_suppressed_when_rate_below_threshold(self):
        fp_filter = self._make_filter(fp_threshold=0.7)
        fp_filter.record("interaction", "rare_pair", is_fp=True)
        fp_filter.record("interaction", "rare_pair", is_fp=False)
        fp_filter.record("interaction", "rare_pair", is_fp=False)
        # rate = 1/3 ≈ 0.33 < 0.7
        suppressed = fp_filter.get_suppressed_pairs()
        self.assertNotIn(("interaction", "rare_pair"), suppressed)

    def test_multiple_suppressed_pairs(self):
        fp_filter = self._make_filter(fp_threshold=0.5)
        fp_filter.record("behavioral", "speed_zscore", is_fp=True)
        fp_filter.record("behavioral", "speed_zscore", is_fp=True)
        fp_filter.record("spatial", "rare_cell", is_fp=True)
        fp_filter.record("spatial", "rare_cell", is_fp=True)
        suppressed = fp_filter.get_suppressed_pairs()
        self.assertIn(("behavioral", "speed_zscore"), suppressed)
        self.assertIn(("spatial", "rare_cell"), suppressed)

    # -- apply_to_ensemble ----------------------------------------------------

    def test_apply_to_ensemble_sets_fp_cooldown_overrides(self):
        fp_filter = self._make_filter(fp_threshold=0.5, cooldown_multiplier=3.0)
        fp_filter.record("behavioral", "speed_zscore", is_fp=True)
        fp_filter.record("behavioral", "speed_zscore", is_fp=True)

        ensemble = _make_warmed_ensemble(cooldown_seconds=60.0)
        fp_filter.apply_to_ensemble(ensemble)

        self.assertTrue(hasattr(ensemble, "_fp_cooldown_overrides"))
        self.assertIn(("behavioral", "speed_zscore"), ensemble._fp_cooldown_overrides)

    def test_apply_to_ensemble_extended_cooldown_value(self):
        fp_filter = self._make_filter(fp_threshold=0.5, cooldown_multiplier=4.0)
        fp_filter.record("spatial", "rare_cell", is_fp=True)
        fp_filter.record("spatial", "rare_cell", is_fp=True)

        ensemble = _make_warmed_ensemble(cooldown_seconds=60.0)
        fp_filter.apply_to_ensemble(ensemble)

        extended = ensemble._fp_cooldown_overrides.get(("spatial", "rare_cell"))
        self.assertAlmostEqual(extended, 240.0)  # 60 * 4

    def test_apply_to_ensemble_no_suppressed_noop(self):
        fp_filter = self._make_filter(fp_threshold=0.9)
        fp_filter.record("behavioral", "speed_zscore", is_fp=True)
        fp_filter.record("behavioral", "speed_zscore", is_fp=False)

        ensemble = _make_warmed_ensemble()
        fp_filter.apply_to_ensemble(ensemble)

        # No _fp_cooldown_overrides if nothing was suppressed
        overrides = getattr(ensemble, "_fp_cooldown_overrides", {})
        self.assertEqual(len(overrides), 0)

    def test_apply_to_ensemble_patches_cooldown_map(self):
        """Existing cooldown map entries for suppressed pairs are extended."""
        fp_filter = self._make_filter(fp_threshold=0.5, cooldown_multiplier=2.0)
        fp_filter.record("behavioral", "speed_zscore", is_fp=True)
        fp_filter.record("behavioral", "speed_zscore", is_fp=True)

        ensemble = _make_warmed_ensemble(cooldown_seconds=60.0)
        # Manually insert a cooldown entry
        recorded_ts = 1000.0
        ensemble._cooldown_map[("behavioral", "speed_zscore", -1)] = recorded_ts

        fp_filter.apply_to_ensemble(ensemble)

        new_ts = ensemble._cooldown_map[("behavioral", "speed_zscore", -1)]
        # Should be extended by (2-1)*60 = 60 seconds
        self.assertAlmostEqual(new_ts, recorded_ts + 60.0)

    # -- get_stats ------------------------------------------------------------

    def test_get_stats_structure(self):
        fp_filter = self._make_filter()
        stats = fp_filter.get_stats()
        self.assertIn("fp_threshold", stats)
        self.assertIn("cooldown_multiplier", stats)
        self.assertIn("pairs", stats)
        self.assertIn("suppressed_count", stats)

    def test_get_stats_empty(self):
        fp_filter = self._make_filter()
        stats = fp_filter.get_stats()
        self.assertEqual(stats["pairs"], [])
        self.assertEqual(stats["suppressed_count"], 0)

    def test_get_stats_pair_entry_keys(self):
        fp_filter = self._make_filter()
        fp_filter.record("behavioral", "speed_zscore", is_fp=True)
        stats = fp_filter.get_stats()
        pair = stats["pairs"][0]
        for key in ("detector", "metric", "total_detections", "fp_count", "fp_rate", "suppressed"):
            self.assertIn(key, pair)

    def test_get_stats_suppressed_flag(self):
        fp_filter = self._make_filter(fp_threshold=0.5)
        fp_filter.record("spatial", "rare_cell", is_fp=True)
        fp_filter.record("spatial", "rare_cell", is_fp=True)
        stats = fp_filter.get_stats()
        pair = next(p for p in stats["pairs"] if p["metric"] == "rare_cell")
        self.assertTrue(pair["suppressed"])
        self.assertEqual(stats["suppressed_count"], 1)

    def test_get_stats_reflects_fp_threshold(self):
        fp_filter = self._make_filter(fp_threshold=0.65, cooldown_multiplier=5.0)
        stats = fp_filter.get_stats()
        self.assertAlmostEqual(stats["fp_threshold"], 0.65)
        self.assertAlmostEqual(stats["cooldown_multiplier"], 5.0)


if __name__ == "__main__":
    unittest.main()
