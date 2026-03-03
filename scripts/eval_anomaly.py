"""CLI tool for anomaly detection evaluation and threshold tuning.

Usage examples::

    # Run demo with 200 synthetic snapshots (100 normal + 100 anomalous)
    python scripts/eval_anomaly.py --demo

    # Evaluate at a specific sigma threshold
    python scripts/eval_anomaly.py --demo --threshold 2.5

    # Grid-search for optimal threshold
    python scripts/eval_anomaly.py --demo --optimize
"""

from __future__ import annotations

import argparse
import math
import sys
import time

# Ensure project root is on the path when run directly
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sopilot.perception.anomaly import AnomalyDetectorEnsemble
from sopilot.perception.anomaly_eval import (
    AnomalyEvaluator,
    AnomalyFalsePositiveFilter,
    LabeledAnomalyEvent,
)
from sopilot.perception.types import (
    BBox,
    EntityEvent,
    PerceptionConfig,
    SceneEntity,
    SceneGraph,
    Track,
    TrackState,
    WorldState,
)


# ---------------------------------------------------------------------------
# Synthetic data generation helpers
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
    frame_number: int,
    timestamp: float,
    entity_count: int = 3,
    velocity: tuple[float, float] = (0.01, 0.0),
    activity: str = "walking",
    entity_x: float = 0.2,
) -> WorldState:
    """Build a synthetic WorldState for testing."""
    entities = [
        _make_entity(entity_id=i + 1, x1=entity_x, y1=0.1 + i * 0.1, x2=entity_x + 0.15, y2=0.4 + i * 0.05)
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


def build_synthetic_dataset(
    n_normal: int = 100,
    n_anomalous: int = 100,
) -> list[LabeledAnomalyEvent]:
    """Generate synthetic labeled events.

    Normal frames:  3 entities, moderate walking speed (~0.01), common area
    Anomalous frames: high speed burst, unusual area (x ≈ 0.95), 10 entities
    """
    events: list[LabeledAnomalyEvent] = []
    base_ts = time.time()

    # --- Normal frames (first half) ---
    for i in range(n_normal):
        ts = base_ts + i * 0.5
        ws = _make_world_state(
            frame_number=i,
            timestamp=ts,
            entity_count=3,
            velocity=(0.01, 0.0),
            activity="walking",
            entity_x=0.2,
        )
        events.append(
            LabeledAnomalyEvent(
                world_state=ws,
                is_anomaly=False,
                frame_number=i,
                timestamp=ts,
                note="normal_walking",
            )
        )

    # --- Anomalous frames (second half) ---
    for j in range(n_anomalous):
        frame_num = n_normal + j
        ts = base_ts + frame_num * 0.5

        # Alternate between two anomaly types for variety
        if j % 2 == 0:
            # Speed anomaly: velocity 20x normal
            ws = _make_world_state(
                frame_number=frame_num,
                timestamp=ts,
                entity_count=3,
                velocity=(0.20, 0.10),
                activity="running",
                entity_x=0.2,
            )
            note = "anomaly_high_speed"
        else:
            # Spatial anomaly: entity in rare corner cell + many entities
            ws = _make_world_state(
                frame_number=frame_num,
                timestamp=ts,
                entity_count=10,
                velocity=(0.01, 0.0),
                activity="walking",
                entity_x=0.93,
            )
            note = "anomaly_rare_zone"

        events.append(
            LabeledAnomalyEvent(
                world_state=ws,
                is_anomaly=True,
                frame_number=frame_num,
                timestamp=ts,
                note=note,
            )
        )

    return events


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------


def _print_result(result, label: str = "") -> None:
    tag = f" [{label}]" if label else ""
    print(f"\n{'='*60}")
    print(f"Evaluation Result{tag}")
    print(f"{'='*60}")
    print(f"  Threshold       : {result.threshold:.2f}")
    print(f"  True  Positives : {result.true_positives}")
    print(f"  False Positives : {result.false_positives}")
    print(f"  True  Negatives : {result.true_negatives}")
    print(f"  False Negatives : {result.false_negatives}")
    print(f"  Precision       : {result.precision:.4f}")
    print(f"  Recall          : {result.recall:.4f}")
    print(f"  F1 Score        : {result.f1:.4f}")
    print(f"  Accuracy        : {result.accuracy:.4f}")


def _print_report(report: dict) -> None:
    print(f"\n{'='*60}")
    print("Full Evaluation Report")
    print(f"{'='*60}")
    print(f"  Total evaluations run : {report['total_evaluations']}")
    for r in report["results"]:
        print(
            f"  threshold={r['threshold']:.2f}  "
            f"P={r['precision']:.3f}  R={r['recall']:.3f}  "
            f"F1={r['f1']:.3f}  Acc={r['accuracy']:.3f}"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Anomaly detection evaluation framework — SOPilot v1.7.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo evaluation with 200 synthetic frames (100 normal + 100 anomalous)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        metavar="FLOAT",
        help="sigma_threshold to evaluate at (default: 2.0)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Grid-search for optimal sigma_threshold via F1 maximization",
    )

    args = parser.parse_args()

    if not args.demo:
        print("No action specified. Use --demo to run a demo evaluation.")
        print("Run with --help for full usage information.")
        return 1

    print("Building synthetic evaluation dataset (200 frames)...")
    labeled_events = build_synthetic_dataset(n_normal=100, n_anomalous=100)
    print(f"  Normal frames    : {sum(1 for e in labeled_events if not e.is_anomaly)}")
    print(f"  Anomalous frames : {sum(1 for e in labeled_events if e.is_anomaly)}")

    # Build ensemble with low warmup so anomaly detection kicks in during eval
    config = PerceptionConfig(
        anomaly_warmup_frames=10,
        anomaly_sigma_threshold=args.threshold,
        anomaly_cooldown_seconds=1.0,  # short cooldown for demo
    )
    ensemble = AnomalyDetectorEnsemble(config)
    evaluator = AnomalyEvaluator(ensemble)

    if args.optimize:
        print(f"\nRunning grid search over thresholds [1.0 .. 4.0]...")
        best_thresh, best_result = evaluator.find_optimal_threshold(labeled_events)
        _print_result(best_result, label=f"OPTIMAL threshold={best_thresh:.2f}")
    else:
        print(f"\nEvaluating at sigma_threshold={args.threshold:.2f}...")
        result = evaluator.evaluate(labeled_events, threshold=args.threshold)
        _print_result(result)

    _print_report(evaluator.generate_report())

    # Demo: FP filter stats
    print(f"\n{'='*60}")
    print("False Positive Filter Demo")
    print(f"{'='*60}")
    fp_filter = AnomalyFalsePositiveFilter(fp_threshold=0.7, cooldown_multiplier=3.0)
    fp_filter.record("behavioral", "speed_zscore", is_fp=True)
    fp_filter.record("behavioral", "speed_zscore", is_fp=True)
    fp_filter.record("behavioral", "speed_zscore", is_fp=False)
    fp_filter.record("spatial", "rare_cell", is_fp=True)
    fp_filter.record("spatial", "rare_cell", is_fp=True)
    fp_filter.record("spatial", "rare_cell", is_fp=True)
    stats = fp_filter.get_stats()
    print(f"  FP threshold     : {stats['fp_threshold']}")
    print(f"  Cooldown mult.   : {stats['cooldown_multiplier']}x")
    print(f"  Suppressed pairs : {stats['suppressed_count']}")
    for pair in stats["pairs"]:
        flag = " [SUPPRESSED]" if pair["suppressed"] else ""
        print(
            f"    {pair['detector']}/{pair['metric']}: "
            f"fp_rate={pair['fp_rate']:.2f} "
            f"({pair['fp_count']}/{pair['total_detections']}){flag}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
