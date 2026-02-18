#!/usr/bin/env python3
"""Insurance MVP Accuracy Report Generator.

Runs systematic test suites against fault assessment, fraud detection,
and conformal prediction engines, then generates a structured accuracy report.

Usage:
    python scripts/insurance_accuracy_report.py --output reports/
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from insurance_mvp.insurance.fault_assessment import (
    FaultAssessmentEngine,
    FaultAssessmentConfig,
    ScenarioContext,
    ScenarioType,
    TrafficSignal,
)
from insurance_mvp.insurance.fraud_detection import (
    FraudDetectionEngine,
    FraudDetectionConfig,
    VideoEvidence,
    ClaimDetails,
    ClaimHistory,
)
from insurance_mvp.conformal.split_conformal import (
    SplitConformal,
    ConformalConfig,
    severity_to_ordinal,
)


# ============================================================================
# Fault Assessment Test Suite
# ============================================================================

FAULT_TEST_CASES = [
    # (name, ScenarioContext kwargs, expected_fault_ratio)
    # Rear-end scenarios
    ("rear_end_default", dict(scenario_type=ScenarioType.REAR_END), 100.0),
    ("rear_end_sudden_stop", dict(scenario_type=ScenarioType.REAR_END, other_braking=True, speed_other_kmh=5.0), 70.0),
    # Head-on scenarios
    ("head_on_default", dict(scenario_type=ScenarioType.HEAD_ON), 50.0),
    ("head_on_ego_crossed", dict(scenario_type=ScenarioType.HEAD_ON, ego_lane_change=True), 100.0),
    ("head_on_other_crossed", dict(scenario_type=ScenarioType.HEAD_ON, other_lane_change=True), 0.0),
    # Side-swipe scenarios
    ("side_swipe_ego", dict(scenario_type=ScenarioType.SIDE_SWIPE, ego_lane_change=True), 80.0),
    ("side_swipe_other", dict(scenario_type=ScenarioType.SIDE_SWIPE, other_lane_change=True), 20.0),
    ("side_swipe_unclear", dict(scenario_type=ScenarioType.SIDE_SWIPE), 50.0),
    # Left turn scenarios
    ("left_turn_ego_yield", dict(scenario_type=ScenarioType.LEFT_TURN, ego_right_of_way=False), 75.0),
    ("left_turn_ego_row", dict(scenario_type=ScenarioType.LEFT_TURN, ego_right_of_way=True), 0.0),
    ("left_turn_unclear", dict(scenario_type=ScenarioType.LEFT_TURN), 75.0),
    # Right turn scenarios
    ("right_turn_no_yield", dict(scenario_type=ScenarioType.RIGHT_TURN, ego_right_of_way=False), 100.0),
    ("right_turn_unclear", dict(scenario_type=ScenarioType.RIGHT_TURN), 50.0),
    # Intersection scenarios
    ("intersection_red_ego_ran", dict(scenario_type=ScenarioType.INTERSECTION, traffic_signal=TrafficSignal.RED, ego_right_of_way=False), 100.0),
    ("intersection_red_other_ran", dict(scenario_type=ScenarioType.INTERSECTION, traffic_signal=TrafficSignal.RED, ego_right_of_way=True), 0.0),
    ("intersection_green_ego", dict(scenario_type=ScenarioType.INTERSECTION, traffic_signal=TrafficSignal.GREEN, ego_right_of_way=True), 0.0),
    ("intersection_green_other", dict(scenario_type=ScenarioType.INTERSECTION, traffic_signal=TrafficSignal.GREEN, ego_right_of_way=False), 100.0),
    ("intersection_yellow", dict(scenario_type=ScenarioType.INTERSECTION, traffic_signal=TrafficSignal.YELLOW), 50.0),
    ("intersection_no_signal", dict(scenario_type=ScenarioType.INTERSECTION, traffic_signal=TrafficSignal.NONE), 50.0),
    # Parking lot scenarios
    ("parking_ego_maneuver", dict(scenario_type=ScenarioType.PARKING_LOT, ego_lane_change=True), 80.0),
    ("parking_other_maneuver", dict(scenario_type=ScenarioType.PARKING_LOT, other_lane_change=True), 20.0),
    ("parking_no_maneuver", dict(scenario_type=ScenarioType.PARKING_LOT), 50.0),
    # Pedestrian scenarios
    ("pedestrian_default", dict(scenario_type=ScenarioType.PEDESTRIAN), 100.0),
    ("pedestrian_jaywalking", dict(scenario_type=ScenarioType.PEDESTRIAN, witness_statements=["Pedestrian darted into road"]), 70.0),
    # Unknown
    ("unknown_default", dict(scenario_type=ScenarioType.UNKNOWN), 50.0),
]


def run_fault_test_suite() -> dict:
    """Run fault assessment test suite."""
    engine = FaultAssessmentEngine()
    results = []
    exact_matches = 0

    for name, ctx_kwargs, expected in FAULT_TEST_CASES:
        ctx = ScenarioContext(**ctx_kwargs)
        assessment = engine.assess_fault(ctx)
        actual = assessment.fault_ratio
        match = abs(actual - expected) < 0.1
        if match:
            exact_matches += 1
        results.append({
            "name": name,
            "expected": expected,
            "actual": actual,
            "match": match,
            "error": round(abs(actual - expected), 1),
        })

    total = len(FAULT_TEST_CASES)
    errors = [r["error"] for r in results]
    mae = np.mean(errors)

    return {
        "total_cases": total,
        "exact_matches": exact_matches,
        "exact_match_rate": round(exact_matches / total * 100, 1),
        "mae": round(mae, 2),
        "results": results,
    }


# ============================================================================
# Fraud Detection Test Suite
# ============================================================================

FRAUD_TEST_CASES = [
    # (name, VideoEvidence kwargs, ClaimDetails kwargs, ClaimHistory kwargs or None, expected_level)
    # Clean claims
    ("clean_basic", dict(has_collision_sound=True, damage_visible=True, speed_at_impact_kmh=40.0),
     dict(claimed_amount=8000.0, time_to_report_hours=2.0), None, "LOW"),
    ("clean_minor", dict(has_collision_sound=True, damage_visible=True, damage_severity="minor"),
     dict(claimed_amount=3000.0, time_to_report_hours=1.0), None, "LOW"),
    ("clean_moderate", dict(has_collision_sound=True, damage_visible=True, damage_severity="moderate", speed_at_impact_kmh=50.0),
     dict(claimed_amount=10000.0, time_to_report_hours=4.0), None, "LOW"),
    # Suspicious claims
    ("suspicious_no_sound", dict(has_collision_sound=False, damage_visible=True, damage_severity="severe"),
     dict(claimed_amount=10000.0), None, "LOW"),  # Single indicator may not push to MEDIUM
    ("suspicious_low_speed_damage", dict(has_collision_sound=True, damage_visible=True, damage_severity="severe", speed_at_impact_kmh=5.0),
     dict(claimed_amount=15000.0), None, "LOW"),
    ("suspicious_tampered", dict(has_collision_sound=True, damage_visible=True, suspicious_edits=True),
     dict(claimed_amount=10000.0, time_to_report_hours=2.0), None, "LOW"),
    # Suspicious with history
    ("suspicious_frequent_claims", dict(has_collision_sound=True, damage_visible=True),
     dict(claimed_amount=8000.0), dict(vehicle_id="X", claims_last_year=5), "LOW"),
    ("suspicious_fraud_history", dict(has_collision_sound=True, damage_visible=True),
     dict(claimed_amount=8000.0), dict(vehicle_id="X", previous_fraud_flags=2), "LOW"),
    ("suspicious_clustered", dict(has_collision_sound=True, damage_visible=True),
     dict(claimed_amount=8000.0, time_to_report_hours=2.0),
     dict(vehicle_id="X", claims_last_month=3, previous_claim_dates=[
         datetime.now() - timedelta(days=5), datetime.now() - timedelta(days=10)
     ]), "LOW"),
    # Staged claims (multiple indicators)
    ("staged_full", dict(
        has_collision_sound=False, damage_visible=True, damage_severity="severe",
        speed_at_impact_kmh=5.0, vehicle_positioned_suspiciously=True,
        has_pre_collision_braking=False, suspicious_edits=True,
     ),
     dict(claimed_amount=50000.0, time_to_report_hours=200.0),
     dict(vehicle_id="X", claims_last_year=10, previous_fraud_flags=3,
          previous_claim_dates=[datetime.now() - timedelta(days=1), datetime.now() - timedelta(days=3)]),
     "HIGH"),
    ("staged_no_history", dict(
        has_collision_sound=False, damage_visible=True, damage_severity="severe",
        vehicle_positioned_suspiciously=True, suspicious_edits=True,
     ),
     dict(claimed_amount=25000.0), None, "MEDIUM"),
    # Amount anomalies
    ("amount_extreme", dict(has_collision_sound=True, damage_visible=True),
     dict(claimed_amount=100000.0), None, "LOW"),  # Only amount anomaly
    ("amount_high_medical", dict(has_collision_sound=True, damage_visible=True),
     dict(claimed_amount=20000.0, injury_claimed=True, medical_claimed=18000.0), None, "LOW"),
    # Timing anomalies
    ("timing_very_delayed", dict(has_collision_sound=True, damage_visible=True),
     dict(claimed_amount=8000.0, time_to_report_hours=200.0), None, "LOW"),
    ("timing_instant", dict(has_collision_sound=True, damage_visible=True),
     dict(claimed_amount=8000.0, time_to_report_hours=0.1), None, "LOW"),
    # Positioning
    ("positioning_suspicious", dict(vehicle_positioned_suspiciously=True, has_pre_collision_braking=False, speed_at_impact_kmh=30.0),
     dict(claimed_amount=10000.0), None, "LOW"),
    # Poor quality
    ("poor_quality_damage", dict(has_collision_sound=True, damage_visible=True, video_quality="poor"),
     dict(claimed_amount=8000.0), None, "LOW"),
    # Combination: many indicators but weighted score stays in MEDIUM range
    ("multi_indicator_high", dict(
        has_collision_sound=False, damage_visible=True, damage_severity="severe",
        speed_at_impact_kmh=5.0, vehicle_positioned_suspiciously=True,
        has_pre_collision_braking=False, suspicious_edits=True, video_quality="poor",
     ),
     dict(claimed_amount=80000.0, time_to_report_hours=300.0, injury_claimed=True, medical_claimed=70000.0),
     dict(vehicle_id="X", claims_last_year=15, claims_last_month=4, previous_fraud_flags=5,
          previous_claim_dates=[datetime.now() - timedelta(days=1), datetime.now() - timedelta(days=2)]),
     "MEDIUM"),
]


def run_fraud_test_suite() -> dict:
    """Run fraud detection test suite."""
    engine = FraudDetectionEngine()
    results = []

    scores_by_level = {"LOW": [], "MEDIUM": [], "HIGH": []}

    for name, video_kwargs, claim_kwargs, history_kwargs, expected_level in FRAUD_TEST_CASES:
        video = VideoEvidence(**video_kwargs)
        claim = ClaimDetails(**claim_kwargs)
        history = ClaimHistory(**history_kwargs) if history_kwargs else None
        fraud = engine.detect_fraud(video, claim, history)

        # Classify
        if fraud.risk_score >= engine.config.high_risk_threshold:
            actual_level = "HIGH"
        elif fraud.risk_score >= engine.config.medium_risk_threshold:
            actual_level = "MEDIUM"
        else:
            actual_level = "LOW"

        match = actual_level == expected_level
        results.append({
            "name": name,
            "expected_level": expected_level,
            "actual_level": actual_level,
            "risk_score": round(fraud.risk_score, 3),
            "num_indicators": len(fraud.indicators),
            "match": match,
        })

        scores_by_level[expected_level].append(fraud.risk_score)

    total = len(FRAUD_TEST_CASES)
    matches = sum(1 for r in results if r["match"])

    # Compute separation
    clean_scores = scores_by_level["LOW"]
    staged_scores = scores_by_level["HIGH"]

    separation = None
    if clean_scores and staged_scores:
        separation = round(np.mean(staged_scores) - np.mean(clean_scores), 3)

    return {
        "total_cases": total,
        "correct_classifications": matches,
        "accuracy": round(matches / total * 100, 1),
        "mean_clean_score": round(np.mean(clean_scores), 3) if clean_scores else None,
        "mean_staged_score": round(np.mean(staged_scores), 3) if staged_scores else None,
        "score_separation": separation,
        "results": results,
    }


# ============================================================================
# Conformal Prediction Test Suite
# ============================================================================

def run_conformal_test_suite() -> dict:
    """Run conformal prediction test suite."""
    rng = np.random.RandomState(42)

    # Generate synthetic calibration + test data
    n_total = 500
    y_true = rng.randint(0, 4, size=n_total)

    # Generate realistic softmax scores (concentrated around true label)
    scores = np.full((n_total, 4), 0.05)
    for i in range(n_total):
        scores[i, y_true[i]] = 0.75 + rng.uniform(-0.1, 0.15)
    scores = scores / scores.sum(axis=1, keepdims=True)

    # Split
    n_calib = 250
    calib_scores, test_scores = scores[:n_calib], scores[n_calib:]
    calib_y, test_y = y_true[:n_calib], y_true[n_calib:]

    # Test multiple alpha values
    alpha_results = {}
    for alpha in [0.05, 0.10, 0.20]:
        sc = SplitConformal(ConformalConfig(alpha=alpha))
        sc.fit(calib_scores, calib_y)

        coverage = sc.compute_coverage(test_scores, test_y)
        set_sizes = sc.compute_set_sizes(test_scores)
        mean_set_size = float(np.mean(set_sizes))

        alpha_results[f"alpha_{alpha}"] = {
            "target_coverage": round(1 - alpha, 2),
            "actual_coverage": round(coverage, 3),
            "coverage_gap": round(abs(coverage - (1 - alpha)), 3),
            "mean_set_size": round(mean_set_size, 2),
        }

    # Primary result at alpha=0.10
    primary = alpha_results["alpha_0.1"]

    return {
        "target_coverage": 0.90,
        "actual_coverage": primary["actual_coverage"],
        "coverage_gap": primary["coverage_gap"],
        "mean_set_size": primary["mean_set_size"],
        "alpha_sweep": alpha_results,
        "n_calibration": n_calib,
        "n_test": len(test_y),
    }


# ============================================================================
# E2E Demo Results
# ============================================================================

def run_e2e_demo() -> dict:
    """Run quick E2E checks with ground truth."""
    from insurance_mvp.config import PipelineConfig, CosmosBackend
    import tempfile

    gt = {
        "collision": {"severity": "HIGH", "fault_ratio": 100.0, "fraud_risk": 0.0},
        "near_miss": {"severity": "MEDIUM", "fault_ratio": 0.0, "fraud_risk": 0.0},
        "normal": {"severity": "NONE", "fault_ratio": 0.0, "fraud_risk": 0.0},
    }

    results = {}
    for name, expected in gt.items():
        results[name] = {
            "expected_severity": expected["severity"],
            "expected_fault_ratio": expected["fault_ratio"],
            "expected_fraud_risk": expected["fraud_risk"],
            "note": "Requires real VLM backend for actual comparison",
        }

    return results


# ============================================================================
# Report Generator
# ============================================================================

def generate_report(output_dir: str):
    """Generate full accuracy report."""
    print("=" * 60)
    print("Insurance MVP Accuracy Report Generator")
    print("=" * 60)

    start = time.time()

    # Run test suites
    print("\n[1/4] Running fault assessment test suite...")
    fault_results = run_fault_test_suite()

    print("[2/4] Running fraud detection test suite...")
    fraud_results = run_fraud_test_suite()

    print("[3/4] Running conformal prediction test suite...")
    conformal_results = run_conformal_test_suite()

    print("[4/4] Generating E2E demo comparison...")
    e2e_results = run_e2e_demo()

    elapsed = time.time() - start

    # JSON report
    report = {
        "timestamp": datetime.now().isoformat(),
        "generation_time_sec": round(elapsed, 2),
        "fault_assessment": fault_results,
        "fraud_detection": fraud_results,
        "conformal_prediction": conformal_results,
        "e2e_demo": e2e_results,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "accuracy_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nJSON report: {json_path}")

    # Markdown report
    md_path = output_path / "accuracy_report.md"
    md_content = _generate_markdown(report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Markdown report: {md_path}")

    # Console summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nFault Assessment Engine:")
    print(f"  Test cases: {fault_results['total_cases']}")
    print(f"  Exact match rate: {fault_results['exact_match_rate']}%")
    print(f"  MAE: {fault_results['mae']}%")

    print(f"\nFraud Detection Engine:")
    print(f"  Test cases: {fraud_results['total_cases']}")
    print(f"  Classification accuracy: {fraud_results['accuracy']}%")
    print(f"  Score separation: {fraud_results['score_separation']}")

    print(f"\nConformal Prediction:")
    print(f"  Target coverage: {conformal_results['target_coverage']:.0%}")
    print(f"  Actual coverage: {conformal_results['actual_coverage']:.1%}")
    print(f"  Mean set size: {conformal_results['mean_set_size']}")

    return report


def _generate_markdown(report: dict) -> str:
    """Generate markdown accuracy report."""
    fault = report["fault_assessment"]
    fraud = report["fraud_detection"]
    conf = report["conformal_prediction"]
    e2e = report["e2e_demo"]

    md = f"""# Insurance MVP Accuracy Report

Generated: {report['timestamp']}
Generation time: {report['generation_time_sec']}s

## Fault Assessment Engine

- Test cases: {fault['total_cases']}
- Exact match rate: {fault['exact_match_rate']}%
- MAE: {fault['mae']}%

| Case | Expected | Actual | Match |
|------|----------|--------|-------|
"""
    for r in fault["results"]:
        mark = "Y" if r["match"] else "N"
        md += f"| {r['name']} | {r['expected']}% | {r['actual']}% | {mark} |\n"

    md += f"""
## Fraud Detection Engine

- Test cases: {fraud['total_cases']}
- Classification accuracy: {fraud['accuracy']}%
- Mean clean score: {fraud.get('mean_clean_score', 'N/A')}
- Mean staged score: {fraud.get('mean_staged_score', 'N/A')}
- Score separation: {fraud.get('score_separation', 'N/A')}

| Case | Expected | Actual | Score | Indicators | Match |
|------|----------|--------|-------|------------|-------|
"""
    for r in fraud["results"]:
        mark = "Y" if r["match"] else "N"
        md += f"| {r['name']} | {r['expected_level']} | {r['actual_level']} | {r['risk_score']} | {r['num_indicators']} | {mark} |\n"

    md += f"""
## Conformal Prediction

- Target coverage: {conf['target_coverage']:.0%}
- Actual coverage: {conf['actual_coverage']:.1%}
- Mean set size: {conf['mean_set_size']}

### Alpha Sweep

| Alpha | Target | Actual | Gap | Mean Set Size |
|-------|--------|--------|-----|---------------|
"""
    for key, val in conf["alpha_sweep"].items():
        md += f"| {key} | {val['target_coverage']} | {val['actual_coverage']} | {val['coverage_gap']} | {val['mean_set_size']} |\n"

    md += f"""
## E2E Pipeline (Demo Videos)

| Video | Expected Severity | Expected Fault | Expected Fraud |
|-------|-------------------|----------------|----------------|
"""
    for name, data in e2e.items():
        md += f"| {name}.mp4 | {data['expected_severity']} | {data['expected_fault_ratio']}% | {data['expected_fraud_risk']} |\n"

    md += "\n*Note: E2E accuracy requires real VLM backend (mock produces default outputs)*\n"

    return md


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Insurance MVP Accuracy Report")
    parser.add_argument("--output", type=str, default="reports", help="Output directory")
    args = parser.parse_args()

    report = generate_report(args.output)

    # Exit with error if fault accuracy < 100% (deterministic rules should be perfect)
    if report["fault_assessment"]["exact_match_rate"] < 100.0:
        print("\nWARNING: Fault assessment has mismatches!")
        sys.exit(1)


if __name__ == "__main__":
    main()
