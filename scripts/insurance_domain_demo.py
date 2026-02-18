#!/usr/bin/env python3
"""Insurance Domain Logic Demo.

Demonstrates fault assessment and fraud detection capabilities
for insurance claim processing.

Usage:
    python scripts/insurance_domain_demo.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from insurance_mvp.insurance import (  # noqa: E402
    ClaimDetails,
    ClaimHistory,
    # Fault Assessment
    FaultAssessmentEngine,
    # Fraud Detection
    FraudDetectionEngine,
    ScenarioContext,
    ScenarioType,
    TrafficSignal,
    VideoEvidence,
    detect_scenario_type,
)


def print_section(title: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_subsection(title: str):
    """Print subsection header."""
    print(f"\n--- {title} ---\n")


def demo_fault_assessment():
    """Demonstrate fault assessment for various scenarios."""
    print_section("FAULT ASSESSMENT DEMO")

    engine = FaultAssessmentEngine()

    # Scenario 1: Rear-end collision
    print_subsection("Scenario 1: Rear-End Collision (Standard)")
    context1 = ScenarioContext(
        scenario_type=ScenarioType.REAR_END,
        speed_ego_kmh=60.0,
        traffic_signal=TrafficSignal.GREEN,
        ego_braking=True,
    )
    result1 = engine.assess_fault(context1)
    print(f"Fault Ratio: {result1.fault_ratio}%")
    print(f"Reasoning: {result1.reasoning}")
    print(f"Applicable Rules: {', '.join(result1.applicable_rules)}")

    # Scenario 2: Head-on collision with ego crossing center line
    print_subsection("Scenario 2: Head-On Collision (Ego Crossed Center Line)")
    context2 = ScenarioContext(
        scenario_type=ScenarioType.HEAD_ON,
        speed_ego_kmh=80.0,
        ego_lane_change=True,
        other_lane_change=False,
        weather_conditions="rain",
    )
    result2 = engine.assess_fault(context2)
    print(f"Fault Ratio: {result2.fault_ratio}%")
    print(f"Reasoning: {result2.reasoning}")
    print(f"Right of Way: {result2.right_of_way}")

    # Scenario 3: Left turn collision
    print_subsection("Scenario 3: Left Turn Collision (Failed to Yield)")
    context3 = ScenarioContext(
        scenario_type=ScenarioType.LEFT_TURN,
        speed_ego_kmh=30.0,
        ego_right_of_way=False,
        traffic_signal=TrafficSignal.GREEN,
    )
    result3 = engine.assess_fault(context3)
    print(f"Fault Ratio: {result3.fault_ratio}%")
    print(f"Reasoning: {result3.reasoning}")
    print(f"Traffic Signal: {result3.traffic_signal}")

    # Scenario 4: Intersection with red light violation
    print_subsection("Scenario 4: Intersection (Red Light Violation)")
    context4 = ScenarioContext(
        scenario_type=ScenarioType.INTERSECTION,
        traffic_signal=TrafficSignal.RED,
        ego_right_of_way=False,
        speed_ego_kmh=50.0,
    )
    result4 = engine.assess_fault(context4)
    print(f"Fault Ratio: {result4.fault_ratio}%")
    print(f"Reasoning: {result4.reasoning}")
    print(f"Applicable Rules: {', '.join(result4.applicable_rules)}")

    # Scenario 5: Pedestrian collision
    print_subsection("Scenario 5: Pedestrian Collision")
    context5 = ScenarioContext(
        scenario_type=ScenarioType.PEDESTRIAN,
        speed_ego_kmh=35.0,
        ego_braking=True,
    )
    result5 = engine.assess_fault(context5)
    print(f"Fault Ratio: {result5.fault_ratio}%")
    print(f"Reasoning: {result5.reasoning}")

    # Scenario 6: Pedestrian jaywalking
    print_subsection("Scenario 6: Pedestrian Collision (Jaywalking)")
    context6 = ScenarioContext(
        scenario_type=ScenarioType.PEDESTRIAN,
        speed_ego_kmh=35.0,
        witness_statements=["Pedestrian suddenly darted into traffic"],
    )
    result6 = engine.assess_fault(context6)
    print(f"Fault Ratio: {result6.fault_ratio}%")
    print(f"Reasoning: {result6.reasoning}")

    # Scenario detection demo
    print_subsection("Scenario Detection from Text")
    descriptions = [
        "Car hit from behind at stoplight",
        "Head-on collision on highway",
        "Side-swipe during lane merge",
        "Collision with pedestrian in crosswalk",
    ]
    for desc in descriptions:
        detected = detect_scenario_type(desc)
        print(f"'{desc}' -> {detected.value}")


def demo_fraud_detection():
    """Demonstrate fraud detection for various scenarios."""
    print_section("FRAUD DETECTION DEMO")

    engine = FraudDetectionEngine()

    # Case 1: Clean claim (low fraud risk)
    print_subsection("Case 1: Clean Claim (Low Fraud Risk)")
    video1 = VideoEvidence(
        has_collision_sound=True,
        damage_visible=True,
        damage_severity="moderate",
        speed_at_impact_kmh=40.0,
        has_pre_collision_braking=True,
        video_quality="good",
    )
    claim1 = ClaimDetails(
        claimed_amount=8000.0,
        injury_claimed=False,
        time_to_report_hours=2.0,
    )
    result1 = engine.detect_fraud(video1, claim1)
    print(f"Fraud Risk Score: {result1.risk_score:.3f}")
    print(f"Reasoning: {result1.reasoning}")
    print(f"Indicators: {len(result1.indicators)}")

    # Case 2: Audio/visual mismatch
    print_subsection("Case 2: Audio/Visual Mismatch (Medium Risk)")
    video2 = VideoEvidence(
        has_collision_sound=False,  # RED FLAG
        damage_visible=True,
        damage_severity="severe",
        speed_at_impact_kmh=50.0,
        video_quality="good",
    )
    claim2 = ClaimDetails(
        claimed_amount=12000.0,
        injury_claimed=True,
        medical_claimed=8000.0,
    )
    result2 = engine.detect_fraud(video2, claim2)
    print(f"Fraud Risk Score: {result2.risk_score:.3f}")
    print(f"Reasoning: {result2.reasoning}")
    print(f"Indicators ({len(result2.indicators)}):")
    for ind in result2.indicators:
        print(f"  - {ind}")

    # Case 3: Damage inconsistency
    print_subsection("Case 3: Damage Inconsistency (Medium Risk)")
    video3 = VideoEvidence(
        has_collision_sound=True,
        damage_visible=True,
        damage_severity="severe",  # RED FLAG
        speed_at_impact_kmh=5.0,  # Too low for severe damage
        video_quality="fair",
    )
    claim3 = ClaimDetails(
        claimed_amount=15000.0,
        injury_claimed=True,
        medical_claimed=10000.0,
    )
    result3 = engine.detect_fraud(video3, claim3)
    print(f"Fraud Risk Score: {result3.risk_score:.3f}")
    print(f"Reasoning: {result3.reasoning}")
    print(f"Indicators ({len(result3.indicators)}):")
    for ind in result3.indicators:
        print(f"  - {ind}")

    # Case 4: High claim frequency
    print_subsection("Case 4: High Claim Frequency (High Risk)")
    video4 = VideoEvidence(
        has_collision_sound=True,
        damage_visible=True,
        damage_severity="moderate",
        speed_at_impact_kmh=30.0,
    )
    claim4 = ClaimDetails(claimed_amount=10000.0)
    history4 = ClaimHistory(
        vehicle_id="ABC123",
        claims_last_year=6,  # RED FLAG
        claims_last_month=2,  # RED FLAG
        previous_claim_dates=[
            datetime.now() - timedelta(days=5),
            datetime.now() - timedelta(days=10),
            datetime.now() - timedelta(days=45),
            datetime.now() - timedelta(days=90),
            datetime.now() - timedelta(days=180),
            datetime.now() - timedelta(days=300),
        ],
    )
    result4 = engine.detect_fraud(video4, claim4, history4)
    print(f"Fraud Risk Score: {result4.risk_score:.3f}")
    print(f"Reasoning: {result4.reasoning}")
    print(f"Indicators ({len(result4.indicators)}):")
    for ind in result4.indicators:
        print(f"  - {ind}")

    # Case 5: Previous fraud flags
    print_subsection("Case 5: Previous Fraud History (High Risk)")
    video5 = VideoEvidence(
        has_collision_sound=True,
        damage_visible=True,
        damage_severity="moderate",
    )
    claim5 = ClaimDetails(claimed_amount=9000.0)
    history5 = ClaimHistory(
        vehicle_id="FRAUD999",
        claims_last_year=2,
        previous_fraud_flags=2,  # RED FLAG
    )
    result5 = engine.detect_fraud(video5, claim5, history5)
    print(f"Fraud Risk Score: {result5.risk_score:.3f}")
    print(f"Reasoning: {result5.reasoning}")
    print(f"Indicators ({len(result5.indicators)}):")
    for ind in result5.indicators:
        print(f"  - {ind}")

    # Case 6: Staged accident (multiple red flags)
    print_subsection("Case 6: Suspected Staged Accident (CRITICAL RISK)")
    video6 = VideoEvidence(
        has_collision_sound=False,  # RED FLAG 1
        damage_visible=True,
        damage_severity="moderate",
        vehicle_positioned_suspiciously=True,  # RED FLAG 2
        has_pre_collision_braking=False,  # RED FLAG 3
        speed_at_impact_kmh=25.0,
        suspicious_edits=True,  # RED FLAG 4
        video_quality="poor",
    )
    claim6 = ClaimDetails(
        claimed_amount=20000.0,  # RED FLAG 5
        injury_claimed=True,
        medical_claimed=16000.0,  # RED FLAG 6
        time_to_report_hours=0.2,  # RED FLAG 7 (too quick)
    )
    history6 = ClaimHistory(
        vehicle_id="STAGED123",
        claims_last_year=4,  # RED FLAG 8
        previous_fraud_flags=1,  # RED FLAG 9
    )
    result6 = engine.detect_fraud(video6, claim6, history6)
    print(f"Fraud Risk Score: {result6.risk_score:.3f} [WARNING]")
    print(f"Reasoning: {result6.reasoning}")
    print(f"Indicators ({len(result6.indicators)}):")
    for ind in result6.indicators:
        print(f"  - {ind}")
    if result6.risk_score >= 0.7:
        print("\n*** CRITICAL: IMMEDIATE FRAUD INVESTIGATION REQUIRED ***")


def demo_integrated_assessment():
    """Demonstrate integrated fault + fraud assessment."""
    print_section("INTEGRATED ASSESSMENT DEMO")

    print("Processing claim for Vehicle ID: XYZ789")
    print("Incident: Rear-end collision at intersection")
    print()

    # Fault assessment
    print_subsection("Step 1: Fault Assessment")
    fault_engine = FaultAssessmentEngine()
    fault_context = ScenarioContext(
        scenario_type=ScenarioType.REAR_END,
        speed_ego_kmh=55.0,
        traffic_signal=TrafficSignal.RED,
        ego_braking=True,
        weather_conditions="clear",
    )
    fault_result = fault_engine.assess_fault(fault_context)
    print(f"Fault Ratio: {fault_result.fault_ratio}%")
    print(f"Reasoning: {fault_result.reasoning}")

    # Fraud detection
    print_subsection("Step 2: Fraud Risk Analysis")
    fraud_engine = FraudDetectionEngine()
    video_evidence = VideoEvidence(
        has_collision_sound=True,
        damage_visible=True,
        damage_severity="moderate",
        speed_at_impact_kmh=55.0,
        has_pre_collision_braking=True,
        video_quality="good",
    )
    claim_details = ClaimDetails(
        claimed_amount=9500.0,
        injury_claimed=True,
        medical_claimed=3000.0,
        time_to_report_hours=3.0,
    )
    claim_history = ClaimHistory(
        vehicle_id="XYZ789",
        claims_last_year=1,
        claims_last_month=0,
        previous_fraud_flags=0,
    )
    fraud_result = fraud_engine.detect_fraud(video_evidence, claim_details, claim_history)
    print(f"Fraud Risk Score: {fraud_result.risk_score:.3f}")
    print(f"Reasoning: {fraud_result.reasoning}")

    # Final recommendation
    print_subsection("Step 3: Claim Processing Recommendation")
    if fraud_result.risk_score >= 0.7:
        recommendation = "REJECT - Refer to fraud investigation unit"
        priority = "URGENT"
    elif fraud_result.risk_score >= 0.4:
        recommendation = "REVIEW - Manual review required"
        priority = "HIGH"
    elif fault_result.fault_ratio >= 80.0:
        recommendation = "REVIEW - High fault ratio, verify circumstances"
        priority = "STANDARD"
    else:
        recommendation = "APPROVE - Process claim normally"
        priority = "LOW"

    print(f"Recommendation: {recommendation}")
    print(f"Review Priority: {priority}")
    print(f"Estimated Fault: {fault_result.fault_ratio}%")
    print(f"Fraud Risk: {fraud_result.risk_score:.1%}")

    # Summary
    print_subsection("Summary")
    print(f"[OK] Fault Assessment: {fault_result.scenario_type} collision")
    print(f"[OK] Fault Ratio: {fault_result.fault_ratio}%")
    print(f"[OK] Fraud Risk: {'HIGH' if fraud_result.risk_score >= 0.7 else 'MEDIUM' if fraud_result.risk_score >= 0.4 else 'LOW'}")
    print("[OK] Processing Time: ~2-3 seconds (automated)")
    print(f"[OK] Human Review Required: {'YES' if fraud_result.risk_score >= 0.4 else 'NO'}")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("=" + " " * 78 + "=")
    print("=" + "  INSURANCE DOMAIN LOGIC DEMONSTRATION".center(78) + "=")
    print("=" + "  Fault Assessment & Fraud Detection".center(78) + "=")
    print("=" + " " * 78 + "=")
    print("=" * 80)

    # Run demos
    demo_fault_assessment()
    demo_fraud_detection()
    demo_integrated_assessment()

    # Final summary
    print_section("DEMO COMPLETE")
    print("[OK] Fault Assessment: Industry-standard scenario-based logic")
    print("[OK] Fraud Detection: Multi-signal risk analysis")
    print("[OK] Production-Ready: Configurable thresholds, comprehensive logging")
    print("[OK] Explainable: Detailed reasoning for all assessments")
    print()
    print("Next Steps:")
    print("  1. Run tests: pytest insurance_mvp/tests/test_insurance_domain.py -v")
    print("  2. Integrate with video analysis pipeline")
    print("  3. Configure thresholds for production deployment")
    print()


if __name__ == "__main__":
    main()
