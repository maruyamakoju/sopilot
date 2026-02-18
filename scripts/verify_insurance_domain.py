#!/usr/bin/env python3
"""Verification Script for Insurance Domain Logic.

Quick smoke test to verify all components are working correctly.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def verify_imports():
    """Verify all imports work correctly."""
    print("Verifying imports...")
    try:
        from insurance_mvp.insurance import (
            ClaimAssessment,  # noqa: F401
            ClaimDetails,  # noqa: F401
            ClaimHistory,  # noqa: F401
            # Schema
            Evidence,  # noqa: F401
            FaultAssessment,  # noqa: F401
            FaultAssessmentConfig,  # noqa: F401
            # Fault Assessment
            FaultAssessmentEngine,  # noqa: F401
            FraudDetectionConfig,  # noqa: F401
            # Fraud Detection
            FraudDetectionEngine,  # noqa: F401
            FraudIndicator,  # noqa: F401
            FraudRisk,  # noqa: F401
            ScenarioContext,  # noqa: F401
            ScenarioType,  # noqa: F401
            TrafficSignal,  # noqa: F401
            VideoEvidence,  # noqa: F401
            # Utils
            VideoMetadata,  # noqa: F401
            detect_scenario_type,  # noqa: F401
            format_timestamp,  # noqa: F401
            parse_timestamp,  # noqa: F401
        )
        print("  [OK] All imports successful")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def verify_fault_assessment():
    """Verify fault assessment engine."""
    print("\nVerifying fault assessment...")
    try:
        from insurance_mvp.insurance import (
            FaultAssessmentEngine,
            ScenarioContext,
            ScenarioType,
        )

        engine = FaultAssessmentEngine()
        context = ScenarioContext(scenario_type=ScenarioType.REAR_END)
        result = engine.assess_fault(context)

        assert result.fault_ratio == 100.0
        assert "rear_end" in result.scenario_type
        assert len(result.reasoning) > 0

        print("  [OK] Fault assessment engine working")
        return True
    except Exception as e:
        print(f"  [FAIL] Fault assessment error: {e}")
        return False


def verify_fraud_detection():
    """Verify fraud detection engine."""
    print("\nVerifying fraud detection...")
    try:
        from insurance_mvp.insurance import (
            ClaimDetails,
            FraudDetectionEngine,
            VideoEvidence,
        )

        engine = FraudDetectionEngine()
        video = VideoEvidence(has_collision_sound=True, damage_visible=True)
        claim = ClaimDetails(claimed_amount=8000.0, time_to_report_hours=2.0)
        result = engine.detect_fraud(video, claim)

        assert 0.0 <= result.risk_score <= 1.0
        assert len(result.reasoning) > 0

        print("  [OK] Fraud detection engine working")
        return True
    except Exception as e:
        print(f"  [FAIL] Fraud detection error: {e}")
        return False


def verify_utilities():
    """Verify utility functions."""
    print("\nVerifying utilities...")
    try:
        from insurance_mvp.insurance import format_timestamp, parse_timestamp

        # Test timestamp formatting
        formatted = format_timestamp(125.5)
        assert formatted == "00:02:05.50"

        # Test timestamp parsing
        parsed = parse_timestamp("02:05.50")
        assert parsed == 125.5

        # Test roundtrip
        original = 3661.25
        roundtrip = parse_timestamp(format_timestamp(original))
        assert abs(roundtrip - original) < 0.01

        print("  [OK] Utility functions working")
        return True
    except Exception as e:
        print(f"  [FAIL] Utility error: {e}")
        return False


def verify_configuration():
    """Verify configuration classes."""
    print("\nVerifying configuration...")
    try:
        from insurance_mvp.insurance import (
            FaultAssessmentConfig,
            FraudDetectionConfig,
        )

        # Test fault config
        fault_config = FaultAssessmentConfig(
            rear_end_default=100.0,
            excessive_speed_adjustment=10.0,
        )
        assert fault_config.rear_end_default == 100.0

        # Test fraud config
        fraud_config = FraudDetectionConfig(
            high_risk_threshold=0.7,
            weight_audio_visual_mismatch=0.25,
        )
        assert fraud_config.high_risk_threshold == 0.7

        print("  [OK] Configuration classes working")
        return True
    except Exception as e:
        print(f"  [FAIL] Configuration error: {e}")
        return False


def verify_scenario_detection():
    """Verify scenario type detection."""
    print("\nVerifying scenario detection...")
    try:
        from insurance_mvp.insurance import ScenarioType, detect_scenario_type

        scenarios = [
            ("car hit from behind", ScenarioType.REAR_END),
            ("head-on collision", ScenarioType.HEAD_ON),
            ("side-swipe during merge", ScenarioType.SIDE_SWIPE),
            ("pedestrian collision", ScenarioType.PEDESTRIAN),
        ]

        for description, expected_type in scenarios:
            detected = detect_scenario_type(description)
            assert detected == expected_type

        print("  [OK] Scenario detection working")
        return True
    except Exception as e:
        print(f"  [FAIL] Scenario detection error: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 70)
    print("  INSURANCE DOMAIN LOGIC VERIFICATION")
    print("=" * 70)

    results = []

    # Run all checks
    results.append(("Imports", verify_imports()))
    results.append(("Fault Assessment", verify_fault_assessment()))
    results.append(("Fraud Detection", verify_fraud_detection()))
    results.append(("Utilities", verify_utilities()))
    results.append(("Configuration", verify_configuration()))
    results.append(("Scenario Detection", verify_scenario_detection()))

    # Summary
    print("\n" + "=" * 70)
    print("  VERIFICATION SUMMARY")
    print("=" * 70)

    total = len(results)
    passed = sum(1 for _, result in results if result)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {name}")

    print(f"\nTotal: {passed}/{total} checks passed")

    if passed == total:
        print("\n[OK] All verification checks passed!")
        print("\nNext steps:")
        print("  1. Run full test suite: pytest insurance_mvp/tests/test_insurance_domain.py -v")
        print("  2. Run demo: python scripts/insurance_domain_demo.py")
        print("  3. Integrate with video analysis pipeline")
        return 0
    else:
        print(f"\n[FAIL] {total - passed} checks failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
