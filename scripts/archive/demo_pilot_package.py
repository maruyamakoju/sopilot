#!/usr/bin/env python3
"""Demo script for SOPilot Manufacturing Pilot Package.

Runs comprehensive evaluation across all SOPs and trainee variants to
demonstrate the complete commercial product capabilities.

Usage:
    python scripts/demo_pilot_package.py
    python scripts/demo_pilot_package.py --output-dir pilot_demo_reports
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Demo scenarios: (gold, trainee, sop, expected_pass, description)
DEMO_SCENARIOS = [
    # Oil Change
    (
        "demo_videos/manufacturing/oil_change_gold.mp4",
        "demo_videos/manufacturing/oil_change_gold.mp4",
        "oil_change",
        True,
        "Oil Change - Gold vs Gold (Perfect)",
    ),
    (
        "demo_videos/manufacturing/oil_change_gold.mp4",
        "demo_videos/manufacturing/oil_change_trainee_1.mp4",
        "oil_change",
        False,
        "Oil Change - Trainee 1 (Missing SAFETY)",
    ),
    (
        "demo_videos/manufacturing/oil_change_gold.mp4",
        "demo_videos/manufacturing/oil_change_trainee_2.mp4",
        "oil_change",
        False,
        "Oil Change - Trainee 2 (Reversed Order)",
    ),
    (
        "demo_videos/manufacturing/oil_change_gold.mp4",
        "demo_videos/manufacturing/oil_change_trainee_3.mp4",
        "oil_change",
        False,
        "Oil Change - Trainee 3 (Multiple Mistakes)",
    ),
    # Brake Pads
    (
        "demo_videos/manufacturing/brake_pads_gold.mp4",
        "demo_videos/manufacturing/brake_pads_gold.mp4",
        "brake_pads",
        True,
        "Brake Pads - Gold vs Gold (Perfect)",
    ),
    (
        "demo_videos/manufacturing/brake_pads_gold.mp4",
        "demo_videos/manufacturing/brake_pads_trainee_1.mp4",
        "brake_pads",
        False,
        "Brake Pads - Trainee 1 (Skip Torque Check)",
    ),
    (
        "demo_videos/manufacturing/brake_pads_gold.mp4",
        "demo_videos/manufacturing/brake_pads_trainee_2.mp4",
        "brake_pads",
        False,
        "Brake Pads - Trainee 2 (Wrong Order)",
    ),
    # PPE Check
    (
        "demo_videos/manufacturing/ppe_check_gold.mp4",
        "demo_videos/manufacturing/ppe_check_gold.mp4",
        "ppe_check",
        True,
        "PPE Check - Gold vs Gold (Perfect)",
    ),
    (
        "demo_videos/manufacturing/ppe_check_gold.mp4",
        "demo_videos/manufacturing/ppe_check_trainee_1.mp4",
        "ppe_check",
        False,
        "PPE Check - Trainee 1 (No Gloves)",
    ),
    (
        "demo_videos/manufacturing/ppe_check_gold.mp4",
        "demo_videos/manufacturing/ppe_check_trainee_2.mp4",
        "ppe_check",
        False,
        "PPE Check - Trainee 2 (No Glasses)",
    ),
]


def main():
    parser = argparse.ArgumentParser(description="Demo SOPilot Manufacturing Pilot Package")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pilot_demo_reports"),
        help="Directory to save all reports",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SOPilot Manufacturing Pilot Package - Comprehensive Demo")
    print("=" * 80)
    print(f"Output Directory: {output_dir.resolve()}")
    print(f"Scenarios: {len(DEMO_SCENARIOS)}")
    print()

    passed = 0
    failed = 0
    errors = 0

    for idx, (gold, trainee, sop, expected_pass, description) in enumerate(DEMO_SCENARIOS, 1):
        print(f"[{idx}/{len(DEMO_SCENARIOS)}] {description}")
        print(f"  Gold:    {gold}")
        print(f"  Trainee: {trainee}")
        print(f"  SOP:     {sop}")

        # Generate output filename
        output_name = f"{idx:02d}_{sop}_{Path(trainee).stem}.json"
        output_path = output_dir / output_name

        # Run pilot evaluation
        cmd = [
            sys.executable,
            "scripts/sopilot_evaluate_pilot.py",
            "--gold",
            gold,
            "--trainee",
            trainee,
            "--sop",
            sop,
            "--out",
            str(output_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Check result
            actual_pass = result.returncode == 0

            if actual_pass == expected_pass:
                status = "PASS (as expected)"
                if expected_pass:
                    passed += 1
                else:
                    passed += 1  # Correctly detected failure
                print(f"  Result:  {status}")
            else:
                status = (
                    f"FAIL (expected {'PASS' if expected_pass else 'FAIL'}, got {'PASS' if actual_pass else 'FAIL'})"
                )
                failed += 1
                print(f"  Result:  {status}")

            print(f"  Report:  {output_path.name}")
            print()

        except subprocess.TimeoutExpired:
            print("  Result:  ERROR (timeout)")
            errors += 1
            print()
        except Exception as e:
            print(f"  Result:  ERROR ({e})")
            errors += 1
            print()

    # Summary
    print("=" * 80)
    print("Demo Summary")
    print("=" * 80)
    print(f"Total Scenarios:  {len(DEMO_SCENARIOS)}")
    print(f"Passed:           {passed}")
    print(f"Failed:           {failed}")
    print(f"Errors:           {errors}")
    print()

    if failed == 0 and errors == 0:
        print("SUCCESS: All scenarios completed as expected!")
        print(f"Reports saved to: {output_dir.resolve()}")
        return 0
    else:
        print("WARNING: Some scenarios did not complete as expected.")
        print("Review individual reports for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
