#!/usr/bin/env python3
"""Convert Nexar annotations to Insurance MVP ground truth format.

Maps Nexar labels to Insurance severity:
- positive (collision) → HIGH
- negative (normal)    → NONE

The download script already creates ground_truth.json, but this script
can enrich it with additional fields or re-map labels.

Usage:
  python scripts/nexar_to_insurance_format.py [--input data/real_dashcam/nexar]
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert Nexar to insurance format")
    parser.add_argument("--input", type=str, default="data/real_dashcam/nexar")
    args = parser.parse_args()

    nexar_dir = Path(args.input)
    metadata_path = nexar_dir / "metadata.json"

    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found")
        print("Run: python scripts/download_nexar.py first")
        return

    with open(metadata_path) as f:
        nexar_data = json.load(f)

    print(f"Loaded {len(nexar_data)} Nexar videos")

    # Map labels to severity
    label_map = {
        "positive": "HIGH",
        "negative": "NONE",
        # Legacy format support
        "collision": "HIGH",
        "near_collision": "MEDIUM",
        "normal": "NONE",
    }

    fault_map = {
        "HIGH": 70.0,
        "MEDIUM": 50.0,
        "LOW": 20.0,
        "NONE": 0.0,
    }

    # Build ground truth
    ground_truth = {}

    for item in nexar_data:
        video_id = item["video_id"]
        nexar_label = item["label"]
        severity = item.get("gt_severity") or label_map.get(nexar_label, "LOW")

        ground_truth[video_id] = {
            "video_id": video_id,
            "video_path": item["video_path"],
            "gt_severity": severity,
            "gt_fault_ratio": fault_map.get(severity, 0.0),
            "gt_fraud_risk": 0.0,
            "nexar_label": nexar_label,
            "time_to_accident": item.get("time_to_accident"),
            "source": "nexar_collision_prediction",
        }

    # Save ground truth
    gt_path = nexar_dir / "ground_truth.json"
    gt_path.write_text(json.dumps(ground_truth, indent=2, ensure_ascii=False))

    # Print distribution
    severity_counts = {}
    for item in ground_truth.values():
        sev = item["gt_severity"]
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    print(f"\nGround truth: {gt_path}")
    print(f"\nSeverity distribution:")
    for sev in ["NONE", "LOW", "MEDIUM", "HIGH"]:
        if sev in severity_counts:
            print(f"  {sev}: {severity_counts[sev]}")

    print(f"\nNext: python scripts/real_data_benchmark.py --input {nexar_dir} --backend mock")


if __name__ == "__main__":
    main()
