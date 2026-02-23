#!/usr/bin/env python3
"""Convert Nexar annotations to Insurance MVP ground truth format.

Maps Nexar labels to Insurance severity:
- collision → HIGH
- near_collision → MEDIUM
- normal → NONE
"""

import json
from pathlib import Path


def main():
    # Load Nexar metadata
    nexar_dir = Path("data/real_dashcam/nexar")
    metadata_path = nexar_dir / "metadata.json"

    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found")
        print("Run: python scripts/download_nexar.py first")
        return

    with open(metadata_path) as f:
        nexar_data = json.load(f)

    print(f"Loaded {len(nexar_data)} Nexar videos")

    # Convert to Insurance format
    insurance_gt = {}

    label_map = {
        "collision": "HIGH",
        "near_collision": "MEDIUM",
        "normal": "NONE",
    }

    for item in nexar_data:
        video_id = item["video_id"]
        nexar_label = item["label"]
        severity = label_map.get(nexar_label, "LOW")  # fallback

        insurance_gt[video_id] = {
            "video_id": video_id,
            "video_path": item["video_path"],
            "gt_severity": severity,
            "gt_fault_ratio": 100.0 if severity == "HIGH" else (50.0 if severity == "MEDIUM" else 0.0),
            "gt_fraud_risk": 0.0,  # Nexar doesn't have fraud labels
            "nexar_label": nexar_label,
            "collision_timestamp": item.get("collision_timestamp"),
            "weather": item.get("weather"),
            "lighting": item.get("lighting"),
            "scene_type": item.get("scene_type"),
        }

    # Save ground truth
    gt_path = nexar_dir / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(insurance_gt, f, indent=2, ensure_ascii=False)

    # Print distribution
    severity_counts = {}
    for item in insurance_gt.values():
        sev = item["gt_severity"]
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    print(f"\nGround truth created: {gt_path}")
    print(f"\nSeverity distribution:")
    for sev, count in sorted(severity_counts.items()):
        print(f"  {sev}: {count}")

    print(f"\nNext: python scripts/real_data_benchmark.py")


if __name__ == "__main__":
    main()
