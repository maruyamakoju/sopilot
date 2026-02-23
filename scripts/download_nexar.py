#!/usr/bin/env python3
"""Download Nexar collision prediction dataset from HuggingFace.

Downloads a stratified sample of videos using huggingface_hub:
- N positive (collision) videos
- N negative (normal driving) videos

Dataset: https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction
License: MIT

Structure:
  train/positive/*.mp4  — 751 collision videos (~12MB each)
  train/negative/*.mp4  — 751 normal videos (~12MB each)
  solution.csv          — time_to_accident labels

Usage:
  python scripts/download_nexar.py --n-per-class 10 --output data/real_dashcam/nexar
"""

import argparse
import csv
import json
import random
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

REPO_ID = "nexar-ai/nexar_collision_prediction"
REPO_TYPE = "dataset"


def list_videos(api: HfApi, split: str = "train") -> tuple[list[str], list[str]]:
    """List positive and negative video paths in the dataset repo."""
    positive = []
    negative = []
    for item in api.list_repo_tree(REPO_ID, repo_type=REPO_TYPE, path_in_repo=f"{split}/positive"):
        if hasattr(item, "rfilename") and item.rfilename.endswith(".mp4"):
            positive.append(item.rfilename)
        elif hasattr(item, "path") and item.path.endswith(".mp4"):
            positive.append(item.path)

    for item in api.list_repo_tree(REPO_ID, repo_type=REPO_TYPE, path_in_repo=f"{split}/negative"):
        if hasattr(item, "rfilename") and item.rfilename.endswith(".mp4"):
            negative.append(item.rfilename)
        elif hasattr(item, "path") and item.path.endswith(".mp4"):
            negative.append(item.path)

    return sorted(positive), sorted(negative)


def load_solution_csv(api: HfApi) -> dict:
    """Load solution.csv with time_to_accident labels."""
    try:
        path = hf_hub_download(REPO_ID, "solution.csv", repo_type=REPO_TYPE)
        labels = {}
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row.get("id") or row.get("video_id", "")
                tta = row.get("time_to_accident", "")
                labels[video_id] = float(tta) if tta else None
        return labels
    except Exception as e:
        print(f"Warning: Could not load solution.csv: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Download Nexar collision dataset")
    parser.add_argument("--n-per-class", type=int, default=10, help="Videos per class (default: 10)")
    parser.add_argument("--output", type=str, default="data/real_dashcam/nexar", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    api = HfApi()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Listing videos in {REPO_ID}...")
    positive_paths, negative_paths = list_videos(api)
    print(f"  Positive (collision): {len(positive_paths)}")
    print(f"  Negative (normal):    {len(negative_paths)}")

    # Stratified sampling
    random.seed(args.seed)
    sampled_pos = random.sample(positive_paths, min(args.n_per_class, len(positive_paths)))
    sampled_neg = random.sample(negative_paths, min(args.n_per_class, len(negative_paths)))

    print(f"\nDownloading {len(sampled_pos)} positive + {len(sampled_neg)} negative videos...")

    # Load solution labels
    solution = load_solution_csv(api)

    metadata = []

    # Download positive videos
    for vpath in tqdm(sampled_pos, desc="Positive (collision)"):
        video_name = Path(vpath).name
        video_id = Path(vpath).stem
        local_path = output_dir / f"pos_{video_name}"

        try:
            downloaded = hf_hub_download(REPO_ID, vpath, repo_type=REPO_TYPE)
            import shutil
            shutil.copy(downloaded, local_path)
        except Exception as e:
            print(f"\n  Error downloading {vpath}: {e}")
            continue

        metadata.append({
            "video_id": f"pos_{video_id}",
            "video_path": str(local_path),
            "label": "positive",
            "gt_severity": "HIGH",
            "time_to_accident": solution.get(video_id),
            "source": "nexar_collision_prediction",
        })

    # Download negative videos
    for vpath in tqdm(sampled_neg, desc="Negative (normal)"):
        video_name = Path(vpath).name
        video_id = Path(vpath).stem
        local_path = output_dir / f"neg_{video_name}"

        try:
            downloaded = hf_hub_download(REPO_ID, vpath, repo_type=REPO_TYPE)
            import shutil
            shutil.copy(downloaded, local_path)
        except Exception as e:
            print(f"\n  Error downloading {vpath}: {e}")
            continue

        metadata.append({
            "video_id": f"neg_{video_id}",
            "video_path": str(local_path),
            "label": "negative",
            "gt_severity": "NONE",
            "time_to_accident": None,
            "source": "nexar_collision_prediction",
        })

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    # Also create ground_truth.json for the benchmark script
    ground_truth = {}
    for m in metadata:
        ground_truth[m["video_id"]] = {
            "gt_severity": m["gt_severity"],
            "gt_fault_ratio": 70.0 if m["label"] == "positive" else 0.0,
            "label": m["label"],
        }
    gt_path = output_dir / "ground_truth.json"
    gt_path.write_text(json.dumps(ground_truth, indent=2, ensure_ascii=False))

    print(f"\nDownload complete!")
    print(f"  Videos:       {output_dir} ({len(metadata)} files)")
    print(f"  Metadata:     {metadata_path}")
    print(f"  Ground truth: {gt_path}")
    print(f"\nNext: python scripts/real_data_benchmark.py --input {output_dir} --backend mock")


if __name__ == "__main__":
    main()
