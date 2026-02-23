#!/usr/bin/env python3
"""Download Nexar collision prediction dataset from HuggingFace.

Downloads a stratified sample of 60 videos:
- 20 collision
- 20 near_collision
- 20 normal

Dataset: https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction
License: MIT
"""

import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main():
    print("Loading Nexar collision prediction dataset from HuggingFace...")

    # Load dataset
    dataset = load_dataset("nexar-ai/nexar_collision_prediction", split="train")

    print(f"Total samples: {len(dataset)}")
    print(f"Features: {dataset.features}")

    # Group by label
    collision = []
    near_collision = []
    normal = []

    for idx, sample in enumerate(dataset):
        label = sample.get("label") or sample.get("collision_type")
        if label == "collision":
            collision.append((idx, sample))
        elif label == "near_collision":
            near_collision.append((idx, sample))
        elif label == "normal":
            normal.append((idx, sample))

    print(f"\nDistribution:")
    print(f"  collision: {len(collision)}")
    print(f"  near_collision: {len(near_collision)}")
    print(f"  normal: {len(normal)}")

    # Stratified sampling
    random.seed(42)
    sample_collision = random.sample(collision, min(20, len(collision)))
    sample_near = random.sample(near_collision, min(20, len(near_collision)))
    sample_normal = random.sample(normal, min(20, len(normal)))

    sampled = sample_collision + sample_near + sample_normal
    print(f"\nSampled {len(sampled)} videos (20 per class)")

    # Create output directory
    output_dir = Path("data/real_dashcam/nexar")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download videos and save metadata
    metadata = []

    for idx, sample in tqdm(sampled, desc="Downloading videos"):
        video_id = sample.get("video_id") or f"nexar_{idx:04d}"
        label = sample.get("label") or sample.get("collision_type")

        # Save video
        video_path = output_dir / f"{video_id}.mp4"

        # HuggingFace datasets stores video as bytes or path
        video_data = sample.get("video")
        if video_data:
            if isinstance(video_data, bytes):
                video_path.write_bytes(video_data)
            elif isinstance(video_data, dict) and "path" in video_data:
                # If video is stored as file reference
                import shutil
                shutil.copy(video_data["path"], video_path)

        # Metadata
        metadata.append({
            "video_id": video_id,
            "video_path": str(video_path),
            "label": label,
            "duration": sample.get("duration"),
            "collision_timestamp": sample.get("collision_timestamp"),
            "weather": sample.get("weather"),
            "lighting": sample.get("lighting"),
            "scene_type": sample.get("scene_type"),
        })

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    print(f"\nDownload complete!")
    print(f"  Videos: {output_dir}")
    print(f"  Metadata: {metadata_path}")
    print(f"\nNext: python scripts/nexar_to_insurance_format.py")


if __name__ == "__main__":
    main()
