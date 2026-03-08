#!/usr/bin/env python
"""
Phase 45 (Unsupervised Local Adaptation): Nightly Clustering Job.

This script runs during idle time to learn the "normal" state of the facility.
It clusters accumulated DINOv2 features and updates the anomaly detection baseline.
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_clustering_job(features_dir: Path, output_dir: Path) -> None:
    """Load extracted features, cluster them, and save prototypes."""
    logger.info(f"Starting unsupervised local adaptation job.")
    
    # 1. Load accumulated features
    features = []
    if not features_dir.exists():
        logger.warning(f"Features directory {features_dir} not found. Skipping.")
        return

    # In a real scenario, this would load from SQLite or VideoMemory
    # For now we simulate loading recent features
    for npy_file in features_dir.glob("*.npy"):
        try:
            arr = np.load(npy_file)
            if arr.ndim == 1:
                features.append(arr)
            elif arr.ndim == 2:
                features.extend(list(arr))
        except Exception as e:
            logger.error(f"Failed to load {npy_file}: {e}")

    if len(features) < 10:
        logger.info("Not enough features collected for clustering (< 10).")
        return

    X = np.stack(features)
    logger.info(f"Loaded {X.shape[0]} feature vectors of dimension {X.shape[1]}.")

    # 2. Cluster features to find normal prototypes
    # We use K-Means as a simple, effective baseline for prototyping
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        logger.error("scikit-learn is required for clustering. Run: pip install scikit-learn")
        return

    n_clusters = min(50, len(X) // 5)
    logger.info(f"Running K-Means clustering with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    kmeans.fit(X)

    prototypes = kmeans.cluster_centers_
    
    # 3. Save the learned prototypes
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = output_dir / "anomaly_baseline_prototypes.npy"
    np.save(baseline_path, prototypes)

    metadata = {
        "n_samples": len(X),
        "n_clusters": n_clusters,
        "feature_dim": X.shape[1],
        "job_status": "success"
    }
    with open(output_dir / "baseline_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Successfully saved {n_clusters} prototypes to {baseline_path}")

if __name__ == "__main__":
    base_dir = Path("data/active_learning")
    features_dir = base_dir / "features"
    output_dir = base_dir / "baseline"
    
    # Simulate some dummy features for the first run if none exist
    if not features_dir.exists():
        logger.info("Creating dummy features for initial adaptation...")
        features_dir.mkdir(parents=True, exist_ok=True)
        np.save(features_dir / "dummy.npy", np.random.randn(100, 768))

    run_clustering_job(features_dir, output_dir)
