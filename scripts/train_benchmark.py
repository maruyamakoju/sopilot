#!/usr/bin/env python
"""SOPilot neural pipeline benchmark & training script.

Generates synthetic SOP-like data, runs the full SOPilotTrainer pipeline
(4 phases), evaluates on held-out pairs, and prints a comprehensive
summary table.

Usage:
    python scripts/train_benchmark.py
    python scripts/train_benchmark.py --device cuda --epochs-multiplier 2
    python scripts/train_benchmark.py --data-dir data/models/neural
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as _nn

# ---------------------------------------------------------------------------
# Ensure sopilot package is importable when running from repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from sopilot.nn.trainer import (  # noqa: E402
    SOPilotTrainer,
    TrainingConfig,
)
from sopilot.step_engine import evaluate_sop  # noqa: E402

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_benchmark")

# Maximum clips per sequence for joint finetuning (Phase 3).
# The dict-based Soft-DTW DP in SoftDTWAlignment has O(M*N) Python loops,
# so we cap sequence length to keep Phase 3 tractable.
# On CUDA, each Python-level tensor op incurs kernel launch overhead,
# so we keep sequences short. 30 clips = 900 DP cells, ~5s per pair.
_JOINT_MAX_CLIPS = 30


# ===================================================================
# 1.  Synthetic data generation
# ===================================================================


def _random_boundaries(n_clips: int, n_steps: int) -> list[int]:
    """Generate sorted random step boundaries [0, b1, ..., n_clips]."""
    inner = sorted(np.random.choice(range(2, n_clips - 1), size=n_steps - 1, replace=False).tolist())
    return [0] + inner + [n_clips]


def _temporal_warp(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random temporal warping: stretch/compress segments."""
    n = seq.shape[0]
    if n < 4:
        return seq.copy()
    warp_factor = rng.uniform(0.7, 1.3, size=n)
    cumulative = np.cumsum(warp_factor)
    cumulative = cumulative / cumulative[-1] * (n - 1)
    indices = np.clip(np.round(cumulative).astype(int), 0, n - 1)
    return seq[indices]


def generate_synthetic_data(
    n_gold: int = 20,
    n_trainee: int = 20,
    dim: int = 128,
    seed: int = 42,
) -> dict:
    """Generate synthetic SOP-like embedding data.

    Returns a dict with keys:
        gold_embeddings:    list of (N_i, D) arrays
        gold_boundaries:    list of boundary lists
        trainee_embeddings: list of (N_j, D) arrays
        trainee_boundaries: list of boundary lists
        target_scores:      (n_trainee,) array in [0, 100]
        gold_meta:          list of metadata dicts per gold video
        trainee_meta:       list of metadata dicts per trainee video
    """
    rng = np.random.default_rng(seed)
    logger.info(
        "Generating synthetic data: %d gold, %d trainee, dim=%d",
        n_gold,
        n_trainee,
        dim,
    )

    gold_embeddings: list[np.ndarray] = []
    gold_boundaries: list[list[int]] = []
    gold_meta_all: list[list[dict]] = []

    for _ in range(n_gold):
        n_clips = int(rng.integers(100, 301))
        n_steps = int(rng.integers(5, 16))
        n_steps = min(n_steps, n_clips - 2)

        bounds = _random_boundaries(n_clips, n_steps)
        embs = np.zeros((n_clips, dim), dtype=np.float32)
        for s in range(len(bounds) - 1):
            centroid = rng.standard_normal(dim).astype(np.float32)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
            start, end = bounds[s], bounds[s + 1]
            noise = rng.standard_normal((end - start, dim)).astype(np.float32) * 0.15
            embs[start:end] = centroid + noise

        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.maximum(norms, 1e-8)

        gold_embeddings.append(embs)
        gold_boundaries.append(bounds)
        gold_meta_all.append([{"start_sec": float(c * 2.0), "end_sec": float((c + 1) * 2.0)} for c in range(n_clips)])

    trainee_embeddings: list[np.ndarray] = []
    trainee_boundaries: list[list[int]] = []
    trainee_meta_all: list[list[dict]] = []
    target_scores = np.zeros(n_trainee, dtype=np.float32)

    for i in range(n_trainee):
        gold_idx = i % n_gold
        gold_emb = gold_embeddings[gold_idx]

        noise_level = rng.uniform(0.05, 0.5)
        noisy = gold_emb + rng.standard_normal(gold_emb.shape).astype(np.float32) * noise_level
        warped = _temporal_warp(noisy, rng)

        norms = np.linalg.norm(warped, axis=1, keepdims=True)
        warped = warped / np.maximum(norms, 1e-8)

        trainee_embeddings.append(warped)

        n_clips_t = warped.shape[0]
        n_steps_t = int(rng.integers(5, 16))
        n_steps_t = min(n_steps_t, n_clips_t - 2)
        trainee_boundaries.append(_random_boundaries(n_clips_t, n_steps_t))

        trainee_meta_all.append(
            [{"start_sec": float(c * 2.0), "end_sec": float((c + 1) * 2.0)} for c in range(n_clips_t)]
        )

        # Target score via mean cosine similarity
        min_len = min(gold_emb.shape[0], warped.shape[0])
        cos_sims = np.sum(gold_emb[:min_len] * warped[:min_len], axis=1)
        score = float(np.clip(np.mean(cos_sims) * 100.0, 0.0, 100.0))
        target_scores[i] = score

    logger.info(
        "Data generated: gold clip counts %d-%d, trainee clip counts %d-%d",
        min(e.shape[0] for e in gold_embeddings),
        max(e.shape[0] for e in gold_embeddings),
        min(e.shape[0] for e in trainee_embeddings),
        max(e.shape[0] for e in trainee_embeddings),
    )
    logger.info(
        "Target scores: mean=%.1f, std=%.1f, min=%.1f, max=%.1f",
        float(np.mean(target_scores)),
        float(np.std(target_scores)),
        float(np.min(target_scores)),
        float(np.max(target_scores)),
    )

    return {
        "gold_embeddings": gold_embeddings,
        "gold_boundaries": gold_boundaries,
        "trainee_embeddings": trainee_embeddings,
        "trainee_boundaries": trainee_boundaries,
        "target_scores": target_scores,
        "gold_meta": gold_meta_all,
        "trainee_meta": trainee_meta_all,
    }


# ===================================================================
# 2.  Generate synthetic 15-metric feature rows for ScoringHead
# ===================================================================


def generate_scoring_data(
    gold_embeddings: list[np.ndarray],
    trainee_embeddings: list[np.ndarray],
    target_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 15-metric feature vectors for each gold/trainee pair.

    Synthesises plausible metric values from the embeddings directly
    rather than running the full evaluate_sop pipeline.
    """
    n = len(trainee_embeddings)
    metrics = np.zeros((n, 15), dtype=np.float32)

    for i in range(n):
        gold_idx = i % len(gold_embeddings)
        g = gold_embeddings[gold_idx]
        t = trainee_embeddings[i]

        min_len = min(g.shape[0], t.shape[0])
        cos = np.sum(g[:min_len] * t[:min_len], axis=1)
        mean_sim = float(np.mean(cos))
        std_sim = float(np.std(cos))

        metrics[i, 0] = max(0.0, (1.0 - mean_sim) * 5.0)  # miss
        metrics[i, 1] = max(0.0, std_sim * 3.0)  # swap
        metrics[i, 2] = max(0.0, 1.0 - mean_sim)  # deviation
        metrics[i, 3] = abs(g.shape[0] - t.shape[0]) / max(g.shape[0], 1)  # over_time
        metrics[i, 4] = std_sim * 0.5  # temporal_warp
        metrics[i, 5] = abs(g.shape[0] - t.shape[0]) / max(g.shape[0], t.shape[0], 1)
        metrics[i, 6] = max(0.0, 1.0 - mean_sim) * 0.3  # duplicate_ratio
        metrics[i, 7] = std_sim * 0.2  # order_violation_ratio
        metrics[i, 8] = std_sim * 0.3  # temporal_drift
        metrics[i, 9] = max(0.0, 1.0 - mean_sim)  # confidence_loss
        metrics[i, 10] = float(np.max(cos) - np.min(cos)) if min_len > 0 else 0.0
        metrics[i, 11] = max(0.3, float(np.median(cos) - 0.01))
        metrics[i, 12] = max(0.3, float(np.median(cos) - 0.01))
        metrics[i, 13] = float(np.mean(cos < 0.3))  # hard_miss_ratio
        metrics[i, 14] = max(0.0, 1.0 - mean_sim) * 0.5  # mean_alignment_cost

    return metrics, target_scores


def _subsample_uniform(arr: np.ndarray, max_len: int) -> np.ndarray:
    """Uniformly subsample a (T, D) array to at most max_len rows."""
    if arr.shape[0] <= max_len:
        return arr
    idx = np.linspace(0, arr.shape[0] - 1, max_len, dtype=int)
    return arr[idx]


# ===================================================================
# 3.  Main benchmark runner
# ===================================================================


def _detect_device(requested: str) -> str:
    """Resolve device string, falling back to CPU if CUDA unavailable."""
    if requested == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("CUDA available: %s", torch.cuda.get_device_name(0))
        else:
            device = "cpu"
            logger.info("CUDA not available, using CPU")
    elif requested.startswith("cuda"):
        if torch.cuda.is_available():
            device = requested
            idx = int(requested.split(":")[-1]) if ":" in requested else 0
            logger.info("Using CUDA device %d: %s", idx, torch.cuda.get_device_name(idx))
        else:
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
    else:
        device = "cpu"
        logger.info("Using CPU")
    return device


def _set_bn_eval(model: _nn.Module) -> None:
    """Set all BatchNorm1d layers to eval mode (use running stats).

    This is required for modules that process batch_size=1 tensors
    during training, since BatchNorm1d raises an error with a single
    sample when in training mode.
    """
    for m in model.modules():
        if isinstance(m, _nn.BatchNorm1d):
            m.eval()


def run_benchmark(args: argparse.Namespace) -> None:
    """Run the full benchmark pipeline."""
    device = _detect_device(args.device)
    em = args.epochs_multiplier
    output_dir = Path(args.data_dir)

    # Phase epoch counts (scaled by multiplier)
    proj_epochs = max(1, int(30 * em))
    seg_epochs = max(1, int(30 * em))
    asformer_epochs = max(1, int(30 * em))
    score_epochs = max(1, int(50 * em))
    joint_epochs = max(1, int(10 * em))

    logger.info("=" * 70)
    logger.info("SOPilot Neural Pipeline Benchmark")
    logger.info("=" * 70)
    logger.info("Device:            %s", device)
    logger.info("Epochs multiplier: %.2f", em)
    logger.info("Output directory:  %s", output_dir)
    logger.info(
        "Phase epochs:      proj=%d  seg=%d  asformer=%d  score=%d  joint=%d",
        proj_epochs,
        seg_epochs,
        asformer_epochs,
        score_epochs,
        joint_epochs,
    )
    logger.info("-" * 70)

    # ------------------------------------------------------------------
    # Step 1: Generate synthetic data
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    data = generate_synthetic_data(n_gold=20, n_trainee=20, dim=128, seed=42)
    data_time = time.perf_counter() - t0
    logger.info("Data generation: %.2fs", data_time)

    gold_embs = data["gold_embeddings"]
    gold_bounds = data["gold_boundaries"]
    trainee_embs = data["trainee_embeddings"]
    target_scores = data["target_scores"]

    # ------------------------------------------------------------------
    # Step 2: Configure and run SOPilotTrainer
    # ------------------------------------------------------------------
    config = TrainingConfig(
        device=device,
        proj_d_in=128,
        proj_d_out=128,
        proj_lr=1e-3,
        proj_epochs=proj_epochs,
        proj_batch_size=256,
        proj_temperature=0.07,
        seg_lr=1e-3,
        seg_epochs=seg_epochs,
        seg_batch_size=16,
        asformer_d_model=64,
        asformer_n_heads=4,
        asformer_n_encoder_layers=10,
        asformer_n_decoder_layers=10,
        asformer_n_decoders=3,
        asformer_lr=1e-3,
        asformer_epochs=asformer_epochs,
        score_lr=1e-3,
        score_epochs=score_epochs,
        score_batch_size=64,
        joint_lr=1e-4,
        joint_epochs=joint_epochs,
        gamma_init=1.0,
        dilate_alpha=0.5,
        conformal_alpha=0.1,
        output_dir=output_dir,
    )

    trainer = SOPilotTrainer(config)
    phase_timings: list[dict] = []

    # -------------------- Phase 1a: ProjectionHead --------------------
    logger.info("")
    logger.info("=" * 50)
    logger.info("PHASE 1a: ProjectionHead (contrastive)")
    logger.info("=" * 50)
    t_start = time.perf_counter()
    log_proj = trainer.train_projection_head(gold_embs, gold_bounds)
    t_proj = time.perf_counter() - t_start
    phase_timings.append(
        {
            "phase": "1a: ProjectionHead",
            "epochs": log_proj.epochs_completed,
            "final_loss": log_proj.final_loss,
            "num_params": log_proj.num_parameters,
            "wall_time": t_proj,
        }
    )
    logger.info("Phase 1a finished in %.2fs", t_proj)

    # -------------------- Phase 1b: StepSegmenter --------------------
    logger.info("")
    logger.info("=" * 50)
    logger.info("PHASE 1b: StepSegmenter (MS-TCN++)")
    logger.info("=" * 50)
    t_start = time.perf_counter()
    log_seg = trainer.train_step_segmenter(gold_embs, gold_bounds)
    t_seg = time.perf_counter() - t_start
    phase_timings.append(
        {
            "phase": "1b: StepSegmenter",
            "epochs": log_seg.epochs_completed,
            "final_loss": log_seg.final_loss,
            "num_params": log_seg.num_parameters,
            "wall_time": t_seg,
        }
    )
    logger.info("Phase 1b finished in %.2fs", t_seg)

    # -------------------- Phase 1c: ASFormer --------------------
    logger.info("")
    logger.info("=" * 50)
    logger.info("PHASE 1c: ASFormer (transformer segmenter)")
    logger.info("=" * 50)
    t_start = time.perf_counter()
    log_asf = trainer.train_asformer(gold_embs, gold_bounds)
    t_asf = time.perf_counter() - t_start
    phase_timings.append(
        {
            "phase": "1c: ASFormer",
            "epochs": log_asf.epochs_completed,
            "final_loss": log_asf.final_loss,
            "num_params": log_asf.num_parameters,
            "wall_time": t_asf,
        }
    )
    logger.info("Phase 1c finished in %.2fs", t_asf)

    # -------------------- Phase 2: ScoringHead --------------------
    logger.info("")
    logger.info("=" * 50)
    logger.info("PHASE 2: ScoringHead (warm-start)")
    logger.info("=" * 50)
    metrics_array, scores_array = generate_scoring_data(
        gold_embs,
        trainee_embs,
        target_scores,
    )
    t_start = time.perf_counter()
    log_score = trainer.train_scoring_head(metrics_array, scores_array)
    t_score = time.perf_counter() - t_start
    phase_timings.append(
        {
            "phase": "2:  ScoringHead",
            "epochs": log_score.epochs_completed,
            "final_loss": log_score.final_loss,
            "num_params": log_score.num_parameters,
            "wall_time": t_score,
        }
    )
    logger.info("Phase 2 finished in %.2fs", t_score)

    # -------------------- Phase 3: Joint fine-tune --------------------
    logger.info("")
    logger.info("=" * 50)
    logger.info("PHASE 3: Joint fine-tune (end-to-end)")
    logger.info("=" * 50)

    # Subsample sequences for joint finetuning: the dict-based Soft-DTW DP
    # has O(M*N) Python-level iterations, so we cap each sequence at
    # _JOINT_MAX_CLIPS to keep training time reasonable.
    joint_gold = [_subsample_uniform(gold_embs[i % len(gold_embs)], _JOINT_MAX_CLIPS) for i in range(len(trainee_embs))]
    joint_trainee = [_subsample_uniform(trainee_embs[i], _JOINT_MAX_CLIPS) for i in range(len(trainee_embs))]
    logger.info(
        "Joint finetune: %d pairs, sequences capped at %d clips",
        len(joint_gold),
        _JOINT_MAX_CLIPS,
    )

    # Phase 3 freezes the scoring head internally and sets BN to eval mode
    # in projection_head for batch_size=1 compatibility.
    _set_bn_eval(trainer.projection_head)

    t_start = time.perf_counter()
    log_joint = trainer.joint_finetune(joint_gold, joint_trainee, target_scores)
    t_joint = time.perf_counter() - t_start
    phase_timings.append(
        {
            "phase": "3:  Joint Finetune",
            "epochs": log_joint.epochs_completed,
            "final_loss": log_joint.final_loss,
            "num_params": log_joint.num_parameters,
            "wall_time": t_joint,
        }
    )
    logger.info("Phase 3 finished in %.2fs", t_joint)

    # -------------------- Phase 4: Calibration --------------------
    logger.info("")
    logger.info("=" * 50)
    logger.info("PHASE 4: Isotonic + Conformal calibration")
    logger.info("=" * 50)

    # Phase 3 freezes the scoring head, so its weights are intact from Phase 2.
    # Just run eval-mode inference to get calibration predictions.
    trainer.scoring_head.eval()
    cal_metrics_t = torch.from_numpy(metrics_array.astype(np.float32)).to(device)
    with torch.no_grad():
        cal_preds = trainer.scoring_head(cal_metrics_t).cpu().numpy().flatten()

    t_start = time.perf_counter()
    trainer.calibrate(cal_preds, scores_array)
    t_cal = time.perf_counter() - t_start
    phase_timings.append(
        {
            "phase": "4:  Calibration",
            "epochs": 0,
            "final_loss": 0.0,
            "num_params": 0,
            "wall_time": t_cal,
        }
    )
    logger.info("Phase 4 finished in %.2fs", t_cal)

    # ------------------------------------------------------------------
    # Step 3: Save models
    # ------------------------------------------------------------------
    # Refresh BatchNorm running statistics in projection_head before saving.
    # Phase 3 set BN to eval mode while weights were updated, so stats may be stale.
    trainer.projection_head.train()
    with torch.no_grad():
        concat = np.concatenate(gold_embs[:5], axis=0)
        real_proj = torch.from_numpy(concat.astype(np.float32)).to(device)
        _ = trainer.projection_head(real_proj)
    trainer.projection_head.eval()

    logger.info("")
    logger.info("Saving models to %s ...", output_dir)
    saved_paths = trainer.save_all(output_dir)
    for name, path in saved_paths.items():
        logger.info("  %s -> %s", name, path)

    # ------------------------------------------------------------------
    # Step 4: Held-out evaluation using evaluate_sop in neural_mode
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 50)
    logger.info("HELD-OUT EVALUATION (neural_mode=True)")
    logger.info("=" * 50)

    # Clear step_engine neural caches so it picks up freshly saved models
    from sopilot.step_engine import invalidate_neural_caches

    invalidate_neural_caches()

    eval_scores: list[float] = []
    n_eval = len(trainee_embs)
    t_eval_start = time.perf_counter()

    for i in range(n_eval):
        gold_idx = i % len(gold_embs)
        g_emb = gold_embs[gold_idx]
        t_emb = trainee_embs[i]
        g_meta = data["gold_meta"][gold_idx]
        t_meta = data["trainee_meta"][i]

        result = evaluate_sop(
            gold_embeddings=g_emb,
            trainee_embeddings=t_emb,
            gold_meta=g_meta,
            trainee_meta=t_meta,
            threshold_factor=1.0,
            min_step_clips=2,
            low_similarity_threshold=0.3,
            w_miss=15.0,
            w_swap=10.0,
            w_dev=8.0,
            w_time=5.0,
            w_warp=12.0,
            use_gpu_dtw=False,
            neural_mode=True,
            neural_model_dir=output_dir,
            neural_device=device,
            neural_soft_dtw_gamma=1.0,
            neural_uncertainty_samples=30,
            neural_calibration_enabled=True,
            neural_cuda_dtw=False,
            neural_ot_alignment=False,
            neural_conformal_alpha=0.1,
        )

        heuristic_score = result["score"]
        neural_info = result.get("neural_score", {})
        neural_score = neural_info.get("score", heuristic_score)

        # Guard against NaN from MC Dropout / BN issues
        if neural_score is None or (isinstance(neural_score, float) and math.isnan(neural_score)):
            neural_score = heuristic_score

        eval_scores.append(neural_score)

        if (i + 1) % 5 == 0 or i == 0:
            uncertainty = neural_info.get("uncertainty", 0.0)
            cal_score = neural_info.get("calibrated_score")
            cal_str = f"{cal_score:.1f}" if cal_score is not None else "N/A"
            logger.info(
                "  Pair %2d/%d: heuristic=%.1f  neural=%.1f +/- %.1f  calibrated=%s",
                i + 1,
                n_eval,
                heuristic_score,
                neural_score,
                uncertainty,
                cal_str,
            )

    t_eval = time.perf_counter() - t_eval_start
    eval_arr = np.array(eval_scores, dtype=np.float64)

    # Replace any remaining NaN for summary stats
    eval_arr = np.where(np.isnan(eval_arr), 0.0, eval_arr)

    # ------------------------------------------------------------------
    # Step 5: Print comprehensive summary table
    # ------------------------------------------------------------------
    total_params = sum(p["num_params"] for p in phase_timings)
    total_time = sum(p["wall_time"] for p in phase_timings)

    print()
    print("=" * 78)
    print("  SOPilot Neural Pipeline -- Benchmark Summary")
    print("=" * 78)
    print()
    print(f"  Device:            {device}")
    print(f"  Epochs multiplier: {em:.2f}")
    print(f"  Output directory:  {output_dir}")
    print()
    print("-" * 78)
    print(f"  {'Phase':<24s}  {'Epochs':>6s}  {'Final Loss':>11s}  {'Parameters':>11s}  {'Wall Time':>10s}")
    print("-" * 78)

    for p in phase_timings:
        epochs_str = str(p["epochs"]) if p["epochs"] > 0 else "-"
        loss_str = f"{p['final_loss']:.6f}" if p["final_loss"] > 0 else "-"
        params_str = f"{p['num_params']:,}" if p["num_params"] > 0 else "-"
        time_str = f"{p['wall_time']:.2f}s"
        print(f"  {p['phase']:<24s}  {epochs_str:>6s}  {loss_str:>11s}  {params_str:>11s}  {time_str:>10s}")

    print("-" * 78)
    print(f"  {'TOTAL':<24s}  {'':>6s}  {'':>11s}  {total_params:>11,}  {total_time:>9.2f}s")
    print("-" * 78)
    print()
    print("  Held-out Evaluation (neural_mode)")
    print("  " + "-" * 40)
    print(f"    Pairs evaluated:  {n_eval}")
    print(f"    Evaluation time:  {t_eval:.2f}s")
    print(f"    Mean score:       {float(np.mean(eval_arr)):.2f}")
    print(f"    Std  score:       {float(np.std(eval_arr)):.2f}")
    print(f"    Min  score:       {float(np.min(eval_arr)):.2f}")
    print(f"    Max  score:       {float(np.max(eval_arr)):.2f}")
    print(f"    Median score:     {float(np.median(eval_arr)):.2f}")
    print()

    # Model file sizes
    print("  Saved Model Files")
    print("  " + "-" * 40)
    for name, fpath in saved_paths.items():
        p = Path(fpath)
        if p.exists():
            size_kb = p.stat().st_size / 1024
            print(f"    {name:<25s}  {size_kb:>8.1f} KB")
    print()
    print("=" * 78)
    print("  Benchmark complete.")
    print("=" * 78)
    print()


# ===================================================================
# CLI entry point
# ===================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SOPilot neural pipeline benchmark & training script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/train_benchmark.py
  python scripts/train_benchmark.py --device cuda
  python scripts/train_benchmark.py --epochs-multiplier 0.5 --data-dir /tmp/models
""",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(_REPO_ROOT / "data" / "models" / "neural"),
        help="Directory to save trained neural models (default: data/models/neural/)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"],
        help="Compute device (default: auto-detect CUDA)",
    )
    parser.add_argument(
        "--epochs-multiplier",
        type=float,
        default=1.0,
        help="Scale all epoch counts by this factor (default: 1.0). Use 0.5 for quick test, 2.0 for longer training.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_benchmark(args)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(1)
    except Exception:
        logger.exception("Benchmark failed")
        sys.exit(2)
