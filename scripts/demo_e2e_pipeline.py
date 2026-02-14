#!/usr/bin/env python
"""SOPilot Neural Pipeline — End-to-End Demo.

Runs the complete scoring pipeline on synthetic data and produces a
multi-panel figure showing every stage from raw embeddings through
to calibrated score with uncertainty.

Pipeline stages:
    1. Embeddings → Step boundary detection (heuristic)
    2. Gold vs Trainee → Temporal alignment (Soft-DTW)
    3. Alignment → 15 penalty metrics extraction
    4. Metrics → ScoringHead MLP → raw score
    5. MC Dropout → epistemic uncertainty
    6. Conformal Prediction → coverage-guaranteed intervals
    7. Integrated Gradients → metric attribution

Usage:
    python scripts/demo_e2e_pipeline.py [--out-dir demo_outputs]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Synthetic data: realistic SOP scenario
# ---------------------------------------------------------------------------

def _generate_realistic_scenario(
    n_steps: int = 5,
    clips_per_step: int = 8,
    embed_dim: int = 64,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    """Generate gold + trainee with known deviations and metadata.

    Trainee deviations:
    - Step 0: slow execution (10 clips instead of 8)
    - Steps 1-2: swapped order
    - Step 3: normal
    - Step 4: skipped

    Returns:
        (gold_emb, trainee_emb, gold_meta, trainee_meta)
    """
    rng = np.random.RandomState(seed)
    centroids = rng.randn(n_steps, embed_dim).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    def _clips(step_id: int, n: int, noise: float = 0.1) -> list[np.ndarray]:
        clips = []
        for _ in range(n):
            c = centroids[step_id] + rng.randn(embed_dim).astype(np.float32) * noise
            clips.append(c / np.linalg.norm(c))
        return clips

    # Gold: 5 steps × 8 clips = 40 clips
    gold_clips = []
    gold_meta = []
    t = 0.0
    for s in range(n_steps):
        for c in _clips(s, clips_per_step):
            gold_clips.append(c)
            gold_meta.append({"start_sec": t, "end_sec": t + 1.0})
            t += 1.0
    gold = np.stack(gold_clips)

    # Trainee: deviations
    trainee_clips = []
    trainee_meta = []
    t = 0.0
    trainee_plan = [
        (0, 10, 0.15),  # step 0: slow, noisier
        (2, 8, 0.1),    # step 2 before step 1 (swap)
        (1, 8, 0.1),    # step 1 after step 2
        (3, 8, 0.1),    # step 3: normal
        # step 4: skipped
    ]
    for step_id, n, noise in trainee_plan:
        for c in _clips(step_id, n, noise):
            trainee_clips.append(c)
            trainee_meta.append({"start_sec": t, "end_sec": t + 1.0})
            t += 1.0
    trainee = np.stack(trainee_clips)

    return gold, trainee, gold_meta, trainee_meta


# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    gold: np.ndarray,
    trainee: np.ndarray,
    gold_meta: list[dict],
    trainee_meta: list[dict],
) -> dict:
    """Run evaluate_sop and compute all neural components."""
    from sopilot.step_engine import evaluate_sop

    result = evaluate_sop(
        gold_embeddings=gold,
        trainee_embeddings=trainee,
        gold_meta=gold_meta,
        trainee_meta=trainee_meta,
        threshold_factor=0.8,
        min_step_clips=3,
        low_similarity_threshold=0.6,
        w_miss=15.0,
        w_swap=10.0,
        w_dev=8.0,
        w_time=5.0,
        w_warp=12.0,
        neural_mode=False,  # We'll do neural scoring manually for visualization
    )
    return result


def run_neural_scoring(metrics_dict: dict) -> dict:
    """Run ScoringHead + MC Dropout + Conformal separately for visualization."""
    from sopilot.nn.conformal import SplitConformalPredictor
    from sopilot.nn.explainability import CounterfactualExplainer
    from sopilot.nn.scoring_head import N_METRICS, ScoringHead

    torch.manual_seed(42)
    rng = np.random.RandomState(42)

    model = ScoringHead(n_inputs=N_METRICS)
    model.eval()

    # Prepare metrics tensor
    from sopilot.nn.scoring_head import METRIC_KEYS
    metrics_vals = [float(metrics_dict.get(k, 0.0)) for k in METRIC_KEYS]
    metrics_tensor = torch.tensor([metrics_vals], dtype=torch.float32)

    # Forward pass
    with torch.no_grad():
        raw_score = float(model._forward_single(metrics_tensor))

    # MC Dropout uncertainty
    uncertainty = model.predict_with_uncertainty(metrics_tensor, n_samples=100)

    # Collect individual MC samples for visualization
    mc_samples = []
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()
    with torch.no_grad():
        for _ in range(100):
            s = model._forward_single(metrics_tensor)
            mc_samples.append(s)
    model.eval()

    # Conformal prediction
    n_cal = 80
    cal_x = torch.randn(n_cal, N_METRICS) * 0.3
    cal_preds = np.array([model._forward_single(cal_x[i:i + 1]) for i in range(n_cal)])
    cal_actuals = np.clip(cal_preds + rng.randn(n_cal) * 8, 0, 100)

    conformal = SplitConformalPredictor(alpha=0.1)
    conformal.calibrate(cal_preds, cal_actuals)
    _, conf_lo, conf_hi = conformal.predict(uncertainty["score"])

    # Sensitivity analysis
    explainer = CounterfactualExplainer(model)
    sensitivity = explainer.compute_sensitivity(metrics_tensor)

    return {
        "raw_score": raw_score,
        "mc_samples": mc_samples,
        "uncertainty": uncertainty,
        "conformal": {"lower": conf_lo, "upper": conf_hi},
        "sensitivity": sensitivity,
        "metrics_tensor": metrics_tensor,
    }


# ---------------------------------------------------------------------------
# Multi-panel figure
# ---------------------------------------------------------------------------

def plot_e2e_figure(
    gold: np.ndarray,
    trainee: np.ndarray,
    pipeline_result: dict,
    neural_result: dict,
    out_dir: Path,
) -> None:
    """Generate the comprehensive end-to-end figure."""
    from sopilot.nn.soft_dtw import soft_dtw_align_numpy

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 4, hspace=0.35, wspace=0.35)

    step_colors = ["#E53935", "#43A047", "#1E88E5", "#FDD835", "#8E24AA"]
    step_names = ["Prepare", "Apply", "Check", "Record", "Clean"]

    # ------------------------------------------------------------------
    # Panel A (0,0): Gold embedding structure (t-SNE-like 2D projection)
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    # Project embeddings to 2D via PCA
    from numpy.linalg import svd
    combined = np.vstack([gold, trainee])
    mean = combined.mean(axis=0)
    _, _, Vt = svd(combined - mean, full_matrices=False)
    proj = (combined - mean) @ Vt[:2].T
    gold_2d = proj[:len(gold)]
    trainee_2d = proj[len(gold):]

    n_per_step = len(gold) // 5
    for s in range(5):
        sl = slice(s * n_per_step, (s + 1) * n_per_step)
        ax_a.scatter(gold_2d[sl, 0], gold_2d[sl, 1], c=step_colors[s],
                     s=30, marker="o", alpha=0.7, label=f"G:{step_names[s]}")
    ax_a.scatter(trainee_2d[:, 0], trainee_2d[:, 1], c="gray", s=15,
                 marker="x", alpha=0.5, label="Trainee")
    ax_a.set_title("A) Embedding Space (PCA)")
    ax_a.legend(fontsize=6, ncol=2, loc="upper right")
    ax_a.set_xlabel("PC1")
    ax_a.set_ylabel("PC2")

    # ------------------------------------------------------------------
    # Panel B (0,1): Step boundaries
    # ------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    g_bounds = pipeline_result["step_boundaries"]["gold"]
    t_bounds = pipeline_result["step_boundaries"]["trainee"]

    for i in range(len(g_bounds) - 1):
        ax_b.barh(1, g_bounds[i + 1] - g_bounds[i], left=g_bounds[i],
                  color=step_colors[i % 5], edgecolor="white", height=0.3)
        ax_b.text(g_bounds[i] + (g_bounds[i + 1] - g_bounds[i]) / 2, 1,
                  step_names[i % 5], ha="center", va="center", fontsize=7, fontweight="bold")
    for i in range(len(t_bounds) - 1):
        ax_b.barh(0, t_bounds[i + 1] - t_bounds[i], left=t_bounds[i],
                  color=step_colors[i % 5], edgecolor="white", height=0.3, alpha=0.7)

    ax_b.set_yticks([0, 1])
    ax_b.set_yticklabels(["Trainee", "Gold"])
    ax_b.set_xlabel("Clip index")
    ax_b.set_title("B) Step Segmentation")
    ax_b.set_ylim(-0.5, 1.5)

    # ------------------------------------------------------------------
    # Panel C (0,2): Soft-DTW alignment matrix
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    _, dtw_cost, alignment_mat = soft_dtw_align_numpy(gold, trainee, gamma=0.5)
    ax_c.imshow(alignment_mat, aspect="auto", cmap="inferno", interpolation="nearest")
    ax_c.set_title(f"C) Soft-DTW Alignment\n(γ=0.5, cost={dtw_cost:.3f})")
    ax_c.set_xlabel("Trainee")
    ax_c.set_ylabel("Gold")
    for b in g_bounds:
        ax_c.axhline(y=b - 0.5, color="cyan", linewidth=0.5, alpha=0.5, linestyle="--")

    # ------------------------------------------------------------------
    # Panel D (0,3): Alignment path (from pipeline)
    # ------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[0, 3])
    path = pipeline_result["alignment_preview"]
    g_idx = [p["gold_clip"] for p in path]
    t_idx = [p["trainee_clip"] for p in path]
    sims = [p["similarity"] for p in path]

    scatter = ax_d.scatter(t_idx, g_idx, c=sims, cmap="RdYlGn", s=8, vmin=0.5, vmax=1.0)
    fig.colorbar(scatter, ax=ax_d, fraction=0.046, pad=0.04, label="Similarity")
    # Perfect timing reference
    max_val = max(max(g_idx), max(t_idx))
    ax_d.plot([0, max_val], [0, max_val], "k--", alpha=0.3, linewidth=1)
    ax_d.set_title("D) Alignment Path")
    ax_d.set_xlabel("Trainee clip")
    ax_d.set_ylabel("Gold clip")

    # ------------------------------------------------------------------
    # Panel E (1,0-1): 15 Penalty Metrics
    # ------------------------------------------------------------------
    ax_e = fig.add_subplot(gs[1, 0:2])
    metrics = pipeline_result["metrics"]
    metric_names = list(metrics.keys())
    metric_vals = list(metrics.values())

    y_pos = np.arange(len(metric_names))
    colors_m = ["#E53935" if v > 0.1 else "#43A047" for v in metric_vals]
    bars = ax_e.barh(y_pos, metric_vals, color=colors_m, alpha=0.8, edgecolor="white")
    ax_e.set_yticks(y_pos)
    ax_e.set_yticklabels([n.replace("_", "\n") for n in metric_names], fontsize=7)
    ax_e.set_xlabel("Value")
    ax_e.set_title("E) 15 Penalty Metrics (from alignment)")
    ax_e.invert_yaxis()
    ax_e.grid(axis="x", alpha=0.3)

    # Annotate non-zero values
    for bar, v in zip(bars, metric_vals, strict=True):
        if v > 0.001:
            ax_e.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                      f"{v:.3f}", va="center", fontsize=7)

    # ------------------------------------------------------------------
    # Panel F (1,2): Heuristic vs Neural Score
    # ------------------------------------------------------------------
    ax_f = fig.add_subplot(gs[1, 2])
    heur_score = pipeline_result["score"]
    neural_score = neural_result["uncertainty"]["score"]

    ax_f.barh([1], [heur_score], color="#42A5F5", height=0.3, label=f"Heuristic: {heur_score:.1f}")
    ax_f.barh([0], [neural_score], color="#FF7043", height=0.3, label=f"Neural MLP: {neural_score:.1f}")

    # Add CI
    unc = neural_result["uncertainty"]
    ax_f.errorbar(neural_score, 0, xerr=[[neural_score - unc["ci_lower"]],
                                          [unc["ci_upper"] - neural_score]],
                  fmt="none", color="black", capsize=5, linewidth=2)

    ax_f.set_yticks([0, 1])
    ax_f.set_yticklabels(["Neural\n(MC Dropout)", "Heuristic\n(formula)"])
    ax_f.set_xlabel("Score [0-100]")
    ax_f.set_xlim(0, 105)
    ax_f.set_title("F) Score Comparison")
    ax_f.grid(axis="x", alpha=0.3)
    ax_f.legend(fontsize=8)

    # ------------------------------------------------------------------
    # Panel G (1,3): MC Dropout distribution
    # ------------------------------------------------------------------
    ax_g = fig.add_subplot(gs[1, 3])
    mc_samples = neural_result["mc_samples"]
    ax_g.hist(mc_samples, bins=20, color="#FF7043", alpha=0.7, edgecolor="white", density=True)
    ax_g.axvline(unc["score"], color="black", linewidth=2, linestyle="-", label=f"Mean: {unc['score']:.1f}")
    ax_g.axvline(unc["ci_lower"], color="red", linewidth=1.5, linestyle="--",
                 label=f"95% CI: [{unc['ci_lower']:.1f}, {unc['ci_upper']:.1f}]")
    ax_g.axvline(unc["ci_upper"], color="red", linewidth=1.5, linestyle="--")
    ax_g.set_xlabel("Score")
    ax_g.set_ylabel("Density")
    ax_g.set_title(f"G) MC Dropout (n=100)\nσ = {unc['uncertainty']:.2f}")
    ax_g.legend(fontsize=7)

    # ------------------------------------------------------------------
    # Panel H (2,0): Conformal Prediction
    # ------------------------------------------------------------------
    ax_h = fig.add_subplot(gs[2, 0])
    conf = neural_result["conformal"]

    # Draw intervals
    ax_h.barh(0, conf["upper"] - conf["lower"], left=conf["lower"],
              color="#66BB6A", alpha=0.4, height=0.3, label="Conformal 90% PI")
    ax_h.barh(0, unc["ci_upper"] - unc["ci_lower"], left=unc["ci_lower"],
              color="#42A5F5", alpha=0.4, height=0.2, label="MC Dropout 95% CI")
    ax_h.plot(unc["score"], 0, "ko", markersize=10, zorder=5)

    ax_h.set_yticks([])
    ax_h.set_xlabel("Score [0-100]")
    ax_h.set_title("H) Uncertainty Intervals")
    ax_h.legend(fontsize=8)
    ax_h.set_xlim(0, 100)

    # Guarantee annotation
    ax_h.annotate(
        r"$\mathrm{P}(Y \in C) \geq 1 - \alpha$" + "\n(finite-sample, distribution-free)",
        xy=(0.5, -0.35), xycoords="axes fraction", fontsize=9,
        ha="center", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    # ------------------------------------------------------------------
    # Panel I (2,1): Metric Sensitivity (Integrated Gradients)
    # ------------------------------------------------------------------
    ax_i = fig.add_subplot(gs[2, 1])
    sensitivity = neural_result["sensitivity"]
    sorted_sens = sorted(sensitivity.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    s_names = [s[0].replace("_", "\n") for s in sorted_sens]
    s_vals = [s[1] for s in sorted_sens]
    s_colors = ["#E53935" if v < 0 else "#43A047" for v in s_vals]

    ax_i.barh(range(len(s_names)), s_vals, color=s_colors, alpha=0.8, edgecolor="white")
    ax_i.set_yticks(range(len(s_names)))
    ax_i.set_yticklabels(s_names, fontsize=7)
    ax_i.set_xlabel("∂score / ∂metric")
    ax_i.set_title("I) Metric Sensitivity\n(Counterfactual)")
    ax_i.invert_yaxis()
    ax_i.axvline(0, color="black", linewidth=0.5)
    ax_i.grid(axis="x", alpha=0.3)

    # ------------------------------------------------------------------
    # Panel J (2,2-3): Detected deviations timeline
    # ------------------------------------------------------------------
    ax_j = fig.add_subplot(gs[2, 2:4])
    deviations = pipeline_result["deviations"]

    # Draw gold timeline
    for i in range(len(g_bounds) - 1):
        ax_j.barh(1, g_bounds[i + 1] - g_bounds[i], left=g_bounds[i],
                  color=step_colors[i % 5], edgecolor="white", height=0.25, alpha=0.5)

    # Draw deviation markers
    dev_colors = {"step_missing": "#E53935", "order_swap": "#FF9800", "execution_deviation": "#FDD835"}
    dev_y = 0.5
    for dev in deviations:
        dtype = dev.get("type", "unknown")
        color = dev_colors.get(dtype, "gray")
        g_time = dev.get("gold_time", {})
        start = g_time.get("start_sec", 0)
        end = g_time.get("end_sec", start + 1)
        ax_j.barh(dev_y, end - start, left=start, color=color, edgecolor="black",
                  height=0.2, alpha=0.8)
        ax_j.text((start + end) / 2, dev_y, dtype.replace("_", "\n"),
                  ha="center", va="center", fontsize=6, fontweight="bold")
        dev_y -= 0.25

    ax_j.set_yticks([1])
    ax_j.set_yticklabels(["Gold steps"])
    ax_j.set_xlabel("Time (sec)")
    ax_j.set_title("J) Detected Deviations")
    ax_j.set_ylim(-0.5, 1.5)

    # Legend for deviation types
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=t.replace("_", " ").title())
                       for t, c in dev_colors.items()]
    ax_j.legend(handles=legend_elements, fontsize=7, loc="upper right")

    # ------------------------------------------------------------------
    # Main title
    # ------------------------------------------------------------------
    fig.suptitle(
        "SOPilot Neural Scoring Pipeline — End-to-End Demonstration\n"
        f"Heuristic Score: {heur_score:.1f} | Neural Score: {neural_score:.1f} "
        f"(σ={unc['uncertainty']:.2f}) | "
        f"Conformal 90% PI: [{conf['lower']:.1f}, {conf['upper']:.1f}]",
        fontsize=14, fontweight="bold", y=1.0,
    )

    fig.savefig(out_dir / "e2e_pipeline.png")
    plt.close(fig)
    logger.info("Saved e2e_pipeline.png")


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_pipeline_summary(pipeline_result: dict, neural_result: dict) -> None:
    """Print a textual summary of the pipeline results."""
    logger.info("\n" + "=" * 70)
    logger.info("END-TO-END PIPELINE RESULTS")
    logger.info("=" * 70)

    metrics = pipeline_result["metrics"]
    logger.info("\nScenario: Gold (5 steps × 8 clips) vs Trainee (swap 1-2, skip 4, slow 0)")
    logger.info(f"  Gold clips:    {pipeline_result['clip_count']['gold']}")
    logger.info(f"  Trainee clips: {pipeline_result['clip_count']['trainee']}")
    logger.info(f"  Gold steps:    {pipeline_result['step_boundaries']['gold']}")
    logger.info(f"  Trainee steps: {pipeline_result['step_boundaries']['trainee']}")

    logger.info("\n15 Penalty Metrics:")
    for k, v in metrics.items():
        flag = " <<<" if isinstance(v, (int, float)) and v > 0.1 else ""
        logger.info(f"  {k:<40} {v:>8.4f}{flag}")

    logger.info(f"\nHeuristic Score: {pipeline_result['score']:.1f} / 100")

    unc = neural_result["uncertainty"]
    conf = neural_result["conformal"]
    logger.info(f"Neural MLP Score: {unc['score']:.1f} ± {unc['uncertainty']:.2f}")
    logger.info(f"  MC Dropout 95% CI: [{unc['ci_lower']:.1f}, {unc['ci_upper']:.1f}]")
    logger.info(f"  Conformal  90% PI: [{conf['lower']:.1f}, {conf['upper']:.1f}]")

    logger.info(f"\nDeviations detected: {len(pipeline_result['deviations'])}")
    for dev in pipeline_result["deviations"]:
        logger.info(f"  {dev.get('type', 'unknown')}: step {dev.get('gold_step', '?')}")

    logger.info("\nTop-5 most sensitive metrics:")
    sorted_sens = sorted(neural_result["sensitivity"].items(),
                         key=lambda x: abs(x[1]), reverse=True)[:5]
    for name, val in sorted_sens:
        direction = "↑ score" if val > 0 else "↓ score"
        logger.info(f"  {name:<35} {val:>+8.4f}  ({direction})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SOPilot E2E Neural Pipeline Demo")
    parser.add_argument("--out-dir", type=str, default="demo_outputs", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("SOPilot Neural Scoring Pipeline — End-to-End Demo")
    logger.info("=" * 70)

    # Step 1: Generate data
    logger.info("\n[1/4] Generating synthetic SOP scenario...")
    gold, trainee, gold_meta, trainee_meta = _generate_realistic_scenario()
    logger.info(f"  Gold: {gold.shape}, Trainee: {trainee.shape}")
    logger.info("  Trainee deviations: swap steps 1-2, skip step 4, slow step 0")

    # Step 2: Run evaluate_sop pipeline
    logger.info("\n[2/4] Running evaluate_sop() pipeline...")
    pipeline_result = run_pipeline(gold, trainee, gold_meta, trainee_meta)
    logger.info(f"  Heuristic score: {pipeline_result['score']:.1f}")
    logger.info(f"  Deviations: {len(pipeline_result['deviations'])}")

    # Step 3: Neural scoring
    logger.info("\n[3/4] Running neural scoring (ScoringHead + MC Dropout + Conformal)...")
    neural_result = run_neural_scoring(pipeline_result["metrics"])
    logger.info(f"  Neural score: {neural_result['uncertainty']['score']:.1f} "
                f"± {neural_result['uncertainty']['uncertainty']:.2f}")

    # Step 4: Generate figure
    logger.info("\n[4/4] Generating end-to-end visualization...")
    plot_e2e_figure(gold, trainee, pipeline_result, neural_result, out_dir)

    # Summary
    print_pipeline_summary(pipeline_result, neural_result)

    logger.info("\n" + "=" * 70)
    logger.info(f"Done! Figure saved to {out_dir}/e2e_pipeline.png")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
