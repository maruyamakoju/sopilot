#!/usr/bin/env python
"""SOPilot Neural Pipeline Visualization Demo.

Generates publication-quality figures demonstrating:
1. Soft-DTW alignment matrix with optimal path
2. Euclidean distance vs Soft-DTW vs Optimal Transport comparison
3. MC Dropout uncertainty + Conformal prediction intervals
4. DILATE loss decomposition (shape vs temporal)
5. Explainability: alignment heatmap + metric sensitivity

Usage:
    python scripts/demo_neural_pipeline.py [--out-dir demo_outputs]

Output: matplotlib figures saved to demo_outputs/
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

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Consistent style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Synthetic Data Generation
# ---------------------------------------------------------------------------

def _generate_sop_sequences(
    n_steps: int = 5,
    clips_per_step: int = 6,
    embed_dim: int = 64,
    noise: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Generate synthetic gold + trainee SOP embedding sequences.

    Gold: perfectly ordered steps.
    Trainee: realistic deviations (swap step 2-3, skip step 4, slow on step 1).

    Returns:
        (gold_emb, trainee_emb, step_centroids, gold_boundaries)
    """
    rng = np.random.RandomState(seed)

    # Step centroids: well-separated points in embedding space
    centroids = rng.randn(n_steps, embed_dim).astype(np.float32)
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    # Gold: steps 0,1,2,3,4 in order, clips_per_step each
    gold_clips = []
    gold_boundaries = []
    for s in range(n_steps):
        gold_boundaries.append(len(gold_clips))
        for _ in range(clips_per_step):
            clip = centroids[s] + rng.randn(embed_dim).astype(np.float32) * noise
            gold_clips.append(clip / np.linalg.norm(clip))
    gold = np.stack(gold_clips)

    # Trainee: deviations
    trainee_clips = []
    # Step 0: slow (8 clips instead of 6)
    for _ in range(8):
        clip = centroids[0] + rng.randn(embed_dim).astype(np.float32) * noise * 1.5
        trainee_clips.append(clip / np.linalg.norm(clip))
    # Step 2 before step 1 (swap)
    for _ in range(clips_per_step):
        clip = centroids[2] + rng.randn(embed_dim).astype(np.float32) * noise
        trainee_clips.append(clip / np.linalg.norm(clip))
    for _ in range(clips_per_step):
        clip = centroids[1] + rng.randn(embed_dim).astype(np.float32) * noise
        trainee_clips.append(clip / np.linalg.norm(clip))
    # Step 3: normal
    for _ in range(clips_per_step):
        clip = centroids[3] + rng.randn(embed_dim).astype(np.float32) * noise
        trainee_clips.append(clip / np.linalg.norm(clip))
    # Step 4: skipped (missing)
    trainee = np.stack(trainee_clips)

    return gold, trainee, centroids, gold_boundaries


# ---------------------------------------------------------------------------
# Figure 1: Soft-DTW Alignment
# ---------------------------------------------------------------------------

def plot_soft_dtw_alignment(gold: np.ndarray, trainee: np.ndarray, out_dir: Path) -> None:
    """Soft-DTW alignment matrix + path visualization."""
    from sopilot.nn.soft_dtw import soft_dtw_align_numpy

    # Get alignment with different gamma values
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])

    gammas = [0.01, 0.5, 5.0]
    titles = [r"$\gamma = 0.01$ (Hard DTW)", r"$\gamma = 0.5$ (Moderate)", r"$\gamma = 5.0$ (Soft)"]

    for idx, (gamma, title) in enumerate(zip(gammas, titles, strict=True)):
        path, cost, alignment = soft_dtw_align_numpy(gold, trainee, gamma=gamma)
        ax = fig.add_subplot(gs[0, idx])
        im = ax.imshow(alignment, aspect="auto", cmap="inferno", interpolation="nearest")

        # Overlay path for hard DTW
        if gamma <= 0.1:
            path_g = [p[0] for p in path]
            path_t = [p[1] for p in path]
            ax.plot(path_t, path_g, "c-", linewidth=1.5, alpha=0.8)

        ax.set_xlabel("Trainee clip index")
        if idx == 0:
            ax.set_ylabel("Gold clip index")
        ax.set_title(f"{title}\ncost = {cost:.3f}")

        # Add step boundary annotations
        for b in [0, 6, 12, 18, 24]:
            ax.axhline(y=b - 0.5, color="white", linewidth=0.5, alpha=0.5, linestyle="--")

    # Shared colorbar
    cbar_ax = fig.add_subplot(gs[0, 3])
    fig.colorbar(im, cax=cbar_ax, label="Alignment probability")

    fig.suptitle("Soft-DTW Alignment: Effect of Smoothing Parameter γ", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "01_soft_dtw_alignment.png")
    plt.close(fig)
    logger.info("Saved 01_soft_dtw_alignment.png")


# ---------------------------------------------------------------------------
# Figure 2: Euclidean vs Soft-DTW vs OT
# ---------------------------------------------------------------------------

def plot_alignment_comparison(gold: np.ndarray, trainee: np.ndarray, out_dir: Path) -> None:
    """Compare Euclidean distance, Soft-DTW, and Optimal Transport alignment."""
    from sopilot.nn.optimal_transport import SinkhornDistance
    from sopilot.nn.soft_dtw import soft_dtw_align_numpy

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Cosine distance matrix (raw, no alignment)
    gold_norm = gold / np.linalg.norm(gold, axis=1, keepdims=True)
    trainee_norm = trainee / np.linalg.norm(trainee, axis=1, keepdims=True)
    cosine_dist = 1.0 - gold_norm @ trainee_norm.T

    ax = axes[0]
    im0 = ax.imshow(cosine_dist, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_title("Cosine Distance Matrix\n(no temporal alignment)")
    ax.set_xlabel("Trainee clip index")
    ax.set_ylabel("Gold clip index")
    fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

    # 2. Soft-DTW alignment
    _, dtw_cost, dtw_align = soft_dtw_align_numpy(gold, trainee, gamma=0.5)
    ax = axes[1]
    im1 = ax.imshow(dtw_align, aspect="auto", cmap="inferno", interpolation="nearest")
    ax.set_title(f"Soft-DTW Alignment\n(γ=0.5, cost={dtw_cost:.3f})")
    ax.set_xlabel("Trainee clip index")
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    # 3. Optimal Transport (Sinkhorn)
    cost_tensor = torch.from_numpy(cosine_dist.astype(np.float32)).unsqueeze(0)  # (1, M, N)
    sinkhorn = SinkhornDistance(epsilon=0.1, max_iter=100)
    ot_cost, ot_plan = sinkhorn(cost_tensor)
    ot_plan_np = ot_plan[0].detach().numpy()

    ax = axes[2]
    im2 = ax.imshow(ot_plan_np, aspect="auto", cmap="inferno", interpolation="nearest")
    ax.set_title(f"Optimal Transport (Sinkhorn)\n(ε=0.1, cost={ot_cost.item():.3f})")
    ax.set_xlabel("Trainee clip index")
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    # Add step labels
    for ax in axes:
        for b in [0, 6, 12, 18, 24]:
            ax.axhline(y=b - 0.5, color="white", linewidth=0.5, alpha=0.4, linestyle="--")

    fig.suptitle("Alignment Method Comparison: Cosine vs Soft-DTW vs Optimal Transport",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "02_alignment_comparison.png")
    plt.close(fig)
    logger.info("Saved 02_alignment_comparison.png")


# ---------------------------------------------------------------------------
# Figure 3: Scoring with Uncertainty + Conformal
# ---------------------------------------------------------------------------

def plot_scoring_uncertainty(out_dir: Path) -> None:
    """MC Dropout uncertainty estimation + Conformal prediction intervals."""
    from sopilot.nn.conformal import SplitConformalPredictor
    from sopilot.nn.scoring_head import N_METRICS, ScoringHead

    rng = np.random.RandomState(42)
    torch.manual_seed(42)

    # Initialize model (random weights — showing the mechanism, not trained results)
    model = ScoringHead(n_inputs=N_METRICS)
    model.eval()

    # Generate synthetic calibration data: 50 samples
    n_cal = 50
    cal_metrics = torch.randn(n_cal, N_METRICS) * 0.3
    with torch.no_grad():
        cal_preds_raw = []
        for i in range(n_cal):
            pred = model._forward_single(cal_metrics[i:i+1])
            cal_preds_raw.append(pred)
    cal_preds = np.array(cal_preds_raw)
    # Simulate "actual" scores with noise around predictions
    cal_actuals = np.clip(cal_preds + rng.randn(n_cal) * 8, 0, 100)

    # Calibrate conformal predictor
    conformal = SplitConformalPredictor(alpha=0.1)  # 90% coverage
    conformal.calibrate(cal_preds, cal_actuals)

    # Generate test scenarios: sweep one metric
    n_test = 30
    test_scenarios = []
    test_scores = []
    test_uncertainties = []
    test_ci_lower = []
    test_ci_upper = []
    conf_lower = []
    conf_upper = []

    base_metrics = torch.zeros(1, N_METRICS)
    for i in range(n_test):
        x = base_metrics.clone()
        x[0, 0] = i / n_test * 2.0  # Sweep "miss" penalty from 0 to 2
        result = model.predict_with_uncertainty(x, n_samples=50)
        test_scenarios.append(i / n_test * 2.0)
        test_scores.append(result["score"])
        test_uncertainties.append(result["uncertainty"])
        test_ci_lower.append(result["ci_lower"])
        test_ci_upper.append(result["ci_upper"])
        _, cl, cu = conformal.predict(result["score"])
        conf_lower.append(cl)
        conf_upper.append(cu)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: MC Dropout uncertainty
    x_vals = np.array(test_scenarios)
    scores = np.array(test_scores)
    ci_lo = np.array(test_ci_lower)
    ci_hi = np.array(test_ci_upper)

    ax1.plot(x_vals, scores, "b-", linewidth=2, label="Score")
    ax1.fill_between(x_vals, ci_lo, ci_hi, alpha=0.3, color="blue", label="MC Dropout 95% CI")
    ax1.set_xlabel("Miss penalty (input metric)")
    ax1.set_ylabel("SOP Score [0-100]")
    ax1.set_title("MC Dropout Uncertainty Estimation\n(Gal & Ghahramani, 2016)")
    ax1.legend()
    ax1.set_ylim(0, 100)
    ax1.grid(alpha=0.3)

    # Right: Conformal prediction intervals
    cf_lo = np.array(conf_lower)
    cf_hi = np.array(conf_upper)
    ax2.plot(x_vals, scores, "r-", linewidth=2, label="Score")
    ax2.fill_between(x_vals, cf_lo, cf_hi, alpha=0.2, color="red",
                     label=f"Conformal 90% PI (n_cal={n_cal})")
    ax2.fill_between(x_vals, ci_lo, ci_hi, alpha=0.3, color="blue",
                     label="MC Dropout 95% CI")
    ax2.set_xlabel("Miss penalty (input metric)")
    ax2.set_ylabel("SOP Score [0-100]")
    ax2.set_title("Conformal Prediction Intervals\n(Lei et al., 2018 — distribution-free)")
    ax2.legend()
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.3)

    # Add guarantee annotation
    ax2.annotate(
        r"$\mathrm{P}(Y \in [\hat{y} \pm q]) \geq 1 - \alpha$" + "\n(finite-sample guarantee)",
        xy=(0.5, 0.02), xycoords="axes fraction",
        fontsize=10, ha="center", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    fig.suptitle("Uncertainty Quantification: MC Dropout + Conformal Prediction",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "03_uncertainty_conformal.png")
    plt.close(fig)
    logger.info("Saved 03_uncertainty_conformal.png")


# ---------------------------------------------------------------------------
# Figure 4: DILATE Loss Decomposition
# ---------------------------------------------------------------------------

def plot_dilate_decomposition(gold: np.ndarray, trainee: np.ndarray, out_dir: Path) -> None:
    """DILATE loss: shape vs temporal distortion components."""
    from sopilot.nn.dilate_loss import DILATELoss

    gold_t = torch.from_numpy(gold)
    trainee_t = torch.from_numpy(trainee)

    # Sweep alpha: shape emphasis → temporal emphasis
    alphas = np.linspace(0.0, 1.0, 21)
    shape_losses = []
    temporal_losses = []
    total_losses = []

    for alpha in alphas:
        loss_fn = DILATELoss(alpha=float(alpha), gamma=0.5)
        total, components = loss_fn(gold_t, trainee_t)
        shape_losses.append(components["shape"].item())
        temporal_losses.append(components["temporal"].item())
        total_losses.append(total.item())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Loss decomposition across alpha
    ax1.plot(alphas, shape_losses, "b-o", markersize=3, linewidth=2, label="Shape loss (Soft-DTW)")
    ax1.plot(alphas, temporal_losses, "r-s", markersize=3, linewidth=2, label="Temporal distortion (TDI)")
    ax1.plot(alphas, total_losses, "k--", linewidth=1.5, alpha=0.6, label=r"Total: $\alpha \cdot S + (1-\alpha) \cdot T$")
    ax1.set_xlabel(r"$\alpha$ (shape weight)")
    ax1.set_ylabel("Loss")
    ax1.set_title(r"DILATE = $\alpha \cdot \mathcal{L}_{shape}$ + $(1-\alpha) \cdot \mathcal{L}_{temporal}$")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Right: Alignment at α=0.5 (balanced)
    loss_fn = DILATELoss(alpha=0.5, gamma=0.5)
    _, components = loss_fn(gold_t, trainee_t)
    alignment = components["alignment"].detach().numpy()

    ax2.imshow(alignment, aspect="auto", cmap="inferno", interpolation="nearest")
    ax2.set_title(f"Soft Alignment (γ=0.5)\nShape={components['shape'].item():.3f}, "
                  f"Temporal={components['temporal'].item():.3f}")
    ax2.set_xlabel("Trainee clip index")
    ax2.set_ylabel("Gold clip index")

    # Annotate diagonal (perfect timing reference)
    m, n = alignment.shape
    diag_x = np.linspace(0, n - 1, 100)
    diag_y = np.linspace(0, m - 1, 100)
    ax2.plot(diag_x, diag_y, "w--", linewidth=1, alpha=0.5, label="Perfect timing")
    ax2.legend(loc="upper left")

    fig.suptitle("DILATE Loss Decomposition (Guen & Thome, NeurIPS 2019)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "04_dilate_decomposition.png")
    plt.close(fig)
    logger.info("Saved 04_dilate_decomposition.png")


# ---------------------------------------------------------------------------
# Figure 5: Explainability — Alignment Heatmap + Metric Sensitivity
# ---------------------------------------------------------------------------

def plot_explainability(gold: np.ndarray, trainee: np.ndarray, out_dir: Path) -> None:
    """Alignment heatmap with per-frame importance + metric sensitivity analysis."""
    from sopilot.nn.explainability import CounterfactualExplainer, TemporalAttentionVisualizer
    from sopilot.nn.scoring_head import N_METRICS, ScoringHead
    from sopilot.nn.soft_dtw import soft_dtw_align_numpy

    # Alignment heatmap
    _, _, alignment = soft_dtw_align_numpy(gold, trainee, gamma=0.5)
    viz_data = TemporalAttentionVisualizer.alignment_heatmap(alignment)

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])

    # Main heatmap
    ax_main = fig.add_subplot(gs[0, 0])
    ax_main.imshow(viz_data["heatmap"], aspect="auto", cmap="inferno", interpolation="nearest")
    ax_main.set_ylabel("Gold clip index")
    ax_main.set_title("Temporal Alignment Heatmap")

    # Annotate step regions
    step_names = ["Step 1\n(Prepare)", "Step 2\n(Apply)", "Step 3\n(Check)",
                  "Step 4\n(Record)", "Step 5\n(Clean)"]
    for b, name in zip([0, 6, 12, 18, 24], step_names, strict=True):
        ax_main.axhline(y=b - 0.5, color="cyan", linewidth=0.8, alpha=0.6, linestyle="--")
        ax_main.text(-0.5, b + 2.5, name, fontsize=7, color="cyan", ha="right",
                     va="center", transform=ax_main.get_yaxis_transform())

    # Peak alignments
    for g_idx, t_idx, strength in viz_data["peak_alignment"][:10]:
        if strength > 0.5:
            ax_main.plot(t_idx, g_idx, "w+", markersize=6, markeredgewidth=1.5)

    # Bottom: trainee importance
    ax_trainee = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_trainee.bar(range(len(viz_data["trainee_importance"])), viz_data["trainee_importance"],
                   color="coral", alpha=0.7)
    ax_trainee.set_xlabel("Trainee clip index")
    ax_trainee.set_ylabel("Importance")
    ax_trainee.set_ylim(0, 1.1)

    # Right: gold importance
    ax_gold = fig.add_subplot(gs[0, 1], sharey=ax_main)
    ax_gold.barh(range(len(viz_data["gold_importance"])), viz_data["gold_importance"],
                 color="steelblue", alpha=0.7)
    ax_gold.set_xlabel("Importance")
    ax_gold.invert_xaxis()

    # Bottom right: Metric sensitivity analysis
    torch.manual_seed(42)
    scoring_head = ScoringHead(n_inputs=N_METRICS)
    scoring_head.eval()
    explainer = CounterfactualExplainer(scoring_head)
    base_metrics = torch.zeros(1, N_METRICS)
    sensitivity = explainer.compute_sensitivity(base_metrics)

    ax_sens = fig.add_subplot(gs[1, 1])
    top_k = 5
    top_metrics = sorted(sensitivity.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    names = [m[0].replace("_", "\n") for m in top_metrics]
    values = [m[1] for m in top_metrics]
    colors = ["red" if v < 0 else "green" for v in values]
    ax_sens.barh(range(top_k), values, color=colors, alpha=0.7)
    ax_sens.set_yticks(range(top_k))
    ax_sens.set_yticklabels(names, fontsize=7)
    ax_sens.set_xlabel("∂score/∂metric")
    ax_sens.set_title("Metric\nSensitivity", fontsize=10)
    ax_sens.invert_yaxis()

    fig.suptitle("Explainability: Temporal Attention + Metric Sensitivity",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "05_explainability.png")
    plt.close(fig)
    logger.info("Saved 05_explainability.png")


# ---------------------------------------------------------------------------
# Figure 6: Architecture Summary (info graphic)
# ---------------------------------------------------------------------------

def plot_architecture_summary(out_dir: Path) -> None:
    """Architecture diagram showing the full neural pipeline."""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Boxes
    boxes = [
        (0.3, 4.5, "Video\nInput", "#E8F5E9"),
        (2.0, 4.5, "Embedder\n(V-JEPA2 /\nHeuristic)", "#E3F2FD"),
        (4.0, 4.5, "Step\nSegmenter\n(ASFormer /\nMS-TCN++)", "#FFF3E0"),
        (6.0, 4.5, "Temporal\nAlignment\n(Soft-DTW / OT)", "#FCE4EC"),
        (8.0, 4.5, "Scoring\nHead\n(MLP + MC\nDropout)", "#F3E5F5"),
        (9.5, 4.5, "Score\n[0-100]", "#E8F5E9"),
    ]

    for x, y, text, color in boxes:
        w, h = 1.5, 1.2
        rect = plt.Rectangle((x, y - h/2), w, h, facecolor=color,
                              edgecolor="black", linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y, text, ha="center", va="center", fontsize=8,
                fontweight="bold", zorder=3)

    # Arrows
    arrow_props = dict(arrowstyle="->", linewidth=2, color="black")
    for x_start in [1.8, 3.5, 5.5, 7.5, 9.5]:
        ax.annotate("", xy=(x_start + 0.2, 4.5), xytext=(x_start, 4.5),
                    arrowprops=arrow_props)

    # Lower modules
    lower_boxes = [
        (2.0, 2.5, "Projection\nHead\n(NT-Xent)", "#BBDEFB"),
        (4.0, 2.5, "DILATE\nLoss\n(Shape+Time)", "#FFCCBC"),
        (6.0, 2.5, "Conformal\nPrediction\n(α=0.05)", "#F8BBD0"),
        (8.0, 2.5, "Explainability\n(IG + Wachter\nCounterfactual)", "#E1BEE7"),
    ]

    for x, y, text, color in lower_boxes:
        w, h = 1.5, 1.0
        rect = plt.Rectangle((x, y - h/2), w, h, facecolor=color,
                              edgecolor="gray", linewidth=1, linestyle="--", zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y, text, ha="center", va="center", fontsize=7, zorder=3)

    # Connect lower to upper
    for x_upper, x_lower in [(2.75, 2.75), (5.0, 4.75), (8.75, 6.75), (8.75, 8.75)]:
        ax.annotate("", xy=(x_lower, 3.0), xytext=(x_upper, 3.9),
                    arrowprops=dict(arrowstyle="->", linewidth=1, color="gray", linestyle="--"))

    # Labels
    ax.text(5.0, 5.8, "SOPilot Neural Scoring Pipeline", ha="center", va="center",
            fontsize=16, fontweight="bold")
    ax.text(5.0, 0.8, "6-Phase Training: Contrastive → Segmentation → DTW Alignment → "
            "Scoring → Calibration → Conformal",
            ha="center", va="center", fontsize=9, style="italic", color="gray")

    # Tech annotations
    annotations = [
        (2.75, 1.5, "SimCLR\nNT-Xent\nτ=0.07"),
        (4.75, 1.5, "α·Shape +\n(1-α)·Temporal\n+ Boundary"),
        (6.75, 1.5, "P(Y∈C) ≥ 1-α\n∀ distribution"),
        (8.75, 1.5, "Sundararajan\n(ICML 2017)"),
    ]
    for x, y, text in annotations:
        ax.text(x, y, text, ha="center", va="center", fontsize=6,
                color="gray", style="italic")

    fig.savefig(out_dir / "06_architecture.png")
    plt.close(fig)
    logger.info("Saved 06_architecture.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SOPilot Neural Pipeline Visualization Demo")
    parser.add_argument("--out-dir", type=str, default="demo_outputs", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SOPilot Neural Pipeline Visualization Demo")
    logger.info("=" * 60)

    # Generate synthetic data
    logger.info("\n[1/6] Generating synthetic SOP sequences...")
    gold, trainee, centroids, boundaries = _generate_sop_sequences()
    logger.info(f"  Gold: {gold.shape}, Trainee: {trainee.shape}, {len(boundaries)} step boundaries")

    # Figure 1: Soft-DTW alignment
    logger.info("\n[2/6] Plotting Soft-DTW alignment (γ sweep)...")
    plot_soft_dtw_alignment(gold, trainee, out_dir)

    # Figure 2: Alignment comparison
    logger.info("\n[3/6] Plotting alignment method comparison...")
    plot_alignment_comparison(gold, trainee, out_dir)

    # Figure 3: Scoring uncertainty + conformal
    logger.info("\n[4/6] Plotting uncertainty quantification...")
    plot_scoring_uncertainty(out_dir)

    # Figure 4: DILATE loss decomposition
    logger.info("\n[5/6] Plotting DILATE loss decomposition...")
    plot_dilate_decomposition(gold, trainee, out_dir)

    # Figure 5: Explainability
    logger.info("\n[6/6] Plotting explainability analysis...")
    plot_explainability(gold, trainee, out_dir)

    # Bonus: Architecture diagram
    plot_architecture_summary(out_dir)

    logger.info("\n" + "=" * 60)
    logger.info(f"Done! {6} figures saved to {out_dir}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
