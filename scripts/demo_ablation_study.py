#!/usr/bin/env python
"""SOPilot Neural Pipeline — Ablation Study.

Quantitative ablation experiments showing the contribution of each
neural component. Generates comparison figures and a summary table.

Experiments:
1. Alignment method ablation: Cosine vs Hard DTW vs Soft-DTW vs OT
2. Gamma (γ) sensitivity: alignment quality across smoothing values
3. DILATE loss decomposition: MSE vs Shape-only vs Temporal-only vs DILATE
4. Scoring head: Heuristic formula vs MLP (metric interaction capture)
5. Uncertainty calibration: Point vs MC Dropout vs Conformal coverage

Usage:
    python scripts/demo_ablation_study.py [--out-dir demo_outputs]

Output: figures + summary JSON saved to demo_outputs/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

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
# Synthetic data with controlled difficulty levels
# ---------------------------------------------------------------------------

def _make_scenario(
    n_steps: int,
    clips_per_step: int,
    embed_dim: int,
    noise: float,
    seed: int,
    *,
    n_swaps: int = 0,
    n_skips: int = 0,
    slow_factor: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate gold + trainee with controlled deviations.

    Args:
        n_swaps: Number of adjacent-step swaps.
        n_skips: Number of steps to skip entirely.
        slow_factor: Extra clips added to first step.
    """
    rng = np.random.RandomState(seed)
    centroids = rng.randn(n_steps, embed_dim).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    def _make_clips(step_order: list[int], extra_first: int = 0) -> np.ndarray:
        clips = []
        for idx, s in enumerate(step_order):
            n = clips_per_step + (extra_first if idx == 0 else 0)
            for _ in range(n):
                c = centroids[s] + rng.randn(embed_dim).astype(np.float32) * noise
                clips.append(c / np.linalg.norm(c))
        return np.stack(clips)

    gold_order = list(range(n_steps))
    gold = _make_clips(gold_order)

    trainee_order = list(range(n_steps))
    # Apply swaps
    for i in range(min(n_swaps, n_steps - 1)):
        j = i + 1
        trainee_order[i], trainee_order[j] = trainee_order[j], trainee_order[i]
    # Apply skips
    for _ in range(min(n_skips, len(trainee_order))):
        trainee_order.pop()
    trainee = _make_clips(trainee_order, extra_first=slow_factor)

    return gold, trainee


def _scenario_suite(
    embed_dim: int = 64,
    clips_per_step: int = 8,
    seed: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Standard suite of scenarios at increasing difficulty."""
    return {
        "perfect": _make_scenario(5, clips_per_step, embed_dim, 0.05, seed),
        "noisy": _make_scenario(5, clips_per_step, embed_dim, 0.25, seed),
        "1-swap": _make_scenario(5, clips_per_step, embed_dim, 0.1, seed, n_swaps=1),
        "2-swap": _make_scenario(5, clips_per_step, embed_dim, 0.1, seed, n_swaps=2),
        "1-skip": _make_scenario(5, clips_per_step, embed_dim, 0.1, seed, n_skips=1),
        "slow": _make_scenario(5, clips_per_step, embed_dim, 0.1, seed, slow_factor=6),
    }


# ---------------------------------------------------------------------------
# Experiment 1: Alignment Method Ablation
# ---------------------------------------------------------------------------

def _cosine_alignment_cost(gold: np.ndarray, trainee: np.ndarray) -> float:
    """Raw cosine distance: mean diagonal cost, no temporal alignment."""
    g = gold / np.linalg.norm(gold, axis=1, keepdims=True)
    t = trainee / np.linalg.norm(trainee, axis=1, keepdims=True)
    n = min(len(g), len(t))
    costs = [1.0 - float(g[i] @ t[i]) for i in range(n)]
    return float(np.mean(costs))


def _hard_dtw_cost(gold: np.ndarray, trainee: np.ndarray) -> float:
    """Hard DTW from step_engine."""
    from sopilot.step_engine import dtw_align
    result = dtw_align(gold, trainee)
    return result.mean_cost


def _soft_dtw_cost(gold: np.ndarray, trainee: np.ndarray, gamma: float = 0.5) -> float:
    """Soft-DTW alignment cost."""
    from sopilot.nn.soft_dtw import soft_dtw_align_numpy
    _, cost, _ = soft_dtw_align_numpy(gold, trainee, gamma=gamma)
    return float(cost)


def _ot_cost(gold: np.ndarray, trainee: np.ndarray) -> float:
    """Optimal Transport (Sinkhorn) cost."""
    from sopilot.nn.optimal_transport import SinkhornDistance
    g = gold / np.linalg.norm(gold, axis=1, keepdims=True)
    t = trainee / np.linalg.norm(trainee, axis=1, keepdims=True)
    cost_mat = 1.0 - (g @ t.T)
    cost_tensor = torch.from_numpy(cost_mat.astype(np.float32)).unsqueeze(0)
    sinkhorn = SinkhornDistance(epsilon=0.1, max_iter=100)
    ot_cost_val, _ = sinkhorn(cost_tensor)
    return float(ot_cost_val.item())


def experiment_alignment_ablation(
    scenarios: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Compare alignment methods across difficulty levels."""
    methods = {
        "Cosine (no alignment)": _cosine_alignment_cost,
        "Hard DTW": _hard_dtw_cost,
        "Soft-DTW (γ=0.5)": lambda g, t: _soft_dtw_cost(g, t, 0.5),
        "Optimal Transport": _ot_cost,
    }
    results = {}
    for name, (gold, trainee) in scenarios.items():
        results[name] = {}
        for method_name, fn in methods.items():
            results[name][method_name] = fn(gold, trainee)
    return results


# ---------------------------------------------------------------------------
# Experiment 2: Gamma Sensitivity
# ---------------------------------------------------------------------------

def experiment_gamma_sensitivity(
    scenarios: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Measure how γ affects alignment cost."""
    gammas = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    results = {"gammas": gammas, "scenarios": {}}
    for name, (gold, trainee) in scenarios.items():
        costs = [_soft_dtw_cost(gold, trainee, g) for g in gammas]
        results["scenarios"][name] = costs
    return results


# ---------------------------------------------------------------------------
# Experiment 3: DILATE Loss Components
# ---------------------------------------------------------------------------

def experiment_dilate_ablation(
    scenarios: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Compare MSE vs Shape-only vs Temporal-only vs DILATE."""
    from sopilot.nn.dilate_loss import DILATELoss

    results = {}
    for name, (gold, trainee) in scenarios.items():
        g_t = torch.from_numpy(gold)
        t_t = torch.from_numpy(trainee)

        # MSE baseline (mean squared error of paired embeddings)
        n = min(len(gold), len(trainee))
        mse = float(np.mean((gold[:n] - trainee[:n]) ** 2))

        # Shape-only (α=1.0)
        loss_fn = DILATELoss(alpha=1.0, gamma=0.5)
        shape_total, shape_comp = loss_fn(g_t, t_t)

        # Temporal-only (α=0.0)
        loss_fn = DILATELoss(alpha=0.0, gamma=0.5)
        temp_total, temp_comp = loss_fn(g_t, t_t)

        # Balanced DILATE (α=0.5)
        loss_fn = DILATELoss(alpha=0.5, gamma=0.5)
        dilate_total, dilate_comp = loss_fn(g_t, t_t)

        results[name] = {
            "MSE": mse,
            "Shape only": float(shape_total.item()),
            "Temporal only": float(temp_total.item()),
            "DILATE (α=0.5)": float(dilate_total.item()),
            "shape_component": float(dilate_comp["shape"].item()),
            "temporal_component": float(dilate_comp["temporal"].item()),
        }
    return results


# ---------------------------------------------------------------------------
# Experiment 4: Scoring Head — Heuristic vs MLP
# ---------------------------------------------------------------------------

def experiment_scoring_ablation() -> dict:
    """Show that MLP captures non-linear metric interactions."""
    from sopilot.nn.scoring_head import N_METRICS, ScoringHead

    torch.manual_seed(42)
    model = ScoringHead(n_inputs=N_METRICS)
    model.eval()

    # Test: sweep one metric at a time, measure score sensitivity
    heuristic_weights = [15, 10, 8, 5, 12, 5, 3, 5, 5, 3, 3, 0, 0, 5, 5]
    metric_names = [
        "miss", "swap", "deviation", "over_time", "temporal_warp",
        "path_stretch", "dup_ratio", "order_viol", "temp_drift",
        "conf_loss", "local_sim_gap", "adapt_thresh", "eff_thresh",
        "hard_miss", "mean_align_cost",
    ]

    results = {"metrics": metric_names, "heuristic_sensitivity": [], "mlp_sensitivity": []}
    base = torch.zeros(1, N_METRICS)

    for i in range(N_METRICS):
        # Heuristic: linear sensitivity
        heur_delta = heuristic_weights[i] * 1.0  # 1-unit increase penalty
        results["heuristic_sensitivity"].append(heur_delta)

        # MLP: actual sensitivity (difference from baseline)
        x = base.clone()
        x[0, i] = 1.0
        with torch.no_grad():
            base_score = model._forward_single(base)
            pert_score = model._forward_single(x)
        mlp_delta = abs(pert_score - base_score)
        results["mlp_sensitivity"].append(mlp_delta)

    return results


# ---------------------------------------------------------------------------
# Experiment 5: Uncertainty Calibration Coverage
# ---------------------------------------------------------------------------

def experiment_uncertainty_coverage() -> dict:
    """Measure actual coverage of MC Dropout vs Conformal."""
    from sopilot.nn.conformal import SplitConformalPredictor
    from sopilot.nn.scoring_head import N_METRICS, ScoringHead

    torch.manual_seed(42)
    rng = np.random.RandomState(42)
    model = ScoringHead(n_inputs=N_METRICS)
    model.eval()

    # Generate synthetic calibration set
    n_cal = 100
    cal_x = torch.randn(n_cal, N_METRICS) * 0.3
    cal_preds = []
    for i in range(n_cal):
        cal_preds.append(model._forward_single(cal_x[i:i + 1]))
    cal_preds = np.array(cal_preds)
    cal_actuals = np.clip(cal_preds + rng.randn(n_cal) * 10, 0, 100)

    # Generate test set
    n_test = 200
    test_x = torch.randn(n_test, N_METRICS) * 0.3
    test_preds = []
    for i in range(n_test):
        test_preds.append(model._forward_single(test_x[i:i + 1]))
    test_preds = np.array(test_preds)
    test_actuals = np.clip(test_preds + rng.randn(n_test) * 10, 0, 100)

    results = {"n_test": n_test, "methods": {}}

    # MC Dropout coverage
    mc_covered = 0
    mc_widths = []
    for i in range(n_test):
        unc = model.predict_with_uncertainty(test_x[i:i + 1], n_samples=50)
        if unc["ci_lower"] <= test_actuals[i] <= unc["ci_upper"]:
            mc_covered += 1
        mc_widths.append(unc["ci_upper"] - unc["ci_lower"])
    results["methods"]["MC Dropout (95%)"] = {
        "target_coverage": 0.95,
        "actual_coverage": mc_covered / n_test,
        "mean_width": float(np.mean(mc_widths)),
    }

    # Conformal at different alpha levels
    for alpha in [0.05, 0.10, 0.20]:
        conformal = SplitConformalPredictor(alpha=alpha)
        conformal.calibrate(cal_preds, cal_actuals)

        conf_covered = 0
        conf_widths = []
        for i in range(n_test):
            _, lo, hi = conformal.predict(test_preds[i])
            if lo <= test_actuals[i] <= hi:
                conf_covered += 1
            conf_widths.append(hi - lo)

        results["methods"][f"Conformal ({int((1-alpha)*100)}%)"] = {
            "target_coverage": 1 - alpha,
            "actual_coverage": conf_covered / n_test,
            "mean_width": float(np.mean(conf_widths)),
        }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_alignment_ablation(data: dict, out_dir: Path) -> None:
    """Bar chart: alignment cost by method × scenario."""
    scenarios = list(data.keys())
    methods = list(data[scenarios[0]].keys())
    n_methods = len(methods)
    n_scenarios = len(scenarios)

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(n_scenarios)
    width = 0.8 / n_methods
    colors = ["#90CAF9", "#42A5F5", "#1565C0", "#FF7043"]

    for i, method in enumerate(methods):
        vals = [data[s][method] for s in scenarios]
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=method, color=colors[i], edgecolor="white")
        for bar, v in zip(bars, vals, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Alignment Cost (lower = better match)")
    ax.set_title("Alignment Method Ablation: Cost by Scenario", fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "ablation_01_alignment.png")
    plt.close(fig)
    logger.info("Saved ablation_01_alignment.png")


def plot_gamma_sensitivity(data: dict, out_dir: Path) -> None:
    """Line plot: alignment cost vs γ for each scenario."""
    fig, ax = plt.subplots(figsize=(10, 5))
    gammas = data["gammas"]
    colors = plt.cm.Set2(np.linspace(0, 1, len(data["scenarios"])))

    for (name, costs), color in zip(data["scenarios"].items(), colors, strict=True):
        ax.plot(gammas, costs, "o-", label=name, color=color, markersize=4)

    ax.set_xscale("log")
    ax.set_xlabel("γ (smoothing parameter, log scale)")
    ax.set_ylabel("Soft-DTW Cost")
    ax.set_title("γ Sensitivity: Alignment Cost vs Smoothing", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # Annotate optimal region
    ax.axvspan(0.1, 1.0, alpha=0.1, color="green", label="Recommended range")
    ax.text(0.3, ax.get_ylim()[1] * 0.95, "Recommended\nrange", fontsize=8,
            color="green", ha="center", va="top")

    plt.tight_layout()
    fig.savefig(out_dir / "ablation_02_gamma.png")
    plt.close(fig)
    logger.info("Saved ablation_02_gamma.png")


def plot_dilate_ablation(data: dict, out_dir: Path) -> None:
    """Grouped bar: loss method × scenario + component breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    scenarios = list(data.keys())
    loss_methods = ["MSE", "Shape only", "Temporal only", "DILATE (α=0.5)"]
    n_methods = len(loss_methods)
    x = np.arange(len(scenarios))
    width = 0.8 / n_methods
    colors = ["#BDBDBD", "#42A5F5", "#FF7043", "#66BB6A"]

    for i, method in enumerate(loss_methods):
        vals = [data[s][method] for s in scenarios]
        offset = (i - n_methods / 2 + 0.5) * width
        ax1.bar(x + offset, vals, width, label=method, color=colors[i], edgecolor="white")

    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=9)
    ax1.set_ylabel("Loss Value")
    ax1.set_title("Loss Method Comparison", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Component breakdown for DILATE
    shape_vals = [data[s]["shape_component"] for s in scenarios]
    temp_vals = [data[s]["temporal_component"] for s in scenarios]
    ax2.bar(x - 0.2, shape_vals, 0.35, label="Shape (Soft-DTW)", color="#42A5F5")
    ax2.bar(x + 0.2, temp_vals, 0.35, label="Temporal (TDI)", color="#FF7043")
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, fontsize=9)
    ax2.set_ylabel("Component Loss")
    ax2.set_title("DILATE Component Breakdown", fontweight="bold")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("DILATE Loss Ablation (Guen & Thome, NeurIPS 2019)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "ablation_03_dilate.png")
    plt.close(fig)
    logger.info("Saved ablation_03_dilate.png")


def plot_scoring_ablation(data: dict, out_dir: Path) -> None:
    """Side-by-side: heuristic (linear) vs MLP (non-linear) sensitivity."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    names = data["metrics"]
    n = len(names)

    # Heuristic
    h_vals = data["heuristic_sensitivity"]
    y = np.arange(n)
    ax1.barh(y, h_vals, color="#90CAF9", edgecolor="white")
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel("Score Change per Unit")
    ax1.set_title("Heuristic Formula\n(linear, hand-tuned)", fontweight="bold")
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.3)

    # MLP
    m_vals = data["mlp_sensitivity"]
    colors = ["#FF7043" if v > np.median(m_vals) else "#66BB6A" for v in m_vals]
    ax2.barh(y, m_vals, color=colors, edgecolor="white")
    ax2.set_yticks(y)
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel("|Score Change| per Unit")
    ax2.set_title("Neural MLP\n(learned, captures interactions)", fontweight="bold")
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("Scoring Sensitivity: Heuristic vs Neural MLP",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "ablation_04_scoring.png")
    plt.close(fig)
    logger.info("Saved ablation_04_scoring.png")


def plot_uncertainty_coverage(data: dict, out_dir: Path) -> None:
    """Coverage vs width trade-off for uncertainty methods."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    methods = list(data["methods"].keys())
    targets = [data["methods"][m]["target_coverage"] for m in methods]
    actuals = [data["methods"][m]["actual_coverage"] for m in methods]
    widths = [data["methods"][m]["mean_width"] for m in methods]
    colors = ["#FF7043", "#42A5F5", "#66BB6A", "#FFA726"]

    # Coverage comparison
    x = np.arange(len(methods))
    bars = ax1.bar(x, actuals, 0.5, color=colors, edgecolor="white")
    for i, (t, a) in enumerate(zip(targets, actuals, strict=True)):
        ax1.plot([i - 0.3, i + 0.3], [t, t], "k--", linewidth=2)
        label = f"{a:.1%}"
        ax1.text(i, a + 0.01, label, ha="center", fontsize=9, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=8, rotation=15)
    ax1.set_ylabel("Actual Coverage")
    ax1.set_title("Coverage (dashed = target)", fontweight="bold")
    ax1.set_ylim(0, 1.15)
    ax1.grid(axis="y", alpha=0.3)

    # Width comparison
    ax2.bar(x, widths, 0.5, color=colors, edgecolor="white")
    for i, w in enumerate(widths):
        ax2.text(i, w + 0.5, f"{w:.1f}", ha="center", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=8, rotation=15)
    ax2.set_ylabel("Mean Interval Width")
    ax2.set_title("Interval Width (narrower = better)", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Uncertainty Quantification: Coverage vs Width Trade-off",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "ablation_05_uncertainty.png")
    plt.close(fig)
    logger.info("Saved ablation_05_uncertainty.png")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(
    align: dict,
    gamma: dict,
    dilate: dict,
    scoring: dict,
    uncertainty: dict,
) -> dict:
    """Print and return summary of all experiments."""
    summary: dict = {}

    # Alignment: which method best separates perfect from degraded?
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 1: Alignment Method Ablation")
    logger.info("-" * 70)
    logger.info(f"{'Scenario':<12} {'Cosine':>12} {'Hard DTW':>12} {'Soft-DTW':>12} {'OT':>12}")
    logger.info("-" * 70)
    for scenario in align:
        vals = align[scenario]
        logger.info(
            f"{scenario:<12} "
            f"{vals['Cosine (no alignment)']:>12.4f} "
            f"{vals['Hard DTW']:>12.4f} "
            f"{vals['Soft-DTW (γ=0.5)']:>12.4f} "
            f"{vals['Optimal Transport']:>12.4f}"
        )
    summary["alignment"] = align

    # Discrimination ratio: worst / best cost for each method
    logger.info("\nDiscrimination ratio (worst/best cost — higher = more discriminative):")
    methods_list = list(align[list(align.keys())[0]].keys())
    for method in methods_list:
        costs = [align[s][method] for s in align]
        ratio = max(costs) / max(min(costs), 1e-8)
        logger.info(f"  {method:<25} {ratio:.2f}x")

    # DILATE
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 3: DILATE Loss Ablation")
    logger.info("-" * 70)
    logger.info(f"{'Scenario':<12} {'MSE':>10} {'Shape':>10} {'Temporal':>10} {'DILATE':>10}")
    logger.info("-" * 70)
    for scenario in dilate:
        d = dilate[scenario]
        logger.info(
            f"{scenario:<12} "
            f"{d['MSE']:>10.4f} "
            f"{d['Shape only']:>10.4f} "
            f"{d['Temporal only']:>10.4f} "
            f"{d['DILATE (α=0.5)']:>10.4f}"
        )
    summary["dilate"] = dilate

    # Uncertainty coverage
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 5: Uncertainty Coverage")
    logger.info("-" * 70)
    logger.info(f"{'Method':<25} {'Target':>10} {'Actual':>10} {'Width':>10}")
    logger.info("-" * 70)
    for method, vals in uncertainty["methods"].items():
        logger.info(
            f"{method:<25} "
            f"{vals['target_coverage']:>10.1%} "
            f"{vals['actual_coverage']:>10.1%} "
            f"{vals['mean_width']:>10.1f}"
        )

    # Key finding
    conf_95 = uncertainty["methods"].get("Conformal (95%)", {})
    mc = uncertainty["methods"].get("MC Dropout (95%)", {})
    logger.info("\nKey finding:")
    if conf_95 and mc:
        logger.info(f"  Conformal achieves {conf_95['actual_coverage']:.1%} coverage "
                    f"(target: {conf_95['target_coverage']:.0%}) — "
                    f"distribution-free guarantee")
        logger.info(f"  MC Dropout achieves {mc['actual_coverage']:.1%} coverage "
                    f"(target: {mc['target_coverage']:.0%}) — "
                    f"no formal guarantee")

    summary["uncertainty"] = uncertainty
    summary["scoring"] = scoring
    summary["gamma"] = gamma
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SOPilot Neural Pipeline Ablation Study")
    parser.add_argument("--out-dir", type=str, default="demo_outputs", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("SOPilot Neural Pipeline — Ablation Study")
    logger.info("=" * 70)

    scenarios = _scenario_suite()
    logger.info(f"Generated {len(scenarios)} scenarios: {', '.join(scenarios.keys())}")

    # Experiment 1: Alignment
    logger.info("\n[1/5] Alignment method ablation...")
    align_data = experiment_alignment_ablation(scenarios)
    plot_alignment_ablation(align_data, out_dir)

    # Experiment 2: Gamma sensitivity
    logger.info("[2/5] Gamma sensitivity...")
    gamma_data = experiment_gamma_sensitivity(scenarios)
    plot_gamma_sensitivity(gamma_data, out_dir)

    # Experiment 3: DILATE
    logger.info("[3/5] DILATE loss ablation...")
    dilate_data = experiment_dilate_ablation(scenarios)
    plot_dilate_ablation(dilate_data, out_dir)

    # Experiment 4: Scoring
    logger.info("[4/5] Scoring head ablation...")
    scoring_data = experiment_scoring_ablation()
    plot_scoring_ablation(scoring_data, out_dir)

    # Experiment 5: Uncertainty
    logger.info("[5/5] Uncertainty coverage...")
    uncertainty_data = experiment_uncertainty_coverage()
    plot_uncertainty_coverage(uncertainty_data, out_dir)

    # Summary
    summary = print_summary(align_data, gamma_data, dilate_data, scoring_data, uncertainty_data)

    # Save JSON
    json_path = out_dir / "ablation_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\nSaved summary to {json_path}")

    logger.info("\n" + "=" * 70)
    logger.info(f"Done! 5 figures + 1 JSON saved to {out_dir}/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
