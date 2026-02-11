"""Explainability module for SOPilot neural pipeline.

Provides:
- TemporalAttentionVisualizer: extracts and visualizes soft-DTW alignment heatmaps
- IntegratedGradientsExplainer: Integrated Gradients (Sundararajan, Taly, Yan, ICML 2017)
- WachterCounterfactualExplainer: Wachter et al. (2017) counterfactual explanations
- CounterfactualExplainer: gradient-based "what-if" analysis for actionable feedback

References:
    - Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks" (ICML)
    - Wachter et al. (2017) "Counterfactual Explanations Without Opening the Black Box"
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TemporalAttentionVisualizer:
    """Visualize temporal alignment from Soft-DTW alignment matrices.

    Extracts per-step importance scores and generates interpretable
    heatmaps showing which trainee frames align to which gold frames.
    """

    @staticmethod
    def alignment_heatmap(alignment_matrix: np.ndarray) -> dict:
        """Extract visualization data from soft alignment matrix.

        Args:
            alignment_matrix: (M, N) soft alignment from SoftDTWAlignment.

        Returns:
            dict with:
                - heatmap: (M, N) normalized alignment matrix
                - gold_importance: (M,) how much each gold frame contributes
                - trainee_importance: (N,) how much each trainee frame contributes
                - peak_alignment: list of (gold_idx, trainee_idx, strength) top alignments
        """
        m, n = alignment_matrix.shape

        # Normalize to [0, 1]
        max_val = alignment_matrix.max()
        heatmap = alignment_matrix / max(max_val, 1e-8)

        # Per-frame importance
        gold_importance = alignment_matrix.sum(axis=1)
        gold_importance = gold_importance / max(gold_importance.max(), 1e-8)
        trainee_importance = alignment_matrix.sum(axis=0)
        trainee_importance = trainee_importance / max(trainee_importance.max(), 1e-8)

        # Peak alignments (top K strongest cells)
        k = min(50, m * n)
        flat_indices = np.argpartition(alignment_matrix.ravel(), -k)[-k:]
        flat_indices = flat_indices[np.argsort(-alignment_matrix.ravel()[flat_indices])]

        peak_alignment = []
        for idx in flat_indices:
            gi = int(idx // n)
            tj = int(idx % n)
            strength = float(alignment_matrix[gi, tj])
            if strength > 1e-4:
                peak_alignment.append((gi, tj, strength))

        return {
            "heatmap": heatmap,
            "gold_importance": gold_importance.tolist(),
            "trainee_importance": trainee_importance.tolist(),
            "peak_alignment": peak_alignment,
        }

    @staticmethod
    def step_alignment_summary(
        alignment_matrix: np.ndarray,
        gold_boundaries: list[int],
        trainee_boundaries: list[int],
    ) -> list[dict]:
        """Summarize alignment strength per gold step.

        Args:
            alignment_matrix: (M, N) soft alignment.
            gold_boundaries: Gold step boundaries [0, b1, ..., M].
            trainee_boundaries: Trainee step boundaries [0, b1, ..., N].

        Returns:
            List of dicts, one per gold step, with alignment statistics.
        """
        summaries = []
        for k in range(len(gold_boundaries) - 1):
            g_start = gold_boundaries[k]
            g_end = gold_boundaries[k + 1]

            step_alignment = alignment_matrix[g_start:g_end, :]
            total_mass = float(step_alignment.sum())
            mean_strength = float(step_alignment.mean()) if step_alignment.size > 0 else 0.0

            # Which trainee step gets the most alignment mass
            trainee_mass = step_alignment.sum(axis=0)
            best_trainee_frame = int(np.argmax(trainee_mass))

            # Map to trainee step
            best_trainee_step = 0
            for ts in range(len(trainee_boundaries) - 1):
                if trainee_boundaries[ts] <= best_trainee_frame < trainee_boundaries[ts + 1]:
                    best_trainee_step = ts
                    break

            summaries.append({
                "gold_step": k,
                "gold_frames": (g_start, g_end),
                "alignment_mass": round(total_mass, 4),
                "mean_strength": round(mean_strength, 4),
                "best_trainee_step": best_trainee_step,
                "best_trainee_frame": best_trainee_frame,
            })

        return summaries


class IntegratedGradientsExplainer:
    """Integrated Gradients (Sundararajan, Taly, Yan, ICML 2017).

    Computes attribution by integrating gradients along a straight-line
    path from a baseline to the input. Satisfies two key axioms:

    - Completeness: attributions sum to f(x) - f(baseline)
    - Sensitivity: if a feature differs between x and baseline and
      changes the output, it receives non-zero attribution

    This is a theoretically grounded attribution method, unlike simple
    gradient magnitude which violates the sensitivity axiom.
    """

    def __init__(self, model: nn.Module, n_steps: int = 50) -> None:
        self.model = model
        self.n_steps = n_steps

    def attribute(
        self, x: torch.Tensor, baseline: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute integrated gradients attributions.

        Approximated with Riemann sum over n_steps.

        Args:
            x: Input tensor (any shape accepted by the model).
            baseline: Reference input. Defaults to zeros.

        Returns:
            Attribution tensor, same shape as x.
        """
        if baseline is None:
            baseline = torch.zeros_like(x)

        accumulated_grads = torch.zeros_like(x)
        for step in range(self.n_steps):
            alpha = (step + 0.5) / self.n_steps
            interpolated = baseline + alpha * (x - baseline)
            interpolated = interpolated.clone().detach().requires_grad_(True)

            output = self.model(interpolated)
            if output.dim() > 0:
                output = output.sum()
            output.backward()

            if interpolated.grad is not None:
                accumulated_grads += interpolated.grad

        attributions = (x - baseline) * accumulated_grads / self.n_steps
        return attributions.detach()


class WachterCounterfactualExplainer:
    """Wachter et al. (2017) counterfactual explanations.

    Finds the minimal perturbation to the input that changes the model
    output to a target value, solving an optimization problem.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def find_counterfactual(
        self,
        x: torch.Tensor,
        target_score: float,
        lambda_: float = 1.0,
        lr: float = 0.01,
        n_steps: int = 100,
    ) -> dict:
        """Find a counterfactual explanation.

        Args:
            x: (1, D) input features.
            target_score: Desired model output.
            lambda_: Weight on the target loss vs proximity.
            lr: Learning rate for the optimizer.
            n_steps: Number of optimization steps.

        Returns:
            dict with counterfactual results.
        """
        self.model.eval()
        x_prime = x.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x_prime], lr=lr)
        target_tensor = torch.tensor(
            [[target_score]], dtype=x.dtype, device=x.device
        )

        for _ in range(n_steps):
            optimizer.zero_grad()
            pred = self.model(x_prime)
            proximity_loss = torch.sum((x_prime - x.detach()) ** 2)
            target_loss = torch.sum((pred - target_tensor) ** 2)
            loss = proximity_loss + lambda_ * target_loss
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            achieved = float(self.model(x_prime).item())

        perturbation = (x_prime - x).detach()
        l2_dist = float(torch.sqrt(torch.sum(perturbation ** 2)).item())

        return {
            "original_input": x.detach().cpu().numpy(),
            "counterfactual": x_prime.detach().cpu().numpy(),
            "perturbation": perturbation.cpu().numpy(),
            "achieved_score": round(achieved, 2),
            "target_score": target_score,
            "l2_distance": round(l2_dist, 4),
        }


class CounterfactualExplainer:
    """Gradient-based counterfactual explanations for scores.

    Answers: "If metric X improved by delta, how would the score change?"
    Uses the ScoringHead's gradient w.r.t. input metrics for local sensitivity.
    """

    def __init__(self, scoring_head: torch.nn.Module) -> None:
        self.scoring_head = scoring_head

    def compute_sensitivity(
        self, metrics_tensor: torch.Tensor
    ) -> dict[str, float]:
        """Compute per-metric sensitivity (partial derivative of score).

        Args:
            metrics_tensor: (1, 15) input metrics.

        Returns:
            dict mapping metric_name → ∂score/∂metric.
        """
        from .scoring_head import METRIC_KEYS

        x = metrics_tensor.clone().detach().requires_grad_(True)
        self.scoring_head.eval()
        score = self.scoring_head(x)
        score.backward()

        grad = x.grad[0].detach().cpu().numpy()
        return {k: float(grad[i]) for i, k in enumerate(METRIC_KEYS)}

    def counterfactual(
        self,
        metrics_tensor: torch.Tensor,
        metric_name: str,
        target_improvement: float,
    ) -> dict:
        """Estimate score change from improving a single metric.

        Args:
            metrics_tensor: (1, 15) current metrics.
            metric_name: Which metric to change.
            target_improvement: Desired improvement in metric value
                (negative = lower value, since most metrics are penalties).

        Returns:
            dict with: current_score, predicted_score, delta, metric_name,
                       metric_change.
        """
        from .scoring_head import METRIC_KEYS

        if metric_name not in METRIC_KEYS:
            raise ValueError(f"Unknown metric: {metric_name}")

        idx = METRIC_KEYS.index(metric_name)

        self.scoring_head.eval()
        with torch.no_grad():
            current_score = float(self.scoring_head(metrics_tensor).item())

        modified = metrics_tensor.clone()
        modified[0, idx] += target_improvement

        with torch.no_grad():
            predicted_score = float(self.scoring_head(modified).item())

        return {
            "current_score": round(current_score, 2),
            "predicted_score": round(predicted_score, 2),
            "delta": round(predicted_score - current_score, 2),
            "metric_name": metric_name,
            "metric_change": target_improvement,
        }

    def top_actionable_improvements(
        self,
        metrics_tensor: torch.Tensor,
        n_top: int = 5,
    ) -> list[dict]:
        """Find the top N most impactful metric improvements.

        For each metric, estimates how much improving it by a small amount
        (reducing penalties) would increase the score.

        Args:
            metrics_tensor: (1, 15) current metrics.
            n_top: Number of top improvements to return.

        Returns:
            List of dicts sorted by potential score gain (descending).
        """
        sensitivity = self.compute_sensitivity(metrics_tensor)

        improvements = []
        for name, grad in sensitivity.items():
            # Negative gradient means reducing this metric increases score
            potential_gain = abs(grad)
            direction = "decrease" if grad < 0 else "increase"
            improvements.append({
                "metric": name,
                "sensitivity": round(grad, 4),
                "direction_to_improve": direction,
                "potential_gain_per_unit": round(potential_gain, 4),
            })

        improvements.sort(key=lambda x: x["potential_gain_per_unit"], reverse=True)
        return improvements[:n_top]
