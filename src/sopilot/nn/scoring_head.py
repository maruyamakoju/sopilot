"""Learned Scoring Head — replaces fixed penalty formula.

Implements an MLP that maps the 15 existing penalty metrics to a [0, 100] score,
with MC Dropout uncertainty estimation (Gal & Ghahramani 2016) and Isotonic
Regression calibration (Niculescu-Mizil & Caruana 2005).

Architecture:
    Linear(15, 64) -> BatchNorm1d -> ReLU -> Dropout(0.2)
    Linear(64, 32)  -> BatchNorm1d -> ReLU -> Dropout(0.2)
    Linear(32, 1)   -> Sigmoid -> * 100
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# The 15 metrics from evaluate_sop() in order
METRIC_KEYS = [
    "miss",
    "swap",
    "deviation",
    "over_time",
    "temporal_warp",
    "path_stretch",
    "duplicate_ratio",
    "order_violation_ratio",
    "temporal_drift",
    "confidence_loss",
    "local_similarity_gap",
    "adaptive_low_similarity_threshold",
    "effective_low_similarity_threshold",
    "hard_miss_ratio",
    "mean_alignment_cost",
]

N_METRICS = len(METRIC_KEYS)


class ScoringHead(nn.Module):
    """MLP scoring head: maps 15 metrics to [0, 100] score."""

    def __init__(
        self,
        n_inputs: int = N_METRICS,
        d_hidden1: int = 64,
        d_hidden2: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_inputs = n_inputs
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, d_hidden1),
            nn.BatchNorm1d(d_hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden1, d_hidden2),
            nn.BatchNorm1d(d_hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden2, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map metrics to score.

        Args:
            x: (B, 15) raw metric values.

        Returns:
            (B, 1) scores in [0, 100].
        """
        return self.layers(x) * 100.0

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 30,
    ) -> dict[str, float]:
        """MC Dropout uncertainty estimation.

        Runs n_samples forward passes with dropout enabled at inference
        to estimate epistemic uncertainty.

        Args:
            x: (1, 15) single input metrics vector.
            n_samples: Number of stochastic forward passes.

        Returns:
            dict with: score, uncertainty, ci_lower, ci_upper (95% CI).
        """
        # Enable dropout but keep BatchNorm in eval mode (avoids batch_size=1 error)
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self._forward_single(x)
                predictions.append(pred)
        self.eval()

        preds = np.array(predictions)

        # Guard against NaN predictions (corrupted weights, degenerate inputs)
        valid = preds[~np.isnan(preds)]
        if len(valid) == 0:
            return {
                "score": 50.0,
                "uncertainty": float("inf"),
                "ci_lower": 0.0,
                "ci_upper": 100.0,
            }

        mean = float(np.mean(valid))
        std = float(np.std(valid))

        # 95% confidence interval
        ci_lower = float(np.percentile(valid, 2.5))
        ci_upper = float(np.percentile(valid, 97.5))

        return {
            "score": np.clip(mean, 0.0, 100.0),
            "uncertainty": std,
            "ci_lower": np.clip(ci_lower, 0.0, 100.0),
            "ci_upper": np.clip(ci_upper, 0.0, 100.0),
        }

    def _forward_single(self, x: torch.Tensor) -> float:
        """Forward pass safe for batch_size=1 (runs BatchNorm in eval mode)."""
        out = x
        for module in self.layers:
            if isinstance(module, nn.BatchNorm1d):
                module.eval()  # Use running stats, not batch stats
            out = module(out)
        return float((out * 100.0).item())

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Isotonic calibration
# ---------------------------------------------------------------------------


class IsotonicCalibrator:
    """Isotonic regression calibrator for predicted scores.

    Ensures that predicted scores are well-calibrated: a predicted score of 80
    should correspond to actual performance near 80 on the calibration set.
    """

    def __init__(self) -> None:
        self._fitted = False
        self._x_sorted: np.ndarray | None = None
        self._y_sorted: np.ndarray | None = None

    def fit(self, predicted: np.ndarray, actual: np.ndarray) -> None:
        """Fit isotonic regression from predicted → actual scores.

        Uses Pool Adjacent Violators Algorithm (PAVA).

        Args:
            predicted: (N,) predicted scores from ScoringHead.
            actual: (N,) ground-truth / human-annotated scores.
        """
        order = np.argsort(predicted)
        x = predicted[order].astype(np.float64)
        y = actual[order].astype(np.float64)

        # PAVA (Pool Adjacent Violators Algorithm)
        n = len(y)
        blocks_start = list(range(n))
        blocks_size = [1] * n
        blocks_sum = y.tolist()

        i = 0
        while i < len(blocks_start) - 1:
            mean_curr = blocks_sum[i] / blocks_size[i]
            mean_next = blocks_sum[i + 1] / blocks_size[i + 1]
            if mean_curr > mean_next:
                # Pool blocks
                blocks_sum[i] += blocks_sum[i + 1]
                blocks_size[i] += blocks_size[i + 1]
                del blocks_start[i + 1]
                del blocks_size[i + 1]
                del blocks_sum[i + 1]
                # Check backward
                if i > 0:
                    i -= 1
            else:
                i += 1

        # Build fitted values
        fitted = np.zeros(n, dtype=np.float64)
        pos = 0
        for k in range(len(blocks_start)):
            mean_val = blocks_sum[k] / blocks_size[k]
            for _ in range(blocks_size[k]):
                fitted[pos] = mean_val
                pos += 1

        self._x_sorted = x
        self._y_sorted = fitted
        self._fitted = True
        logger.info("Fitted IsotonicCalibrator on %d samples", n)

    def calibrate(self, score: float) -> float:
        """Calibrate a single predicted score.

        Args:
            score: Raw predicted score.

        Returns:
            Calibrated score.
        """
        if not self._fitted or self._x_sorted is None or self._y_sorted is None:
            return score

        # Linear interpolation
        idx = np.searchsorted(self._x_sorted, score)
        if idx == 0:
            return float(self._y_sorted[0])
        if idx >= len(self._x_sorted):
            return float(self._y_sorted[-1])

        # Interpolate between adjacent points
        x0, x1 = self._x_sorted[idx - 1], self._x_sorted[idx]
        y0, y1 = self._y_sorted[idx - 1], self._y_sorted[idx]
        if abs(x1 - x0) < 1e-12:
            return float(y0)
        t = (score - x0) / (x1 - x0)
        return float(y0 + t * (y1 - y0))

    def calibrate_batch(self, scores: np.ndarray) -> np.ndarray:
        """Calibrate an array of scores."""
        return np.array([self.calibrate(float(s)) for s in scores])

    def save(self, path: Path) -> None:
        """Save calibrator to npz."""
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, x=self._x_sorted, y=self._y_sorted)
        logger.info("Saved IsotonicCalibrator to %s", path)

    @classmethod
    def load(cls, path: Path) -> IsotonicCalibrator:
        """Load calibrator from npz."""
        data = np.load(path)
        cal = cls()
        cal._x_sorted = data["x"]
        cal._y_sorted = data["y"]
        cal._fitted = True
        logger.info("Loaded IsotonicCalibrator from %s", path)
        return cal


# ---------------------------------------------------------------------------
# Utility: extract metrics vector from evaluate_sop() result
# ---------------------------------------------------------------------------


def metrics_to_tensor(metrics: dict, device: str = "cpu") -> torch.Tensor:
    """Convert evaluate_sop() metrics dict to (1, 15) tensor.

    Args:
        metrics: The 'metrics' dict from evaluate_sop() output.

    Returns:
        (1, 15) float tensor in METRIC_KEYS order.
    """
    values = [float(metrics.get(k, 0.0)) for k in METRIC_KEYS]
    return torch.tensor([values], dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Save / load scoring head
# ---------------------------------------------------------------------------


def save_scoring_head(model: ScoringHead, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"n_inputs": model.n_inputs, "state_dict": model.state_dict()},
        path,
    )
    logger.info("Saved ScoringHead (%d params) to %s", model.num_parameters, path)


def load_scoring_head(path: Path, device: str = "cpu") -> ScoringHead:
    data = torch.load(path, map_location=device, weights_only=True)
    model = ScoringHead(n_inputs=data["n_inputs"])
    model.load_state_dict(data["state_dict"])
    model.to(device)
    model.eval()
    logger.info("Loaded ScoringHead from %s", path)
    return model
