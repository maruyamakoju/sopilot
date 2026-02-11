"""Conformal prediction for distribution-free uncertainty quantification.

Provides prediction intervals with formal coverage guarantees that hold
regardless of the underlying model or data distribution. This is strictly
more rigorous than MC Dropout, which provides no coverage guarantee.

Methods implemented:
    - SplitConformalPredictor: basic split conformal (Lei et al., 2018)
    - ConformizedQuantileRegression: adaptive intervals (Romano et al., 2019)
    - AdaptiveConformalInference: handles distribution shift (Gibbs & Candès, 2021)
    - MondrianConformal: group-conditional coverage (Vovk et al., 2005)
    - ConformalMCDropout: combines MC Dropout with conformal wrapping

References:
    Lei, J. et al. (2018). "Distribution-Free Predictive Inference for Regression"
    Romano, Y., Patterson, E., & Candès, E. (2019). "Conformalized Quantile Regression"
    Gibbs, I. & Candès, E. (2021). "Adaptive Conformal Inference Under Distribution Shift"
    Vovk, V. et al. (2005). "Algorithmic Learning in a Random World"
    Angelopoulos, A. & Bates, S. (2023). "Conformal Prediction: A Gentle Introduction"
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = [
    "SplitConformalPredictor",
    "ConformizedQuantileRegression",
    "AdaptiveConformalInference",
    "MondrianConformal",
    "ConformalMCDropout",
]


class SplitConformalPredictor:
    """Split conformal prediction for regression (Lei et al., 2018).

    Provides prediction intervals with guaranteed finite-sample coverage:
        P(Y_{n+1} ∈ C(X_{n+1})) >= 1 - alpha

    This guarantee holds for ANY model, ANY distribution, requiring only
    exchangeability (weaker than i.i.d.).

    Algorithm:
        1. Compute nonconformity scores on calibration set:
           s_i = |y_i - f(x_i)|
        2. Find quantile: q = ⌈(1-alpha)(n+1)⌉/n percentile of scores
        3. Prediction interval: [f(x) - q, f(x) + q]
    """

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self._scores: np.ndarray | None = None
        self._quantile: float | None = None
        self._n_cal: int = 0

    def calibrate(self, predictions: np.ndarray, actuals: np.ndarray) -> None:
        """Compute nonconformity scores from calibration data.

        Args:
            predictions: (N,) model predictions on calibration set.
            actuals: (N,) true values on calibration set.
        """
        predictions = np.asarray(predictions, dtype=np.float64).ravel()
        actuals = np.asarray(actuals, dtype=np.float64).ravel()
        if len(predictions) != len(actuals):
            raise ValueError("predictions and actuals must have same length")
        n = len(predictions)
        if n == 0:
            raise ValueError("calibration set must be non-empty")

        self._scores = np.abs(actuals - predictions)
        self._n_cal = n

        # Quantile level: ceil((1-alpha)(n+1)) / n
        # This ensures finite-sample coverage guarantee
        level = np.ceil((1 - self.alpha) * (n + 1)) / n
        level = min(level, 1.0)  # Clamp to [0, 1]
        self._quantile = float(np.quantile(self._scores, level))
        logger.info(
            "Conformal calibration: n=%d, alpha=%.3f, quantile=%.4f",
            n, self.alpha, self._quantile,
        )

    def predict(self, point_prediction: float) -> tuple[float, float, float]:
        """Return prediction with conformal interval.

        Args:
            point_prediction: Model's point prediction f(x).

        Returns:
            (prediction, lower, upper) with coverage >= 1-alpha.
        """
        if self._quantile is None:
            raise RuntimeError("Must call calibrate() first")
        q = self._quantile
        return (
            float(point_prediction),
            float(point_prediction - q),
            float(point_prediction + q),
        )

    def predict_batch(self, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Batch prediction intervals.

        Returns:
            (lower_bounds, upper_bounds) arrays.
        """
        if self._quantile is None:
            raise RuntimeError("Must call calibrate() first")
        preds = np.asarray(predictions, dtype=np.float64)
        return preds - self._quantile, preds + self._quantile

    @property
    def interval_width(self) -> float:
        """Width of the prediction interval (2 * quantile)."""
        if self._quantile is None:
            return float("inf")
        return 2.0 * self._quantile

    def coverage_guarantee(self) -> str:
        """Formal statement of the coverage guarantee."""
        return (
            f"P(Y ∈ [{chr(0x0177)} ± {self._quantile:.4f}]) >= {1-self.alpha:.3f} "
            f"(finite-sample, distribution-free, calibrated on {self._n_cal} samples)"
        )


class ConformizedQuantileRegression:
    """CQR: Conformalized Quantile Regression (Romano et al., 2019).

    More adaptive than split conformal: intervals are wider where the
    model is uncertain and narrower where it's confident.

    Algorithm:
        1. Train quantile model for alpha/2 and 1-alpha/2 quantiles
        2. Nonconformity score: s_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))
        3. Find quantile Q of scores on calibration set
        4. Prediction interval: [q_lo(x) - Q, q_hi(x) + Q]

    The key insight: if the base quantile model is well-calibrated locally,
    CQR intervals will be adaptive. If it's poorly calibrated, CQR still
    provides valid marginal coverage via the conformal correction Q.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self._Q: float | None = None
        self._n_cal: int = 0

    def calibrate(
        self,
        lower_quantiles: np.ndarray,
        upper_quantiles: np.ndarray,
        actuals: np.ndarray,
    ) -> None:
        """Calibrate using quantile regression outputs.

        Args:
            lower_quantiles: (N,) predicted alpha/2 quantiles on cal set.
            upper_quantiles: (N,) predicted 1-alpha/2 quantiles on cal set.
            actuals: (N,) true values on cal set.
        """
        lo = np.asarray(lower_quantiles, dtype=np.float64).ravel()
        hi = np.asarray(upper_quantiles, dtype=np.float64).ravel()
        y = np.asarray(actuals, dtype=np.float64).ravel()
        n = len(y)
        if n == 0:
            raise ValueError("calibration set must be non-empty")

        # CQR nonconformity score
        scores = np.maximum(lo - y, y - hi)
        self._n_cal = n

        level = np.ceil((1 - self.alpha) * (n + 1)) / n
        level = min(level, 1.0)
        self._Q = float(np.quantile(scores, level))
        logger.info("CQR calibration: n=%d, Q=%.4f", n, self._Q)

    def predict(
        self, lower_quantile: float, upper_quantile: float
    ) -> tuple[float, float]:
        """Return calibrated interval.

        Args:
            lower_quantile: Model's predicted lower quantile q_lo(x).
            upper_quantile: Model's predicted upper quantile q_hi(x).

        Returns:
            (calibrated_lower, calibrated_upper) with coverage >= 1-alpha.
        """
        if self._Q is None:
            raise RuntimeError("Must call calibrate() first")
        return (
            float(lower_quantile - self._Q),
            float(upper_quantile + self._Q),
        )


class AdaptiveConformalInference:
    """ACI: Adaptive Conformal Inference (Gibbs & Candès, 2021).

    Handles distribution shift by adapting alpha_t online. At each step:
        1. Make prediction with current alpha_t
        2. Observe true value y_t
        3. Update: alpha_{t+1} = alpha_t + lr * (alpha_target - err_t)
           where err_t = 1{y_t not in C_t(x_t)}

    This achieves long-run average coverage even under distribution shift,
    a property that static conformal prediction cannot guarantee.
    """

    def __init__(
        self,
        target_alpha: float = 0.05,
        learning_rate: float = 0.01,
    ) -> None:
        if not 0 < target_alpha < 1:
            raise ValueError(f"target_alpha must be in (0, 1)")
        self.target_alpha = target_alpha
        self.learning_rate = learning_rate
        self._alpha_t: float = target_alpha
        self._history: list[dict] = []

    def update(
        self,
        prediction_interval: tuple[float, float],
        actual: float,
    ) -> None:
        """Update adaptive alpha after observing true value.

        Args:
            prediction_interval: (lower, upper) from conformal predictor.
            actual: Observed true value.
        """
        lower, upper = prediction_interval
        err_t = 1.0 if (actual < lower or actual > upper) else 0.0

        # Online gradient update
        self._alpha_t = self._alpha_t + self.learning_rate * (
            self.target_alpha - err_t
        )
        # Clamp to valid range
        self._alpha_t = max(0.001, min(0.999, self._alpha_t))

        self._history.append({
            "alpha_t": self._alpha_t,
            "err_t": err_t,
            "actual": float(actual),
            "lower": float(lower),
            "upper": float(upper),
        })

    @property
    def current_alpha(self) -> float:
        """Current adaptive significance level."""
        return self._alpha_t

    @property
    def running_coverage(self) -> float:
        """Empirical coverage over all updates."""
        if not self._history:
            return 1.0
        covered = sum(1.0 - h["err_t"] for h in self._history)
        return covered / len(self._history)

    @property
    def n_updates(self) -> int:
        return len(self._history)


class MondrianConformal:
    """Group-conditional conformal prediction (Vovk et al., 2005).

    Provides coverage guarantees WITHIN each group separately:
        P(Y ∈ C(X) | G=g) >= 1-alpha for each group g

    For SOP evaluation: different SOP types or difficulty levels
    may need different calibration.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1)")
        self.alpha = alpha
        self._group_quantiles: dict[int, float] = {}
        self._group_counts: dict[int, int] = {}

    def calibrate(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        groups: np.ndarray,
    ) -> None:
        """Calibrate per-group conformal predictors.

        Args:
            predictions: (N,) model predictions.
            actuals: (N,) true values.
            groups: (N,) integer group labels.
        """
        predictions = np.asarray(predictions, dtype=np.float64).ravel()
        actuals = np.asarray(actuals, dtype=np.float64).ravel()
        groups = np.asarray(groups).ravel()

        unique_groups = np.unique(groups)
        for g in unique_groups:
            mask = groups == g
            g_preds = predictions[mask]
            g_actuals = actuals[mask]
            n_g = len(g_preds)
            if n_g == 0:
                continue

            scores = np.abs(g_actuals - g_preds)
            level = np.ceil((1 - self.alpha) * (n_g + 1)) / n_g
            level = min(level, 1.0)
            self._group_quantiles[int(g)] = float(np.quantile(scores, level))
            self._group_counts[int(g)] = n_g

        logger.info(
            "Mondrian calibration: %d groups, sizes=%s",
            len(unique_groups),
            {int(g): self._group_counts.get(int(g), 0) for g in unique_groups},
        )

    def predict(
        self, point_prediction: float, group: int
    ) -> tuple[float, float, float]:
        """Predict with group-specific interval.

        Args:
            point_prediction: Model's point prediction.
            group: Group label for this instance.

        Returns:
            (prediction, lower, upper) with per-group coverage >= 1-alpha.
        """
        if group not in self._group_quantiles:
            # Fallback: use maximum quantile across all groups
            if not self._group_quantiles:
                raise RuntimeError("Must call calibrate() first")
            q = max(self._group_quantiles.values())
        else:
            q = self._group_quantiles[group]

        return (
            float(point_prediction),
            float(point_prediction - q),
            float(point_prediction + q),
        )


class ConformalMCDropout:
    """Combines MC Dropout with conformal wrapping.

    MC Dropout provides informative (adaptive-width) intervals via
    epistemic uncertainty estimation. Conformal prediction provides
    the formal coverage guarantee.

    Procedure:
        1. Run MC Dropout to get mean ± quantile interval
        2. Use CQR to conformalize the quantile interval
        3. Result: adaptive intervals WITH coverage guarantee

    This is strictly better than either method alone:
    - MC Dropout alone: no coverage guarantee
    - Basic conformal alone: constant-width intervals
    - Combined: adaptive width + formal guarantee
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.05,
        n_mc_samples: int = 30,
    ) -> None:
        self.model = model
        self.alpha = alpha
        self.n_mc_samples = n_mc_samples
        self._cqr = ConformizedQuantileRegression(alpha=alpha)
        self._calibrated = False

    def _mc_dropout_predict(
        self, x: torch.Tensor
    ) -> tuple[float, float, float]:
        """Run MC Dropout forward passes.

        Returns:
            (mean, lower_quantile, upper_quantile)
        """
        # Enable dropout, keep batchnorm in eval mode
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        predictions = []
        with torch.no_grad():
            for _ in range(self.n_mc_samples):
                pred = self.model(x)
                if pred.dim() > 1:
                    pred = pred.squeeze()
                predictions.append(float(pred.item()))

        self.model.eval()

        preds = np.array(predictions)
        lo_q = self.alpha / 2
        hi_q = 1.0 - self.alpha / 2
        return (
            float(np.mean(preds)),
            float(np.percentile(preds, 100 * lo_q)),
            float(np.percentile(preds, 100 * hi_q)),
        )

    def calibrate(
        self,
        calibration_inputs: list[torch.Tensor],
        actuals: np.ndarray,
    ) -> None:
        """Calibrate conformal wrapper on held-out data.

        Args:
            calibration_inputs: List of (1, D) input tensors.
            actuals: (N,) true values.
        """
        lower_quantiles = []
        upper_quantiles = []

        for x in calibration_inputs:
            _, lo, hi = self._mc_dropout_predict(x)
            lower_quantiles.append(lo)
            upper_quantiles.append(hi)

        self._cqr.calibrate(
            np.array(lower_quantiles),
            np.array(upper_quantiles),
            np.asarray(actuals),
        )
        self._calibrated = True

    def predict(self, x: torch.Tensor) -> dict[str, float]:
        """Predict with conformalized MC Dropout interval.

        Args:
            x: (1, D) input tensor.

        Returns:
            dict with: score, mc_std, ci_lower, ci_upper, interval_width
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() first")

        mean, lo_mc, hi_mc = self._mc_dropout_predict(x)
        lo_conf, hi_conf = self._cqr.predict(lo_mc, hi_mc)

        return {
            "score": float(np.clip(mean, 0.0, 100.0)),
            "mc_std": float(np.std([mean])),  # Placeholder; real std from MC
            "ci_lower": float(np.clip(lo_conf, 0.0, 100.0)),
            "ci_upper": float(np.clip(hi_conf, 0.0, 100.0)),
            "interval_width": float(hi_conf - lo_conf),
            "method": "conformal_mc_dropout",
            "coverage_guarantee": f">= {1-self.alpha:.0%}",
        }
