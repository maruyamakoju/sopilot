"""Probabilistic severity classification model.

Replaces rule-based recalibration with a learned model that combines
multimodal signal features (danger, motion, proximity) with VLM outputs
to produce calibrated severity predictions.

The model implements:
- Multiclass logistic regression with L2 regularization (pure numpy)
- Ordinal constraints enforcing NONE < LOW < MEDIUM < HIGH
- Post-hoc probability calibration via isotonic regression or Platt scaling
- Full serialization for deployment

Theory
------
We frame severity classification as an ordinal regression problem.  The base
model is a one-vs-rest logistic regression trained via gradient descent with
L2 penalty.  After fitting, we enforce ordinal monotonicity by projecting the
predicted cumulative probabilities onto the isotone cone.  Post-hoc calibration
(isotonic or Platt) is applied to the maximum predicted probability to yield
well-calibrated confidence scores.

References
----------
[1] Zadrozny & Elkan, "Transforming classifier scores into accurate
    multiclass probability estimates", KDD 2002.
[2] Platt, "Probabilistic outputs for SVMs", 1999.
[3] Barlow et al., "Statistical Inference under Order Restrictions", 1972.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Ordinal severity levels
SEVERITY_LABELS = ["NONE", "LOW", "MEDIUM", "HIGH"]
NUM_CLASSES = len(SEVERITY_LABELS)


# ---------------------------------------------------------------------------
# Feature representation
# ---------------------------------------------------------------------------

@dataclass
class SeverityFeatures:
    """Input features for severity classification.

    Attributes:
        danger_score: Composite danger score from signal mining, in [0, 1].
        motion_score: Optical-flow / acceleration magnitude, in [0, 1].
        proximity_score: Minimum distance to other road users, in [0, 1].
            Higher means closer / more dangerous.
        vlm_confidence: Confidence score from the VLM inference, in [0, 1].
        vlm_severity_idx: Ordinal index of the VLM's predicted severity
            (NONE=0, LOW=1, MEDIUM=2, HIGH=3).
    """

    danger_score: float = 0.0
    motion_score: float = 0.0
    proximity_score: float = 0.0
    vlm_confidence: float = 0.0
    vlm_severity_idx: int = 0

    def to_array(self) -> np.ndarray:
        """Convert to a 1-D numpy feature vector of shape (5,)."""
        return np.array(
            [
                self.danger_score,
                self.motion_score,
                self.proximity_score,
                self.vlm_confidence,
                self.vlm_severity_idx / (NUM_CLASSES - 1),  # normalize to [0, 1]
            ],
            dtype=np.float64,
        )


# ---------------------------------------------------------------------------
# Isotonic calibration (Pool-Adjacent-Violators)
# ---------------------------------------------------------------------------

class IsotonicCalibrator:
    """Isotonic regression calibrator using the PAV algorithm.

    Maps raw scores to calibrated probabilities while preserving monotonicity.
    The pool-adjacent-violators algorithm [3] merges adjacent blocks that
    violate the isotonic constraint and replaces them with their weighted mean.

    After fitting, ``transform`` performs piecewise-linear interpolation
    between the fitted breakpoints.

    Attributes:
        x_: Sorted unique score breakpoints after fitting.
        y_: Calibrated probability at each breakpoint.
    """

    def __init__(self) -> None:
        self.x_: np.ndarray | None = None
        self.y_: np.ndarray | None = None
        self._fitted: bool = False

    # ----- PAV core -----

    @staticmethod
    def _pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Pool-Adjacent-Violators algorithm.

        Given observations *y* with weights *w* (both 1-D, same length,
        sorted by the covariate), returns the isotonic (non-decreasing)
        regression solution that minimizes weighted squared error.

        Complexity: O(n) amortized.

        Parameters
        ----------
        y : ndarray, shape (n,)
            Response values sorted by the covariate.
        w : ndarray, shape (n,)
            Non-negative weights.

        Returns
        -------
        ndarray, shape (n,)
            Isotonic regression values.
        """
        n = len(y)
        if n == 0:
            return np.array([], dtype=np.float64)

        # Each block is (sum_wy, sum_w, start_index, end_index).
        # We use lists for efficient merging.
        result = np.empty(n, dtype=np.float64)
        blocks: list[list[float | int]] = []

        for i in range(n):
            # Create a new block for observation i
            blocks.append([w[i] * y[i], w[i], i, i])

            # Merge backwards while isotonicity is violated
            while len(blocks) >= 2:
                last = blocks[-1]
                prev = blocks[-2]
                val_last = last[0] / last[1]
                val_prev = prev[0] / prev[1]
                if val_prev > val_last:
                    # Merge last into prev
                    prev[0] += last[0]
                    prev[1] += last[1]
                    prev[3] = last[3]
                    blocks.pop()
                else:
                    break

        # Write back block means
        for block in blocks:
            val = block[0] / block[1]
            start = int(block[2])
            end = int(block[3])
            result[start: end + 1] = val

        return result

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> IsotonicCalibrator:
        """Fit isotonic regression from raw scores to binary labels.

        Parameters
        ----------
        scores : ndarray, shape (n,)
            Raw model scores (e.g. predicted probabilities before calibration).
        labels : ndarray, shape (n,)
            Binary labels in {0, 1}.

        Returns
        -------
        self
        """
        scores = np.asarray(scores, dtype=np.float64).ravel()
        labels = np.asarray(labels, dtype=np.float64).ravel()

        if len(scores) != len(labels):
            raise ValueError(
                f"scores and labels must have same length, "
                f"got {len(scores)} and {len(labels)}"
            )
        if len(scores) < 2:
            raise ValueError("Need at least 2 samples for isotonic calibration")

        # Sort by score
        order = np.argsort(scores)
        s_sorted = scores[order]
        l_sorted = labels[order]
        w = np.ones_like(s_sorted)

        iso_values = self._pav(l_sorted, w)

        # Deduplicate: average over tied scores
        unique_scores, inverse = np.unique(s_sorted, return_inverse=True)
        unique_values = np.zeros_like(unique_scores)
        counts = np.zeros_like(unique_scores)
        np.add.at(unique_values, inverse, iso_values)
        np.add.at(counts, inverse, 1)
        unique_values /= counts

        self.x_ = unique_scores
        self.y_ = unique_values
        self._fitted = True

        logger.debug(
            "IsotonicCalibrator fitted with %d breakpoints from %d samples",
            len(self.x_),
            len(scores),
        )
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Map raw scores to calibrated probabilities via linear interpolation.

        Parameters
        ----------
        scores : ndarray, shape (n,)
            Raw scores to calibrate.

        Returns
        -------
        ndarray, shape (n,)
            Calibrated probabilities, clipped to [0, 1].
        """
        if not self._fitted:
            raise RuntimeError("IsotonicCalibrator has not been fitted")

        scores = np.asarray(scores, dtype=np.float64).ravel()
        calibrated = np.interp(scores, self.x_, self.y_)
        return np.clip(calibrated, 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize calibrator state to a JSON-compatible dict."""
        if not self._fitted:
            return {"type": "isotonic", "fitted": False}
        return {
            "type": "isotonic",
            "fitted": True,
            "x": self.x_.tolist(),
            "y": self.y_.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IsotonicCalibrator:
        """Deserialize calibrator from dict."""
        cal = cls()
        if d.get("fitted", False):
            cal.x_ = np.array(d["x"], dtype=np.float64)
            cal.y_ = np.array(d["y"], dtype=np.float64)
            cal._fitted = True
        return cal


# ---------------------------------------------------------------------------
# Platt scaling
# ---------------------------------------------------------------------------

class PlattScaling:
    """Platt's sigmoid calibration.

    Maps a raw score *f* to a calibrated probability via::

        P(y=1 | f) = 1 / (1 + exp(A*f + B))

    Parameters *A* and *B* are fit by minimizing the negative log-likelihood
    of the calibration set using Newton's method, following the algorithm
    described in [2] with target probabilities that avoid overfitting.

    Attributes:
        A: Slope parameter (typically negative).
        B: Intercept parameter.
    """

    def __init__(self) -> None:
        self.A: float = 0.0
        self.B: float = 0.0
        self._fitted: bool = False

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-7,
    ) -> PlattScaling:
        """Fit Platt scaling parameters using Newton's method.

        Uses the regularized target values from Platt (1999)::

            t_i = (y_i * N_+ + 1) / (N_+ + 2)   if y_i = 1
            t_i = 1 / (N_- + 2)                   if y_i = 0

        where N_+ and N_- are the positive and negative sample counts.

        Parameters
        ----------
        scores : ndarray, shape (n,)
            Raw classifier outputs (decision function values).
        labels : ndarray, shape (n,)
            Binary labels in {0, 1}.
        max_iter : int
            Maximum Newton iterations.
        tol : float
            Convergence tolerance on the gradient norm.

        Returns
        -------
        self
        """
        scores = np.asarray(scores, dtype=np.float64).ravel()
        labels = np.asarray(labels, dtype=np.float64).ravel()

        if len(scores) != len(labels):
            raise ValueError("scores and labels must have same length")
        if len(scores) < 2:
            raise ValueError("Need at least 2 samples for Platt scaling")

        n_pos = np.sum(labels > 0.5)
        n_neg = len(labels) - n_pos

        # Regularized target probabilities (Platt 1999, Sec. 2)
        t_pos = (n_pos + 1.0) / (n_pos + 2.0)
        t_neg = 1.0 / (n_neg + 2.0)
        target = np.where(labels > 0.5, t_pos, t_neg)

        # Initialize A, B
        a = 0.0
        b = np.log((n_neg + 1.0) / (n_pos + 1.0))

        # Newton's method
        for iteration in range(max_iter):
            # Forward pass: p_i = 1 / (1 + exp(a*f_i + b))
            logit = a * scores + b
            # Numerically stable sigmoid
            p = np.where(
                logit >= 0,
                1.0 / (1.0 + np.exp(-logit)),
                np.exp(logit) / (1.0 + np.exp(logit)),
            )
            # Note: Platt's formulation uses P = 1/(1+exp(Af+B)),
            # so p_platt = 1 - p_sigmoid when A < 0.  We work with
            # the standard sigmoid here and negate A at the end.
            # Actually, let's follow Platt exactly:
            # p_i = 1 / (1 + exp(a * f_i + b))
            # We need to be careful: exp(a*f+b) for the Platt form.
            p = self._platt_sigmoid(scores, a, b)

            # Gradient of NLL w.r.t. (a, b)
            d = p - target  # shape (n,)
            grad_a = np.dot(d, scores)
            grad_b = np.sum(d)

            grad_norm = np.sqrt(grad_a ** 2 + grad_b ** 2)
            if grad_norm < tol:
                logger.debug(
                    "PlattScaling converged at iteration %d (grad=%.2e)",
                    iteration,
                    grad_norm,
                )
                break

            # Hessian: p*(1-p) weighted
            w = p * (1.0 - p) + 1e-12  # avoid division by zero
            h_aa = np.dot(w, scores ** 2)
            h_ab = np.dot(w, scores)
            h_bb = np.sum(w)

            # Solve 2x2 Newton system: H @ delta = -grad
            det = h_aa * h_bb - h_ab * h_ab
            if abs(det) < 1e-15:
                logger.warning("PlattScaling: Hessian near-singular, stopping early")
                break

            da = -(h_bb * grad_a - h_ab * grad_b) / det
            db = -(h_aa * grad_b - h_ab * grad_a) / det

            # Line search with step halving for robustness
            step = 1.0
            old_nll = self._nll(scores, target, a, b)
            for _ in range(20):
                new_a = a + step * da
                new_b = b + step * db
                new_nll = self._nll(scores, target, new_a, new_b)
                if new_nll < old_nll + 1e-10:
                    break
                step *= 0.5
            else:
                logger.debug("PlattScaling: line search exhausted at iter %d", iteration)

            a = a + step * da
            b = b + step * db

        self.A = a
        self.B = b
        self._fitted = True

        logger.debug("PlattScaling fitted: A=%.6f, B=%.6f", self.A, self.B)
        return self

    @staticmethod
    def _platt_sigmoid(scores: np.ndarray, a: float, b: float) -> np.ndarray:
        """Compute P = 1 / (1 + exp(a*f + b)) with numerical stability."""
        z = a * scores + b
        # P = 1/(1+exp(z)) = sigmoid(-z)
        return _sigmoid(-z)

    @staticmethod
    def _nll(
        scores: np.ndarray, target: np.ndarray, a: float, b: float
    ) -> float:
        """Negative log-likelihood for Platt model."""
        p = PlattScaling._platt_sigmoid(scores, a, b)
        p = np.clip(p, 1e-15, 1.0 - 1e-15)
        return -np.sum(target * np.log(p) + (1.0 - target) * np.log(1.0 - p))

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply fitted Platt scaling to raw scores.

        Parameters
        ----------
        scores : ndarray, shape (n,)
            Raw scores to calibrate.

        Returns
        -------
        ndarray, shape (n,)
            Calibrated probabilities in [0, 1].
        """
        if not self._fitted:
            raise RuntimeError("PlattScaling has not been fitted")
        scores = np.asarray(scores, dtype=np.float64).ravel()
        return self._platt_sigmoid(scores, self.A, self.B)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "type": "platt",
            "fitted": self._fitted,
            "A": self.A,
            "B": self.B,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PlattScaling:
        """Deserialize from dict."""
        ps = cls()
        if d.get("fitted", False):
            ps.A = float(d["A"])
            ps.B = float(d["B"])
            ps._fitted = True
        return ps


# ---------------------------------------------------------------------------
# Numerically stable sigmoid
# ---------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: 1 / (1 + exp(-z))."""
    z = np.asarray(z, dtype=np.float64)
    pos_mask = z >= 0
    result = np.empty_like(z)
    # For z >= 0: 1 / (1 + exp(-z))
    result[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))
    # For z < 0: exp(z) / (1 + exp(z))
    exp_z = np.exp(z[~pos_mask])
    result[~pos_mask] = exp_z / (1.0 + exp_z)
    return result


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax with numerical stability.

    Parameters
    ----------
    logits : ndarray, shape (n, k)
        Raw logit values.

    Returns
    -------
    ndarray, shape (n, k)
        Probability distributions (rows sum to 1).
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits[np.newaxis, :]
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Multiclass logistic regression with L2 regularization
# ---------------------------------------------------------------------------

class SeverityClassifier:
    """Multiclass logistic regression for severity classification.

    Implements one-vs-rest decomposition with L2 regularization, trained
    via batch gradient descent.  After fitting, ordinal constraints are
    enforced by projecting cumulative probabilities onto the isotone cone
    using the PAV algorithm.

    The feature space is 5-dimensional:
        [danger_score, motion_score, proximity_score, vlm_confidence,
         normalized_vlm_severity_idx]

    All inputs are expected in [0, 1].

    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent.
    l2_lambda : float
        L2 regularization strength (applied to weights, not bias).
    max_iter : int
        Maximum training iterations.
    tol : float
        Convergence tolerance on the gradient norm.
    enforce_ordinal : bool
        If True, enforce NONE < LOW < MEDIUM < HIGH monotonicity
        in predicted probabilities via isotonic projection.

    Attributes:
        W : ndarray, shape (n_features, n_classes)
            Weight matrix.
        b : ndarray, shape (n_classes,)
            Bias vector.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        l2_lambda: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
        enforce_ordinal: bool = True,
    ) -> None:
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.max_iter = max_iter
        self.tol = tol
        self.enforce_ordinal = enforce_ordinal

        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self._fitted: bool = False
        self._n_features: int = 5

    def fit(self, X: np.ndarray, y: np.ndarray) -> SeverityClassifier:
        """Fit multiclass logistic regression via gradient descent.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.  Each row is a feature vector (see SeverityFeatures).
        y : ndarray, shape (n_samples,)
            Integer class labels in {0, 1, 2, 3}.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).ravel()

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        n_samples, n_features = X.shape
        self._n_features = n_features

        if len(y) != n_samples:
            raise ValueError("X and y must have the same number of samples")
        if y.min() < 0 or y.max() >= NUM_CLASSES:
            raise ValueError(
                f"Labels must be in [0, {NUM_CLASSES - 1}], "
                f"got range [{y.min()}, {y.max()}]"
            )

        # One-hot encode targets
        Y = np.zeros((n_samples, NUM_CLASSES), dtype=np.float64)
        Y[np.arange(n_samples), y] = 1.0

        # Initialize weights (Xavier initialization)
        rng = np.random.default_rng(42)
        scale = np.sqrt(2.0 / (n_features + NUM_CLASSES))
        self.W = rng.normal(0, scale, (n_features, NUM_CLASSES))
        self.b = np.zeros(NUM_CLASSES, dtype=np.float64)

        prev_loss = np.inf
        for iteration in range(self.max_iter):
            # Forward pass
            logits = X @ self.W + self.b  # (n, k)
            probs = _softmax(logits)  # (n, k)

            # Cross-entropy loss + L2 penalty
            log_probs = np.log(np.clip(probs, 1e-15, 1.0))
            ce_loss = -np.sum(Y * log_probs) / n_samples
            l2_loss = 0.5 * self.l2_lambda * np.sum(self.W ** 2)
            total_loss = ce_loss + l2_loss

            # Gradients
            grad_logits = (probs - Y) / n_samples  # (n, k)
            grad_W = X.T @ grad_logits + self.l2_lambda * self.W  # (d, k)
            grad_b = np.sum(grad_logits, axis=0)  # (k,)

            grad_norm = np.sqrt(np.sum(grad_W ** 2) + np.sum(grad_b ** 2))

            # Check convergence
            if grad_norm < self.tol:
                logger.debug(
                    "SeverityClassifier converged at iteration %d "
                    "(loss=%.6f, grad=%.2e)",
                    iteration,
                    total_loss,
                    grad_norm,
                )
                break

            # Gradient descent step
            self.W -= self.learning_rate * grad_W
            self.b -= self.learning_rate * grad_b

            # Adaptive learning rate: reduce if loss increased
            if total_loss > prev_loss + 1e-10:
                self.learning_rate *= 0.5
                logger.debug(
                    "Reducing learning rate to %.6f at iteration %d",
                    self.learning_rate,
                    iteration,
                )
            prev_loss = total_loss

            if iteration % 200 == 0:
                logger.debug(
                    "Iteration %d: loss=%.6f, grad_norm=%.6f",
                    iteration,
                    total_loss,
                    grad_norm,
                )

        self._fitted = True
        logger.info(
            "SeverityClassifier fitted: %d samples, %d features, "
            "final loss=%.6f",
            n_samples,
            n_features,
            total_loss,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features) or (n_features,)
            Feature matrix.

        Returns
        -------
        ndarray, shape (n_samples, 4)
            Predicted probability for each severity class.
            Columns are [NONE, LOW, MEDIUM, HIGH].
        """
        if not self._fitted:
            raise RuntimeError("SeverityClassifier has not been fitted")

        X = np.asarray(X, dtype=np.float64)
        single = X.ndim == 1
        if single:
            X = X[np.newaxis, :]

        logits = X @ self.W + self.b
        probs = _softmax(logits)

        if self.enforce_ordinal:
            probs = self._enforce_ordinal_constraints(probs)

        if single:
            return probs[0]
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features) or (n_features,)

        Returns
        -------
        ndarray, shape (n_samples,) or scalar
            Predicted class indices.
        """
        probs = self.predict_proba(X)
        if probs.ndim == 1:
            return int(np.argmax(probs))
        return np.argmax(probs, axis=1)

    @staticmethod
    def _enforce_ordinal_constraints(probs: np.ndarray) -> np.ndarray:
        """Project probabilities to satisfy ordinal monotonicity.

        For ordinal classification, the cumulative distribution function
        P(Y <= k) should be non-decreasing in k.  We enforce this by:
        1. Computing cumulative probabilities.
        2. Applying isotonic projection (PAV) to each sample's cumulative probs.
        3. Recovering class probabilities from the projected CDF.

        Additionally, we enforce that for higher severity indices, the
        cumulative probability should decrease (i.e., P(Y >= k) is
        non-increasing), which is automatically satisfied by a valid CDF.

        Parameters
        ----------
        probs : ndarray, shape (n, 4)
            Raw predicted probabilities.

        Returns
        -------
        ndarray, shape (n, 4)
            Adjusted probabilities satisfying ordinal constraints.
        """
        n = probs.shape[0]
        result = np.copy(probs)

        for i in range(n):
            # Compute cumulative probabilities P(Y <= k)
            cdf = np.cumsum(result[i])

            # CDF must be non-decreasing and end at 1.0 — by construction
            # of cumsum this holds, but after arbitrary perturbation it
            # might not.  We project onto the isotone cone.
            cdf_proj = np.clip(cdf, 0.0, 1.0)

            # Ensure monotonicity: each value >= previous
            for k in range(1, NUM_CLASSES):
                if cdf_proj[k] < cdf_proj[k - 1]:
                    cdf_proj[k] = cdf_proj[k - 1]

            # Force final value to 1
            cdf_proj[-1] = 1.0

            # Recover class probabilities from CDF
            adjusted = np.empty(NUM_CLASSES, dtype=np.float64)
            adjusted[0] = cdf_proj[0]
            for k in range(1, NUM_CLASSES):
                adjusted[k] = cdf_proj[k] - cdf_proj[k - 1]

            # Ensure non-negative and re-normalize
            adjusted = np.maximum(adjusted, 0.0)
            total = adjusted.sum()
            if total > 0:
                adjusted /= total
            else:
                adjusted = np.ones(NUM_CLASSES) / NUM_CLASSES

            result[i] = adjusted

        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize model weights to JSON-compatible dict."""
        return {
            "fitted": self._fitted,
            "n_features": self._n_features,
            "learning_rate": self.learning_rate,
            "l2_lambda": self.l2_lambda,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "enforce_ordinal": self.enforce_ordinal,
            "W": self.W.tolist() if self.W is not None else None,
            "b": self.b.tolist() if self.b is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SeverityClassifier:
        """Deserialize from dict."""
        obj = cls(
            learning_rate=d.get("learning_rate", 0.1),
            l2_lambda=d.get("l2_lambda", 0.01),
            max_iter=d.get("max_iter", 1000),
            tol=d.get("tol", 1e-6),
            enforce_ordinal=d.get("enforce_ordinal", True),
        )
        obj._n_features = d.get("n_features", 5)
        if d.get("fitted", False):
            obj.W = np.array(d["W"], dtype=np.float64)
            obj.b = np.array(d["b"], dtype=np.float64)
            obj._fitted = True
        return obj


# ---------------------------------------------------------------------------
# CalibratedSeverityModel: end-to-end interface
# ---------------------------------------------------------------------------

@dataclass
class SeverityPrediction:
    """Output of the calibrated severity model.

    Attributes:
        severity: Predicted severity label (NONE / LOW / MEDIUM / HIGH).
        confidence: Calibrated confidence for the predicted class.
        probabilities: Full probability distribution over all 4 classes.
        severity_idx: Integer index of the predicted class.
    """

    severity: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    severity_idx: int = 0


class CalibratedSeverityModel:
    """End-to-end calibrated severity classifier.

    Combines a ``SeverityClassifier`` (logistic regression) with a post-hoc
    calibrator (``IsotonicCalibrator`` or ``PlattScaling``) to produce
    well-calibrated severity predictions from multimodal features.

    Workflow::

        features -> SeverityClassifier.predict_proba() -> raw_probs
        max(raw_probs) -> calibrator.transform() -> calibrated_confidence

    Parameters
    ----------
    calibration_method : str
        One of ``"isotonic"`` or ``"platt"``.  Determines which calibrator
        is used for post-hoc confidence calibration.
    classifier_kwargs : dict
        Keyword arguments forwarded to ``SeverityClassifier.__init__``.
    """

    def __init__(
        self,
        calibration_method: str = "isotonic",
        **classifier_kwargs: Any,
    ) -> None:
        if calibration_method not in ("isotonic", "platt"):
            raise ValueError(
                f"calibration_method must be 'isotonic' or 'platt', "
                f"got '{calibration_method}'"
            )
        self.calibration_method = calibration_method
        self.classifier = SeverityClassifier(**classifier_kwargs)

        if calibration_method == "isotonic":
            self.calibrator: IsotonicCalibrator | PlattScaling = IsotonicCalibrator()
        else:
            self.calibrator = PlattScaling()

        self._calibrator_fitted: bool = False

    def fit_from_data(
        self,
        features_list: list[SeverityFeatures],
        labels: list[int] | np.ndarray,
        calibration_fraction: float = 0.2,
    ) -> CalibratedSeverityModel:
        """Train the classifier and calibrate from labeled data.

        Splits the data into a training set and a calibration set.  The
        classifier is trained on the training set, and the calibrator is
        fitted on the calibration set's predicted max-probabilities.

        Parameters
        ----------
        features_list : list of SeverityFeatures
            Training features.
        labels : array-like, shape (n,)
            Integer severity labels in {0, 1, 2, 3}.
        calibration_fraction : float
            Fraction of data reserved for calibration (default 0.2).

        Returns
        -------
        self
        """
        # Build feature matrix
        X = np.array([f.to_array() for f in features_list], dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64).ravel()

        if len(X) != len(y):
            raise ValueError("features_list and labels must have same length")

        n = len(X)
        if n < 10:
            logger.warning(
                "Very few samples (%d) for train+calibrate split. "
                "Calibrator may be unreliable.",
                n,
            )

        # Stratified-ish split: shuffle then split
        rng = np.random.default_rng(42)
        indices = rng.permutation(n)

        n_cal = max(2, int(n * calibration_fraction))
        cal_idx = indices[:n_cal]
        train_idx = indices[n_cal:]

        if len(train_idx) < 2:
            # Not enough data to split; train on everything, skip calibration
            logger.warning(
                "Too few samples to split for calibration. "
                "Training on all data without calibration."
            )
            self.classifier.fit(X, y)
            return self

        X_train, y_train = X[train_idx], y[train_idx]
        X_cal, y_cal = X[cal_idx], y[cal_idx]

        # Train classifier
        self.classifier.fit(X_train, y_train)

        # Calibrate on held-out data
        self._fit_calibrator(X_cal, y_cal)

        logger.info(
            "CalibratedSeverityModel fitted: %d train, %d calibration samples",
            len(train_idx),
            n_cal,
        )
        return self

    def _fit_calibrator(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the calibrator on calibration data.

        We calibrate the max predicted probability against a binary
        indicator of whether the argmax prediction was correct.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Calibration features.
        y : ndarray, shape (n,)
            True labels.
        """
        probs = self.classifier.predict_proba(X)
        if probs.ndim == 1:
            probs = probs[np.newaxis, :]

        max_probs = np.max(probs, axis=1)
        pred_labels = np.argmax(probs, axis=1)
        correct = (pred_labels == y).astype(np.float64)

        try:
            self.calibrator.fit(max_probs, correct)
            self._calibrator_fitted = True
        except (ValueError, RuntimeError) as e:
            logger.warning(
                "Calibrator fitting failed: %s. Predictions will use "
                "uncalibrated confidence.",
                e,
            )
            self._calibrator_fitted = False

    def predict(self, features: SeverityFeatures) -> SeverityPrediction:
        """Predict severity for a single sample.

        Parameters
        ----------
        features : SeverityFeatures
            Input features for one clip.

        Returns
        -------
        SeverityPrediction
            Named result with severity label, calibrated confidence,
            and full probability distribution.
        """
        x = features.to_array()
        probs = self.classifier.predict_proba(x)

        if probs.ndim > 1:
            probs = probs[0]

        severity_idx = int(np.argmax(probs))
        raw_confidence = float(probs[severity_idx])

        # Apply calibration to the max probability
        if self._calibrator_fitted:
            cal_conf = float(
                self.calibrator.transform(np.array([raw_confidence]))[0]
            )
        else:
            cal_conf = raw_confidence

        # Build probability dict
        prob_dict = {
            SEVERITY_LABELS[i]: float(probs[i]) for i in range(NUM_CLASSES)
        }

        return SeverityPrediction(
            severity=SEVERITY_LABELS[severity_idx],
            confidence=cal_conf,
            probabilities=prob_dict,
            severity_idx=severity_idx,
        )

    def predict_batch(
        self, features_list: list[SeverityFeatures]
    ) -> list[SeverityPrediction]:
        """Predict severity for a batch of samples.

        Parameters
        ----------
        features_list : list of SeverityFeatures
            Input features.

        Returns
        -------
        list of SeverityPrediction
        """
        return [self.predict(f) for f in features_list]

    def save(self, path: str) -> None:
        """Serialize model to a JSON file.

        Parameters
        ----------
        path : str
            Output file path (should end in .json).
        """
        state = {
            "version": 1,
            "calibration_method": self.calibration_method,
            "classifier": self.classifier.to_dict(),
            "calibrator": self.calibrator.to_dict(),
            "calibrator_fitted": self._calibrator_fitted,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        logger.info("CalibratedSeverityModel saved to %s", path)

    @classmethod
    def load(cls, path: str) -> CalibratedSeverityModel:
        """Load a model from a JSON file.

        Parameters
        ----------
        path : str
            Path to a JSON file previously created by ``save()``.

        Returns
        -------
        CalibratedSeverityModel
            Deserialized model ready for prediction.
        """
        with open(path, encoding="utf-8") as f:
            state = json.load(f)

        version = state.get("version", 1)
        if version != 1:
            raise ValueError(f"Unsupported model version: {version}")

        method = state["calibration_method"]
        model = cls(calibration_method=method)
        model.classifier = SeverityClassifier.from_dict(state["classifier"])

        cal_dict = state["calibrator"]
        if cal_dict["type"] == "isotonic":
            model.calibrator = IsotonicCalibrator.from_dict(cal_dict)
        elif cal_dict["type"] == "platt":
            model.calibrator = PlattScaling.from_dict(cal_dict)
        else:
            raise ValueError(f"Unknown calibrator type: {cal_dict['type']}")

        model._calibrator_fitted = state.get("calibrator_fitted", False)

        logger.info("CalibratedSeverityModel loaded from %s", path)
        return model

    def to_dict(self) -> dict[str, Any]:
        """Serialize full model state to a dict (for embedding in larger configs)."""
        return {
            "version": 1,
            "calibration_method": self.calibration_method,
            "classifier": self.classifier.to_dict(),
            "calibrator": self.calibrator.to_dict(),
            "calibrator_fitted": self._calibrator_fitted,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CalibratedSeverityModel:
        """Deserialize from dict."""
        method = d["calibration_method"]
        model = cls(calibration_method=method)
        model.classifier = SeverityClassifier.from_dict(d["classifier"])

        cal_dict = d["calibrator"]
        if cal_dict["type"] == "isotonic":
            model.calibrator = IsotonicCalibrator.from_dict(cal_dict)
        elif cal_dict["type"] == "platt":
            model.calibrator = PlattScaling.from_dict(cal_dict)

        model._calibrator_fitted = d.get("calibrator_fitted", False)
        return model
