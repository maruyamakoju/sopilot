"""Operator learning curve analysis and certification pathway prediction.

Supports three model classes selected via BIC:
  - Linear:      score(n) = slope * n + intercept
  - Exponential: score(n) = L - (L - s0) * exp(-n / tau)
  - Gaussian Process: Bayesian non-parametric (RBF kernel), provides
                       calibrated credible intervals on future scores.

Changepoint detection via CUSUM alerts to sudden performance shifts.

References:
    Rasmussen & Williams (2006). Gaussian Processes for Machine Learning.
    Page & Barnard (1954). Continuous Inspection Schemes (CUSUM origin).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data structures (backward-compatible with v0.7 schema)
# ---------------------------------------------------------------------------

@dataclass
class LearningCurveResult:
    operator_id: str
    job_count: int                    # Number of scored evaluations
    scores: list[float]               # Historical scores in chronological order
    avg_score: float                  # Mean of all scores
    trend_slope: float                # Linear regression slope (points per evaluation)
    latest_score: float               # Most recent score
    pass_threshold: float             # Certification pass threshold
    is_certified: bool                # Whether latest score >= pass_threshold

    # Prediction fields (None if insufficient data or already certified)
    evaluations_to_certification: int | None = None
    confidence: str = "low"           # 'high' / 'medium' / 'low'
    trajectory: str = "improving"     # 'improving' / 'plateau' / 'declining'
    model_type: str = "linear"        # 'linear' / 'exponential' / 'gaussian_process' / 'insufficient_data'
    projected_scores: list[dict] = field(default_factory=list)  # Next 5 projected scores

    # --- New research-grade fields (all Optional, default None) ---
    bic_scores: dict[str, float] | None = None
    """BIC value for each candidate model.  Lower = better fit."""

    gp_uncertainty_bands: list[dict] | None = None
    """GP posterior credible bands for the next 10 evaluations.
    Each entry: {"evaluation_number": int, "mean": float, "lower": float, "upper": float}"""

    changepoints: list[int] | None = None
    """0-indexed positions in scores where CUSUM detected a sudden shift."""

    tau: float | None = None
    """Exponential time constant (evaluations to reach 63% of asymptote)."""

    plateau_score: float | None = None
    """Predicted asymptote from exponential model (= L parameter)."""

    evaluations_to_cert_ci: tuple[float, float] | None = None
    """Bootstrap 95% CI on evaluations_to_certification (from exponential model)."""


# ---------------------------------------------------------------------------
# Internal helpers: linear regression
# ---------------------------------------------------------------------------

def _linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Return (slope, intercept) from simple linear regression."""
    n = len(xs)
    if n < 2:
        return 0.0, ys[0] if ys else 0.0
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys, strict=False))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-10:
        return 0.0, sy / n
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def _linear_bic(xs: list[float], ys: list[float], slope: float, intercept: float) -> float:
    """BIC for linear model (k=2 parameters)."""
    n = len(ys)
    residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    sse = sum(r * r for r in residuals)
    sigma2 = max(sse / n, 1e-10)
    log_lik = -0.5 * n * math.log(2 * math.pi * sigma2) - sse / (2 * sigma2)
    return 2 * math.log(n) - 2 * log_lik  # BIC = k*ln(n) - 2*ln(L)


# ---------------------------------------------------------------------------
# Internal helpers: exponential saturation model
# ---------------------------------------------------------------------------

def _exp_model(n: float, L: float, s0: float, tau: float) -> float:
    """Exponential saturation: L - (L - s0) * exp(-n / tau)."""
    return L - (L - s0) * math.exp(-n / tau)


def _fit_exponential(
    xs: list[float],
    ys: list[float],
) -> tuple[float, float, float, float] | None:
    """Fit exponential model; returns (L, s0, tau, bic) or None on failure."""
    try:
        import numpy as np
        from scipy.optimize import curve_fit  # type: ignore[import]
    except ImportError:
        return None

    x_arr = np.array(xs, dtype=np.float64)
    y_arr = np.array(ys, dtype=np.float64)
    n = len(ys)

    def _model(x: Any, L: float, s0: float, tau: float) -> Any:
        return L - (L - s0) * np.exp(-x / tau)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p0 = [min(100.0, max(ys) + 5), ys[0], max(1.0, n / 2)]
            bounds = ([max(ys), 0.0, 0.01], [110.0, 105.0, float(n * 10 + 1)])
            popt, _ = curve_fit(
                _model, x_arr, y_arr, p0=p0, bounds=bounds, maxfev=5000
            )
    except Exception:
        return None

    L, s0, tau = float(popt[0]), float(popt[1]), float(popt[2])
    residuals = y_arr - _model(x_arr, L, s0, tau)
    sse = float(np.sum(residuals ** 2))
    sigma2 = max(sse / n, 1e-10)
    log_lik = -0.5 * n * math.log(2 * math.pi * sigma2) - sse / (2 * sigma2)
    bic = 3 * math.log(n) - 2 * log_lik  # k=3 parameters
    return L, s0, tau, bic


def _bootstrap_tau_ci(
    xs: list[float],
    ys: list[float],
    pass_threshold: float,
    n_current: int,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> tuple[float, float] | None:
    """Bootstrap 95% CI on evaluations-to-certification from exponential model."""
    try:
        import numpy as np
        from scipy.optimize import curve_fit  # type: ignore[import]
    except ImportError:
        return None

    rng = np.random.default_rng(seed)
    x_arr = np.array(xs, dtype=np.float64)
    y_arr = np.array(ys, dtype=np.float64)
    n = len(ys)
    cert_counts: list[float] = []

    def _model(x: Any, L: float, s0: float, tau: float) -> Any:
        return L - (L - s0) * np.exp(-x / tau)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        xb, yb = x_arr[idx], y_arr[idx]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    _model,
                    xb, yb,
                    p0=[min(100.0, float(yb.max()) + 5), float(yb[0]), max(1.0, n / 2)],
                    bounds=([float(yb.max()) if yb.max() < 100 else 50, 0.0, 0.01],
                            [110.0, 105.0, float(n * 10 + 1)]),
                    maxfev=2000,
                )
        except Exception:
            continue
        L, s0, tau = float(popt[0]), float(popt[1]), float(popt[2])
        if L < pass_threshold or tau <= 0:
            continue
        # Solve L - (L-s0)*exp(-n_cert/tau) = threshold
        ratio = (L - pass_threshold) / max(L - s0, 1e-8)
        if ratio <= 0 or ratio >= 1:
            continue
        n_cert = -tau * math.log(ratio)
        evals_needed = max(0.0, n_cert - n_current)
        cert_counts.append(evals_needed)

    if len(cert_counts) < 20:
        return None
    arr = sorted(cert_counts)
    lo = arr[int(0.025 * len(arr))]
    hi = arr[int(0.975 * len(arr))]
    return round(lo, 1), round(hi, 1)


# ---------------------------------------------------------------------------
# Internal helpers: Gaussian Process (manual numpy implementation)
# ---------------------------------------------------------------------------

def _rbf_kernel(x1: Any, x2: Any, sigma_f: float, length: float) -> Any:
    """RBF (squared-exponential) kernel: σ_f² exp(−½(x−x')²/l²)."""
    import numpy as np
    diff = x1[:, None] - x2[None, :]
    return sigma_f ** 2 * np.exp(-0.5 * diff ** 2 / length ** 2)


def _gp_log_marginal_likelihood(
    log_params: Any,
    x_train: Any,
    y_train: Any,
) -> float:
    """Negative log marginal likelihood for GP hyperparameter optimization."""
    import numpy as np
    sigma_f = float(np.exp(log_params[0]))
    length = float(np.exp(log_params[1]))
    sigma_n = float(np.exp(log_params[2]))

    n = len(x_train)
    K = _rbf_kernel(x_train, x_train, sigma_f, length)
    K += (sigma_n ** 2 + 1e-6) * np.eye(n)

    # Cholesky with jitter fallback
    jitter = 1e-8
    for _ in range(6):
        try:
            L = np.linalg.cholesky(K)
            break
        except np.linalg.LinAlgError:
            K += jitter * np.eye(n)
            jitter *= 10
    else:
        return 1e15

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    log_lik = (
        -0.5 * float(y_train @ alpha)
        - float(np.sum(np.log(np.diagonal(L))))
        - 0.5 * n * math.log(2 * math.pi)
    )
    return -log_lik


def _fit_gp(
    xs: list[float],
    ys: list[float],
) -> dict | None:
    """Fit GP hyperparameters and return posterior parameters or None."""
    try:
        import numpy as np
        from scipy.optimize import minimize  # type: ignore[import]
    except ImportError:
        return None

    x_arr = np.array(xs, dtype=np.float64)
    y_arr = np.array(ys, dtype=np.float64)
    n = len(ys)

    # Multiple random restarts for hyperparameter optimization
    best_nlml = 1e15
    best_params: Any = None
    rng = np.random.default_rng(0)

    for _ in range(5):
        log_p0 = rng.uniform([-1, -1, -3], [3, 2, 0])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    _gp_log_marginal_likelihood,
                    log_p0,
                    args=(x_arr, y_arr),
                    method="L-BFGS-B",
                    options={"maxiter": 200},
                )
            if res.fun < best_nlml:
                best_nlml = res.fun
                best_params = res.x
        except Exception:
            continue

    if best_params is None:
        return None

    sigma_f = float(np.exp(best_params[0]))
    length = float(np.exp(best_params[1]))
    sigma_n = float(np.exp(best_params[2]))

    K = _rbf_kernel(x_arr, x_arr, sigma_f, length)
    K += (sigma_n ** 2 + 1e-6) * np.eye(n)

    jitter = 1e-8
    for _ in range(6):
        try:
            L_chol = np.linalg.cholesky(K)
            break
        except np.linalg.LinAlgError:
            K += jitter * np.eye(n)
            jitter *= 10
    else:
        return None

    alpha = np.linalg.solve(L_chol.T, np.linalg.solve(L_chol, y_arr))

    # Compute BIC: GP has k=3 parameters
    nlml = _gp_log_marginal_likelihood(best_params, x_arr, y_arr)
    bic = 3 * math.log(n) + 2 * nlml

    return {
        "x_train": x_arr,
        "y_train": y_arr,
        "alpha": alpha,
        "L_chol": L_chol,
        "sigma_f": sigma_f,
        "length": length,
        "sigma_n": sigma_n,
        "bic": bic,
    }


def _gp_predict(gp: dict, x_star: Any) -> tuple[Any, Any]:
    """GP posterior mean and std at test points x_star."""
    import numpy as np
    x_arr = gp["x_train"]
    sigma_f, length, sigma_n = gp["sigma_f"], gp["length"], gp["sigma_n"]
    L_chol = gp["L_chol"]
    alpha = gp["alpha"]

    K_s = _rbf_kernel(x_arr, x_star, sigma_f, length)   # (n, m)
    K_ss = _rbf_kernel(x_star, x_star, sigma_f, length)  # (m, m)

    mu = K_s.T @ alpha
    v = np.linalg.solve(L_chol, K_s)
    cov = K_ss - v.T @ v
    std = np.sqrt(np.maximum(np.diag(cov) + sigma_n ** 2, 1e-8))
    return mu, std


# ---------------------------------------------------------------------------
# Internal helpers: CUSUM changepoint detection
# ---------------------------------------------------------------------------

def detect_changepoints(
    scores: list[float],
    k: float = 0.5,
    threshold: float = 2.5,
    min_gap: int = 3,
) -> list[int]:
    """Detect sudden performance shifts via CUSUM algorithm.

    Args:
        scores:     Chronological score list.
        k:          Allowance parameter (slack for expected drift).
        threshold:  Alert threshold in units of score standard deviation.
        min_gap:    Minimum evaluations between consecutive detections.

    Returns:
        List of 0-indexed positions where a changepoint was detected.
    """
    n = len(scores)
    if n < 4:
        return []

    mean_s = sum(scores) / n
    var_s = sum((s - mean_s) ** 2 for s in scores) / max(n - 1, 1)
    std_s = math.sqrt(var_s)
    if std_s < 1e-6:
        return []

    z = [(s - mean_s) / std_s for s in scores]
    cusum_pos = 0.0
    cusum_neg = 0.0
    changepoints: list[int] = []
    last_detection = -min_gap - 1

    for i, zi in enumerate(z):
        cusum_pos = max(0.0, cusum_pos + zi - k)
        cusum_neg = min(0.0, cusum_neg + zi + k)
        if (cusum_pos > threshold or abs(cusum_neg) > threshold) and i - last_detection >= min_gap:
            changepoints.append(i)
            last_detection = i
            cusum_pos = 0.0
            cusum_neg = 0.0

    return changepoints


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def analyze_learning_curve(
    operator_id: str,
    scores: list[float],
    *,
    pass_threshold: float = 90.0,
) -> LearningCurveResult:
    """Analyze an operator's score trajectory and project certification pathway.

    Selects the best-fitting model (linear / exponential / gaussian_process)
    via BIC, provides calibrated uncertainty bands, and detects sudden
    performance shifts via CUSUM.

    Args:
        operator_id:    Operator identifier.
        scores:         Chronological list of scores (oldest first).
        pass_threshold: Score required for certification.

    Returns:
        LearningCurveResult — backward-compatible with v0.7; extended with
        research-grade fields (bic_scores, gp_uncertainty_bands, changepoints,
        tau, plateau_score, evaluations_to_cert_ci).
    """
    n = len(scores)

    if n == 0:
        return LearningCurveResult(
            operator_id=operator_id,
            job_count=0,
            scores=[],
            avg_score=0.0,
            trend_slope=0.0,
            latest_score=0.0,
            pass_threshold=pass_threshold,
            is_certified=False,
            model_type="insufficient_data",
            confidence="low",
            trajectory="improving",
        )

    avg_score = sum(scores) / n
    latest_score = scores[-1]
    is_certified = latest_score >= pass_threshold

    if n < 2:
        return LearningCurveResult(
            operator_id=operator_id,
            job_count=n,
            scores=scores,
            avg_score=round(avg_score, 2),
            trend_slope=0.0,
            latest_score=round(latest_score, 2),
            pass_threshold=pass_threshold,
            is_certified=is_certified,
            model_type="insufficient_data",
            confidence="low",
            trajectory="improving",
        )

    xs = [float(i) for i in range(n)]

    # ---- Changepoint detection (always run) ----
    changepoints = detect_changepoints(scores)

    # ---- Linear model (always computed, baseline) ----
    slope, intercept = _linear_regression(xs, scores)
    linear_bic = _linear_bic(xs, scores, slope, intercept)
    bic_scores: dict[str, float] = {"linear": round(linear_bic, 2)}

    # ---- Exponential model ----
    exp_result = None
    exp_bic = 1e15
    tau_val: float | None = None
    plateau_val: float | None = None
    if n >= 4:
        exp_result = _fit_exponential(xs, scores)
        if exp_result is not None:
            L_fit, s0_fit, tau_fit, exp_bic_fit = exp_result
            exp_bic = exp_bic_fit
            tau_val = round(tau_fit, 2)
            plateau_val = round(L_fit, 2)
            bic_scores["exponential"] = round(exp_bic, 2)

    # ---- Gaussian Process model ----
    gp_result = None
    gp_bic = 1e15
    gp_bands: list[dict] | None = None
    if n >= 5:
        gp_result = _fit_gp(xs, scores)
        if gp_result is not None:
            gp_bic = gp_result["bic"]
            bic_scores["gaussian_process"] = round(gp_bic, 2)

            # Always compute GP bands regardless of model selection
            try:
                import numpy as np
                x_future = np.array([float(i) for i in range(n, n + 10)])
                mu, std = _gp_predict(gp_result, x_future)
                gp_bands = []
                for i_f, (m_val, s_val) in enumerate(zip(mu, std)):
                    gp_bands.append({
                        "evaluation_number": n + i_f + 1,
                        "mean": round(float(np.clip(m_val, 0, 100)), 1),
                        "lower": round(float(np.clip(m_val - 1.96 * s_val, 0, 100)), 1),
                        "upper": round(float(np.clip(m_val + 1.96 * s_val, 0, 100)), 1),
                    })
            except Exception:
                gp_bands = None

    # ---- Model selection via BIC ----
    min_bic = linear_bic
    selected_model = "linear"
    if exp_bic < min_bic:
        min_bic = exp_bic
        selected_model = "exponential"
    if gp_bic < min_bic:
        selected_model = "gaussian_process"

    # ---- Trajectory classification (use linear slope for consistency) ----
    if slope > 1.0:
        trajectory = "improving"
    elif slope < -1.0:
        trajectory = "declining"
    else:
        trajectory = "plateau"

    # Override trajectory if changepoint recently detected (last 3 evals)
    if changepoints and n - changepoints[-1] <= 3:
        recent_mean = sum(scores[changepoints[-1]:]) / max(1, n - changepoints[-1])
        prior_mean = sum(scores[:changepoints[-1]]) / max(1, changepoints[-1])
        if recent_mean > prior_mean + 2:
            trajectory = "improving"
        elif recent_mean < prior_mean - 2:
            trajectory = "declining"

    # ---- Confidence ----
    if n >= 10:
        confidence = "high"
    elif n >= 5:
        confidence = "medium"
    else:
        confidence = "low"

    # ---- Projected scores (next 5, from selected model) ----
    projected_scores: list[dict] = []
    for future_i in range(n, n + 10):
        if selected_model == "exponential" and exp_result is not None:
            L_f, s0_f, tau_f, _ = exp_result
            proj = _exp_model(float(future_i), L_f, s0_f, tau_f)
        elif selected_model == "gaussian_process" and gp_result is not None:
            try:
                import numpy as np
                mu_p, _ = _gp_predict(gp_result, np.array([float(future_i)]))
                proj = float(mu_p[0])
            except Exception:
                proj = intercept + slope * future_i
        else:
            proj = intercept + slope * future_i
        proj = max(0.0, min(100.0, proj))
        projected_scores.append({
            "evaluation_number": future_i + 1,
            "projected_score": round(proj, 1),
        })

    # ---- Evaluations to certification ----
    evaluations_to_cert: int | None = None
    cert_ci: tuple[float, float] | None = None

    if is_certified:
        evaluations_to_cert = 0
    elif selected_model == "exponential" and exp_result is not None:
        L_f, s0_f, tau_f, _ = exp_result
        if L_f >= pass_threshold and tau_f > 0:
            ratio = (L_f - pass_threshold) / max(L_f - s0_f, 1e-8)
            if 0 < ratio < 1:
                n_cert = -tau_f * math.log(ratio)
                evals_needed = max(1, math.ceil(n_cert - n))
                evaluations_to_cert = evals_needed if evals_needed <= 100 else None
                cert_ci = _bootstrap_tau_ci(xs, scores, pass_threshold, n)
    elif selected_model == "gaussian_process" and gp_result is not None:
        # Use GP mean projection to find first crossing
        if projected_scores:
            for proj_entry in projected_scores:
                if proj_entry["projected_score"] >= pass_threshold:
                    evals_needed = proj_entry["evaluation_number"] - n
                    evaluations_to_cert = evals_needed if evals_needed <= 100 else None
                    break
    else:
        # Fallback: linear
        if slope > 0:
            n_cert = (pass_threshold - intercept) / slope
            if n_cert > n:
                evals_needed = max(1, math.ceil(n_cert - n))
                evaluations_to_cert = evals_needed if evals_needed <= 100 else None

    return LearningCurveResult(
        operator_id=operator_id,
        job_count=n,
        scores=[round(s, 2) for s in scores],
        avg_score=round(avg_score, 2),
        trend_slope=round(slope, 3),
        latest_score=round(latest_score, 2),
        pass_threshold=pass_threshold,
        is_certified=is_certified,
        evaluations_to_certification=evaluations_to_cert,
        confidence=confidence,
        trajectory=trajectory,
        model_type=selected_model,
        projected_scores=projected_scores[:5],
        # New research-grade fields
        bic_scores=bic_scores,
        gp_uncertainty_bands=gp_bands,
        changepoints=changepoints if changepoints else None,
        tau=tau_val,
        plateau_score=plateau_val,
        evaluations_to_cert_ci=cert_ci,
    )
