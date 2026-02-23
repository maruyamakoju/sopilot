"""Research-grade experiment tracking and ablation study framework.

Provides reproducible experiment management for severity classification
research, including hyperparameter tracking, ablation analysis, and
cross-experiment comparison with statistical rigour.

Design Principles:
  - Self-contained: depends only on numpy and the standard library.
  - Deterministic: seed management guarantees bitwise reproducibility.
  - Serialisable: all state round-trips through JSON for archival.
  - Composable: ExperimentTracker and AblationRunner are independent
    but interoperate through shared ExperimentConfig/ExperimentResult.

Typical usage::

    tracker = ExperimentTracker(storage_path="experiments.json")
    config = ExperimentConfig(
        name="baseline_v2",
        seed=42,
        description="Baseline with recalibration enabled",
        hyperparams={"lr": 1e-3, "alpha": 0.1},
        ablation_flags={"recalibration": True, "fraud_detection": True},
    )
    tracker.register(config)

    # ... run evaluation, collect y_true, y_pred ...
    metrics = tracker.compute_metrics(y_true, y_pred)
    result = tracker.record(config.name, metrics, timing_sec=12.5)

    # Ablation study
    runner = AblationRunner(base_config=config, eval_fn=my_eval_fn)
    report = runner.run()
    print(format_ablation_report(report))

References:
  Cohen (1960) "A Coefficient of Agreement for Nominal Scales"
  Fleiss, Levin & Paik (2003) "Statistical Methods for Rates and Proportions"
  Lipton & Steinhardt (2019) "Troubling Trends in Machine Learning Scholarship"
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Immutable specification of a single experiment.

    Attributes:
        name: Unique human-readable identifier (e.g. "baseline_v3").
        seed: Master random seed for full reproducibility.
        description: Free-text purpose statement for the lab notebook.
        hyperparams: Arbitrary key-value hyperparameters (learning rate,
            alpha, threshold, etc.).  Values must be JSON-serialisable.
        ablation_flags: Boolean feature toggles. Each key names a
            subsystem (e.g. "recalibration", "fraud_detection"); True
            means the subsystem is active in this configuration.
    """

    name: str
    seed: int = 42
    description: str = ""
    hyperparams: dict[str, Any] = field(default_factory=dict)
    ablation_flags: dict[str, bool] = field(default_factory=dict)

    def derive(self, *, name: str | None = None, **overrides) -> ExperimentConfig:
        """Create a modified copy, preserving unmodified fields.

        This is the canonical way to produce ablation variants::

            ablated = base.derive(
                name="no_recalibration",
                ablation_flags={**base.ablation_flags, "recalibration": False},
            )
        """
        data = asdict(self)
        if name is not None:
            data["name"] = name
        data.update(overrides)
        return ExperimentConfig(**data)

    @property
    def fingerprint(self) -> str:
        """Deterministic content hash for deduplication.

        Two configs with identical fields produce the same fingerprint,
        regardless of insertion order of dict keys.
        """
        canonical = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclass
class ExperimentResult:
    """Immutable record of a completed experiment run.

    Attributes:
        config: The ExperimentConfig that produced this result.
        metrics: Evaluation metrics (accuracy, macro_f1, kappa, etc.).
        timing_sec: Wall-clock duration of the evaluation in seconds.
        timestamp: ISO-8601 UTC timestamp of completion.
        git_hash: Short SHA of the working tree (None if unavailable).
        metadata: Arbitrary additional information (hardware, notes).
    """

    config: ExperimentConfig
    metrics: dict[str, float]
    timing_sec: float
    timestamp: str = ""
    git_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "config": asdict(self.config),
            "metrics": self.metrics,
            "timing_sec": self.timing_sec,
            "timestamp": self.timestamp,
            "git_hash": self.git_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentResult:
        """Reconstruct from a dictionary (inverse of to_dict)."""
        config = ExperimentConfig(**data["config"])
        return cls(
            config=config,
            metrics=data["metrics"],
            timing_sec=data["timing_sec"],
            timestamp=data.get("timestamp", ""),
            git_hash=data.get("git_hash"),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Metric computation (numpy-only, no sklearn)
# ---------------------------------------------------------------------------

def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Compute an n_classes x n_classes confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def _per_class_f1(cm: np.ndarray) -> np.ndarray:
    """Compute per-class F1 from a confusion matrix.

    Returns an array of length n_classes.  Classes with zero support
    receive F1 = 0.0 (the convention used by macro-averaging).
    """
    n_classes = cm.shape[0]
    f1s = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = precision + recall
        f1s[c] = 2.0 * precision * recall / denom if denom > 0 else 0.0
    return f1s


def compute_standard_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    n_classes: int = 4,
) -> dict[str, float]:
    """Compute the standard metric suite for severity classification.

    Returns a dictionary with the following keys:
      accuracy, macro_f1, weighted_f1, cohen_kappa, per_class_f1_0 .. N-1

    All computations use integer class indices.  The caller is
    responsible for mapping string labels to indices beforehand.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        n_classes: Number of distinct severity levels.

    Returns:
        Dictionary mapping metric name to float value.
    """
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    n = len(yt)
    if n == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "cohen_kappa": 0.0}

    cm = _confusion_matrix(yt, yp, n_classes)

    # Accuracy
    accuracy = float(np.trace(cm)) / n

    # Per-class F1
    f1s = _per_class_f1(cm)

    # Macro F1 (unweighted mean)
    macro_f1 = float(f1s.mean())

    # Weighted F1 (weighted by class support)
    supports = cm.sum(axis=1).astype(np.float64)
    weighted_f1 = float(np.dot(f1s, supports) / supports.sum()) if supports.sum() > 0 else 0.0

    # Cohen's kappa
    p_observed = accuracy
    p_expected = float(np.dot(cm.sum(axis=0), cm.sum(axis=1))) / (n * n)
    cohen_kappa = (p_observed - p_expected) / (1.0 - p_expected) if p_expected < 1.0 else 1.0

    metrics: dict[str, float] = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "cohen_kappa": cohen_kappa,
    }
    for c in range(n_classes):
        metrics[f"per_class_f1_{c}"] = float(f1s[c])

    return metrics


# ---------------------------------------------------------------------------
# Git helper
# ---------------------------------------------------------------------------

def _get_git_hash() -> str | None:
    """Return the short git SHA of HEAD, or None if not in a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


# ---------------------------------------------------------------------------
# ExperimentTracker
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """Registry and persistence layer for experiment results.

    Maintains an ordered history of ExperimentResult objects with
    JSON round-trip serialisation.  Provides comparison utilities
    for evaluating the impact of configuration changes.

    Thread Safety:
        Not thread-safe.  If concurrent writes are needed, protect
        calls to ``register``, ``record``, and ``save`` with a lock.
    """

    def __init__(self, storage_path: str | Path | None = None):
        """Initialise the tracker.

        Args:
            storage_path: Path to a JSON file for persistence.
                If the file exists, history is loaded on construction.
                If None, the tracker operates in memory only.
        """
        self._configs: dict[str, ExperimentConfig] = {}
        self._results: list[ExperimentResult] = []
        self._storage_path: Path | None = Path(storage_path) if storage_path else None

        if self._storage_path and self._storage_path.exists():
            self.load()

    # -- Registration -------------------------------------------------------

    def register(self, config: ExperimentConfig) -> None:
        """Register an experiment configuration.

        Raises ValueError if a different config with the same name
        already exists.  Re-registering an identical config is a no-op.
        """
        if config.name in self._configs:
            existing = self._configs[config.name]
            if existing.fingerprint != config.fingerprint:
                raise ValueError(
                    f"Config name '{config.name}' already registered with "
                    f"different parameters (fingerprint {existing.fingerprint} "
                    f"vs {config.fingerprint}).  Use a unique name."
                )
            return
        self._configs[config.name] = copy.deepcopy(config)
        logger.info("Registered experiment: %s (seed=%d)", config.name, config.seed)

    # -- Metric computation -------------------------------------------------

    @staticmethod
    def compute_metrics(
        y_true: Sequence[int],
        y_pred: Sequence[int],
        n_classes: int = 4,
    ) -> dict[str, float]:
        """Convenience proxy for ``compute_standard_metrics``."""
        return compute_standard_metrics(y_true, y_pred, n_classes)

    # -- Recording ----------------------------------------------------------

    def record(
        self,
        name: str,
        metrics: dict[str, float],
        timing_sec: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> ExperimentResult:
        """Record the result of a completed experiment.

        Args:
            name: Name of a previously registered config.
            metrics: Evaluation metrics dictionary.
            timing_sec: Wall-clock evaluation time in seconds.
            metadata: Optional additional data (hardware info, notes).

        Returns:
            The constructed ExperimentResult.

        Raises:
            KeyError: If ``name`` has not been registered.
        """
        if name not in self._configs:
            raise KeyError(
                f"Experiment '{name}' not registered.  "
                f"Call tracker.register(config) first."
            )
        config = self._configs[name]
        result = ExperimentResult(
            config=config,
            metrics=metrics,
            timing_sec=timing_sec,
            git_hash=_get_git_hash(),
            metadata=metadata or {},
        )
        self._results.append(result)
        logger.info(
            "Recorded result for '%s': accuracy=%.4f, macro_f1=%.4f (%.2fs)",
            name,
            metrics.get("accuracy", float("nan")),
            metrics.get("macro_f1", float("nan")),
            timing_sec,
        )
        if self._storage_path:
            self.save()
        return result

    # -- Querying -----------------------------------------------------------

    @property
    def configs(self) -> dict[str, ExperimentConfig]:
        """Read-only view of registered configurations."""
        return dict(self._configs)

    @property
    def results(self) -> list[ExperimentResult]:
        """Read-only copy of the result history."""
        return list(self._results)

    def results_for(self, name: str) -> list[ExperimentResult]:
        """Return all results for a given experiment name, in order."""
        return [r for r in self._results if r.config.name == name]

    def best_result(self, metric: str = "accuracy", higher_is_better: bool = True) -> ExperimentResult | None:
        """Return the single best result across all experiments.

        Args:
            metric: Metric key to optimise.
            higher_is_better: Direction of optimisation.

        Returns:
            The best ExperimentResult, or None if no results exist.
        """
        if not self._results:
            return None
        key = (lambda r: r.metrics.get(metric, float("-inf"))) if higher_is_better else (
            lambda r: -r.metrics.get(metric, float("inf"))
        )
        return max(self._results, key=key)

    # -- Comparison ---------------------------------------------------------

    def compare(
        self,
        name_a: str,
        name_b: str,
        metrics: Sequence[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Compare the latest results of two experiments.

        Returns a nested dictionary::

            {
                "accuracy": {"a": 0.85, "b": 0.82, "delta": 0.03, "relative_pct": 3.66},
                "macro_f1": { ... },
            }

        Args:
            name_a: First experiment name.
            name_b: Second experiment name.
            metrics: Metric keys to compare (default: all shared keys).

        Returns:
            Comparison dictionary.

        Raises:
            KeyError: If either experiment has no recorded results.
        """
        results_a = self.results_for(name_a)
        results_b = self.results_for(name_b)
        if not results_a:
            raise KeyError(f"No results recorded for experiment '{name_a}'")
        if not results_b:
            raise KeyError(f"No results recorded for experiment '{name_b}'")

        latest_a = results_a[-1]
        latest_b = results_b[-1]

        if metrics is None:
            shared_keys = set(latest_a.metrics.keys()) & set(latest_b.metrics.keys())
            metrics = sorted(shared_keys)

        comparison: dict[str, dict[str, float]] = {}
        for m in metrics:
            val_a = latest_a.metrics.get(m, 0.0)
            val_b = latest_b.metrics.get(m, 0.0)
            delta = val_a - val_b
            relative_pct = (delta / abs(val_b) * 100.0) if val_b != 0.0 else float("inf") if delta != 0.0 else 0.0
            comparison[m] = {
                "a": val_a,
                "b": val_b,
                "delta": delta,
                "relative_pct": relative_pct,
            }
        return comparison

    # -- Seed management ----------------------------------------------------

    @staticmethod
    def set_seed(seed: int) -> None:
        """Set the global numpy random seed for reproducibility.

        This does NOT affect Python's built-in ``random`` module.
        For full determinism, also set ``PYTHONHASHSEED=seed`` in
        the environment before process start.
        """
        np.random.seed(seed)
        logger.debug("Numpy random seed set to %d", seed)

    # -- Persistence --------------------------------------------------------

    def save(self, path: str | Path | None = None) -> Path:
        """Serialise tracker state to JSON.

        Args:
            path: Override the default storage path.

        Returns:
            The path that was written.

        Raises:
            ValueError: If no storage path is configured or provided.
        """
        target = Path(path) if path else self._storage_path
        if target is None:
            raise ValueError("No storage path configured.  Pass a path or set storage_path on init.")

        data = {
            "schema_version": 1,
            "configs": {name: asdict(cfg) for name, cfg in self._configs.items()},
            "results": [r.to_dict() for r in self._results],
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Saved %d configs and %d results to %s", len(self._configs), len(self._results), target)
        return target

    def load(self, path: str | Path | None = None) -> None:
        """Load tracker state from JSON, replacing current state.

        Args:
            path: Override the default storage path.

        Raises:
            ValueError: If no storage path is configured or provided.
            FileNotFoundError: If the file does not exist.
        """
        target = Path(path) if path else self._storage_path
        if target is None:
            raise ValueError("No storage path configured.")
        if not target.exists():
            raise FileNotFoundError(f"Experiment history file not found: {target}")

        with open(target, encoding="utf-8") as f:
            data = json.load(f)

        version = data.get("schema_version", 1)
        if version != 1:
            logger.warning("Unknown schema version %d; attempting best-effort load", version)

        self._configs = {
            name: ExperimentConfig(**cfg_data)
            for name, cfg_data in data.get("configs", {}).items()
        }
        self._results = [
            ExperimentResult.from_dict(r) for r in data.get("results", [])
        ]
        logger.info("Loaded %d configs and %d results from %s", len(self._configs), len(self._results), target)


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    """Result of a single ablation (one feature disabled).

    Attributes:
        flag_name: The ablation flag that was toggled off.
        config: The ablation ExperimentConfig.
        metrics: Evaluation metrics with the flag disabled.
        delta: Metric deltas relative to the base (positive means
            the base is better, i.e. the feature helps).
        timing_sec: Wall-clock duration of this ablation run.
    """

    flag_name: str
    config: ExperimentConfig
    metrics: dict[str, float]
    delta: dict[str, float]
    timing_sec: float


@dataclass
class AblationReport:
    """Complete ablation study over all flags.

    Attributes:
        base_config: The full-feature baseline configuration.
        base_metrics: Metrics with all features enabled.
        base_timing_sec: Timing for the baseline run.
        ablations: One AblationResult per toggled flag.
        feature_importance: Flags sorted by impact (most impactful first),
            measured by accuracy delta.
        total_timing_sec: Total wall-clock time for the full study.
    """

    base_config: ExperimentConfig
    base_metrics: dict[str, float]
    base_timing_sec: float
    ablations: list[AblationResult]
    feature_importance: list[tuple[str, float]]
    total_timing_sec: float


class AblationRunner:
    """Systematic leave-one-out ablation over boolean feature flags.

    For a base configuration with N ablation flags set to True, this
    runner executes N+1 evaluations: one baseline (all flags on) and
    N ablations (one flag off at a time).  The impact of each feature
    is measured as the delta between the baseline and its ablation.

    The ``eval_fn`` callback has the signature::

        def eval_fn(config: ExperimentConfig) -> tuple[dict[str, float], float]

    It receives the (possibly modified) config, runs the evaluation,
    and returns ``(metrics_dict, timing_seconds)``.  The runner does
    NOT modify global state; seed management is the callback's
    responsibility (``config.seed`` is always available).

    Args:
        base_config: Baseline configuration with all features enabled.
        eval_fn: Evaluation callback.
        primary_metric: Metric key used to rank feature importance.
        tracker: Optional ExperimentTracker for automatic recording.
    """

    def __init__(
        self,
        base_config: ExperimentConfig,
        eval_fn: Callable[[ExperimentConfig], tuple[dict[str, float], float]],
        primary_metric: str = "accuracy",
        tracker: ExperimentTracker | None = None,
    ):
        self.base_config = base_config
        self.eval_fn = eval_fn
        self.primary_metric = primary_metric
        self.tracker = tracker

        # Validate: at least one ablation flag should be True
        active_flags = [k for k, v in base_config.ablation_flags.items() if v]
        if not active_flags:
            logger.warning(
                "Base config '%s' has no active ablation flags; "
                "ablation study will only run the baseline.",
                base_config.name,
            )
        self._active_flags = active_flags

    def run(self) -> AblationReport:
        """Execute the full ablation study.

        Returns:
            AblationReport with baseline metrics, per-flag deltas,
            and a feature importance ranking.
        """
        total_start = time.time()

        # -- Baseline ---------------------------------------------------------
        logger.info("Ablation study: running baseline '%s'", self.base_config.name)
        ExperimentTracker.set_seed(self.base_config.seed)
        base_metrics, base_timing = self.eval_fn(self.base_config)

        if self.tracker:
            self.tracker.register(self.base_config)
            self.tracker.record(
                self.base_config.name,
                base_metrics,
                timing_sec=base_timing,
                metadata={"ablation_role": "baseline"},
            )

        logger.info(
            "Baseline: %s=%.4f (%.2fs)",
            self.primary_metric,
            base_metrics.get(self.primary_metric, float("nan")),
            base_timing,
        )

        # -- Ablations --------------------------------------------------------
        ablations: list[AblationResult] = []
        for flag_name in self._active_flags:
            ablated_flags = dict(self.base_config.ablation_flags)
            ablated_flags[flag_name] = False

            ablated_config = self.base_config.derive(
                name=f"{self.base_config.name}__no_{flag_name}",
                ablation_flags=ablated_flags,
            )

            logger.info("Ablation: disabling '%s'", flag_name)
            ExperimentTracker.set_seed(self.base_config.seed)
            abl_metrics, abl_timing = self.eval_fn(ablated_config)

            # Compute deltas (positive = base is better = feature helps)
            delta = {
                k: base_metrics.get(k, 0.0) - abl_metrics.get(k, 0.0)
                for k in base_metrics
            }

            ablations.append(AblationResult(
                flag_name=flag_name,
                config=ablated_config,
                metrics=abl_metrics,
                delta=delta,
                timing_sec=abl_timing,
            ))

            if self.tracker:
                self.tracker.register(ablated_config)
                self.tracker.record(
                    ablated_config.name,
                    abl_metrics,
                    timing_sec=abl_timing,
                    metadata={"ablation_role": "ablated", "disabled_flag": flag_name},
                )

            logger.info(
                "  %s=%.4f (delta=%.4f, %.2fs)",
                self.primary_metric,
                abl_metrics.get(self.primary_metric, float("nan")),
                delta.get(self.primary_metric, float("nan")),
                abl_timing,
            )

        # -- Feature importance ranking ---------------------------------------
        feature_importance = sorted(
            [(a.flag_name, a.delta.get(self.primary_metric, 0.0)) for a in ablations],
            key=lambda pair: abs(pair[1]),
            reverse=True,
        )

        total_timing = time.time() - total_start
        logger.info("Ablation study complete: %d ablations in %.2fs", len(ablations), total_timing)

        return AblationReport(
            base_config=self.base_config,
            base_metrics=base_metrics,
            base_timing_sec=base_timing,
            ablations=ablations,
            feature_importance=feature_importance,
            total_timing_sec=total_timing,
        )


# ---------------------------------------------------------------------------
# Human-readable report formatting
# ---------------------------------------------------------------------------

def format_ablation_report(report: AblationReport, primary_metric: str = "accuracy") -> str:
    """Format an ablation report as a human-readable string.

    Suitable for logging, paper appendices, or terminal display.
    The output includes a baseline summary, per-ablation deltas,
    and a ranked feature importance table.

    Args:
        report: The AblationReport to format.
        primary_metric: Metric key highlighted in the summary.

    Returns:
        Multi-line formatted string.
    """
    width = 74
    lines: list[str] = []

    # -- Header ---------------------------------------------------------------
    lines.append("=" * width)
    lines.append("ABLATION STUDY REPORT")
    lines.append("=" * width)
    lines.append(f"  Experiment : {report.base_config.name}")
    lines.append(f"  Seed       : {report.base_config.seed}")
    lines.append(f"  Description: {report.base_config.description or '(none)'}")
    lines.append(f"  Ablations  : {len(report.ablations)}")
    lines.append(f"  Total time : {report.total_timing_sec:.2f}s")
    lines.append("")

    # -- Baseline metrics -----------------------------------------------------
    lines.append("-" * width)
    lines.append("BASELINE METRICS (all features enabled)")
    lines.append("-" * width)
    for key in sorted(report.base_metrics.keys()):
        lines.append(f"  {key:<24s} {report.base_metrics[key]:>10.4f}")
    lines.append(f"  {'timing_sec':<24s} {report.base_timing_sec:>10.2f}")
    lines.append("")

    # -- Per-ablation results -------------------------------------------------
    if report.ablations:
        lines.append("-" * width)
        lines.append("ABLATION RESULTS (one feature disabled at a time)")
        lines.append("-" * width)
        lines.append("")

        for abl in report.ablations:
            direction = "HURTS" if abl.delta.get(primary_metric, 0.0) > 0 else "HELPS" if abl.delta.get(primary_metric, 0.0) < 0 else "NEUTRAL"
            pm_delta = abl.delta.get(primary_metric, 0.0)
            lines.append(f"  [Disable: {abl.flag_name}]  ({direction}: {primary_metric} delta = {pm_delta:+.4f})")
            for key in sorted(abl.metrics.keys()):
                base_val = report.base_metrics.get(key, 0.0)
                abl_val = abl.metrics[key]
                delta = abl.delta.get(key, 0.0)
                sign = "+" if delta >= 0 else ""
                lines.append(f"    {key:<22s}  base={base_val:.4f}  ablated={abl_val:.4f}  delta={sign}{delta:.4f}")
            lines.append(f"    {'timing_sec':<22s}  {abl.timing_sec:.2f}s")
            lines.append("")

    # -- Feature importance ---------------------------------------------------
    if report.feature_importance:
        lines.append("-" * width)
        lines.append(f"FEATURE IMPORTANCE (ranked by |{primary_metric} delta|)")
        lines.append("-" * width)
        lines.append(f"  {'Rank':<6s} {'Feature':<28s} {'Delta':>10s} {'|Delta|':>10s} {'Verdict'}")
        lines.append(f"  {'----':<6s} {'-------':<28s} {'-----':>10s} {'-------':>10s} {'-------'}")

        for rank, (flag_name, delta) in enumerate(report.feature_importance, start=1):
            abs_delta = abs(delta)
            if delta > 0.005:
                verdict = "HELPFUL"
            elif delta < -0.005:
                verdict = "HARMFUL"
            else:
                verdict = "NEGLIGIBLE"
            lines.append(
                f"  {rank:<6d} {flag_name:<28s} {delta:>+10.4f} {abs_delta:>10.4f} {verdict}"
            )
        lines.append("")

    # -- Hyperparameters for reproducibility ----------------------------------
    if report.base_config.hyperparams:
        lines.append("-" * width)
        lines.append("HYPERPARAMETERS")
        lines.append("-" * width)
        for key in sorted(report.base_config.hyperparams.keys()):
            lines.append(f"  {key:<28s} = {report.base_config.hyperparams[key]}")
        lines.append("")

    lines.append("=" * width)
    return "\n".join(lines)
