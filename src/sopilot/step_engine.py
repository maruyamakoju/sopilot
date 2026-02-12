from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np

from .utils import normalize_rows as _normalize_rows

logger = logging.getLogger(__name__)

STRUCT_DUP_SCALE = 1.2
STRUCT_ORDER_SCALE = 2.0
STRUCT_DRIFT_SCALE = 1.0
STRUCT_CONF_MEAN_SCALE = 10.0
STRUCT_CONF_GAP_SCALE = 20.0
STRUCT_STRETCH_SCALE = 1.2
STRUCT_ADAPTIVE_MARGIN = 0.01
MAX_DTW_CLIPS = 2000


# ---------------------------------------------------------------------------
# Neural component loaders (lazy, cached with path tracking)
# ---------------------------------------------------------------------------


class NeuralModelCache:
    """Lazy-loaded cache for neural models with path-based invalidation.

    Tracks the source path of each cached model so that loading from a
    different directory (e.g. after retraining) correctly replaces the cache
    instead of returning a stale model.
    """

    def __init__(self) -> None:
        self._models: dict[str, object] = {}
        self._paths: dict[str, Path] = {}

    def get(self, key: str, path: Path) -> object | None:
        if key in self._models and self._paths.get(key) == path:
            return self._models[key]
        return None

    def put(self, key: str, path: Path, model: object) -> None:
        self._models[key] = model
        self._paths[key] = path

    def invalidate(self) -> None:
        self._models.clear()
        self._paths.clear()


_model_cache = NeuralModelCache()


def _load_cached_model(
    key: str,
    model_dir: Path,
    filename: str,
    load_fn_getter,
    *,
    device: str | None = None,
) -> object | None:
    """Generic loader: check cache, check file, import + load, cache result."""
    path = model_dir / filename
    cached = _model_cache.get(key, path)
    if cached is not None:
        return cached
    if not path.exists():
        return None
    try:
        load_fn = load_fn_getter()
        model = load_fn(path, device=device) if device is not None else load_fn(path)
        _model_cache.put(key, path, model)
        return model
    except Exception as e:
        logger.warning("Failed to load %s: %s", key, e)
        return None


# Registry of neural model loaders: (cache_key, filename, module_path, attr_name).
# Each entry is lazily imported and loaded via _load_cached_model.
_NEURAL_MODEL_REGISTRY: list[tuple[str, str, str, str]] = [
    ("segmenter", "step_segmenter.pt", ".nn.step_segmenter", "load_segmenter"),
    ("scoring", "scoring_head.pt", ".nn.scoring_head", "load_scoring_head"),
    ("calibrator", "isotonic_calibrator.npz", ".nn.scoring_head", "IsotonicCalibrator.load"),
    ("asformer", "asformer.pt", ".nn.asformer", "load_asformer"),
]


def _load_neural_model(key: str, model_dir: Path, device: str | None = None):
    """Load a neural model from the registry by key."""
    for cache_key, filename, module_path, attr_path in _NEURAL_MODEL_REGISTRY:
        if cache_key == key:

            def _get_loader(_mp=module_path, _ap=attr_path):
                import importlib

                mod = importlib.import_module(_mp, package="sopilot")
                parts = _ap.split(".")
                obj = getattr(mod, parts[0])
                for p in parts[1:]:
                    obj = getattr(obj, p)
                return obj

            return _load_cached_model(cache_key, model_dir, filename, _get_loader, device=device)
    raise KeyError(f"Unknown neural model: {key}")


def _load_neural_segmenter(model_dir: Path, device: str = "cpu"):
    return _load_neural_model("segmenter", model_dir, device=device)


def _load_neural_scoring(model_dir: Path, device: str = "cpu"):
    return _load_neural_model("scoring", model_dir, device=device)


def _load_neural_calibrator(model_dir: Path):
    return _load_neural_model("calibrator", model_dir)


def _load_neural_asformer(model_dir: Path, device: str = "cpu"):
    return _load_neural_model("asformer", model_dir, device=device)


def _load_neural_conformal(model_dir: Path):
    def _get_loader():
        def _load_conformal(path):
            from .nn.conformal import SplitConformalPredictor

            cp = SplitConformalPredictor()
            data = np.load(path)
            cp._quantile = float(data["quantile"].item() if hasattr(data["quantile"], "item") else data["quantile"])
            cp._calibrated = True
            return cp

        return _load_conformal

    return _load_cached_model("conformal", model_dir, "conformal_predictor.npz", _get_loader)


def invalidate_neural_caches() -> None:
    """Clear all cached neural models (call after retraining)."""
    _model_cache.invalidate()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    an = a / max(float(np.linalg.norm(a)), 1e-12)
    bn = b / max(float(np.linalg.norm(b)), 1e-12)
    return float(np.dot(an, bn))


@dataclass
class AlignmentResult:
    path: list[tuple[int, int, float]]
    mean_cost: float


def detect_step_boundaries(
    embeddings: np.ndarray,
    threshold_factor: float,
    min_step_clips: int,
    *,
    neural_model_dir: Path | None = None,
    neural_device: str = "cpu",
) -> list[int]:
    n = int(embeddings.shape[0])
    if n <= 1:
        return [0, n]

    # Try neural segmenters if model dir provided
    # Priority: ASFormer (transformer) → MS-TCN++ → threshold heuristic
    if neural_model_dir is not None:
        # Try ASFormer first (state-of-the-art transformer segmenter)
        asformer = _load_neural_asformer(neural_model_dir, device=neural_device)
        if asformer is not None:
            try:
                from .nn.asformer import predict_boundaries_asformer

                boundaries, _ = predict_boundaries_asformer(
                    asformer,
                    embeddings,
                    min_step_clips=min_step_clips,
                    device=neural_device,
                )
                logger.debug("ASFormer segmenter produced %d boundaries", len(boundaries))
                return boundaries
            except Exception as e:
                logger.warning("ASFormer segmenter failed, trying MS-TCN: %s", e)

        # Fallback to MS-TCN++ (dilated convolution segmenter)
        segmenter = _load_neural_segmenter(neural_model_dir, device=neural_device)
        if segmenter is not None:
            try:
                boundaries = segmenter.predict_boundaries(embeddings, min_step_clips=min_step_clips)
                logger.debug("MS-TCN segmenter produced %d boundaries", len(boundaries))
                return boundaries
            except Exception as e:
                logger.warning("MS-TCN segmenter failed, falling back to threshold: %s", e)

    # Fallback: threshold-based detection
    normed = _normalize_rows(embeddings.astype(np.float32))
    sims = np.sum(normed[1:] * normed[:-1], axis=1)
    dists = 1.0 - sims

    mean = float(np.mean(dists))
    std = float(np.std(dists))
    threshold = mean + threshold_factor * std

    raw_points = [i + 1 for i, d in enumerate(dists) if float(d) >= threshold]

    filtered: list[int] = []
    last = 0
    for point in raw_points:
        if point - last >= max(1, min_step_clips):
            filtered.append(point)
            last = point

    boundaries = [0] + filtered
    if boundaries[-1] != n:
        boundaries.append(n)
    return boundaries


def dtw_align(gold: np.ndarray, trainee: np.ndarray) -> AlignmentResult:
    g = _normalize_rows(gold.astype(np.float32))
    t = _normalize_rows(trainee.astype(np.float32))

    m = int(g.shape[0])
    n = int(t.shape[0])
    if m == 0 or n == 0:
        return AlignmentResult(path=[], mean_cost=1.0)

    if m > MAX_DTW_CLIPS or n > MAX_DTW_CLIPS:
        raise ValueError(f"clip count exceeds DTW limit: gold={m}, trainee={n}, max={MAX_DTW_CLIPS}")

    cost = (1.0 - (g @ t.T)).astype(np.float64)

    dp = np.full((m + 1, n + 1), np.inf, dtype=np.float64)
    trace = np.zeros((m + 1, n + 1), dtype=np.int8)
    dp[0, 0] = 0.0

    # Anti-diagonal wavefront: all cells on the same anti-diagonal are
    # independent because their predecessors lie on earlier diagonals.
    for d in range(2, m + n + 2):
        i_start = max(1, d - n)
        i_end = min(m, d - 1)
        if i_start > i_end:
            continue
        ii = np.arange(i_start, i_end + 1)
        jj = d - ii

        diag = dp[ii - 1, jj - 1]
        above = dp[ii - 1, jj]
        left = dp[ii, jj - 1]

        min_da = np.minimum(diag, above)
        min_all = np.minimum(min_da, left)

        dp[ii, jj] = cost[ii - 1, jj - 1] + min_all

        # trace: 0=diagonal, 1=above, 2=left
        tr = np.where(min_all == diag, np.int8(0), np.where(min_all == above, np.int8(1), np.int8(2)))
        trace[ii, jj] = tr

    i, j = m, n
    rev_path: list[tuple[int, int, float]] = []
    while i > 0 and j > 0:
        sim = 1.0 - float(cost[i - 1, j - 1])
        rev_path.append((i - 1, j - 1, sim))
        move = int(trace[i, j])
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1

    while i > 0:
        sim = _cosine_similarity(g[i - 1], t[0])
        rev_path.append((i - 1, 0, sim))
        i -= 1

    while j > 0:
        sim = _cosine_similarity(g[0], t[j - 1])
        rev_path.append((0, j - 1, sim))
        j -= 1

    path = list(reversed(rev_path))
    mean_cost = float(dp[m, n] / max(1, len(path)))
    return AlignmentResult(path=path, mean_cost=mean_cost)


def _clip_to_step(boundaries: list[int], clip_count: int) -> list[int]:
    mapping = [0] * max(clip_count, 0)
    for sidx in range(max(0, len(boundaries) - 1)):
        start = int(boundaries[sidx])
        end = int(boundaries[sidx + 1])
        for c in range(start, min(end, clip_count)):
            mapping[c] = sidx
    return mapping


def _clip_time_range(meta: list[dict], indices: list[int]) -> dict:
    if not indices:
        return {"start_sec": None, "end_sec": None}
    s = min(indices)
    e = max(indices)
    return {
        "start_sec": float(meta[s]["start_sec"]),
        "end_sec": float(meta[e]["end_sec"]),
    }


def _best_match_structure(
    gold_norm: np.ndarray,
    trainee_norm: np.ndarray,
    low_similarity_threshold: float,
) -> dict:
    m = int(gold_norm.shape[0])
    n = int(trainee_norm.shape[0])
    if m == 0 or n == 0:
        return {
            "duplicate_ratio": 1.0,
            "order_violation_ratio": 1.0,
            "temporal_drift": 1.0,
            "confidence_loss": 1.0,
            "local_similarity_gap": 1.0,
            "adaptive_low_similarity_threshold": float(low_similarity_threshold),
            "hard_miss_ratio": 1.0,
        }

    sims = gold_norm @ trainee_norm.T
    best_idx = np.argmax(sims, axis=1).astype(np.int32)
    best_sims = np.max(sims, axis=1).astype(np.float32)

    duplicates = int(max(0, m - int(np.unique(best_idx).shape[0])))
    duplicate_ratio = float(duplicates / max(1, m))

    violations = 0
    for i in range(1, m):
        if int(best_idx[i]) < int(best_idx[i - 1]):
            violations += 1
    order_violation_ratio = float(violations / max(1, m - 1))

    expected = np.linspace(0, max(n - 1, 0), num=max(1, m), dtype=np.float32)
    temporal_drift = float(np.mean(np.abs(best_idx.astype(np.float32) - expected)) / max(1.0, float(n - 1)))

    mean_best = float(np.mean(best_sims))
    median_best = float(np.median(best_sims))
    min_best = float(np.min(best_sims))
    confidence_loss = float(max(0.0, 1.0 - mean_best))
    local_similarity_gap = float(max(0.0, median_best - min_best))

    adaptive_low = float(max(low_similarity_threshold, median_best - STRUCT_ADAPTIVE_MARGIN))
    hard_miss_ratio = float(np.mean(best_sims < adaptive_low))

    return {
        "duplicate_ratio": duplicate_ratio,
        "order_violation_ratio": order_violation_ratio,
        "temporal_drift": temporal_drift,
        "confidence_loss": confidence_loss,
        "local_similarity_gap": local_similarity_gap,
        "adaptive_low_similarity_threshold": adaptive_low,
        "hard_miss_ratio": hard_miss_ratio,
    }


def _hard_dtw_fallback(
    gold_embeddings: np.ndarray,
    trainee_embeddings: np.ndarray,
    use_gpu_dtw: bool | None,
) -> AlignmentResult:
    """Hard DTW with GPU/CPU fallback chain."""
    if use_gpu_dtw is None:
        use_gpu_dtw = os.getenv("SOPILOT_DTW_USE_GPU", "auto").strip().lower() in {"1", "true", "auto"}
    if use_gpu_dtw:
        try:
            from .dtw_gpu import dtw_align_auto

            gpu_result = dtw_align_auto(gold_embeddings, trainee_embeddings, prefer_gpu=True)
            return AlignmentResult(path=gpu_result.path, mean_cost=gpu_result.mean_cost)
        except Exception as e:
            logger.warning("GPU DTW failed, falling back to CPU: %s", e)
    return dtw_align(gold_embeddings, trainee_embeddings)


def _resolve_alignment(
    gold_embeddings: np.ndarray,
    trainee_embeddings: np.ndarray,
    gold_norm: np.ndarray,
    trainee_norm: np.ndarray,
    use_gpu_dtw: bool | None,
    *,
    neural_mode: bool,
    neural_model_dir: Path | None,
    neural_soft_dtw_gamma: float,
    neural_cuda_dtw: bool,
    neural_ot_alignment: bool,
) -> AlignmentResult:
    """Resolve alignment via priority chain: OT → CUDA Soft-DTW → CPU Soft-DTW → hard DTW."""
    if not (neural_mode and neural_model_dir is not None):
        return _hard_dtw_fallback(gold_embeddings, trainee_embeddings, use_gpu_dtw)

    # Option 1: Optimal Transport alignment (Sinkhorn + Gromov-Wasserstein)
    if neural_ot_alignment:
        try:
            import torch

            from .nn.optimal_transport import HierarchicalOTAlignment

            ot_aligner = HierarchicalOTAlignment(
                sinkhorn_reg=0.05, n_clusters=min(8, max(2, gold_embeddings.shape[0] // 5))
            )
            g_t = torch.from_numpy(gold_embeddings.astype(np.float32))
            t_t = torch.from_numpy(trainee_embeddings.astype(np.float32))
            ot_plan = ot_aligner(g_t, t_t)
            ot_np = ot_plan.detach().cpu().numpy()
            path_list = []
            for i in range(ot_np.shape[0]):
                j = int(np.argmax(ot_np[i]))
                sim = float(np.dot(gold_norm[i], trainee_norm[j]))
                path_list.append((i, j, sim))
            mean_c = float(1.0 - np.mean([s for _, _, s in path_list])) if path_list else 1.0
            logger.debug("Using OT alignment (Sinkhorn + GW)")
            return AlignmentResult(path=path_list, mean_cost=mean_c)
        except Exception as e:
            logger.warning("OT alignment failed: %s", e)

    # Option 2: CUDA-accelerated Soft-DTW (distance only, path via CPU fallback)
    if neural_cuda_dtw:
        try:
            import torch

            from .nn.soft_dtw_cuda import SoftDTWCuda

            g_t = torch.from_numpy(gold_embeddings.astype(np.float32)).unsqueeze(0)
            t_t = torch.from_numpy(trainee_embeddings.astype(np.float32)).unsqueeze(0)
            sdtw = SoftDTWCuda(gamma=neural_soft_dtw_gamma)
            with torch.no_grad():
                dist = sdtw(g_t, t_t)
            logger.debug("CUDA Soft-DTW dist=%.4f, getting path via CPU", dist.item())
        except Exception as e:
            logger.debug("CUDA Soft-DTW not available: %s", e)

    # Option 3: CPU Soft-DTW (always provides path + alignment matrix)
    try:
        from .nn.soft_dtw import soft_dtw_align_numpy

        soft_path, soft_cost, _ = soft_dtw_align_numpy(gold_embeddings, trainee_embeddings, gamma=neural_soft_dtw_gamma)
        logger.debug("Using CPU Soft-DTW alignment (γ=%.2f)", neural_soft_dtw_gamma)
        return AlignmentResult(path=soft_path, mean_cost=soft_cost)
    except Exception as e:
        logger.warning("Soft-DTW failed, falling back to hard DTW: %s", e)

    return _hard_dtw_fallback(gold_embeddings, trainee_embeddings, use_gpu_dtw)


def _detect_deviations(
    path: list[tuple[int, int, float]],
    gold_steps: list[int],
    trainee_steps: list[int],
    gold_meta: list[dict],
    trainee_meta: list[dict],
    g_step_map: list[int],
    effective_threshold: float,
) -> tuple[int, int, float, list[dict]]:
    """Detect misses, swaps, and execution deviations from alignment path.

    Returns (miss_count, swap_count, deviation_ratio, deviations_list).
    """
    pairs_by_g_step: dict[int, list[tuple[int, int, float]]] = {}
    for gi, tj, sim in path:
        gs = g_step_map[gi]
        pairs_by_g_step.setdefault(gs, []).append((gi, tj, sim))

    miss = 0
    deviations: list[dict] = []
    step_target_positions: list[tuple[int, float]] = []

    for gs in range(max(0, len(gold_steps) - 1)):
        pairs = pairs_by_g_step.get(gs, [])
        if not pairs:
            miss += 1
            deviations.append(
                {
                    "type": "step_missing",
                    "gold_step": gs,
                    "gold_time": _clip_time_range(gold_meta, list(range(gold_steps[gs], gold_steps[gs + 1]))),
                    "trainee_time": {"start_sec": None, "end_sec": None},
                    "confidence": 1.0,
                    "reason": "no aligned trainee clips",
                }
            )
            continue

        sims = [p[2] for p in pairs]
        if max(sims) < effective_threshold:
            miss += 1
            deviations.append(
                {
                    "type": "step_missing",
                    "gold_step": gs,
                    "gold_time": _clip_time_range(gold_meta, list(range(gold_steps[gs], gold_steps[gs + 1]))),
                    "trainee_time": _clip_time_range(trainee_meta, [p[1] for p in pairs]),
                    "confidence": float(1.0 - max(sims)),
                    "reason": "aligned similarity below threshold",
                }
            )

        step_target_positions.append((gs, float(median([p[1] for p in pairs]))))

    # Detect order swaps
    swap = 0
    for idx in range(1, len(step_target_positions)):
        prev = step_target_positions[idx - 1][1]
        cur = step_target_positions[idx][1]
        if cur + 1.0 < prev:
            swap += 1
            gs = step_target_positions[idx][0]
            deviations.append(
                {
                    "type": "order_swap",
                    "gold_step": gs,
                    "gold_time": _clip_time_range(gold_meta, list(range(gold_steps[gs], gold_steps[gs + 1]))),
                    "trainee_time": {"start_sec": None, "end_sec": None},
                    "confidence": float(min(1.0, (prev - cur) / max(prev, 1.0))),
                    "reason": "step order likely inverted",
                }
            )

    # Detect execution deviations (contiguous low-similarity regions)
    low_pairs = [(gi, tj, sim) for gi, tj, sim in path if sim < effective_threshold]
    deviation_ratio = float(len(low_pairs) / max(1, len(path)))

    if low_pairs:
        chunk: list[tuple[int, int, float]] = [low_pairs[0]]
        for pair in low_pairs[1:]:
            pg, pt, _ = chunk[-1]
            cg, ct, _ = pair
            if abs(cg - pg) <= 1 and abs(ct - pt) <= 1:
                chunk.append(pair)
            else:
                _emit_deviation_chunk(chunk, g_step_map, gold_meta, trainee_meta, deviations)
                chunk = [pair]
        _emit_deviation_chunk(chunk, g_step_map, gold_meta, trainee_meta, deviations)

    return miss, swap, deviation_ratio, deviations


def _emit_deviation_chunk(
    chunk: list[tuple[int, int, float]],
    g_step_map: list[int],
    gold_meta: list[dict],
    trainee_meta: list[dict],
    deviations: list[dict],
) -> None:
    """Append an execution_deviation entry from a contiguous chunk of low-similarity pairs."""
    g_idxs = [x[0] for x in chunk]
    t_idxs = [x[1] for x in chunk]
    deviations.append(
        {
            "type": "execution_deviation",
            "gold_step": g_step_map[g_idxs[0]],
            "gold_time": _clip_time_range(gold_meta, g_idxs),
            "trainee_time": _clip_time_range(trainee_meta, t_idxs),
            "confidence": float(1.0 - np.mean([x[2] for x in chunk])),
            "reason": "low local similarity",
        }
    )


def _compute_sop_score(
    miss: int,
    swap: int,
    deviation: float,
    over_time: float,
    temporal_warp: float,
    path_stretch: float,
    structure: dict,
    w_miss: float,
    w_swap: float,
    w_dev: float,
    w_time: float,
    w_warp: float,
) -> float:
    """Compute the final 0-100 SOP compliance score."""
    duplicate_penalty = w_miss * STRUCT_DUP_SCALE * structure["duplicate_ratio"]
    order_penalty = w_swap * STRUCT_ORDER_SCALE * structure["order_violation_ratio"]
    drift_penalty = w_time * STRUCT_DRIFT_SCALE * structure["temporal_drift"]
    confidence_penalty = w_dev * (
        STRUCT_CONF_MEAN_SCALE * structure["confidence_loss"]
        + STRUCT_CONF_GAP_SCALE * structure["local_similarity_gap"]
    )
    stretch_penalty = w_time * STRUCT_STRETCH_SCALE * path_stretch

    score = 100.0 - (
        w_miss * miss
        + w_swap * swap
        + w_dev * deviation
        + w_time * over_time
        + w_warp * temporal_warp
        + duplicate_penalty
        + order_penalty
        + drift_penalty
        + confidence_penalty
        + stretch_penalty
    )
    return float(np.clip(score, 0.0, 100.0))


def _apply_neural_scoring(
    metrics_dict: dict,
    result: dict,
    neural_model_dir: Path,
    neural_device: str,
    neural_uncertainty_samples: int,
    neural_calibration_enabled: bool,
) -> None:
    """Run neural scoring head with uncertainty, calibration, and conformal prediction."""
    scoring_head = _load_neural_scoring(neural_model_dir, device=neural_device)
    if scoring_head is None:
        return

    try:
        from .nn.scoring_head import metrics_to_tensor

        metrics_tensor = metrics_to_tensor(metrics_dict, device=neural_device)
        uncertainty = scoring_head.predict_with_uncertainty(metrics_tensor, n_samples=neural_uncertainty_samples)

        calibrated_score = None
        if neural_calibration_enabled:
            calibrator = _load_neural_calibrator(neural_model_dir)
            if calibrator is not None:
                calibrated_score = calibrator.calibrate(uncertainty["score"])

        conformal_lo, conformal_hi = None, None
        conformal_predictor = _load_neural_conformal(neural_model_dir)
        if conformal_predictor is not None:
            try:
                _, conformal_lo, conformal_hi = conformal_predictor.predict(uncertainty["score"])
                conformal_lo = float(max(0.0, conformal_lo))
                conformal_hi = float(min(100.0, conformal_hi))
            except Exception as ce:
                logger.warning("Conformal prediction failed: %s", ce)

        result["neural_score"] = {
            "score": uncertainty["score"],
            "uncertainty": uncertainty["uncertainty"],
            "ci_lower": uncertainty["ci_lower"],
            "ci_upper": uncertainty["ci_upper"],
            "calibrated_score": calibrated_score,
            "conformal_ci_lower": conformal_lo,
            "conformal_ci_upper": conformal_hi,
            "n_samples": neural_uncertainty_samples,
        }
        logger.debug(
            "Neural score: %.1f ± %.1f (CI: [%.1f, %.1f])",
            uncertainty["score"],
            uncertainty["uncertainty"],
            uncertainty["ci_lower"],
            uncertainty["ci_upper"],
        )
    except Exception as e:
        logger.warning("Neural scoring failed: %s", e)


def evaluate_sop(
    gold_embeddings: np.ndarray,
    trainee_embeddings: np.ndarray,
    gold_meta: list[dict],
    trainee_meta: list[dict],
    threshold_factor: float,
    min_step_clips: int,
    low_similarity_threshold: float,
    w_miss: float,
    w_swap: float,
    w_dev: float,
    w_time: float,
    w_warp: float = 12.0,
    use_gpu_dtw: bool | None = None,
    *,
    neural_mode: bool = False,
    neural_model_dir: Path | None = None,
    neural_device: str = "cpu",
    neural_soft_dtw_gamma: float = 1.0,
    neural_uncertainty_samples: int = 30,
    neural_calibration_enabled: bool = True,
    neural_cuda_dtw: bool = True,
    neural_ot_alignment: bool = False,
    neural_conformal_alpha: float = 0.1,
) -> dict:
    """Evaluate SOP compliance between gold and trainee videos."""
    gold_norm = _normalize_rows(gold_embeddings.astype(np.float32))
    trainee_norm = _normalize_rows(trainee_embeddings.astype(np.float32))

    # Phase 1: Step boundary detection
    seg_kwargs: dict = {}
    if neural_mode and neural_model_dir is not None:
        seg_kwargs["neural_model_dir"] = neural_model_dir
        seg_kwargs["neural_device"] = neural_device

    gold_steps = detect_step_boundaries(gold_embeddings, threshold_factor, min_step_clips, **seg_kwargs)
    trainee_steps = detect_step_boundaries(trainee_embeddings, threshold_factor, min_step_clips, **seg_kwargs)

    # Phase 2: Alignment
    alignment = _resolve_alignment(
        gold_embeddings,
        trainee_embeddings,
        gold_norm,
        trainee_norm,
        use_gpu_dtw,
        neural_mode=neural_mode,
        neural_model_dir=neural_model_dir,
        neural_soft_dtw_gamma=neural_soft_dtw_gamma,
        neural_cuda_dtw=neural_cuda_dtw,
        neural_ot_alignment=neural_ot_alignment,
    )

    path = alignment.path
    structure = _best_match_structure(gold_norm, trainee_norm, low_similarity_threshold)
    effective_threshold = float(max(low_similarity_threshold, float(structure["adaptive_low_similarity_threshold"])))

    # Phase 3: Deviation detection
    g_step_map = _clip_to_step(gold_steps, int(gold_embeddings.shape[0]))
    t_step_map = _clip_to_step(trainee_steps, int(trainee_embeddings.shape[0]))

    miss, swap, deviation, deviations = _detect_deviations(
        path,
        gold_steps,
        trainee_steps,
        gold_meta,
        trainee_meta,
        g_step_map,
        effective_threshold,
    )

    # Phase 4: Temporal metrics
    n_gold = int(gold_embeddings.shape[0])
    n_trainee = int(trainee_embeddings.shape[0])
    over_time = float(max(0.0, n_trainee - n_gold) / max(1.0, float(n_gold)))

    if path:
        gm = max(1.0, float(n_gold - 1))
        tn = max(1.0, float(n_trainee - 1))
        temporal_warp = float(np.mean([abs((gi / gm) - (tj / tn)) for gi, tj, _ in path]))
    else:
        temporal_warp = 1.0

    path_stretch = float(max(0.0, len(path) - max(n_gold, n_trainee)) / max(1.0, float(max(n_gold, n_trainee))))

    # Phase 5: Score computation
    score = _compute_sop_score(
        miss,
        swap,
        deviation,
        over_time,
        temporal_warp,
        path_stretch,
        structure,
        w_miss,
        w_swap,
        w_dev,
        w_time,
        w_warp,
    )

    metrics_dict = {
        "miss": int(miss),
        "swap": int(swap),
        "deviation": float(deviation),
        "over_time": float(over_time),
        "temporal_warp": float(temporal_warp),
        "path_stretch": float(path_stretch),
        "duplicate_ratio": float(structure["duplicate_ratio"]),
        "order_violation_ratio": float(structure["order_violation_ratio"]),
        "temporal_drift": float(structure["temporal_drift"]),
        "confidence_loss": float(structure["confidence_loss"]),
        "local_similarity_gap": float(structure["local_similarity_gap"]),
        "adaptive_low_similarity_threshold": float(structure["adaptive_low_similarity_threshold"]),
        "effective_low_similarity_threshold": float(effective_threshold),
        "hard_miss_ratio": float(structure["hard_miss_ratio"]),
        "mean_alignment_cost": float(alignment.mean_cost),
    }

    result: dict = {
        "score": score,
        "metrics": metrics_dict,
        "step_boundaries": {"gold": gold_steps, "trainee": trainee_steps},
        "deviations": deviations,
        "alignment_preview": [
            {"gold_clip": gi, "trainee_clip": tj, "similarity": round(float(sim), 4)} for gi, tj, sim in path[:300]
        ],
        "clip_count": {"gold": n_gold, "trainee": n_trainee},
        "step_map_preview": {"gold": g_step_map[:100], "trainee": t_step_map[:100]},
        "neural_mode": neural_mode,
    }

    # Phase 6: Neural scoring with uncertainty
    if neural_mode and neural_model_dir is not None:
        _apply_neural_scoring(
            metrics_dict,
            result,
            neural_model_dir,
            neural_device,
            neural_uncertainty_samples,
            neural_calibration_enabled,
        )

    return result
