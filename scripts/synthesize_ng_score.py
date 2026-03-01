"""
synthesize_ng_score.py
──────────────────────
Standalone script: reads gold embeddings from an existing DB, synthesizes a
"step-skipping" NG trainee sequence, runs DTW + scoring, and prints the full
score JSON.  No server required, no video file needed.

Usage:
    python scripts/synthesize_ng_score.py
    python scripts/synthesize_ng_score.py --db data_tesda_cm/sopilot.db --gold-id 2
    python scripts/synthesize_ng_score.py --out ng_score_result.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _ensure_repo_on_path() -> None:
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def make_ng_embeddings(
    gold_embs: np.ndarray,
    rng: np.random.Generator,
    normalize: Callable[[np.ndarray], np.ndarray],
    n_correct: int = 1,
    n_total: int = 4,
) -> np.ndarray:
    """
    Produce synthetic NG embeddings that represent a trainee who:
      - Executes only the first `n_correct` steps roughly correctly
      - Then performs random/wrong actions for the rest
      - Stops early (total clips << gold clips)

    This reliably produces missing_step (critical) deviations.
    """
    dim = gold_embs.shape[1]
    clips = []

    for i in range(n_total):
        if i < n_correct:
            # Slightly noisy version of gold (step done, but imprecisely)
            noise = rng.normal(0, 0.25, dim).astype(np.float32)
            vec = normalize(gold_embs[i] + noise)
        else:
            # Completely wrong action — orthogonal to gold
            # Use a random vector that's far from all gold embeddings
            raw = rng.standard_normal(dim).astype(np.float32)
            vec = normalize(raw)
        clips.append(vec)

    return np.stack(clips, axis=0)


def apply_deviation_policy(deviations: list[dict], policy: dict[str, str]) -> list[dict]:
    """Inject severity from deviation policy into raw deviations."""
    result = []
    for dev in deviations:
        d = dict(dev)
        dev_type = d.get("type", "")
        d["severity"] = policy.get(dev_type, "quality")
        result.append(d)
    return result


def count_severity(deviations: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {"critical": 0, "quality": 0, "efficiency": 0}
    for d in deviations:
        sev = d.get("severity", "quality")
        if sev in counts:
            counts[sev] += 1
        else:
            counts[sev] = counts.get(sev, 0) + 1
    return counts


def decide(score: float, severity_counts: dict[str, int], pass_score: float, retrain_score: float) -> tuple[str, str]:
    if severity_counts.get("critical", 0) > 0:
        return "fail", "Critical deviation detected"
    if score >= pass_score:
        return "pass", f"score >= pass_score ({pass_score})"
    if score >= retrain_score:
        return "needs_review", "between retrain_score and pass_score"
    return "retrain", f"score < retrain_score ({retrain_score})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize NG score without a running server.")
    parser.add_argument("--db", default="data_tesda_cm/sopilot.db",
                        help="Path to sopilot.db (default: data_tesda_cm/sopilot.db)")
    parser.add_argument("--gold-id", type=int, default=2,
                        help="Video ID of the gold video to score against (default: 2)")
    parser.add_argument("--n-correct", type=int, default=1,
                        help="Number of NG trainee clips that partially match gold (default: 1)")
    parser.add_argument("--n-clips", type=int, default=4,
                        help="Total NG trainee clip count (default: 4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="Write result JSON to this file")
    parser.add_argument("--task-id", default="filter_change")
    args = parser.parse_args()

    _ensure_repo_on_path()
    from sopilot.core.dtw import dtw_align
    from sopilot.core.math_utils import l2_normalize
    from sopilot.core.scoring import ScoreWeights, score_alignment
    from sopilot.core.segmentation import detect_step_boundaries
    from sopilot.database import Database

    db_path = ROOT / args.db if not Path(args.db).is_absolute() else Path(args.db)
    if not db_path.exists():
        # Try other known DB paths
        for fallback in [
            ROOT / "data_tesda_cm" / "sopilot.db",
            ROOT / "data_tesda" / "sopilot.db",
            ROOT / "data" / "sopilot.db",
        ]:
            if fallback.exists():
                db_path = fallback
                print(f"[warn] using fallback DB: {db_path}", file=sys.stderr)
                break
        else:
            sys.exit(f"[error] DB not found: {db_path}")

    print(f"[info] DB: {db_path}", file=sys.stderr)
    db = Database(db_path)

    # Load gold clips
    gold_clips = db.get_video_clips(args.gold_id)
    if not gold_clips:
        sys.exit(f"[error] no clips for gold video_id={args.gold_id}")

    gold_embs = np.array([c["embedding"] for c in gold_clips], dtype=np.float32)
    gold_len = len(gold_embs)
    print(f"[info] gold video_id={args.gold_id}  clips={gold_len}  dim={gold_embs.shape[1]}", file=sys.stderr)

    # Load task profile for policy / thresholds
    profile = db.get_task_profile(args.task_id)
    if profile:
        pass_score = float(profile["pass_score"])
        retrain_score = float(profile["retrain_score"])
        weights_raw = profile.get("default_weights") or {}
        policy = profile.get("deviation_policy") or {}
    else:
        pass_score, retrain_score = 90.0, 80.0
        weights_raw = {"w_miss": 0.40, "w_swap": 0.25, "w_dev": 0.25, "w_time": 0.10}
        policy = {"missing_step": "critical", "step_deviation": "quality",
                  "order_swap": "quality", "over_time": "efficiency"}

    weights = ScoreWeights(
        w_miss=float(weights_raw.get("w_miss", 0.40)),
        w_swap=float(weights_raw.get("w_swap", 0.25)),
        w_dev=float(weights_raw.get("w_dev", 0.25)),
        w_time=float(weights_raw.get("w_time", 0.10)),
    )

    # Detect gold step boundaries
    gold_boundaries = detect_step_boundaries(gold_embs, min_gap=2, z_threshold=1.0)
    print(f"[info] gold boundaries: {gold_boundaries}", file=sys.stderr)

    # Synthesize NG trainee
    rng = np.random.default_rng(args.seed)
    n_clips = max(1, min(args.n_clips, gold_len - 1))
    n_correct = max(0, min(args.n_correct, n_clips))
    trainee_embs = make_ng_embeddings(
        gold_embs,
        rng,
        normalize=l2_normalize,
        n_correct=n_correct,
        n_total=n_clips,
    )
    trainee_len = len(trainee_embs)
    print(f"[info] NG trainee clips={trainee_len}  n_correct={n_correct}", file=sys.stderr)

    # Detect trainee boundaries
    trainee_boundaries = detect_step_boundaries(trainee_embs, min_gap=2, z_threshold=1.0)

    # DTW align
    alignment = dtw_align(gold_embs, trainee_embs)
    print(f"[info] DTW normalized_cost={alignment.normalized_cost:.4f}", file=sys.stderr)

    # Score
    raw = score_alignment(
        alignment=alignment,
        gold_len=gold_len,
        trainee_len=trainee_len,
        gold_boundaries=gold_boundaries,
        trainee_boundaries=trainee_boundaries,
        weights=weights,
        deviation_threshold=0.25,
    )

    # Apply severity policy
    deviations_with_sev = apply_deviation_policy(raw["deviations"], policy)
    raw["deviations"] = deviations_with_sev

    severity_counts = count_severity(deviations_with_sev)
    decision, decision_reason = decide(raw["score"], severity_counts, pass_score, retrain_score)

    raw["gold_video_id"] = args.gold_id
    raw["trainee_video_id"] = -1   # synthetic
    raw["task_id"] = args.task_id
    raw["summary"] = {
        "decision": decision,
        "decision_reason": decision_reason,
        "severity_counts": severity_counts,
        "pass_score": pass_score,
        "retrain_score": retrain_score,
    }
    raw["_meta"] = {
        "synthesized": True,
        "gold_clips": gold_len,
        "trainee_clips": trainee_len,
        "n_correct": n_correct,
        "db": str(db_path),
        "gold_boundaries": gold_boundaries,
        "trainee_boundaries": trainee_boundaries,
    }

    out = json.dumps(raw, ensure_ascii=False, indent=2)
    print(out)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out, encoding="utf-8")
        print(f"[info] wrote: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
