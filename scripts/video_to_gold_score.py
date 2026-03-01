"""
video_to_gold_score.py
──────────────────────
Standalone script:
  1. Processes a real video file with ColorMotionEmbedder
  2. Registers it as a Gold video in the DB
  3. Synthesizes an NG trainee from the real embeddings
  4. Runs DTW + scoring
  5. Prints the full score JSON

Usage:
    python scripts/video_to_gold_score.py --video jr23_720p.mp4
    python scripts/video_to_gold_score.py --video jr23_720p.mp4 --out jr23_ng_score.json
    python scripts/video_to_gold_score.py --video jr23_720p.mp4 --sample-fps 2 --clip-seconds 2
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _ensure_repo_on_path() -> None:
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


# ── reuse helpers from synthesize_ng_score ──────────────────────────
def make_ng_embeddings(
    gold_embs: np.ndarray,
    rng: np.random.Generator,
    normalize: Callable[[np.ndarray], np.ndarray],
    n_correct: int = 1,
    n_total: int = 4,
) -> np.ndarray:
    """Synthesise NG trainee: n_correct clips roughly right, rest random/wrong."""
    dim = gold_embs.shape[1]
    clips = []
    for i in range(n_total):
        if i < n_correct:
            noise = rng.normal(0, 0.25, dim).astype(np.float32)
            vec = normalize(gold_embs[i] + noise)
        else:
            raw = rng.standard_normal(dim).astype(np.float32)
            vec = normalize(raw)
        clips.append(vec)
    return np.stack(clips, axis=0)


def apply_deviation_policy(deviations: list[dict], policy: dict[str, str]) -> list[dict]:
    result = []
    for dev in deviations:
        d = dict(dev)
        d["severity"] = policy.get(d.get("type", ""), "quality")
        result.append(d)
    return result


def count_severity(deviations: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {"critical": 0, "quality": 0, "efficiency": 0}
    for d in deviations:
        sev = d.get("severity", "quality")
        counts[sev] = counts.get(sev, 0) + 1
    return counts


def decide(score: float, severity_counts: dict[str, int],
           pass_score: float, retrain_score: float) -> tuple[str, str]:
    if severity_counts.get("critical", 0) > 0:
        return "fail", "Critical deviation detected"
    if score >= pass_score:
        return "pass", f"score >= pass_score ({pass_score})"
    if score >= retrain_score:
        return "needs_review", "between retrain_score and pass_score"
    return "retrain", f"score < retrain_score ({retrain_score})"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process a real video as Gold, then score a synthesized NG trainee.")
    parser.add_argument("--video", required=True, help="Path to the input video file")
    parser.add_argument("--db", default="data_tesda_cm/sopilot.db")
    parser.add_argument("--task-id", default="filter_change")
    parser.add_argument("--sample-fps", type=int, default=4,
                        help="Frames to sample per second (default: 4)")
    parser.add_argument("--clip-seconds", type=int, default=2,
                        help="Seconds per clip window (default: 2)")
    parser.add_argument("--frame-size", type=int, default=256)
    parser.add_argument("--n-correct", type=int, default=1,
                        help="NG trainee clips that partially match gold (default: 1)")
    parser.add_argument("--n-clips", type=int, default=4,
                        help="Total NG trainee clip count (default: 4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="Write result JSON to file")
    parser.add_argument("--no-db", action="store_true",
                        help="Skip DB registration, run fully in memory")
    args = parser.parse_args()

    _ensure_repo_on_path()
    from sopilot.core.dtw import dtw_align
    from sopilot.core.math_utils import l2_normalize
    from sopilot.core.scoring import ScoreWeights, score_alignment
    from sopilot.core.segmentation import detect_step_boundaries
    from sopilot.database import Database
    from sopilot.services.embedder import ColorMotionEmbedder
    from sopilot.services.video_processor import VideoProcessor

    # ── resolve paths ──────────────────────────────────────────────────
    video_path = Path(args.video)
    if not video_path.is_absolute():
        # try relative to cwd, then to repo root
        if (Path.cwd() / video_path).exists():
            video_path = Path.cwd() / video_path
        elif (ROOT / video_path).exists():
            video_path = ROOT / video_path
    if not video_path.exists():
        sys.exit(f"[error] Video not found: {video_path}")
    print(f"[info] video: {video_path}", file=sys.stderr)

    db_path: Path | None = None
    if not args.no_db:
        db_path_raw = ROOT / args.db if not Path(args.db).is_absolute() else Path(args.db)
        if not db_path_raw.exists():
            for fallback in [
                ROOT / "data_tesda_cm" / "sopilot.db",
                ROOT / "data_tesda"    / "sopilot.db",
                ROOT / "data"          / "sopilot.db",
            ]:
                if fallback.exists():
                    db_path_raw = fallback
                    print(f"[warn] using fallback DB: {db_path_raw}", file=sys.stderr)
                    break
            else:
                print("[warn] DB not found — running in --no-db mode", file=sys.stderr)
                db_path_raw = None
        db_path = db_path_raw

    # ── extract clips from real video ──────────────────────────────────
    print(f"[info] processing video (sample_fps={args.sample_fps}, "
          f"clip_seconds={args.clip_seconds}, frame_size={args.frame_size})", file=sys.stderr)
    embedder = ColorMotionEmbedder()
    processor = VideoProcessor(
        sample_fps=args.sample_fps,
        clip_seconds=args.clip_seconds,
        frame_size=args.frame_size,
        embedder=embedder,
    )
    clips = processor.process(str(video_path))
    gold_embs = np.array([c.embedding for c in clips], dtype=np.float32)
    gold_len = len(gold_embs)
    print(f"[info] extracted {gold_len} clips, dim={gold_embs.shape[1]}", file=sys.stderr)

    # ── register in DB (optional) ──────────────────────────────────────
    gold_video_id: int | None = None
    if db_path is not None:
        db = Database(db_path)
        now = datetime.now(UTC).isoformat()
        gold_video_id = db.insert_video(
            task_id=args.task_id,
            site_id=None,
            camera_id=None,
            operator_id_hash=None,
            recorded_at=now,
            is_gold=True,
        )
        clip_dicts = [
            {
                "clip_index": c.clip_index,
                "start_sec": c.start_sec,
                "end_sec": c.end_sec,
                "embedding": c.embedding,
                "quality_flag": c.quality_flag,
            }
            for c in clips
        ]
        # detect boundaries for storage
        boundaries = detect_step_boundaries(gold_embs, min_gap=2, z_threshold=1.0)
        db.finalize_video(
            video_id=gold_video_id,
            file_path=str(video_path),
            step_boundaries=boundaries,
            clips=clip_dicts,
            embedding_model=embedder.name,
        )
        print(f"[info] registered Gold video_id={gold_video_id} in DB", file=sys.stderr)
    else:
        boundaries = detect_step_boundaries(gold_embs, min_gap=2, z_threshold=1.0)

    # ── load task profile (thresholds / weights / policy) ──────────────
    if db_path is not None:
        db = Database(db_path)
        profile = db.get_task_profile(args.task_id)
    else:
        profile = None

    if profile:
        pass_score    = float(profile["pass_score"])
        retrain_score = float(profile["retrain_score"])
        weights_raw   = profile.get("default_weights") or {}
        policy        = profile.get("deviation_policy") or {}
    else:
        pass_score, retrain_score = 90.0, 80.0
        weights_raw = {"w_miss": 0.40, "w_swap": 0.25, "w_dev": 0.25, "w_time": 0.10}
        policy = {"missing_step": "critical", "step_deviation": "quality",
                  "order_swap": "quality", "over_time": "efficiency"}

    weights = ScoreWeights(
        w_miss=float(weights_raw.get("w_miss", 0.40)),
        w_swap=float(weights_raw.get("w_swap", 0.25)),
        w_dev =float(weights_raw.get("w_dev",  0.25)),
        w_time=float(weights_raw.get("w_time", 0.10)),
    )

    # ── gold boundaries ────────────────────────────────────────────────
    gold_boundaries = detect_step_boundaries(gold_embs, min_gap=2, z_threshold=1.0)
    print(f"[info] gold boundaries: {gold_boundaries}", file=sys.stderr)

    # ── synthesise NG trainee ──────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    n_clips   = max(1, min(args.n_clips,   gold_len - 1))
    n_correct = max(0, min(args.n_correct, n_clips))
    trainee_embs = make_ng_embeddings(
        gold_embs,
        rng,
        normalize=l2_normalize,
        n_correct=n_correct,
        n_total=n_clips,
    )
    trainee_len  = len(trainee_embs)
    print(f"[info] NG trainee clips={trainee_len}  n_correct={n_correct}", file=sys.stderr)

    trainee_boundaries = detect_step_boundaries(trainee_embs, min_gap=2, z_threshold=1.0)

    # ── DTW align ─────────────────────────────────────────────────────
    alignment = dtw_align(gold_embs, trainee_embs)
    print(f"[info] DTW normalized_cost={alignment.normalized_cost:.4f}", file=sys.stderr)

    # ── score ─────────────────────────────────────────────────────────
    raw = score_alignment(
        alignment=alignment,
        gold_len=gold_len,
        trainee_len=trainee_len,
        gold_boundaries=gold_boundaries,
        trainee_boundaries=trainee_boundaries,
        weights=weights,
        deviation_threshold=0.25,
    )

    deviations_with_sev = apply_deviation_policy(raw["deviations"], policy)
    raw["deviations"] = deviations_with_sev
    severity_counts = count_severity(deviations_with_sev)
    decision, decision_reason = decide(raw["score"], severity_counts, pass_score, retrain_score)

    raw["gold_video_id"]    = gold_video_id if gold_video_id is not None else "in-memory"
    raw["trainee_video_id"] = -1
    raw["task_id"]          = args.task_id
    raw["summary"] = {
        "decision":        decision,
        "decision_reason": decision_reason,
        "severity_counts": severity_counts,
        "pass_score":      pass_score,
        "retrain_score":   retrain_score,
    }
    raw["_meta"] = {
        "synthesized":         True,
        "source_video":        str(video_path),
        "gold_clips":          gold_len,
        "trainee_clips":       trainee_len,
        "n_correct":           n_correct,
        "sample_fps":          args.sample_fps,
        "clip_seconds":        args.clip_seconds,
        "embedder":            embedder.name,
        "db":                  str(db_path) if db_path else None,
        "gold_boundaries":     gold_boundaries,
        "trainee_boundaries":  trainee_boundaries,
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
