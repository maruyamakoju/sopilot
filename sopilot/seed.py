"""Generate demo seed data for SOPilot.

Usage:
    python -m sopilot.seed          # populate data/sopilot.db with demo data
    python -m sopilot.seed --reset  # wipe existing data first

Creates Gold/Trainee videos with synthetic embeddings and runs the real
DTW scoring pipeline so dashboards display fully populated results.
"""

from __future__ import annotations

import argparse
import random
from datetime import UTC, datetime, timedelta

import numpy as np

from sopilot.config import Settings
from sopilot.constants import DEFAULT_DEVIATION_POLICY, DEFAULT_WEIGHTS
from sopilot.core.dtw import dtw_align
from sopilot.core.scoring import ScoreWeights, score_alignment
from sopilot.core.segmentation import detect_step_boundaries
from sopilot.database import Database

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TASK_ID = "pilot_task"
TASK_NAME = "PoC Primary Task"
EMBED_DIM = 512
CLIP_SEC = 4.0
SAMPLE_FPS = 4

# Gold video definitions: (filename, num_clips, site_id)
GOLD_VIDEOS = [
    ("手洗い手順_Gold_A.mp4", 20, "site_tokyo"),
    ("手洗い手順_Gold_B.mp4", 18, "site_tokyo"),
    ("器具消毒手順_Gold.mp4", 25, "site_osaka"),
    ("検体採取手順_Gold.mp4", 22, "site_osaka"),
    ("薬品管理手順_Gold.mp4", 16, "site_nagoya"),
]

# Trainee video definitions:
# (filename, gold_index, deviated_frac, skip_count, site_id, operator)
#   deviated_frac = fraction of steps with high cosine distance (quality severity)
#   skip_count    = number of steps completely omitted (critical severity → fail)
TRAINEE_VIDEOS = [
    # Excellent trainees (96-100): no deviations
    ("田中_手洗い_01.mp4",   0, 0.00, 0, "site_tokyo",  "op_tanaka"),
    ("佐藤_手洗い_02.mp4",   0, 0.00, 0, "site_tokyo",  "op_sato"),
    # Good trainees (90-95): 1 deviated step, no skips
    ("鈴木_手洗い_03.mp4",   1, 0.25, 0, "site_tokyo",  "op_suzuki"),
    ("高橋_器具消毒_01.mp4", 2, 0.17, 0, "site_osaka",  "op_takahashi"),
    ("伊藤_検体採取_01.mp4", 3, 0.20, 0, "site_osaka",  "op_ito"),
    # Needs review (82-89): 2 deviated steps, no skips
    ("渡辺_手洗い_04.mp4",   0, 0.40, 0, "site_nagoya", "op_watanabe"),
    ("山本_器具消毒_02.mp4", 2, 0.33, 0, "site_osaka",  "op_yamamoto"),
    # Retrain (70-80): many deviated steps, no critical
    ("中村_薬品管理_01.mp4", 4, 0.75, 0, "site_nagoya", "op_nakamura"),
    ("小林_検体採取_02.mp4", 3, 0.60, 0, "site_tokyo",  "op_kobayashi"),
    # Fail: 1-2 skipped steps (critical) + some deviations
    ("加藤_手洗い_05.mp4",   0, 0.20, 2, "site_osaka",  "op_kato"),
    ("吉田_器具消毒_03.mp4", 2, 0.17, 3, "site_nagoya", "op_yoshida"),
    # Extra variety
    ("山田_手洗い_06.mp4",   1, 0.00, 0, "site_tokyo",  "op_yamada"),
    ("松本_検体採取_03.mp4", 3, 0.20, 0, "site_osaka",  "op_matsumoto"),
    ("井上_薬品管理_02.mp4", 4, 0.50, 0, "site_nagoya", "op_inoue"),
    ("木村_手洗い_07.mp4",   0, 0.30, 0, "site_tokyo",  "op_kimura"),
]


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------


def _make_gold_embeddings(num_clips: int, rng: np.random.Generator) -> np.ndarray:
    """Create a coherent sequence of normalized embeddings for a gold video.

    Generates step-like structure: clips within a step are similar,
    clips across steps are more different.
    """
    # Create ~5 distinct step prototypes
    num_steps = max(3, num_clips // 4)
    prototypes = rng.standard_normal((num_steps, EMBED_DIM)).astype(np.float32)
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)

    embeddings = []
    clips_per_step = num_clips / num_steps
    for i in range(num_clips):
        step_idx = min(int(i / clips_per_step), num_steps - 1)
        proto = prototypes[step_idx]
        # Small intra-step variation
        noise = rng.standard_normal(EMBED_DIM).astype(np.float32) * 0.03
        vec = proto + noise
        vec /= np.linalg.norm(vec)
        embeddings.append(vec)

    return np.array(embeddings, dtype=np.float32)


def _perturb_embedding(
    base: np.ndarray,
    target_cos_dist: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a vector with approximately the given cosine distance from base."""
    alpha = max(0.0, min(1.0, 1.0 - target_cos_dist))
    rand = rng.standard_normal(len(base)).astype(np.float32)
    rand -= np.dot(rand, base) * base
    norm = np.linalg.norm(rand)
    if norm > 1e-8:
        rand /= norm
    vec = alpha * base + np.sqrt(max(0, 1 - alpha * alpha)) * rand
    vec /= np.linalg.norm(vec)
    return np.asarray(vec, dtype=np.float32)


def _make_trainee_embeddings(
    gold_embeddings: np.ndarray,
    deviated_frac: float,
    skip_count: int,
    gold_boundaries: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Create trainee embeddings with step-level control over quality.

    deviated_frac: fraction of steps with high cosine distance (step_deviation → quality)
    skip_count: number of steps completely omitted (missing_step → critical → fail)
    """
    num_gold = len(gold_embeddings)

    # Build step spans from boundaries
    points = [0] + sorted([b for b in gold_boundaries if 0 < b < num_gold]) + [num_gold]
    step_spans = [(points[i], points[i + 1]) for i in range(len(points) - 1) if points[i + 1] > points[i]]
    num_steps = len(step_spans)

    # Decide which steps are deviated (quality severity)
    num_deviated = max(0, int(round(num_steps * deviated_frac)))
    all_indices = list(range(num_steps))
    rng.shuffle(all_indices)

    skip_indices = set(all_indices[:min(skip_count, num_steps)])
    remaining = [i for i in all_indices if i not in skip_indices]
    deviated_indices = set(remaining[:min(num_deviated, len(remaining))])

    # Slight over-time for imperfect trainees
    over_time = 0.0 if (deviated_frac == 0 and skip_count == 0) else rng.uniform(0.0, 0.12)

    trainee = []
    for step_idx, (g_start, g_end) in enumerate(step_spans):
        step_len = g_end - g_start
        t_len = max(1, int(step_len * (1.0 + rng.uniform(-0.05, over_time))))

        if step_idx in skip_indices:
            # Completely skip this step → triggers missing_step (critical)
            continue

        if step_idx in deviated_indices:
            # High cosine distance → triggers step_deviation (quality)
            for j in range(t_len):
                frac = j / max(t_len - 1, 1)
                gold_idx = min(g_start + int(frac * (step_len - 1)), g_end - 1)
                cos_dist = rng.uniform(0.35, 0.55)
                vec = _perturb_embedding(gold_embeddings[gold_idx], cos_dist, rng)
                trainee.append(vec)
        else:
            # Good step: low cosine distance from gold
            for j in range(t_len):
                frac = j / max(t_len - 1, 1)
                gold_idx = min(g_start + int(frac * (step_len - 1)), g_end - 1)
                cos_dist = rng.uniform(0.02, 0.12)
                vec = _perturb_embedding(gold_embeddings[gold_idx], cos_dist, rng)
                trainee.append(vec)

    # Ensure minimum clip count
    if len(trainee) < 3:
        for _ in range(3 - len(trainee)):
            vec = rng.standard_normal(EMBED_DIM).astype(np.float32)
            vec /= np.linalg.norm(vec)
            trainee.append(vec)

    return np.array(trainee, dtype=np.float32)


# ---------------------------------------------------------------------------
# Database insertion helpers
# ---------------------------------------------------------------------------

def _utc_iso(dt: datetime | None = None) -> str:
    return (dt or datetime.now(UTC)).isoformat()


def _insert_video(
    db: Database,
    *,
    filename: str,
    task_id: str,
    site_id: str,
    is_gold: bool,
    embeddings: np.ndarray,
    operator_id_hash: str | None = None,
    recorded_at: str | None = None,
) -> int:
    """Insert a video record with clips and embeddings."""
    num_clips = len(embeddings)
    video_id = db.insert_video(
        task_id=task_id,
        site_id=site_id,
        camera_id="cam_01",
        operator_id_hash=operator_id_hash,
        recorded_at=recorded_at,
        is_gold=is_gold,
        original_filename=filename,
    )

    boundaries = detect_step_boundaries(
        embeddings,
        min_gap=2,
        z_threshold=1.0,
    )

    clips = []
    for i in range(num_clips):
        clips.append({
            "clip_index": i,
            "start_sec": round(i * CLIP_SEC, 3),
            "end_sec": round((i + 1) * CLIP_SEC, 3),
            "embedding": embeddings[i].tolist(),
            "quality_flag": None,
        })

    db.finalize_video(
        video_id=video_id,
        file_path=f"demo/{video_id:08d}.mp4",
        step_boundaries=boundaries,
        clips=clips,
        embedding_model="demo-seed-v1",
    )

    return video_id


def _run_score(
    db: Database,
    gold_video_id: int,
    trainee_video_id: int,
    gold_embeddings: np.ndarray,
    trainee_embeddings: np.ndarray,
    gold_boundaries: list[int],
    trainee_boundaries: list[int],
    weights: ScoreWeights,
) -> tuple[int, dict]:
    """Run DTW scoring and insert the result."""
    alignment = dtw_align(gold_embeddings, trainee_embeddings)
    result = score_alignment(
        alignment=alignment,
        gold_len=len(gold_embeddings),
        trainee_len=len(trainee_embeddings),
        gold_boundaries=gold_boundaries,
        trainee_boundaries=trainee_boundaries,
        weights=weights,
        deviation_threshold=0.25,
    )
    result["gold_video_id"] = gold_video_id
    result["trainee_video_id"] = trainee_video_id
    result["task_id"] = TASK_ID

    # Attach timecodes to deviations
    gold_clips = [{"start_sec": i * CLIP_SEC, "end_sec": (i + 1) * CLIP_SEC}
                  for i in range(len(gold_embeddings))]
    trainee_clips = [{"start_sec": i * CLIP_SEC, "end_sec": (i + 1) * CLIP_SEC}
                     for i in range(len(trainee_embeddings))]

    for dev in result.get("deviations", []):
        gr = dev.get("gold_clip_range")
        tr = dev.get("trainee_clip_range")
        if isinstance(gr, list) and len(gr) == 2:
            s = max(0, min(gr[0], len(gold_clips) - 1))
            e = max(0, min(gr[1], len(gold_clips) - 1))
            dev["gold_timecode"] = [
                round(gold_clips[s]["start_sec"], 3),
                round(gold_clips[e]["end_sec"], 3),
            ]
        if isinstance(tr, list) and len(tr) == 2:
            s = max(0, min(tr[0], len(trainee_clips) - 1))
            e = max(0, min(tr[1], len(trainee_clips) - 1))
            dev["trainee_timecode"] = [
                round(trainee_clips[s]["start_sec"], 3),
                round(trainee_clips[e]["end_sec"], 3),
            ]

    # Apply simple task policy
    policy = dict(DEFAULT_DEVIATION_POLICY)
    for dev in result.get("deviations", []):
        dev_type = str(dev.get("type", "unknown"))
        dev["severity"] = policy.get(dev_type, "quality")

    # Build summary
    counts = {"critical": 0, "quality": 0, "efficiency": 0}
    for dev in result.get("deviations", []):
        sev = str(dev.get("severity", "quality"))
        counts[sev] = counts.get(sev, 0) + 1

    score = float(result.get("score", 0.0))
    pass_score = 90.0
    retrain_score = 80.0
    if counts.get("critical", 0) > 0:
        decision = "fail"
        reason = "critical deviation detected"
    elif score >= pass_score:
        decision = "pass"
        reason = f"score >= pass_score ({pass_score:.1f})"
    elif score < retrain_score:
        decision = "retrain"
        reason = f"score < retrain_score ({retrain_score:.1f})"
    else:
        decision = "needs_review"
        reason = "between retrain_score and pass_score"

    result["summary"] = {
        "decision": decision,
        "decision_reason": reason,
        "severity_counts": counts,
        "pass_score": pass_score,
        "retrain_score": retrain_score,
    }

    # Add metrics that the UI expects
    result["metrics"]["gold_length"] = len(gold_embeddings)
    result["metrics"]["trainee_length"] = len(trainee_embeddings)

    # Create score job directly in DB
    job_id = db.create_score_job(
        gold_video_id=gold_video_id,
        trainee_video_id=trainee_video_id,
        weights={"w_miss": 0.4, "w_swap": 0.25, "w_dev": 0.25, "w_time": 0.1},
    )

    # Claim and complete
    db.claim_score_job(job_id)
    db.complete_score_job(job_id, result)

    return job_id, result


# ---------------------------------------------------------------------------
# Main seeding logic
# ---------------------------------------------------------------------------

def seed(reset: bool = False) -> None:
    """Populate the database with demo data."""
    settings = Settings.from_env()
    db_path = settings.database_path
    print(f"Database: {db_path}")

    if reset and db_path.exists():
        print("Resetting database...")
        db_path.unlink()

    db = Database(db_path)

    # Check if data already exists
    with db.connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
        if count > 0 and not reset:
            print(f"Database already has {count} videos. Use --reset to wipe.")
            return

    rng = np.random.default_rng(42)  # reproducible
    random.seed(42)

    # Ensure task profile exists
    db.upsert_task_profile(
        task_id=TASK_ID,
        task_name=TASK_NAME,
        pass_score=90.0,
        retrain_score=80.0,
        default_weights=dict(DEFAULT_WEIGHTS),
        deviation_policy=dict(DEFAULT_DEVIATION_POLICY),
    )
    print("Task profile created")

    # --- Create Gold videos ---
    gold_data: list[dict] = []
    base_time = datetime.now(UTC) - timedelta(days=30)

    print("\n--- Gold Videos ---")
    for i, (filename, num_clips, site_id) in enumerate(GOLD_VIDEOS):
        recorded = base_time + timedelta(days=i)
        embeddings = _make_gold_embeddings(num_clips, rng)
        boundaries = detect_step_boundaries(embeddings, min_gap=2, z_threshold=1.0)

        video_id = _insert_video(
            db,
            filename=filename,
            task_id=TASK_ID,
            site_id=site_id,
            is_gold=True,
            embeddings=embeddings,
            recorded_at=_utc_iso(recorded),
        )
        gold_data.append({
            "video_id": video_id,
            "embeddings": embeddings,
            "boundaries": boundaries,
            "num_clips": num_clips,
        })
        print(f"  Gold #{video_id}: {filename} ({num_clips} clips, {len(boundaries)} boundaries)")

    # --- Create Trainee videos and score them ---
    weights = ScoreWeights()
    job_results: list[tuple[int, dict, str]] = []

    print("\n--- Trainee Videos ---")
    for i, (filename, gold_idx, dev_frac, skip_n, site_id, operator) in enumerate(TRAINEE_VIDEOS):
        recorded = base_time + timedelta(days=5 + i)
        gold = gold_data[gold_idx]
        trainee_embeddings = _make_trainee_embeddings(
            gold["embeddings"], dev_frac, skip_n, gold["boundaries"], rng,
        )
        trainee_boundaries = detect_step_boundaries(trainee_embeddings, min_gap=2, z_threshold=1.0)

        video_id = _insert_video(
            db,
            filename=filename,
            task_id=TASK_ID,
            site_id=site_id,
            is_gold=False,
            embeddings=trainee_embeddings,
            operator_id_hash=operator,
            recorded_at=_utc_iso(recorded),
        )

        job_id, result = _run_score(
            db=db,
            gold_video_id=gold["video_id"],
            trainee_video_id=video_id,
            gold_embeddings=gold["embeddings"],
            trainee_embeddings=trainee_embeddings,
            gold_boundaries=gold["boundaries"],
            trainee_boundaries=trainee_boundaries,
            weights=weights,
        )

        score = result["score"]
        decision = result["summary"]["decision"]
        job_results.append((job_id, result, filename))
        print(f"  Trainee #{video_id}: {filename} -> Score: {score:.1f} ({decision}) [Job #{job_id}]")

    # --- Add reviews to some completed jobs ---
    print("\n--- Reviews ---")
    review_data = [
        # (job_index, verdict, note)
        (0, "pass", "手順完璧。模範的な実施。"),
        (1, "pass", "わずかな手順の遅れはあるが問題なし。"),
        (2, "pass", "合格基準を満たしている。継続的な改善を期待。"),
        (4, "pass", "検体採取手順を正確に実施。"),
        (5, "needs_review", "一部手順に不安あり。上長確認が必要。"),
        (6, "needs_review", "器具の持ち方に改善の余地。再確認推奨。"),
        (7, "retrain", "薬品の取り扱いに重大な手順漏れ。再訓練必須。"),
        (9, "fail", "複数の手順省略を確認。安全上の問題あり。即再訓練。"),
        (10, "fail", "消毒手順の大幅な逸脱。現場投入不可。"),
    ]
    for job_idx, verdict, note in review_data:
        if job_idx < len(job_results):
            job_id = job_results[job_idx][0]
            db.upsert_score_review(job_id=job_id, verdict=verdict, note=note)
            print(f"  Job #{job_id}: {verdict} - {note[:30]}...")

    # --- Summary ---
    with db.connect() as conn:
        v_count = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
        c_count = conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0]
        j_count = conn.execute("SELECT COUNT(*) FROM score_jobs").fetchone()[0]
        r_count = conn.execute("SELECT COUNT(*) FROM score_reviews").fetchone()[0]

    print(f"\n{'='*50}")
    print("Seed complete!")
    print(f"  Videos:  {v_count} ({len(GOLD_VIDEOS)} gold + {len(TRAINEE_VIDEOS)} trainee)")
    print(f"  Clips:   {c_count}")
    print(f"  Jobs:    {j_count}")
    print(f"  Reviews: {r_count}")
    print(f"{'='*50}")
    print("\nStart the server:")
    print("  python -m uvicorn sopilot.main:create_app --factory --reload")
    print("  Open http://localhost:8000")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SOPilot demo seed data generator")
    parser.add_argument("--reset", action="store_true", help="Wipe existing data before seeding")
    args = parser.parse_args()
    seed(reset=args.reset)


if __name__ == "__main__":
    main()
