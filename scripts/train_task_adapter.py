"""
train_task_adapter.py — CLI for training a SOPAdapterHead on stored clip embeddings.

Usage
-----
python scripts/train_task_adapter.py \\
    --db data_release_baseline/sopilot.db \\
    --task-id filter_change \\
    --output-dir artifacts/adapters/ \\
    --epochs 50 \\
    --device auto

The script:
1. Connects to the SQLite DB (WAL, same pattern as sopilot.database).
2. Loads all completed score jobs for the given task.
3. Loads clip embeddings for every referenced video.
4. Uses gold video step boundaries from the sop_steps table as step labels.
   If no step definitions exist, pseudo-labels are derived from step_boundaries_json
   stored on the video row (set during video finalization).
5. Trains a SOPAdapterHead with TripletLoss.
6. Saves the adapter checkpoint to --output-dir/<task_id>_adapter.pt.
7. Prints before/after evaluation metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("train_task_adapter")


# ---------------------------------------------------------------------------
# Torch guard (graceful degradation)
# ---------------------------------------------------------------------------
try:
    import torch  # noqa: F401 — just probing availability

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Minimal DB helpers (mirrors sopilot.database patterns without importing it,
# so the script can run stand-alone against any DB path).
# ---------------------------------------------------------------------------

@contextmanager
def _connect(db_path: str) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _fetch_all(db_path: str, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        return [dict(row) for row in conn.execute(query, params).fetchall()]


def _fetch_one(db_path: str, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(query, params).fetchone()
        return dict(row) if row is not None else None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_completed_jobs(db_path: str, task_id: str) -> list[dict[str, Any]]:
    """Return all completed score_jobs whose gold video belongs to task_id."""
    rows = _fetch_all(
        db_path,
        """
        SELECT sj.id          AS job_id,
               sj.gold_video_id,
               sj.trainee_video_id,
               sj.status,
               v_gold.task_id AS task_id
        FROM   score_jobs sj
        JOIN   videos v_gold ON v_gold.id = sj.gold_video_id
        WHERE  sj.status = 'completed'
          AND  v_gold.task_id = ?
        """,
        (task_id,),
    )
    logger.info("Found %d completed score jobs for task '%s'", len(rows), task_id)
    return rows


def load_clip_embeddings(db_path: str, video_id: int) -> np.ndarray | None:
    """
    Load all clip embeddings for video_id from the clips table.

    Returns (n_clips, D) float32 array ordered by clip_index, or None if the
    video has no clips.
    """
    rows = _fetch_all(
        db_path,
        "SELECT clip_index, embedding_json FROM clips WHERE video_id = ? ORDER BY clip_index",
        (video_id,),
    )
    if not rows:
        return None
    vectors = [np.asarray(json.loads(row["embedding_json"]), dtype=np.float32) for row in rows]
    return np.stack(vectors)


def load_video_meta(db_path: str, video_id: int) -> dict[str, Any] | None:
    """Return the videos row for video_id (with step_boundaries_json parsed)."""
    row = _fetch_one(db_path, "SELECT * FROM videos WHERE id = ?", (video_id,))
    if row is None:
        return None
    raw = row.get("step_boundaries_json")
    row["step_boundaries"] = json.loads(raw) if raw else []
    return row


def load_sop_steps(db_path: str, task_id: str) -> list[dict[str, Any]]:
    """Return all sop_steps rows for task_id, ordered by step_index."""
    return _fetch_all(
        db_path,
        "SELECT * FROM sop_steps WHERE task_id = ? ORDER BY step_index",
        (task_id,),
    )


# ---------------------------------------------------------------------------
# Label building
# ---------------------------------------------------------------------------

def build_labels_from_step_boundaries(
    step_boundaries: list[int],
    n_clips: int,
) -> list[int | None]:
    """
    Map each clip to a step index using step_boundaries (clip-index boundary points).

    Boundary semantics: boundary B means clips [0..B-1] belong to step i,
    clips [B..next_B-1] belong to step i+1.  This matches the convention
    used in sopilot.core.scoring._build_step_spans.
    """
    boundaries = sorted(b for b in step_boundaries if 0 < b < n_clips)
    clip_labels: list[int | None] = []
    for clip_idx in range(n_clips):
        step = 0
        for b in boundaries:
            if clip_idx >= b:
                step += 1
            else:
                break
        clip_labels.append(step)
    return clip_labels


def build_labels_from_sop_steps(
    sop_steps: list[dict[str, Any]],
    step_boundaries: list[int],
    n_clips: int,
) -> list[int | None]:
    """
    Use sop_steps definitions + step_boundaries to label each clip.

    sop_steps provides the ground-truth count and ordering of steps.
    step_boundaries is used to split clips across those steps.
    If step count from sop_steps differs from boundaries+1, we fall back
    to plain boundary-based labelling.
    """
    n_steps_defined = len(sop_steps)
    n_steps_from_boundaries = len(step_boundaries) + 1
    if n_steps_defined != n_steps_from_boundaries:
        logger.debug(
            "sop_steps count (%d) != boundary-derived count (%d); "
            "using boundary-based labels",
            n_steps_defined,
            n_steps_from_boundaries,
        )
    return build_labels_from_step_boundaries(step_boundaries, n_clips)


# ---------------------------------------------------------------------------
# Train / evaluate helpers
# ---------------------------------------------------------------------------

def _infer_embedding_dim(embeddings_dict: dict[str, np.ndarray]) -> int:
    for emb in embeddings_dict.values():
        if len(emb) > 0:
            return int(emb.shape[1])
    raise ValueError("No embeddings found to infer dimensionality from")


def _train_val_split(
    video_ids: list[str],
    val_fraction: float = 0.2,
    rng_seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Shuffle and split video IDs into train/val sets."""
    rng = np.random.default_rng(rng_seed)
    shuffled = list(video_ids)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


def _print_metrics(metrics: dict[str, float]) -> None:
    before = metrics.get("before_rank", float("nan"))
    after  = metrics.get("after_rank",  float("nan"))
    improv = metrics.get("improvement_pct", float("nan"))
    print(f"  Before adapter — mean step rank : {before:.3f}")
    print(f"  After  adapter — mean step rank : {after:.3f}")
    if not (before != before or improv != improv):  # NaN check
        sign = "+" if improv >= 0 else ""
        print(f"  Rank improvement               : {sign}{improv:.1f}%")
    else:
        print("  Rank improvement               : n/a (insufficient labelled clips)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a SOPAdapterHead on stored clip embeddings for a given task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", required=True, help="Path to sopilot.db SQLite file")
    parser.add_argument("--task-id", required=True, help="task_id to train on")
    parser.add_argument(
        "--output-dir",
        default="artifacts/adapters",
        help="Directory where the .pt checkpoint is saved",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument(
        "--hidden-dim", type=int, default=256, help="MLP hidden dimension"
    )
    parser.add_argument(
        "--output-dim", type=int, default=128, help="Adapter output dimension"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="Dropout probability"
    )
    parser.add_argument(
        "--margin", type=float, default=0.3, help="Triplet loss margin"
    )
    parser.add_argument(
        "--mining",
        choices=["hard", "semi-hard"],
        default="hard",
        help="Triplet mining strategy",
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early-stopping patience (epochs)"
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of videos held out for validation",
    )
    parser.add_argument(
        "--triplets-per-epoch",
        type=int,
        default=2000,
        help="Triplets sampled per epoch from the training set",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: 'auto', 'cpu', 'cuda', 'mps', …",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--gold-only",
        action="store_true",
        help="Use only gold videos for training (recommended when labelled data is scarce)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip before/after evaluation (faster, useful for large datasets)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ------------------------------------------------------------------ #
    # 0. Torch availability check
    # ------------------------------------------------------------------ #
    if not _TORCH_AVAILABLE:
        print(
            "[ERROR] torch is not installed.  Install with:\n"
            "  pip install 'sopilot[vjepa2]'\n"
            "or:\n"
            "  pip install torch>=2.0",
            file=sys.stderr,
        )
        return 1

    # Import fine_tuning only after confirming torch is present so that the
    # ImportWarning from the module level is surfaced cleanly.
    try:
        from sopilot.core.fine_tuning import (
            AdapterTrainer,
            SOPAdapterHead,
            evaluate_adapter_quality,
        )
        from sopilot.core.dtw import dtw_align
    except ImportError as exc:
        print(f"[ERROR] Failed to import sopilot.core.fine_tuning: {exc}", file=sys.stderr)
        return 1

    db_path = str(Path(args.db).resolve())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load completed jobs
    # ------------------------------------------------------------------ #
    jobs = load_completed_jobs(db_path, args.task_id)
    if not jobs:
        print(
            f"[ERROR] No completed score jobs found for task '{args.task_id}' in {db_path}.\n"
            "        Run some scoring jobs first (POST /score), or check the task-id.",
            file=sys.stderr,
        )
        return 1

    # Collect unique video IDs referenced by jobs.
    video_ids_set: set[int] = set()
    for job in jobs:
        video_ids_set.add(int(job["gold_video_id"]))
        if not args.gold_only:
            video_ids_set.add(int(job["trainee_video_id"]))

    logger.info(
        "Collecting embeddings for %d unique videos (gold_only=%s)",
        len(video_ids_set),
        args.gold_only,
    )

    # ------------------------------------------------------------------ #
    # 2. Load embeddings + metadata
    # ------------------------------------------------------------------ #
    sop_steps = load_sop_steps(db_path, args.task_id)
    has_step_defs = len(sop_steps) > 0
    if has_step_defs:
        logger.info(
            "Found %d SOP step definitions for task '%s'", len(sop_steps), args.task_id
        )
    else:
        logger.info(
            "No SOP step definitions found for task '%s'. "
            "Will use step_boundaries_json from video rows as pseudo-labels.",
            args.task_id,
        )

    embeddings_dict: dict[str, np.ndarray] = {}
    labels_dict: dict[str, list[int | None]] = {}
    missing_embeddings: list[int] = []

    for vid_id in sorted(video_ids_set):
        embs = load_clip_embeddings(db_path, vid_id)
        if embs is None or len(embs) == 0:
            missing_embeddings.append(vid_id)
            continue

        meta = load_video_meta(db_path, vid_id)
        if meta is None:
            missing_embeddings.append(vid_id)
            continue

        step_boundaries: list[int] = meta.get("step_boundaries") or []

        if has_step_defs:
            clip_labels = build_labels_from_sop_steps(sop_steps, step_boundaries, len(embs))
        else:
            clip_labels = build_labels_from_step_boundaries(step_boundaries, len(embs))

        # If all clips map to step 0 (no boundaries), mark as unlabelled —
        # the model can't learn from single-step videos.
        unique_labels = set(l for l in clip_labels if l is not None)
        if len(unique_labels) < 2:
            logger.debug(
                "Video %d has only %d distinct step(s); skipping (need ≥2).",
                vid_id,
                len(unique_labels),
            )
            continue

        key = str(vid_id)
        embeddings_dict[key] = embs
        labels_dict[key] = clip_labels

    if missing_embeddings:
        logger.warning(
            "%d video(s) had no clip embeddings and were skipped: %s\n"
            "  Run video embedding extraction first (POST /videos with a video file).",
            len(missing_embeddings),
            missing_embeddings[:10],
        )

    if not embeddings_dict:
        print(
            "[ERROR] No usable embeddings found.  All videos either have no clips "
            "or only a single step.\n"
            "        Upload and process videos via POST /videos before training.",
            file=sys.stderr,
        )
        return 1

    logger.info(
        "Using %d videos with ≥2 step labels for training.",
        len(embeddings_dict),
    )

    # ------------------------------------------------------------------ #
    # 3. Train/val split
    # ------------------------------------------------------------------ #
    all_vids = list(embeddings_dict.keys())
    if len(all_vids) >= 4:
        train_vids, val_vids = _train_val_split(
            all_vids, val_fraction=args.val_fraction, rng_seed=args.seed
        )
        logger.info(
            "Train/val split: %d train videos, %d val videos",
            len(train_vids),
            len(val_vids),
        )
    else:
        # Too few videos for a meaningful split; train on everything.
        train_vids = all_vids
        val_vids = []
        logger.info(
            "Only %d video(s) available — training on all (no validation split).",
            len(all_vids),
        )

    train_embs = {v: embeddings_dict[v] for v in train_vids}
    train_lbls = {v: labels_dict[v] for v in train_vids}
    val_embs   = {v: embeddings_dict[v] for v in val_vids} if val_vids else None
    val_lbls   = {v: labels_dict[v] for v in val_vids}    if val_vids else None

    # ------------------------------------------------------------------ #
    # 4. Build adapter + trainer
    # ------------------------------------------------------------------ #
    input_dim = _infer_embedding_dim(embeddings_dict)
    logger.info("Embedding dimensionality detected: %d", input_dim)

    adapter = SOPAdapterHead(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
    )

    trainer = AdapterTrainer(
        adapter=adapter,
        device=args.device,
        loss_margin=args.margin,
        loss_mining=args.mining,
        triplets_per_epoch=args.triplets_per_epoch,
        rng_seed=args.seed,
    )

    param_count = sum(p.numel() for p in adapter.parameters())
    logger.info(
        "Adapter architecture: %d → %d → %d → %d  (%.1fK parameters)",
        input_dim, args.hidden_dim, args.hidden_dim // 2, args.output_dim,
        param_count / 1000,
    )

    # ------------------------------------------------------------------ #
    # 5. Before-adapter evaluation
    # ------------------------------------------------------------------ #
    before_metrics: dict[str, float] | None = None
    if not args.no_eval:
        logger.info("Computing pre-training evaluation metrics …")
        try:
            before_metrics = evaluate_adapter_quality(
                trainer, embeddings_dict, labels_dict, dtw_align
            )
        except Exception as exc:
            logger.warning("Pre-training evaluation failed: %s", exc)

    # ------------------------------------------------------------------ #
    # 6. Train
    # ------------------------------------------------------------------ #
    print(f"\nTraining SOPAdapterHead for task '{args.task_id}' …")
    print(
        f"  Videos  : {len(train_vids)} train"
        + (f" / {len(val_vids)} val" if val_vids else " (no val split)")
    )
    print(
        f"  Adapter : {input_dim}→{args.hidden_dim}→{args.hidden_dim//2}→{args.output_dim}"
        f"  ({param_count/1000:.1f}K params)"
    )
    print(f"  Device  : {args.device}  |  max epochs: {args.epochs}"
          f"  |  patience: {args.patience}")
    print()

    result = trainer.fit(
        train_embeddings=train_embs,
        train_labels=train_lbls,
        val_embeddings=val_embs,
        val_labels=val_lbls,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )

    # ------------------------------------------------------------------ #
    # 7. Save checkpoint
    # ------------------------------------------------------------------ #
    checkpoint_path = output_dir / f"{args.task_id}_adapter.pt"
    trainer.save(checkpoint_path)
    result.model_path = str(checkpoint_path)

    print(f"\nAdapter saved to: {checkpoint_path}")
    print(
        f"Training summary: {result.total_epochs_run} epochs run"
        f"  |  best epoch: {result.best_epoch + 1}"
        f"  |  final train loss: {result.final_train_loss:.4f}"
    )
    if result.val_losses:
        best_val = min(result.val_losses)
        print(f"  best val loss: {best_val:.4f}")

    # ------------------------------------------------------------------ #
    # 8. After-adapter evaluation + comparison
    # ------------------------------------------------------------------ #
    if not args.no_eval:
        logger.info("Computing post-training evaluation metrics …")
        try:
            after_metrics = evaluate_adapter_quality(
                trainer, embeddings_dict, labels_dict, dtw_align
            )
            print("\nStep-alignment rank (lower = better):")
            _print_metrics(after_metrics)

            if before_metrics is not None:
                before_rank  = before_metrics.get("before_rank", float("nan"))
                after_rank   = after_metrics.get("after_rank",  float("nan"))
                # before_metrics["before_rank"] and after_metrics["after_rank"]
                # use the same raw embeddings for the "before" comparison so the
                # two runs are directly comparable.
                if not (after_rank != after_rank):  # NaN check
                    delta = before_rank - after_rank
                    sign  = "+" if delta >= 0 else ""
                    print(
                        f"\n  Rank delta (pre→post training): {sign}{delta:.3f}"
                        f" ({sign}{delta / max(before_rank, 1e-9) * 100:.1f}%)"
                    )
        except Exception as exc:
            logger.warning("Post-training evaluation failed: %s", exc)

    # ------------------------------------------------------------------ #
    # 9. Print usage hint
    # ------------------------------------------------------------------ #
    print(
        f"\nTo apply this adapter:\n"
        f"  from sopilot.core.fine_tuning import AdapterTrainer\n"
        f"  trainer = AdapterTrainer.load('{checkpoint_path}')\n"
        f"  adapted_embs = trainer.transform(raw_embs)  # (N, {args.output_dim}) float32\n"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
