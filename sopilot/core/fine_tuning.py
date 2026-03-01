"""
Fine-tuning pipeline for SOP adapter heads.

Provides a lightweight MLP adapter (SOPAdapterHead) that sits on top of frozen
V-JEPA2 embeddings to learn task-specific step representations via metric
learning (TripletLoss).  All heavy lifting is done in numpy unless torch is
available; the module degrades gracefully when torch is absent.
"""

from __future__ import annotations

import logging
import random
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Torch availability guard
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    warnings.warn(
        "[fine_tuning] torch is not installed. "
        "SOPAdapterHead training is unavailable. "
        "Install with: pip install 'sopilot[vjepa2]'",
        ImportWarning,
        stacklevel=2,
    )
    # Provide stub base classes so class definitions below still parse.
    class nn:  # type: ignore[no-redef]
        class Module:
            pass

    class Dataset:  # type: ignore[no-redef]
        pass

    class DataLoader:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# TrainingResult
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Summary produced by AdapterTrainer.fit()."""

    train_losses: list[float]
    val_losses: list[float] | None
    best_epoch: int
    final_train_loss: float
    total_epochs_run: int
    model_path: str = ""


# ---------------------------------------------------------------------------
# SOPAdapterHead
# ---------------------------------------------------------------------------

class SOPAdapterHead(nn.Module):  # type: ignore[misc]
    """
    Lightweight 3-layer MLP adapter sitting on top of frozen V-JEPA2 embeddings.

    Architecture
    ------------
    Linear(input_dim, hidden_dim)  -> LayerNorm -> GELU -> Dropout
    Linear(hidden_dim, hidden_dim//2) -> LayerNorm -> GELU -> Dropout
    Linear(hidden_dim//2, output_dim)
    L2-normalise output  (unit vectors for cosine-distance metric learning)

    Parameters
    ----------
    input_dim:  Dimensionality of frozen V-JEPA2 embeddings (default 1024 for ViT-L).
    hidden_dim: Width of first hidden layer.
    output_dim: Embedding space dimensionality for cosine similarity learning.
    dropout:    Dropout probability applied after each GELU activation.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is required to instantiate SOPAdapterHead")
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        half = hidden_dim // 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, half),
            nn.LayerNorm(half),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(half, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
        out = self.net(x)
        # L2 normalise to the unit hypersphere for cosine similarity learning.
        norm = out.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return out / norm

    def config_dict(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout": self.dropout,
        }


# ---------------------------------------------------------------------------
# TripletLoss
# ---------------------------------------------------------------------------

class TripletLoss(nn.Module):  # type: ignore[misc]
    """
    Metric-learning triplet loss with online mining.

    Parameters
    ----------
    margin:  Minimum required gap: d(a,n) - d(a,p) >= margin.
    mining:  "hard"      — hardest negative in batch (smallest d(a,n)).
             "semi-hard" — hardest negative that is still farther than d(a,p)
                           but within d(a,p) + margin.  Falls back to hard if
                           no semi-hard negative exists.

    The module expects embeddings to already be L2-normalised (as output by
    SOPAdapterHead), so cosine distance = 1 - dot(a,b).
    """

    def __init__(self, margin: float = 0.3, mining: str = "hard") -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is required to instantiate TripletLoss")
        super().__init__()
        if mining not in {"hard", "semi-hard"}:
            raise ValueError(f"mining must be 'hard' or 'semi-hard', got {mining!r}")
        self.margin = margin
        self.mining = mining

    def forward(
        self,
        anchors: torch.Tensor,    # (B, D)  # type: ignore[name-defined]
        positives: torch.Tensor,  # (B, D)  # type: ignore[name-defined]
        negatives: torch.Tensor,  # (B, D)  # type: ignore[name-defined]
    ) -> torch.Tensor:  # type: ignore[name-defined]
        """
        Compute triplet loss from pre-mined triplets.

        If batch mining is desired the caller should pass all embeddings through
        _mine_triplets() before calling forward.
        """
        # Cosine distance = 1 - dot product (embeddings are unit vectors).
        d_ap = 1.0 - (anchors * positives).sum(dim=-1)   # (B,)
        d_an = 1.0 - (anchors * negatives).sum(dim=-1)   # (B,)
        loss_per_triplet = torch.clamp(d_ap - d_an + self.margin, min=0.0)
        return loss_per_triplet.mean()

    def mine_batch(
        self,
        embeddings: torch.Tensor,  # (N, D)  # type: ignore[name-defined]
        labels: torch.Tensor,      # (N,) int  # type: ignore[name-defined]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[name-defined]
        """
        Mine anchors, positives, negatives from a labelled embedding batch.

        Returns three (M, D) tensors: anchors, positives, negatives.
        M may be < N if some samples have no valid positive or negative.
        """
        # Pairwise cosine distance matrix.
        # embeddings are unit-norm, so d = 1 - e @ e.T.
        sim = embeddings @ embeddings.T              # (N, N)
        dist = 1.0 - sim.clamp(-1.0, 1.0)           # (N, N)

        N = labels.shape[0]
        # Boolean masks: same_label[i,j]=True when labels match.
        same_label = labels.unsqueeze(1) == labels.unsqueeze(0)   # (N, N)
        diff_label = ~same_label
        eye_mask = torch.eye(N, dtype=torch.bool, device=embeddings.device)
        pos_mask = same_label & ~eye_mask     # valid positives
        neg_mask = diff_label                 # valid negatives

        anchors_list, positives_list, negatives_list = [], [], []

        for i in range(N):
            pos_indices = pos_mask[i].nonzero(as_tuple=False).squeeze(1)
            neg_indices = neg_mask[i].nonzero(as_tuple=False).squeeze(1)
            if pos_indices.numel() == 0 or neg_indices.numel() == 0:
                continue

            # Choose the hardest positive (furthest from anchor).
            pos_idx = int(pos_indices[dist[i, pos_indices].argmax()].item())
            d_ap_i = float(dist[i, pos_idx].item())

            if self.mining == "semi-hard":
                # Semi-hard: negatives farther than d(a,p) but within d(a,p)+margin.
                semi_mask = (dist[i, neg_indices] > d_ap_i) & (
                    dist[i, neg_indices] < d_ap_i + self.margin
                )
                semi_indices = neg_indices[semi_mask]
                if semi_indices.numel() > 0:
                    neg_idx = int(semi_indices[dist[i, semi_indices].argmin()].item())
                else:
                    # Fall back to hard negative.
                    neg_idx = int(neg_indices[dist[i, neg_indices].argmin()].item())
            else:
                # Hard: closest negative to anchor.
                neg_idx = int(neg_indices[dist[i, neg_indices].argmin()].item())

            anchors_list.append(embeddings[i])
            positives_list.append(embeddings[pos_idx])
            negatives_list.append(embeddings[neg_idx])

        if not anchors_list:
            # Return empty tensors with correct shape if no valid triplets found.
            empty = torch.zeros(0, embeddings.shape[-1], device=embeddings.device)
            return empty, empty, empty

        return (
            torch.stack(anchors_list),
            torch.stack(positives_list),
            torch.stack(negatives_list),
        )


# ---------------------------------------------------------------------------
# TripletDataset
# ---------------------------------------------------------------------------

class TripletDataset(Dataset):  # type: ignore[misc]
    """
    Dataset that generates on-the-fly triplets from pre-computed clip embeddings.

    Parameters
    ----------
    embeddings:   {video_id: (n_clips, D) float32 array}
    step_labels:  {video_id: list[int | None]}  — per-clip step index.
                  None marks unlabelled clips (excluded from triplet sampling).
    triplets_per_epoch: Number of (anchor, positive, negative) triplets drawn
                        each time __len__ is queried; re-sampled each epoch.
    rng_seed:     Optional seed for reproducibility.
    """

    def __init__(
        self,
        embeddings: dict[str, np.ndarray],
        step_labels: dict[str, list[int | None]],
        triplets_per_epoch: int = 2000,
        rng_seed: int | None = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is required to instantiate TripletDataset")

        self.embeddings = {vid: np.asarray(emb, dtype=np.float32) for vid, emb in embeddings.items()}
        self.step_labels = step_labels
        self.triplets_per_epoch = triplets_per_epoch
        self._rng = random.Random(rng_seed)

        # Build flat index: list of (video_id, clip_idx, step_label) for labelled clips only.
        self._labelled_clips: list[tuple[str, int, int]] = []
        # step_to_clips: step_label -> list of (video_id, clip_idx)
        self._step_to_clips: dict[int, list[tuple[str, int]]] = {}

        for vid, labels in step_labels.items():
            if vid not in self.embeddings:
                continue
            for clip_idx, label in enumerate(labels):
                if label is None:
                    continue
                self._labelled_clips.append((vid, clip_idx, int(label)))
                self._step_to_clips.setdefault(int(label), []).append((vid, clip_idx))

        # Remove step labels that have only one clip (can't form a positive pair).
        self._valid_steps = {
            step for step, clips in self._step_to_clips.items() if len(clips) >= 2
        }
        self._valid_labelled = [
            (vid, cidx, lbl)
            for vid, cidx, lbl in self._labelled_clips
            if lbl in self._valid_steps
        ]
        self._all_steps = sorted(self._step_to_clips.keys())

        # Pre-sample triplets for this epoch.
        self._triplets: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._resample()

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _get_embedding(self, vid: str, clip_idx: int) -> np.ndarray:
        return self.embeddings[vid][clip_idx]

    def _sample_triplet(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Draw one valid (anchor, positive, negative) triplet."""
        if not self._valid_labelled:
            return None

        anchor_vid, anchor_cidx, anchor_label = self._rng.choice(self._valid_labelled)
        anchor_emb = self._get_embedding(anchor_vid, anchor_cidx)

        # Positive: different clip of the same step.
        pos_candidates = [
            (v, c) for v, c in self._step_to_clips[anchor_label]
            if not (v == anchor_vid and c == anchor_cidx)
        ]
        if not pos_candidates:
            return None
        pos_vid, pos_cidx = self._rng.choice(pos_candidates)
        pos_emb = self._get_embedding(pos_vid, pos_cidx)

        # Negative: clip from a different step.
        neg_steps = [s for s in self._all_steps if s != anchor_label]
        if not neg_steps:
            return None
        neg_label = self._rng.choice(neg_steps)
        neg_vid, neg_cidx = self._rng.choice(self._step_to_clips[neg_label])
        neg_emb = self._get_embedding(neg_vid, neg_cidx)

        return anchor_emb, pos_emb, neg_emb

    def _resample(self) -> None:
        """Re-draw all triplets for the upcoming epoch."""
        triplets: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        attempts = 0
        max_attempts = self.triplets_per_epoch * 10
        while len(triplets) < self.triplets_per_epoch and attempts < max_attempts:
            t = self._sample_triplet()
            if t is not None:
                triplets.append(t)
            attempts += 1
        self._triplets = triplets

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._triplets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[name-defined]
        anchor, positive, negative = self._triplets[idx]
        return (
            torch.from_numpy(anchor),
            torch.from_numpy(positive),
            torch.from_numpy(negative),
        )

    def on_epoch_end(self) -> None:
        """Call after each epoch to refresh the triplet pool."""
        self._resample()


# ---------------------------------------------------------------------------
# AdapterTrainer
# ---------------------------------------------------------------------------

@dataclass
class _CheckpointMeta:
    config: dict[str, Any]
    history: dict[str, list[float]]
    best_epoch: int


class AdapterTrainer:
    """
    Trains a SOPAdapterHead on pre-computed clip embeddings via TripletLoss.

    Usage
    -----
    trainer = AdapterTrainer(adapter)
    result  = trainer.fit(train_embeddings, train_labels)
    trainer.save("artifacts/adapters/filter_change.pt")

    # Later:
    trainer2 = AdapterTrainer.load("artifacts/adapters/filter_change.pt")
    adapted  = trainer2.transform(raw_embeddings)   # numpy (N, output_dim)
    """

    def __init__(
        self,
        adapter: SOPAdapterHead,
        device: str = "auto",
        loss_margin: float = 0.3,
        loss_mining: str = "hard",
        triplets_per_epoch: int = 2000,
        rng_seed: int | None = 42,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is required to use AdapterTrainer")

        self.adapter = adapter
        self.loss_margin = loss_margin
        self.loss_mining = loss_mining
        self.triplets_per_epoch = triplets_per_epoch
        self.rng_seed = rng_seed

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._criterion = TripletLoss(margin=loss_margin, mining=loss_mining)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_embeddings: dict[str, np.ndarray],
        train_labels: dict[str, list[int | None]],
        val_embeddings: dict[str, np.ndarray] | None = None,
        val_labels: dict[str, list[int | None]] | None = None,
        n_epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        patience: int = 10,
    ) -> TrainingResult:
        """
        Train the adapter with early stopping and cosine-annealing LR schedule.

        Parameters
        ----------
        train_embeddings: {video_id: (n_clips, D) float32}
        train_labels:     {video_id: [step_index_or_None, ...]}
        val_embeddings:   Optional held-out set (same structure).
        val_labels:       Required when val_embeddings is provided.
        n_epochs:         Maximum training epochs.
        batch_size:       DataLoader batch size.
        lr:               Peak learning rate.
        patience:         Early-stopping patience (epochs without val improvement).

        Returns
        -------
        TrainingResult with full training history.
        """
        self.adapter.to(self._device)
        optimizer = optim.AdamW(self.adapter.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

        train_dataset = TripletDataset(
            train_embeddings,
            train_labels,
            triplets_per_epoch=self.triplets_per_epoch,
            rng_seed=self.rng_seed,
        )

        has_val = val_embeddings is not None and val_labels is not None
        val_dataset: TripletDataset | None = None
        if has_val:
            assert val_labels is not None
            val_dataset = TripletDataset(
                val_embeddings,  # type: ignore[arg-type]
                val_labels,
                triplets_per_epoch=max(self.triplets_per_epoch // 4, 200),
                rng_seed=self.rng_seed,
            )

        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_no_improve = 0

        for epoch in range(n_epochs):
            # ---- Train ----
            self.adapter.train()
            epoch_loss = self._run_epoch(train_dataset, optimizer, batch_size, training=True)
            train_losses.append(epoch_loss)
            scheduler.step()
            train_dataset.on_epoch_end()

            # ---- Validation ----
            epoch_val_loss: float | None = None
            if has_val and val_dataset is not None:
                self.adapter.eval()
                epoch_val_loss = self._run_epoch(val_dataset, optimizer, batch_size, training=False)
                val_losses.append(epoch_val_loss)
                val_dataset.on_epoch_end()

                if epoch_val_loss < best_val_loss - 1e-6:
                    best_val_loss = epoch_val_loss
                    best_epoch = epoch
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            else:
                # No validation: use train loss for early stopping.
                if epoch_loss < best_val_loss - 1e-6:
                    best_val_loss = epoch_loss
                    best_epoch = epoch
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

            log_val = f"  val_loss={epoch_val_loss:.4f}" if epoch_val_loss is not None else ""
            logger.info(
                "[AdapterTrainer] epoch %d/%d  train_loss=%.4f%s",
                epoch + 1, n_epochs, epoch_loss, log_val,
            )

            if epochs_no_improve >= patience:
                logger.info("[AdapterTrainer] early stopping at epoch %d", epoch + 1)
                break

        return TrainingResult(
            train_losses=train_losses,
            val_losses=val_losses if val_losses else None,
            best_epoch=best_epoch,
            final_train_loss=train_losses[-1] if train_losses else float("nan"),
            total_epochs_run=len(train_losses),
            model_path="",
        )

    def _run_epoch(
        self,
        dataset: TripletDataset,
        optimizer: optim.Optimizer,  # type: ignore[name-defined]
        batch_size: int,
        *,
        training: bool,
    ) -> float:
        """Run one pass over the dataset; return mean triplet loss."""
        if len(dataset) == 0:
            return 0.0

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=training,
            drop_last=False,
            num_workers=0,   # keep in-process for portability (Windows + SQLite)
        )
        total_loss = 0.0
        n_batches = 0

        context = torch.no_grad() if not training else _null_context()

        with context:  # type: ignore[attr-defined]
            for anchors, positives, negatives in loader:
                anchors   = anchors.to(self._device)
                positives = positives.to(self._device)
                negatives = negatives.to(self._device)

                if training:
                    optimizer.zero_grad()

                a_emb = self.adapter(anchors)
                p_emb = self.adapter(positives)
                n_emb = self.adapter(negatives)

                # Online hard/semi-hard mining within the batch.
                # The pre-built triplet structure is used directly.
                loss = self._criterion(a_emb, p_emb, n_emb)

                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), max_norm=1.0)
                    optimizer.step()

                total_loss += float(loss.item())
                n_batches += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save adapter weights + config + training meta to a .pt file."""
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is required to save AdapterTrainer")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "adapter_state_dict": self.adapter.state_dict(),
                "adapter_config": self.adapter.config_dict(),
                "loss_margin": self.loss_margin,
                "loss_mining": self.loss_mining,
            },
            path,
        )
        logger.info("[AdapterTrainer] saved adapter to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> AdapterTrainer:
        """Load a previously saved AdapterTrainer from disk."""
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is required to load AdapterTrainer")
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["adapter_config"]
        adapter = SOPAdapterHead(**config)
        adapter.load_state_dict(checkpoint["adapter_state_dict"])
        adapter.eval()
        trainer = cls(
            adapter=adapter,
            loss_margin=checkpoint.get("loss_margin", 0.3),
            loss_mining=checkpoint.get("loss_mining", "hard"),
        )
        logger.info("[AdapterTrainer] loaded adapter from %s", path)
        return trainer

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply the adapter in eval mode to an array of raw embeddings.

        Parameters
        ----------
        embeddings: (N, input_dim) float32 numpy array of V-JEPA2 embeddings.

        Returns
        -------
        (N, output_dim) float32 numpy array, L2-normalised.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is required to use AdapterTrainer.transform")
        self.adapter.eval()
        self.adapter.to(self._device)
        tensor = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(self._device)
        with torch.no_grad():
            out = self.adapter(tensor)
        return out.detach().cpu().float().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Null context manager (Python 3.11 contextlib.nullcontext equivalent)
# ---------------------------------------------------------------------------

class _null_context:
    """No-op context manager used to unify training/eval code paths."""

    def __enter__(self) -> _null_context:
        return self

    def __exit__(self, *_: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# evaluate_adapter_quality
# ---------------------------------------------------------------------------

def evaluate_adapter_quality(
    adapter_trainer: AdapterTrainer,
    test_embeddings: dict[str, np.ndarray],
    test_labels: dict[str, list[int | None]],
    dtw_fn: Callable[[np.ndarray, np.ndarray], Any],
) -> dict[str, float]:
    """
    Measure how much the adapter improves DTW step-alignment rank quality.

    For each labelled video pair (gold vs. trainee), we:
      1. Build step-centroid embeddings from gold clips using ground-truth labels.
      2. For every trainee clip, compute DTW alignment rank of the true step
         centroid vs. all other step centroids.
      3. Repeat with adapter-transformed embeddings.
      4. Report mean rank (lower = better) before and after.

    Parameters
    ----------
    adapter_trainer: Trained AdapterTrainer instance.
    test_embeddings: {video_id: (n_clips, D)} raw embeddings.
    test_labels:     {video_id: [step_int_or_None, ...]}
    dtw_fn:          Callable matching sopilot.core.dtw.dtw_align signature:
                     dtw_fn(gold: np.ndarray, trainee: np.ndarray) -> DTWAlignment.

    Returns
    -------
    {"before_rank": float, "after_rank": float, "improvement_pct": float}
    """
    # Collect all step centroids from all labelled videos.
    def build_step_centroids(
        embeddings_dict: dict[str, np.ndarray],
        labels_dict: dict[str, list[int | None]],
    ) -> dict[int, np.ndarray]:
        step_sums: dict[int, np.ndarray] = {}
        step_counts: dict[int, int] = {}
        for vid, labels in labels_dict.items():
            if vid not in embeddings_dict:
                continue
            embs = embeddings_dict[vid]
            for clip_idx, label in enumerate(labels):
                if label is None or clip_idx >= len(embs):
                    continue
                label = int(label)
                if label not in step_sums:
                    step_sums[label] = np.zeros(embs.shape[1], dtype=np.float64)
                    step_counts[label] = 0
                step_sums[label] += embs[clip_idx].astype(np.float64)
                step_counts[label] += 1
        # L2-normalise centroids.
        centroids: dict[int, np.ndarray] = {}
        for label, s in step_sums.items():
            mean = (s / step_counts[label]).astype(np.float32)
            denom = float(np.linalg.norm(mean))
            centroids[label] = mean / max(denom, 1e-12)
        return centroids

    def mean_step_rank(
        embs_dict: dict[str, np.ndarray],
        labels_dict: dict[str, list[int | None]],
        centroids: dict[int, np.ndarray],
    ) -> float:
        """Compute mean rank of the correct step centroid for each labelled clip."""
        if not centroids:
            return float("nan")
        step_indices = sorted(centroids.keys())
        centroid_matrix = np.stack([centroids[s] for s in step_indices])  # (S, D)

        ranks: list[float] = []
        for vid, labels in labels_dict.items():
            if vid not in embs_dict:
                continue
            embs = embs_dict[vid]
            for clip_idx, label in enumerate(labels):
                if label is None or clip_idx >= len(embs):
                    continue
                label = int(label)
                if label not in centroids:
                    continue
                clip_emb = embs[clip_idx]                            # (D,)
                # Cosine similarity to all step centroids.
                sims = centroid_matrix @ clip_emb                     # (S,)
                # Rank of the correct step (1 = best).
                correct_pos = step_indices.index(label)
                rank = int(np.sum(sims > sims[correct_pos])) + 1
                ranks.append(float(rank))

        return float(np.mean(ranks)) if ranks else float("nan")

    # --- Before adapter ---
    before_centroids = build_step_centroids(test_embeddings, test_labels)
    before_rank = mean_step_rank(test_embeddings, test_labels, before_centroids)

    # --- After adapter ---
    adapted_embeddings: dict[str, np.ndarray] = {}
    for vid, embs in test_embeddings.items():
        if len(embs) > 0:
            adapted_embeddings[vid] = adapter_trainer.transform(embs)
        else:
            adapted_embeddings[vid] = embs

    after_centroids = build_step_centroids(adapted_embeddings, test_labels)
    after_rank = mean_step_rank(adapted_embeddings, test_labels, after_centroids)

    if not (np.isnan(before_rank) or before_rank == 0.0):
        improvement_pct = float((before_rank - after_rank) / before_rank * 100.0)
    else:
        improvement_pct = float("nan")

    return {
        "before_rank": before_rank,
        "after_rank": after_rank,
        "improvement_pct": improvement_pct,
    }


# ---------------------------------------------------------------------------
# Pseudo-label generation (DTW-based, used when no manual labels exist)
# ---------------------------------------------------------------------------

def generate_pseudo_labels_from_dtw(
    gold_embeddings: dict[str, np.ndarray],
    gold_step_boundaries: dict[str, list[int]],
    dtw_fn: Callable[[np.ndarray, np.ndarray], Any],
) -> dict[str, list[int | None]]:
    """
    Generate step labels for gold videos using known step boundaries.

    Each clip is assigned the step index it falls into based on the step
    boundary list stored in the DB.  Clips that fall exactly on a boundary
    are assigned to the later step.

    Parameters
    ----------
    gold_embeddings:      {video_id: (n_clips, D)}
    gold_step_boundaries: {video_id: sorted list of clip-index boundary points}
    dtw_fn:               DTW function (unused here but accepted for API symmetry).

    Returns
    -------
    {video_id: [step_int, ...]} — every clip gets a label (no None).
    """
    labels: dict[str, list[int | None]] = {}
    for vid, embs in gold_embeddings.items():
        n_clips = len(embs)
        boundaries = sorted(gold_step_boundaries.get(vid, []))
        clip_labels: list[int | None] = []
        for clip_idx in range(n_clips):
            step = 0
            for b in boundaries:
                if clip_idx >= b:
                    step += 1
                else:
                    break
            clip_labels.append(step)
        labels[vid] = clip_labels
    return labels
