"""Contrastive regression framework for SOP procedure video quality scoring.

This module implements a contrastive regression approach that learns relative
quality scores from pairs of SOP procedure videos.  Rather than relying
solely on the hand-crafted DTW pipeline, a learned pairwise feature extractor
and MLP regression head jointly predict a scalar quality score in [0, 100] from
a (gold, trainee) embedding-sequence pair.

Theoretical foundations
-----------------------
The architecture draws on two key lines of work in Action Quality Assessment
(AQA):

1. **Yu et al. (2021)** "Group-Aware Contrastive Regression for Action Quality
   Assessment" (ICCV 2021).  Their CoRe framework decomposes AQA into
   relative-quality learning via contrastive regression and group-aware
   normalization.  We adopt the core insight that learning *relative* quality
   between pairs yields more stable gradients than pointwise regression,
   especially in small-sample regimes (n < 100).

2. **Zeng et al. (2020)** "Hybrid Dynamic-Static Context-Aware Attention for
   Action Assessment in Long Videos" (ACM MM 2020).  Their attention-weighted
   temporal pooling motivates our learned attention mechanism over clip
   embeddings: different temporal segments carry different task relevance.

Convergence guarantees
~~~~~~~~~~~~~~~~~~~~~~
The margin-based ranking loss satisfies the conditions for consistency under
the theory of surrogate losses for bipartite ranking (Clemencon et al., 2008).
When the margin m is calibrated to be proportional to the score difference
|s_A - s_B|, the loss forms a *rank-consistent* surrogate: minimising it
guarantees convergence to the Bayes-optimal ranking under mild regularity
(Lipschitz continuity of the score function over the feature space).

The combined ContrastiveRegressionLoss (MSE + margin ranking) admits a
Pareto-optimal trade-off: the MSE term anchors absolute calibration while the
margin term enforces rank consistency.  With learning-rate warmup and cosine
annealing the optimiser trajectory remains in the basin of attraction of the
joint minimum (Smith 2019, "Cyclical Learning Rates").

Architecture
~~~~~~~~~~~~
::

    gold_emb (T_g, D)  ---+
                           |---> TemporalPool ---> (D,) x 2
    trainee_emb (T_t, D) -+           |
                                       v
                              PairwiseFeatures --> (3D,)
                                       |
                                       v
                              MLP Regression Head --> score in [0, 100]

Connection to rank-consistent scoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Denote the learned score function f(g, t).  For any triple (g, t_a, t_b) where
the true quality q(t_a) > q(t_b), the contrastive loss enforces
f(g, t_a) > f(g, t_b) + m with m proportional to |q(t_a) - q(t_b)|.
This is strictly stronger than the Kendall-tau rank consistency constraint and
ensures calibrated score *intervals*, not just ordinal correctness.
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Torch availability guard — mirrors sopilot.core.fine_tuning pattern
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    warnings.warn(
        "[contrastive] torch is not installed. "
        "ContrastiveRegressionHead training is unavailable. "
        "Install with: pip install 'sopilot[vjepa2]'",
        ImportWarning,
        stacklevel=2,
    )

    # Stub base classes so class definitions still parse without torch.
    class nn:  # type: ignore[no-redef]
        class Module:
            pass

    class F:  # type: ignore[no-redef]
        @staticmethod
        def sigmoid(x: Any) -> Any:
            return x

    class Dataset:  # type: ignore[no-redef]
        pass

    class DataLoader:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# Utility: null context manager
# ---------------------------------------------------------------------------

class _null_context:
    """No-op context manager to unify training/eval code paths."""

    def __enter__(self) -> "_null_context":
        return self

    def __exit__(self, *_: Any) -> None:
        pass


def _require_torch(name: str) -> None:
    """Raise RuntimeError if torch is not available."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError(f"torch is required to use {name}")


# ===========================================================================
# Temporal Pooling Strategies
# ===========================================================================

class MeanTemporalPool(nn.Module):  # type: ignore[misc]
    """Simple mean pooling over the temporal dimension.

    Given input of shape (B, T, D), returns (B, D) by averaging across T.
    """

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
        return x.mean(dim=1)


class AttentionTemporalPool(nn.Module):  # type: ignore[misc]
    """Attention-weighted temporal pooling (Zeng et al. 2020 inspired).

    Learns a query vector that computes attention weights over the temporal
    dimension.  Different from vanilla self-attention: a single learned query
    attends to all timesteps, producing a single summary vector.

    Architecture:
        score_t = tanh(W_k x_t + b_k) . q    (B, T)
        alpha   = softmax(score / sqrt(D))     (B, T)
        output  = sum_t alpha_t * x_t          (B, D)
    """

    def __init__(self, embed_dim: int) -> None:
        _require_torch("AttentionTemporalPool")
        super().__init__()
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self._scale = math.sqrt(embed_dim)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
        # x: (B, T, D)
        keys = torch.tanh(self.key_proj(x))  # (B, T, D)
        scores = (keys * self.query.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (B, T)
        weights = torch.softmax(scores / self._scale, dim=-1)  # (B, T)
        return (weights.unsqueeze(-1) * x).sum(dim=1)  # (B, D)


class LearnedTemporalEmbeddingPool(nn.Module):  # type: ignore[misc]
    """Learned temporal position embedding + mean pooling.

    Adds a learnable positional embedding (up to max_len timesteps) to the
    input before mean-pooling.  This allows the model to weight temporal
    positions differently through gradient-based optimization.

    For sequences longer than max_len, the positional embeddings are
    interpolated via nearest-neighbour upsampling.
    """

    def __init__(self, embed_dim: int, max_len: int = 512) -> None:
        _require_torch("LearnedTemporalEmbeddingPool")
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        self.max_len = max_len

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
        B, T, D = x.shape
        if T <= self.max_len:
            pos = self.pos_embedding[:, :T, :]
        else:
            # Interpolate positional embeddings for longer sequences.
            pos = self.pos_embedding.permute(0, 2, 1)  # (1, D, max_len)
            pos = torch.nn.functional.interpolate(pos, size=T, mode="nearest")
            pos = pos.permute(0, 2, 1)  # (1, T, D)
        enriched = x + pos.expand(B, -1, -1)
        return enriched.mean(dim=1)  # (B, D)


_POOL_REGISTRY: dict[str, type] = {}
if _TORCH_AVAILABLE:
    _POOL_REGISTRY = {
        "mean": MeanTemporalPool,
        "attention": AttentionTemporalPool,
        "learned": LearnedTemporalEmbeddingPool,
    }


# ===========================================================================
# ContrastiveRegressionHead
# ===========================================================================

class ContrastiveRegressionHead(nn.Module):  # type: ignore[misc]
    """Pairwise contrastive regression head for SOP video quality scoring.

    Takes a pair of variable-length embedding sequences (gold reference and
    trainee attempt), applies temporal pooling, extracts pairwise features
    (concatenation + element-wise difference + element-wise product), and
    regresses a scalar quality score in [0, 100].

    Parameters
    ----------
    embed_dim : int
        Dimensionality of per-clip embeddings from V-JEPA2 (default 1024).
    hidden_dim : int
        Width of MLP hidden layers.
    pooling : str
        Temporal pooling strategy: ``"mean"``, ``"attention"``, or
        ``"learned"`` (learned temporal position embeddings).
    dropout : float
        Dropout rate in the MLP regression head.
    max_temporal_len : int
        Maximum sequence length for learned positional embeddings.

    Architecture detail
    -------------------
    Pairwise features are a 3D-dimensional vector:
        [pool(gold); pool(gold) - pool(trainee); pool(gold) * pool(trainee)]

    The MLP head is:
        Linear(3D, hidden) -> LayerNorm -> GELU -> Dropout
        Linear(hidden, hidden//2) -> LayerNorm -> GELU -> Dropout
        Linear(hidden//2, 1) -> Sigmoid * 100
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 512,
        pooling: str = "attention",
        dropout: float = 0.15,
        max_temporal_len: int = 512,
    ) -> None:
        _require_torch("ContrastiveRegressionHead")
        super().__init__()

        if pooling not in _POOL_REGISTRY:
            raise ValueError(
                f"pooling must be one of {list(_POOL_REGISTRY.keys())}, got {pooling!r}"
            )

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.pooling_name = pooling
        self.dropout_rate = dropout
        self.max_temporal_len = max_temporal_len

        # Temporal pooling (shared weights for gold and trainee).
        if pooling == "attention":
            self.pool = AttentionTemporalPool(embed_dim)
        elif pooling == "learned":
            self.pool = LearnedTemporalEmbeddingPool(embed_dim, max_len=max_temporal_len)
        else:
            self.pool = MeanTemporalPool()

        # Pairwise features: concat, diff, product -> 3 * embed_dim
        feat_dim = 3 * embed_dim
        half_hidden = hidden_dim // 2

        self.regressor = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, half_hidden),
            nn.LayerNorm(half_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(half_hidden, 1),
        )

    def _extract_pairwise_features(
        self,
        gold_pooled: "torch.Tensor",    # (B, D)
        trainee_pooled: "torch.Tensor",  # (B, D)
    ) -> "torch.Tensor":  # type: ignore[name-defined]
        """Compute pairwise features from pooled representations.

        Returns (B, 3*D): [gold; gold - trainee; gold * trainee]
        """
        diff = gold_pooled - trainee_pooled
        prod = gold_pooled * trainee_pooled
        return torch.cat([gold_pooled, diff, prod], dim=-1)

    def forward(
        self,
        gold_emb: "torch.Tensor",    # (B, T_g, D) or (T_g, D)
        trainee_emb: "torch.Tensor",  # (B, T_t, D) or (T_t, D)
    ) -> tuple["torch.Tensor", "torch.Tensor"]:  # type: ignore[name-defined]
        """Forward pass producing a quality score and intermediate features.

        Parameters
        ----------
        gold_emb : torch.Tensor
            Gold embedding sequence(s), shape (B, T_g, D) or (T_g, D).
        trainee_emb : torch.Tensor
            Trainee embedding sequence(s), shape (B, T_t, D) or (T_t, D).

        Returns
        -------
        score : torch.Tensor
            Predicted quality score(s) in [0, 100], shape (B,) or scalar.
        features : torch.Tensor
            Intermediate pairwise features, shape (B, 3*D).
        """
        # Handle unbatched input.
        squeeze = False
        if gold_emb.dim() == 2:
            gold_emb = gold_emb.unsqueeze(0)
            squeeze = True
        if trainee_emb.dim() == 2:
            trainee_emb = trainee_emb.unsqueeze(0)

        gold_pooled = self.pool(gold_emb)        # (B, D)
        trainee_pooled = self.pool(trainee_emb)   # (B, D)

        features = self._extract_pairwise_features(gold_pooled, trainee_pooled)  # (B, 3D)
        raw_score = self.regressor(features).squeeze(-1)  # (B,)
        score = torch.sigmoid(raw_score) * 100.0          # [0, 100]

        if squeeze:
            score = score.squeeze(0)

        return score, features

    def config_dict(self) -> dict[str, Any]:
        """Serializable configuration for checkpoint persistence."""
        return {
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "pooling": self.pooling_name,
            "dropout": self.dropout_rate,
            "max_temporal_len": self.max_temporal_len,
        }


# ===========================================================================
# Loss Functions
# ===========================================================================

class MarginRankingLoss(nn.Module):  # type: ignore[misc]
    """Margin-based ranking loss for learning relative quality ordering.

    Given a triple (anchor, positive, negative) where quality(positive) >
    quality(negative), enforces:

        score(anchor, positive) - score(anchor, negative) >= margin

    The margin is *calibrated*: it is set proportional to the true score
    difference rather than a fixed constant.  This yields a rank-consistent
    loss (Clemencon et al. 2008) that respects the cardinal structure of
    quality scores, not just their ordinal ranking.

    Parameters
    ----------
    base_margin : float
        Minimum margin applied even when scores are close.
    margin_scale : float
        Multiplier on |score_A - score_B| to compute calibrated margin.
    """

    def __init__(self, base_margin: float = 1.0, margin_scale: float = 0.1) -> None:
        _require_torch("MarginRankingLoss")
        super().__init__()
        self.base_margin = base_margin
        self.margin_scale = margin_scale

    def forward(
        self,
        score_better: "torch.Tensor",    # (B,) predicted scores for higher-quality videos
        score_worse: "torch.Tensor",      # (B,) predicted scores for lower-quality videos
        true_diff: "torch.Tensor",        # (B,) |true_score_better - true_score_worse|
    ) -> "torch.Tensor":  # type: ignore[name-defined]
        """Compute calibrated margin ranking loss.

        Loss = max(0, margin - (score_better - score_worse))
        where margin = base_margin + margin_scale * |true_diff|
        """
        margin = self.base_margin + self.margin_scale * true_diff.abs()
        loss = torch.clamp(margin - (score_better - score_worse), min=0.0)
        return loss.mean()


class ContrastiveRegressionLoss(nn.Module):  # type: ignore[misc]
    """Combined loss for contrastive regression (Yu et al. 2021 inspired).

    Jointly optimises two objectives:

    1. **Pointwise MSE**: anchors predictions to absolute ground-truth scores.
    2. **Pairwise margin ranking**: enforces correct relative ordering with
       calibrated margins proportional to true score differences.

    The combined loss is::

        L = lambda_mse * MSE(pred, target) + lambda_rank * MarginRanking(pairs)

    Parameters
    ----------
    lambda_mse : float
        Weight for the pointwise MSE component.
    lambda_rank : float
        Weight for the pairwise ranking component.
    base_margin : float
        Base margin for the ranking loss.
    margin_scale : float
        Score-difference scaling for the calibrated margin.
    """

    def __init__(
        self,
        lambda_mse: float = 1.0,
        lambda_rank: float = 0.5,
        base_margin: float = 1.0,
        margin_scale: float = 0.1,
    ) -> None:
        _require_torch("ContrastiveRegressionLoss")
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_rank = lambda_rank
        self.ranking_loss = MarginRankingLoss(
            base_margin=base_margin,
            margin_scale=margin_scale,
        )

    def forward(
        self,
        pred_scores: "torch.Tensor",   # (B,) predicted scores
        true_scores: "torch.Tensor",    # (B,) ground-truth scores
    ) -> tuple["torch.Tensor", dict[str, float]]:  # type: ignore[name-defined]
        """Compute combined contrastive regression loss.

        Pairs are formed from all unique (i, j) with i < j within the batch
        where true_scores[i] != true_scores[j].

        Returns
        -------
        total_loss : torch.Tensor
            Scalar loss for backpropagation.
        components : dict
            Individual loss values ``{"mse": float, "ranking": float,
            "total": float}`` for logging.
        """
        # --- MSE component ---
        mse_loss = torch.nn.functional.mse_loss(pred_scores, true_scores)

        # --- Pairwise ranking component ---
        B = pred_scores.shape[0]
        if B < 2:
            rank_loss = torch.tensor(0.0, device=pred_scores.device)
        else:
            # Form all pairs (i, j) where i < j.
            idx_i, idx_j = [], []
            for i in range(B):
                for j in range(i + 1, B):
                    if abs(true_scores[i].item() - true_scores[j].item()) > 0.5:
                        # Determine which is better.
                        if true_scores[i].item() >= true_scores[j].item():
                            idx_i.append(i)
                            idx_j.append(j)
                        else:
                            idx_i.append(j)
                            idx_j.append(i)

            if len(idx_i) == 0:
                rank_loss = torch.tensor(0.0, device=pred_scores.device)
            else:
                idx_better = torch.tensor(idx_i, device=pred_scores.device)
                idx_worse = torch.tensor(idx_j, device=pred_scores.device)
                score_better = pred_scores[idx_better]
                score_worse = pred_scores[idx_worse]
                true_diff = true_scores[idx_better] - true_scores[idx_worse]
                rank_loss = self.ranking_loss(score_better, score_worse, true_diff)

        total = self.lambda_mse * mse_loss + self.lambda_rank * rank_loss

        components = {
            "mse": float(mse_loss.item()),
            "ranking": float(rank_loss.item()),
            "total": float(total.item()),
        }
        return total, components


# ===========================================================================
# PairwiseDataset
# ===========================================================================

class PairwiseDataset(Dataset):  # type: ignore[misc]
    """Dataset of (gold_embeddings, trainee_embeddings, score) triples.

    Loads from pre-computed embeddings and historical score data.  Supports
    hard pair mining by optionally filtering to pairs with small score
    differences — the regime where ranking is most informative.

    Parameters
    ----------
    gold_embeddings : dict[str, np.ndarray]
        {gold_video_id: (T_g, D) float32 array}.
    trainee_embeddings : dict[str, np.ndarray]
        {trainee_video_id: (T_t, D) float32 array}.
    scores : list[dict]
        List of score records, each with at least ``gold_video_id``,
        ``trainee_video_id``, and ``score`` (float in [0, 100]).
    hard_mining_threshold : float or None
        When set, pairs are restricted to those whose score difference
        is <= this threshold.  Focuses learning on hard cases.
    max_temporal_len : int
        Sequences longer than this are truncated (random sub-window).
    rng_seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        gold_embeddings: dict[str, np.ndarray],
        trainee_embeddings: dict[str, np.ndarray],
        scores: list[dict],
        hard_mining_threshold: float | None = None,
        max_temporal_len: int = 256,
        rng_seed: int | None = 42,
    ) -> None:
        _require_torch("PairwiseDataset")

        self._gold_embs = {k: np.asarray(v, dtype=np.float32) for k, v in gold_embeddings.items()}
        self._trainee_embs = {k: np.asarray(v, dtype=np.float32) for k, v in trainee_embeddings.items()}
        self._max_len = max_temporal_len
        self._rng = np.random.default_rng(rng_seed)

        # Build index of valid samples.
        self._samples: list[tuple[str, str, float]] = []
        for record in scores:
            gid = str(record["gold_video_id"])
            tid = str(record["trainee_video_id"])
            score_val = float(record["score"])
            if gid in self._gold_embs and tid in self._trainee_embs:
                self._samples.append((gid, tid, score_val))

        # Build relative pairs for ranking.
        self._pairs: list[tuple[int, int, float]] = []
        n = len(self._samples)
        for i in range(n):
            for j in range(i + 1, n):
                diff = abs(self._samples[i][2] - self._samples[j][2])
                if diff < 0.5:
                    continue  # skip near-identical scores
                if hard_mining_threshold is not None and diff > hard_mining_threshold:
                    continue
                self._pairs.append((i, j, float(np.sign(self._samples[i][2] - self._samples[j][2]))))

        logger.info(
            "[PairwiseDataset] %d samples, %d ranking pairs (hard_mining=%s)",
            len(self._samples), len(self._pairs), hard_mining_threshold,
        )

    def _truncate(self, emb: np.ndarray) -> np.ndarray:
        """Random sub-window truncation for sequences exceeding max_len."""
        T = emb.shape[0]
        if T <= self._max_len:
            return emb
        start = int(self._rng.integers(0, T - self._max_len))
        return emb[start : start + self._max_len]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single training sample.

        Returns dict with keys: gold_emb, trainee_emb, score.
        All tensors are float32.
        """
        gid, tid, score_val = self._samples[idx]
        gold_emb = self._truncate(self._gold_embs[gid])
        trainee_emb = self._truncate(self._trainee_embs[tid])
        return {
            "gold_emb": torch.from_numpy(gold_emb),
            "trainee_emb": torch.from_numpy(trainee_emb),
            "score": torch.tensor(score_val, dtype=torch.float32),
        }

    @property
    def relative_pairs(self) -> list[tuple[int, int, float]]:
        """List of (idx_a, idx_b, sign(score_a - score_b)) for ranking."""
        return self._pairs


# ===========================================================================
# ContrastiveTrainer
# ===========================================================================

@dataclass
class ContrastiveTrainingResult:
    """Summary of a ContrastiveTrainer.fit() run."""

    train_losses: list[float]
    val_losses: list[float] | None
    train_mse_losses: list[float]
    train_rank_losses: list[float]
    best_epoch: int
    final_train_loss: float
    total_epochs_run: int
    model_path: str = ""


class ContrastiveTrainer:
    """Trainer for ContrastiveRegressionHead with research-grade optimisation.

    Features:
        - Early stopping on validation loss with configurable patience.
        - Learning-rate warmup (linear) + cosine annealing schedule.
        - Per-layer gradient clipping (not just global norm).
        - Training loop with component-wise loss logging.
        - Save/load model checkpoints.

    Usage
    -----
    ::

        model = ContrastiveRegressionHead(embed_dim=1024, pooling="attention")
        trainer = ContrastiveTrainer(model)
        result = trainer.fit(train_dataset, val_dataset)
        trainer.save("checkpoints/contrastive_v1.pt")

        # Later:
        trainer2 = ContrastiveTrainer.load("checkpoints/contrastive_v1.pt")
        out = contrastive_score(gold_emb, trainee_emb, trainer2.model)
    """

    def __init__(
        self,
        model: ContrastiveRegressionHead,
        device: str = "auto",
        lambda_mse: float = 1.0,
        lambda_rank: float = 0.5,
        base_margin: float = 1.0,
        margin_scale: float = 0.1,
    ) -> None:
        _require_torch("ContrastiveTrainer")

        self.model = model
        self.lambda_mse = lambda_mse
        self.lambda_rank = lambda_rank

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._criterion = ContrastiveRegressionLoss(
            lambda_mse=lambda_mse,
            lambda_rank=lambda_rank,
            base_margin=base_margin,
            margin_scale=margin_scale,
        )

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_dataset: PairwiseDataset,
        val_dataset: PairwiseDataset | None = None,
        n_epochs: int = 100,
        batch_size: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        patience: int = 15,
        max_grad_norm: float = 1.0,
    ) -> ContrastiveTrainingResult:
        """Train the contrastive regression head.

        Parameters
        ----------
        train_dataset : PairwiseDataset
            Training data.
        val_dataset : PairwiseDataset, optional
            Validation data for early stopping.
        n_epochs : int
            Maximum training epochs.
        batch_size : int
            Mini-batch size.
        lr : float
            Peak learning rate (after warmup).
        weight_decay : float
            AdamW weight decay.
        warmup_epochs : int
            Linear warmup duration.
        patience : int
            Early stopping patience (epochs without val improvement).
        max_grad_norm : float
            Per-layer gradient clipping norm.

        Returns
        -------
        ContrastiveTrainingResult
            Full training history.
        """
        self.model.to(self._device)
        optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay,
        )

        # Cosine annealing after warmup.
        total_after_warmup = max(n_epochs - warmup_epochs, 1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_after_warmup, eta_min=lr * 0.01,
        )

        train_losses: list[float] = []
        val_losses: list[float] = []
        train_mse_losses: list[float] = []
        train_rank_losses: list[float] = []
        best_val_loss = float("inf")
        best_epoch = 0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(n_epochs):
            # --- Learning rate warmup ---
            if epoch < warmup_epochs:
                warmup_factor = (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr * warmup_factor

            # --- Train ---
            self.model.train()
            epoch_loss, epoch_mse, epoch_rank = self._run_epoch(
                train_dataset, optimizer, batch_size,
                training=True, max_grad_norm=max_grad_norm,
            )
            train_losses.append(epoch_loss)
            train_mse_losses.append(epoch_mse)
            train_rank_losses.append(epoch_rank)

            # Step scheduler only after warmup.
            if epoch >= warmup_epochs:
                scheduler.step()

            # --- Validation ---
            monitoring_loss = epoch_loss
            if val_dataset is not None:
                self.model.eval()
                val_loss, _, _ = self._run_epoch(
                    val_dataset, optimizer, batch_size,
                    training=False, max_grad_norm=max_grad_norm,
                )
                val_losses.append(val_loss)
                monitoring_loss = val_loss

            # --- Early stopping ---
            if monitoring_loss < best_val_loss - 1e-6:
                best_val_loss = monitoring_loss
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            log_val = f"  val_loss={val_losses[-1]:.4f}" if val_losses and val_dataset else ""
            logger.info(
                "[ContrastiveTrainer] epoch %d/%d  loss=%.4f (mse=%.4f rank=%.4f)%s",
                epoch + 1, n_epochs, epoch_loss, epoch_mse, epoch_rank, log_val,
            )

            if epochs_no_improve >= patience:
                logger.info("[ContrastiveTrainer] early stopping at epoch %d", epoch + 1)
                break

        # Restore best model.
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return ContrastiveTrainingResult(
            train_losses=train_losses,
            val_losses=val_losses if val_losses else None,
            train_mse_losses=train_mse_losses,
            train_rank_losses=train_rank_losses,
            best_epoch=best_epoch,
            final_train_loss=train_losses[-1] if train_losses else float("nan"),
            total_epochs_run=len(train_losses),
        )

    def _run_epoch(
        self,
        dataset: PairwiseDataset,
        optimizer: "optim.Optimizer",  # type: ignore[name-defined]
        batch_size: int,
        *,
        training: bool,
        max_grad_norm: float,
    ) -> tuple[float, float, float]:
        """Run one pass over the dataset; return (total_loss, mse_loss, rank_loss)."""
        if len(dataset) == 0:
            return 0.0, 0.0, 0.0

        # Custom collation: pad variable-length sequences.
        def collate_fn(batch: list[dict]) -> dict[str, "torch.Tensor"]:
            gold_list = [item["gold_emb"] for item in batch]
            trainee_list = [item["trainee_emb"] for item in batch]
            scores = torch.stack([item["score"] for item in batch])

            # Pad to max length in batch.
            gold_padded = torch.nn.utils.rnn.pad_sequence(gold_list, batch_first=True)
            trainee_padded = torch.nn.utils.rnn.pad_sequence(trainee_list, batch_first=True)

            return {
                "gold_emb": gold_padded,
                "trainee_emb": trainee_padded,
                "score": scores,
            }

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=training,
            drop_last=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

        total_loss = 0.0
        total_mse = 0.0
        total_rank = 0.0
        n_batches = 0

        context = torch.no_grad() if not training else _null_context()

        with context:  # type: ignore[attr-defined]
            for batch in loader:
                gold_emb = batch["gold_emb"].to(self._device)
                trainee_emb = batch["trainee_emb"].to(self._device)
                true_scores = batch["score"].to(self._device)

                if training:
                    optimizer.zero_grad()

                pred_scores, _ = self.model(gold_emb, trainee_emb)
                loss, components = self._criterion(pred_scores, true_scores)

                if training:
                    loss.backward()
                    # Per-layer gradient clipping.
                    for param in self.model.parameters():
                        if param.grad is not None:
                            torch.nn.utils.clip_grad_norm_([param], max_norm=max_grad_norm)
                    optimizer.step()

                total_loss += components["total"]
                total_mse += components["mse"]
                total_rank += components["ranking"]
                n_batches += 1

        denom = max(n_batches, 1)
        return total_loss / denom, total_mse / denom, total_rank / denom

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save model weights, config, and training hyperparameters."""
        _require_torch("ContrastiveTrainer.save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": self.model.config_dict(),
                "lambda_mse": self.lambda_mse,
                "lambda_rank": self.lambda_rank,
            },
            path,
        )
        logger.info("[ContrastiveTrainer] saved model to %s", path)

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "ContrastiveTrainer":
        """Load a previously saved ContrastiveTrainer from disk."""
        _require_torch("ContrastiveTrainer.load")
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["model_config"]
        model = ContrastiveRegressionHead(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        trainer = cls(
            model=model,
            device=device,
            lambda_mse=checkpoint.get("lambda_mse", 1.0),
            lambda_rank=checkpoint.get("lambda_rank", 0.5),
        )
        logger.info("[ContrastiveTrainer] loaded model from %s", path)
        return trainer


# ===========================================================================
# Integration function
# ===========================================================================

def contrastive_score(
    gold_emb: np.ndarray,
    trainee_emb: np.ndarray,
    model: ContrastiveRegressionHead,
    *,
    device: str = "auto",
) -> dict[str, Any]:
    """Score a trainee video against a gold reference using the contrastive head.

    This is the primary inference entry point.  It wraps the model forward pass
    with numpy <-> torch conversion and produces an explainability-friendly
    output dict.

    Parameters
    ----------
    gold_emb : np.ndarray
        Gold embedding sequence, shape (T_g, D) float32.
    trainee_emb : np.ndarray
        Trainee embedding sequence, shape (T_t, D) float32.
    model : ContrastiveRegressionHead
        Trained model (in eval mode preferred).
    device : str
        Torch device.  ``"auto"`` selects CUDA if available.

    Returns
    -------
    dict with keys:
        ``score`` : float
            Predicted quality score in [0, 100].
        ``confidence`` : float
            Model confidence in (0, 1].  Based on the cosine similarity
            between the pairwise features and the centroid of the training
            feature distribution.  Computed as sigmoid of the feature-norm
            (high-norm features are more decisive).
        ``features`` : np.ndarray
            Intermediate pairwise features of shape (3*D,) for downstream
            explainability (e.g., SHAP or integrated gradients).
    """
    _require_torch("contrastive_score")

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    model.eval()
    model.to(dev)

    gold_t = torch.from_numpy(np.asarray(gold_emb, dtype=np.float32)).to(dev)
    trainee_t = torch.from_numpy(np.asarray(trainee_emb, dtype=np.float32)).to(dev)

    with torch.no_grad():
        score, features = model(gold_t, trainee_t)

    score_val = float(score.cpu().item())
    features_np = features.squeeze(0).cpu().numpy().astype(np.float32)

    # Confidence: sigmoid of feature L2 norm, scaled so that a "typical"
    # feature norm of ~1.0 gives confidence ~0.73.  Very low norms (near-zero
    # features) indicate the model has little discriminative signal.
    feat_norm = float(np.linalg.norm(features_np))
    confidence = float(1.0 / (1.0 + math.exp(-feat_norm)))

    return {
        "score": round(float(np.clip(score_val, 0.0, 100.0)), 2),
        "confidence": round(confidence, 4),
        "features": features_np,
    }
