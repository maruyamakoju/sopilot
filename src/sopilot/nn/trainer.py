"""SOPilot joint training pipeline.

Four-phase training strategy:
    Phase 1 (Self-supervised): Contrastive projection + pseudo-label segmentation
    Phase 2 (Warm-start scoring): Train ScoringHead to match current penalty formula
    Phase 3 (Fine-tune with human labels): End-to-end gradient flow
    Phase 4 (Calibration): Isotonic calibration + statistical evaluation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .projection_head import (
    ProjectionHead,
    NTXentLoss,
    StepPairMiner,
    save_projection_head,
)
from .soft_dtw import SoftDTW, SoftDTWAlignment
from .step_segmenter import (
    NeuralStepSegmenter,
    SegmentationLoss,
    generate_pseudo_labels,
    save_segmenter,
)
from .scoring_head import (
    ScoringHead,
    IsotonicCalibrator,
    METRIC_KEYS,
    save_scoring_head,
)
from .asformer import (
    ASFormer,
    ASFormerLoss,
    save_asformer,
)
from .dilate_loss import SOPDilateLoss
from .conformal import SplitConformalPredictor

logger = logging.getLogger(__name__)


class AlignmentFeatureBridge(nn.Module):
    """Extract scoring-compatible features from Soft-DTW alignment.

    Computes differentiable approximations of the 15 penalty metrics
    from the soft alignment matrix, enabling true end-to-end training.
    """

    def __init__(self, d_embedding: int = 128) -> None:
        super().__init__()
        # Small network to extract features from alignment matrix.
        # Sigmoid output clamps features to [0, 1], matching the typical
        # scale of the 15 raw penalty metrics the ScoringHead expects.
        self.alignment_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),  # Normalize alignment to fixed size
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 15),  # Output 15 features to match ScoringHead input
            nn.Sigmoid(),       # Clamp to [0, 1] — prevents BN scale mismatch
        )

    def forward(
        self, alignment_matrix: torch.Tensor, sdtw_distance: torch.Tensor
    ) -> torch.Tensor:
        """Extract 15 scoring features from alignment matrix and distance.

        The alignment matrix already encodes the Soft-DTW distance signal
        (it is computed from the forward/backward DP that produces the
        distance), so the sdtw_distance scalar is accepted for API
        compatibility but not used — the MLP learns to extract all
        relevant features from the alignment structure.

        Args:
            alignment_matrix: (M, N) soft alignment from SoftDTWAlignment.
            sdtw_distance: scalar Soft-DTW distance (unused, retained for API).

        Returns:
            (1, 15) feature vector in [0, 1] compatible with ScoringHead.
        """
        # Add batch and channel dims: (1, 1, M, N)
        x = alignment_matrix.unsqueeze(0).unsqueeze(0)
        features = self.alignment_encoder(x)  # (1, 15) — Sigmoid clamped to [0, 1]
        return features


@dataclass
class TrainingConfig:
    """Configuration for neural training pipeline."""

    # Device
    device: str = "cpu"

    # Projection head
    proj_d_in: int = 1280
    proj_d_out: int = 128
    proj_lr: float = 1e-3
    proj_epochs: int = 50
    proj_batch_size: int = 256
    proj_temperature: float = 0.07

    # Step segmenter
    seg_lr: float = 1e-3
    seg_epochs: int = 30
    seg_batch_size: int = 16

    # Scoring head
    score_lr: float = 1e-3
    score_epochs: int = 100
    score_batch_size: int = 64

    # Joint fine-tuning
    joint_lr: float = 1e-4
    joint_epochs: int = 20

    # ASFormer
    asformer_d_model: int = 64
    asformer_n_heads: int = 4
    asformer_n_encoder_layers: int = 10
    asformer_n_decoder_layers: int = 10
    asformer_n_decoders: int = 3
    asformer_lr: float = 1e-3
    asformer_epochs: int = 50

    # Soft-DTW
    gamma_init: float = 1.0

    # DILATE loss
    dilate_alpha: float = 0.5

    # Conformal prediction
    conformal_alpha: float = 0.1

    # Output
    output_dir: Path = field(default_factory=lambda: Path("models/neural"))


@dataclass
class TrainingLog:
    """Collects training metrics across phases."""

    phase: str = ""
    epoch_losses: list[float] = field(default_factory=list)
    final_loss: float = 0.0
    num_parameters: int = 0
    epochs_completed: int = 0


class SOPilotTrainer:
    """Orchestrates multi-phase neural training for SOPilot."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = config.device
        self.logs: list[TrainingLog] = []

        # Models (initialized lazily)
        self.projection_head: ProjectionHead | None = None
        self.segmenter: NeuralStepSegmenter | None = None
        self.scoring_head: ScoringHead | None = None
        self.asformer: ASFormer | None = None
        self.soft_dtw: SoftDTW | None = None
        self.calibrator: IsotonicCalibrator | None = None
        self.conformal: SplitConformalPredictor | None = None
        self.alignment_bridge: AlignmentFeatureBridge | None = None

    # ------------------------------------------------------------------
    # Phase 1: Self-supervised contrastive learning
    # ------------------------------------------------------------------

    def train_projection_head(
        self,
        embeddings_list: list[np.ndarray],
        boundaries_list: list[list[int]],
    ) -> TrainingLog:
        """Phase 1a: Train contrastive projection head.

        Args:
            embeddings_list: List of (N_i, D) embedding matrices, one per gold video.
            boundaries_list: Corresponding step boundaries for each video.

        Returns:
            TrainingLog with loss history.
        """
        log = TrainingLog(phase="projection_head")
        d_in = embeddings_list[0].shape[1]

        self.projection_head = ProjectionHead(
            d_in=d_in, d_out=self.config.proj_d_out
        ).to(self.device)
        log.num_parameters = self.projection_head.num_parameters

        criterion = NTXentLoss(temperature=self.config.proj_temperature).to(self.device)
        optimizer = optim.Adam(self.projection_head.parameters(), lr=self.config.proj_lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.proj_epochs
        )

        # Build dataset: all clips with step labels
        all_embeddings = []
        all_labels = []
        label_offset = 0
        for embs, bounds in zip(embeddings_list, boundaries_list):
            labels = StepPairMiner.assign_step_labels(embs.shape[0], bounds)
            labels += label_offset
            label_offset = int(labels.max().item()) + 1
            all_embeddings.append(torch.from_numpy(embs.astype(np.float32)))
            all_labels.append(labels)

        all_emb = torch.cat(all_embeddings, dim=0)
        all_lab = torch.cat(all_labels, dim=0)
        dataset = TensorDataset(all_emb, all_lab)
        loader = DataLoader(
            dataset, batch_size=self.config.proj_batch_size, shuffle=True, drop_last=True
        )

        self.projection_head.train()
        for epoch in range(self.config.proj_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch_emb, batch_lab in loader:
                batch_emb = batch_emb.to(self.device)
                batch_lab = batch_lab.to(self.device)

                projected = self.projection_head(batch_emb)
                loss = criterion(projected, batch_lab)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(1, n_batches)
            log.epoch_losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "Phase 1a [ProjectionHead] epoch %d/%d loss=%.4f",
                    epoch + 1,
                    self.config.proj_epochs,
                    avg_loss,
                )

        log.final_loss = log.epoch_losses[-1] if log.epoch_losses else 0.0
        log.epochs_completed = self.config.proj_epochs
        self.logs.append(log)

        logger.info(
            "Phase 1a complete: %d params, final_loss=%.4f",
            log.num_parameters,
            log.final_loss,
        )
        return log

    def train_step_segmenter(
        self,
        embeddings_list: list[np.ndarray],
        boundaries_list: list[list[int]],
        alignment_paths: list[list[tuple[int, int, float]]] | None = None,
    ) -> TrainingLog:
        """Phase 1b: Train step segmenter with pseudo-labels or gold boundaries.

        Args:
            embeddings_list: List of (N_i, D) embedding matrices.
            boundaries_list: Step boundaries (gold truth or pseudo-labels).
            alignment_paths: Optional DTW paths for pseudo-label generation.

        Returns:
            TrainingLog.
        """
        log = TrainingLog(phase="step_segmenter")
        d_in = embeddings_list[0].shape[1]

        self.segmenter = NeuralStepSegmenter(d_in=d_in).to(self.device)
        log.num_parameters = self.segmenter.num_parameters

        criterion = SegmentationLoss(smoothing_weight=0.15).to(self.device)
        optimizer = optim.Adam(self.segmenter.parameters(), lr=self.config.seg_lr)

        # Build targets: binary boundary indicators
        X_list = []
        Y_list = []
        for embs, bounds in zip(embeddings_list, boundaries_list):
            n_clips = embs.shape[0]
            if n_clips < 3:
                continue
            target = np.zeros(n_clips, dtype=np.int64)
            for b in bounds[1:-1]:  # Skip first (0) and last (N)
                if 0 < b < n_clips:
                    target[b] = 1
            X_list.append(torch.from_numpy(embs.astype(np.float32)).T)  # (D, T)
            Y_list.append(torch.from_numpy(target))

        if not X_list:
            logger.warning("No valid training data for step segmenter")
            return log

        self.segmenter.train()
        for epoch in range(self.config.seg_epochs):
            epoch_loss = 0.0
            for x, y in zip(X_list, Y_list):
                x = x.unsqueeze(0).to(self.device)  # (1, D, T)
                y = y.unsqueeze(0).to(self.device)  # (1, T)

                logits1, logits2 = self.segmenter(x)
                loss = criterion(logits1, logits2, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(1, len(X_list))
            log.epoch_losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "Phase 1b [StepSegmenter] epoch %d/%d loss=%.4f",
                    epoch + 1,
                    self.config.seg_epochs,
                    avg_loss,
                )

        log.final_loss = log.epoch_losses[-1] if log.epoch_losses else 0.0
        log.epochs_completed = self.config.seg_epochs
        self.logs.append(log)
        return log

    def train_asformer(
        self,
        embeddings_list: list[np.ndarray],
        boundaries_list: list[list[int]],
    ) -> TrainingLog:
        """Phase 1c: Train ASFormer temporal action segmenter.

        Uses transformer-based architecture with exponential dilations,
        multi-head self/cross-attention, and iterative refinement decoders.

        Args:
            embeddings_list: List of (N_i, D) embedding matrices.
            boundaries_list: Step boundaries (gold truth).

        Returns:
            TrainingLog.
        """
        log = TrainingLog(phase="asformer")
        d_in = embeddings_list[0].shape[1]

        self.asformer = ASFormer(
            d_in=d_in,
            d_model=self.config.asformer_d_model,
            n_heads=self.config.asformer_n_heads,
            n_encoder_layers=self.config.asformer_n_encoder_layers,
            n_decoder_layers=self.config.asformer_n_decoder_layers,
            n_decoders=self.config.asformer_n_decoders,
        ).to(self.device)
        log.num_parameters = self.asformer.num_parameters

        criterion = ASFormerLoss(n_classes=2).to(self.device)
        optimizer = optim.Adam(self.asformer.parameters(), lr=self.config.asformer_lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.asformer_epochs
        )

        # Build targets: binary boundary indicators
        X_list = []
        Y_list = []
        for embs, bounds in zip(embeddings_list, boundaries_list):
            n_clips = embs.shape[0]
            if n_clips < 3:
                continue
            target = np.zeros(n_clips, dtype=np.int64)
            for b in bounds[1:-1]:
                if 0 < b < n_clips:
                    target[b] = 1
            X_list.append(torch.from_numpy(embs.astype(np.float32)).T)  # (D, T)
            Y_list.append(torch.from_numpy(target))

        if not X_list:
            logger.warning("No valid training data for ASFormer")
            return log

        self.asformer.train()
        for epoch in range(self.config.asformer_epochs):
            epoch_loss = 0.0
            for x, y in zip(X_list, Y_list):
                x = x.unsqueeze(0).to(self.device)  # (1, D, T)
                y = y.unsqueeze(0).to(self.device)  # (1, T)

                all_logits = self.asformer(x)
                loss = criterion(all_logits, y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.asformer.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step()
            avg_loss = epoch_loss / max(1, len(X_list))
            log.epoch_losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "Phase 1c [ASFormer] epoch %d/%d loss=%.4f",
                    epoch + 1,
                    self.config.asformer_epochs,
                    avg_loss,
                )

        log.final_loss = log.epoch_losses[-1] if log.epoch_losses else 0.0
        log.epochs_completed = self.config.asformer_epochs
        self.logs.append(log)

        logger.info(
            "Phase 1c complete: %d params, final_loss=%.4f",
            log.num_parameters,
            log.final_loss,
        )
        return log

    # ------------------------------------------------------------------
    # Phase 2: Warm-start scoring (match existing penalty formula)
    # ------------------------------------------------------------------

    def train_scoring_head(
        self,
        metrics_array: np.ndarray,
        scores_array: np.ndarray,
    ) -> TrainingLog:
        """Phase 2: Train scoring MLP on existing formula outputs.

        Args:
            metrics_array: (N, 15) matrix of evaluation metrics.
            scores_array: (N,) target scores (from formula or human).

        Returns:
            TrainingLog.
        """
        log = TrainingLog(phase="scoring_head")

        self.scoring_head = ScoringHead().to(self.device)
        log.num_parameters = self.scoring_head.num_parameters

        X = torch.from_numpy(metrics_array.astype(np.float32)).to(self.device)
        Y = torch.from_numpy(scores_array.astype(np.float32)).to(self.device).unsqueeze(-1)

        dataset = TensorDataset(X, Y)
        loader = DataLoader(
            dataset, batch_size=self.config.score_batch_size, shuffle=True
        )

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.scoring_head.parameters(), lr=self.config.score_lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.score_epochs
        )

        self.scoring_head.train()
        for epoch in range(self.config.score_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch_x, batch_y in loader:
                pred = self.scoring_head(batch_x)
                loss = criterion(pred, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(1, n_batches)
            log.epoch_losses.append(avg_loss)

            if (epoch + 1) % 20 == 0:
                logger.info(
                    "Phase 2 [ScoringHead] epoch %d/%d MSE=%.4f",
                    epoch + 1,
                    self.config.score_epochs,
                    avg_loss,
                )

        log.final_loss = log.epoch_losses[-1] if log.epoch_losses else 0.0
        log.epochs_completed = self.config.score_epochs
        self.logs.append(log)
        return log

    # ------------------------------------------------------------------
    # Phase 3: Joint fine-tuning (end-to-end)
    # ------------------------------------------------------------------

    def joint_finetune(
        self,
        gold_embeddings_list: list[np.ndarray],
        trainee_embeddings_list: list[np.ndarray],
        target_scores: np.ndarray,
    ) -> TrainingLog:
        """Phase 3: End-to-end joint fine-tuning through differentiable pipeline.

        Uses AlignmentFeatureBridge to extract meaningful 15-feature vectors
        from the Soft-DTW alignment matrix, replacing the previous zero-padded
        approach that broke end-to-end training.

        Requires: projection_head, scoring_head already initialized.

        Args:
            gold_embeddings_list: List of gold video embeddings.
            trainee_embeddings_list: Corresponding trainee embeddings.
            target_scores: (N,) human-annotated target scores.

        Returns:
            TrainingLog.
        """
        log = TrainingLog(phase="joint_finetune")

        if self.projection_head is None or self.scoring_head is None:
            raise RuntimeError("Must train projection_head and scoring_head first")

        self.soft_dtw = SoftDTW(gamma=self.config.gamma_init).to(self.device)

        # SoftDTWAlignment for extracting alignment matrices.
        # Freeze gamma: the d/d(gamma) of logsumexp(-x/gamma) is numerically
        # unstable when DP cells are INF, producing NaN gradients.  Gamma is
        # a smoothing hyperparameter, not an end-to-end learnable weight.
        soft_dtw_alignment = SoftDTWAlignment(gamma=self.config.gamma_init).to(
            self.device
        )
        soft_dtw_alignment.gamma.requires_grad = False

        # Initialize AlignmentFeatureBridge to convert alignment matrix to 15 features
        self.alignment_bridge = AlignmentFeatureBridge(
            d_embedding=self.config.proj_d_out
        ).to(self.device)

        # FREEZE the ScoringHead — it acts as a differentiable surrogate loss.
        # Gradients flow through it to improve upstream components, but its
        # weights (trained on raw 15 metrics in Phase 2) stay intact.
        # At inference, the scoring head receives raw metrics, not bridge
        # features, so its Phase 2 weights must be preserved.
        for p in self.scoring_head.parameters():
            p.requires_grad = False
        self.scoring_head.eval()

        # Only optimise: projection head + alignment bridge.
        # DTW gamma is frozen (numerical stability), scoring head is frozen
        # (preserves Phase 2 weights for inference on raw metrics).
        params = (
            list(self.projection_head.parameters())
            + list(self.alignment_bridge.parameters())
        )
        optimizer = optim.Adam(params, lr=self.config.joint_lr)
        criterion = nn.MSELoss()

        n = len(gold_embeddings_list)
        log.num_parameters = sum(p.numel() for p in params)

        nan_early_stop = False
        for epoch in range(self.config.joint_epochs):
            epoch_loss = 0.0
            nan_count = 0

            for i in range(n):
                gold = torch.from_numpy(gold_embeddings_list[i].astype(np.float32)).to(
                    self.device
                )
                trainee = torch.from_numpy(
                    trainee_embeddings_list[i].astype(np.float32)
                ).to(self.device)
                target = torch.tensor(
                    [[target_scores[i]]], dtype=torch.float32, device=self.device
                )

                # Forward: project -> alignment matrix -> feature bridge -> score
                gold_proj = self.projection_head(gold)
                trainee_proj = self.projection_head(trainee)

                # Get alignment matrix and distance from SoftDTWAlignment
                alignment_matrix, sdtw_dist = soft_dtw_alignment(
                    gold_proj, trainee_proj
                )

                # Use AlignmentFeatureBridge to extract 15 meaningful features
                score_features = self.alignment_bridge(alignment_matrix, sdtw_dist)

                # ScoringHead is frozen (eval mode), acts as surrogate loss
                predicted_score = self.scoring_head(score_features)

                loss = criterion(predicted_score, target)

                # NaN guard: skip this sample if loss is NaN
                if torch.isnan(loss):
                    nan_count += 1
                    if nan_count > max(1, n // 5):
                        logger.warning(
                            "Phase 3 epoch %d: >20%% NaN losses (%d/%d), stopping early",
                            epoch + 1, nan_count, n,
                        )
                        nan_early_stop = True
                        break
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

            if nan_early_stop:
                break

            valid_count = n - nan_count
            avg_loss = epoch_loss / max(1, valid_count)
            log.epoch_losses.append(avg_loss)

            if (epoch + 1) % 5 == 0:
                logger.info(
                    "Phase 3 [Joint] epoch %d/%d MSE=%.4f gamma=%.4f%s",
                    epoch + 1,
                    self.config.joint_epochs,
                    avg_loss,
                    float(self.soft_dtw.gamma.item()),
                    f" ({nan_count} NaN skipped)" if nan_count else "",
                )

        log.final_loss = log.epoch_losses[-1] if log.epoch_losses else 0.0
        log.epochs_completed = self.config.joint_epochs
        self.logs.append(log)

        # Unfreeze scoring head for subsequent use (save, calibration, etc.)
        for p in self.scoring_head.parameters():
            p.requires_grad = True

        return log

    # ------------------------------------------------------------------
    # Phase 4: Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        predicted_scores: np.ndarray,
        actual_scores: np.ndarray,
    ) -> None:
        """Phase 4: Fit isotonic calibrator on held-out data.

        Args:
            predicted_scores: (N,) model predictions on calibration set.
            actual_scores: (N,) ground-truth scores.
        """
        self.calibrator = IsotonicCalibrator()
        self.calibrator.fit(predicted_scores, actual_scores)

        # Conformal prediction for distribution-free coverage
        self.conformal = SplitConformalPredictor(alpha=self.config.conformal_alpha)
        self.conformal.calibrate(predicted_scores, actual_scores)
        logger.info(
            "Phase 4: Isotonic + conformal calibration (α=%.2f) on %d samples",
            self.config.conformal_alpha,
            len(actual_scores),
        )

    # ------------------------------------------------------------------
    # Save / load full pipeline
    # ------------------------------------------------------------------

    def save_all(self, output_dir: Path | None = None) -> dict[str, str]:
        """Save all trained models to output directory.

        Returns:
            dict mapping component name → saved path.
        """
        out = output_dir or self.config.output_dir
        out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}

        if self.projection_head is not None:
            p = out / "projection_head.pt"
            save_projection_head(self.projection_head, p)
            paths["projection_head"] = str(p)

        if self.segmenter is not None:
            p = out / "step_segmenter.pt"
            save_segmenter(self.segmenter, p)
            paths["step_segmenter"] = str(p)

        if self.asformer is not None:
            p = out / "asformer.pt"
            save_asformer(self.asformer, p)
            paths["asformer"] = str(p)

        if self.scoring_head is not None:
            p = out / "scoring_head.pt"
            save_scoring_head(self.scoring_head, p)
            paths["scoring_head"] = str(p)

        if self.soft_dtw is not None:
            p = out / "soft_dtw.pt"
            torch.save({"gamma": float(self.soft_dtw.gamma.item())}, p)
            paths["soft_dtw"] = str(p)

        if self.alignment_bridge is not None:
            p = out / "alignment_bridge.pt"
            torch.save(self.alignment_bridge.state_dict(), p)
            paths["alignment_bridge"] = str(p)

        if self.calibrator is not None:
            p = out / "isotonic_calibrator.npz"
            self.calibrator.save(p)
            paths["isotonic_calibrator"] = str(p)

        if self.conformal is not None:
            p = out / "conformal_predictor.npz"
            np.savez(p, quantile=np.float64(self.conformal._quantile))
            paths["conformal_predictor"] = str(p)

        logger.info("Saved all models to %s: %s", out, list(paths.keys()))
        return paths

    def training_summary(self) -> dict:
        """Generate summary of all training phases."""
        summary: dict = {"phases": []}
        for log in self.logs:
            summary["phases"].append({
                "phase": log.phase,
                "epochs": log.epochs_completed,
                "final_loss": round(log.final_loss, 6),
                "num_parameters": log.num_parameters,
            })
        total_params = sum(log.num_parameters for log in self.logs)
        summary["total_trainable_parameters"] = total_params
        return summary
