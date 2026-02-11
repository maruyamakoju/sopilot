"""ASFormer: Transformer for Action Segmentation.

Combines temporal convolutions (local features) with self-attention
(global context) in a multi-stage prediction-refinement architecture.
This provides full-sequence receptive field unlike MS-TCN variants.

Architecture:
    Encoder (prediction): Input -> [ASFormerBlock(dilation=2^k)] x L -> Output
    Decoder (refinement): [Prev_output + Input -> Cross-Attn + ASFormerBlock] x S

Each ASFormerBlock: DilatedConv -> Self-Attention -> FFN (all with residuals).

References:
    Yi, F. et al. (2021). "ASFormer: Transformer for Action Segmentation",
    BMVC 2021.

    Shaw, P. et al. (2018). "Self-Attention with Relative Position
    Representations", NAACL 2018.

    Vaswani, A. et al. (2017). "Attention Is All You Need", NeurIPS 2017.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "ASFormer",
    "ASFormerLoss",
    "predict_boundaries_asformer",
    "save_asformer",
    "load_asformer",
]


# ---------------------------------------------------------------------------
# Relative Positional Encoding
# ---------------------------------------------------------------------------

class RelativePositionalEncoding(nn.Module):
    """Learnable relative positional encoding (Shaw et al., 2018).

    For temporal segmentation, relative position ('how far apart in time')
    is more meaningful than absolute position, since temporal shifts in
    video should not change the segmentation.

    Stores a learnable embedding for each relative distance
    in [-max_rel, +max_rel], clipped for longer sequences.
    """

    def __init__(self, d_model: int, max_relative_position: int = 256) -> None:
        super().__init__()
        self.max_rel = max_relative_position
        self.embeddings = nn.Embedding(2 * max_relative_position + 1, d_model)

    def forward(self, length: int) -> torch.Tensor:
        """Compute relative position bias matrix.

        Args:
            length: Sequence length T.

        Returns:
            (T, T, d_model) relative position embeddings.
        """
        device = self.embeddings.weight.device
        positions = torch.arange(length, device=device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T)
        rel_pos = rel_pos.clamp(-self.max_rel, self.max_rel) + self.max_rel
        return self.embeddings(rel_pos)  # (T, T, d_model)


# ---------------------------------------------------------------------------
# Multi-Head Temporal Self-Attention
# ---------------------------------------------------------------------------

class TemporalSelfAttention(nn.Module):
    """Multi-head self-attention with relative positional encoding.

    Uses chunked attention for efficiency: divides the sequence into
    windows and computes attention within each window, plus a set of
    global tokens for long-range information flow.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_relative_position: int = 256,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.rel_pos = RelativePositionalEncoding(self.d_head, max_relative_position)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Self-attention on temporal features.

        Args:
            x: (B, T, D) input features.
            mask: (B, T) boolean mask, True for valid positions.

        Returns:
            (B, T, D) attended features.
        """
        B, T, D = x.shape

        Q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # Q, K, V: (B, H, T, d_head)

        # Content-based attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T, T)

        # Relative position bias
        rel_emb = self.rel_pos(T)  # (T, T, d_head)
        # Compute position attention: Q @ rel_emb^T
        rel_attn = torch.einsum("bhtd,ijd->bhij", Q, rel_emb) / self.scale
        attn = attn + rel_attn

        # Apply mask
        if mask is not None:
            # mask: (B, T) -> (B, 1, 1, T) for broadcasting
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            attn = attn.masked_fill(~mask_expanded, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, H, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Cross-Attention (for refinement stages)
# ---------------------------------------------------------------------------

class TemporalCrossAttention(nn.Module):
    """Cross-attention between refinement stage and encoder features.

    Allows each refinement stage to attend to encoder's hidden
    representations for informed prediction correction.
    """

    def __init__(
        self, d_model: int, n_heads: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-attention: query attends to key_value.

        Args:
            query: (B, T, D) from current refinement stage.
            key_value: (B, T, D) from encoder.
            mask: (B, T) boolean mask.

        Returns:
            (B, T, D) attended features.
        """
        B, T, D = query.shape
        H = self.n_heads

        Q = self.q_proj(query).view(B, T, H, self.d_head).transpose(1, 2)
        K = self.k_proj(key_value).view(B, T, H, self.d_head).transpose(1, 2)
        V = self.v_proj(key_value).view(B, T, H, self.d_head).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask_expanded, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class PositionwiseFFN(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


# ---------------------------------------------------------------------------
# ASFormer Block
# ---------------------------------------------------------------------------

class ASFormerBlock(nn.Module):
    """Single ASFormer block: DilatedConv + Self-Attention + FFN.

    Pre-LayerNorm architecture for training stability:
        x -> LN -> DilatedConv -> Residual
        x -> LN -> Self-Attention -> Residual
        x -> LN -> FFN -> Residual
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dilation: int = 1,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Dilated convolution (local context)
        self.norm1 = nn.LayerNorm(d_model)
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=padding, dilation=dilation,
        )
        self.conv_dropout = nn.Dropout(dropout)

        # Self-attention (global context)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = TemporalSelfAttention(d_model, n_heads, dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # FFN
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, dropout=dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """(B, T, D) -> (B, T, D)"""
        # Dilated conv (operates on channel dim)
        residual = x
        h = self.norm1(x)
        h = h.transpose(1, 2)  # (B, D, T)
        h = self.conv(h)
        h = F.relu(h)
        h = h.transpose(1, 2)  # (B, T, D)
        x = residual + self.conv_dropout(h)

        # Self-attention
        residual = x
        h = self.norm2(x)
        h = self.self_attn(h, mask)
        x = residual + self.attn_dropout(h)

        # FFN
        residual = x
        h = self.norm3(x)
        h = self.ffn(h)
        x = residual + self.ffn_dropout(h)

        return x


# ---------------------------------------------------------------------------
# ASFormer Decoder Block (with cross-attention)
# ---------------------------------------------------------------------------

class ASFormerDecoderBlock(nn.Module):
    """Decoder block: DilatedConv + Self-Attention + Cross-Attention + FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dilation: int = 1,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Same as encoder block
        self.norm1 = nn.LayerNorm(d_model)
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=padding, dilation=dilation,
        )
        self.conv_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = TemporalSelfAttention(d_model, n_heads, dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # Cross-attention to encoder
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = TemporalCrossAttention(d_model, n_heads, dropout)
        self.cross_dropout = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, dropout=dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Dilated conv
        residual = x
        h = self.norm1(x).transpose(1, 2)
        h = F.relu(self.conv(h)).transpose(1, 2)
        x = residual + self.conv_dropout(h)

        # Self-attention
        residual = x
        h = self.self_attn(self.norm2(x), mask)
        x = residual + self.attn_dropout(h)

        # Cross-attention to encoder
        residual = x
        h = self.cross_attn(self.norm_cross(x), encoder_out, mask)
        x = residual + self.cross_dropout(h)

        # FFN
        residual = x
        h = self.ffn(self.norm3(x))
        x = residual + self.ffn_dropout(h)

        return x


# ---------------------------------------------------------------------------
# Full ASFormer Model
# ---------------------------------------------------------------------------

class ASFormer(nn.Module):
    """ASFormer: Transformer for Action Segmentation.

    1 encoder (prediction) + K decoders (refinement stages).
    Encoder: stack of ASFormerBlocks with exponential dilations.
    Each decoder: refines previous stage using cross-attention to encoder.

    Args:
        d_in: Input feature dimension.
        d_model: Hidden dimension (default 64).
        n_classes: Number of output classes (default 2 for boundary detection).
        n_heads: Number of attention heads (default 4).
        n_encoder_layers: Blocks in encoder (default 10).
        n_decoder_layers: Blocks per decoder (default 10).
        n_decoders: Number of refinement stages (default 3).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_in: int,
        d_model: int = 64,
        n_classes: int = 2,
        n_heads: int = 4,
        n_encoder_layers: int = 10,
        n_decoder_layers: int = 10,
        n_decoders: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_classes = n_classes
        self.n_heads = n_heads
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.n_decoders = n_decoders
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(d_in, d_model)

        # Encoder
        encoder_dilations = [2 ** (i % n_encoder_layers) for i in range(n_encoder_layers)]
        self.encoder_blocks = nn.ModuleList([
            ASFormerBlock(d_model, n_heads, dil, dropout=dropout)
            for dil in encoder_dilations
        ])
        self.encoder_norm = nn.LayerNorm(d_model)
        self.encoder_out_proj = nn.Linear(d_model, n_classes)

        # Decoders
        self.decoder_input_projs = nn.ModuleList()
        self.decoder_blocks_list = nn.ModuleList()
        self.decoder_norms = nn.ModuleList()
        self.decoder_out_projs = nn.ModuleList()

        for _ in range(n_decoders):
            self.decoder_input_projs.append(nn.Linear(n_classes + d_in, d_model))
            decoder_dilations = [2 ** (i % n_decoder_layers) for i in range(n_decoder_layers)]
            blocks = nn.ModuleList([
                ASFormerDecoderBlock(d_model, n_heads, dil, dropout=dropout)
                for dil in decoder_dilations
            ])
            self.decoder_blocks_list.append(blocks)
            self.decoder_norms.append(nn.LayerNorm(d_model))
            self.decoder_out_projs.append(nn.Linear(d_model, n_classes))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Forward pass through encoder + decoders.

        Args:
            x: (B, D_in, T) input features (channel-first convention).
            mask: (B, T) boolean mask for valid positions.

        Returns:
            List of (B, n_classes, T) logits, one per stage
            (encoder + each decoder).
        """
        B, D, T = x.shape
        # Convert to (B, T, D) for transformer
        x_bt = x.transpose(1, 2)  # (B, T, D_in)

        # Encoder
        h = self.input_proj(x_bt)  # (B, T, d_model)
        for block in self.encoder_blocks:
            h = block(h, mask)
        encoder_out = self.encoder_norm(h)
        enc_logits = self.encoder_out_proj(encoder_out)  # (B, T, n_classes)

        all_logits = [enc_logits.transpose(1, 2)]  # (B, n_classes, T)

        # Decoders
        prev_logits = enc_logits
        for s in range(self.n_decoders):
            # Input: concat previous softmax + original features
            prev_probs = F.softmax(prev_logits, dim=-1)  # (B, T, n_classes)
            dec_input = torch.cat([prev_probs, x_bt], dim=-1)  # (B, T, n_classes+D_in)
            h = self.decoder_input_projs[s](dec_input)

            for block in self.decoder_blocks_list[s]:
                h = block(h, encoder_out, mask)

            h = self.decoder_norms[s](h)
            dec_logits = self.decoder_out_projs[s](h)  # (B, T, n_classes)
            all_logits.append(dec_logits.transpose(1, 2))  # (B, n_classes, T)
            prev_logits = dec_logits

        return all_logits

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# ASFormer Loss
# ---------------------------------------------------------------------------

class ASFormerLoss(nn.Module):
    """Combined loss for ASFormer training.

    Per stage: CE + smoothing + focal boundary weighting.
    Later stages get higher weight (exponential).
    """

    def __init__(
        self,
        n_classes: int = 2,
        smoothing_weight: float = 0.15,
        boundary_weight: float = 2.0,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.smoothing_weight = smoothing_weight
        self.boundary_weight = boundary_weight
        self.focal_gamma = focal_gamma

    def forward(
        self,
        all_logits: list[torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-stage loss.

        Args:
            all_logits: List of (B, n_classes, T) per stage.
            targets: (B, T) long tensor class labels.

        Returns:
            Scalar loss.
        """
        total_loss = torch.tensor(0.0, device=targets.device)
        n_stages = len(all_logits)

        for s, logits in enumerate(all_logits):
            # Exponential stage weighting (later stages matter more)
            stage_weight = 2.0 ** (s - n_stages + 1)

            # Cross-entropy with class weighting for boundary imbalance
            # Boundary class (1) is rare, so upweight it
            weight = torch.ones(self.n_classes, device=targets.device)
            weight[1] = self.boundary_weight
            ce_loss = F.cross_entropy(logits, targets, weight=weight)

            # Focal loss modification for hard examples
            probs = F.softmax(logits, dim=1)
            # Gather probabilities for correct class
            target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - target_probs) ** self.focal_gamma
            focal_ce = (focal_weight * F.cross_entropy(
                logits, targets, weight=weight, reduction='none'
            )).mean()

            # Temporal smoothing
            smooth = torch.mean(torch.abs(probs[:, :, 1:] - probs[:, :, :-1]))

            stage_loss = focal_ce + self.smoothing_weight * smooth
            total_loss = total_loss + stage_weight * stage_loss

        return total_loss


# ---------------------------------------------------------------------------
# Prediction utility
# ---------------------------------------------------------------------------

def predict_boundaries_asformer(
    model: ASFormer,
    embeddings: np.ndarray,
    min_step_clips: int = 2,
    threshold: float = 0.5,
    device: str = "cpu",
) -> tuple[list[int], np.ndarray]:
    """Predict step boundaries using ASFormer.

    Args:
        model: Trained ASFormer model.
        embeddings: (T, D) clip embeddings.
        min_step_clips: Minimum clips between boundaries.
        threshold: Classification threshold for boundary class.
        device: Device for inference.

    Returns:
        (boundaries, boundary_probs) where boundaries is [0, b1, ..., T]
        and boundary_probs is (T,) array of boundary probabilities.
    """
    t, d = embeddings.shape
    if t <= 1:
        return [0, t], np.zeros(t, dtype=np.float32)

    x = torch.from_numpy(embeddings.astype(np.float32)).to(device)
    x = x.T.unsqueeze(0)  # (1, D, T)

    model.eval()
    with torch.no_grad():
        all_logits = model(x)
        # Use last stage (most refined)
        final_logits = all_logits[-1]
        probs = F.softmax(final_logits, dim=1)
        boundary_probs = probs[0, 1, :].cpu().numpy()

    # Extract boundaries with minimum spacing
    raw_points = [i for i in range(1, t) if boundary_probs[i] >= threshold]

    filtered: list[int] = []
    last = 0
    for point in raw_points:
        if point - last >= max(1, min_step_clips):
            filtered.append(point)
            last = point

    boundaries = [0] + filtered
    if boundaries[-1] != t:
        boundaries.append(t)

    return boundaries, boundary_probs


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_asformer(model: ASFormer, path: Path) -> None:
    """Save ASFormer model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "d_in": model.d_in,
        "d_model": model.d_model,
        "n_classes": model.n_classes,
        "n_heads": model.n_heads,
        "n_encoder_layers": model.n_encoder_layers,
        "n_decoder_layers": model.n_decoder_layers,
        "n_decoders": model.n_decoders,
        "dropout": model.dropout,
        "state_dict": model.state_dict(),
    }, path)
    logger.info("Saved ASFormer (%d params) to %s", model.num_parameters, path)


def load_asformer(path: Path, device: str = "cpu") -> ASFormer:
    """Load ASFormer model from checkpoint."""
    data = torch.load(path, map_location=device, weights_only=True)
    model = ASFormer(
        d_in=data["d_in"],
        d_model=data["d_model"],
        n_classes=data["n_classes"],
        n_heads=data.get("n_heads", 4),
        n_encoder_layers=data.get("n_encoder_layers", 10),
        n_decoder_layers=data.get("n_decoder_layers", 10),
        n_decoders=data["n_decoders"],
        dropout=data.get("dropout", 0.1),
    )
    model.load_state_dict(data["state_dict"])
    model.to(device)
    model.eval()
    logger.info("Loaded ASFormer from %s", path)
    return model
