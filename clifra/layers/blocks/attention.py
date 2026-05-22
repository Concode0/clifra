# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from clifra.core.foundation.layout import GradeLayout
from clifra.core.foundation.module import AlgebraLike, CliffordModule
from clifra.core.foundation.numerics import eps_like
from clifra.core.runtime.attention import GeometricAttentionScorer
from clifra.core.storage import resolve_layer_layout_contract

from ..primitives.linear import CliffordLinear
from ..primitives.product import GeometricProductLayer

# Memory-bounded block size for chunked attention computation
_BLOCK_SIZE = 64
_G2_BLADE_CHUNK_SIZE = 16
_SCORE_PRECOMPUTE_LIMIT = 8_000_000


def _merge_attention_mask(left: torch.Tensor | None, right: torch.Tensor | None) -> torch.Tensor | None:
    if left is None:
        return right
    if right is None:
        return left
    return left | right


def _safe_masked_softmax(scores: torch.Tensor, mask: torch.Tensor | None, dim: int = -1) -> torch.Tensor:
    """Softmax with finite zero output for fully masked rows."""
    if mask is None:
        return F.softmax(scores, dim=dim)

    masked_scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    weights = F.softmax(masked_scores, dim=dim).masked_fill(mask, 0.0)
    normalizer = weights.sum(dim=dim, keepdim=True)
    eps = eps_like(weights, min_value=torch.finfo(weights.dtype).tiny)
    weights = weights / normalizer.clamp_min(eps)
    return torch.where(normalizer > 0, weights, torch.zeros_like(weights))


class GeometricProductAttention(CliffordModule):
    """Multi-head attention using geometric product scoring.

    Standard attention: score(Q, K) = <Q, K> / sqrt(d)  (scalar only)

    GA attention:
        product = Q_c * reverse(K_c)    (geometric product per head-channel)
        score   = (<product>_0 + lambda_ * ||<product>_2||_F) / sqrt(H_c * dim)

    The grade-0 (scalar) part measures alignment (like dot product).
    The grade-2 (bivector) part measures relative orientation - novel.

    Memory: naive [B, H, L, L, H_c, D] is too large. We chunk over L_q
    in blocks of BLOCK_SIZE to bound peak VRAM.

    Attributes:
        num_heads (int): Number of attention heads.
        head_channels (int): Channels per head.
        causal (bool): If True, apply autoregressive causal mask.
        bivector_weight (float): lambda_ - weight of bivector score component.
    """

    def __init__(
        self,
        algebra: AlgebraLike,
        channels: int,
        num_heads: int,
        causal: bool = True,
        bivector_weight: float = 0.5,
        dropout: float = 0.0,
        score_blade_chunk_size: int = _G2_BLADE_CHUNK_SIZE,
        score_precompute_limit: int = _SCORE_PRECOMPUTE_LIMIT,
        grades=None,
        layout: GradeLayout = None,
    ):
        """Sets up geometric product attention.

        Args:
            algebra: Clifford algebra instance.
            channels: Total number of multivector channels.
            num_heads: Number of attention heads.
            causal: Apply causal mask for autoregressive generation.
            bivector_weight: lambda_ weight on bivector score component.
            dropout: Dropout rate on attention weights.
            score_blade_chunk_size: Grade-2 output blades processed per dense
                chunk when exact dense scoring is used.
            score_precompute_limit: Maximum temporary ``K_g2`` elements allowed
                before exact dense scoring switches to chunked grade-2 blades.
            grades: Optional compact input/output grades. Compact attention is
                currently grade-1 only and uses planned pairwise products.
            layout: Optional compact input/output layout.
        """
        super().__init__(algebra)
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"

        self.channels = channels
        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        self.causal = causal
        self.bivector_weight = bivector_weight
        self.score_blade_chunk_size = max(1, int(score_blade_chunk_size))
        self.score_precompute_limit = max(0, int(score_precompute_limit))
        self.layout_contract = resolve_layer_layout_contract(algebra, layout=layout, grades=grades)
        self.layout = self.layout_contract.layout
        self.lane_dim = self.layout_contract.lane_dim

        # Q, K, V projections operate on [B*L, channels, dim]
        self.q_proj = CliffordLinear(algebra, channels, channels, layout=self.layout)
        self.k_proj = CliffordLinear(algebra, channels, channels, layout=self.layout)
        self.v_proj = CliffordLinear(algebra, channels, channels, layout=self.layout)
        self.out_proj = CliffordLinear(algebra, channels, channels, layout=self.layout)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        score_product = None
        if self.layout is not None and self.layout.dim != self.algebra.dim:
            score_product = GeometricProductLayer(
                algebra,
                left_layout=self.layout,
                right_layout=self.layout,
                output_layout=algebra.layout((0, 2)),
                pairwise=True,
            )
        self.scorer = GeometricAttentionScorer(
            algebra,
            head_channels=self.head_channels,
            bivector_weight=bivector_weight,
            layout=self.layout,
            pairwise_product=score_product,
            score_blade_chunk_size=self.score_blade_chunk_size,
            score_precompute_limit=self.score_precompute_limit,
        )

    def _compute_score(
        self,
        q_head: torch.Tensor,
        k_head: torch.Tensor,
    ) -> torch.Tensor:
        """Compute GA attention scores for one query block."""
        return self.scorer(q_head, k_head)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Computes geometric product attention.

        Args:
            x: Input multivectors [B, L, C, D].
            key_padding_mask: Optional [B, L] bool mask where True = padded (ignored).

        Returns:
            Output multivectors [B, L, C, D].
        """
        self.layout_contract.validate_input(
            x,
            channels=self.channels,
            name="GeometricProductAttention input",
            allow_full=self.layout is None or self.layout.dim == self.algebra.dim,
        )
        B, L, C, D = x.shape

        # Project Q, K, V (CliffordLinear expects [B, C, D])
        x_flat = x.reshape(B * L, C, D)
        Q = self.q_proj(x_flat).reshape(B, L, C, D)
        K = self.k_proj(x_flat).reshape(B, L, C, D)
        V = self.v_proj(x_flat).reshape(B, L, C, D)

        H = self.num_heads
        Hc = self.head_channels

        # Reshape to [B, H, L, Hc, D]
        Q = Q.reshape(B, L, H, Hc, D).permute(0, 2, 1, 3, 4)  # [B, H, L, Hc, D]
        K = K.reshape(B, L, H, Hc, D).permute(0, 2, 1, 3, 4)
        V = V.reshape(B, L, H, Hc, D).permute(0, 2, 1, 3, 4)

        # Build causal mask once [L, L]
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
            )  # True = masked (future)
        else:
            causal_mask = None

        # Chunked attention over query positions to bound memory
        output_chunks = []
        for q_start in range(0, L, _BLOCK_SIZE):
            q_end = min(q_start + _BLOCK_SIZE, L)

            Q_block = Q[:, :, q_start:q_end]  # [B, H, Lq, Hc, D]

            # Compute scores: [B, H, Lq, L]
            scores = self._compute_score(Q_block, K)

            score_mask = None
            if causal_mask is not None:
                mask_block = causal_mask[q_start:q_end, :]  # [Lq, L]
                score_mask = mask_block.unsqueeze(0).unsqueeze(0)

            if key_padding_mask is not None:
                # key_padding_mask: [B, L] -> [B, 1, 1, L]
                padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
                score_mask = _merge_attention_mask(score_mask, padding_mask)

            # Softmax + dropout
            attn_weights = _safe_masked_softmax(scores, score_mask, dim=-1)  # [B, H, Lq, L]
            if self.attn_dropout is not None:
                attn_weights = self.attn_dropout(attn_weights)

            # Aggregate values: sum_k attn[b,h,i,k] * V[b,h,k,Hc,D]
            # attn_weights: [B, H, Lq, L]
            # V:            [B, H, L,  Hc, D]
            # out:          [B, H, Lq, Hc, D]
            out_block = torch.einsum("bhij,bhjcd->bhicd", attn_weights, V)
            output_chunks.append(out_block)

        # Reassemble: [B, H, L, Hc, D]
        output = torch.cat(output_chunks, dim=2)

        # Merge heads back: [B, L, C, D]
        output = output.permute(0, 2, 1, 3, 4).reshape(B, L, C, D)

        # Output projection
        output = self.out_proj(output.reshape(B * L, C, D)).reshape(B, L, C, D)

        return output
