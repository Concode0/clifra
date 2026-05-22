# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from clifra.core.execution.attention import GeometricAttentionScoreExecutor
from clifra.core.foundation.layout import GradeLayout
from clifra.core.foundation.module import AlgebraLike, CliffordModule
from clifra.core.storage import resolve_layer_layout_contract

from ..primitives.linear import CliffordLinear


def _merge_attention_mask(left: torch.Tensor | None, right: torch.Tensor | None) -> torch.Tensor | None:
    if left is None:
        return right
    if right is None:
        return left
    return left | right


def _score_bias(scores: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Return additive SDPA score bias with masked keys removed."""
    if mask is None:
        return scores
    return scores.masked_fill(mask, float("-inf"))


def _sdpa_from_scores(scores: torch.Tensor, values: torch.Tensor, *, dropout_p: float) -> torch.Tensor:
    """Aggregate ``values`` with precomputed GA scores through Torch SDPA.

    SDPA cannot compute the geometric algebra score itself. Zero query/key
    tensors make the dot-product contribution vanish, while ``scores`` enters
    as the float attention bias. This keeps the softmax/dropout/value matmul on
    Torch's attention primitive without changing the GA scoring contract.
    """
    B, H, Lq, Lk = scores.shape
    value_shape = values.shape
    flat_values = values.reshape(B, H, Lk, -1)
    query = scores.new_zeros(B, H, Lq, 1)
    key = scores.new_zeros(B, H, Lk, 1)
    output = F.scaled_dot_product_attention(
        query,
        key,
        flat_values,
        attn_mask=scores,
        dropout_p=dropout_p,
        scale=1.0,
    )
    return output.reshape(*value_shape[:2], Lq, *value_shape[3:])


class GeometricProductAttention(CliffordModule):
    """Multi-head attention using geometric product scoring.

    Standard attention: score(Q, K) = <Q, K> / sqrt(d)  (scalar only)

    GA attention:
        product = Q_c * reverse(K_c)    (geometric product per head-channel)
        score   = (<product>_0 + lambda_ * ||<product>_2||_F) / sqrt(H_c * dim)

    The grade-0 (scalar) part measures alignment (like dot product).
    The grade-2 (bivector) part measures relative orientation - novel.

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
            grades: Optional compact input/output grades.
            layout: Optional compact input/output layout.
        """
        super().__init__(algebra)
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"

        self.channels = channels
        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        self.causal = causal
        self.bivector_weight = bivector_weight
        self.layout_contract = resolve_layer_layout_contract(algebra, layout=layout, grades=grades)
        self.layout = self.layout_contract.layout
        self.lane_dim = self.layout_contract.lane_dim

        # Q, K, V projections operate on [B*L, channels, dim]
        self.q_proj = CliffordLinear(algebra, channels, channels, layout=self.layout)
        self.k_proj = CliffordLinear(algebra, channels, channels, layout=self.layout)
        self.v_proj = CliffordLinear(algebra, channels, channels, layout=self.layout)
        self.out_proj = CliffordLinear(algebra, channels, channels, layout=self.layout)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        self.scorer = GeometricAttentionScoreExecutor(
            algebra,
            head_channels=self.head_channels,
            bivector_weight=bivector_weight,
            layout=self.layout,
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

        # Build score mask once [B, H, L, L], where True means masked out.
        score_mask = None
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
            )  # True = masked (future)
            score_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        if key_padding_mask is not None:
            # key_padding_mask: [B, L] -> [B, 1, 1, L]
            padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            score_mask = _merge_attention_mask(score_mask, padding_mask)

        scores = _score_bias(self._compute_score(Q, K), score_mask)
        dropout_p = self.attn_dropout.p if self.attn_dropout is not None and self.training else 0.0
        output = _sdpa_from_scores(scores, V, dropout_p=dropout_p)

        # Merge heads back: [B, L, C, D]
        output = output.permute(0, 2, 1, 3, 4).reshape(B, L, C, D)

        # Output projection
        output = self.out_proj(output.reshape(B * L, C, D)).reshape(B, L, C, D)

        return output
