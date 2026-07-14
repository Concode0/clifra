# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Attention blocks driven by planned Clifford products."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from clifra.core.execution.attention import GeometricAttentionScoreExecutor
from clifra.core.foundation.layout import GradeLayout
from clifra.core.foundation.module import AlgebraLike, CliffordModule
from clifra.core.foundation.numerics import eps_like
from clifra.core.runtime.tensors import resolve_contract

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

    Standard attention uses a scalar query-key score.

    This layer forms a geometric product per head-channel and combines its
    grade-0 component with the coefficient norm of its grade-2 component.

    The grade-0 (scalar) part measures alignment (like dot product).
    The grade-2 contribution records oriented-plane content in the product.

    Attributes:
        num_heads (int): Number of attention heads.
        head_channels (int): Channels per head.
        causal (bool): If True, apply autoregressive causal mask.
        bivector_weight (float): Weight of the bivector score component.
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
            bivector_weight: Weight on the bivector score component.
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
        self.layout_contract = resolve_contract(algebra, layout=layout, grades=grades)
        self.layout = self.layout_contract.layout
        self.lane_dim = self.layout_contract.lane_dim

        self.qkv_weight = nn.Parameter(torch.empty(3, channels, channels))
        self.qkv_bias = nn.Parameter(torch.empty(3, channels, self.lane_dim))
        self.out_proj = CliffordLinear(algebra, channels, channels, layout=self.layout)
        self.reset_parameters()

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        self.scorer = GeometricAttentionScoreExecutor(
            algebra,
            head_channels=self.head_channels,
            bivector_weight=bivector_weight,
            layout=self.layout,
        )

    def reset_parameters(self) -> None:
        """Initialize the fused QKV channel projections."""
        for weight in self.qkv_weight:
            nn.init.xavier_uniform_(weight)
        nn.init.zeros_(self.qkv_bias)

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

        qkv = torch.einsum("poi,...id->...pod", self.qkv_weight, x)
        bias_shape = (1,) * (x.ndim - 2) + (3, C, D)
        qkv = qkv + self.qkv_bias.view(bias_shape)
        Q = qkv.select(-3, 0)
        K = qkv.select(-3, 1)
        V = qkv.select(-3, 2)

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

        return self.out_proj(output)


class EntropyGatedAttention(CliffordModule):
    """Geometric attention with an example bivector-entropy gate.

    This is intentionally implemented as a layout-first block: inputs and
    outputs use the compact lanes declared by ``layout`` or ``grades``.
    """

    def __init__(
        self,
        algebra: AlgebraLike,
        channels: int,
        num_heads: int,
        eta: float = 1.0,
        H_base: float = 0.5,
        *,
        grades=None,
        layout: GradeLayout = None,
    ):
        """Initializes entropy-gated attention."""
        super().__init__(algebra)
        self.channels = channels
        self.eta = eta
        self.H_base = H_base
        self.layout_contract = resolve_contract(algebra, layout=layout, grades=grades)
        self.layout = self.layout_contract.layout
        self.base_attention = GeometricProductAttention(
            algebra,
            channels,
            num_heads,
            causal=False,
            layout=self.layout,
        )

        g2_idx = self.layout_contract.grade_positions(2, device=algebra.device)
        g2_mask = torch.zeros(self.layout_contract.lane_dim, device=algebra.device, dtype=torch.float32)
        if g2_idx.numel() > 0:
            g2_mask.index_fill_(0, g2_idx, 1.0)
        self.register_buffer("g2_idx", g2_idx)
        self.register_buffer("_g2_float_mask", g2_mask)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor = None, return_gating: bool = False
    ) -> torch.Tensor:
        """Applies entropy-gated geometric attention to ``[B, L, C, D]`` inputs."""
        self.layout_contract.validate_input(
            x,
            channels=self.channels,
            name="EntropyGatedAttention input",
        )
        if self.g2_idx.numel() > 0:
            g2_values = torch.index_select(x, -1, self.g2_idx)
            g2_energy = g2_values.square().sum(dim=(-1, -2))
        else:
            g2_energy = x.new_zeros(x.shape[0], x.shape[1])

        if key_padding_mask is not None:
            g2_energy = g2_energy.masked_fill(key_padding_mask, 0.0)

        total_energy = g2_energy.sum(dim=1, keepdim=True)
        eps = eps_like(g2_energy, min_value=torch.finfo(g2_energy.dtype).tiny)
        p = torch.where(total_energy > 0, g2_energy / total_energy.clamp_min(eps), torch.zeros_like(g2_energy))
        entropy = -(p * torch.log(p.clamp_min(eps))).sum(dim=1)

        gate = self.eta * torch.sigmoid(entropy - self.H_base)
        gate_view = gate.view(-1, 1, 1, 1)
        scale = 1.0 + (gate_view - 1.0) * self._g2_float_mask
        output = self.base_attention(x * scale, key_padding_mask=key_padding_mask)

        if return_gating:
            return output, entropy, gate
        return output
