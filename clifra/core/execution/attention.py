"""Attention score executor assembled from planned pairwise products."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from clifra.core.foundation.layout import GradeLayout


class GeometricAttentionScoreExecutor(nn.Module):
    """Compute geometric attention scores from a declared pairwise product plan."""

    def __init__(
        self,
        algebra,
        *,
        head_channels: int,
        bivector_weight: float,
        layout: GradeLayout,
    ):
        super().__init__()
        self.algebra = algebra
        self.head_channels = int(head_channels)
        self.bivector_weight = float(bivector_weight)
        self.layout = layout
        self.score_output_layout = algebra.layout((0, 2))
        self.reverse_executor = algebra.planner.unary_executor(
            op="reverse",
            input_grades=layout.grades,
            output_grades=layout.grades,
            dtype=algebra.dtype,
            device=algebra.device,
            cache=True,
        )
        self.score_product = algebra.product_executor(
            left_grades=layout.grades,
            right_grades=layout.grades,
            output_grades=self.score_output_layout.grades,
            op="gp",
            dtype=algebra.dtype,
            device=algebra.device,
            cache=True,
        )
        self.register_buffer(
            "_score_scalar_positions",
            self.score_output_layout.positions_for_grades((0,), device=algebra.device),
            persistent=False,
        )
        self.register_buffer(
            "_score_bivector_positions",
            self.score_output_layout.positions_for_grades((2,), device=algebra.device),
            persistent=False,
        )

    def forward(self, q_head: torch.Tensor, k_head: torch.Tensor) -> torch.Tensor:
        """Return attention scores for heads shaped ``[B, H, L, Hc, D]``."""
        if q_head.shape[-1] != self.layout.dim:
            raise ValueError(f"q_head last dim must be {self.layout.dim}, got {q_head.shape[-1]}")
        if k_head.shape[-1] != self.layout.dim:
            raise ValueError(f"k_head last dim must be {self.layout.dim}, got {k_head.shape[-1]}")

        B, H, Lq, Hc, lane_dim = q_head.shape
        Lk = k_head.shape[2]
        q_by_channel = q_head.permute(0, 1, 3, 2, 4).reshape(B, H, Hc, Lq, lane_dim)
        k_by_channel = k_head.permute(0, 1, 3, 2, 4).reshape(B, H, Hc, Lk, lane_dim)
        k_by_channel = self.reverse_executor.forward_compact(k_by_channel)
        product = self.score_product.forward_pairwise_compact(q_by_channel, k_by_channel)

        scalar = torch.index_select(product, -1, self._score_scalar_positions.to(device=product.device))
        score_g0 = scalar.sum(dim=(2, -1))

        bivectors = torch.index_select(product, -1, self._score_bivector_positions.to(device=product.device))
        if bivectors.shape[-1] > 0:
            score_g2 = bivectors.pow(2).sum(dim=(2, -1)).clamp_min(0.0).sqrt()
        else:
            score_g2 = torch.zeros_like(score_g0)

        scale = math.sqrt(self.head_channels * self.layout.dim)
        return (score_g0 + self.bivector_weight * score_g2) / scale
