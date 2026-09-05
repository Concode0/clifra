# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Attention score executor assembled from planned pairwise products."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from clifra.core._kernel.basis import reverse_sign
from clifra.core._kernel.contracts import resolve_contract
from clifra.core._kernel.numerics import eps_like
from clifra.core.layout import GradeLayout


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
        self.layout_contract = resolve_contract(algebra, layout=layout, name="layout")
        self.layout = self.layout_contract.layout
        self.score_output_layout = algebra.layout((0, 2))
        self.score_product = algebra.plan_product(
            op="geometric_product",
            left=self.layout,
            right=self.layout,
            output=algebra.layout(self.score_output_layout.grades),
        )._kernel
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
        self.register_buffer(
            "_right_reverse_signs",
            torch.tensor(
                [reverse_sign(index) for index in self.layout.basis_indices],
                dtype=algebra.dtype,
                device=algebra.device,
            ),
            persistent=False,
        )

    def forward(self, q_head: torch.Tensor, k_head: torch.Tensor) -> torch.Tensor:
        """Return attention scores for heads shaped ``[B, H, L, Hc, D]``."""
        self.layout_contract.validate(q_head, name="q_head")
        self.layout_contract.validate(k_head, name="k_head")

        B, H, Lq, Hc, lane_dim = q_head.shape
        Lk = k_head.shape[2]
        q_by_channel = q_head.permute(0, 1, 3, 2, 4).reshape(B, H, Hc, Lq, lane_dim)
        k_by_channel = k_head.permute(0, 1, 3, 2, 4).reshape(B, H, Hc, Lk, lane_dim)
        product = self.score_product.forward_pairwise_compact_right_signed(
            q_by_channel,
            k_by_channel,
            self._right_reverse_signs,
        )

        scalar = torch.index_select(product, -1, self._score_scalar_positions)
        score_g0 = scalar.sum(dim=(2, -1))

        bivectors = torch.index_select(product, -1, self._score_bivector_positions)
        if bivectors.shape[-1] > 0:
            score_g2 = bivectors.pow(2).sum(dim=(2, -1)).clamp_min(eps_like(bivectors)).sqrt()
        else:
            score_g2 = torch.zeros_like(score_g0)

        scale = math.sqrt(self.head_channels * self.layout.dim)
        return (score_g0 + self.bivector_weight * score_g2) / scale
