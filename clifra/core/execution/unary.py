# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Unary executors for static Clifford unary plans."""

from __future__ import annotations

import torch
import torch.nn as nn

from clifra.core.planning.unary import GradeUnaryPlan


class GradeUnaryExecutor(nn.Module):
    """Torch module for planned unary gather/sign execution."""

    def __init__(self, plan: GradeUnaryPlan):
        super().__init__()
        self.spec = plan.spec
        self.op = plan.op
        self.input_layout = plan.input_layout
        self.output_layout = plan.output_layout
        self.dim = plan.dim
        self.register_buffer("input_positions", plan.input_positions, persistent=False)
        self.register_buffer("output_indices", plan.output_indices, persistent=False)
        self.register_buffer("signs", plan.signs, persistent=False)

    @property
    def output_dim(self) -> int:
        """Return the compact output lane count."""
        return self.output_layout.dim

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return compact output lanes for full-layout input coefficients."""
        if values.shape[-1] != self.dim:
            raise ValueError(f"full-layout last dimension must be {self.dim}, got {values.shape[-1]}")
        output = torch.index_select(values, -1, self.output_indices)
        return output * self.signs

    def forward_compact(self, values: torch.Tensor) -> torch.Tensor:
        """Return compact output lanes for compact input coefficients."""
        if values.shape[-1] != self.input_layout.dim:
            raise ValueError(f"compact last dimension must be {self.input_layout.dim}, got {values.shape[-1]}")
        output = torch.index_select(values, -1, self.input_positions)
        return output * self.signs

    def forward_full(self, values: torch.Tensor) -> torch.Tensor:
        """Return full-layout output coefficients for full-layout input coefficients."""
        compact = self.forward(values)
        return self.output_layout.full(compact)


__all__ = ["GradeUnaryExecutor"]
