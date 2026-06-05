# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Permutation executors for static Clifford lane-map plans."""

from __future__ import annotations

import torch
import torch.nn as nn

from clifra.core.planning.permutation import DualPlan


class DualExecutor(nn.Module):
    """Compile-friendly dual/pseudoscalar permutation executor."""

    executor_family = "unary_permutation"
    op = "dual"

    def __init__(self, plan: DualPlan):
        super().__init__()
        self.spec = plan.spec
        self.input_layout = plan.input_layout
        self.output_layout = plan.output_layout
        self.input_grades = plan.input_grades
        self.output_grades = plan.output_grades
        self.input_dim = plan.input_layout.dim
        self.output_dim = plan.output_layout.dim
        self.register_buffer("input_positions", plan.input_positions, persistent=False)
        self.register_buffer("signs", plan.signs, persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return dual values in ``output_layout`` lanes."""
        if values.shape[-1] != self.input_dim:
            raise ValueError(f"values last dimension must be {self.input_dim}, got {values.shape[-1]}")
        gathered = torch.index_select(values, -1, self.input_positions)
        return gathered * self.signs


__all__ = ["DualExecutor"]
