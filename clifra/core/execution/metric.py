# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Metric executors for static Clifford metric plans."""

from __future__ import annotations

import torch
import torch.nn as nn

from clifra.core.planning.metric import NormSquaredPlan


class NormSquaredExecutor(nn.Module):
    """Compile-friendly diagonal executor for algebraic squared norm."""

    executor_family = "metric_diagonal"
    op = "norm_sq"

    def __init__(self, plan: NormSquaredPlan):
        super().__init__()
        self.spec = plan.spec
        self.input_layout = plan.input_layout
        self.input_grades = plan.input_grades
        self.input_dim = plan.input_dim
        self.register_buffer("signs", plan.signs, persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return ``<values reverse(values)>_0`` as ``[..., 1]``."""
        if values.shape[-1] != self.input_dim:
            raise ValueError(f"values last dimension must be {self.input_dim}, got {values.shape[-1]}")
        return (values * values * self.signs).sum(dim=-1, keepdim=True)


__all__ = ["NormSquaredExecutor"]
