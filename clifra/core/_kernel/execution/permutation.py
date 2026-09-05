# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Permutation executors for static Clifford lane-map plans."""

from __future__ import annotations

import torch
import torch.nn as nn

from clifra.core._kernel.planning.permutation import PseudoscalarProductPlan


class PseudoscalarProductExecutor(nn.Module):
    """Compile-friendly right-pseudoscalar product permutation executor."""

    route = "pseudoscalar"
    op = "pseudoscalar_product"

    def __init__(self, plan: PseudoscalarProductPlan):
        super().__init__()
        self.spec = plan.spec
        self.input_layout = plan.input_layout
        self.output_layout = plan.output_layout
        self.input_contract = plan.input_contract
        self.output_contract = plan.output_contract
        self.input_grades = plan.input_grades
        self.output_grades = plan.output_grades
        self.input_dim = plan.input_layout.dim
        self.output_dim = plan.output_layout.dim
        self.register_buffer("input_positions", plan.input_positions, persistent=False)
        self.register_buffer("signs", plan.signs, persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return right-pseudoscalar product values in ``output_layout`` lanes."""
        self.input_contract.validate(values, name="values")
        gathered = torch.index_select(values, -1, self.input_positions)
        return gathered * self.signs


__all__ = ["PseudoscalarProductExecutor"]
