# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Permutation executors for static Clifford lane-map plans."""

from __future__ import annotations

import torch
import torch.nn as nn

from clifra.core._kernel.basis import operation_coefficient
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
        # Tiny CPU gathers are faster; larger complement maps benefit from flip.
        mask = self.spec.dim - 1
        source_indices = tuple(index ^ mask for index in self.output_layout.basis_indices)
        self._reverse_order = source_indices == self.input_layout.basis_indices[::-1] and self.input_dim >= 64
        signs = {
            operation_coefficient(index, mask, self.spec.p, self.spec.q, self.spec.r, "geometric_product")
            for index in source_indices
        }
        self._uniform_sign = 1 if signs <= {1} else (-1 if signs == {-1} else 0)
        self.register_buffer("input_positions", plan.input_positions, persistent=False)
        self.register_buffer("signs", plan.signs, persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return right-pseudoscalar product values in ``output_layout`` lanes."""
        self.input_contract.validate(values, name="values")
        gathered = values.flip((-1,)) if self._reverse_order else torch.index_select(values, -1, self.input_positions)
        if gathered.dtype != self.signs.dtype or gathered.device != self.signs.device:
            return gathered * self.signs
        if self._uniform_sign == 1:
            return gathered
        if self._uniform_sign == -1:
            return -gathered
        return gathered * self.signs


__all__ = ["PseudoscalarProductExecutor"]
