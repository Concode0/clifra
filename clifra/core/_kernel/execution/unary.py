# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Unary executors for static Clifford unary plans."""

from __future__ import annotations

import torch
import torch.nn as nn

from clifra.core._kernel.basis import unary_sign
from clifra.core._kernel.planning.unary import GradeUnaryPlan
from clifra.core.tensors import TensorContract


class GradeUnaryExecutor(nn.Module):
    """Torch module for planned unary gather/sign execution."""

    def __init__(self, plan: GradeUnaryPlan):
        super().__init__()
        self.spec = plan.spec
        self.op = plan.op
        self.input_layout = plan.input_layout
        self.output_layout = plan.output_layout
        self.input_contract = plan.input_contract
        self.output_contract = plan.output_contract
        self.canonical_input_contract = TensorContract.canonical(self.input_layout)
        self.dim = plan.dim
        self._same_layout = self.input_layout.basis_indices == self.output_layout.basis_indices
        self._canonical_output = self.output_layout.dim == self.dim
        signs = {unary_sign(self.op, index) for index in self.output_layout.basis_indices}
        self._uniform_sign = 1 if signs <= {1} else (-1 if signs == {-1} else 0)
        self.register_buffer("input_positions", plan.input_positions, persistent=False)
        self.register_buffer("output_indices", plan.output_indices, persistent=False)
        self.register_buffer("signs", plan.signs, persistent=False)

    @property
    def output_dim(self) -> int:
        """Return the compact output lane count."""
        return self.output_layout.dim

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return compact output lanes for full-layout input coefficients."""
        self.canonical_input_contract.validate(values, name="values")
        output = values if self._canonical_output else torch.index_select(values, -1, self.output_indices)
        return self._apply_sign(output)

    def forward_compact(self, values: torch.Tensor) -> torch.Tensor:
        """Return compact output lanes for compact input coefficients."""
        self.input_contract.validate(values, name="values")
        output = values if self._same_layout else torch.index_select(values, -1, self.input_positions)
        return self._apply_sign(output)

    def _apply_sign(self, values):
        if values.dtype != self.signs.dtype or values.device != self.signs.device:
            return values * self.signs
        if self._uniform_sign == 1:
            return values
        if self._uniform_sign == -1:
            return -values
        return values * self.signs


__all__ = ["GradeUnaryExecutor"]
