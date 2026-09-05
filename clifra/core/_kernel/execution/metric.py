# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Metric executors for static Clifford metric plans."""

from __future__ import annotations

import torch
import torch.nn as nn

from clifra.core._kernel.planning.metric import SignatureNormSquaredPlan


class SignatureNormSquaredExecutor(nn.Module):
    """Compile-friendly diagonal executor for signed signature norm squared."""

    route = "diagonal"
    op = "signature_norm_squared"

    def __init__(self, plan: SignatureNormSquaredPlan):
        super().__init__()
        self.spec = plan.spec
        self.input_layout = plan.input_layout
        self.input_contract = plan.input_contract
        self.input_grades = plan.input_grades
        self.input_dim = plan.input_dim
        self.register_buffer("signs", plan.signs, persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return ``<values reverse(values)>_0`` as ``[..., 1]``."""
        self.input_contract.validate(values, name="values")
        return (values * values * self.signs).sum(dim=-1, keepdim=True)


__all__ = ["SignatureNormSquaredExecutor"]
