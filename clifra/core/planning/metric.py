# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Static metric plans for diagonal Clifford forms."""

from __future__ import annotations

import torch

from clifra.core.foundation.basis import operation_coefficient, reverse_sign
from clifra.core.foundation.layout import AlgebraSpec, GradeLayout


class NormSquaredPlan:
    """Static diagonal plan for ``<x reverse(x)>_0`` over one layout."""

    def __init__(
        self,
        *,
        spec: AlgebraSpec,
        input_layout: GradeLayout,
        signs: torch.Tensor,
    ):
        self.spec = spec
        self.input_layout = input_layout
        self.signs = signs

    @property
    def input_grades(self) -> tuple[int, ...]:
        """Return input grades represented by the plan."""
        return self.input_layout.grades

    @property
    def input_dim(self) -> int:
        """Return the compact input lane count."""
        return self.input_layout.dim


def build_norm_squared_plan(
    spec: AlgebraSpec,
    *,
    input_layout: GradeLayout,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> NormSquaredPlan:
    """Build a static diagonal signed norm plan for ``input_layout``."""
    if input_layout.spec != spec:
        raise ValueError(f"input_layout signature {input_layout.spec} does not match algebra signature {spec}")
    signs = [
        reverse_sign(index) * operation_coefficient(index, index, spec.p, spec.q, spec.r, "gp")
        for index in input_layout.basis_indices
    ]
    return NormSquaredPlan(
        spec=spec,
        input_layout=input_layout,
        signs=torch.tensor(signs, dtype=dtype, device=device),
    )
