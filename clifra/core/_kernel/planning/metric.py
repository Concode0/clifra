# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Static metric plans for diagonal Clifford forms."""

from __future__ import annotations

import torch

from clifra.core._kernel.basis import operation_coefficient, reverse_sign
from clifra.core._kernel.contracts import _check_contract_spec
from clifra.core.layout import AlgebraSpec, GradeLayout
from clifra.core.tensors import TensorContract


class SignatureNormSquaredPlan:
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
        self.input_contract = TensorContract.compact(input_layout)
        self.signs = signs

    @property
    def input_grades(self) -> tuple[int, ...]:
        """Return input grades represented by the plan."""
        return self.input_layout.grades

    @property
    def input_dim(self) -> int:
        """Return the compact input lane count."""
        return self.input_layout.dim


def build_signature_norm_squared_plan(
    spec: AlgebraSpec,
    *,
    input_layout: GradeLayout,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> SignatureNormSquaredPlan:
    """Build a static diagonal signed norm plan for ``input_layout``."""
    input_contract = TensorContract.compact(input_layout)
    input_layout = _check_contract_spec(spec, input_contract, "input_layout").layout
    signs = [
        reverse_sign(index) * operation_coefficient(index, index, spec.p, spec.q, spec.r, "geometric_product")
        for index in input_layout.basis_indices
    ]
    return SignatureNormSquaredPlan(
        spec=spec,
        input_layout=input_layout,
        signs=torch.tensor(signs, dtype=dtype, device=device),
    )


__all__ = [
    "SignatureNormSquaredPlan",
    "build_signature_norm_squared_plan",
]
