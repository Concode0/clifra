# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Static permutation plans for Clifford lane maps."""

from __future__ import annotations

import torch

from clifra.core.foundation.basis import operation_coefficient
from clifra.core.foundation.layout import AlgebraSpec, GradeLayout


class DualPlan:
    """Static gather/sign plan for right multiplication by pseudoscalar."""

    def __init__(
        self,
        *,
        spec: AlgebraSpec,
        input_layout: GradeLayout,
        output_layout: GradeLayout,
        input_positions: torch.Tensor,
        signs: torch.Tensor,
    ):
        self.spec = spec
        self.input_layout = input_layout
        self.output_layout = output_layout
        self.input_positions = input_positions
        self.signs = signs

    @property
    def input_grades(self) -> tuple[int, ...]:
        """Return input grades."""
        return self.input_layout.grades

    @property
    def output_grades(self) -> tuple[int, ...]:
        """Return output grades."""
        return self.output_layout.grades


def build_dual_plan(
    spec: AlgebraSpec,
    *,
    input_layout: GradeLayout,
    output_layout: GradeLayout | None = None,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> DualPlan:
    """Build a static right-pseudoscalar multiplication plan."""
    if input_layout.spec != spec:
        raise ValueError(f"input_layout signature {input_layout.spec} does not match algebra signature {spec}")
    if output_layout is None:
        output_layout = spec.layout(tuple(spec.n - grade for grade in input_layout.grades))
    if output_layout.spec != spec:
        raise ValueError(f"output_layout signature {output_layout.spec} does not match algebra signature {spec}")

    pseudoscalar_index = spec.dim - 1
    input_position_by_index = {index: position for position, index in enumerate(input_layout.basis_indices)}
    input_positions = []
    signs = []
    for output_index in output_layout.basis_indices:
        source_index = output_index ^ pseudoscalar_index
        source_position = input_position_by_index.get(source_index)
        if source_position is None:
            raise ValueError(
                f"output basis index {output_index} requires source basis index {source_index}, "
                f"which is not in input grades {input_layout.grades}"
            )
        input_positions.append(source_position)
        signs.append(operation_coefficient(source_index, pseudoscalar_index, spec.p, spec.q, spec.r, "gp"))

    return DualPlan(
        spec=spec,
        input_layout=input_layout,
        output_layout=output_layout,
        input_positions=torch.tensor(input_positions, dtype=torch.long, device=device),
        signs=torch.tensor(signs, dtype=dtype, device=device),
    )
