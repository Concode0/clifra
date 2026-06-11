# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Static bivector exponential plans and executor-family selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from clifra.core.foundation.layout import AlgebraSpec, GradeLayout


@dataclass(frozen=True)
class BivectorExpPlan:
    """Static layout contract for a bivector exponential executor."""

    spec: AlgebraSpec
    input_layout: GradeLayout
    vector_layout: GradeLayout
    grade4_layout: GradeLayout | None
    operator_layout: GradeLayout
    output_layout: GradeLayout
    executor_family: str
    bivector_squared_signs: torch.Tensor
    vector_seed: torch.Tensor
    output_scalar_mask: torch.Tensor
    operator_scalar_mask: torch.Tensor
    bivector_to_output: torch.Tensor
    bivector_to_operator: torch.Tensor
    grade4_to_output: torch.Tensor
    operator_to_output: torch.Tensor
    operator_eye: torch.Tensor
    operator_scalar_position: int
    component_count: int
    fixed_iterations: int
    decomposition_tolerance: float
    eps: float
    eps_sq: float


def build_bivector_exp_plan(
    spec: AlgebraSpec,
    *,
    input_layout: GradeLayout,
    output_layout: GradeLayout,
    dtype: torch.dtype,
    device,
    fixed_iterations: int = 20,
) -> BivectorExpPlan:
    """Build a static plan for ``exp(B)`` where ``B`` is grade-2."""
    if input_layout.spec != spec:
        raise ValueError(f"input_layout signature {input_layout.spec} does not match algebra signature {spec}")
    if output_layout.spec != spec:
        raise ValueError(f"output_layout signature {output_layout.spec} does not match algebra signature {spec}")
    if input_layout.grades != (2,):
        raise ValueError(f"bivector exp requires grade-2 input layout, got {input_layout.grades}")

    resolved_device = torch.device(device)
    vector_layout = spec.layout((1,))
    grade4_layout = spec.layout((4,)) if spec.n >= 4 else None
    operator_layout = spec.layout(range(0, spec.n + 1, 2))
    finfo = torch.finfo(dtype)
    operator_position_by_index = {index: position for position, index in enumerate(operator_layout.basis_indices)}

    signs = []
    for index in input_layout.basis_indices:
        bits = [bit for bit in range(spec.n) if index & (1 << bit)]
        if len(bits) != 2:
            signs.append(0.0)
            continue
        a, b = bits
        s_a = 1.0 if a < spec.p else (-1.0 if a < spec.p + spec.q else 0.0)
        s_b = 1.0 if b < spec.p else (-1.0 if b < spec.p + spec.q else 0.0)
        signs.append(-s_a * s_b)

    return BivectorExpPlan(
        spec=spec,
        input_layout=input_layout,
        vector_layout=vector_layout,
        grade4_layout=grade4_layout,
        operator_layout=operator_layout,
        output_layout=output_layout,
        executor_family=select_bivector_exp_executor_family(spec, resolved_device),
        bivector_squared_signs=torch.tensor(signs, dtype=dtype, device=resolved_device),
        vector_seed=torch.full((vector_layout.dim,), 1.0 / (spec.n**0.5), dtype=dtype, device=resolved_device),
        output_scalar_mask=_scalar_mask(output_layout, dtype=dtype, device=resolved_device),
        operator_scalar_mask=_scalar_mask(operator_layout, dtype=dtype, device=resolved_device),
        bivector_to_output=_layout_map(input_layout, output_layout, dtype=dtype, device=resolved_device),
        bivector_to_operator=_layout_map(input_layout, operator_layout, dtype=dtype, device=resolved_device),
        grade4_to_output=_layout_map(grade4_layout, output_layout, dtype=dtype, device=resolved_device),
        operator_to_output=_layout_map(operator_layout, output_layout, dtype=dtype, device=resolved_device),
        operator_eye=torch.eye(operator_layout.dim, dtype=dtype, device=resolved_device),
        operator_scalar_position=operator_position_by_index[0],
        component_count=max(spec.n // 2, 1),
        fixed_iterations=int(fixed_iterations),
        decomposition_tolerance=1e-6,
        eps=float(finfo.eps),
        eps_sq=float(finfo.eps**2),
    )


def _scalar_mask(layout: GradeLayout, *, dtype: torch.dtype, device) -> torch.Tensor:
    mask = torch.zeros(layout.dim, dtype=dtype, device=device)
    scalar_position = {index: position for position, index in enumerate(layout.basis_indices)}.get(0)
    if scalar_position is not None:
        mask[scalar_position] = 1.0
    return mask


def _layout_map(source: GradeLayout | None, target: GradeLayout, *, dtype: torch.dtype, device) -> torch.Tensor:
    if source is None:
        return torch.zeros((0, target.dim), dtype=dtype, device=device)

    matrix = torch.zeros((source.dim, target.dim), dtype=dtype, device=device)
    target_positions = {index: position for position, index in enumerate(target.basis_indices)}
    for source_position, index in enumerate(source.basis_indices):
        target_position = target_positions.get(index)
        if target_position is not None:
            matrix[source_position, target_position] = 1.0
    return matrix


def select_bivector_exp_executor_family(spec: AlgebraSpec, device) -> str:
    """Return the planner-selected bivector-exp executor family."""
    if spec.n <= 3:
        return "closed_simple"
    if spec.n <= 5:
        return "closed_biquadratic"
    if torch.device(device).type == "mps":
        return "decomposed"
    return "left_matrix_exp"
