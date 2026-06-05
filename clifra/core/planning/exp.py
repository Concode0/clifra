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
    operator_layout: GradeLayout
    output_layout: GradeLayout
    executor_family: str
    bivector_squared_signs: torch.Tensor
    vector_seed: torch.Tensor
    scalar_output_position: int
    bivector_input_positions: torch.Tensor
    bivector_output_positions: torch.Tensor
    bivector_operator_positions: torch.Tensor
    output_from_operator_positions: torch.Tensor
    operator_to_output_positions: torch.Tensor
    scalar_output_index: torch.Tensor
    operator_scalar_index: torch.Tensor
    operator_eye: torch.Tensor
    operator_scalar_position: int
    component_count: int
    fixed_iterations: int
    decomposition_tolerance: float
    eps: float
    eps_sq: float
    regime: str


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
    operator_layout = spec.layout(range(0, spec.n + 1, 2))
    finfo = torch.finfo(dtype)
    input_positions: list[int] = []
    output_positions: list[int] = []
    output_position_by_index = {index: position for position, index in enumerate(output_layout.basis_indices)}
    for input_position, index in enumerate(input_layout.basis_indices):
        output_position = output_position_by_index.get(index)
        if output_position is None:
            continue
        input_positions.append(input_position)
        output_positions.append(output_position)

    operator_position_by_index = {index: position for position, index in enumerate(operator_layout.basis_indices)}
    bivector_operator_positions = [operator_position_by_index[index] for index in input_layout.basis_indices]
    operator_positions: list[int] = []
    target_positions: list[int] = []
    for output_position, index in enumerate(output_layout.basis_indices):
        operator_position = operator_position_by_index.get(index)
        if operator_position is None:
            continue
        operator_positions.append(operator_position)
        target_positions.append(output_position)

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

    if spec.p == 0 or spec.q == 0:
        regime = "elliptic"
    elif spec.p == 1 and spec.q == 1 and spec.r == 0:
        regime = "hyperbolic"
    else:
        regime = "mixed"

    return BivectorExpPlan(
        spec=spec,
        input_layout=input_layout,
        vector_layout=vector_layout,
        operator_layout=operator_layout,
        output_layout=output_layout,
        executor_family=select_bivector_exp_executor_family(spec, resolved_device),
        bivector_squared_signs=torch.tensor(signs, dtype=dtype, device=resolved_device),
        vector_seed=torch.full((vector_layout.dim,), 1.0 / (spec.n**0.5), dtype=dtype, device=resolved_device),
        scalar_output_position=output_position_by_index.get(0, -1),
        bivector_input_positions=torch.tensor(input_positions, dtype=torch.long, device=resolved_device),
        bivector_output_positions=torch.tensor(output_positions, dtype=torch.long, device=resolved_device),
        bivector_operator_positions=torch.tensor(bivector_operator_positions, dtype=torch.long, device=resolved_device),
        output_from_operator_positions=torch.tensor(operator_positions, dtype=torch.long, device=resolved_device),
        operator_to_output_positions=torch.tensor(target_positions, dtype=torch.long, device=resolved_device),
        scalar_output_index=torch.tensor([output_position_by_index.get(0, -1)], dtype=torch.long, device=resolved_device),
        operator_scalar_index=torch.tensor([operator_position_by_index[0]], dtype=torch.long, device=resolved_device),
        operator_eye=torch.eye(operator_layout.dim, dtype=dtype, device=resolved_device),
        operator_scalar_position=operator_position_by_index[0],
        component_count=max(spec.n // 2, 1),
        fixed_iterations=int(fixed_iterations),
        decomposition_tolerance=1e-6,
        eps=float(finfo.eps),
        eps_sq=float(finfo.eps**2),
        regime=regime,
    )


def select_bivector_exp_executor_family(spec: AlgebraSpec, device) -> str:
    """Return the planner-selected bivector-exp executor family."""
    if spec.n <= 3:
        return "closed_simple"
    if torch.device(device).type == "mps":
        return "decomposed"
    return "left_matrix_exp"
