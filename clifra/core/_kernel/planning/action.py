# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Intent and layout plans for linear and versor-style actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

import torch

from clifra.core._kernel.basis import operation_coefficient
from clifra.core._kernel.contracts import _check_contract_spec
from clifra.core._kernel.planning.policy import ActionFacts
from clifra.core._kernel.planning.resources import ResourceRequirements
from clifra.core.layout import AlgebraSpec, GradeLayout
from clifra.core.tensors import TensorContract


def _contract(spec, layout, role: str) -> TensorContract:
    return _check_contract_spec(spec, TensorContract.compact(layout), f"{role}_layout")


def _bind_contracts(plan, *roles: str) -> None:
    spec = plan.input_layout.spec
    for role in roles:
        object.__setattr__(plan, f"{role}_contract", _contract(spec, getattr(plan, f"{role}_layout"), role))


@dataclass(frozen=True)
class LinearActionPlan:
    """Resolved contract for a vector-space action lifted to multivector grades."""

    input_layout: GradeLayout
    output_layout: GradeLayout
    input_contract: TensorContract = field(init=False, repr=False)
    output_contract: TensorContract = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _bind_contracts(self, "input", "output")

    @property
    def input_grades(self) -> tuple[int, ...]:
        """Return the grades accepted by the action input layout."""
        return self.input_layout.grades

    @property
    def output_grades(self) -> tuple[int, ...]:
        """Return the grades emitted by the action output layout."""
        return self.output_layout.grades


@dataclass(frozen=True)
class VersorActionPlan:
    """Resolved contract for grade-1 or grade-2 versor actions."""

    grade: int
    input_layout: GradeLayout
    output_layout: GradeLayout
    parameter_layout: GradeLayout
    route: str
    input_contract: TensorContract = field(init=False, repr=False)
    output_contract: TensorContract = field(init=False, repr=False)
    parameter_contract: TensorContract = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _bind_contracts(self, "input", "output", "parameter")


def build_linear_action_plan(
    *,
    input_layout: GradeLayout,
    output_layout: GradeLayout | None = None,
) -> LinearActionPlan:
    """Build a plan-only linear action contract."""
    spec = input_layout.spec
    output_layout = input_layout if output_layout is None else output_layout
    _contract(spec, output_layout, "output")
    return LinearActionPlan(input_layout=input_layout, output_layout=output_layout)


def build_versor_action_plan(
    algebra,
    *,
    grade: int,
    input_layout: GradeLayout,
    output_layout: GradeLayout | None = None,
    parameter_layout: GradeLayout | None = None,
) -> VersorActionPlan:
    """Build a plan-only versor action contract."""
    spec = AlgebraSpec.from_algebra(algebra)
    grade = int(grade)
    if grade not in {1, 2}:
        raise ValueError("planned versor actions currently support grade=1 and grade=2")
    output_layout = input_layout if output_layout is None else output_layout
    parameter_layout = algebra.layout((grade,)) if parameter_layout is None else parameter_layout
    input_layout = _contract(spec, input_layout, "input").layout
    output_layout = _contract(spec, output_layout, "output").layout
    parameter_layout = _contract(spec, parameter_layout, "parameter").layout
    if parameter_layout.grades != (grade,):
        raise ValueError(f"parameter_layout must contain grade {grade}, got {parameter_layout.grades}")
    decision = _select_versor_action_route(
        algebra,
        grade=grade,
        input_layout=input_layout,
        output_layout=output_layout,
        parameter_layout=parameter_layout,
    )
    return VersorActionPlan(
        grade=grade,
        input_layout=input_layout,
        output_layout=output_layout,
        parameter_layout=parameter_layout,
        route=decision.route,
    )


def _select_versor_action_route(algebra, **parameters):
    return _select_action_route(algebra, "versor", parameters)


def _select_action_route(algebra, operation, parameters):
    from clifra.core._kernel.providers import action_execution_request

    request = action_execution_request(algebra, operation, **parameters)
    return algebra._planner.router.select(request, algebra._planner.policy, algebra._planner.limits)


def _basis_bits_tuple(index: int, n: int) -> tuple[int, ...]:
    return tuple(bit for bit in range(n) if index & (1 << bit))


def _scalar_action_positions(input_layout: GradeLayout, output_layout: GradeLayout) -> torch.Tensor:
    positions: list[int] = []
    for output_position, output_index in enumerate(output_layout.basis_indices):
        if output_index != 0:
            continue
        for input_position, input_index in enumerate(input_layout.basis_indices):
            if input_index == 0:
                positions.append(output_position * input_layout.dim + input_position)
    return torch.tensor(positions, dtype=torch.long)


def _graded_action_plan_tensors(
    input_layout: GradeLayout,
    output_layout: GradeLayout,
    *,
    grade: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_positions: list[int] = []
    row_indices: list[tuple[int, ...]] = []
    col_indices: list[tuple[int, ...]] = []
    input_items = [
        (input_position, _basis_bits_tuple(input_index, input_layout.spec.n))
        for input_position, input_index in enumerate(input_layout.basis_indices)
        if input_index.bit_count() == grade
    ]
    for output_position, output_index in enumerate(output_layout.basis_indices):
        if output_index.bit_count() != grade:
            continue
        output_bits = _basis_bits_tuple(output_index, input_layout.spec.n)
        for input_position, input_bits in input_items:
            flat_positions.append(output_position * input_layout.dim + input_position)
            row_indices.append(output_bits)
            col_indices.append(input_bits)

    if not flat_positions:
        empty = torch.empty(0, dtype=torch.long)
        return empty, torch.empty(0, grade, dtype=torch.long), torch.empty(0, grade, dtype=torch.long)
    return (
        torch.tensor(flat_positions, dtype=torch.long),
        torch.tensor(row_indices, dtype=torch.long),
        torch.tensor(col_indices, dtype=torch.long),
    )


def build_bivector_vector_generator_buffers(bivector_layout, *, dtype, device):
    lane_positions: list[int] = []
    flat_positions: list[int] = []
    coefficients: list[float] = []
    vector_layout = bivector_layout.spec.layout((1,))
    vector_positions = {index: position for position, index in enumerate(vector_layout.basis_indices)}
    for bivector_position, bivector_index in enumerate(bivector_layout.basis_indices):
        for input_position, input_index in enumerate(vector_layout.basis_indices):
            output_index = bivector_index ^ input_index
            output_position = vector_positions.get(output_index)
            if output_position is None:
                continue
            coefficient = -0.5 * operation_coefficient(
                bivector_index,
                input_index,
                bivector_layout.spec.p,
                bivector_layout.spec.q,
                bivector_layout.spec.r,
                "commutator_product",
            )
            if coefficient == 0.0:
                continue
            lane_positions.append(bivector_position)
            flat_positions.append(output_position * bivector_layout.spec.n + input_position)
            coefficients.append(coefficient)
    return (
        torch.tensor(lane_positions, dtype=torch.long, device=device),
        torch.tensor(flat_positions, dtype=torch.long, device=device),
        torch.tensor(coefficients, dtype=dtype, device=device),
    )


def build_full_sandwich_action_buffers(layout, *, device=None, dtype=torch.float32):
    dim = layout.spec.dim
    indices = torch.arange(dim, dtype=torch.long, device=device)
    cayley_indices = indices.unsqueeze(0) ^ indices.unsqueeze(1)
    sign_rows: list[list[float]] = []
    for left_index in range(dim):
        row = []
        for output_index in range(dim):
            right_index = left_index ^ output_index
            row.append(
                operation_coefficient(
                    left_index, right_index, layout.spec.p, layout.spec.q, layout.spec.r, "geometric_product"
                )
            )
        sign_rows.append(row)
    geometric_product_signs = torch.tensor(sign_rows, dtype=dtype, device=device)
    output_indices = torch.arange(dim, dtype=torch.long, device=device).unsqueeze(0).expand(dim, dim)
    left_sign_t = geometric_product_signs[cayley_indices, output_indices].T.contiguous()
    return cayley_indices, left_sign_t, geometric_product_signs.T.contiguous()


def _bivector_generator_term_count(bivector_layout) -> int:
    if bivector_layout.grades != (2,):
        raise ValueError("vector-generator facts require a grade-2 parameter layout")
    spec = bivector_layout.spec
    # Each non-null basis direction contributes in both bivectors containing
    # it and the corresponding generator row; null overlaps vanish.
    return (spec.p + spec.q) * max(spec.n - 1, 0)


def _linear_action_structure(input_layout, output_layout, *, generator_layout=None):
    """Describe lifted coefficients and bound determinant allocations.

    These are per broadcast item, like other static resource estimates. Count
    grade blocks without constructing their Cartesian-product index buffers.
    """
    n = input_layout.spec.n
    grades = set(input_layout.grades) & set(output_layout.grades)
    vector_only = input_layout.grades == output_layout.grades == (1,)
    blocks = {} if vector_only else {g: comb(n, g) ** 2 for g in grades if g > 0}
    dense = input_layout.dim * output_layout.dim
    minors = max((count * g * g for g, count in blocks.items()), default=0)
    indices = sum(count * (1 + 2 * g) for g, count in blocks.items())
    scalar = int(0 in input_layout.grades and 0 in output_layout.grades)
    lifted = scalar + sum(blocks.values())
    minor_entries = sum(count * g * g for g, count in blocks.items() if g >= 4)
    determinant_work = sum(count * (2 if g == 2 else 9 if g == 3 else g**3) for g, count in blocks.items() if g >= 2)
    generator_terms = 0 if generator_layout is None else _bivector_generator_term_count(generator_layout)
    pairs = max(n * n, dense, minors)
    lanes = max(n, input_layout.dim, output_layout.dim, 0 if generator_layout is None else generator_layout.dim)
    return (
        ActionFacts(
            generator_terms=generator_terms,
            lifted_coefficients=lifted,
            minor_entries=minor_entries,
            determinant_work=determinant_work,
        ),
        ResourceRequirements(lanes, max(pairs, indices + n * n)),
    )
