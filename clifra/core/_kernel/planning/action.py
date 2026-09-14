# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Intent and layout plans for linear and versor-style actions."""

from __future__ import annotations

from math import comb

import torch

from clifra.core._kernel.basis import operation_coefficient
from clifra.core._kernel.planning.resources import ResourceRequirements
from clifra.core.layout import GradeLayout


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


def _direct_grade_counts(n: int, grade: int) -> tuple[int, int, int]:
    """Transition work, largest state, and largest gathered transition."""
    work = sum(comb(n, step + 1) * comb(n, grade - step - 1) * (n - grade + step + 1) for step in range(grade))
    state = max(comb(n, step) * comb(n, grade - step) for step in range(grade + 1))
    terms = max(comb(n, step + 1) * comb(n, grade - step - 1) * (n - grade + step + 1) for step in range(grade))
    return work, state, terms


def _direct_action_grade(input_layout: GradeLayout, output_layout: GradeLayout, grade: int, device) -> bool:
    """Choose a prepared matvec only when it reduces both work and peak terms."""
    n = input_layout.spec.n
    if grade <= 1:
        return False
    width = comb(n, grade)
    compound_coefficients = width * width
    # A small full lift amortizes one dense matrix application particularly well.
    if (
        input_layout.grades == output_layout.grades == tuple(range(n + 1))
        and input_layout.dim * output_layout.dim <= 4096
    ):
        return False
    # MPS favors the existing few-launch compound path while its block is no
    # wider than a vector-space n²-by-n² coefficient footprint.
    if torch.device(device).type == "mps" and compound_coefficients <= n**4:
        return False
    direct_work, _, direct_terms = _direct_grade_counts(n, grade)
    compound_work = compound_coefficients * (2 if grade == 2 else 9 if grade == 3 else grade**3)
    compound_minor_entries = compound_coefficients * grade * grade
    return direct_work < compound_work and direct_terms < compound_minor_entries


def _direct_action_plan_tensors(spec, grade: int, *, dtype, device):
    """Prepare row-contraction transitions for one exterior-power matvec."""
    n = spec.n

    def blades(size):
        return tuple(_basis_bits_tuple(mask, n) for mask in spec.layout((size,)).basis_indices)

    stages = []
    for step in range(grade):
        prior_outputs = {item: pos for pos, item in enumerate(blades(step))}
        prior_remaining = {item: pos for pos, item in enumerate(blades(grade - step))}
        outputs = blades(step + 1)
        remaining = blades(grade - step - 1)
        previous, rows, cols, signs = [], [], [], []
        for output in outputs:
            row = output[-1]
            prefix = prior_outputs[output[:-1]]
            for tail in remaining:
                for col in range(n):
                    if col in tail:
                        continue
                    joined = tuple(sorted((*tail, col)))
                    previous.append(prefix * len(prior_remaining) + prior_remaining[joined])
                    rows.append(row)
                    cols.append(col)
                    signs.append((-1) ** joined.index(col))
        stages.append(
            (
                torch.tensor(previous, dtype=torch.long, device=device),
                torch.tensor(rows, dtype=torch.long, device=device),
                torch.tensor(cols, dtype=torch.long, device=device),
                torch.tensor(signs, dtype=dtype, device=device),
                len(outputs),
                len(remaining),
                n - (grade - step - 1),
            )
        )
    return tuple(stages)


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


def build_versor_vector_buffers(parameter_layout, *, grade, dtype, device):
    """Prepare route-local vector-action buffers after route selection."""
    if grade == 2:
        generator = build_bivector_vector_generator_buffers(parameter_layout, dtype=dtype, device=device)
        empty = torch.empty(0, dtype=dtype, device=device)
        return generator, empty, empty
    if grade != 1 or parameter_layout.grades != (1,):
        raise ValueError("vector action parameters must have the selected grade layout")
    signs = [
        operation_coefficient(
            index,
            index,
            parameter_layout.spec.p,
            parameter_layout.spec.q,
            parameter_layout.spec.r,
            "geometric_product",
        )
        for index in parameter_layout.basis_indices
    ]
    return (
        None,
        torch.tensor(signs, dtype=dtype, device=device),
        torch.eye(parameter_layout.spec.n, dtype=dtype, device=device),
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


def _linear_action_structure(input_layout, output_layout, *, generator_layout=None, device="cpu"):
    """Describe the lift and bound the selected prepared execution storage.

    These are per broadcast item, like other static resource estimates. Count
    grade blocks without constructing their Cartesian-product index buffers.
    """
    n = input_layout.spec.n
    grades = set(input_layout.grades) & set(output_layout.grades)
    vector_only = input_layout.grades == output_layout.grades == (1,)
    blocks = {} if vector_only else {g: comb(n, g) ** 2 for g in grades if g > 0}
    direct = {g for g in blocks if _direct_action_grade(input_layout, output_layout, g, device)}
    dense = input_layout.dim * output_layout.dim
    minors = max((count * g * g for g, count in blocks.items()), default=0)
    indices = sum(count * (1 + 2 * g) for g, count in blocks.items())
    scalar = int(0 in input_layout.grades and 0 in output_layout.grades)
    # Resources, unlike route-level facts, bound the selected implementation.
    generator_terms = 0 if generator_layout is None else _bivector_generator_term_count(generator_layout)
    if direct:
        compound = {g: count for g, count in blocks.items() if g not in direct}
        compound_indices = sum(count * (1 + 2 * g) for g, count in compound.items())
        compound_minors = sum(count * g * g for g, count in compound.items() if g >= 4)
        compound_blocks = sum(compound.values())
        direct_counts = [_direct_grade_counts(n, g) for g in direct]
        # Each transition retains three index vectors and one sign vector;
        # backward may retain the gathered state and matrix coefficients.
        direct_buffers = 4 * sum(item[0] for item in direct_counts)
        direct_saved_terms = 2 * sum(item[0] for item in direct_counts)
        direct_states = sum(item[1] for item in direct_counts)
        direct_peak_terms = 2 * max(item[2] for item in direct_counts)
        # Mixed direct execution assembles an output vector, not a dense lift.
        pairs = (
            scalar
            + compound_indices
            + compound_minors
            + compound_blocks
            + direct_buffers
            + direct_saved_terms
            + direct_states
            + direct_peak_terms
            + input_layout.dim
            + output_layout.dim
        )
    else:
        # Pure compound execution still assembles the previous dense lift.
        pairs = scalar + indices + dense + minors
    if generator_layout is not None:
        pairs += 3 * generator_terms + 2 * n * n
    lanes = max(n, input_layout.dim, output_layout.dim, 0 if generator_layout is None else generator_layout.dim)
    required_pairs = pairs if direct else max(pairs, indices + n * n)
    return generator_terms, ResourceRequirements(lanes, required_pairs)
