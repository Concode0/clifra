# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Static grade-path plans for Clifford product execution.

This module describes the lower-level AoT shape needed by high-dimensional
product execution: input grades are declared or inferred at construction time
and all basis interactions are expanded once. Hot tensor execution lives in
``clifra.core.execution.product``.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch

from clifra.core.foundation.basis import (
    GradeProductOp,
    basis_index_tuple_for_grades,
    basis_indices_tensor,
    operation_coefficient,
)
from clifra.core.foundation.layout import AlgebraSpec
from clifra.core.planning.layouts import ProductRequest
from clifra.core.planning.tree import GradePlanTree, build_grade_plan_tree


class GradeProductPlan:
    """AoT basis interaction plan for one grade-restricted bilinear product."""

    def __init__(
        self,
        *,
        p: int,
        q: int,
        r: int,
        op: GradeProductOp,
        left_grades: tuple[int, ...],
        right_grades: tuple[int, ...],
        output_grades: tuple[int, ...],
        left_indices: torch.Tensor,
        right_indices: torch.Tensor,
        output_indices: torch.Tensor,
        output_positions: torch.Tensor,
        left_active_positions: torch.Tensor,
        right_active_positions: torch.Tensor,
        coefficients: torch.Tensor,
        active_output_indices: torch.Tensor,
        pairwise_contract_left: bool,
        pairwise_gather_positions: torch.Tensor,
        pairwise_coefficients: torch.Tensor,
        tree: GradePlanTree,
    ):
        self.spec = AlgebraSpec(p, q, r)
        self.op = op
        self.left_layout = self.spec.layout(left_grades)
        self.right_layout = self.spec.layout(right_grades)
        self.output_layout = self.spec.layout(output_grades)
        self.left_indices = left_indices
        self.right_indices = right_indices
        self.output_indices = output_indices
        self.output_positions = output_positions
        self.left_active_positions = left_active_positions
        self.right_active_positions = right_active_positions
        self.coefficients = coefficients
        self.active_output_indices = active_output_indices
        self.pairwise_contract_left = bool(pairwise_contract_left)
        self.pairwise_gather_positions = pairwise_gather_positions
        self.pairwise_coefficients = pairwise_coefficients
        self.tree = tree

    @property
    def p(self) -> int:
        """Return the positive metric dimension."""
        return self.spec.p

    @property
    def q(self) -> int:
        """Return the negative metric dimension."""
        return self.spec.q

    @property
    def r(self) -> int:
        """Return the null metric dimension."""
        return self.spec.r

    @property
    def left_grades(self) -> tuple[int, ...]:
        """Return the left operand grades."""
        return self.left_layout.grades

    @property
    def right_grades(self) -> tuple[int, ...]:
        """Return the right operand grades."""
        return self.right_layout.grades

    @property
    def output_grades(self) -> tuple[int, ...]:
        """Return the output grades."""
        return self.output_layout.grades

    @property
    def n(self) -> int:
        """Return the total algebra dimension."""
        return self.p + self.q + self.r

    @property
    def dim(self) -> int:
        """Return the full multivector lane count."""
        return 1 << self.n

    @property
    def pair_count(self) -> int:
        """Return the number of nonzero basis interactions."""
        return int(self.left_indices.numel())

    @property
    def output_dim(self) -> int:
        """Return the compact output lane count."""
        return int(self.active_output_indices.numel())

    @property
    def is_empty(self) -> bool:
        """Return whether the product has no nonzero basis interactions."""
        return self.pair_count == 0

    @property
    def density(self) -> float:
        """Return realized interaction density relative to the grade tree estimate."""
        if self.tree.estimated_pairs == 0:
            return 0.0
        return self.pair_count / self.tree.estimated_pairs


class FullTableProductPlan:
    """Full-layout Cayley-table product plan owned by the planner.

    This executor family is available for requests where both operands and the
    output are the canonical all-grades layout.
    """

    def __init__(
        self,
        *,
        spec: AlgebraSpec,
        op: GradeProductOp,
        cayley_indices: torch.Tensor,
        signs: torch.Tensor,
        pair_count: int,
    ):
        self.spec = spec
        self.op = op
        self.left_layout = spec.full_layout()
        self.right_layout = spec.full_layout()
        self.output_layout = spec.full_layout()
        self.cayley_indices = cayley_indices
        self.signs = signs
        self.pair_count = int(pair_count)

    @property
    def p(self) -> int:
        """Return the positive metric dimension."""
        return self.spec.p

    @property
    def q(self) -> int:
        """Return the negative metric dimension."""
        return self.spec.q

    @property
    def r(self) -> int:
        """Return the null metric dimension."""
        return self.spec.r

    @property
    def n(self) -> int:
        """Return the total algebra dimension."""
        return self.spec.n

    @property
    def dim(self) -> int:
        """Return the full multivector lane count."""
        return self.spec.dim

    @property
    def output_dim(self) -> int:
        """Return the output lane count."""
        return self.output_layout.dim

    @property
    def left_grades(self) -> tuple[int, ...]:
        """Return all left operand grades."""
        return self.left_layout.grades

    @property
    def right_grades(self) -> tuple[int, ...]:
        """Return all right operand grades."""
        return self.right_layout.grades

    @property
    def output_grades(self) -> tuple[int, ...]:
        """Return all output grades."""
        return self.output_layout.grades


def build_grade_product_plan(
    p: int,
    q: int = 0,
    r: int = 0,
    *,
    left_grades: Iterable[int],
    right_grades: Iterable[int],
    output_grades: Optional[Iterable[int]] = None,
    op: GradeProductOp = "gp",
    device=None,
    dtype: torch.dtype = torch.float32,
) -> GradeProductPlan:
    """Build an exact static basis-pair plan for a grade-restricted operation."""
    spec = AlgebraSpec(int(p), int(q), int(r))
    tree = build_grade_plan_tree(
        spec,
        left_grades=left_grades,
        right_grades=right_grades,
        output_grades=output_grades,
        op=op,
    )
    return build_grade_product_plan_from_tree(tree, device=device, dtype=dtype)


def build_grade_product_plan_from_request(
    request: ProductRequest,
    *,
    device=None,
    dtype: Optional[torch.dtype] = None,
) -> GradeProductPlan:
    """Build a plan from a normalized product request."""
    tree = build_grade_plan_tree(
        request.spec,
        left_grades=request.left_grades,
        right_grades=request.right_grades,
        output_grades=request.output_grades,
        op=request.op,
    )
    return build_grade_product_plan_from_tree(
        tree,
        device=request.device if device is None else device,
        dtype=request.dtype if dtype is None else dtype,
    )


def build_grade_product_plan_from_tree(
    tree: GradePlanTree,
    *,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> GradeProductPlan:
    """Lower a planner tree into flat Torch gather/reduce buffers."""
    spec = tree.spec
    p, q, r = spec.p, spec.q, spec.r
    n = spec.n
    left_grade_tuple = tree.left_grades
    right_grade_tuple = tree.right_grades
    output_grade_tuple = tree.output_grades
    left_layout = spec.layout(left_grade_tuple)
    right_layout = spec.layout(right_grade_tuple)

    left_basis_by_grade = {
        grade: basis_indices_tensor(
            basis_index_tuple_for_grades(n, (grade,)),
            n=n,
            role="left product basis indices",
        )
        for grade in left_grade_tuple
    }
    right_basis_by_grade = {
        grade: basis_indices_tensor(
            basis_index_tuple_for_grades(n, (grade,)),
            n=n,
            role="right product basis indices",
        )
        for grade in right_grade_tuple
    }
    left_layout_indices = basis_indices_tensor(left_layout.basis_indices, n=n, role="left layout basis indices")
    right_layout_indices = basis_indices_tensor(right_layout.basis_indices, n=n, role="right layout basis indices")
    active_outputs = basis_index_tuple_for_grades(n, output_grade_tuple)
    active_output_indices = basis_indices_tensor(active_outputs, n=n, role="active output basis indices")

    left_chunks: list[torch.Tensor] = []
    right_chunks: list[torch.Tensor] = []
    output_chunks: list[torch.Tensor] = []
    output_position_chunks: list[torch.Tensor] = []
    left_position_chunks: list[torch.Tensor] = []
    right_position_chunks: list[torch.Tensor] = []
    coefficient_chunks: list[torch.Tensor] = []
    negative_mask = sum(1 << bit for bit in range(p, p + q))
    null_mask = sum(1 << bit for bit in range(p + q, n))
    pair_limit = tree.chunk_pair_limit if tree.chunk_pair_limit and tree.chunk_pair_limit > 0 else 262_144

    for path in tree.paths:
        left_basis = left_basis_by_grade[path.left_grade]
        right_basis = right_basis_by_grade[path.right_grade]
        if left_basis.numel() == 0 or right_basis.numel() == 0:
            continue
        rows_per_chunk = max(1, min(left_basis.numel(), int(pair_limit) // max(int(right_basis.numel()), 1)))
        right = right_basis.view(1, -1)
        right_positions_template = _positions_in_sorted_indices(right, right_layout_indices).expand(rows_per_chunk, -1)
        allowed_output_grades = torch.tensor(path.output_grades, dtype=torch.long)

        for start in range(0, left_basis.numel(), rows_per_chunk):
            stop = min(start + rows_per_chunk, left_basis.numel())
            left = left_basis[start:stop].view(-1, 1)
            output_indices = torch.bitwise_xor(left, right)
            output_positions = _positions_in_sorted_indices(output_indices, active_output_indices)
            output_grades = _bit_count_tensor(output_indices, n)
            valid = output_positions >= 0
            valid = valid & (output_grades.unsqueeze(-1) == allowed_output_grades).any(dim=-1)
            coefficients = _operation_coefficients(
                left,
                right,
                output_indices,
                op=tree.op,
                left_grade=path.left_grade,
                right_grade=path.right_grade,
                n=n,
                negative_mask=negative_mask,
                null_mask=null_mask,
                dtype=dtype,
            )
            valid = valid & (coefficients != 0)

            left_indices = left.expand(-1, right_basis.numel())
            right_indices = right.expand(stop - start, -1)
            left_positions = _positions_in_sorted_indices(left, left_layout_indices).expand(-1, right_basis.numel())
            right_positions = right_positions_template[: stop - start]

            left_chunks.append(left_indices[valid])
            right_chunks.append(right_indices[valid])
            output_chunks.append(output_indices[valid])
            output_position_chunks.append(output_positions[valid])
            left_position_chunks.append(left_positions[valid])
            right_position_chunks.append(right_positions[valid])
            coefficient_chunks.append(coefficients[valid])

    plan_left = _cat_long_chunks(left_chunks)
    plan_right = _cat_long_chunks(right_chunks)
    plan_output = _cat_long_chunks(output_chunks)
    plan_positions = _cat_long_chunks(output_position_chunks)
    plan_left_active_positions = _cat_long_chunks(left_position_chunks)
    plan_right_active_positions = _cat_long_chunks(right_position_chunks)
    plan_coefficients = _cat_float_chunks(coefficient_chunks, dtype=dtype)
    pairwise_contract_left, pairwise_gather_positions, pairwise_coefficients = _build_pairwise_contraction_buffers(
        left_dim=left_layout.dim,
        right_dim=right_layout.dim,
        output_dim=len(active_outputs),
        left_positions=plan_left_active_positions,
        right_positions=plan_right_active_positions,
        output_positions=plan_positions,
        coefficients=plan_coefficients,
        dtype=dtype,
        device=device,
    )

    return GradeProductPlan(
        p=p,
        q=q,
        r=r,
        op=tree.op,
        left_grades=left_grade_tuple,
        right_grades=right_grade_tuple,
        output_grades=output_grade_tuple,
        left_indices=plan_left.to(device=device),
        right_indices=plan_right.to(device=device),
        output_indices=plan_output.to(device=device),
        output_positions=plan_positions.to(device=device),
        left_active_positions=plan_left_active_positions.to(device=device),
        right_active_positions=plan_right_active_positions.to(device=device),
        coefficients=plan_coefficients.to(device=device),
        active_output_indices=basis_indices_tensor(
            active_outputs, n=n, role="active output basis indices", device=device
        ),
        pairwise_contract_left=pairwise_contract_left,
        pairwise_gather_positions=pairwise_gather_positions,
        pairwise_coefficients=pairwise_coefficients,
        tree=tree,
    )


def _build_pairwise_contraction_buffers(
    *,
    left_dim: int,
    right_dim: int,
    output_dim: int,
    left_positions: torch.Tensor,
    right_positions: torch.Tensor,
    output_positions: torch.Tensor,
    coefficients: torch.Tensor,
    dtype: torch.dtype,
    device,
) -> tuple[bool, torch.Tensor, torch.Tensor]:
    """Return static factorized buffers for pairwise compact products.

    Each Clifford basis interaction has ``output = left xor right``. For a fixed
    output lane and one operand lane, the matching opposite operand lane is
    therefore unique. Pairwise execution can contract over the smaller operand
    width instead of materializing one term per planned basis interaction for
    every pair of items.
    """
    contract_left = left_dim <= right_dim
    contract_dim = left_dim if contract_left else right_dim
    gather_rows = torch.zeros((contract_dim, output_dim), dtype=torch.long)
    coefficient_rows = torch.zeros((contract_dim, output_dim), dtype=dtype)

    if contract_left:
        gather_rows[left_positions, output_positions] = right_positions
        coefficient_rows.index_put_((left_positions, output_positions), coefficients, accumulate=True)
    else:
        gather_rows[right_positions, output_positions] = left_positions
        coefficient_rows.index_put_((right_positions, output_positions), coefficients, accumulate=True)
    return (
        contract_left,
        gather_rows.to(device=device),
        coefficient_rows.to(device=device),
    )


def _positions_in_sorted_indices(values: torch.Tensor, sorted_indices: torch.Tensor) -> torch.Tensor:
    if sorted_indices.numel() == 0:
        return torch.full_like(values, -1)
    positions = torch.searchsorted(sorted_indices, values)
    clamped = positions.clamp_max(sorted_indices.numel() - 1)
    found = (positions < sorted_indices.numel()) & (torch.index_select(sorted_indices, 0, clamped.reshape(-1)).reshape_as(values) == values)
    return torch.where(found, positions, torch.full_like(positions, -1))


def _bit_count_tensor(values: torch.Tensor, n: int) -> torch.Tensor:
    counts = torch.zeros_like(values)
    for bit in range(int(n)):
        counts = counts + ((torch.bitwise_and(values, 1 << bit) != 0).to(torch.long))
    return counts


def _parity_tensor(values: torch.Tensor, n: int) -> torch.Tensor:
    return (_bit_count_tensor(values, n) & 1).to(torch.bool)


def _cat_long_chunks(chunks: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(chunks) if chunks else torch.zeros(0, dtype=torch.long)


def _cat_float_chunks(chunks: list[torch.Tensor], *, dtype: torch.dtype) -> torch.Tensor:
    return torch.cat(chunks) if chunks else torch.zeros(0, dtype=dtype)


def _operation_coefficients(
    left: torch.Tensor,
    right: torch.Tensor,
    outputs: torch.Tensor,
    *,
    op: GradeProductOp,
    left_grade: int,
    right_grade: int,
    n: int,
    negative_mask: int,
    null_mask: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    overlap = torch.bitwise_and(left, right)
    valid = torch.ones_like(outputs, dtype=torch.bool)
    if op == "wedge":
        valid = overlap == 0
    elif null_mask:
        valid = torch.bitwise_and(overlap, null_mask) == 0

    output_grade = _bit_count_tensor(outputs, n)
    if op == "left_contraction":
        valid = valid & (left_grade <= right_grade) & (output_grade == right_grade - left_grade)
    elif op == "right_contraction":
        valid = valid & (left_grade >= right_grade) & (output_grade == left_grade - right_grade)
    elif op in {"inner", "commutator", "anti_commutator"}:
        overlap_grade = _bit_count_tensor(overlap, n)
        parity_odd = ((int(left_grade) * int(right_grade) - overlap_grade) & 1).to(torch.bool)
        if op == "commutator":
            valid = valid & parity_odd
        else:
            valid = valid & ~parity_odd
    elif op not in {"gp", "wedge"}:
        raise ValueError(f"Unsupported grade product op {op!r}")

    swap_parity = torch.zeros_like(outputs, dtype=torch.bool)
    for bit in range(n):
        left_has_bit = torch.bitwise_and(left, 1 << bit) != 0
        lower_right_parity = _parity_tensor(torch.bitwise_and(right, (1 << bit) - 1), bit)
        swap_parity = swap_parity ^ (left_has_bit & lower_right_parity)
    if negative_mask:
        metric_parity = _parity_tensor(torch.bitwise_and(overlap, negative_mask), n)
        swap_parity = swap_parity ^ metric_parity

    factor = 2.0 if op in {"commutator", "anti_commutator"} else 1.0
    coefficients = torch.where(
        swap_parity,
        torch.full((), -factor, dtype=dtype),
        torch.full((), factor, dtype=dtype),
    )
    return torch.where(valid, coefficients, torch.zeros((), dtype=dtype))


def build_full_table_product_plan_from_request(
    request: ProductRequest,
    *,
    device=None,
    dtype: Optional[torch.dtype] = None,
) -> FullTableProductPlan:
    """Build a full-table product plan from a normalized request."""
    if not request_is_full_layout_product(request):
        raise ValueError("full-table product plans require full left, right, and output layouts")
    return build_full_table_product_plan(
        request.spec,
        op=request.op,
        device=request.device if device is None else device,
        dtype=request.dtype if dtype is None else dtype,
    )


def build_full_table_product_plan(
    spec: AlgebraSpec,
    *,
    op: GradeProductOp,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> FullTableProductPlan:
    """Build Cayley-style buffers for one full-layout product."""
    dim = spec.dim
    indices = torch.arange(dim, dtype=torch.long, device=device)
    cayley_indices = indices.unsqueeze(0) ^ indices.unsqueeze(1)
    sign_rows: list[list[float]] = []
    pair_count = 0
    for left_index in range(dim):
        row = []
        for output_index in range(dim):
            right_index = left_index ^ output_index
            coefficient = operation_coefficient(left_index, right_index, spec.p, spec.q, spec.r, op)
            row.append(coefficient)
            if coefficient != 0.0:
                pair_count += 1
        sign_rows.append(row)
    signs = torch.tensor(sign_rows, dtype=dtype, device=device)
    return FullTableProductPlan(
        spec=spec,
        op=op,
        cayley_indices=cayley_indices,
        signs=signs,
        pair_count=pair_count,
    )


def request_is_full_layout_product(request: ProductRequest) -> bool:
    """Return whether a request is the canonical full-layout product."""
    full_grades = tuple(range(request.spec.n + 1))
    return (
        request.left_grades == full_grades
        and request.right_grades == full_grades
        and request.output_grades == full_grades
    )
