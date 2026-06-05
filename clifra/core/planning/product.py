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

    left_basis_by_grade = {grade: basis_index_tuple_for_grades(n, (grade,)) for grade in left_grade_tuple}
    right_basis_by_grade = {grade: basis_index_tuple_for_grades(n, (grade,)) for grade in right_grade_tuple}
    left_position_by_index = {
        index: position for position, index in enumerate(spec.layout(left_grade_tuple).basis_indices)
    }
    right_position_by_index = {
        index: position for position, index in enumerate(spec.layout(right_grade_tuple).basis_indices)
    }
    active_outputs = basis_index_tuple_for_grades(n, output_grade_tuple)
    output_position_by_index = {index: position for position, index in enumerate(active_outputs)}

    plan_left: list[int] = []
    plan_right: list[int] = []
    plan_output: list[int] = []
    plan_positions: list[int] = []
    plan_left_active_positions: list[int] = []
    plan_right_active_positions: list[int] = []
    plan_coefficients: list[float] = []

    for path in tree.paths:
        path_output_grades = set(path.output_grades)
        for left_index in left_basis_by_grade[path.left_grade]:
            for right_index in right_basis_by_grade[path.right_grade]:
                output_index = left_index ^ right_index
                if output_index.bit_count() not in path_output_grades:
                    continue
                output_position = output_position_by_index.get(output_index)
                if output_position is None:
                    continue
                coefficient = operation_coefficient(left_index, right_index, p, q, r, tree.op)
                if coefficient == 0.0:
                    continue
                plan_left.append(left_index)
                plan_right.append(right_index)
                plan_output.append(output_index)
                plan_positions.append(output_position)
                plan_left_active_positions.append(left_position_by_index[left_index])
                plan_right_active_positions.append(right_position_by_index[right_index])
                plan_coefficients.append(coefficient)

    return GradeProductPlan(
        p=p,
        q=q,
        r=r,
        op=tree.op,
        left_grades=left_grade_tuple,
        right_grades=right_grade_tuple,
        output_grades=output_grade_tuple,
        left_indices=basis_indices_tensor(plan_left, n=n, role="left product basis indices", device=device),
        right_indices=basis_indices_tensor(plan_right, n=n, role="right product basis indices", device=device),
        output_indices=basis_indices_tensor(plan_output, n=n, role="output product basis indices", device=device),
        output_positions=torch.tensor(plan_positions, dtype=torch.long, device=device),
        left_active_positions=torch.tensor(plan_left_active_positions, dtype=torch.long, device=device),
        right_active_positions=torch.tensor(plan_right_active_positions, dtype=torch.long, device=device),
        coefficients=torch.tensor(plan_coefficients, dtype=dtype, device=device),
        active_output_indices=basis_indices_tensor(
            active_outputs, n=n, role="active output basis indices", device=device
        ),
        tree=tree,
    )


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
