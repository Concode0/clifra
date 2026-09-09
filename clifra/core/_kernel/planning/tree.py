# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Planner-only grade tree metadata.

The tree here is not a runtime backend. It groups declared grade routes before
they are lowered into flat Torch executor buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from clifra.core._kernel.basis import (
    GradeProductOp,
    expand_output_grades,
    normalize_grade_product_op,
    normalize_grades,
    product_output_grades,
)
from clifra.core.layout import AlgebraSpec


@dataclass(frozen=True)
class GradePathNode:
    """One homogeneous left-grade/right-grade route in a product plan."""

    left_grade: int
    right_grade: int
    output_grades: tuple[int, ...]


@dataclass(frozen=True)
class GradePlanTree:
    """Planner-side grouping for a grade-restricted product."""

    spec: AlgebraSpec
    op: GradeProductOp
    left_grades: tuple[int, ...]
    right_grades: tuple[int, ...]
    output_grades: tuple[int, ...]
    paths: tuple[GradePathNode, ...]


def build_grade_plan_tree(
    spec: AlgebraSpec,
    *,
    left_grades: Iterable[int],
    right_grades: Iterable[int],
    output_grades: Optional[Iterable[int]] = None,
    op: GradeProductOp = "geometric_product",
) -> GradePlanTree:
    """Build planner metadata for grade route grouping."""
    op = normalize_grade_product_op(op)
    left = normalize_grades(left_grades, spec.n, name="left_grades")
    right = normalize_grades(right_grades, spec.n, name="right_grades")
    output = (
        expand_output_grades(left, right, spec.n, op=op)
        if output_grades is None
        else normalize_grades(output_grades, spec.n, name="output_grades")
    )
    output_set = set(output)

    paths = []
    for left_grade in left:
        for right_grade in right:
            route_outputs = product_output_grades(left_grade, right_grade, spec.n, op=op)
            route_outputs = tuple(grade for grade in route_outputs if grade in output_set)
            if not route_outputs:
                continue
            paths.append(
                GradePathNode(
                    left_grade=left_grade,
                    right_grade=right_grade,
                    output_grades=route_outputs,
                )
            )

    return GradePlanTree(
        spec=spec,
        op=op,
        left_grades=left,
        right_grades=right,
        output_grades=output,
        paths=tuple(paths),
    )
