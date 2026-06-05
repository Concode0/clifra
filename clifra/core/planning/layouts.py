# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Layout and request normalization for static grade planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from clifra.core.foundation.basis import GradeProductOp, expand_output_grades, normalize_grades
from clifra.core.foundation.layout import AlgebraSpec, GradeLayout
from clifra.core.storage import (
    ValueLayout,
    check_layout_spec,
    resolve_operand_layout,
    resolve_value_layout,
    tensor_uses_active_lanes,
)

_VALID_PRODUCT_OPS = {
    "gp",
    "wedge",
    "inner",
    "commutator",
    "anti_commutator",
    "left_contraction",
    "right_contraction",
}

__all__ = [
    "ProductRequest",
    "build_product_request",
    "check_layout_spec",
    "normalize_product_op",
    "resolve_operand_layout",
    "resolve_output_layout",
    "tensor_uses_active_lanes",
]


@dataclass(frozen=True)
class ProductRequest:
    """Fully resolved static request for one bilinear product.

    The request is the planner's intermediate representation. It removes
    ambiguity from caller input before any executor is built: layouts and lane
    widths are normalized, and output grades are inferred when callers do not
    explicitly project them.
    """

    spec: AlgebraSpec
    op: GradeProductOp
    left_value: ValueLayout
    right_value: ValueLayout
    output_value: ValueLayout
    dtype: torch.dtype
    device: torch.device

    @property
    def left_layout(self) -> GradeLayout:
        """Return the resolved layout for the left operand."""
        return self.left_value.layout

    @property
    def right_layout(self) -> GradeLayout:
        """Return the resolved layout for the right operand."""
        return self.right_value.layout

    @property
    def output_layout(self) -> GradeLayout:
        """Return the resolved layout for the product output."""
        return self.output_value.layout

    @property
    def left_uses_active_lanes(self) -> bool:
        """Return whether the left tensor is already compact."""
        return self.left_value.uses_active_lanes

    @property
    def right_uses_active_lanes(self) -> bool:
        """Return whether the right tensor is already compact."""
        return self.right_value.uses_active_lanes

    @property
    def left_grades(self) -> tuple[int, ...]:
        """Return the grades selected from the left operand."""
        return self.left_layout.grades

    @property
    def right_grades(self) -> tuple[int, ...]:
        """Return the grades selected from the right operand."""
        return self.right_layout.grades

    @property
    def output_grades(self) -> tuple[int, ...]:
        """Return the grades selected for the product output."""
        return self.output_layout.grades

    @property
    def cache_key(self) -> tuple[object, ...]:
        """Stable key for executor caching."""
        return (
            self.spec,
            str(self.device),
            str(self.dtype),
            self.op,
            self.left_grades,
            self.right_grades,
            self.output_grades,
        )


def build_product_request(
    spec: AlgebraSpec,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    op: str = "gp",
    left_grades=None,
    right_grades=None,
    output_grades=None,
    left_layout: Optional[GradeLayout] = None,
    right_layout: Optional[GradeLayout] = None,
    output_layout: Optional[GradeLayout] = None,
    left_active_lanes: bool = False,
    right_active_lanes: bool = False,
) -> ProductRequest:
    """Resolve caller input into a static product request."""
    normalized_op = normalize_product_op(op)
    left_value = resolve_value_layout(
        spec,
        left,
        grades=left_grades,
        layout=left_layout,
        active_lanes=left_active_lanes,
        side="left",
    )
    right_value = resolve_value_layout(
        spec,
        right,
        grades=right_grades,
        layout=right_layout,
        active_lanes=right_active_lanes,
        side="right",
    )
    output_layout = resolve_output_layout(
        spec,
        op=normalized_op,
        left_layout=left_value.layout,
        right_layout=right_value.layout,
        output_grades=output_grades,
        output_layout=output_layout,
    )
    output_value = ValueLayout.active(spec, output_layout)

    return ProductRequest(
        spec=spec,
        op=normalized_op,
        left_value=left_value,
        right_value=right_value,
        output_value=output_value,
        dtype=torch.promote_types(left.dtype, right.dtype),
        device=left.device,
    )


def normalize_product_op(op: str) -> GradeProductOp:
    """Validate and normalize a product operation name."""
    normalized = str(op)
    if normalized not in _VALID_PRODUCT_OPS:
        raise ValueError(f"Unsupported grade product op {op!r}")
    return normalized  # type: ignore[return-value]


def resolve_output_layout(
    spec: AlgebraSpec,
    *,
    op: GradeProductOp,
    left_layout: GradeLayout,
    right_layout: GradeLayout,
    output_grades=None,
    output_layout: Optional[GradeLayout] = None,
) -> GradeLayout:
    """Resolve the output layout for a product request."""
    if output_layout is not None:
        check_layout_spec(spec, output_layout, "output_layout")
        if output_grades is not None and output_layout.grades != normalize_grades(
            output_grades, spec.n, name="output_grades"
        ):
            raise ValueError("output_layout and output_grades disagree")
        return output_layout

    if output_grades is None:
        full_grades = tuple(range(spec.n + 1))
        if left_layout.grades == full_grades and right_layout.grades == full_grades:
            return spec.full_layout()
        output_grades = expand_output_grades(left_layout.grades, right_layout.grades, spec.n, op=op)
    return spec.layout(output_grades)
