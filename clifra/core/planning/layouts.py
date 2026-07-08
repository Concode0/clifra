# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Layout and request normalization for static grade planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from clifra.core.foundation.basis import (
    GradeProductOp,
    expand_output_grades,
    normalize_grade_product_op,
    normalize_grades,
)
from clifra.core.foundation.layout import AlgebraSpec, GradeLayout
from clifra.core.runtime.tensors import (
    LaneStorage,
    TensorContract,
    check_layout_spec,
    infer_contract,
    normalize_lane_storage,
)

__all__ = [
    "ProductRequest",
    "build_product_request",
    "check_layout_spec",
    "normalize_product_op",
    "resolve_output_layout",
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
    left: TensorContract
    right: TensorContract
    output: TensorContract
    dtype: torch.dtype
    device: torch.device

    @property
    def left_layout(self) -> GradeLayout:
        """Return the resolved layout for the left operand."""
        return self.left.layout

    @property
    def right_layout(self) -> GradeLayout:
        """Return the resolved layout for the right operand."""
        return self.right.layout

    @property
    def output_layout(self) -> GradeLayout:
        """Return the resolved layout for the product output."""
        return self.output.layout

    @property
    def left_uses_compact_storage(self) -> bool:
        """Return whether the left tensor is already compact."""
        return self.left.uses_compact_storage

    @property
    def right_uses_compact_storage(self) -> bool:
        """Return whether the right tensor is already compact."""
        return self.right.uses_compact_storage

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
    left_storage: LaneStorage | str | None = None,
    right_storage: LaneStorage | str | None = None,
    output_storage: LaneStorage | str = LaneStorage.COMPACT,
) -> ProductRequest:
    """Resolve caller input into a static product request."""
    normalized_op = normalize_product_op(op)
    left_contract = infer_contract(
        spec,
        left,
        grades=left_grades,
        layout=left_layout,
        storage=left_storage,
        side="left",
    )
    right_contract = infer_contract(
        spec,
        right,
        grades=right_grades,
        layout=right_layout,
        storage=right_storage,
        side="right",
    )
    output_layout = resolve_output_layout(
        spec,
        op=normalized_op,
        left_layout=left_contract.layout,
        right_layout=right_contract.layout,
        output_grades=output_grades,
        output_layout=output_layout,
    )
    output_contract = TensorContract(spec=spec, layout=output_layout, storage=normalize_lane_storage(output_storage))

    return ProductRequest(
        spec=spec,
        op=normalized_op,
        left=left_contract,
        right=right_contract,
        output=output_contract,
        dtype=torch.promote_types(left.dtype, right.dtype),
        device=left.device,
    )


def normalize_product_op(op: str) -> GradeProductOp:
    """Validate and normalize a product operation name."""
    return normalize_grade_product_op(op)


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
