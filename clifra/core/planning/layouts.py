# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Layout and request normalization for static grade planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from clifra.core.foundation.basis import GradeProductOp, expand_output_grades, normalize_grades
from clifra.core.foundation.layout import AlgebraSpec, GradeLayout
from clifra.core.storage import (
    TensorStorage,
    check_layout_spec,
    resolve_operand_layout,
    resolve_tensor_storage,
)
from clifra.core.storage import (
    tensor_is_compact as is_compact_tensor,
)

_VALID_PRODUCT_OPS = {"gp", "wedge", "inner", "commutator", "anti_commutator"}

__all__ = [
    "ProductRequest",
    "build_product_request",
    "check_layout_spec",
    "is_compact_tensor",
    "normalize_product_op",
    "resolve_operand_layout",
    "resolve_output_layout",
]


@dataclass(frozen=True)
class ProductRequest:
    """Fully resolved static request for one bilinear product.

    The request is the planner's intermediate representation. It removes
    ambiguity from caller input before any executor is built: layouts are
    normalized, physical operand storage is known, and output grades are
    inferred when callers do not explicitly project them.
    """

    spec: AlgebraSpec
    op: GradeProductOp
    left_storage: TensorStorage
    right_storage: TensorStorage
    output_storage: TensorStorage
    dtype: torch.dtype
    device: torch.device

    @property
    def left_layout(self) -> GradeLayout:
        return self.left_storage.layout

    @property
    def right_layout(self) -> GradeLayout:
        return self.right_storage.layout

    @property
    def output_layout(self) -> GradeLayout:
        return self.output_storage.layout

    @property
    def left_compact(self) -> bool:
        return self.left_storage.is_compact

    @property
    def right_compact(self) -> bool:
        return self.right_storage.is_compact

    @property
    def left_grades(self) -> tuple[int, ...]:
        return self.left_layout.grades

    @property
    def right_grades(self) -> tuple[int, ...]:
        return self.right_layout.grades

    @property
    def output_grades(self) -> tuple[int, ...]:
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
    left_compact: bool = False,
    right_compact: bool = False,
    full_layout_allowed: bool = True,
) -> ProductRequest:
    """Resolve caller input into a static product request."""
    normalized_op = normalize_product_op(op)
    left_storage = resolve_tensor_storage(
        spec,
        left,
        grades=left_grades,
        layout=left_layout,
        compact=left_compact,
        side="left",
        full_layout_allowed=full_layout_allowed,
    )
    right_storage = resolve_tensor_storage(
        spec,
        right,
        grades=right_grades,
        layout=right_layout,
        compact=right_compact,
        side="right",
        full_layout_allowed=full_layout_allowed,
    )
    output_layout = resolve_output_layout(
        spec,
        op=normalized_op,
        left_layout=left_storage.layout,
        right_layout=right_storage.layout,
        output_grades=output_grades,
        output_layout=output_layout,
    )
    output_storage = TensorStorage.compact(spec, output_layout)

    return ProductRequest(
        spec=spec,
        op=normalized_op,
        left_storage=left_storage,
        right_storage=right_storage,
        output_storage=output_storage,
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
        output_grades = expand_output_grades(left_layout.grades, right_layout.grades, spec.n, op=op)
    return spec.layout(output_grades)
