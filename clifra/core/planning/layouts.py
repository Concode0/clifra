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

_VALID_PRODUCT_OPS = {"gp", "wedge", "inner", "commutator", "anti_commutator"}


@dataclass(frozen=True)
class ProductRequest:
    """Fully resolved static request for one bilinear product.

    The request is the planner's intermediate representation. It removes
    ambiguity from caller input before any executor is built: layouts are
    normalized, compact-vs-dense operand storage is known, and output grades are
    inferred when callers do not explicitly project them.
    """

    spec: AlgebraSpec
    op: GradeProductOp
    left_layout: GradeLayout
    right_layout: GradeLayout
    output_layout: GradeLayout
    left_compact: bool
    right_compact: bool
    dtype: torch.dtype
    device: torch.device

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
    left_layout = resolve_operand_layout(
        spec,
        left,
        grades=left_grades,
        layout=left_layout,
        compact=left_compact,
        side="left",
        full_layout_allowed=full_layout_allowed,
    )
    right_layout = resolve_operand_layout(
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
        left_layout=left_layout,
        right_layout=right_layout,
        output_grades=output_grades,
        output_layout=output_layout,
    )

    left_compact = left_compact or is_compact_tensor(spec, left, left_layout)
    right_compact = right_compact or is_compact_tensor(spec, right, right_layout)

    return ProductRequest(
        spec=spec,
        op=normalized_op,
        left_layout=left_layout,
        right_layout=right_layout,
        output_layout=output_layout,
        left_compact=left_compact,
        right_compact=right_compact,
        dtype=torch.promote_types(left.dtype, right.dtype),
        device=left.device,
    )


def normalize_product_op(op: str) -> GradeProductOp:
    """Validate and normalize a product operation name."""
    normalized = str(op)
    if normalized not in _VALID_PRODUCT_OPS:
        raise ValueError(f"Unsupported grade product op {op!r}")
    return normalized  # type: ignore[return-value]


def resolve_operand_layout(
    spec: AlgebraSpec,
    tensor: torch.Tensor,
    *,
    grades=None,
    layout: Optional[GradeLayout] = None,
    compact: bool = False,
    side: str,
    full_layout_allowed: bool = True,
) -> GradeLayout:
    """Resolve one operand's grade layout from explicit metadata or tensor shape."""
    if layout is not None:
        check_layout_spec(spec, layout, f"{side}_layout")
        if grades is not None and layout.grades != normalize_grades(grades, spec.n, name=f"{side}_grades"):
            raise ValueError(f"{side}_layout and {side}_grades disagree")
        _check_operand_shape(spec, tensor, layout, compact=compact, side=side)
        return layout

    if grades is not None:
        layout = spec.layout(grades)
        _check_operand_shape(spec, tensor, layout, compact=compact, side=side)
        return layout

    if compact:
        raise ValueError(f"{side}_layout or {side}_grades is required for compact {side} input")
    if tensor.shape[-1] != spec.dim:
        raise ValueError(
            f"{side} input has last dimension {tensor.shape[-1]}; declare {side}_layout or "
            f"{side}_grades for compact planned execution"
        )
    if not full_layout_allowed:
        raise ValueError(
            f"{side} input would require a full Cl({spec.p},{spec.q},{spec.r}) layout. "
            "Declare active grades or enable an explicit low-dimensional full-layout fallback."
        )
    return spec.layout(range(spec.n + 1))


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


def check_layout_spec(spec: AlgebraSpec, layout: GradeLayout, name: str) -> None:
    """Validate that a layout belongs to ``spec``."""
    if layout.spec != spec:
        raise ValueError(f"{name} signature {layout.spec} does not match product spec {spec}")


def is_compact_tensor(spec: AlgebraSpec, tensor: torch.Tensor, layout: GradeLayout) -> bool:
    """Return whether ``tensor`` already uses ``layout``'s compact lane count."""
    return layout.dim != spec.dim and tensor.shape[-1] == layout.dim


def _check_operand_shape(
    spec: AlgebraSpec,
    tensor: torch.Tensor,
    layout: GradeLayout,
    *,
    compact: bool,
    side: str,
) -> None:
    expected = layout.dim if compact or is_compact_tensor(spec, tensor, layout) else spec.dim
    if tensor.shape[-1] != expected:
        storage = "compact" if expected == layout.dim else "dense"
        raise ValueError(f"{side} {storage} last dimension must be {expected}, got {tensor.shape[-1]}")
