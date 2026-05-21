# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Compile-friendly planned unary operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

from clifra.core.foundation.basis import normalize_grades, reverse_sign
from clifra.core.foundation.layout import AlgebraSpec, GradeLayout
from clifra.core.planning.layouts import check_layout_spec, is_compact_tensor, resolve_operand_layout

GradeUnaryOp = Literal["identity", "reverse", "grade_involution", "clifford_conjugation", "grade_projection"]
_VALID_UNARY_OPS = {"identity", "reverse", "grade_involution", "clifford_conjugation", "grade_projection"}


@dataclass(frozen=True)
class UnaryRequest:
    """Fully resolved request for one unary planned operation."""

    spec: AlgebraSpec
    op: GradeUnaryOp
    input_layout: GradeLayout
    output_layout: GradeLayout
    input_compact: bool
    dtype: torch.dtype
    device: torch.device

    @property
    def input_grades(self) -> tuple[int, ...]:
        return self.input_layout.grades

    @property
    def output_grades(self) -> tuple[int, ...]:
        return self.output_layout.grades

    @property
    def cache_key(self) -> tuple[object, ...]:
        return (
            self.spec,
            str(self.device),
            str(self.dtype),
            self.op,
            self.input_grades,
            self.output_grades,
        )


class GradeUnaryPlan:
    """Static gather/sign plan for one unary operation."""

    def __init__(
        self,
        *,
        spec: AlgebraSpec,
        op: GradeUnaryOp,
        input_grades: tuple[int, ...],
        output_grades: tuple[int, ...],
        input_positions: torch.Tensor,
        output_indices: torch.Tensor,
        signs: torch.Tensor,
    ):
        self.spec = spec
        self.op = op
        self.input_layout = spec.layout(input_grades)
        self.output_layout = spec.layout(output_grades)
        self.input_positions = input_positions
        self.output_indices = output_indices
        self.signs = signs

    @property
    def dim(self) -> int:
        return self.spec.dim

    @property
    def output_dim(self) -> int:
        return self.output_layout.dim


class GradeUnaryExecutor(nn.Module):
    """Torch module for planned unary gather/sign execution."""

    def __init__(self, plan: GradeUnaryPlan):
        super().__init__()
        self.spec = plan.spec
        self.op = plan.op
        self.input_layout = plan.input_layout
        self.output_layout = plan.output_layout
        self.dim = plan.dim
        self.register_buffer("input_positions", plan.input_positions, persistent=False)
        self.register_buffer("output_indices", plan.output_indices, persistent=False)
        self.register_buffer("signs", plan.signs, persistent=False)

    @property
    def output_dim(self) -> int:
        return self.output_layout.dim

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return compact output lanes for dense input coefficients."""
        if values.shape[-1] != self.dim:
            raise ValueError(f"dense last dimension must be {self.dim}, got {values.shape[-1]}")
        output = torch.index_select(values, -1, self.output_indices)
        return output * self.signs.to(dtype=output.dtype)

    def forward_compact(self, values: torch.Tensor) -> torch.Tensor:
        """Return compact output lanes for compact input coefficients."""
        if values.shape[-1] != self.input_layout.dim:
            raise ValueError(f"compact last dimension must be {self.input_layout.dim}, got {values.shape[-1]}")
        output = torch.index_select(values, -1, self.input_positions)
        return output * self.signs.to(dtype=output.dtype)

    def forward_dense(self, values: torch.Tensor) -> torch.Tensor:
        """Return dense output coefficients for dense input coefficients."""
        compact = self.forward(values)
        return self.output_layout.dense(compact)


def build_unary_request(
    spec: AlgebraSpec,
    values: torch.Tensor,
    *,
    op: str,
    input_grades=None,
    output_grades=None,
    input_layout: Optional[GradeLayout] = None,
    output_layout: Optional[GradeLayout] = None,
    input_compact: bool = False,
    full_layout_allowed: bool = True,
) -> UnaryRequest:
    """Resolve caller input into a static unary request."""
    op = normalize_unary_op(op)
    if op == "grade_projection" and input_grades is None and input_layout is None and not input_compact:
        if output_layout is not None:
            input_layout = output_layout
        elif output_grades is not None:
            input_grades = output_grades
    input_layout = resolve_operand_layout(
        spec,
        values,
        grades=input_grades,
        layout=input_layout,
        compact=input_compact,
        side="input",
        full_layout_allowed=full_layout_allowed,
    )
    output_layout = resolve_unary_output_layout(
        spec,
        op=op,
        input_layout=input_layout,
        output_grades=output_grades,
        output_layout=output_layout,
    )
    input_compact = input_compact or is_compact_tensor(spec, values, input_layout)
    return UnaryRequest(
        spec=spec,
        op=op,
        input_layout=input_layout,
        output_layout=output_layout,
        input_compact=input_compact,
        dtype=values.dtype,
        device=values.device,
    )


def build_unary_plan_from_request(request: UnaryRequest) -> GradeUnaryPlan:
    """Lower a unary request into static gather/sign buffers."""
    input_position_by_index = {index: pos for pos, index in enumerate(request.input_layout.basis_indices)}
    input_positions = []
    signs = []
    for index in request.output_layout.basis_indices:
        position = input_position_by_index.get(index)
        if position is None:
            raise ValueError(
                f"output basis index {index} is not available in input grades {request.input_layout.grades}"
            )
        input_positions.append(position)
        signs.append(_unary_sign(request.op, index))

    return GradeUnaryPlan(
        spec=request.spec,
        op=request.op,
        input_grades=request.input_grades,
        output_grades=request.output_grades,
        input_positions=torch.tensor(input_positions, dtype=torch.long, device=request.device),
        output_indices=torch.tensor(request.output_layout.basis_indices, dtype=torch.long, device=request.device),
        signs=torch.tensor(signs, dtype=request.dtype, device=request.device),
    )


def normalize_unary_op(op: str) -> GradeUnaryOp:
    """Validate and normalize a unary operation name."""
    normalized = str(op)
    if normalized not in _VALID_UNARY_OPS:
        raise ValueError(f"Unsupported grade unary op {op!r}")
    return normalized  # type: ignore[return-value]


def resolve_unary_output_layout(
    spec: AlgebraSpec,
    *,
    op: GradeUnaryOp,
    input_layout: GradeLayout,
    output_grades=None,
    output_layout: Optional[GradeLayout] = None,
) -> GradeLayout:
    """Resolve output layout for a planned unary operation."""
    if output_layout is not None:
        check_layout_spec(spec, output_layout, "output_layout")
        if output_grades is not None and output_layout.grades != normalize_grades(
            output_grades, spec.n, name="output_grades"
        ):
            raise ValueError("output_layout and output_grades disagree")
        return output_layout

    if op == "grade_projection":
        if output_grades is None:
            raise ValueError("output_grades is required for grade_projection")
        projected = normalize_grades(output_grades, spec.n, name="output_grades")
        missing = tuple(grade for grade in projected if grade not in input_layout.grades)
        if missing:
            raise ValueError(f"Cannot project missing grades {missing} from input grades {input_layout.grades}")
        return spec.layout(projected)

    if output_grades is not None:
        projected = normalize_grades(output_grades, spec.n, name="output_grades")
        missing = tuple(grade for grade in projected if grade not in input_layout.grades)
        if missing:
            raise ValueError(f"Cannot project missing grades {missing} from input grades {input_layout.grades}")
        return spec.layout(projected)
    return input_layout


def _unary_sign(op: GradeUnaryOp, index: int) -> float:
    grade = int(index).bit_count()
    if op in {"identity", "grade_projection"}:
        return 1.0
    if op == "reverse":
        return reverse_sign(index)
    if op == "grade_involution":
        return -1.0 if grade % 2 else 1.0
    if op == "clifford_conjugation":
        return (-1.0 if grade % 2 else 1.0) * reverse_sign(index)
    raise ValueError(f"Unsupported grade unary op {op!r}")
