# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Static unary requests and plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from clifra.core.foundation.basis import normalize_grades, reverse_sign
from clifra.core.foundation.layout import AlgebraSpec, GradeLayout
from clifra.core.runtime.tensors import (
    LaneStorage,
    TensorContract,
    check_layout_spec,
    infer_contract,
)

GradeUnaryOp = Literal["identity", "reverse", "grade_involution", "clifford_conjugation", "grade_projection"]
_VALID_UNARY_OPS = {"identity", "reverse", "grade_involution", "clifford_conjugation", "grade_projection"}


@dataclass(frozen=True)
class UnaryRequest:
    """Fully resolved request for one unary planned operation."""

    spec: AlgebraSpec
    op: GradeUnaryOp
    input: TensorContract
    output: TensorContract
    dtype: torch.dtype
    device: torch.device

    @property
    def input_layout(self) -> GradeLayout:
        """Return the resolved input layout."""
        return self.input.layout

    @property
    def output_layout(self) -> GradeLayout:
        """Return the resolved output layout."""
        return self.output.layout

    @property
    def input_uses_compact_storage(self) -> bool:
        """Return whether the input tensor is already compact."""
        return self.input.uses_compact_storage

    @property
    def input_grades(self) -> tuple[int, ...]:
        """Return the grades accepted by the unary operation."""
        return self.input_layout.grades

    @property
    def output_grades(self) -> tuple[int, ...]:
        """Return the grades emitted by the unary operation."""
        return self.output_layout.grades

    @property
    def cache_key(self) -> tuple[object, ...]:
        """Return a stable key for executor caching."""
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
        """Return the full algebra lane dimension."""
        return self.spec.dim

    @property
    def output_dim(self) -> int:
        """Return the compact output lane count."""
        return self.output_layout.dim


def build_unary_request(
    spec: AlgebraSpec,
    values: torch.Tensor,
    *,
    op: str,
    input_grades=None,
    output_grades=None,
    input_layout: Optional[GradeLayout] = None,
    output_layout: Optional[GradeLayout] = None,
    input_storage: LaneStorage | str | None = None,
    output_storage: LaneStorage | str = LaneStorage.COMPACT,
) -> UnaryRequest:
    """Resolve caller input into a static unary request."""
    op = normalize_unary_op(op)
    if op == "grade_projection" and input_grades is None and input_layout is None and input_storage is None:
        if output_layout is not None:
            input_layout = output_layout
        elif output_grades is not None:
            input_grades = output_grades
    input_contract = infer_contract(
        spec,
        values,
        grades=input_grades,
        layout=input_layout,
        storage=input_storage,
        side="input",
    )
    output_layout = resolve_unary_output_layout(
        spec,
        op=op,
        input_layout=input_contract.layout,
        output_grades=output_grades,
        output_layout=output_layout,
    )
    output_contract = TensorContract(spec=spec, layout=output_layout, storage=output_storage)
    return UnaryRequest(
        spec=spec,
        op=op,
        input=input_contract,
        output=output_contract,
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
