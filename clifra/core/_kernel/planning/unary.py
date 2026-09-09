# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Static unary requests and plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from clifra.core._kernel.basis import unary_sign
from clifra.core._kernel.contracts import _check_contract_spec
from clifra.core.layout import AlgebraSpec, GradeLayout
from clifra.core.tensors import TensorContract

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

    @classmethod
    def compact(
        cls,
        spec: AlgebraSpec,
        *,
        op: str,
        input_layout: GradeLayout,
        output_layout: GradeLayout,
        dtype: torch.dtype,
        device,
    ) -> "UnaryRequest":
        """Build a normalized request for compact planned execution."""
        return cls(
            spec=spec,
            op=normalize_unary_op(op),
            input=TensorContract.compact(input_layout),
            output=TensorContract.compact(output_layout),
            dtype=dtype,
            device=torch.device(device),
        )

    def __post_init__(self) -> None:
        self.validate(self.spec)

    def validate(self, spec: AlgebraSpec) -> None:
        """Validate all contracts against a receiving algebra specification."""
        if self.spec != spec:
            raise ValueError(f"request signature {self.spec} does not match algebra signature {spec}")
        _check_contract_spec(spec, self.input, "input_layout")
        _check_contract_spec(spec, self.output, "output_layout")

    @property
    def input_layout(self) -> GradeLayout:
        """Return the resolved input layout."""
        return self.input.layout

    @property
    def output_layout(self) -> GradeLayout:
        """Return the resolved output layout."""
        return self.output.layout

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
        self.input_contract = TensorContract.compact(self.input_layout)
        self.output_contract = TensorContract.compact(self.output_layout)
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
        signs.append(unary_sign(request.op, index))

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
    return normalized
