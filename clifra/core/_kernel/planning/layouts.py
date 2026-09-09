# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Explicit product requests for static grade planning."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from clifra.core._kernel.basis import (
    GradeProductOp,
    normalize_grade_product_op,
)
from clifra.core._kernel.contracts import (
    _check_contract_spec,
)
from clifra.core.layout import AlgebraSpec, GradeLayout
from clifra.core.tensors import TensorContract

__all__ = [
    "ProductRequest",
]


@dataclass(frozen=True)
class ProductRequest:
    """Fully resolved static request for one bilinear product.

    Inputs and output carry explicit tensor contracts before lowering.
    The algebra boundary resolves omitted output grades and storage adaptation.
    """

    spec: AlgebraSpec
    op: GradeProductOp
    left: TensorContract
    right: TensorContract
    output: TensorContract
    dtype: torch.dtype
    device: torch.device

    @classmethod
    def compact(
        cls,
        spec: AlgebraSpec,
        *,
        op: str,
        left_layout: GradeLayout,
        right_layout: GradeLayout,
        output_layout: GradeLayout,
        dtype: torch.dtype,
        device,
    ) -> "ProductRequest":
        """Build a normalized request for compact planned execution."""
        return cls(
            spec=spec,
            op=normalize_grade_product_op(op),
            left=TensorContract.compact(left_layout),
            right=TensorContract.compact(right_layout),
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
        _check_contract_spec(spec, self.left, "left_layout")
        _check_contract_spec(spec, self.right, "right_layout")
        _check_contract_spec(spec, self.output, "output_layout")

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
