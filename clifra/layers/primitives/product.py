# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Module wrappers around declared geometric algebra products."""

from __future__ import annotations

from typing import Iterable, Optional

import torch

from clifra.core.foundation.basis import normalize_grades
from clifra.core.foundation.layout import GradeLayout
from clifra.core.foundation.module import CliffordModule
from clifra.core.runtime.tensors import LaneStorage, check_layout_spec
from clifra.functional.products import canonical_product_op, product


def _normalize_optional_grades(grades: Optional[Iterable[int]], n: int, *, name: str) -> Optional[tuple[int, ...]]:
    if grades is None:
        return None
    if isinstance(grades, int):
        grades = (grades,)
    return normalize_grades(grades, n, name=name)


class ProductLayer(CliffordModule):
    """Apply a geometric algebra product inside ``nn.Module`` graphs.

    Grade and layout declarations define the tensor lane contract. If a caller
    declares any input or output layout, the layer returns the planned output
    layout lanes. If no layout is declared, it uses the algebra's direct full
    multivector product.
    """

    def __init__(
        self,
        algebra,
        *,
        op: str = "gp",
        left_grades: Optional[Iterable[int]] = None,
        right_grades: Optional[Iterable[int]] = None,
        output_grades: Optional[Iterable[int]] = None,
        left_layout: GradeLayout = None,
        right_layout: GradeLayout = None,
        output_layout: GradeLayout = None,
        pairwise: bool = False,
    ):
        """Initialize a product layer.

        Args:
            algebra: Algebra host.
            op: Product route: ``"gp"``, ``"wedge"``,
                ``"symmetric_product"``, ``"commutator_product"``,
                ``"anti_commutator_product"``, ``"left_contraction"``, or
                ``"right_contraction"``.
            left_grades: Declared input grades for the left operand.
            right_grades: Declared input grades for the right operand.
            output_grades: Optional output grade projection.
            left_layout: Optional explicit left operand layout.
            right_layout: Optional explicit right operand layout.
            output_layout: Optional explicit output layout.
            pairwise: Treat the penultimate dimension of each declared-layout operand
                as independent left/right item axes.
        """
        super().__init__(algebra)
        self.op = canonical_product_op(op)
        self.left_grades = _normalize_optional_grades(left_grades, algebra.n, name="left_grades")
        self.right_grades = _normalize_optional_grades(right_grades, algebra.n, name="right_grades")
        self.output_grades = _normalize_optional_grades(output_grades, algebra.n, name="output_grades")
        self.left_layout = _validate_optional_layout(algebra, left_layout, self.left_grades, "left")
        self.right_layout = _validate_optional_layout(algebra, right_layout, self.right_grades, "right")
        self.output_layout = _validate_optional_layout(algebra, output_layout, self.output_grades, "output")
        self.pairwise = bool(pairwise)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Apply the configured product to ``left`` and ``right``."""
        kwargs = self._product_kwargs()
        return product(self.algebra, left, right, op=self.op, **kwargs)

    def _product_kwargs(self) -> dict:
        kwargs = {}
        if self.left_grades is not None:
            kwargs["left_grades"] = self.left_grades
        if self.right_grades is not None:
            kwargs["right_grades"] = self.right_grades
        if self.output_grades is not None:
            kwargs["output_grades"] = self.output_grades
        if self.left_layout is not None:
            kwargs["left_layout"] = self.left_layout
        if self.right_layout is not None:
            kwargs["right_layout"] = self.right_layout
        if self.output_layout is not None:
            kwargs["output_layout"] = self.output_layout
        if kwargs:
            kwargs["output_storage"] = LaneStorage.COMPACT
        if self.pairwise:
            kwargs["pairwise"] = True
        return kwargs

    def extra_repr(self) -> str:
        """Return constructor fields shown by ``nn.Module`` repr."""
        parts = [f"op={self.op!r}"]
        if self.left_grades is not None:
            parts.append(f"left_grades={self.left_grades}")
        if self.right_grades is not None:
            parts.append(f"right_grades={self.right_grades}")
        if self.output_grades is not None:
            parts.append(f"output_grades={self.output_grades}")
        if self.left_layout is not None:
            parts.append(f"left_layout={self.left_layout.grades}")
        if self.right_layout is not None:
            parts.append(f"right_layout={self.right_layout.grades}")
        if self.output_layout is not None:
            parts.append(f"output_layout={self.output_layout.grades}")
        if self.pairwise:
            parts.append("pairwise=True")
        return ", ".join(parts)


class GeometricProductLayer(ProductLayer):
    """Layer form of the geometric product."""

    def __init__(self, algebra, **kwargs):
        super().__init__(algebra, op="gp", **kwargs)


class WedgeLayer(ProductLayer):
    """Layer form of the exterior product."""

    def __init__(self, algebra, **kwargs):
        super().__init__(algebra, op="wedge", **kwargs)


class SymmetricProductLayer(ProductLayer):
    """Layer form of the parity-selected symmetric product route."""

    def __init__(self, algebra, **kwargs):
        super().__init__(algebra, op="symmetric_product", **kwargs)


class CommutatorProductLayer(ProductLayer):
    """Layer form of the commutator product."""

    def __init__(self, algebra, **kwargs):
        super().__init__(algebra, op="commutator_product", **kwargs)


class AntiCommutatorProductLayer(ProductLayer):
    """Layer form of the anti-commutator product."""

    def __init__(self, algebra, **kwargs):
        super().__init__(algebra, op="anti_commutator_product", **kwargs)


class LeftContractionLayer(ProductLayer):
    """Layer form of the left contraction."""

    def __init__(self, algebra, **kwargs):
        super().__init__(algebra, op="left_contraction", **kwargs)


class RightContractionLayer(ProductLayer):
    """Layer form of the right contraction."""

    def __init__(self, algebra, **kwargs):
        super().__init__(algebra, op="right_contraction", **kwargs)


InnerProductLayer = SymmetricProductLayer
CommutatorLayer = CommutatorProductLayer
AntiCommutatorLayer = AntiCommutatorProductLayer


def _validate_optional_layout(algebra, layout: GradeLayout | None, grades, side: str) -> GradeLayout | None:
    if layout is None:
        return None
    check_layout_spec(algebra.planner.spec, layout, f"{side}_layout")
    if grades is not None and layout.grades != grades:
        raise ValueError(f"{side}_layout and {side}_grades disagree")
    return layout
