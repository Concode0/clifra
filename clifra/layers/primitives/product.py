"""Module wrappers around declared geometric algebra products."""

from __future__ import annotations

from typing import Iterable, Optional

import torch

from clifra.core.foundation.basis import normalize_grades
from clifra.core.foundation.module import CliffordModule
from clifra.functional.products import canonical_product_op, product


def _normalize_optional_grades(grades: Optional[Iterable[int]], n: int, *, name: str) -> Optional[tuple[int, ...]]:
    if grades is None:
        return None
    if isinstance(grades, int):
        grades = (grades,)
    return normalize_grades(grades, n, name=name)


class ProductLayer(CliffordModule):
    """Apply a geometric algebra product inside ``nn.Module`` graphs.

    The layer is intentionally thin: grade declarations and compact/pairwise
    behavior are forwarded to ``algebra.projected_product`` when supplied,
    while the dense no-declaration path uses the algebra's direct kernels.
    """

    def __init__(
        self,
        algebra,
        *,
        op: str = "gp",
        left_grades: Optional[Iterable[int]] = None,
        right_grades: Optional[Iterable[int]] = None,
        output_grades: Optional[Iterable[int]] = None,
        left_compact: bool = False,
        right_compact: bool = False,
        compact_output: bool = False,
        pairwise: bool = False,
    ):
        """Initialize a product layer.

        Args:
            algebra: Dense ``CliffordAlgebra`` or planned ``AlgebraContext``.
            op: Product route: ``"gp"``, ``"wedge"``, ``"inner"``,
                ``"commutator"``, or ``"anti_commutator"``.
            left_grades: Declared input grades for the left operand.
            right_grades: Declared input grades for the right operand.
            output_grades: Optional output grade projection.
            left_compact: Whether the left operand is already compact.
            right_compact: Whether the right operand is already compact.
            compact_output: Return compact output instead of dense coefficients.
            pairwise: Treat the penultimate dimension of each compact operand
                as independent left/right item axes.
        """
        super().__init__(algebra)
        self.op = canonical_product_op(op)
        self.left_grades = _normalize_optional_grades(left_grades, algebra.n, name="left_grades")
        self.right_grades = _normalize_optional_grades(right_grades, algebra.n, name="right_grades")
        self.output_grades = _normalize_optional_grades(output_grades, algebra.n, name="output_grades")
        self.left_compact = bool(left_compact)
        self.right_compact = bool(right_compact)
        self.compact_output = bool(compact_output)
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
        if self.left_compact:
            kwargs["left_compact"] = True
        if self.right_compact:
            kwargs["right_compact"] = True
        if self.compact_output:
            kwargs["compact_output"] = True
        if self.pairwise:
            kwargs["pairwise"] = True
        return kwargs

    def extra_repr(self) -> str:
        parts = [f"op={self.op!r}"]
        if self.left_grades is not None:
            parts.append(f"left_grades={self.left_grades}")
        if self.right_grades is not None:
            parts.append(f"right_grades={self.right_grades}")
        if self.output_grades is not None:
            parts.append(f"output_grades={self.output_grades}")
        if self.compact_output:
            parts.append("compact_output=True")
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


class InnerProductLayer(ProductLayer):
    """Layer form of the inner product."""

    def __init__(self, algebra, **kwargs):
        super().__init__(algebra, op="inner", **kwargs)


class CommutatorLayer(ProductLayer):
    """Layer form of the commutator product."""

    def __init__(self, algebra, **kwargs):
        super().__init__(algebra, op="commutator", **kwargs)


class AntiCommutatorLayer(ProductLayer):
    """Layer form of the anti-commutator product."""

    def __init__(self, algebra, **kwargs):
        super().__init__(algebra, op="anti_commutator", **kwargs)
