"""Stateless geometric algebra product helpers.

These wrappers keep model code concise while preserving the algebra host as the
single execution authority. Dense kernels, compact planned kernels, and pairwise
planned kernels all flow through the same public calls.
"""

from __future__ import annotations

from typing import Any

import torch

_PRODUCT_METHODS = {
    "gp": ("gp", "geometric_product"),
    "geometric_product": ("gp", "geometric_product"),
    "wedge": ("wedge", "wedge"),
    "outer": ("wedge", "wedge"),
    "inner": ("inner", "inner_product"),
    "inner_product": ("inner", "inner_product"),
    "commutator": ("commutator", "commutator"),
    "anti_commutator": ("anti_commutator", "anti_commutator"),
    "anticommutator": ("anti_commutator", "anti_commutator"),
}


def canonical_product_op(op: str) -> str:
    """Return the planner op name for a supported product alias."""
    return _resolve_product_op(op)[0]


def _resolve_product_op(op: str) -> tuple[str, str]:
    op_key = op.lower()
    if op_key not in _PRODUCT_METHODS:
        supported = ", ".join(sorted(_PRODUCT_METHODS))
        raise ValueError(f"Unsupported product op {op!r}. Supported ops: {supported}")
    return _PRODUCT_METHODS[op_key]


def product(algebra, left: torch.Tensor, right: torch.Tensor, *, op: str = "gp", **kwargs: Any) -> torch.Tensor:
    """Apply a binary geometric algebra product.

    Args:
        algebra: Algebra host.
        left: Left operand.
        right: Right operand.
        op: ``"gp"``, ``"wedge"``, ``"inner"``, ``"commutator"``, or
            ``"anti_commutator"``.
        **kwargs: Optional grade/layout declarations accepted by
            ``algebra.projected_product``.

    Returns:
        Product values in dense or compact form according to ``kwargs``.
    """
    planned_op, method_name = _resolve_product_op(op)
    if kwargs:
        return algebra.projected_product(left, right, op=planned_op, **kwargs)
    return getattr(algebra, method_name)(left, right)


def projected_product(
    algebra,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    op: str = "gp",
    **kwargs: Any,
) -> torch.Tensor:
    """Apply a declared grade-restricted product through the planner."""
    return product(algebra, left, right, op=op, **kwargs)


def geometric_product(algebra, left: torch.Tensor, right: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Apply the geometric product."""
    return product(algebra, left, right, op="gp", **kwargs)


def wedge(algebra, left: torch.Tensor, right: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Apply the exterior product."""
    return product(algebra, left, right, op="wedge", **kwargs)


def inner_product(algebra, left: torch.Tensor, right: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Apply the inner product."""
    return product(algebra, left, right, op="inner", **kwargs)


def commutator(algebra, left: torch.Tensor, right: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Apply the commutator product."""
    return product(algebra, left, right, op="commutator", **kwargs)


def anti_commutator(algebra, left: torch.Tensor, right: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Apply the anti-commutator product."""
    return product(algebra, left, right, op="anti_commutator", **kwargs)


def grade_projection(algebra, values: torch.Tensor, grade: int, **kwargs: Any) -> torch.Tensor:
    """Project multivectors to one grade."""
    return algebra.grade_projection(values, grade, **kwargs)


def reverse(algebra, values: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Apply reversion."""
    return algebra.reverse(values, **kwargs)


def grade_involution(algebra, values: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Apply grade involution."""
    return algebra.grade_involution(values, **kwargs)


def clifford_conjugation(algebra, values: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Apply Clifford conjugation."""
    return algebra.clifford_conjugation(values, **kwargs)


def dual(algebra, values: torch.Tensor) -> torch.Tensor:
    """Apply the Hodge dual."""
    return algebra.dual(values)


def norm_sq(algebra, values: torch.Tensor) -> torch.Tensor:
    """Return the algebraic squared norm."""
    return algebra.norm_sq(values)


def embed_vector(algebra, vectors: torch.Tensor) -> torch.Tensor:
    """Embed coordinate vectors into the grade-1 subspace."""
    return algebra.embed_vector(vectors)
