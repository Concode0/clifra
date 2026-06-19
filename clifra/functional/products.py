# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Stateless geometric algebra product helpers.

Functional helpers use the final axis as the Clifford lane axis. Full-lane
multivectors are ``[..., D]``, where ``D = 2 ** algebra.n``. Active layout
values are ``[..., L]``, where ``L`` is the compact lane count for the declared
layout. Leading axes ``...`` are ordinary PyTorch batch, item, or channel axes.

These wrappers keep model code concise while preserving the algebra host as the
single execution authority. Full-lane, compact-lane, and pairwise planned
executors all flow through the same public calls.
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
    "left_contraction": ("left_contraction", "left_contraction"),
    "right_contraction": ("right_contraction", "right_contraction"),
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
        left: Left operand with full-lane shape ``[..., D]`` or active declared
            shape ``[..., L_left]``.
        right: Right operand with leading axes broadcast-compatible with
            ``left`` and lane shape ``[..., D]`` or ``[..., L_right]``.
        op: ``"gp"``, ``"wedge"``, ``"inner"``, ``"commutator"``,
            ``"anti_commutator"``, ``"left_contraction"``, or
            ``"right_contraction"``.
        **kwargs: Optional grade/layout declarations accepted by
            ``algebra.projected_product``.

    Returns:
        Product values with full-lane shape ``[..., D]`` or declared compact shape
        ``[..., L_out]``.
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
    """Apply a declared grade-restricted product through the planner.

    Operands use compact lane shapes ``[..., L_left]`` and ``[..., L_right]``
    when compact layouts are declared; the output uses ``[..., L_out]``.
    """
    return product(algebra, left, right, op=op, **kwargs)


def geometric_product(algebra, left: torch.Tensor, right: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Apply the geometric product to full ``[..., D]`` or declared compact lanes."""
    return product(algebra, left, right, op="gp", **kwargs)


def wedge(algebra, left: torch.Tensor, right: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Apply the exterior product to full ``[..., D]`` or declared compact lanes."""
    return product(algebra, left, right, op="wedge", **kwargs)


def inner_product(algebra, left: torch.Tensor, right: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Apply the inner product to full ``[..., D]`` or declared compact lanes."""
    return product(algebra, left, right, op="inner", **kwargs)


def commutator(algebra, left: torch.Tensor, right: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Apply the commutator product to full ``[..., D]`` or declared compact lanes."""
    return product(algebra, left, right, op="commutator", **kwargs)


def anti_commutator(algebra, left: torch.Tensor, right: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Apply the anti-commutator product to full ``[..., D]`` or declared compact lanes."""
    return product(algebra, left, right, op="anti_commutator", **kwargs)


def left_contraction(algebra, left: torch.Tensor, right: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Apply left contraction to full ``[..., D]`` or declared compact lanes."""
    return product(algebra, left, right, op="left_contraction", **kwargs)


def right_contraction(algebra, left: torch.Tensor, right: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Apply right contraction to full ``[..., D]`` or declared compact lanes."""
    return product(algebra, left, right, op="right_contraction", **kwargs)


def grade_projection(algebra, values: torch.Tensor, grade: int, **kwargs: Any) -> torch.Tensor:
    """Project multivectors with shape ``[..., D]`` or ``[..., L]`` to one grade."""
    return algebra.grade_projection(values, grade, **kwargs)


def reverse(algebra, values: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Apply reversion to values with shape ``[..., D]`` or ``[..., L]``."""
    return algebra.reverse(values, **kwargs)


def grade_involution(algebra, values: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Apply grade involution to values with shape ``[..., D]`` or ``[..., L]``."""
    return algebra.grade_involution(values, **kwargs)


def clifford_conjugation(algebra, values: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Apply Clifford conjugation to values with shape ``[..., D]`` or ``[..., L]``."""
    return algebra.clifford_conjugation(values, **kwargs)


def dual(algebra, values: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Apply the Hodge dual to full-lane or declared compact values."""
    return algebra.dual(values, **kwargs)


def norm_sq(algebra, values: torch.Tensor) -> torch.Tensor:
    """Return the algebraic squared norm of values with shape ``[..., D]`` or ``[..., L]``."""
    return algebra.norm_sq(values)


def embed_vector(algebra, vectors: torch.Tensor) -> torch.Tensor:
    """Embed coordinate vectors with shape ``[..., algebra.n]`` into full-lane multivectors."""
    return algebra.embed_vector(vectors)
