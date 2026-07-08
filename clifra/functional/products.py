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

from clifra.core.legacy import canonical_product_alias, product_method_entry


def canonical_product_op(op: str) -> str:
    """Return the preferred public product operation name for a supported alias."""
    return canonical_product_alias(op)


def _resolve_product_op(op: str) -> tuple[str, str]:
    return product_method_entry(op)


def product(algebra, left: torch.Tensor, right: torch.Tensor, *, op: str = "gp", **kwargs: Any) -> torch.Tensor:
    """Apply a binary geometric algebra product.

    Args:
        algebra: Algebra host.
        left: Left operand with full-lane shape ``[..., D]`` or active declared
            shape ``[..., L_left]``.
        right: Right operand with leading axes broadcast-compatible with
            ``left`` and lane shape ``[..., D]`` or ``[..., L_right]``.
        op: ``"gp"``, ``"wedge"``, ``"symmetric_product"``,
            ``"commutator_product"``, ``"anti_commutator_product"``,
            ``"left_contraction"``, or ``"right_contraction"``.
        **kwargs: Optional grade/layout declarations accepted by
            ``algebra.projected_product``.

    Returns:
        Product values with full-lane shape ``[..., D]`` or declared compact shape
        ``[..., L_out]``.
    """
    _, method_name = _resolve_product_op(op)
    if kwargs:
        return algebra.projected_product(left, right, op=canonical_product_alias(op), **kwargs)
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
    """Legacy alias for ``symmetric_product``."""
    return symmetric_product(algebra, left, right, **kwargs)


def symmetric_product(algebra, left: torch.Tensor, right: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Apply the parity-selected symmetric product route to full or declared compact lanes."""
    return product(algebra, left, right, op="symmetric_product", **kwargs)


def commutator(algebra, left: torch.Tensor, right: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Legacy alias for ``commutator_product``."""
    return commutator_product(algebra, left, right, **kwargs)


def commutator_product(algebra, left: torch.Tensor, right: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Apply the unnormalized commutator product to full or declared compact lanes."""
    return product(algebra, left, right, op="commutator_product", **kwargs)


def anti_commutator(algebra, left: torch.Tensor, right: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Legacy alias for ``anti_commutator_product``."""
    return anti_commutator_product(algebra, left, right, **kwargs)


def anti_commutator_product(algebra, left: torch.Tensor, right: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Apply the unnormalized anti-commutator product to full or declared compact lanes."""
    return product(algebra, left, right, op="anti_commutator_product", **kwargs)


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
    """Legacy alias for ``pseudoscalar_product``."""
    return pseudoscalar_product(algebra, values, **kwargs)


def pseudoscalar_product(algebra, values: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Apply right multiplication by the unit pseudoscalar."""
    return algebra.pseudoscalar_product(values, **kwargs)


def norm_sq(algebra, values: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Legacy alias for ``signature_norm_squared``."""
    return signature_norm_squared(algebra, values, **kwargs)


def signature_norm_squared(algebra, values: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Return the signed Clifford signature norm squared."""
    return algebra.signature_norm_squared(values, **kwargs)


def bivector_exp(algebra, values: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Exponentiate a declared bivector."""
    return algebra.bivector_exp(values, **kwargs)


def embed_vector(algebra, vectors: torch.Tensor) -> torch.Tensor:
    """Embed coordinate vectors with shape ``[..., algebra.n]`` into full-lane multivectors."""
    return algebra.embed_vector(vectors)


def blade_inverse(algebra, blade: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Return the regularized inverse of a declared blade."""
    return algebra.blade_inverse(blade, **kwargs)


def blade_project(algebra, values: torch.Tensor, blade: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Project values onto a declared blade subspace."""
    return algebra.blade_project(values, blade, **kwargs)


def blade_reject(algebra, values: torch.Tensor, blade: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Reject values from a declared blade subspace."""
    return algebra.blade_reject(values, blade, **kwargs)


def reflect(algebra, values: torch.Tensor, normal: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Reflect values through a declared vector normal."""
    return algebra.reflect(values, normal, **kwargs)


def versor_product(algebra, versor: torch.Tensor, values: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Apply ``grade_involution(versor) * values * inverse(versor)``."""
    return algebra.versor_product(versor, values, **kwargs)


def planned_linear_action(algebra, values: torch.Tensor, matrix: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Apply a declared vector-space linear action."""
    return algebra.planned_linear_action(values, matrix, **kwargs)


def sandwich_action_matrices(algebra, left: torch.Tensor, right: torch.Tensor = None, **kwargs: Any) -> torch.Tensor:
    """Return full-layout sandwich action matrices."""
    return algebra.sandwich_action_matrices(left, right, **kwargs)


def sandwich_product(
    algebra,
    left: torch.Tensor,
    values: torch.Tensor,
    right: torch.Tensor = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Apply one full-layout sandwich action per leading batch item."""
    return algebra.sandwich_product(left, values, right, **kwargs)


def per_channel_sandwich(
    algebra,
    left: torch.Tensor,
    values: torch.Tensor,
    right: torch.Tensor = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Apply one full-layout sandwich action per channel."""
    return algebra.per_channel_sandwich(left, values, right, **kwargs)


def multi_rotor_sandwich(
    algebra,
    left: torch.Tensor,
    values: torch.Tensor,
    right: torch.Tensor = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Apply every full-layout sandwich action to every input channel."""
    return algebra.multi_rotor_sandwich(left, values, right, **kwargs)


def versor_action(algebra, values: torch.Tensor, weights: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Execute a planned grade-1 or grade-2 versor action."""
    return algebra.versor_action(values, weights, **kwargs)


def multi_versor_action(
    algebra,
    values: torch.Tensor,
    weights: torch.Tensor,
    mix: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    """Execute a planned weighted grade-1 or grade-2 versor action."""
    return algebra.multi_versor_action(values, weights, mix, **kwargs)


def paired_bivector_action(
    algebra,
    values: torch.Tensor,
    left_weights: torch.Tensor,
    right_weights: torch.Tensor,
    channel_to_pair: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    """Execute a planned independent left/right bivector rotor action."""
    return algebra.paired_bivector_action(values, left_weights, right_weights, channel_to_pair, **kwargs)


def grade_norms(algebra, values: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Return per-grade coefficient norms for declared-layout values."""
    return algebra.grade_norms(values, **kwargs)
