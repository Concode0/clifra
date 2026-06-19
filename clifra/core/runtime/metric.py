# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Metric-facing helpers with explicit positive and signed forms.

Lane geometry is the positive-definite coefficient space used for optimization.
Algebraic scalar forms preserve Clifford-signature information and may be
indefinite or degenerate.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch

from clifra.core.foundation.layout import GradeLayout
from clifra.core.foundation.module import AlgebraLike
from clifra.core.runtime.energy import (
    lane_distance,
    lane_energy,
    lane_grade_distribution,
    lane_grade_energy,
    lane_grade_norms,
    lane_inner_product,
    lane_norm,
)
from clifra.core.runtime.forms import (
    conjugate_form_distance_like,
    conjugate_form_magnitude,
    conjugate_grade_magnitude_spectrum,
    conjugate_scalar_form,
    conjugate_scalar_form_signs,
    signature_norm_squared,
    signature_trace_form,
)
from clifra.core.runtime.tensors import LaneStorage, compact_values


def scalar_product(
    algebra: AlgebraLike,
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    left_layout: Optional[GradeLayout] = None,
    right_layout: Optional[GradeLayout] = None,
    left_grades: Optional[Iterable[int]] = None,
    right_grades: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Return the Clifford scalar product ``<A B>_0``.

    This is an algebraic projection, not the positive-definite optimizer
    geometry. Use ``lane_inner_product`` for Euclidean coefficient geometry.
    """
    return algebra.projected_geometric_product(
        A,
        B,
        left_layout=left_layout,
        right_layout=right_layout,
        left_grades=left_grades,
        right_grades=right_grades,
        output_grades=(0,),
        output_storage=LaneStorage.COMPACT,
    )


def signature_magnitude(algebra: AlgebraLike, values: torch.Tensor) -> torch.Tensor:
    """Return ``sqrt(abs(<~A A>_0))`` for the signed signature form."""
    return torch.sqrt(torch.abs(signature_norm_squared(algebra, values)))


def signature_distance_like(algebra: AlgebraLike, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Return ``sqrt(abs(<~(A-B) (A-B)>_0))``; not a metric in mixed signatures."""
    return signature_magnitude(algebra, A - B)


def grade_purity(
    algebra: AlgebraLike,
    values: torch.Tensor,
    grade: int,
    *,
    layout: Optional[GradeLayout] = None,
    grades: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Return the fraction of positive lane energy carried by one grade."""
    compact, resolved = compact_values(algebra, values, layout=layout, grades=grades)
    total = compact.pow(2).sum(dim=-1).clamp_min(float(algebra.eps))
    positions = resolved.positions_for_grades((int(grade),), device=compact.device)
    selected = torch.index_select(compact, -1, positions)
    return selected.pow(2).sum(dim=-1) / total


def mean_grade(
    algebra: AlgebraLike,
    values: torch.Tensor,
    *,
    layout: Optional[GradeLayout] = None,
    grades: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Return the lane-energy weighted mean grade."""
    energy = lane_grade_energy(algebra, values, layout=layout, grades=grades)
    weights = torch.arange(algebra.n + 1, device=energy.device, dtype=energy.dtype)
    total = energy.sum(dim=-1).clamp_min(float(algebra.eps))
    return (energy * weights).sum(dim=-1) / total


def clifford_conjugate(algebra: AlgebraLike, values: torch.Tensor) -> torch.Tensor:
    """Return Clifford conjugation, i.e. grade involution after reversion."""
    return algebra.clifford_conjugation(values)


__all__ = [
    "scalar_product",
    "signature_magnitude",
    "signature_distance_like",
    "grade_purity",
    "mean_grade",
    "clifford_conjugate",
    "lane_inner_product",
    "lane_energy",
    "lane_norm",
    "lane_distance",
    "lane_grade_energy",
    "lane_grade_norms",
    "lane_grade_distribution",
    "conjugate_scalar_form_signs",
    "conjugate_scalar_form",
    "conjugate_form_magnitude",
    "conjugate_form_distance_like",
    "conjugate_grade_magnitude_spectrum",
    "signature_trace_form",
    "signature_norm_squared",
]
