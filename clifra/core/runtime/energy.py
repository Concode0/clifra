# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Positive-definite coefficient-lane geometry."""

from __future__ import annotations

from typing import Optional

import torch

from clifra.core.foundation.layout import GradeLayout
from clifra.core.runtime.tensors import compact_pair_values, compact_values


def lane_dot_product(algebra, A: torch.Tensor, B: torch.Tensor, *, layout: Optional[GradeLayout] = None, grades=None) -> torch.Tensor:
    """Return the positive-definite coefficient dot product over compact lanes."""
    A_values, B_values, _ = compact_pair_values(algebra, A, B, layout=layout, grades=grades)
    return (A_values * B_values).sum(dim=-1, keepdim=True)


def lane_energy(algebra, values: torch.Tensor, *, layout: Optional[GradeLayout] = None, grades=None) -> torch.Tensor:
    """Return positive-definite coefficient energy ``sum_i x_i**2``."""
    compact, _ = compact_values(algebra, values, layout=layout, grades=grades)
    return compact.pow(2).sum(dim=-1, keepdim=True)


def lane_norm(algebra, values: torch.Tensor, *, layout: Optional[GradeLayout] = None, grades=None) -> torch.Tensor:
    """Return positive-definite coefficient norm."""
    energy = lane_energy(algebra, values, layout=layout, grades=grades)
    return energy.sqrt()


def lane_distance(algebra, A: torch.Tensor, B: torch.Tensor, *, layout: Optional[GradeLayout] = None, grades=None) -> torch.Tensor:
    """Return positive-definite coefficient distance."""
    A_values, B_values, _ = compact_pair_values(algebra, A, B, layout=layout, grades=grades)
    diff = A_values - B_values
    energy = diff.pow(2).sum(dim=-1, keepdim=True)
    return energy.sqrt()


def lane_grade_energy(algebra, values: torch.Tensor, *, layout: Optional[GradeLayout] = None, grades=None) -> torch.Tensor:
    """Return per-grade positive coefficient energy."""
    compact, resolved = compact_values(algebra, values, layout=layout, grades=grades)
    flat = compact.pow(2).reshape(-1, resolved.dim)
    grade_ids = resolved.grade_indices_tensor(device=compact.device).unsqueeze(0).expand_as(flat)
    result = compact.new_zeros(flat.shape[0], resolved.spec.n + 1)
    result.scatter_add_(1, grade_ids, flat)
    return result.reshape(*compact.shape[:-1], resolved.spec.n + 1)


def lane_grade_norms(algebra, values: torch.Tensor, *, layout: Optional[GradeLayout] = None, grades=None) -> torch.Tensor:
    """Return per-grade positive coefficient norms."""
    energy = lane_grade_energy(algebra, values, layout=layout, grades=grades)
    return energy.sqrt()


def lane_grade_distribution(
    algebra,
    values: torch.Tensor,
    *,
    layout: Optional[GradeLayout] = None,
    grades=None,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Return normalized per-grade lane-energy distribution."""
    energy = lane_grade_energy(algebra, values, layout=layout, grades=grades)
    return energy / (energy.sum(dim=-1, keepdim=True) + float(eps))
