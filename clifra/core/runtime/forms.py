# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Algebraic scalar forms that may be indefinite or degenerate."""

from __future__ import annotations

from typing import Iterable, Optional

import torch

from clifra.core.foundation.basis import operation_coefficient, reverse_sign
from clifra.core.foundation.layout import AlgebraSpec, GradeLayout
from clifra.core.runtime.tensors import compact_pair_values, compact_values, resolve_layout


def conjugate_scalar_form_signs(
    algebra,
    layout: Optional[GradeLayout] = None,
    *,
    grades: Optional[Iterable[int]] = None,
    device=None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Return signs for ``<bar(A) B>_0`` in a compact layout."""
    resolved = resolve_layout(algebra, layout=layout, grades=grades)
    if device is None:
        device = getattr(algebra, "device", None)
    if dtype is None:
        dtype = getattr(algebra, "dtype", torch.float32)
    values = [_conjugate_scalar_form_sign_for_index(resolved.spec, index) for index in resolved.basis_indices]
    return torch.tensor(values, dtype=dtype, device=device)


def conjugate_scalar_form(
    algebra,
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    layout: Optional[GradeLayout] = None,
    left_layout: Optional[GradeLayout] = None,
    right_layout: Optional[GradeLayout] = None,
    grades: Optional[Iterable[int]] = None,
    left_grades: Optional[Iterable[int]] = None,
    right_grades: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Return the signed Clifford-conjugation scalar form ``<bar(A) B>_0``."""
    A_values, B_values, resolved = compact_pair_values(
        algebra,
        A,
        B,
        layout=layout,
        left_layout=left_layout,
        right_layout=right_layout,
        grades=grades,
        left_grades=left_grades,
        right_grades=right_grades,
    )
    dtype = torch.promote_types(A_values.dtype, B_values.dtype)
    signs = conjugate_scalar_form_signs(algebra, layout=resolved, device=A_values.device, dtype=dtype)
    return (signs * A_values * B_values).sum(dim=-1, keepdim=True)


def conjugate_form_magnitude(
    algebra,
    values: torch.Tensor,
    *,
    layout: Optional[GradeLayout] = None,
    grades: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Return ``sqrt(abs(<bar(A) A>_0))`` for the signed conjugate form."""
    compact, resolved = compact_values(algebra, values, layout=layout, grades=grades)
    sq = conjugate_scalar_form(algebra, compact, compact, layout=resolved)
    return torch.sqrt(torch.abs(sq))


def conjugate_form_distance_like(
    algebra,
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    layout: Optional[GradeLayout] = None,
    left_layout: Optional[GradeLayout] = None,
    right_layout: Optional[GradeLayout] = None,
    grades: Optional[Iterable[int]] = None,
    left_grades: Optional[Iterable[int]] = None,
    right_grades: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Return ``sqrt(abs(<bar(A-B) (A-B)>_0))``; not a metric in general."""
    A_values, B_values, resolved = compact_pair_values(
        algebra,
        A,
        B,
        layout=layout,
        left_layout=left_layout,
        right_layout=right_layout,
        grades=grades,
        left_grades=left_grades,
        right_grades=right_grades,
    )
    diff = A_values - B_values
    return conjugate_form_magnitude(algebra, diff, layout=resolved)


def conjugate_grade_magnitude_spectrum(
    algebra,
    values: torch.Tensor,
    *,
    layout: Optional[GradeLayout] = None,
    grades: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Return per-grade absolute signed conjugate-form magnitudes."""
    compact, resolved = compact_values(algebra, values, layout=layout, grades=grades)
    signs = conjugate_scalar_form_signs(algebra, layout=resolved, device=compact.device, dtype=compact.dtype)
    signed = signs * compact * compact
    flat = signed.reshape(-1, resolved.dim)
    grade_ids = resolved.grade_indices_tensor(device=compact.device).unsqueeze(0).expand_as(flat)
    spectrum = signed.new_zeros(flat.shape[0], algebra.n + 1)
    spectrum.scatter_add_(1, grade_ids, flat)
    return spectrum.reshape(*compact.shape[:-1], algebra.n + 1).abs()


def signature_trace_form(algebra, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Return the signed Clifford scalar product ``<~A B>_0``."""
    A_rev = algebra.reverse(A)
    prod = algebra.geometric_product(A_rev, B)
    return prod[..., 0:1]


def signature_norm_squared(algebra, A: torch.Tensor) -> torch.Tensor:
    """Return the raw signed signature norm squared ``<A~A>_0``."""
    return signature_trace_form(algebra, A, A)


def _conjugate_scalar_form_sign_for_index(spec: AlgebraSpec, index: int) -> float:
    grade = int(index).bit_count()
    grade_sign = -1.0 if grade % 2 else 1.0
    metric_sign = operation_coefficient(index, index, spec.p, spec.q, spec.r, "gp")
    return grade_sign * reverse_sign(index) * metric_sign
