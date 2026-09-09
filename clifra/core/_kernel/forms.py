# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Algebraic scalar forms that may be indefinite or degenerate."""

from __future__ import annotations

from typing import Iterable, Optional

import torch

from clifra.core._kernel.basis import operation_coefficient, reverse_sign
from clifra.core._kernel.contracts import resolve_contract
from clifra.core.layout import AlgebraSpec, GradeLayout


def conjugate_scalar_form_signs(
    algebra,
    layout: Optional[GradeLayout] = None,
    *,
    grades: Optional[Iterable[int]] = None,
    device=None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Return signs for ``<bar(A) B>_0`` in a compact layout."""
    resolved = resolve_contract(algebra, layout=layout, grades=grades).layout
    if device is None:
        device = getattr(algebra, "device", None)
    if dtype is None:
        dtype = getattr(algebra, "dtype", torch.float32)
    values = [_conjugate_scalar_form_sign_for_index(resolved.spec, index) for index in resolved.basis_indices]
    return torch.tensor(values, dtype=dtype, device=device)


def _conjugate_scalar_form_sign_for_index(spec: AlgebraSpec, index: int) -> float:
    grade = int(index).bit_count()
    grade_sign = -1.0 if grade % 2 else 1.0
    metric_sign = operation_coefficient(index, index, spec.p, spec.q, spec.r, "geometric_product")
    return grade_sign * reverse_sign(index) * metric_sign
