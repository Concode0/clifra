"""Internal utilities for the analysis toolkit."""

from __future__ import annotations

from typing import Iterable

import torch

from clifra.core.foundation.device import resolve_dtype
from clifra.core.runtime.algebra import CliffordAlgebra


def analysis_dtype(dtype=None) -> torch.dtype:
    """Resolve the floating-point dtype used by analysis routines."""
    return resolve_dtype(torch.float32 if dtype is None else dtype)


def as_analysis_tensor(data: torch.Tensor, *, device, dtype=None) -> torch.Tensor:
    """Move data to the requested analysis device and floating dtype."""
    resolved = analysis_dtype(dtype if dtype is not None else data.dtype)
    if not resolved.is_floating_point:
        resolved = torch.float32
    return data.to(device=device, dtype=resolved)


def full_grades(algebra) -> tuple[int, ...]:
    """Return all grades for explicit full-layout planned calls."""
    return tuple(range(int(algebra.n) + 1))


def is_dense_algebra(algebra) -> bool:
    """Return whether ``algebra`` owns dense Clifford kernels."""
    return isinstance(algebra, CliffordAlgebra)


def require_dense_algebra(algebra, feature: str) -> None:
    """Raise a clear error when a feature requires dense kernels."""
    if not is_dense_algebra(algebra):
        raise ValueError(f"{feature} requires CliffordAlgebra dense kernels; use a lower-dimensional dense algebra.")


def dense_analysis_supported(algebra, *, max_n: int) -> bool:
    """Return whether explicit dense-style analysis is allowed by policy."""
    return int(getattr(algebra, "n", 0)) <= int(max_n)


def declared_full_product_kwargs(algebra) -> dict[str, Iterable[int]]:
    """Return explicit full-grade metadata for planned dense-width products."""
    grades = full_grades(algebra)
    return {
        "left_grades": grades,
        "right_grades": grades,
        "output_grades": grades,
        "compact_output": True,
    }
