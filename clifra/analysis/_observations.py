# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Input handling shared by declared-algebra diagnostics."""

import torch

from clifra.core.algebra import AlgebraContext


def canonical_observations(data: torch.Tensor, algebra: AlgebraContext) -> torch.Tensor:
    """Validate canonical [N, algebra.dim] floats and use the algebra's dtype/device."""
    if not isinstance(algebra, AlgebraContext):
        raise TypeError("analysis requires an explicit AlgebraContext")
    if data.ndim != 2 or data.shape[-1] != algebra.dim:
        raise ValueError(f"expected canonical [N, {algebra.dim}] data")
    if data.shape[0] == 0:
        raise ValueError("analysis requires nonempty observations")
    if not data.dtype.is_floating_point:
        raise TypeError("analysis requires floating-point observations")
    return data.to(device=algebra.device, dtype=algebra.dtype)
