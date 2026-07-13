# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Small helpers for layout-first adapter examples."""

from __future__ import annotations

import torch

from clifra.core.foundation.layout import GradeLayout


def basis_vector_indices(algebra, axes, *, name: str) -> tuple[int, ...]:
    """Return canonical grade-1 basis indices for coordinate axes."""
    vector_layout = algebra.layout((1,))
    normalized = tuple(int(axis) for axis in axes)
    invalid = [axis for axis in normalized if axis < 0 or axis >= algebra.n]
    if invalid:
        raise ValueError(f"{name} contains invalid basis-vector axes for n={algebra.n}: {invalid}")
    return tuple(vector_layout.basis_indices[axis] for axis in normalized)


def basis_positions(layout: GradeLayout, basis_indices, *, name: str) -> torch.Tensor:
    """Return compact-lane positions for canonical basis indices."""
    position_by_index = {index: position for position, index in enumerate(layout.basis_indices)}
    missing = [int(index) for index in basis_indices if int(index) not in position_by_index]
    if missing:
        raise ValueError(
            f"{name} requires basis indices {missing}, but layout grades {layout.grades} only expose "
            f"{layout.dim} compact lanes"
        )
    positions = [position_by_index[int(index)] for index in basis_indices]
    return torch.tensor(positions, dtype=torch.long)


def basis_position(layout: GradeLayout, basis_index: int, *, name: str) -> torch.Tensor:
    """Return one compact-lane position for a canonical basis index."""
    return basis_positions(layout, (basis_index,), name=name).squeeze(0)
