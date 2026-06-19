# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Small helpers for layout-first adapter examples."""

from __future__ import annotations

import torch

from clifra.core.foundation.layout import GradeLayout


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
