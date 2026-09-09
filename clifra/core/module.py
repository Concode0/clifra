# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0
"""PyTorch ownership of a shared algebra reference."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from .algebra import AlgebraContext


class CliffordModule(nn.Module):
    """Own an algebra reference and move its defaults with this module."""

    def __init__(self, algebra: AlgebraContext):
        super().__init__()
        object.__setattr__(self, "_algebra", algebra)

    @property
    def algebra(self) -> AlgebraContext:
        return self._algebra

    def _apply(self, fn):
        probe = fn(self._algebra._placement_probe())
        result = super()._apply(fn)
        object.__setattr__(self, "_algebra", self._algebra._copy_with_placement(probe.device, probe.dtype))
        return result
