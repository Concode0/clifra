# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Explicit tensor contracts for Clifford coefficient lanes.

``GradeLayout`` describes the semantic basis subset and order. ``LaneStorage``
describes the physical last-axis storage used by an executor or layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from clifra.core.layout import AlgebraSpec, GradeLayout, Layout


class LaneStorage(str, Enum):
    """Physical lane storage for a tensor's final coefficient axis."""

    COMPACT = "compact"
    CANONICAL = "canonical"


@dataclass(frozen=True)
class TensorContract:
    """Resolved tensor lane contract: one semantic layout plus one storage form."""

    layout: Layout
    storage: LaneStorage = LaneStorage.COMPACT

    def __post_init__(self) -> None:
        object.__setattr__(self, "storage", LaneStorage(self.storage))

    @property
    def spec(self) -> AlgebraSpec:
        """Signature derived from the semantic layout."""
        return self.layout.spec

    @classmethod
    def compact(cls, layout: Layout) -> "TensorContract":
        """Return a compact-lane contract for ``layout``."""
        return cls(layout=layout, storage=LaneStorage.COMPACT)

    @classmethod
    def canonical(cls, layout: Layout) -> "TensorContract":
        """Return a canonical full-basis storage contract with layout metadata."""
        return cls(layout=layout, storage=LaneStorage.CANONICAL)

    @property
    def uses_compact_storage(self) -> bool:
        """Return whether tensor lanes are stored as ``layout.dim`` compact lanes."""
        return self.storage is LaneStorage.COMPACT

    @property
    def uses_canonical_storage(self) -> bool:
        """Return whether tensor lanes are stored as canonical ``spec.dim`` lanes."""
        return self.storage is LaneStorage.CANONICAL

    @property
    def lane_dim(self) -> int:
        """Return required last-axis lane width."""
        return self.layout.dim if self.uses_compact_storage else self.spec.dim

    @property
    def grades(self) -> tuple[int, ...]:
        """Return semantic grades represented by this contract."""
        if isinstance(self.layout, GradeLayout):
            return self.layout.grades
        return tuple(sorted({blade.bit_count() for blade in self.layout.basis_indices}))

    def validate(self, values: torch.Tensor, *, name: str = "value") -> None:
        """Validate that ``values`` obeys this contract."""
        if values.ndim < 1:
            raise ValueError(f"{name} must include a coefficient lane dimension, got shape {tuple(values.shape)}")
        if values.shape[-1] != self.lane_dim:
            raise ValueError(
                f"{name} {self.storage.value} last dimension must be {self.lane_dim}, got {values.shape[-1]}"
            )

    def to_compact(self, values: torch.Tensor) -> torch.Tensor:
        """Return compact values for this contract's semantic layout."""
        self.validate(values)
        return values if self.uses_compact_storage else self.layout.compact(values)

    def to_canonical(self, values: torch.Tensor) -> torch.Tensor:
        """Return canonical full-basis values for this contract's semantic layout."""
        self.validate(values)
        return self.layout.full(values) if self.uses_compact_storage else values
