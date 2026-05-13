# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Static algebra and compact grade-layout value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch

from core.foundation.basis import basis_index_tuple_for_grades, normalize_grades


@dataclass(frozen=True)
class AlgebraSpec:
    """Immutable Clifford signature metadata used by grade planners."""

    p: int
    q: int = 0
    r: int = 0

    def __post_init__(self) -> None:
        if self.p < 0 or self.q < 0 or self.r < 0:
            raise ValueError(f"signature counts must be non-negative, got Cl({self.p},{self.q},{self.r})")

    @classmethod
    def from_algebra(cls, algebra) -> "AlgebraSpec":
        """Build a spec from any algebra-like object with ``p``, ``q``, and ``r`` attributes."""
        return cls(int(algebra.p), int(algebra.q), int(algebra.r))

    @property
    def n(self) -> int:
        """Number of basis vectors."""
        return self.p + self.q + self.r

    @property
    def dim(self) -> int:
        """Number of canonical basis blades."""
        return 1 << self.n

    def layout(self, grades: Iterable[int]) -> "GradeLayout":
        """Return a compact layout for ``grades``."""
        return GradeLayout(self, normalize_grades(grades, self.n))


@dataclass(frozen=True)
class GradeLayout:
    """Compact basis-lane layout for a fixed grade set."""

    spec: AlgebraSpec
    grades: tuple[int, ...]
    _basis_indices: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        grades = normalize_grades(self.grades, self.spec.n)
        object.__setattr__(self, "grades", grades)
        object.__setattr__(self, "_basis_indices", basis_index_tuple_for_grades(self.spec.n, grades))

    @property
    def basis_indices(self) -> tuple[int, ...]:
        """Canonical dense basis indices represented by this compact layout."""
        return self._basis_indices

    @property
    def dim(self) -> int:
        """Number of compact lanes."""
        return len(self.basis_indices)

    @property
    def dense_dim(self) -> int:
        """Full dense multivector dimension."""
        return self.spec.dim

    def contains_grade(self, grade: int) -> bool:
        """Return whether ``grade`` is present in this layout."""
        return int(grade) in self.grades

    def indices_tensor(self, *, device=None) -> torch.Tensor:
        """Return basis indices as a tensor on ``device``."""
        return torch.tensor(self.basis_indices, dtype=torch.long, device=device)

    def compact(self, dense: torch.Tensor) -> torch.Tensor:
        """Gather compact lanes from a dense multivector tensor."""
        if dense.shape[-1] != self.dense_dim:
            raise ValueError(f"dense last dimension must be {self.dense_dim}, got {dense.shape[-1]}")
        return torch.index_select(dense, -1, self.indices_tensor(device=dense.device))

    def dense(self, values: torch.Tensor) -> torch.Tensor:
        """Materialize compact lane values into a dense multivector tensor."""
        if values.shape[-1] != self.dim:
            raise ValueError(f"values last dimension must be {self.dim}, got {values.shape[-1]}")
        output = values.new_zeros(*values.shape[:-1], self.dense_dim)
        return output.index_copy(-1, self.indices_tensor(device=values.device), values)
