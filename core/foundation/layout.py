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

from core.foundation.basis import basis_index_tuple_for_grades, basis_indices_tensor, normalize_grades


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
        return basis_indices_tensor(self.basis_indices, n=self.spec.n, role="layout basis indices", device=device)

    def convert(self, values: torch.Tensor, source: "GradeLayout") -> torch.Tensor:
        """Convert compact values from ``source`` into this layout.

        Shared basis lanes are copied by canonical basis index. Lanes present in
        this layout but absent from ``source`` are filled with zeros, which makes
        the method usable for both projections and sparse layout unions without
        materializing a full dense multivector.
        """
        if source.spec != self.spec:
            raise ValueError(f"source layout signature {source.spec} does not match target spec {self.spec}")
        if values.shape[-1] != source.dim:
            raise ValueError(f"source values last dimension must be {source.dim}, got {values.shape[-1]}")
        if source == self:
            return values

        source_positions = {index: position for position, index in enumerate(source.basis_indices)}
        gather_positions: list[int] = []
        scatter_positions: list[int] = []
        for target_position, index in enumerate(self.basis_indices):
            source_position = source_positions.get(index)
            if source_position is None:
                continue
            gather_positions.append(source_position)
            scatter_positions.append(target_position)

        output = values.new_zeros(*values.shape[:-1], self.dim)
        if not gather_positions:
            return output

        gather = torch.tensor(gather_positions, dtype=torch.long, device=values.device)
        scatter = torch.tensor(scatter_positions, dtype=torch.long, device=values.device)
        copied = torch.index_select(values, -1, gather)
        return output.index_copy(-1, scatter, copied)

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
