# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Static algebra and compact grade-layout value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, runtime_checkable

import torch

from clifra.core._kernel.basis import (
    basis_count_for_grades,
    basis_index_tuple_for_grades,
    basis_indices_tensor,
    normalize_grades,
)


@runtime_checkable
class Layout(Protocol):
    """Ordered coefficient lanes, independent of a planner's representation.

    GradeLayout is the supported implementation today. A future selected-blade
    layout can implement this contract without changing tensor declarations.
    """

    spec: "AlgebraSpec"

    @property
    def dim(self) -> int: ...

    @property
    def basis_indices(self) -> tuple[int, ...]: ...

    def compact(self, full: torch.Tensor) -> torch.Tensor: ...

    def full(self, values: torch.Tensor) -> torch.Tensor: ...

    def convert(self, values: torch.Tensor, source: "Layout") -> torch.Tensor: ...


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
        return GradeLayout(self, tuple(grades))

    def full_layout(self) -> "GradeLayout":
        """Return the canonical all-grades layout.

        ``Full`` is only a grade set: grades ``0..n`` in canonical basis order.
        It is not a separate runtime storage mode.
        """
        return self.layout(range(self.n + 1))


@dataclass(frozen=True)
class GradeLayout:
    """Compact fixed-grade layout whose basis indices are lowered on demand."""

    spec: AlgebraSpec
    grades: tuple[int, ...]
    _dim: int = field(init=False, repr=False, compare=False)
    _basis_indices: tuple[int, ...] | None = field(default=None, init=False, repr=False, compare=False)
    _indices_cache: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False, compare=False)
    _grade_index_cache: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False, compare=False)
    _conversion_cache: dict[tuple[AlgebraSpec, tuple[int, ...], str], tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        grades = normalize_grades(self.grades, self.spec.n) if self.grades else ()
        object.__setattr__(self, "grades", grades)
        object.__setattr__(self, "_dim", basis_count_for_grades(self.spec.n, grades) if grades else 0)

    @property
    def basis_indices(self) -> tuple[int, ...]:
        """Canonical basis indices represented by this compact layout."""
        indices = self._basis_indices
        if indices is None:
            indices = basis_index_tuple_for_grades(self.spec.n, self.grades) if self.grades else ()
            object.__setattr__(self, "_basis_indices", indices)
        return indices

    @property
    def dim(self) -> int:
        """Number of compact lanes."""
        return self._dim

    @property
    def full_dim(self) -> int:
        """Full multivector lane dimension."""
        return self.spec.dim

    def contains_grade(self, grade: int) -> bool:
        """Return whether ``grade`` is present in this layout."""
        return int(grade) in self.grades

    def indices_tensor(self, *, device=None) -> torch.Tensor:
        """Return basis indices as a tensor on ``device``."""
        key = _device_cache_key(device)
        cached = self._indices_cache.get(key)
        if cached is None:
            cached = basis_indices_tensor(self.basis_indices, n=self.spec.n, role="layout basis indices", device=device)
            self._indices_cache[key] = cached
        return cached

    def grade_indices_tensor(self, *, device=None) -> torch.Tensor:
        """Return per-lane grade ids on ``device``."""
        key = _device_cache_key(device)
        cached = self._grade_index_cache.get(key)
        if cached is None:
            grades = tuple(index.bit_count() for index in self.basis_indices)
            cached = torch.tensor(grades, dtype=torch.long, device=device)
            self._grade_index_cache[key] = cached
        return cached

    def positions_for_grades(self, grades: Iterable[int], *, device=None) -> torch.Tensor:
        """Return compact lane positions for the requested grades."""
        grades = tuple(grades)
        requested = set(normalize_grades(grades, self.spec.n)) if grades else set()
        positions = [position for position, index in enumerate(self.basis_indices) if index.bit_count() in requested]
        return torch.tensor(positions, dtype=torch.long, device=device)

    def positions_for_basis(self, basis_indices: Iterable[int], *, device=None) -> torch.Tensor:
        """Locate basis blades in requested order, rejecting absent blades."""
        positions = {blade: lane for lane, blade in enumerate(self.basis_indices)}
        requested = tuple(basis_indices)
        missing = [blade for blade in requested if blade not in positions]
        if missing:
            raise ValueError(f"basis blades {missing} are absent from layout")
        return torch.tensor([positions[blade] for blade in requested], dtype=torch.long, device=device)

    def grade_mask(self, grades: Iterable[int], *, device=None) -> torch.Tensor:
        """Boolean mask in compact-lane order for the selected grades."""
        grades = tuple(grades)
        requested = set(normalize_grades(grades, self.spec.n)) if grades else set()
        return torch.tensor(
            [blade.bit_count() in requested for blade in self.basis_indices], dtype=torch.bool, device=device
        )

    def union(self, other: "GradeLayout") -> "GradeLayout":
        """Return the union of two grade layouts."""
        self._check_operand(other)
        return self.spec.layout(set(self.grades) | set(other.grades))

    def intersection(self, other: "GradeLayout") -> "GradeLayout":
        """Return shared grades, including an empty layout when disjoint."""
        self._check_operand(other)
        return self.spec.layout(set(self.grades) & set(other.grades))

    def difference(self, other: "GradeLayout") -> "GradeLayout":
        """Return grades present only in this layout."""
        self._check_operand(other)
        return self.spec.layout(set(self.grades) - set(other.grades))

    def _check_operand(self, other: "GradeLayout") -> None:
        if self.spec != other.spec:
            raise ValueError("layout signatures must match")

    def convert(self, values: torch.Tensor, source: Layout) -> torch.Tensor:
        """Convert compact values from ``source`` into this layout.

        Shared basis lanes are copied by canonical basis index. Lanes present in
        this layout but absent from ``source`` are filled with zeros, which makes
        the method usable for both projections and sparse layout unions without
        materializing a full-lane multivector.
        """
        if source.spec != self.spec:
            raise ValueError(f"source layout signature {source.spec} does not match target spec {self.spec}")
        if values.ndim < 1 or values.shape[-1] != source.dim:
            raise ValueError(f"source values last dimension must be {source.dim}, got shape {tuple(values.shape)}")
        if source == self:
            return values

        output = values.new_zeros(*values.shape[:-1], self.dim)
        gather, scatter = self._conversion_tensors(source, device=values.device)
        copied = torch.index_select(values, -1, gather)
        return output.index_copy(-1, scatter, copied)

    def compact(self, full: torch.Tensor) -> torch.Tensor:
        """Gather compact lanes from a full-lane multivector tensor."""
        if full.ndim < 1 or full.shape[-1] != self.full_dim:
            raise ValueError(f"full last dimension must be {self.full_dim}, got shape {tuple(full.shape)}")
        return torch.index_select(full, -1, self.indices_tensor(device=full.device))

    def full(self, values: torch.Tensor) -> torch.Tensor:
        """Materialize compact lane values into a full-lane multivector tensor."""
        if values.ndim < 1 or values.shape[-1] != self.dim:
            raise ValueError(f"values last dimension must be {self.dim}, got shape {tuple(values.shape)}")
        output = values.new_zeros(*values.shape[:-1], self.full_dim)
        return output.index_copy(-1, self.indices_tensor(device=values.device), values)

    def _conversion_tensors(self, source: "GradeLayout", *, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        key = (source.spec, source.basis_indices, _device_cache_key(device))
        cached = self._conversion_cache.get(key)
        if cached is not None:
            return cached

        source_positions = {index: position for position, index in enumerate(source.basis_indices)}
        gather_positions: list[int] = []
        scatter_positions: list[int] = []
        for target_position, index in enumerate(self.basis_indices):
            source_position = source_positions.get(index)
            if source_position is None:
                continue
            gather_positions.append(source_position)
            scatter_positions.append(target_position)

        gather = torch.tensor(gather_positions, dtype=torch.long, device=device)
        scatter = torch.tensor(scatter_positions, dtype=torch.long, device=device)
        cached = (gather, scatter)
        self._conversion_cache[key] = cached
        return cached


def _device_cache_key(device) -> str:
    if device is None:
        return "cpu"
    return str(torch.device(device))
