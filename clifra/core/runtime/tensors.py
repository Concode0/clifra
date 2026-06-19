# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Explicit tensor contracts for Clifford coefficient lanes.

``GradeLayout`` describes the semantic basis subset and order. ``LaneStorage``
describes the physical last-axis storage used by an executor or layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

from clifra.core.foundation.basis import normalize_grades, operation_coefficient
from clifra.core.foundation.layout import AlgebraSpec, GradeLayout


class LaneStorage(str, Enum):
    """Physical lane storage for a tensor's final coefficient axis."""

    COMPACT = "compact"
    CANONICAL = "canonical"


def normalize_lane_storage(storage: LaneStorage | str) -> LaneStorage:
    """Return a validated lane storage enum."""
    if isinstance(storage, LaneStorage):
        return storage
    try:
        return LaneStorage(str(storage))
    except ValueError as exc:
        raise ValueError("storage must be 'compact' or 'canonical'") from exc


@dataclass(frozen=True)
class TensorContract:
    """Resolved tensor lane contract: one semantic layout plus one storage form."""

    spec: AlgebraSpec
    layout: GradeLayout
    storage: LaneStorage = LaneStorage.COMPACT

    def __post_init__(self) -> None:
        check_layout_spec(self.spec, self.layout, "layout")
        object.__setattr__(self, "storage", normalize_lane_storage(self.storage))

    @classmethod
    def compact(cls, spec: AlgebraSpec, layout: GradeLayout) -> "TensorContract":
        """Return a compact-lane contract for ``layout``."""
        return cls(spec=spec, layout=layout, storage=LaneStorage.COMPACT)

    @classmethod
    def canonical(cls, spec: AlgebraSpec, layout: GradeLayout) -> "TensorContract":
        """Return a canonical full-basis storage contract with layout metadata."""
        return cls(spec=spec, layout=layout, storage=LaneStorage.CANONICAL)

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
        return self.layout.grades

    def validate(self, values: torch.Tensor, *, name: str = "value") -> None:
        """Validate that ``values`` obeys this contract."""
        if values.ndim < 1:
            raise ValueError(f"{name} must include a coefficient lane dimension, got shape {tuple(values.shape)}")
        if values.shape[-1] != self.lane_dim:
            raise ValueError(
                f"{name} {self.storage.value} last dimension must be {self.lane_dim}, got {values.shape[-1]}"
            )

    def validate_input(self, values: torch.Tensor, *, channels: int, name: str) -> None:
        """Validate layer input with a channel axis before the lane axis."""
        if values.ndim < 3:
            raise ValueError(f"{name}: expected ndim >= 3, got shape {tuple(values.shape)}")
        if values.shape[-2] != int(channels):
            raise ValueError(
                f"{name}: expected {channels} channels, got {values.shape[-2]} (shape {tuple(values.shape)})"
            )
        self.validate(values, name=name)

    def to_compact(self, values: torch.Tensor) -> torch.Tensor:
        """Return compact values for this contract's semantic layout."""
        self.validate(values)
        return values if self.uses_compact_storage else self.layout.compact(values)

    def to_canonical(self, values: torch.Tensor) -> torch.Tensor:
        """Return canonical full-basis values for this contract's semantic layout."""
        self.validate(values)
        return self.layout.full(values) if self.uses_compact_storage else values

    def scalar_mask(self, *, device=None, dtype=None) -> torch.Tensor:
        """Return a scalar-lane mask for this contract's storage form."""
        dtype = torch.float32 if dtype is None else dtype
        if self.uses_canonical_storage:
            values = [1.0] + [0.0] * (self.spec.dim - 1)
        else:
            values = [1.0 if index == 0 else 0.0 for index in self.layout.basis_indices]
        return torch.tensor(values, device=device, dtype=dtype)

    def grade_positions(self, grade: int, *, device=None) -> torch.Tensor:
        """Return lane positions for one grade in this contract's storage form."""
        if self.uses_canonical_storage:
            basis = self.spec.full_layout().basis_indices
        else:
            basis = self.layout.basis_indices
        positions = [position for position, index in enumerate(basis) if index.bit_count() == int(grade)]
        return torch.tensor(positions, dtype=torch.long, device=device)


def check_layout_spec(spec: AlgebraSpec, layout: GradeLayout, name: str) -> None:
    """Validate that ``layout`` belongs to ``spec``."""
    if layout.spec != spec:
        raise ValueError(f"{name} signature {layout.spec} does not match algebra signature {spec}")


def resolve_layout(algebra_or_spec, *, layout: Optional[GradeLayout] = None, grades=None) -> GradeLayout:
    """Resolve explicit layout metadata without inspecting tensor shapes."""
    spec = algebra_or_spec if isinstance(algebra_or_spec, AlgebraSpec) else AlgebraSpec.from_algebra(algebra_or_spec)
    if layout is not None:
        check_layout_spec(spec, layout, "layout")
        if grades is not None and layout.grades != normalize_grades(grades, spec.n, name="grades"):
            raise ValueError("layout and grades disagree")
        return layout
    if grades is not None:
        return spec.layout(grades)
    default_grades = getattr(algebra_or_spec, "_default_grades", None)
    if default_grades is not None:
        return spec.layout(default_grades)
    if hasattr(algebra_or_spec, "default_layout"):
        return algebra_or_spec.default_layout()
    return spec.full_layout()


def resolve_contract(
    algebra_or_spec,
    *,
    layout: Optional[GradeLayout] = None,
    grades=None,
    storage: LaneStorage | str = LaneStorage.COMPACT,
) -> TensorContract:
    """Resolve a tensor contract from explicit semantic layout and storage."""
    spec = algebra_or_spec if isinstance(algebra_or_spec, AlgebraSpec) else AlgebraSpec.from_algebra(algebra_or_spec)
    return TensorContract(spec=spec, layout=resolve_layout(algebra_or_spec, layout=layout, grades=grades), storage=storage)


def infer_contract(
    spec: AlgebraSpec,
    tensor: torch.Tensor,
    *,
    layout: Optional[GradeLayout] = None,
    grades=None,
    storage: LaneStorage | str | None = None,
    side: str = "value",
) -> TensorContract:
    """Infer storage at a public boundary, then return an explicit contract."""
    resolved = resolve_layout(spec, layout=layout, grades=grades)
    if storage is None:
        if tensor.shape[-1] == resolved.dim:
            storage = LaneStorage.COMPACT
        elif tensor.shape[-1] == spec.dim:
            storage = LaneStorage.CANONICAL
        else:
            raise ValueError(
                f"{side} last dimension must be {resolved.dim} for compact grades {resolved.grades} "
                f"or {spec.dim} canonical lanes, got {tensor.shape[-1]}"
            )
    contract = TensorContract(spec=spec, layout=resolved, storage=normalize_lane_storage(storage))
    contract.validate(tensor, name=side)
    return contract


def compact_values(algebra, value: torch.Tensor, *, layout: Optional[GradeLayout] = None, grades=None) -> tuple[torch.Tensor, GradeLayout]:
    """Return compact values and resolved semantic layout."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected Tensor value, got {type(value)!r}")
    contract = infer_contract(
        AlgebraSpec.from_algebra(algebra),
        value,
        layout=layout,
        grades=grades,
        storage=None,
    )
    return contract.to_compact(value), contract.layout


def canonical_values(algebra, value: torch.Tensor, *, layout: Optional[GradeLayout] = None, grades=None) -> torch.Tensor:
    """Return canonical full-basis values."""
    values, resolved = compact_values(algebra, value, layout=layout, grades=grades)
    if resolved.dim == resolved.spec.dim and resolved.grades == tuple(range(resolved.spec.n + 1)):
        return values
    return resolved.full(values)


def metric_self_signs(layout: GradeLayout, *, device=None, dtype=None) -> torch.Tensor:
    """Return basis self-product signs for a layout."""
    signs = [
        operation_coefficient(index, index, layout.spec.p, layout.spec.q, layout.spec.r, "gp")
        for index in layout.basis_indices
    ]
    return torch.tensor(signs, device=device, dtype=torch.float32 if dtype is None else dtype)
