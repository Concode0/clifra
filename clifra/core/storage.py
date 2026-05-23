# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Layout and lane-width contracts for multivector tensors.

The public framework contract is layout-first: values are tensors over a
logical :class:`GradeLayout`. Whether a tensor carries every basis lane or only
the declared active lanes is a lane-width detail used by the planner, not a
separate algebra concept.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

import torch

from clifra.core.foundation.basis import normalize_grades, operation_coefficient, reverse_sign
from clifra.core.foundation.layout import AlgebraSpec, GradeLayout


class LaneFormat(str, Enum):
    """Tensor lane width relative to a logical layout."""

    FULL = "full"
    ACTIVE = "active"


class ExecutorPath(str, Enum):
    """Execution path selected after layout and lane-width resolution."""

    PLANNED_ACTIVE = "planned_active"
    PLANNED_FULL = "planned_full"


@dataclass(frozen=True)
class ValueLayout:
    """Resolved lane contract for one tensor plus its logical grade layout."""

    spec: AlgebraSpec
    layout: GradeLayout
    lane_format: LaneFormat

    @classmethod
    def full(cls, spec: AlgebraSpec, layout: GradeLayout) -> "ValueLayout":
        """Return a full-basis lane contract with declared layout metadata."""
        check_layout_spec(spec, layout, "layout")
        return cls(spec=spec, layout=layout, lane_format=LaneFormat.FULL)

    @classmethod
    def active(cls, spec: AlgebraSpec, layout: GradeLayout) -> "ValueLayout":
        """Return an active-lane contract for ``layout``."""
        check_layout_spec(spec, layout, "layout")
        return cls(spec=spec, layout=layout, lane_format=LaneFormat.ACTIVE)

    @property
    def uses_active_lanes(self) -> bool:
        """Return whether tensors use only declared layout lanes."""
        return self.lane_format is LaneFormat.ACTIVE

    @property
    def uses_full_lanes(self) -> bool:
        """Return whether tensors use all canonical basis lanes."""
        return self.lane_format is LaneFormat.FULL

    @property
    def lane_dim(self) -> int:
        """Return the tensor last-dimension width for this lane contract."""
        return self.layout.dim if self.uses_active_lanes else self.spec.dim

    @property
    def full_dim(self) -> int:
        """Return the full canonical basis width."""
        return self.spec.dim

    @property
    def grades(self) -> tuple[int, ...]:
        """Return logical active grades."""
        return self.layout.grades

    def validate_tensor(self, values: torch.Tensor, *, name: str = "value") -> None:
        """Validate that ``values`` matches this lane contract."""
        if values.ndim < 1:
            raise ValueError(f"{name} must include a coefficient lane dimension, got shape {tuple(values.shape)}")
        if values.shape[-1] != self.lane_dim:
            raise ValueError(
                f"{name} {self.lane_format.value} last dimension must be {self.lane_dim}, got {values.shape[-1]}"
            )

    def active_values(self, values: torch.Tensor) -> torch.Tensor:
        """Return declared layout lanes from values following this contract."""
        self.validate_tensor(values)
        return values if self.uses_active_lanes else self.layout.compact(values)

    def full_values(self, values: torch.Tensor) -> torch.Tensor:
        """Return full-basis values from values following this contract."""
        self.validate_tensor(values)
        return self.layout.dense(values) if self.uses_active_lanes else values


@dataclass(frozen=True)
class LayerLayout:
    """Resolved lane contract for layer inputs and outputs."""

    algebra: object
    layout: GradeLayout

    @property
    def lane_dim(self) -> int:
        """Return the coefficient lane count accepted by this contract."""
        return self.layout.dim

    @property
    def uses_active_lanes(self) -> bool:
        """Return whether this layer contract uses the declared layout lanes."""
        return True

    @property
    def grades(self) -> tuple[int, ...]:
        """Return active grades when layout metadata is known."""
        return self.layout.grades

    def validate_input(
        self,
        values: torch.Tensor,
        *,
        channels: int,
        name: str,
    ) -> None:
        """Validate layer input against this declared layout contract."""
        if values.ndim < 3:
            raise ValueError(f"{name}: expected ndim >= 3, got shape {tuple(values.shape)}")
        if values.shape[-2] != channels:
            raise ValueError(
                f"{name}: expected {channels} channels, got {values.shape[-2]} (shape {tuple(values.shape)})"
            )
        if values.shape[-1] != self.layout.dim:
            raise ValueError(
                f"{name}: last dim must be {self.layout.dim} for grades {self.layout.grades}, "
                f"got {values.shape[-1]}"
            )

    def scalar_mask(self, *, device=None, dtype=None) -> torch.Tensor:
        """Return a scalar-lane mask for this contract."""
        dtype = torch.float32 if dtype is None else dtype
        return torch.tensor(
            [1.0 if index == 0 else 0.0 for index in self.layout.basis_indices],
            device=device,
            dtype=dtype,
        )

    def grade_positions(self, grade: int, *, device=None) -> torch.Tensor:
        """Return lane positions for one grade."""
        positions = [
            position for position, index in enumerate(self.layout.basis_indices) if index.bit_count() == int(grade)
        ]
        return torch.tensor(positions, dtype=torch.long, device=device)

    def active_grade_norms(self, values: torch.Tensor) -> torch.Tensor:
        """Return per-grade coefficient norms for active-lane values."""
        return compact_grade_norms(self.algebra, values, self.layout)

    def metric_signs(self, *, device=None, dtype=None) -> torch.Tensor:
        """Return basis self-product signs for this contract."""
        return metric_self_signs(self.layout, device=device, dtype=dtype)


@dataclass(frozen=True)
class ExecutionBoundary:
    """Output lane boundary chosen for a planned operation."""

    path: ExecutorPath
    output_value: ValueLayout
    reason: str

    @property
    def materializes_full(self) -> bool:
        """Return whether the path crosses a full-basis materialization boundary."""
        return self.output_value.uses_full_lanes


def resolve_value_layout(
    spec: AlgebraSpec,
    tensor: torch.Tensor,
    *,
    grades=None,
    layout: Optional[GradeLayout] = None,
    active_lanes: bool = False,
    side: str = "value",
) -> ValueLayout:
    """Resolve a tensor's logical layout and lane-width contract."""
    layout = resolve_operand_layout(
        spec,
        tensor,
        grades=grades,
        layout=layout,
        active_lanes=active_lanes,
        side=side,
    )
    if tensor.shape[-1] == layout.dim:
        value_layout = ValueLayout.active(spec, layout)
    elif tensor.shape[-1] == spec.dim:
        value_layout = ValueLayout.full(spec, layout)
    else:
        raise ValueError(
            f"{side} last dimension must be {layout.dim} for grades {layout.grades} or {spec.dim} full lanes, "
            f"got {tensor.shape[-1]}"
        )
    value_layout.validate_tensor(tensor, name=side)
    return value_layout


def resolve_operand_layout(
    spec: AlgebraSpec,
    tensor: torch.Tensor,
    *,
    grades=None,
    layout: Optional[GradeLayout] = None,
    active_lanes: bool = False,
    side: str,
) -> GradeLayout:
    """Resolve one operand's logical grade layout from metadata or tensor shape."""
    if layout is not None:
        check_layout_spec(spec, layout, f"{side}_layout")
        if grades is not None and layout.grades != normalize_grades(grades, spec.n, name=f"{side}_grades"):
            raise ValueError(f"{side}_layout and {side}_grades disagree")
        _check_operand_shape(spec, tensor, layout, active_lanes=active_lanes, side=side)
        return layout

    if grades is not None:
        layout = spec.layout(grades)
        _check_operand_shape(spec, tensor, layout, active_lanes=active_lanes, side=side)
        return layout

    if tensor.shape[-1] != spec.dim:
        raise ValueError(
            f"{side} input has last dimension {tensor.shape[-1]}; declare {side}_layout or "
            f"{side}_grades for layout-planned execution"
        )
    return spec.full_layout()


def resolve_layer_layout_contract(algebra, *, layout: GradeLayout = None, grades=None) -> LayerLayout:
    """Resolve optional layer grade/layout metadata into a layer lane contract."""
    return LayerLayout(algebra, resolve_layer_layout(algebra, layout=layout, grades=grades))


def resolve_layer_layout(algebra, *, layout: GradeLayout = None, grades=None) -> GradeLayout | None:
    """Resolve optional layer layout metadata."""
    spec = AlgebraSpec.from_algebra(algebra)
    if layout is not None:
        check_layout_spec(spec, layout, "layout")
        return layout
    if grades is not None:
        return algebra.layout(grades)
    default_grades = getattr(algebra, "_default_grades", None)
    if default_grades is not None:
        return algebra.layout(default_grades)
    if hasattr(algebra, "default_layout"):
        return algebra.default_layout()
    return spec.full_layout()


def check_layout_spec(spec: AlgebraSpec, layout: GradeLayout, name: str) -> None:
    """Validate that a layout belongs to ``spec``."""
    if layout.spec != spec:
        raise ValueError(f"{name} signature {layout.spec} does not match algebra signature {spec}")


def tensor_uses_active_lanes(spec: AlgebraSpec, tensor: torch.Tensor, layout: GradeLayout) -> bool:
    """Return whether ``tensor`` uses ``layout``'s lane count."""
    return tensor.shape[-1] == layout.dim


def layout_for_values(
    spec: AlgebraSpec,
    values: torch.Tensor,
    *,
    layout: Optional[GradeLayout] = None,
    grades: Optional[Iterable[int]] = None,
    active_lanes: bool = False,
    side: str = "value",
) -> ValueLayout:
    """Alias for readability at runtime call sites."""
    return resolve_value_layout(
        spec,
        values,
        layout=layout,
        grades=grades,
        active_lanes=active_lanes,
        side=side,
    )


def resolve_output_boundary(request, *, active_output: bool, reason: str | None = None) -> ExecutionBoundary:
    """Resolve the output lane boundary for a planned operation request."""
    output_value = request.output_value
    path = ExecutorPath.PLANNED_ACTIVE
    if reason is None:
        reason = "planner resolved output layout"
    return ExecutionBoundary(path=path, output_value=output_value, reason=reason)


def _check_operand_shape(
    spec: AlgebraSpec,
    tensor: torch.Tensor,
    layout: GradeLayout,
    *,
    active_lanes: bool,
    side: str,
) -> None:
    if tensor.ndim < 1:
        raise ValueError(f"{side} must include a coefficient lane dimension, got shape {tuple(tensor.shape)}")
    if tensor.shape[-1] not in {layout.dim, spec.dim}:
        raise ValueError(
            f"{side} last dimension must be {layout.dim} for grades {layout.grades} or {spec.dim} full lanes, "
            f"got {tensor.shape[-1]}"
        )


def active_values(
    algebra,
    value: torch.Tensor,
    *,
    layout: Optional[GradeLayout] = None,
    grades: Optional[Iterable[int]] = None,
) -> tuple[torch.Tensor, GradeLayout]:
    """Return tensor values and their resolved layout without materializing hidden lanes."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected Tensor value, got {type(value)!r}")
    resolved = resolve_layer_layout(algebra, layout=layout, grades=grades)
    if value.shape[-1] == resolved.dim:
        return value, resolved
    if value.shape[-1] == resolved.spec.dim:
        return resolved.compact(value), resolved
    raise ValueError(
        f"value last dimension must be {resolved.dim} for grades {resolved.grades} or {resolved.spec.dim} full lanes, "
        f"got {value.shape[-1]}"
    )


def materialize_full(
    algebra,
    value: torch.Tensor,
    *,
    layout: Optional[GradeLayout] = None,
    grades: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Explicitly materialize values into the canonical all-grades basis."""
    values, resolved = active_values(algebra, value, layout=layout, grades=grades)
    if resolved.dim == resolved.spec.dim and resolved.grades == tuple(range(resolved.spec.n + 1)):
        return values
    return resolved.dense(values)


def metric_self_signs(layout: GradeLayout, *, device=None, dtype=None) -> torch.Tensor:
    """Return basis self-product signs for a layout."""
    signs = [
        operation_coefficient(index, index, layout.spec.p, layout.spec.q, layout.spec.r, "gp")
        for index in layout.basis_indices
    ]
    return torch.tensor(signs, device=device, dtype=torch.float32 if dtype is None else dtype)


def hermitian_signs(
    algebra,
    layout: Optional[GradeLayout] = None,
    *,
    grades: Optional[Iterable[int]] = None,
    device=None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Return Hermitian metric signs for a declared layout."""
    resolved = resolve_layer_layout(algebra, layout=layout, grades=grades)
    if device is None:
        device = getattr(algebra, "device", None)
    if dtype is None:
        dtype = getattr(algebra, "dtype", torch.float32)

    full_signs = getattr(algebra, "_hermitian_signs", None)
    if full_signs is not None:
        indices = resolved.indices_tensor(device=full_signs.device)
        signs = torch.index_select(full_signs, -1, indices)
        return signs.to(device=device, dtype=dtype)

    values = [_hermitian_sign_for_index(resolved.spec, index) for index in resolved.basis_indices]
    return torch.tensor(values, dtype=dtype, device=device)


def compact_grade_norms(algebra, values: torch.Tensor, layout: GradeLayout) -> torch.Tensor:
    """Return per-grade coefficient norms for declared-layout values."""
    if values.shape[-1] != layout.dim:
        raise ValueError(f"values last dimension must be {layout.dim}, got {values.shape[-1]}")
    flat = values.pow(2).reshape(-1, layout.dim)
    grade_ids = layout.grade_indices_tensor(device=values.device).unsqueeze(0).expand_as(flat)
    result = values.new_zeros(flat.shape[0], layout.spec.n + 1)
    result.scatter_add_(1, grade_ids, flat)
    eps = getattr(algebra, "eps", torch.finfo(values.dtype).eps)
    return result.reshape(*values.shape[:-1], layout.spec.n + 1).clamp(min=eps).sqrt()


def _hermitian_sign_for_index(spec: AlgebraSpec, index: int) -> float:
    grade = int(index).bit_count()
    grade_sign = -1.0 if grade % 2 else 1.0
    metric_sign = operation_coefficient(index, index, spec.p, spec.q, spec.r, "gp")
    return grade_sign * reverse_sign(index) * metric_sign
