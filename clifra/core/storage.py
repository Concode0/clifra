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

from clifra.core.foundation.basis import normalize_grades
from clifra.core.foundation.layout import AlgebraSpec, GradeLayout
from clifra.core.planning.action import metric_self_signs


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
    layout: GradeLayout | None = None

    @property
    def lane_dim(self) -> int:
        """Return the coefficient lane count accepted by this contract."""
        return self.algebra.dim if self.layout is None else self.layout.dim

    @property
    def uses_active_lanes(self) -> bool:
        """Return whether this layer contract uses only declared layout lanes."""
        return self.layout is not None and self.layout.dim != self.algebra.dim

    @property
    def grades(self) -> tuple[int, ...] | None:
        """Return active grades when layout metadata is known."""
        return None if self.layout is None else self.layout.grades

    def validate_input(
        self,
        values: torch.Tensor,
        *,
        channels: int,
        name: str,
        allow_full: bool | None = None,
    ) -> bool:
        """Validate layer input and return whether it uses active layout lanes."""
        if values.ndim < 3:
            raise ValueError(f"{name}: expected ndim >= 3, got shape {tuple(values.shape)}")
        if values.shape[-2] != channels:
            raise ValueError(
                f"{name}: expected {channels} channels, got {values.shape[-2]} (shape {tuple(values.shape)})"
            )

        if self.layout is not None and values.shape[-1] == self.layout.dim:
            return self.uses_active_lanes

        if allow_full is None:
            allow_full = self.layout is None or self.layout.dim == self.algebra.dim
        if allow_full and values.shape[-1] == self.algebra.dim:
            return False

        expected = [str(self.algebra.dim)] if allow_full else []
        if self.layout is not None:
            expected.insert(0, f"{self.layout.dim} for grades {self.layout.grades}")
        raise ValueError(f"{name}: last dim must be {' or '.join(expected)}, got {values.shape[-1]}")

    def scalar_mask(self, *, device=None, dtype=None) -> torch.Tensor:
        """Return a scalar-lane mask for this contract."""
        dtype = torch.float32 if dtype is None else dtype
        if self.layout is None:
            mask = torch.zeros(self.algebra.dim, device=device, dtype=dtype)
            mask[0] = 1.0
            return mask
        return torch.tensor(
            [1.0 if index == 0 else 0.0 for index in self.layout.basis_indices],
            device=device,
            dtype=dtype,
        )

    def grade_positions(self, grade: int, *, device=None) -> torch.Tensor:
        """Return lane positions for one grade."""
        if self.layout is None:
            return self.algebra.grade_indices((grade,), device=device)
        positions = [
            position for position, index in enumerate(self.layout.basis_indices) if index.bit_count() == int(grade)
        ]
        return torch.tensor(positions, dtype=torch.long, device=device)

    def active_grade_norms(self, values: torch.Tensor) -> torch.Tensor:
        """Return per-grade coefficient norms for active-lane values."""
        from clifra.core.runtime.actions import compact_grade_norms

        if self.layout is None:
            return self.algebra.grade_norms(values)
        return compact_grade_norms(self.algebra, values, self.layout)

    def metric_signs(self, *, device=None, dtype=None) -> torch.Tensor:
        """Return basis self-product signs for this contract."""
        if self.layout is None:
            return metric_self_signs(self.algebra.default_layout(), device=device, dtype=dtype)
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
    full_layout_allowed: bool = True,
) -> ValueLayout:
    """Resolve a tensor's logical layout and lane-width contract."""
    layout = resolve_operand_layout(
        spec,
        tensor,
        grades=grades,
        layout=layout,
        active_lanes=active_lanes,
        side=side,
        full_layout_allowed=full_layout_allowed,
    )
    lane_format = (
        LaneFormat.ACTIVE if active_lanes or tensor_uses_active_lanes(spec, tensor, layout) else LaneFormat.FULL
    )
    value_layout = (
        ValueLayout.active(spec, layout) if lane_format is LaneFormat.ACTIVE else ValueLayout.full(spec, layout)
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
    full_layout_allowed: bool = True,
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

    if active_lanes:
        raise ValueError(f"{side}_layout or {side}_grades is required for active-lane {side} input")
    if tensor.shape[-1] != spec.dim:
        raise ValueError(
            f"{side} input has last dimension {tensor.shape[-1]}; declare {side}_layout or "
            f"{side}_grades for layout-planned execution"
        )
    if not full_layout_allowed:
        raise ValueError(
            f"{side} input would require a full Cl({spec.p},{spec.q},{spec.r}) layout. "
            "Declare active grades or enable an explicit low-dimensional full-layout fallback."
        )
    return spec.layout(range(spec.n + 1))


def resolve_layer_layout_contract(algebra, *, layout: GradeLayout = None, grades=None) -> LayerLayout:
    """Resolve optional layer grade/layout metadata into a layer lane contract."""
    return LayerLayout(algebra, resolve_layer_layout(algebra, layout=layout, grades=grades))


def resolve_layer_layout(algebra, *, layout: GradeLayout = None, grades=None) -> GradeLayout | None:
    """Resolve optional layer layout metadata."""
    if layout is not None:
        spec = algebra.planner.spec
        check_layout_spec(spec, layout, "layout")
        return layout
    if grades is not None:
        return algebra.layout(grades)
    default_grades = getattr(algebra, "_default_grades", None)
    if default_grades is not None:
        return algebra.layout(default_grades)
    return None


def check_layout_spec(spec: AlgebraSpec, layout: GradeLayout, name: str) -> None:
    """Validate that a layout belongs to ``spec``."""
    if layout.spec != spec:
        raise ValueError(f"{name} signature {layout.spec} does not match algebra signature {spec}")


def tensor_uses_active_lanes(spec: AlgebraSpec, tensor: torch.Tensor, layout: GradeLayout) -> bool:
    """Return whether ``tensor`` already uses ``layout``'s active lane count."""
    return layout.dim != spec.dim and tensor.shape[-1] == layout.dim


def layout_for_values(
    spec: AlgebraSpec,
    values: torch.Tensor,
    *,
    layout: Optional[GradeLayout] = None,
    grades: Optional[Iterable[int]] = None,
    active_lanes: bool = False,
    side: str = "value",
    full_layout_allowed: bool = True,
) -> ValueLayout:
    """Alias for readability at runtime call sites."""
    return resolve_value_layout(
        spec,
        values,
        layout=layout,
        grades=grades,
        active_lanes=active_lanes,
        side=side,
        full_layout_allowed=full_layout_allowed,
    )


def resolve_output_boundary(request, *, active_output: bool, reason: str | None = None) -> ExecutionBoundary:
    """Resolve the output lane boundary for a planned operation request."""
    output_value = request.output_value if active_output else ValueLayout.full(request.spec, request.output_layout)
    path = ExecutorPath.PLANNED_ACTIVE if output_value.uses_active_lanes else ExecutorPath.PLANNED_FULL
    if reason is None:
        reason = "active output layout" if active_output else "caller requested full-basis output materialization"
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
    expected = layout.dim if active_lanes or tensor_uses_active_lanes(spec, tensor, layout) else spec.dim
    if tensor.shape[-1] != expected:
        lane_width = "active" if expected == layout.dim else "full"
        raise ValueError(f"{side} {lane_width} last dimension must be {expected}, got {tensor.shape[-1]}")
