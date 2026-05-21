"""Central storage contracts for dense and compact multivector values.

Dense and compact are physical storage modes, not separate algebraic planning
concepts. A value always has a logical :class:`GradeLayout`; storage only says
whether the tensor stores all canonical basis lanes or just that layout's
active lanes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

import torch

from clifra.core.foundation.basis import normalize_grades
from clifra.core.foundation.layout import AlgebraSpec, GradeLayout
from clifra.core.planning.action import metric_self_signs


class StorageMode(str, Enum):
    """Physical tensor storage mode for a multivector value."""

    DENSE = "dense"
    COMPACT = "compact"


class DispatchPath(str, Enum):
    """Execution path selected after storage and layout resolution."""

    DENSE_KERNEL = "dense_kernel"
    PLANNED_COMPACT = "planned_compact"
    PLANNED_DENSE_OUTPUT = "planned_dense_output"


@dataclass(frozen=True)
class TensorStorage:
    """Resolved physical storage for a tensor plus its logical grade layout."""

    spec: AlgebraSpec
    layout: GradeLayout
    mode: StorageMode

    @classmethod
    def dense(cls, spec: AlgebraSpec, layout: GradeLayout) -> "TensorStorage":
        """Return dense full-width storage with declared active layout metadata."""
        check_layout_spec(spec, layout, "layout")
        return cls(spec=spec, layout=layout, mode=StorageMode.DENSE)

    @classmethod
    def compact(cls, spec: AlgebraSpec, layout: GradeLayout) -> "TensorStorage":
        """Return compact active-lane storage for ``layout``."""
        check_layout_spec(spec, layout, "layout")
        return cls(spec=spec, layout=layout, mode=StorageMode.COMPACT)

    @property
    def is_compact(self) -> bool:
        """Return whether tensors use compact active lanes."""
        return self.mode is StorageMode.COMPACT

    @property
    def is_dense(self) -> bool:
        """Return whether tensors use full canonical basis lanes."""
        return self.mode is StorageMode.DENSE

    @property
    def lane_dim(self) -> int:
        """Return the tensor last-dimension width for this storage."""
        return self.layout.dim if self.is_compact else self.spec.dim

    @property
    def dense_dim(self) -> int:
        """Return the full canonical basis width."""
        return self.spec.dim

    @property
    def grades(self) -> tuple[int, ...]:
        """Return logical active grades."""
        return self.layout.grades

    def validate_tensor(self, values: torch.Tensor, *, name: str = "value") -> None:
        """Validate that ``values`` matches this storage width."""
        if values.ndim < 1:
            raise ValueError(f"{name} must include a coefficient lane dimension, got shape {tuple(values.shape)}")
        if values.shape[-1] != self.lane_dim:
            raise ValueError(f"{name} {self.mode.value} last dimension must be {self.lane_dim}, got {values.shape[-1]}")

    def compact_values(self, values: torch.Tensor) -> torch.Tensor:
        """Return compact active lanes from values in this storage."""
        self.validate_tensor(values)
        return values if self.is_compact else self.layout.compact(values)

    def dense_values(self, values: torch.Tensor) -> torch.Tensor:
        """Return dense full-width values from values in this storage."""
        self.validate_tensor(values)
        return self.layout.dense(values) if self.is_compact else values


@dataclass(frozen=True)
class LayerStorage:
    """Resolved storage contract for layer inputs and outputs."""

    algebra: object
    layout: GradeLayout | None = None

    @property
    def lane_dim(self) -> int:
        """Return the coefficient lane count accepted by this storage."""
        return self.algebra.dim if self.layout is None else self.layout.dim

    @property
    def is_compact(self) -> bool:
        """Return whether this storage is compact relative to the full algebra."""
        return self.layout is not None and self.layout.dim != self.algebra.dim

    @property
    def grades(self) -> tuple[int, ...] | None:
        """Return active grades when compact metadata is known."""
        return None if self.layout is None else self.layout.grades

    def validate_input(
        self,
        values: torch.Tensor,
        *,
        channels: int,
        name: str,
        allow_dense: bool | None = None,
    ) -> bool:
        """Validate layer input and return whether it is compact."""
        if values.ndim < 3:
            raise ValueError(f"{name}: expected ndim >= 3, got shape {tuple(values.shape)}")
        if values.shape[-2] != channels:
            raise ValueError(
                f"{name}: expected {channels} channels, got {values.shape[-2]} (shape {tuple(values.shape)})"
            )

        if self.layout is not None and values.shape[-1] == self.layout.dim:
            return self.is_compact

        if allow_dense is None:
            allow_dense = self.layout is None or self.layout.dim == self.algebra.dim
        if allow_dense and values.shape[-1] == self.algebra.dim:
            return False

        expected = [str(self.algebra.dim)] if allow_dense else []
        if self.layout is not None:
            expected.insert(0, f"{self.layout.dim} for grades {self.layout.grades}")
        raise ValueError(f"{name}: last dim must be {' or '.join(expected)}, got {values.shape[-1]}")

    def scalar_mask(self, *, device=None, dtype=None) -> torch.Tensor:
        """Return a scalar-lane mask for this storage."""
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
        """Return compact lane positions for one grade."""
        if self.layout is None:
            return self.algebra.grade_indices((grade,), device=device)
        positions = [
            position for position, index in enumerate(self.layout.basis_indices) if index.bit_count() == int(grade)
        ]
        return torch.tensor(positions, dtype=torch.long, device=device)

    def compact_grade_norms(self, values: torch.Tensor) -> torch.Tensor:
        """Return per-grade coefficient norms for compact values."""
        from clifra.core.runtime.actions import compact_grade_norms

        if self.layout is None:
            return self.algebra.grade_norms(values)
        return compact_grade_norms(self.algebra, values, self.layout)

    def metric_signs(self, *, device=None, dtype=None) -> torch.Tensor:
        """Return basis self-product signs for this storage."""
        if self.layout is None:
            return metric_self_signs(self.algebra.default_layout(), device=device, dtype=dtype)
        return metric_self_signs(self.layout, device=device, dtype=dtype)


@dataclass(frozen=True)
class DispatchDecision:
    """Storage-aware execution choice for a planned operation."""

    path: DispatchPath
    output_storage: TensorStorage
    reason: str

    @property
    def materializes_dense(self) -> bool:
        """Return whether the path crosses a dense materialization boundary."""
        return self.output_storage.is_dense

    @property
    def uses_planned_executor(self) -> bool:
        """Return whether execution goes through static planned kernels."""
        return self.path in {DispatchPath.PLANNED_COMPACT, DispatchPath.PLANNED_DENSE_OUTPUT}


def resolve_tensor_storage(
    spec: AlgebraSpec,
    tensor: torch.Tensor,
    *,
    grades=None,
    layout: Optional[GradeLayout] = None,
    compact: bool = False,
    side: str = "value",
    full_layout_allowed: bool = True,
) -> TensorStorage:
    """Resolve a tensor's logical layout and physical storage mode."""
    layout = resolve_operand_layout(
        spec,
        tensor,
        grades=grades,
        layout=layout,
        compact=compact,
        side=side,
        full_layout_allowed=full_layout_allowed,
    )
    mode = StorageMode.COMPACT if compact or tensor_is_compact(spec, tensor, layout) else StorageMode.DENSE
    storage = TensorStorage.compact(spec, layout) if mode is StorageMode.COMPACT else TensorStorage.dense(spec, layout)
    storage.validate_tensor(tensor, name=side)
    return storage


def resolve_operand_layout(
    spec: AlgebraSpec,
    tensor: torch.Tensor,
    *,
    grades=None,
    layout: Optional[GradeLayout] = None,
    compact: bool = False,
    side: str,
    full_layout_allowed: bool = True,
) -> GradeLayout:
    """Resolve one operand's logical grade layout from metadata or tensor shape."""
    if layout is not None:
        check_layout_spec(spec, layout, f"{side}_layout")
        if grades is not None and layout.grades != normalize_grades(grades, spec.n, name=f"{side}_grades"):
            raise ValueError(f"{side}_layout and {side}_grades disagree")
        _check_operand_shape(spec, tensor, layout, compact=compact, side=side)
        return layout

    if grades is not None:
        layout = spec.layout(grades)
        _check_operand_shape(spec, tensor, layout, compact=compact, side=side)
        return layout

    if compact:
        raise ValueError(f"{side}_layout or {side}_grades is required for compact {side} input")
    if tensor.shape[-1] != spec.dim:
        raise ValueError(
            f"{side} input has last dimension {tensor.shape[-1]}; declare {side}_layout or "
            f"{side}_grades for compact planned execution"
        )
    if not full_layout_allowed:
        raise ValueError(
            f"{side} input would require a full Cl({spec.p},{spec.q},{spec.r}) layout. "
            "Declare active grades or enable an explicit low-dimensional full-layout fallback."
        )
    return spec.layout(range(spec.n + 1))


def resolve_layer_storage(algebra, *, layout: GradeLayout = None, grades=None) -> LayerStorage:
    """Resolve optional layer grade/layout metadata into a storage contract."""
    return LayerStorage(algebra, resolve_layer_layout(algebra, layout=layout, grades=grades))


def resolve_layer_layout(algebra, *, layout: GradeLayout = None, grades=None) -> GradeLayout | None:
    """Resolve an optional layer storage layout."""
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


def tensor_is_compact(spec: AlgebraSpec, tensor: torch.Tensor, layout: GradeLayout) -> bool:
    """Return whether ``tensor`` already uses ``layout``'s compact lane count."""
    return layout.dim != spec.dim and tensor.shape[-1] == layout.dim


def storage_for_values(
    spec: AlgebraSpec,
    values: torch.Tensor,
    *,
    layout: Optional[GradeLayout] = None,
    grades: Optional[Iterable[int]] = None,
    compact: bool = False,
    side: str = "value",
    full_layout_allowed: bool = True,
) -> TensorStorage:
    """Alias for readability at runtime call sites."""
    return resolve_tensor_storage(
        spec,
        values,
        layout=layout,
        grades=grades,
        compact=compact,
        side=side,
        full_layout_allowed=full_layout_allowed,
    )


def resolve_planned_dispatch(request, *, compact_output: bool, reason: str | None = None) -> DispatchDecision:
    """Resolve the output storage boundary for a planned operation request."""
    output_storage = (
        request.output_storage
        if compact_output
        else TensorStorage.dense(request.spec, request.output_layout)
    )
    path = DispatchPath.PLANNED_COMPACT if output_storage.is_compact else DispatchPath.PLANNED_DENSE_OUTPUT
    if reason is None:
        reason = "compact_output=True" if compact_output else "caller requested dense output materialization"
    return DispatchDecision(path=path, output_storage=output_storage, reason=reason)


def _check_operand_shape(
    spec: AlgebraSpec,
    tensor: torch.Tensor,
    layout: GradeLayout,
    *,
    compact: bool,
    side: str,
) -> None:
    if tensor.ndim < 1:
        raise ValueError(f"{side} must include a coefficient lane dimension, got shape {tuple(tensor.shape)}")
    expected = layout.dim if compact or tensor_is_compact(spec, tensor, layout) else spec.dim
    if tensor.shape[-1] != expected:
        storage = "compact" if expected == layout.dim else "dense"
        raise ValueError(f"{side} {storage} last dimension must be {expected}, got {tensor.shape[-1]}")
