"""Runtime helpers for layer-facing multivector storage."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from core.foundation.layout import GradeLayout
from core.planning.action import metric_self_signs


@dataclass(frozen=True)
class LayerStorage:
    """Resolved storage contract for a layer input or output."""

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
        if self.layout is None:
            return self.algebra.get_grade_norms(values)
        flat = values.pow(2).reshape(-1, self.layout.dim)
        grade_ids = self.layout.grade_indices_tensor(device=values.device).unsqueeze(0).expand_as(flat)
        result = values.new_zeros(flat.shape[0], self.algebra.num_grades)
        result.scatter_add_(1, grade_ids, flat)
        return result.reshape(*values.shape[:-1], self.algebra.num_grades).clamp(min=self.algebra.eps).sqrt()

    def metric_signs(self, *, device=None, dtype=None) -> torch.Tensor:
        """Return basis self-product signs for this storage."""
        if self.layout is None:
            return metric_self_signs(self.algebra.default_layout(), device=device, dtype=dtype)
        return metric_self_signs(self.layout, device=device, dtype=dtype)


def resolve_layer_storage(algebra, *, layout: GradeLayout = None, grades=None) -> LayerStorage:
    """Resolve optional layer grade/layout metadata into a storage contract."""
    return LayerStorage(algebra, resolve_layer_layout(algebra, layout=layout, grades=grades))


def resolve_layer_layout(algebra, *, layout: GradeLayout = None, grades=None) -> GradeLayout | None:
    """Resolve an optional layer storage layout."""
    if layout is not None:
        spec = algebra.planner.spec
        if layout.spec != spec:
            raise ValueError(f"layout signature {layout.spec} does not match algebra signature {spec}")
        return layout
    if grades is not None:
        return algebra.layout(grades)
    default_grades = getattr(algebra, "_default_grades", None)
    if default_grades is not None:
        return algebra.layout(default_grades)
    return None
