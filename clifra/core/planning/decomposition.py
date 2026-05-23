# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Plan-only contracts for fixed-shape bivector decomposition executors."""

from __future__ import annotations

from dataclasses import dataclass

from clifra.core.foundation.layout import GradeLayout


@dataclass(frozen=True)
class BivectorDecompositionPlan:
    """Static layout and buffer contract for bivector decomposition."""

    bivector_layout: GradeLayout
    vector_layout: GradeLayout
    rotor_layout: GradeLayout
    components: int
    fixed_iterations: int

    @property
    def residual_shape_tail(self) -> tuple[int]:
        """Return the trailing shape for residual bivector buffers."""
        return (self.bivector_layout.dim,)

    @property
    def vector_shape_tail(self) -> tuple[int]:
        """Return the trailing shape for extracted vector buffers."""
        return (self.vector_layout.dim,)

    @property
    def rotor_shape_tail(self) -> tuple[int]:
        """Return the trailing shape for reconstructed rotor buffers."""
        return (self.rotor_layout.dim,)


def build_bivector_decomposition_plan(
    algebra,
    *,
    input_layout: GradeLayout,
    components: int | None = None,
    fixed_iterations: int | None = None,
) -> BivectorDecompositionPlan:
    """Resolve decomposition layouts and fixed loop sizes without executing tensors."""
    if input_layout.grades != (2,):
        raise ValueError(f"bivector decomposition requires grade-2 input layout, got {input_layout.grades}")
    spec = input_layout.spec
    vector_layout = spec.layout((1,))
    rotor_layout = spec.full_layout()
    resolved_components = components if components is not None else max(spec.n // 2, 1)
    resolved_iterations = (
        int(fixed_iterations)
        if fixed_iterations is not None
        else int(getattr(algebra, "_exp_fixed_iterations", 20))
    )
    return BivectorDecompositionPlan(
        bivector_layout=input_layout,
        vector_layout=vector_layout,
        rotor_layout=rotor_layout,
        components=int(resolved_components),
        fixed_iterations=resolved_iterations,
    )
