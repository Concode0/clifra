# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Shared algebra host API for planner-owned layout contracts."""

from __future__ import annotations

from typing import Iterable, Optional

import torch

from clifra.core.execution.action import (
    GradedLinearActionExecutor,
    MultiVersorActionExecutor,
    PairedBivectorActionExecutor,
    VersorActionExecutor,
)
from clifra.core.execution.handles import (
    FullSandwichActionHandle,
    MultiVersorActionHandle,
    PairedBivectorActionHandle,
    ProductPlanHandle,
    UnaryPlanHandle,
    VersorActionHandle,
)
from clifra.core.foundation.basis import expand_output_grades, normalize_grades
from clifra.core.foundation.layout import AlgebraSpec, GradeLayout
from clifra.core.foundation.numerics import signed_clamp_min
from clifra.core.planning.layouts import normalize_product_op
from clifra.core.runtime.energy import lane_grade_norms
from clifra.core.runtime.forms import conjugate_scalar_form_signs as _conjugate_scalar_form_signs
from clifra.core.runtime.tensors import LaneStorage, normalize_lane_storage


class AlgebraHostMixin:
    """Planner-facing algebra API with no runtime layout inference."""

    def layout(self, grades: Optional[Iterable[int]] = None) -> GradeLayout:
        """Return the declared grade layout, or the host default layout."""
        if grades is None:
            return self.default_layout()
        return self.planner.layout(grades)

    def default_layout(self) -> GradeLayout:
        """Return the configured default layout, falling back to all grades."""
        cached = getattr(self, "_default_layout", None)
        if cached is not None:
            return cached
        spec = AlgebraSpec.from_algebra(self)
        default_grades = getattr(self, "_default_grades", None)
        resolved = spec.full_layout() if default_grades is None else spec.layout(default_grades)
        if hasattr(self, "_default_layout"):
            self._default_layout = resolved
        return resolved

    def resolve_layout(
        self,
        *,
        layout: Optional[GradeLayout] = None,
        grades: Optional[Iterable[int]] = None,
        mv=None,
    ) -> GradeLayout:
        """Resolve explicit layout metadata. Tensor inspection is intentionally not supported."""
        if mv is not None:
            raise TypeError("Multivector wrappers are not part of the core layout contract")
        if layout is not None:
            if layout.spec != AlgebraSpec.from_algebra(self):
                raise ValueError(f"layout signature {layout.spec} does not match algebra signature")
            return layout
        if grades is not None:
            return self.layout(grades)
        return self.default_layout()

    def grade_indices(self, grades: Iterable[int], *, device=None) -> torch.Tensor:
        """Return canonical basis indices for ``grades``."""
        if device is None:
            device = getattr(self, "device", None)
        return self.layout(grades).indices_tensor(device=device)

    def conjugate_scalar_form_signs(
        self,
        layout: Optional[GradeLayout] = None,
        *,
        grades: Optional[Iterable[int]] = None,
        device=None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return signs for the signed Clifford-conjugation scalar form."""
        return _conjugate_scalar_form_signs(self, layout=layout, grades=grades, device=device, dtype=dtype)

    def product_executor(
        self,
        *,
        left_grades,
        right_grades,
        op: str = "gp",
        output_grades=None,
        dtype: Optional[torch.dtype] = None,
        device=None,
        cache: bool = True,
    ):
        """Return a preplanned product executor."""
        if dtype is None:
            dtype = getattr(self, "dtype", torch.float32)
        if device is None:
            device = getattr(self, "device", None)
        return self.planner.product_executor(
            op=op,
            left_grades=left_grades,
            right_grades=right_grades,
            output_grades=output_grades,
            dtype=dtype,
            device=device,
            cache=cache,
        )

    def plan_product(
        self,
        *,
        op: str = "gp",
        left_grades=None,
        right_grades=None,
        output_grades=None,
        left_layout: Optional[GradeLayout] = None,
        right_layout: Optional[GradeLayout] = None,
        output_layout: Optional[GradeLayout] = None,
        dtype: Optional[torch.dtype] = None,
        device=None,
        cache: bool = True,
    ) -> ProductPlanHandle:
        """Return an compact-lane product handle with no runtime request inference."""
        if dtype is None:
            dtype = getattr(self, "dtype", torch.float32)
        if device is None:
            device = getattr(self, "device", None)
        left_layout = self._declared_layout(left_grades, left_layout)
        right_layout = self._declared_layout(right_grades, right_layout)
        output_layout = self._product_output_layout(
            op=op,
            left_layout=left_layout,
            right_layout=right_layout,
            output_grades=output_grades,
            output_layout=output_layout,
        )
        executor = self.planner.product_executor_for_layouts(
            op=normalize_product_op(op),
            left_layout=left_layout,
            right_layout=right_layout,
            output_layout=output_layout,
            dtype=dtype,
            device=device,
            cache=cache,
        )
        return ProductPlanHandle(executor)

    def plan_unary(
        self,
        *,
        op: str,
        input_grades=None,
        output_grades=None,
        input_layout: Optional[GradeLayout] = None,
        output_layout: Optional[GradeLayout] = None,
        dtype: Optional[torch.dtype] = None,
        device=None,
        cache: bool = True,
    ) -> UnaryPlanHandle:
        """Return an compact-lane unary handle with no runtime request inference."""
        if dtype is None:
            dtype = getattr(self, "dtype", torch.float32)
        if device is None:
            device = getattr(self, "device", None)
        input_layout = self._declared_layout(input_grades, input_layout)
        output_layout = self._optional_layout(output_grades, output_layout)
        executor = self.planner.unary_executor(
            op=op,
            input_grades=input_layout.grades,
            output_grades=None if output_layout is None else output_layout.grades,
            dtype=dtype,
            device=device,
            cache=cache,
        )
        return UnaryPlanHandle(executor)

    def plan_signature_norm_squared(
        self,
        *,
        grades=None,
        input_grades=None,
        layout: Optional[GradeLayout] = None,
        input_layout: Optional[GradeLayout] = None,
        dtype: Optional[torch.dtype] = None,
        device=None,
        cache: bool = True,
    ):
        """Return a signed signature-norm executor for declared compact-lane values."""
        if input_grades is None:
            input_grades = grades
        if input_layout is None:
            input_layout = layout
        if dtype is None:
            dtype = getattr(self, "dtype", torch.float32)
        if device is None:
            device = getattr(self, "device", None)
        input_layout = self._declared_layout(input_grades, input_layout)
        return self.planner.signature_norm_squared_executor_for_layout(
            input_layout=input_layout,
            dtype=dtype,
            device=device,
            cache=cache,
        )

    def plan_pseudoscalar_product(
        self,
        *,
        input_grades=None,
        output_grades=None,
        input_layout: Optional[GradeLayout] = None,
        output_layout: Optional[GradeLayout] = None,
        dtype: Optional[torch.dtype] = None,
        device=None,
        cache: bool = True,
    ):
        """Return a right-pseudoscalar product executor for compact-lane values."""
        if dtype is None:
            dtype = getattr(self, "dtype", torch.float32)
        if device is None:
            device = getattr(self, "device", None)
        input_layout = self._declared_layout(input_grades, input_layout)
        output_layout = self._optional_layout(output_grades, output_layout)
        return self.planner.pseudoscalar_product_executor_for_layout(
            input_layout=input_layout,
            output_layout=output_layout,
            dtype=dtype,
            device=device,
            cache=cache,
        )

    def plan_bivector_exp(
        self,
        *,
        input_grades=None,
        output_grades=None,
        input_layout: Optional[GradeLayout] = None,
        output_layout: Optional[GradeLayout] = None,
        dtype: Optional[torch.dtype] = None,
        device=None,
        cache: bool = True,
        spectral_max_planes: Optional[int] = None,
        spectral_tol_abs: Optional[float] = None,
        spectral_tol_rel: Optional[float] = None,
        spectral_dominant_rel: Optional[float] = None,
        spectral_transition_n: Optional[int] = None,
        spectral_allow_degenerate: Optional[bool] = None,
        spectral_allow_truncated_degenerate: Optional[bool] = None,
    ):
        """Return a bivector exponential executor for compact-lane values."""
        if dtype is None:
            dtype = getattr(self, "dtype", torch.float32)
        if device is None:
            device = getattr(self, "device", None)
        input_layout = self._declared_layout(input_grades, input_layout)
        output_layout = self._optional_layout(output_grades, output_layout)
        if output_layout is None:
            output_layout = self.layout(range(0, self.n + 1, 2))
        if input_layout.grades != (2,):
            raise ValueError(f"bivector exp requires grade-2 input layout, got {input_layout.grades}")
        return self.planner.bivector_exp_executor_for_layouts(
            input_layout=input_layout,
            output_layout=output_layout,
            dtype=dtype,
            device=device,
            cache=cache,
            spectral_max_planes=spectral_max_planes,
            spectral_tol_abs=spectral_tol_abs,
            spectral_tol_rel=spectral_tol_rel,
            spectral_dominant_rel=spectral_dominant_rel,
            spectral_transition_n=spectral_transition_n,
            spectral_allow_degenerate=spectral_allow_degenerate,
            spectral_allow_truncated_degenerate=spectral_allow_truncated_degenerate,
        )

    def plan_sandwich_action(
        self,
        *,
        layout: Optional[GradeLayout] = None,
        dtype: Optional[torch.dtype] = None,
        device=None,
        cache: bool = True,
    ) -> FullSandwichActionHandle:
        """Return a full-layout sandwich action handle."""
        if dtype is None:
            dtype = getattr(self, "dtype", torch.float32)
        if device is None:
            device = getattr(self, "device", None)
        layout = self._full_sandwich_layout(layout)
        executor = self.planner.full_sandwich_action_executor_for_layout(
            layout=layout,
            dtype=dtype,
            device=device,
            cache=cache,
        )
        return FullSandwichActionHandle(executor)

    def plan_versor_action(
        self,
        *,
        grade: int,
        input_grades=None,
        output_grades=None,
        input_layout: Optional[GradeLayout] = None,
        output_layout: Optional[GradeLayout] = None,
        parameter_layout: Optional[GradeLayout] = None,
    ) -> VersorActionHandle:
        """Return a grade-1 or grade-2 versor action handle."""
        input_layout = self._declared_layout(input_grades, input_layout)
        output_layout = self._optional_layout(output_grades, output_layout) or input_layout
        parameter_layout = parameter_layout or self.layout((int(grade),))
        executor = VersorActionExecutor(
            self,
            grade=int(grade),
            input_layout=input_layout,
            output_layout=output_layout,
            parameter_layout=parameter_layout,
        )
        return VersorActionHandle(executor)

    def plan_multi_versor_action(
        self,
        *,
        grade: int,
        input_grades=None,
        output_grades=None,
        input_layout: Optional[GradeLayout] = None,
        output_layout: Optional[GradeLayout] = None,
        parameter_layout: Optional[GradeLayout] = None,
    ) -> MultiVersorActionHandle:
        """Return a weighted multi-versor action handle."""
        input_layout = self._declared_layout(input_grades, input_layout)
        output_layout = self._optional_layout(output_grades, output_layout) or input_layout
        parameter_layout = parameter_layout or self.layout((int(grade),))
        executor = MultiVersorActionExecutor(
            self,
            grade=int(grade),
            input_layout=input_layout,
            output_layout=output_layout,
            parameter_layout=parameter_layout,
        )
        return MultiVersorActionHandle(executor)

    def plan_paired_bivector_action(
        self,
        *,
        input_grades=None,
        output_grades=None,
        input_layout: Optional[GradeLayout] = None,
        output_layout: Optional[GradeLayout] = None,
        parameter_layout: Optional[GradeLayout] = None,
    ) -> PairedBivectorActionHandle:
        """Return an independent left/right bivector action handle."""
        input_layout = self._declared_layout(input_grades, input_layout)
        output_layout = self._optional_layout(output_grades, output_layout)
        parameter_layout = parameter_layout or self.layout((2,))
        plan = self.planner.paired_bivector_action_plan(
            input_layout=input_layout,
            output_layout=output_layout,
            parameter_layout=parameter_layout,
        )
        executor = PairedBivectorActionExecutor(
            self,
            input_layout=plan.input_layout,
            output_layout=plan.output_layout,
            parameter_layout=plan.parameter_layout,
            rotor_layout=plan.rotor_layout,
            middle_layout=plan.middle_layout,
        )
        return PairedBivectorActionHandle(executor)

    def projected_product(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        *,
        left_grades=None,
        right_grades=None,
        output_grades=None,
        left_layout=None,
        right_layout=None,
        output_layout=None,
        op: str = "gp",
        left_storage: LaneStorage | str | None = None,
        right_storage: LaneStorage | str | None = None,
        output_storage: LaneStorage | str = LaneStorage.COMPACT,
        return_layout: bool = False,
        pairwise: bool = False,
    ):
        """Execute a planner-resolved grade-restricted product."""
        resolved_left_storage = None if left_storage is None else normalize_lane_storage(left_storage)
        resolved_right_storage = None if right_storage is None else normalize_lane_storage(right_storage)
        resolved_output_storage = normalize_lane_storage(output_storage)
        if (
            left_layout is not None
            and right_layout is not None
            and output_layout is not None
            and resolved_left_storage == LaneStorage.COMPACT
            and resolved_right_storage == LaneStorage.COMPACT
        ):
            output = self._projected_product_with_explicit_layouts(
                A,
                B,
                op=op,
                left_grades=left_grades,
                right_grades=right_grades,
                output_grades=output_grades,
                left_layout=left_layout,
                right_layout=right_layout,
                output_layout=output_layout,
                pairwise=pairwise,
            )
            if resolved_output_storage is LaneStorage.CANONICAL:
                output = output_layout.full(output)
            return (output, output_layout) if return_layout else output

        request = self.planner.product_request(
            A,
            B,
            left_grades=left_grades,
            right_grades=right_grades,
            output_grades=output_grades,
            left_layout=left_layout,
            right_layout=right_layout,
            output_layout=output_layout,
            op=op,
            left_storage=resolved_left_storage,
            right_storage=resolved_right_storage,
            output_storage=resolved_output_storage,
        )
        executor = self.planner.product_executor_for_request(request)
        request.left.validate(A, name="left")
        request.right.validate(B, name="right")
        left_values = request.left.to_compact(A)
        right_values = request.right.to_compact(B)
        if pairwise:
            self._check_pairwise_prefix(left_values, right_values)
            output = executor.forward_pairwise_compact(left_values, right_values)
        else:
            self._check_elementwise_prefix(left_values, right_values)
            output = executor.forward_compact(left_values, right_values)
        if request.output.uses_canonical_storage:
            output = request.output.layout.full(output)
        return (output, request.output_layout) if return_layout else output

    def _projected_product_with_explicit_layouts(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        op: str,
        left_grades,
        right_grades,
        output_grades,
        left_layout: GradeLayout,
        right_layout: GradeLayout,
        output_layout: GradeLayout,
        pairwise: bool,
    ) -> torch.Tensor:
        """Fast path for already-resolved layout contracts.

        Layer/executor internals normally call products with concrete layouts.
        This avoids re-inferring the same request on every forward while still
        validating tensor lane widths and reusing the planner executor cache.
        """
        self._check_declared_layout(left_layout, left_grades, "left")
        self._check_declared_layout(right_layout, right_grades, "right")
        self._check_declared_layout(output_layout, output_grades, "output")
        if left.device != right.device:
            raise ValueError(f"product operands must be on the same device, got {left.device} and {right.device}")

        executor = self.planner.product_executor_for_layouts(
            op=normalize_product_op(op),
            left_layout=left_layout,
            right_layout=right_layout,
            output_layout=output_layout,
            dtype=torch.promote_types(left.dtype, right.dtype),
            device=left.device,
        )
        left_values = self._compact_values_for_layout(left, left_layout, "left")
        right_values = self._compact_values_for_layout(right, right_layout, "right")
        if pairwise:
            self._check_pairwise_prefix(left_values, right_values)
            return executor.forward_pairwise_compact(left_values, right_values)
        self._check_elementwise_prefix(left_values, right_values)
        return executor.forward_compact(left_values, right_values)

    def projected_geometric_product(self, A: torch.Tensor, B: torch.Tensor, **kwargs):
        """Projected geometric product convenience wrapper."""
        return self.projected_product(A, B, op="gp", **kwargs)

    def projected_wedge(self, A: torch.Tensor, B: torch.Tensor, **kwargs):
        """Projected wedge product convenience wrapper."""
        return self.projected_product(A, B, op="wedge", **kwargs)

    def symmetric_product(self, A: torch.Tensor, B: torch.Tensor, **kwargs):
        """Apply the parity-selected symmetric product route."""
        return self.projected_product(A, B, op="symmetric_product", **kwargs)

    def commutator_product(self, A: torch.Tensor, B: torch.Tensor, **kwargs):
        """Apply the unnormalized commutator product."""
        return self.projected_product(A, B, op="commutator_product", **kwargs)

    def anti_commutator_product(self, A: torch.Tensor, B: torch.Tensor, **kwargs):
        """Apply the unnormalized anti-commutator product."""
        return self.projected_product(A, B, op="anti_commutator_product", **kwargs)

    def projected_left_contraction(self, A: torch.Tensor, B: torch.Tensor, **kwargs):
        """Projected left contraction convenience wrapper."""
        return self.projected_product(A, B, op="left_contraction", **kwargs)

    def projected_right_contraction(self, A: torch.Tensor, B: torch.Tensor, **kwargs):
        """Projected right contraction convenience wrapper."""
        return self.projected_product(A, B, op="right_contraction", **kwargs)

    def left_contraction(self, A: torch.Tensor, B: torch.Tensor, **kwargs):
        """Apply left contraction through a planned product executor."""
        return self.projected_product(A, B, op="left_contraction", **kwargs)

    def right_contraction(self, A: torch.Tensor, B: torch.Tensor, **kwargs):
        """Apply right contraction through a planned product executor."""
        return self.projected_product(A, B, op="right_contraction", **kwargs)

    def planned_unary(
        self,
        values: torch.Tensor,
        *,
        op: str,
        input_grades=None,
        output_grades=None,
        input_layout: Optional[GradeLayout] = None,
        output_layout: Optional[GradeLayout] = None,
        input_storage: LaneStorage | str | None = None,
        output_storage: LaneStorage | str = LaneStorage.COMPACT,
        return_layout: bool = False,
    ):
        """Execute a unary operation through the planner."""
        request = self.planner.unary_request(
            values,
            op=op,
            input_grades=input_grades,
            output_grades=output_grades,
            input_layout=input_layout,
            output_layout=output_layout,
            input_storage=input_storage,
            output_storage=output_storage,
        )
        executor = self.planner.unary_executor_for_request(request)
        output = executor.forward_compact(request.input.to_compact(values))
        if request.output.uses_canonical_storage:
            output = request.output.layout.full(output)
        return (output, request.output_layout) if return_layout else output

    def signature_norm_squared(
        self,
        values: torch.Tensor,
        *,
        grades=None,
        input_grades=None,
        layout: Optional[GradeLayout] = None,
        input_layout: Optional[GradeLayout] = None,
    ) -> torch.Tensor:
        """Return ``<values reverse(values)>_0`` through a planned diagonal executor."""
        if input_grades is None:
            input_grades = grades
        if input_layout is None:
            input_layout = layout
        resolved = self._declared_layout(input_grades, input_layout)
        active_values = self._compact_values_for_layout(values, resolved, "signature_norm_squared values")
        executor = self.planner.signature_norm_squared_executor_for_layout(
            input_layout=resolved,
            dtype=active_values.dtype,
            device=active_values.device,
        )
        return executor(active_values)

    def pseudoscalar_product(
        self,
        values: torch.Tensor,
        *,
        input_grades=None,
        output_grades=None,
        input_layout: Optional[GradeLayout] = None,
        output_layout: Optional[GradeLayout] = None,
        output_storage: LaneStorage | str = LaneStorage.COMPACT,
        return_layout: bool = False,
    ) -> torch.Tensor:
        """Apply right multiplication by the unit pseudoscalar."""
        resolved_output_storage = normalize_lane_storage(output_storage)
        input_layout = self._declared_layout(input_grades, input_layout)
        output_layout = self._optional_layout(output_grades, output_layout)
        active_values = self._compact_values_for_layout(values, input_layout, "pseudoscalar_product values")
        executor = self.planner.pseudoscalar_product_executor_for_layout(
            input_layout=input_layout,
            output_layout=output_layout,
            dtype=active_values.dtype,
            device=active_values.device,
        )
        output = executor(active_values)
        if resolved_output_storage is LaneStorage.CANONICAL:
            output = executor.output_layout.full(output)
        return (output, executor.output_layout) if return_layout else output

    def bivector_exp(
        self,
        values: torch.Tensor,
        *,
        input_grades=None,
        output_grades=None,
        input_layout: Optional[GradeLayout] = None,
        output_layout: Optional[GradeLayout] = None,
        spectral_max_planes: Optional[int] = None,
        spectral_tol_abs: Optional[float] = None,
        spectral_tol_rel: Optional[float] = None,
        spectral_dominant_rel: Optional[float] = None,
        spectral_transition_n: Optional[int] = None,
        spectral_allow_degenerate: Optional[bool] = None,
        spectral_allow_truncated_degenerate: Optional[bool] = None,
        output_storage: LaneStorage | str = LaneStorage.COMPACT,
        return_layout: bool = False,
    ) -> torch.Tensor:
        """Exponentiate a declared bivector through a planner-owned executor."""
        resolved_output_storage = normalize_lane_storage(output_storage)
        input_layout, output_layout = self._bivector_exp_layouts(
            values,
            input_grades=input_grades,
            output_grades=output_grades,
            input_layout=input_layout,
            output_layout=output_layout,
        )
        active_values = self._compact_values_for_layout(values, input_layout, "exp values")
        executor = self.planner.bivector_exp_executor_for_layouts(
            input_layout=input_layout,
            output_layout=output_layout,
            dtype=active_values.dtype,
            device=active_values.device,
            spectral_max_planes=spectral_max_planes,
            spectral_tol_abs=spectral_tol_abs,
            spectral_tol_rel=spectral_tol_rel,
            spectral_dominant_rel=spectral_dominant_rel,
            spectral_transition_n=spectral_transition_n,
            spectral_allow_degenerate=spectral_allow_degenerate,
            spectral_allow_truncated_degenerate=spectral_allow_truncated_degenerate,
        )
        output = executor(active_values)
        if resolved_output_storage is LaneStorage.CANONICAL:
            output = output_layout.full(output)
        return (output, output_layout) if return_layout else output

    def blade_inverse(
        self,
        blade: torch.Tensor,
        *,
        grades=None,
        input_grades=None,
        layout: Optional[GradeLayout] = None,
        input_layout: Optional[GradeLayout] = None,
        return_layout: bool = False,
    ) -> torch.Tensor:
        """Return ``reverse(blade) / <blade reverse(blade)>_0`` through planned executors."""
        if input_grades is None:
            input_grades = grades
        if input_layout is None:
            input_layout = layout
        resolved = self._declared_layout(input_grades, input_layout)
        active_blade = self._compact_values_for_layout(blade, resolved, "blade_inverse values")
        blade_rev = self.planned_unary(
            active_blade,
            op="reverse",
            input_layout=resolved,
            output_layout=resolved,
        )
        scalar = signed_clamp_min(self.signature_norm_squared(active_blade, input_layout=resolved), self.eps_sq)
        output = blade_rev / scalar
        return (output, resolved) if return_layout else output

    def blade_project(
        self,
        values: torch.Tensor,
        blade: torch.Tensor,
        *,
        input_grades=None,
        blade_grades=None,
        output_grades=None,
        input_layout: Optional[GradeLayout] = None,
        blade_layout: Optional[GradeLayout] = None,
        output_layout: Optional[GradeLayout] = None,
        return_layout: bool = False,
    ) -> torch.Tensor:
        """Project ``values`` onto a blade subspace through planned products."""
        input_layout = self._declared_layout(input_grades, input_layout)
        blade_layout = self._declared_layout(blade_grades, blade_layout)
        output_layout = self._optional_layout(output_grades, output_layout) or input_layout
        active_values = self._compact_values_for_layout(values, input_layout, "blade_project values")
        active_blade = self._compact_values_for_layout(blade, blade_layout, "blade_project blade")
        inner_layout = self.layout(
            expand_output_grades(input_layout.grades, blade_layout.grades, self.n, op="symmetric_product")
        )
        inner = self.symmetric_product(
            active_values,
            active_blade,
            left_layout=input_layout,
            right_layout=blade_layout,
            output_layout=inner_layout,
        )
        blade_inv = self.blade_inverse(active_blade, input_layout=blade_layout)
        output = self.geometric_product(
            inner,
            blade_inv,
            left_layout=inner_layout,
            right_layout=blade_layout,
            output_layout=output_layout,
        )
        return (output, output_layout) if return_layout else output

    def blade_reject(
        self,
        values: torch.Tensor,
        blade: torch.Tensor,
        *,
        input_grades=None,
        blade_grades=None,
        output_grades=None,
        input_layout: Optional[GradeLayout] = None,
        blade_layout: Optional[GradeLayout] = None,
        output_layout: Optional[GradeLayout] = None,
        return_layout: bool = False,
    ) -> torch.Tensor:
        """Reject ``values`` from a blade subspace through planned projection."""
        input_layout = self._declared_layout(input_grades, input_layout)
        output_layout = self._optional_layout(output_grades, output_layout) or input_layout
        if output_layout != input_layout:
            raise ValueError("blade_reject output_layout must match input_layout for subtraction")
        active_values = self._compact_values_for_layout(values, input_layout, "blade_reject values")
        projection = self.blade_project(
            active_values,
            blade,
            input_layout=input_layout,
            blade_grades=blade_grades,
            blade_layout=blade_layout,
            output_layout=output_layout,
        )
        output = active_values - projection
        return (output, output_layout) if return_layout else output

    def reflect(
        self,
        values: torch.Tensor,
        normal: torch.Tensor,
        *,
        input_grades=None,
        normal_grades=None,
        output_grades=None,
        input_layout: Optional[GradeLayout] = None,
        normal_layout: Optional[GradeLayout] = None,
        output_layout: Optional[GradeLayout] = None,
        return_layout: bool = False,
    ) -> torch.Tensor:
        """Reflect ``values`` through a vector normal using planned products."""
        return self.versor_product(
            normal,
            values,
            versor_grades=normal_grades,
            input_grades=input_grades,
            output_grades=output_grades,
            versor_layout=normal_layout,
            input_layout=input_layout,
            output_layout=output_layout,
            return_layout=return_layout,
        )

    def versor_product(
        self,
        versor: torch.Tensor,
        values: torch.Tensor,
        *,
        versor_grades=None,
        input_grades=None,
        output_grades=None,
        versor_layout: Optional[GradeLayout] = None,
        input_layout: Optional[GradeLayout] = None,
        output_layout: Optional[GradeLayout] = None,
        return_layout: bool = False,
    ) -> torch.Tensor:
        """Apply ``grade_involution(versor) * values * inverse(versor)`` through planned products."""
        versor_layout = self._declared_layout(versor_grades, versor_layout)
        input_layout = self._declared_layout(input_grades, input_layout)
        output_layout = self._optional_layout(output_grades, output_layout) or input_layout
        active_versor = self._compact_values_for_layout(versor, versor_layout, "versor_product versor")
        active_values = self._compact_values_for_layout(values, input_layout, "versor_product values")
        versor_hat = self.grade_involution(
            active_versor,
            input_layout=versor_layout,
            output_layout=versor_layout,
        )
        versor_inv = self.blade_inverse(active_versor, input_layout=versor_layout)
        middle_layout = self.layout(expand_output_grades(versor_layout.grades, input_layout.grades, self.n, op="gp"))
        middle = self.geometric_product(
            versor_hat,
            active_values,
            left_layout=versor_layout,
            right_layout=input_layout,
            output_layout=middle_layout,
        )
        output = self.geometric_product(
            middle,
            versor_inv,
            left_layout=middle_layout,
            right_layout=versor_layout,
            output_layout=output_layout,
        )
        return (output, output_layout) if return_layout else output

    def planned_linear_action(
        self,
        values: torch.Tensor,
        matrix: torch.Tensor,
        *,
        input_grades=None,
        output_grades=None,
        input_layout: Optional[GradeLayout] = None,
        output_layout: Optional[GradeLayout] = None,
        return_layout: bool = False,
    ):
        """Execute a planned vector-space linear action."""
        input_layout = self._declared_layout(input_grades, input_layout)
        output_layout = self._optional_layout(output_grades, output_layout)
        if output_layout is None:
            output_layout = input_layout
        executor = GradedLinearActionExecutor(input_layout=input_layout, output_layout=output_layout)
        action_values = values if values.shape[-1] == input_layout.dim else input_layout.compact(values)
        output = executor(action_values, matrix)
        return (output, output_layout) if return_layout else output

    def sandwich_action_matrices(
        self,
        left: torch.Tensor,
        right: torch.Tensor = None,
        *,
        layout: Optional[GradeLayout] = None,
    ) -> torch.Tensor:
        """Return full-layout sandwich action matrices through the planner."""
        layout = self._full_sandwich_layout(layout)
        left = self._compact_values_for_layout(left, layout, "sandwich_action_matrices left")
        if right is None:
            right = self.reverse(left, input_layout=layout, output_layout=layout)
        else:
            right = self._compact_values_for_layout(right, layout, "sandwich_action_matrices right")
        left, right = self._promote_action_tensors(left, right)
        handle = self.plan_sandwich_action(layout=layout, dtype=left.dtype, device=left.device)
        return handle.checked_action_matrices(left, right)

    def sandwich_product(
        self,
        left: torch.Tensor,
        values: torch.Tensor,
        right: torch.Tensor = None,
        *,
        layout: Optional[GradeLayout] = None,
    ) -> torch.Tensor:
        """Apply one full-layout sandwich action per leading batch item."""
        layout = self._full_sandwich_layout(layout)
        left = self._compact_values_for_layout(left, layout, "sandwich_product left")
        values = self._compact_values_for_layout(values, layout, "sandwich_product values")
        if right is None:
            right = self.reverse(left, input_layout=layout, output_layout=layout)
        else:
            right = self._compact_values_for_layout(right, layout, "sandwich_product right")
        left, values, right = self._promote_action_tensors(left, values, right)
        handle = self.plan_sandwich_action(layout=layout, dtype=left.dtype, device=left.device)
        return handle.checked_batched(left, values, right)

    def per_channel_sandwich(
        self,
        left: torch.Tensor,
        values: torch.Tensor,
        right: torch.Tensor = None,
        *,
        layout: Optional[GradeLayout] = None,
    ) -> torch.Tensor:
        """Apply one full-layout sandwich action per channel."""
        layout = self._full_sandwich_layout(layout)
        left = self._compact_values_for_layout(left, layout, "per_channel_sandwich left")
        values = self._compact_values_for_layout(values, layout, "per_channel_sandwich values")
        if right is None:
            right = self.reverse(left, input_layout=layout, output_layout=layout)
        else:
            right = self._compact_values_for_layout(right, layout, "per_channel_sandwich right")
        left, values, right = self._promote_action_tensors(left, values, right)
        handle = self.plan_sandwich_action(layout=layout, dtype=left.dtype, device=left.device)
        return handle.checked_per_channel(left, values, right)

    def multi_rotor_sandwich(
        self,
        left: torch.Tensor,
        values: torch.Tensor,
        right: torch.Tensor = None,
        *,
        layout: Optional[GradeLayout] = None,
    ) -> torch.Tensor:
        """Apply every full-layout sandwich action to every input channel."""
        layout = self._full_sandwich_layout(layout)
        left = self._compact_values_for_layout(left, layout, "multi_rotor_sandwich left")
        values = self._compact_values_for_layout(values, layout, "multi_rotor_sandwich values")
        if right is None:
            right = self.reverse(left, input_layout=layout, output_layout=layout)
        else:
            right = self._compact_values_for_layout(right, layout, "multi_rotor_sandwich right")
        left, values, right = self._promote_action_tensors(left, values, right)
        handle = self.plan_sandwich_action(layout=layout, dtype=left.dtype, device=left.device)
        return handle.checked_multi(left, values, right)

    def versor_action(self, values: torch.Tensor, weights: torch.Tensor, **kwargs):
        """Execute a planned grade-1 or grade-2 versor action."""
        grade = int(kwargs["grade"])
        input_layout = self._declared_layout(kwargs.get("input_grades"), kwargs.get("input_layout"))
        output_layout = self._optional_layout(kwargs.get("output_grades"), kwargs.get("output_layout")) or input_layout
        parameter_layout = kwargs.get("parameter_layout") or self.layout((grade,))
        handle = self.plan_versor_action(
            grade=grade,
            input_layout=input_layout,
            output_layout=output_layout,
            parameter_layout=parameter_layout,
        )
        action_values = values if values.shape[-1] == input_layout.dim else input_layout.compact(values)
        return handle.checked(action_values, weights)

    def multi_versor_action(self, values: torch.Tensor, weights: torch.Tensor, mix: torch.Tensor, **kwargs):
        """Execute a planned weighted grade-1 or grade-2 versor action."""
        grade = int(kwargs["grade"])
        input_layout = self._declared_layout(kwargs.get("input_grades"), kwargs.get("input_layout"))
        output_layout = self._optional_layout(kwargs.get("output_grades"), kwargs.get("output_layout")) or input_layout
        parameter_layout = kwargs.get("parameter_layout") or self.layout((grade,))
        handle = self.plan_multi_versor_action(
            grade=grade,
            input_layout=input_layout,
            output_layout=output_layout,
            parameter_layout=parameter_layout,
        )
        action_values = values if values.shape[-1] == input_layout.dim else input_layout.compact(values)
        return handle.checked(action_values, weights, mix)

    def paired_bivector_action(
        self,
        values: torch.Tensor,
        left_weights: torch.Tensor,
        right_weights: torch.Tensor,
        channel_to_pair: torch.Tensor,
        **kwargs,
    ):
        """Execute a planned independent left/right bivector rotor action."""
        input_layout = self._declared_layout(kwargs.get("input_grades"), kwargs.get("input_layout"))
        parameter_layout = kwargs.get("parameter_layout") or self.layout((2,))
        handle = self.plan_paired_bivector_action(
            input_layout=input_layout,
            parameter_layout=parameter_layout,
            output_grades=kwargs.get("output_grades"),
            output_layout=kwargs.get("output_layout"),
        )
        action_values = values if values.shape[-1] == input_layout.dim else input_layout.compact(values)
        return handle.checked(action_values, left_weights, right_weights, channel_to_pair)

    def grade_norms(self, values: torch.Tensor, *, input_grades=None, layout: GradeLayout = None) -> torch.Tensor:
        """Return per-grade coefficient norms for declared-layout values."""
        resolved = self._declared_layout(input_grades, layout)
        return lane_grade_norms(self, values, layout=resolved)

    def multivector(self, values: torch.Tensor, **kwargs):
        """Return a debug-only multivector formatter for ``values``."""
        from clifra.core.formatting import Multivector

        return Multivector(self, values, **kwargs)

    def format_multivector(self, values: torch.Tensor, **kwargs) -> str:
        """Format ``values`` as basis-blade terms for debugging."""
        from clifra.core.formatting import format_multivector

        return format_multivector(self, values, **kwargs)

    def _declared_layout(self, grades, layout: GradeLayout | None) -> GradeLayout:
        if layout is not None:
            return layout
        if grades is not None:
            return self.layout(grades)
        return self.default_layout()

    def _optional_layout(self, grades, layout: GradeLayout | None) -> GradeLayout | None:
        if layout is not None:
            return layout
        if grades is not None:
            return self.layout(grades)
        return None

    def _product_output_layout(
        self,
        *,
        op: str,
        left_layout: GradeLayout,
        right_layout: GradeLayout,
        output_grades,
        output_layout: GradeLayout | None,
    ) -> GradeLayout:
        resolved = self._optional_layout(output_grades, output_layout)
        if resolved is not None:
            return resolved
        full_grades = tuple(range(self.n + 1))
        if left_layout.grades == full_grades and right_layout.grades == full_grades:
            return self.planner.full_layout()
        return self.layout(expand_output_grades(left_layout.grades, right_layout.grades, self.n, op=op))

    def _full_sandwich_layout(self, layout: Optional[GradeLayout]) -> GradeLayout:
        resolved = self.planner.full_layout() if layout is None else layout
        full_grades = tuple(range(self.n + 1))
        if resolved.grades != full_grades:
            raise ValueError(f"full sandwich action requires full layout {full_grades}, got {resolved.grades}")
        return resolved

    @staticmethod
    def _promote_action_tensors(*values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if not values:
            return ()
        device = values[0].device
        dtype = values[0].dtype
        for value in values[1:]:
            if value.device != device:
                raise ValueError(f"action tensors must share a device, got {device} and {value.device}")
            dtype = torch.promote_types(dtype, value.dtype)
        return tuple(value if value.dtype == dtype else value.to(dtype=dtype) for value in values)

    def _bivector_exp_layouts(
        self,
        values: torch.Tensor,
        *,
        input_grades,
        output_grades,
        input_layout: Optional[GradeLayout],
        output_layout: Optional[GradeLayout],
    ) -> tuple[GradeLayout, GradeLayout]:
        if self.n < 2:
            raise ValueError("bivector exp requires an algebra with at least two basis vectors")
        full_layout = self.planner.full_layout()
        bivector_layout = self.layout((2,))
        explicit_input = input_layout is not None or input_grades is not None
        explicit_output = output_layout is not None or output_grades is not None

        if explicit_input:
            resolved_input = self._declared_layout(input_grades, input_layout)
        elif values.shape[-1] == bivector_layout.dim:
            resolved_input = bivector_layout
        elif values.shape[-1] == full_layout.dim:
            resolved_input = bivector_layout
        else:
            resolved_input = self.default_layout()

        if resolved_input.grades != (2,):
            raise ValueError(f"bivector exp requires grade-2 input layout, got {resolved_input.grades}")

        if explicit_output:
            resolved_output = self._optional_layout(output_grades, output_layout)
        elif not explicit_input and values.shape[-1] == full_layout.dim:
            resolved_output = full_layout
        else:
            resolved_output = self.layout(range(0, self.n + 1, 2))

        if resolved_output is None:
            raise ValueError("bivector exp could not resolve an output layout")
        if resolved_output.spec != resolved_input.spec:
            raise ValueError(f"output layout signature {resolved_output.spec} does not match input signature")
        return resolved_input, resolved_output

    @staticmethod
    def _check_declared_layout(layout: GradeLayout, grades, side: str) -> None:
        if grades is not None and layout.grades != normalize_grades(grades, layout.spec.n, name=f"{side}_grades"):
            raise ValueError(f"{side}_layout and {side}_grades disagree")

    @staticmethod
    def _compact_values_for_layout(values: torch.Tensor, layout: GradeLayout, name: str) -> torch.Tensor:
        if values.ndim < 1:
            raise ValueError(f"{name} must include a coefficient lane dimension, got shape {tuple(values.shape)}")
        if values.shape[-1] == layout.dim:
            return values
        if values.shape[-1] == layout.spec.dim:
            return layout.compact(values)
        raise ValueError(
            f"{name} last dimension must be {layout.dim} for grades {layout.grades} or "
            f"{layout.spec.dim} full lanes, got {values.shape[-1]}"
        )

    @staticmethod
    def _check_elementwise_prefix(left: torch.Tensor, right: torch.Tensor) -> None:
        try:
            torch.broadcast_shapes(left.shape[:-1], right.shape[:-1])
        except RuntimeError as exc:
            raise ValueError(
                "projected_product elementwise prefixes must be broadcastable; "
                f"got left prefix {tuple(left.shape[:-1])} and right prefix {tuple(right.shape[:-1])}. "
                "Use pairwise=True when left and right have distinct item axes."
            ) from exc

    @staticmethod
    def _check_pairwise_prefix(left: torch.Tensor, right: torch.Tensor) -> None:
        if left.ndim < 2 or right.ndim < 2:
            raise ValueError(
                "pairwise projected_product requires explicit item axes before the lane dimension; "
                f"got left shape {tuple(left.shape)} and right shape {tuple(right.shape)}"
            )
        try:
            torch.broadcast_shapes(left.shape[:-2], right.shape[:-2])
        except RuntimeError as exc:
            raise ValueError(
                "pairwise projected_product batch prefixes must be broadcastable; "
                f"got left prefix {tuple(left.shape[:-2])} and right prefix {tuple(right.shape[:-2])}"
            ) from exc
