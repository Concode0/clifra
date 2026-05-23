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
from clifra.core.foundation.basis import normalize_grades
from clifra.core.foundation.layout import AlgebraSpec, GradeLayout
from clifra.core.planning.layouts import normalize_product_op
from clifra.core.storage import compact_grade_norms
from clifra.core.storage import hermitian_signs as _hermitian_signs


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

    def hermitian_signs(
        self,
        layout: Optional[GradeLayout] = None,
        *,
        grades: Optional[Iterable[int]] = None,
        device=None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return Hermitian signs for a declared layout."""
        return _hermitian_signs(self, layout=layout, grades=grades, device=device, dtype=dtype)

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
        left_active_lanes: bool = True,
        right_active_lanes: bool = True,
        active_output: bool = True,
        return_layout: bool = False,
        pairwise: bool = False,
    ):
        """Execute a planner-resolved grade-restricted product."""
        if (
            left_layout is not None
            and right_layout is not None
            and output_layout is not None
            and left_active_lanes
            and right_active_lanes
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
            left_active_lanes=left_active_lanes,
            right_active_lanes=right_active_lanes,
        )
        executor = self.planner.product_executor_for_request(request)
        request.left_value.validate_tensor(A, name="left")
        request.right_value.validate_tensor(B, name="right")
        left_values = request.left_value.active_values(A)
        right_values = request.right_value.active_values(B)
        if pairwise:
            self._check_pairwise_prefix(left_values, right_values)
            output = executor.forward_pairwise_compact(left_values, right_values)
        else:
            self._check_elementwise_prefix(left_values, right_values)
            output = executor.forward_compact(left_values, right_values)
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
        left_values = self._active_values_for_layout(left, left_layout, "left")
        right_values = self._active_values_for_layout(right, right_layout, "right")
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

    def projected_inner_product(self, A: torch.Tensor, B: torch.Tensor, **kwargs):
        """Projected inner product convenience wrapper."""
        return self.projected_product(A, B, op="inner", **kwargs)

    def projected_commutator(self, A: torch.Tensor, B: torch.Tensor, **kwargs):
        """Projected commutator convenience wrapper."""
        return self.projected_product(A, B, op="commutator", **kwargs)

    def projected_anti_commutator(self, A: torch.Tensor, B: torch.Tensor, **kwargs):
        """Projected anti-commutator convenience wrapper."""
        return self.projected_product(A, B, op="anti_commutator", **kwargs)

    def planned_unary(
        self,
        values: torch.Tensor,
        *,
        op: str,
        input_grades=None,
        output_grades=None,
        input_layout: Optional[GradeLayout] = None,
        output_layout: Optional[GradeLayout] = None,
        input_active_lanes: bool = True,
        active_output: bool = True,
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
            input_active_lanes=input_active_lanes,
        )
        executor = self.planner.unary_executor_for_request(request)
        output = executor.forward_compact(request.input_value.active_values(values))
        return (output, request.output_layout) if return_layout else output

    def planned_linear_action(
        self,
        values: torch.Tensor,
        matrix: torch.Tensor,
        *,
        input_grades=None,
        output_grades=None,
        input_layout: Optional[GradeLayout] = None,
        output_layout: Optional[GradeLayout] = None,
        input_active_lanes: bool = True,
        active_output: bool = True,
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

    def versor_action(self, values: torch.Tensor, weights: torch.Tensor, **kwargs):
        """Execute a planned grade-1 or grade-2 versor action."""
        grade = int(kwargs["grade"])
        input_layout = self._declared_layout(kwargs.get("input_grades"), kwargs.get("input_layout"))
        output_layout = self._optional_layout(kwargs.get("output_grades"), kwargs.get("output_layout")) or input_layout
        parameter_layout = kwargs.get("parameter_layout") or self.layout((grade,))
        executor = VersorActionExecutor(
            self,
            grade=grade,
            input_layout=input_layout,
            output_layout=output_layout,
            parameter_layout=parameter_layout,
        )
        action_values = values if values.shape[-1] == input_layout.dim else input_layout.compact(values)
        return executor(action_values, weights)

    def multi_versor_action(self, values: torch.Tensor, weights: torch.Tensor, mix: torch.Tensor, **kwargs):
        """Execute a planned weighted grade-1 or grade-2 versor action."""
        grade = int(kwargs["grade"])
        input_layout = self._declared_layout(kwargs.get("input_grades"), kwargs.get("input_layout"))
        output_layout = self._optional_layout(kwargs.get("output_grades"), kwargs.get("output_layout")) or input_layout
        parameter_layout = kwargs.get("parameter_layout") or self.layout((grade,))
        executor = MultiVersorActionExecutor(
            self,
            grade=grade,
            input_layout=input_layout,
            output_layout=output_layout,
            parameter_layout=parameter_layout,
        )
        action_values = values if values.shape[-1] == input_layout.dim else input_layout.compact(values)
        return executor(action_values, weights, mix)

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
        output_layout = self._optional_layout(kwargs.get("output_grades"), kwargs.get("output_layout"))
        parameter_layout = kwargs.get("parameter_layout") or self.layout((2,))
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
        action_values = values if values.shape[-1] == input_layout.dim else input_layout.compact(values)
        return executor(action_values, left_weights, right_weights, channel_to_pair)

    def grade_norms(self, values: torch.Tensor, *, input_grades=None, layout: GradeLayout = None) -> torch.Tensor:
        """Return per-grade coefficient norms for declared-layout values."""
        resolved = self._declared_layout(input_grades, layout)
        return compact_grade_norms(self, values, resolved)

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

    @staticmethod
    def _check_declared_layout(layout: GradeLayout, grades, side: str) -> None:
        if grades is not None and layout.grades != normalize_grades(grades, layout.spec.n, name=f"{side}_grades"):
            raise ValueError(f"{side}_layout and {side}_grades disagree")

    @staticmethod
    def _active_values_for_layout(values: torch.Tensor, layout: GradeLayout, name: str) -> torch.Tensor:
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
