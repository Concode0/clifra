# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Grade-aware planner from algebraic intent to static executors."""

from __future__ import annotations

import torch

from clifra.core.foundation.basis import operation_coefficient
from clifra.core.foundation.layout import AlgebraSpec, GradeLayout
from clifra.core.planning.action import (
    LinearActionPlan,
    PairedBivectorActionPlan,
    VersorActionPlan,
    build_linear_action_plan,
    build_paired_bivector_action_plan,
    build_versor_action_plan,
)
from clifra.core.planning.decomposition import BivectorDecompositionPlan, build_bivector_decomposition_plan
from clifra.core.planning.layouts import ProductRequest, build_product_request, normalize_product_op
from clifra.core.planning.policy import (
    validate_grades_cost,
    validate_layout_cost,
    validate_product_grades_cost,
    validate_product_request,
    validate_unary_request,
)
from clifra.core.planning.product import GradeProductExecutor, build_grade_product_plan_from_request
from clifra.core.planning.tree import build_grade_plan_tree
from clifra.core.planning.unary import (
    GradeUnaryExecutor,
    UnaryRequest,
    build_unary_plan_from_request,
    build_unary_request,
    normalize_unary_op,
)
from clifra.core.storage import ValueLayout


class GradePlanner:
    """Owns layout and product-plan lowering for one algebra instance.

    The planner is deliberately not an ``nn.Module``. It builds static
    executor modules keyed by signature, grades, dtype, and device, while the
    algebra remains the source of truth for buffers and dense reference paths.
    """

    def __init__(self, algebra):
        self.algebra = algebra
        self.spec = AlgebraSpec.from_algebra(algebra)
        self._product_executors = {}
        self._unary_executors = {}

    def layout(self, grades):
        """Return the compact layout for ``grades``."""
        return self.spec.layout(validate_grades_cost(self.algebra, self.spec, grades))

    def full_layout(self) -> GradeLayout:
        """Return the full dense basis layout."""
        return self.spec.full_layout()

    def grade_indices(self, grades, *, device=None) -> torch.Tensor:
        """Return canonical dense basis indices for ``grades``."""
        if device is None:
            device = getattr(self.algebra, "device", None)
        return self.layout(grades).indices_tensor(device=device)

    def convert_values(self, values: torch.Tensor, *, source_layout: GradeLayout, target_layout: GradeLayout):
        """Convert compact values between layouts without full dense materialization."""
        return target_layout.convert(values, source_layout)

    def bivector_squared_signs(self, *, device=None, dtype: torch.dtype = None) -> torch.Tensor:
        """Return ``(e_ab)^2`` signs in canonical grade-2 layout order."""
        if device is None:
            device = getattr(self.algebra, "device", None)
        if dtype is None:
            dtype = getattr(self.algebra, "dtype", torch.float32)
        layout = self.layout((2,))
        signs = [
            operation_coefficient(index, index, self.spec.p, self.spec.q, self.spec.r, "gp")
            for index in layout.basis_indices
        ]
        return torch.tensor(signs, dtype=dtype, device=device)

    def clear_cache(self) -> None:
        """Drop cached executor modules."""
        self._product_executors.clear()
        self._unary_executors.clear()

    def _apply(self, fn):
        """Apply a PyTorch module-style transform to cached executor buffers."""
        product_executors = list(self._product_executors.values())
        self._product_executors.clear()
        for executor in product_executors:
            executor._apply(fn)
            self._product_executors[self._product_cache_key(executor)] = executor

        unary_executors = list(self._unary_executors.values())
        self._unary_executors.clear()
        for executor in unary_executors:
            executor._apply(fn)
            self._unary_executors[self._unary_cache_key(executor)] = executor
        return self

    def product_executor(
        self,
        *,
        op: str,
        left_grades,
        right_grades,
        output_grades,
        dtype,
        device,
        cache: bool = True,
    ):
        """Return a cached static executor for a projected bilinear product."""
        left_grades, right_grades, output_grades = validate_product_grades_cost(
            self.algebra,
            self.spec,
            op=op,
            left_grades=left_grades,
            right_grades=right_grades,
            output_grades=output_grades,
        )
        request = ProductRequest(
            spec=self.spec,
            op=normalize_product_op(op),
            left_value=ValueLayout.active(self.spec, self.layout(left_grades)),
            right_value=ValueLayout.active(self.spec, self.layout(right_grades)),
            output_value=ValueLayout.active(self.spec, self.layout(output_grades)),
            dtype=dtype,
            device=torch.device(device),
        )
        return self.product_executor_for_request(request, cache=cache)

    def product_request(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        op: str = "gp",
        left_grades=None,
        right_grades=None,
        output_grades=None,
        left_layout: GradeLayout = None,
        right_layout: GradeLayout = None,
        output_layout: GradeLayout = None,
        left_active_lanes: bool = False,
        right_active_lanes: bool = False,
    ) -> ProductRequest:
        """Normalize product intent into a static request without executing tensors."""
        left_grades = self._default_operand_grades(left_grades, left_layout)
        right_grades = self._default_operand_grades(right_grades, right_layout)
        self._validate_product_grade_cost_before_layouts(
            op=op,
            left_grades=left_grades,
            right_grades=right_grades,
            output_grades=output_grades,
            left_layout=left_layout,
            right_layout=right_layout,
            output_layout=output_layout,
        )
        request = build_product_request(
            self.spec,
            left,
            right,
            op=op,
            left_grades=left_grades,
            right_grades=right_grades,
            output_grades=output_grades,
            left_layout=left_layout,
            right_layout=right_layout,
            output_layout=output_layout,
            left_active_lanes=left_active_lanes,
            right_active_lanes=right_active_lanes,
        )
        validate_product_request(self.algebra, request)
        return request

    def product_executor_for_request(self, request: ProductRequest, *, cache: bool = True) -> GradeProductExecutor:
        """Return an executor for an already normalized product request."""
        validate_product_request(self.algebra, request)
        key = request.cache_key
        executor = self._product_executors.get(key) if cache else None
        if executor is None:
            plan = build_grade_product_plan_from_request(request)
            executor = GradeProductExecutor(plan)
            if cache:
                self._product_executors[key] = executor
        return executor

    def product_tree(self, *, op: str, left_grades, right_grades, output_grades=None):
        """Return planner-only grade tree metadata for a product route."""
        return build_grade_plan_tree(
            self.spec,
            op=op,
            left_grades=left_grades,
            right_grades=right_grades,
            output_grades=output_grades,
        )

    def unary_request(
        self,
        values: torch.Tensor,
        *,
        op: str,
        input_grades=None,
        output_grades=None,
        input_layout: GradeLayout = None,
        output_layout: GradeLayout = None,
        input_active_lanes: bool = False,
    ) -> UnaryRequest:
        """Normalize unary intent into a static request without executing tensors."""
        if not (op == "grade_projection" and output_grades is not None):
            input_grades = self._default_operand_grades(input_grades, input_layout)
        request = build_unary_request(
            self.spec,
            values,
            op=op,
            input_grades=input_grades,
            output_grades=output_grades,
            input_layout=input_layout,
            output_layout=output_layout,
            input_active_lanes=input_active_lanes,
        )
        validate_unary_request(self.algebra, request)
        return request

    def unary_executor(
        self,
        *,
        op: str,
        input_grades,
        output_grades=None,
        dtype,
        device,
        cache: bool = True,
    ) -> GradeUnaryExecutor:
        """Return a cached static executor for a unary operation."""
        op = normalize_unary_op(op)
        if op == "grade_projection" and output_grades is None:
            raise ValueError("output_grades is required for grade_projection")
        input_layout = self.layout(input_grades)
        output_layout = input_layout if output_grades is None else self.layout(output_grades)
        request = UnaryRequest(
            spec=self.spec,
            op=op,
            input_value=ValueLayout.active(self.spec, input_layout),
            output_value=ValueLayout.active(self.spec, output_layout),
            dtype=dtype,
            device=torch.device(device),
        )
        return self.unary_executor_for_request(request, cache=cache)

    def linear_action_plan(
        self,
        *,
        input_layout: GradeLayout,
        output_layout: GradeLayout = None,
    ) -> LinearActionPlan:
        """Return a plan-only contract for a grade-preserving linear action."""
        return build_linear_action_plan(input_layout=input_layout, output_layout=output_layout)

    def versor_action_plan(
        self,
        *,
        grade: int,
        input_layout: GradeLayout,
        output_layout: GradeLayout = None,
        parameter_layout: GradeLayout = None,
    ) -> VersorActionPlan:
        """Return a plan-only contract for a grade-1 or grade-2 versor action."""
        return build_versor_action_plan(
            self.algebra,
            grade=grade,
            input_layout=input_layout,
            output_layout=output_layout,
            parameter_layout=parameter_layout,
        )

    def paired_bivector_action_plan(
        self,
        *,
        input_layout: GradeLayout,
        output_layout: GradeLayout = None,
        parameter_layout: GradeLayout = None,
    ) -> PairedBivectorActionPlan:
        """Return a plan-only contract for independent bivector rotor pairs."""
        return build_paired_bivector_action_plan(
            self.algebra,
            input_layout=input_layout,
            output_layout=output_layout,
            parameter_layout=parameter_layout,
        )

    def bivector_decomposition_plan(
        self,
        *,
        input_layout: GradeLayout = None,
        components: int = None,
        fixed_iterations: int = None,
    ) -> BivectorDecompositionPlan:
        """Return static layouts and loop sizes for bivector decomposition."""
        input_layout = self.layout((2,)) if input_layout is None else input_layout
        return build_bivector_decomposition_plan(
            self.algebra,
            input_layout=input_layout,
            components=components,
            fixed_iterations=fixed_iterations,
        )

    def unary_executor_for_request(self, request: UnaryRequest, *, cache: bool = True) -> GradeUnaryExecutor:
        """Return an executor for an already normalized unary request."""
        key = request.cache_key
        executor = self._unary_executors.get(key) if cache else None
        if executor is None:
            plan = build_unary_plan_from_request(request)
            executor = GradeUnaryExecutor(plan)
            if cache:
                self._unary_executors[key] = executor
        return executor

    def _product_cache_key(self, executor: GradeProductExecutor) -> tuple[object, ...]:
        return (
            self.spec,
            str(executor.coefficients.device),
            str(executor.coefficients.dtype),
            executor.op,
            executor.left_grades,
            executor.right_grades,
            executor.output_grades,
        )

    def _unary_cache_key(self, executor: GradeUnaryExecutor) -> tuple[object, ...]:
        return (
            self.spec,
            str(executor.signs.device),
            str(executor.signs.dtype),
            executor.op,
            executor.input_layout.grades,
            executor.output_layout.grades,
        )

    def _default_operand_grades(self, grades, layout: GradeLayout = None):
        if grades is not None or layout is not None:
            return grades
        return getattr(self.algebra, "_default_grades", None)

    def _validate_product_grade_cost_before_layouts(
        self,
        *,
        op: str,
        left_grades,
        right_grades,
        output_grades,
        left_layout: GradeLayout = None,
        right_layout: GradeLayout = None,
        output_layout: GradeLayout = None,
    ) -> None:
        left = left_layout.grades if left_layout is not None else left_grades
        right = right_layout.grades if right_layout is not None else right_grades
        if left is None or right is None:
            return
        output = output_layout.grades if output_layout is not None else output_grades
        validate_product_grades_cost(
            self.algebra,
            self.spec,
            op=op,
            left_grades=left,
            right_grades=right,
            output_grades=output,
        )
