# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Grade-aware planner from algebraic intent to static executors."""

from __future__ import annotations

import torch

from clifra.core.execution.action import FullSandwichActionExecutor
from clifra.core.execution.exp import BivectorExpExecutor
from clifra.core.execution.metric import NormSquaredExecutor
from clifra.core.execution.permutation import DualExecutor
from clifra.core.execution.product import FullTableProductExecutor, GradeProductExecutor
from clifra.core.execution.unary import GradeUnaryExecutor
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
from clifra.core.planning.exp import build_bivector_exp_plan
from clifra.core.planning.layouts import ProductRequest, build_product_request, normalize_product_op
from clifra.core.planning.metric import build_norm_squared_plan
from clifra.core.planning.permutation import build_dual_plan
from clifra.core.planning.policy import (
    estimate_product_executor_cost,
    validate_grades_cost,
    validate_product_grades_cost,
    validate_product_request,
    validate_unary_request,
)
from clifra.core.planning.product import (
    build_full_table_product_plan_from_request,
    build_grade_product_plan_from_request,
)
from clifra.core.planning.tree import build_grade_plan_tree
from clifra.core.planning.unary import (
    UnaryRequest,
    build_unary_plan_from_request,
    build_unary_request,
    normalize_unary_op,
)
from clifra.core.storage import ValueLayout


class GradePlanner:
    """Owns layout and product-plan lowering for one algebra instance.

    The planner is deliberately not an ``nn.Module``. It builds static
    executor modules keyed by signature, grades, dtype, and device.
    """

    def __init__(self, algebra):
        self.algebra = algebra
        self.spec = AlgebraSpec.from_algebra(algebra)
        self._product_executors = {}
        self._unary_executors = {}
        self._norm_sq_executors = {}
        self._dual_executors = {}
        self._bivector_exp_executors = {}
        self._full_sandwich_action_executors = {}
        self._bivector_signs_cache = {}

    def layout(self, grades):
        """Return the compact layout for ``grades``."""
        return self.spec.layout(validate_grades_cost(self.algebra, self.spec, grades))

    def full_layout(self) -> GradeLayout:
        """Return the canonical all-grades layout."""
        return self.spec.full_layout()

    def grade_indices(self, grades, *, device=None) -> torch.Tensor:
        """Return canonical basis indices for ``grades``."""
        if device is None:
            device = getattr(self.algebra, "device", None)
        return self.layout(grades).indices_tensor(device=device)

    def convert_values(self, values: torch.Tensor, *, source_layout: GradeLayout, target_layout: GradeLayout):
        """Convert active values between layouts without full-lane materialization."""
        return target_layout.convert(values, source_layout)

    def bivector_squared_signs(self, *, device=None, dtype: torch.dtype = None) -> torch.Tensor:
        """Return ``(e_ab)^2`` signs in canonical grade-2 layout order."""
        if device is None:
            device = getattr(self.algebra, "device", None)
        if dtype is None:
            dtype = getattr(self.algebra, "dtype", torch.float32)
        layout = self.layout((2,))
        key = (layout.grades, str(torch.device(device)), str(dtype))
        cached = self._bivector_signs_cache.get(key)
        if cached is None:
            signs = [
                operation_coefficient(index, index, self.spec.p, self.spec.q, self.spec.r, "gp")
                for index in layout.basis_indices
            ]
            cached = torch.tensor(signs, dtype=dtype, device=device)
            self._bivector_signs_cache[key] = cached
        return cached

    def clear_cache(self) -> None:
        """Drop cached executor modules."""
        self._product_executors.clear()
        self._unary_executors.clear()
        self._norm_sq_executors.clear()
        self._dual_executors.clear()
        self._bivector_exp_executors.clear()
        self._full_sandwich_action_executors.clear()
        self._bivector_signs_cache.clear()

    def _apply(self, fn):
        """Apply a PyTorch module-style transform to cached executor buffers."""
        product_executors = list(self._product_executors.values())
        self._product_executors.clear()
        self._bivector_signs_cache.clear()
        for executor in product_executors:
            executor._apply(fn)
            self._product_executors[self._product_cache_key(executor)] = executor

        unary_executors = list(self._unary_executors.values())
        self._unary_executors.clear()
        for executor in unary_executors:
            executor._apply(fn)
            self._unary_executors[self._unary_cache_key(executor)] = executor

        norm_sq_executors = list(self._norm_sq_executors.values())
        self._norm_sq_executors.clear()
        for executor in norm_sq_executors:
            executor._apply(fn)
            self._norm_sq_executors[self._norm_sq_cache_key(executor)] = executor

        dual_executors = list(self._dual_executors.values())
        self._dual_executors.clear()
        for executor in dual_executors:
            executor._apply(fn)
            self._dual_executors[self._dual_cache_key(executor)] = executor

        bivector_exp_executors = list(self._bivector_exp_executors.values())
        self._bivector_exp_executors.clear()
        for executor in bivector_exp_executors:
            executor._apply(fn)
            self._bivector_exp_executors[self._bivector_exp_cache_key(executor)] = executor

        full_action_executors = list(self._full_sandwich_action_executors.values())
        self._full_sandwich_action_executors.clear()
        for executor in full_action_executors:
            executor._apply(fn)
            self._full_sandwich_action_executors[self._full_sandwich_action_cache_key(executor)] = executor
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

    def product_executor_for_request(
        self, request: ProductRequest, *, cache: bool = True
    ) -> FullTableProductExecutor | GradeProductExecutor:
        """Return an executor for an already normalized product request."""
        validate_product_request(self.algebra, request)
        family = self._product_executor_family(request)
        key = self._product_request_cache_key(request, family)
        executor = self._product_executors.get(key) if cache else None
        if executor is None:
            if family == "full_table":
                plan = build_full_table_product_plan_from_request(request)
                executor = FullTableProductExecutor(plan)
            else:
                plan = build_grade_product_plan_from_request(request)
                executor = GradeProductExecutor(plan)
            if cache:
                self._product_executors[key] = executor
        return executor

    def product_executor_for_layouts(
        self,
        *,
        op: str,
        left_layout: GradeLayout,
        right_layout: GradeLayout,
        output_layout: GradeLayout,
        dtype: torch.dtype,
        device,
        cache: bool = True,
    ) -> FullTableProductExecutor | GradeProductExecutor:
        """Return a cached executor when layout resolution is already complete."""
        normalized_op = normalize_product_op(op)
        resolved_device = torch.device(device)
        family = self._product_executor_family_for_layouts(
            op=normalized_op,
            left_layout=left_layout,
            right_layout=right_layout,
            output_layout=output_layout,
            dtype=dtype,
            device=resolved_device,
        )
        key = (
            self.spec,
            str(resolved_device),
            str(dtype),
            family,
            normalized_op,
            left_layout.grades,
            right_layout.grades,
            output_layout.grades,
        )
        executor = self._product_executors.get(key) if cache else None
        if executor is None:
            request = ProductRequest(
                spec=self.spec,
                op=normalized_op,
                left_value=ValueLayout.active(self.spec, left_layout),
                right_value=ValueLayout.active(self.spec, right_layout),
                output_value=ValueLayout.active(self.spec, output_layout),
                dtype=dtype,
                device=resolved_device,
            )
            executor = self.product_executor_for_request(request, cache=cache)
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

    def norm_sq_executor(
        self,
        *,
        input_grades,
        dtype,
        device,
        cache: bool = True,
    ) -> NormSquaredExecutor:
        """Return a cached diagonal executor for algebraic squared norm."""
        return self.norm_sq_executor_for_layout(
            input_layout=self.layout(input_grades),
            dtype=dtype,
            device=device,
            cache=cache,
        )

    def norm_sq_executor_for_layout(
        self,
        *,
        input_layout: GradeLayout,
        dtype,
        device,
        cache: bool = True,
    ) -> NormSquaredExecutor:
        """Return a cached diagonal norm executor for a resolved layout."""
        if input_layout.spec != self.spec:
            raise ValueError(f"input_layout signature {input_layout.spec} does not match algebra signature {self.spec}")
        resolved_device = torch.device(device)
        key = (
            self.spec,
            str(resolved_device),
            str(dtype),
            "norm_sq",
            input_layout.grades,
        )
        executor = self._norm_sq_executors.get(key) if cache else None
        if executor is None:
            plan = build_norm_squared_plan(
                self.spec,
                input_layout=input_layout,
                dtype=dtype,
                device=resolved_device,
            )
            executor = NormSquaredExecutor(plan)
            if cache:
                self._norm_sq_executors[key] = executor
        return executor

    def dual_executor_for_layout(
        self,
        *,
        input_layout: GradeLayout,
        output_layout: GradeLayout = None,
        dtype,
        device,
        cache: bool = True,
    ) -> DualExecutor:
        """Return a cached dual/pseudoscalar permutation executor."""
        if input_layout.spec != self.spec:
            raise ValueError(f"input_layout signature {input_layout.spec} does not match algebra signature {self.spec}")
        if output_layout is None:
            output_layout = self.spec.layout(tuple(self.spec.n - grade for grade in input_layout.grades))
        if output_layout.spec != self.spec:
            raise ValueError(f"output_layout signature {output_layout.spec} does not match algebra signature {self.spec}")
        resolved_device = torch.device(device)
        key = (
            self.spec,
            str(resolved_device),
            str(dtype),
            "dual",
            input_layout.grades,
            output_layout.grades,
        )
        executor = self._dual_executors.get(key) if cache else None
        if executor is None:
            plan = build_dual_plan(
                self.spec,
                input_layout=input_layout,
                output_layout=output_layout,
                dtype=dtype,
                device=resolved_device,
            )
            executor = DualExecutor(plan)
            if cache:
                self._dual_executors[key] = executor
        return executor

    def bivector_exp_executor_for_layouts(
        self,
        *,
        input_layout: GradeLayout,
        output_layout: GradeLayout,
        dtype,
        device,
        cache: bool = True,
    ) -> BivectorExpExecutor:
        """Return a cached executor for ``exp(B)`` with grade-2 input."""
        if input_layout.spec != self.spec:
            raise ValueError(f"input_layout signature {input_layout.spec} does not match algebra signature {self.spec}")
        if output_layout.spec != self.spec:
            raise ValueError(f"output_layout signature {output_layout.spec} does not match algebra signature {self.spec}")
        if input_layout.grades != (2,):
            raise ValueError(f"bivector exp requires grade-2 input layout, got {input_layout.grades}")
        resolved_device = torch.device(device)
        plan = build_bivector_exp_plan(
            self.spec,
            input_layout=input_layout,
            output_layout=output_layout,
            dtype=dtype,
            device=resolved_device,
            fixed_iterations=getattr(self.algebra, "_exp_fixed_iterations", 20),
        )
        key = (
            self.spec,
            str(resolved_device),
            str(dtype),
            "bivector_exp",
            plan.executor_family,
            input_layout.grades,
            output_layout.grades,
        )
        executor = self._bivector_exp_executors.get(key) if cache else None
        if executor is None:
            left_product = None
            bivector_wedge = None
            grade4_square = None
            bivector_grade4_product = None
            vector_contraction = None
            vector_wedge = None
            rotor_product = None
            if plan.executor_family == "left_matrix_exp":
                left_product = self.product_executor_for_layouts(
                    op="gp",
                    left_layout=plan.input_layout,
                    right_layout=plan.operator_layout,
                    output_layout=plan.operator_layout,
                    dtype=dtype,
                    device=resolved_device,
                    cache=cache,
                )
            elif plan.executor_family == "closed_biquadratic":
                if plan.grade4_layout is None:
                    raise RuntimeError("closed_biquadratic bivector exp requires a grade-4 layout")
                scalar_layout = self.spec.layout((0,))
                bivector_wedge = self.product_executor_for_layouts(
                    op="wedge",
                    left_layout=plan.input_layout,
                    right_layout=plan.input_layout,
                    output_layout=plan.grade4_layout,
                    dtype=dtype,
                    device=resolved_device,
                    cache=cache,
                )
                grade4_square = self.product_executor_for_layouts(
                    op="gp",
                    left_layout=plan.grade4_layout,
                    right_layout=plan.grade4_layout,
                    output_layout=scalar_layout,
                    dtype=dtype,
                    device=resolved_device,
                    cache=cache,
                )
                bivector_grade4_product = self.product_executor_for_layouts(
                    op="gp",
                    left_layout=plan.input_layout,
                    right_layout=plan.grade4_layout,
                    output_layout=plan.output_layout,
                    dtype=dtype,
                    device=resolved_device,
                    cache=cache,
                )
            elif plan.executor_family == "decomposed":
                vector_contraction = self.product_executor_for_layouts(
                    op="right_contraction",
                    left_layout=plan.input_layout,
                    right_layout=plan.vector_layout,
                    output_layout=plan.vector_layout,
                    dtype=dtype,
                    device=resolved_device,
                    cache=cache,
                )
                vector_wedge = self.product_executor_for_layouts(
                    op="wedge",
                    left_layout=plan.vector_layout,
                    right_layout=plan.vector_layout,
                    output_layout=plan.input_layout,
                    dtype=dtype,
                    device=resolved_device,
                    cache=cache,
                )
                rotor_product = self.product_executor_for_layouts(
                    op="gp",
                    left_layout=plan.operator_layout,
                    right_layout=plan.operator_layout,
                    output_layout=plan.operator_layout,
                    dtype=dtype,
                    device=resolved_device,
                    cache=cache,
                )
            executor = BivectorExpExecutor(
                plan,
                left_product,
                bivector_wedge=bivector_wedge,
                grade4_square=grade4_square,
                bivector_grade4_product=bivector_grade4_product,
                vector_contraction=vector_contraction,
                vector_wedge=vector_wedge,
                rotor_product=rotor_product,
            )
            if cache:
                self._bivector_exp_executors[key] = executor
        return executor

    def full_sandwich_action_executor_for_layout(
        self,
        *,
        layout: GradeLayout,
        dtype,
        device,
        cache: bool = True,
    ) -> FullSandwichActionExecutor:
        """Return a cached full-layout sandwich action executor."""
        if layout.spec != self.spec:
            raise ValueError(f"layout signature {layout.spec} does not match algebra signature {self.spec}")
        full_grades = tuple(range(self.spec.n + 1))
        if layout.grades != full_grades:
            raise ValueError(f"full sandwich action requires full layout {full_grades}, got {layout.grades}")
        resolved_device = torch.device(device)
        key = (
            self.spec,
            str(resolved_device),
            str(dtype),
            "full_sandwich_action",
            layout.grades,
        )
        executor = self._full_sandwich_action_executors.get(key) if cache else None
        if executor is None:
            executor = FullSandwichActionExecutor.from_layout(layout, device=resolved_device, dtype=dtype)
            if cache:
                self._full_sandwich_action_executors[key] = executor
        return executor

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

    def _product_cache_key(self, executor: FullTableProductExecutor | GradeProductExecutor) -> tuple[object, ...]:
        buffer = getattr(executor, "coefficients", None)
        if buffer is None:
            buffer = executor.signs
        return (
            self.spec,
            str(buffer.device),
            str(buffer.dtype),
            getattr(executor, "executor_family", "sparse"),
            executor.op,
            executor.left_grades,
            executor.right_grades,
            executor.output_grades,
        )

    def _product_request_cache_key(self, request: ProductRequest, family: str) -> tuple[object, ...]:
        return (
            request.spec,
            str(request.device),
            str(request.dtype),
            family,
            request.op,
            request.left_grades,
            request.right_grades,
            request.output_grades,
        )

    def _product_executor_family(self, request: ProductRequest) -> str:
        cost = estimate_product_executor_cost(
            self.algebra,
            op=request.op,
            left_layout=request.left_layout,
            right_layout=request.right_layout,
            output_layout=request.output_layout,
            dtype=request.dtype,
            device=request.device,
        )
        return cost.executor_family

    def _product_executor_family_for_layouts(
        self,
        *,
        op: str,
        left_layout: GradeLayout,
        right_layout: GradeLayout,
        output_layout: GradeLayout,
        dtype: torch.dtype,
        device,
    ) -> str:
        cost = estimate_product_executor_cost(
            self.algebra,
            op=op,
            left_layout=left_layout,
            right_layout=right_layout,
            output_layout=output_layout,
            dtype=dtype,
            device=device,
        )
        return cost.executor_family

    def _unary_cache_key(self, executor: GradeUnaryExecutor) -> tuple[object, ...]:
        return (
            self.spec,
            str(executor.signs.device),
            str(executor.signs.dtype),
            executor.op,
            executor.input_layout.grades,
            executor.output_layout.grades,
        )

    def _norm_sq_cache_key(self, executor: NormSquaredExecutor) -> tuple[object, ...]:
        return (
            self.spec,
            str(executor.signs.device),
            str(executor.signs.dtype),
            executor.op,
            executor.input_layout.grades,
        )

    def _dual_cache_key(self, executor: DualExecutor) -> tuple[object, ...]:
        return (
            self.spec,
            str(executor.signs.device),
            str(executor.signs.dtype),
            executor.op,
            executor.input_layout.grades,
            executor.output_layout.grades,
        )

    def _bivector_exp_cache_key(self, executor: BivectorExpExecutor) -> tuple[object, ...]:
        return (
            self.spec,
            str(executor.operator_eye.device),
            str(executor.operator_eye.dtype),
            executor.op,
            executor.executor_family,
            executor.input_layout.grades,
            executor.output_layout.grades,
        )

    def _full_sandwich_action_cache_key(self, executor: FullSandwichActionExecutor) -> tuple[object, ...]:
        return (
            self.spec,
            str(executor.cayley_indices.device),
            str(executor.left_sign_t.dtype),
            executor.op,
            executor.layout.grades,
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
