# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Grade-aware planner from algebraic intent to static executors."""

from __future__ import annotations

import torch

from clifra.core._kernel.basis import operation_coefficient
from clifra.core._kernel.contracts import _check_contract_spec
from clifra.core._kernel.execution.action import FullSandwichActionExecutor
from clifra.core._kernel.execution.exp import BivectorExpExecutor
from clifra.core._kernel.execution.metric import SignatureNormSquaredExecutor
from clifra.core._kernel.execution.permutation import PseudoscalarProductExecutor
from clifra.core._kernel.execution.product import FullTableProductExecutor, GradeProductExecutor
from clifra.core._kernel.execution.unary import GradeUnaryExecutor
from clifra.core._kernel.planning.action import (
    LinearActionPlan,
    PairedBivectorActionPlan,
    VersorActionPlan,
    build_linear_action_plan,
    build_paired_bivector_action_plan,
    build_versor_action_plan,
)
from clifra.core._kernel.planning.exp import DEFAULT_BIVECTOR_EXP_OPTIONS
from clifra.core._kernel.planning.layouts import ProductRequest, build_product_request
from clifra.core._kernel.planning.product import (
    select_product_route,
)
from clifra.core._kernel.planning.resources import (
    validate_grades_cost,
    validate_product_grades_cost,
    validate_product_request,
    validate_unary_request,
)
from clifra.core._kernel.planning.tree import build_grade_plan_tree
from clifra.core._kernel.planning.unary import (
    UnaryRequest,
    build_unary_request,
)
from clifra.core.layout import AlgebraSpec, GradeLayout
from clifra.core.tensors import LaneStorage, TensorContract


class GradePlanner:
    """Owns layout and product-plan lowering for one algebra instance.

    The planner is deliberately not an ``nn.Module``. It builds static
    executor modules keyed by signature, grades, dtype, and device.
    """

    def __init__(self, algebra):
        self.algebra = algebra
        self.spec = AlgebraSpec.from_algebra(algebra)
        from clifra.core._kernel.routing import ExecutorRouter

        self.router = ExecutorRouter(algebra.registry.providers)
        self.policy = algebra._planning_policy
        self.limits = algebra._resource_limits
        self._product_executors = {}
        self._unary_executors = {}
        self._signature_norm_squared_executors = {}
        self._pseudoscalar_product_executors = {}
        self._bivector_exp_executors = {}
        self._full_sandwich_action_executors = {}
        self._versor_action_plans = {}
        self._paired_bivector_action_plans = {}
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
        """Convert compact values between layouts without full-lane materialization."""
        source_layout = self._compact_contract(source_layout, "source_layout").layout
        target_layout = self._compact_contract(target_layout, "target_layout").layout
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
                operation_coefficient(index, index, self.spec.p, self.spec.q, self.spec.r, "geometric_product")
                for index in layout.basis_indices
            ]
            cached = torch.tensor(signs, dtype=dtype, device=device)
            self._bivector_signs_cache[key] = cached
        return cached

    def clear_cache(self) -> None:
        """Drop cached executor modules."""
        self._product_executors.clear()
        self._unary_executors.clear()
        self._signature_norm_squared_executors.clear()
        self._pseudoscalar_product_executors.clear()
        self._bivector_exp_executors.clear()
        self._full_sandwich_action_executors.clear()
        self._versor_action_plans.clear()
        self._paired_bivector_action_plans.clear()
        self._bivector_signs_cache.clear()

    def product_executor(
        self,
        request: ProductRequest,
        *,
        cache: bool = True,
    ) -> FullTableProductExecutor | GradeProductExecutor:
        """Return the cached executor for one normalized product request."""
        request.validate(self.spec)
        validate_product_request(self.algebra, request)
        key = self._product_request_cache_key(request)
        executor = self._product_executors.get(key) if cache else None
        if executor is not None:
            return executor
        if executor is None:
            from clifra.core._kernel.execution.providers import product_execution_request

            executor = self.router.execute_plan(
                product_execution_request(self.algebra, request),
                self.policy,
                self.limits,
            )
            if cache:
                self._product_executors[key] = executor
        return executor

    def product_request(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        op: str = "geometric_product",
        left_grades=None,
        right_grades=None,
        output_grades=None,
        left_layout: GradeLayout = None,
        right_layout: GradeLayout = None,
        output_layout: GradeLayout = None,
        left_storage: LaneStorage | str | None = None,
        right_storage: LaneStorage | str | None = None,
        output_storage: LaneStorage | str = LaneStorage.COMPACT,
    ) -> ProductRequest:
        """Normalize product intent into a static request without executing tensors."""
        if left_layout is not None:
            left_layout = self._compact_contract(left_layout, "left_layout").layout
        if right_layout is not None:
            right_layout = self._compact_contract(right_layout, "right_layout").layout
        if output_layout is not None:
            output_layout = self._compact_contract(output_layout, "output_layout").layout
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
            left_storage=left_storage,
            right_storage=right_storage,
            output_storage=output_storage,
        )
        validate_product_request(self.algebra, request)
        return request

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
        input_storage: LaneStorage | str | None = None,
        output_storage: LaneStorage | str = LaneStorage.COMPACT,
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
            input_storage=input_storage,
            output_storage=output_storage,
        )
        validate_unary_request(self.algebra, request)
        return request

    def unary_executor(
        self,
        request: UnaryRequest,
        *,
        cache: bool = True,
    ) -> GradeUnaryExecutor:
        """Return the cached executor for one normalized unary request."""
        request.validate(self.spec)
        validate_unary_request(self.algebra, request)
        key = request.cache_key
        executor = self._unary_executors.get(key) if cache else None
        if executor is None:
            executor = self._single_executor(
                "unary",
                request.op,
                request.input_layout,
                request.output_layout,
                request.dtype,
                request.device,
                declaration=request,
            )
            if cache:
                self._unary_executors[key] = executor
        return executor

    def signature_norm_squared_executor(
        self,
        *,
        input_layout: GradeLayout,
        dtype,
        device,
        cache: bool = True,
    ) -> SignatureNormSquaredExecutor:
        """Return a cached signed signature-norm executor for a resolved layout."""
        input_layout = self._compact_contract(input_layout, "input_layout").layout
        resolved_device = torch.device(device)
        key = (
            self.spec,
            str(resolved_device),
            str(dtype),
            "signature_norm_squared",
            input_layout.grades,
        )
        executor = self._signature_norm_squared_executors.get(key) if cache else None
        if executor is None:
            executor = self._single_executor(
                "metric",
                "signature_norm_squared",
                input_layout,
                self.spec.layout((0,)),
                dtype,
                resolved_device,
            )
            if cache:
                self._signature_norm_squared_executors[key] = executor
        return executor

    def pseudoscalar_product_executor(
        self,
        *,
        input_layout: GradeLayout,
        output_layout: GradeLayout = None,
        dtype,
        device,
        cache: bool = True,
    ) -> PseudoscalarProductExecutor:
        """Return a cached right-pseudoscalar product permutation executor."""
        input_layout = self._compact_contract(input_layout, "input_layout").layout
        if output_layout is None:
            output_layout = self.spec.layout(tuple(self.spec.n - grade for grade in input_layout.grades))
        output_layout = self._compact_contract(output_layout, "output_layout").layout
        resolved_device = torch.device(device)
        key = (
            self.spec,
            str(resolved_device),
            str(dtype),
            "pseudoscalar_product",
            input_layout.grades,
            output_layout.grades,
        )
        executor = self._pseudoscalar_product_executors.get(key) if cache else None
        if executor is None:
            executor = self._single_executor(
                "permutation",
                "pseudoscalar_product",
                input_layout,
                output_layout,
                dtype,
                resolved_device,
            )
            if cache:
                self._pseudoscalar_product_executors[key] = executor
        return executor

    def bivector_exp_executor(
        self,
        *,
        input_layout: GradeLayout,
        output_layout: GradeLayout,
        dtype=None,
        device=None,
        cache: bool = True,
        spectral_max_planes: int | None = None,
        spectral_tol_abs: float | None = None,
        spectral_tol_rel: float | None = None,
        spectral_dominant_rel: float | None = None,
        spectral_allow_degenerate: bool | None = None,
        spectral_allow_truncated_degenerate: bool | None = None,
    ) -> BivectorExpExecutor:
        """Return a cached executor for the bivector exponential ``exp(B)``."""
        dtype = self.algebra.dtype if dtype is None else dtype
        device = self.algebra.device if device is None else device
        input_layout = self._compact_contract(input_layout, "input_layout").layout
        output_layout = self._compact_contract(output_layout, "output_layout").layout
        if input_layout.grades != (2,):
            raise ValueError(f"bivector exp requires grade-2 input layout, got {input_layout.grades}")
        resolved_device = torch.device(device)
        options = getattr(self.algebra, "_bivector_exp_options", DEFAULT_BIVECTOR_EXP_OPTIONS)
        resolved_spectral_max_planes = (
            options.spectral_max_planes if spectral_max_planes is None else spectral_max_planes
        )
        resolved_spectral_tol_abs = options.spectral_tol_abs if spectral_tol_abs is None else spectral_tol_abs
        resolved_spectral_tol_rel = options.spectral_tol_rel if spectral_tol_rel is None else spectral_tol_rel
        resolved_spectral_dominant_rel = (
            options.spectral_dominant_rel if spectral_dominant_rel is None else spectral_dominant_rel
        )
        resolved_spectral_allow_degenerate = (
            options.spectral_allow_degenerate if spectral_allow_degenerate is None else bool(spectral_allow_degenerate)
        )
        resolved_spectral_allow_truncated_degenerate = (
            options.spectral_allow_truncated_degenerate
            if spectral_allow_truncated_degenerate is None
            else bool(spectral_allow_truncated_degenerate)
        )
        full_rank_planes = (self.spec.p + self.spec.q) // 2
        resolved_spectral_max_planes = (
            min(full_rank_planes, 4)
            if resolved_spectral_max_planes is None
            else min(int(resolved_spectral_max_planes), full_rank_planes, 4)
        )
        resolved_spectral_tol_abs = (
            float(torch.finfo(dtype).eps * 32.0)
            if resolved_spectral_tol_abs is None
            else float(resolved_spectral_tol_abs)
        )
        resolved_spectral_dominant_rel = (
            float(max(torch.finfo(dtype).eps ** 0.5, torch.finfo(dtype).eps * 32.0))
            if resolved_spectral_dominant_rel is None
            else float(resolved_spectral_dominant_rel)
        )
        key = (
            self.spec,
            str(resolved_device),
            str(dtype),
            "bivector_exp",
            resolved_spectral_max_planes,
            resolved_spectral_tol_abs,
            resolved_spectral_tol_rel,
            resolved_spectral_dominant_rel,
            resolved_spectral_allow_degenerate,
            resolved_spectral_allow_truncated_degenerate,
            input_layout.grades,
            output_layout.grades,
        )
        executor = self._bivector_exp_executors.get(key) if cache else None
        if executor is not None:
            return executor
        from clifra.core._kernel.execution.providers import exp_execution_request
        from clifra.core._kernel.planning.exp import BivectorExpOptions, spectral_exp_preselection

        options = BivectorExpOptions(
            spectral_max_planes=None if full_rank_planes == 0 else resolved_spectral_max_planes,
            spectral_tol_abs=resolved_spectral_tol_abs,
            spectral_tol_rel=resolved_spectral_tol_rel,
            spectral_dominant_rel=resolved_spectral_dominant_rel,
            spectral_allow_degenerate=resolved_spectral_allow_degenerate,
            spectral_allow_truncated_degenerate=resolved_spectral_allow_truncated_degenerate,
        )
        preselection = spectral_exp_preselection(
            self.spec,
            resolved_device,
            dtype=dtype,
            max_planes=options.spectral_max_planes,
            tol_abs=options.spectral_tol_abs,
            tol_rel=options.spectral_tol_rel,
            dominant_rel=options.spectral_dominant_rel,
            allow_degenerate=options.spectral_allow_degenerate,
            allow_truncated_degenerate=options.spectral_allow_truncated_degenerate,
        )
        request = exp_execution_request(
            self.spec,
            resolved_device,
            dtype,
            output_layout,
            preselection,
            planner=self,
            options=options,
            cache=cache,
        )
        executor = self.router.execute_plan(request, self.policy, self.limits)
        if cache:
            self._bivector_exp_executors[key] = executor
        return executor

    def full_sandwich_action_executor(
        self,
        *,
        layout: GradeLayout,
        dtype,
        device,
        cache: bool = True,
    ) -> FullSandwichActionExecutor:
        """Return a cached full-layout sandwich action executor."""
        layout = self._compact_contract(layout, "layout").layout
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
            from dataclasses import replace

            from clifra.core._kernel.execution.providers import action_execution_request

            request = action_execution_request(self.algebra, "sandwich", input_layout=layout)
            request = replace(request, dtype=dtype, device=resolved_device)
            executor = self.router.execute_plan(request, self.policy, self.limits)
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
        input_layout = self._compact_contract(input_layout, "input_layout").layout
        if output_layout is not None:
            output_layout = self._compact_contract(output_layout, "output_layout").layout
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
        input_layout = self._compact_contract(input_layout, "input_layout").layout
        if output_layout is not None:
            output_layout = self._compact_contract(output_layout, "output_layout").layout
        if parameter_layout is not None:
            parameter_layout = self._compact_contract(parameter_layout, "parameter_layout").layout
        key = self._action_plan_cache_key(
            "versor_action",
            int(grade),
            input_layout,
            output_layout,
            parameter_layout,
        )
        plan = self._versor_action_plans.get(key)
        if plan is None:
            plan = build_versor_action_plan(
                self.algebra,
                grade=grade,
                input_layout=input_layout,
                output_layout=output_layout,
                parameter_layout=parameter_layout,
            )
            self._versor_action_plans[key] = plan
        return plan

    def paired_bivector_action_plan(
        self,
        *,
        input_layout: GradeLayout,
        output_layout: GradeLayout = None,
        parameter_layout: GradeLayout = None,
    ) -> PairedBivectorActionPlan:
        """Return a plan-only contract for independent bivector rotor pairs."""
        input_layout = self._compact_contract(input_layout, "input_layout").layout
        if output_layout is not None:
            output_layout = self._compact_contract(output_layout, "output_layout").layout
        if parameter_layout is not None:
            parameter_layout = self._compact_contract(parameter_layout, "parameter_layout").layout
        key = self._action_plan_cache_key(
            "paired_bivector_action",
            2,
            input_layout,
            output_layout,
            parameter_layout,
        )
        plan = self._paired_bivector_action_plans.get(key)
        if plan is None:
            plan = build_paired_bivector_action_plan(
                self.algebra,
                input_layout=input_layout,
                output_layout=output_layout,
                parameter_layout=parameter_layout,
            )
            self._paired_bivector_action_plans[key] = plan
        return plan

    def _single_executor(self, family, operation, inputs, output, dtype, device, declaration=None):
        from clifra.core._kernel.execution.providers import UnaryExecutionRequest
        from clifra.core.executors import ExecutorRequest

        arguments = (
            family,
            operation,
            (TensorContract.compact(inputs),),
            TensorContract.compact(output),
            dtype,
            device,
        )
        request = UnaryExecutionRequest(*arguments, declaration) if family == "unary" else ExecutorRequest(*arguments)
        return self.router.execute_plan(request, self.policy, self.limits)

    def action_executor(self, operation, **parameters):
        from clifra.core._kernel.execution.providers import action_execution_request

        request = action_execution_request(self.algebra, operation, **parameters)
        return self.router.execute_plan(request, self.policy, self.limits)

    def _product_request_cache_key(self, request: ProductRequest) -> tuple[object, ...]:
        return (
            request.spec,
            str(request.device),
            str(request.dtype),
            request.op,
            request.left_grades,
            request.right_grades,
            request.output_grades,
        )

    def _product_executor_family(self, request: ProductRequest) -> str:
        decision = select_product_route(
            self.algebra,
            op=request.op,
            left_layout=request.left_layout,
            right_layout=request.right_layout,
            output_layout=request.output_layout,
            dtype=request.dtype,
            device=request.device,
        )
        return decision.route

    def _action_plan_cache_key(
        self,
        family: str,
        grade: int,
        input_layout: GradeLayout,
        output_layout: GradeLayout | None,
        parameter_layout: GradeLayout | None,
    ) -> tuple[object, ...]:
        return (
            self.spec,
            str(self.algebra.device),
            str(self.algebra.dtype),
            self.algebra._bivector_exp_options,
            family,
            grade,
            input_layout.grades,
            None if output_layout is None else output_layout.grades,
            None if parameter_layout is None else parameter_layout.grades,
        )

    def _default_operand_grades(self, grades, layout: GradeLayout = None):
        if grades is not None or layout is not None:
            return grades
        return getattr(self.algebra, "_default_grades", None)

    def _compact_contract(self, layout: GradeLayout, name: str) -> TensorContract:
        contract = TensorContract.compact(layout)
        return _check_contract_spec(self.spec, contract, name)

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
