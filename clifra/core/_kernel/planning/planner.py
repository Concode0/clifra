# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Grade-aware planner from algebraic intent to static executors."""

from __future__ import annotations

import torch

from clifra.core._kernel.contracts import _check_contract_spec
from clifra.core._kernel.planning.action import (
    LinearActionPlan,
    PairedBivectorActionPlan,
    VersorActionPlan,
    build_linear_action_plan,
    build_paired_bivector_action_plan,
    build_versor_action_plan,
)
from clifra.core._kernel.planning.layouts import ProductRequest
from clifra.core._kernel.planning.resources import (
    validate_grades_cost,
    validate_product_request,
    validate_unary_request,
)
from clifra.core._kernel.planning.unary import (
    UnaryRequest,
)
from clifra.core.layout import AlgebraSpec, GradeLayout
from clifra.core.tensors import TensorContract


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

    def layout(self, grades):
        """Return the compact layout for ``grades``."""
        return self.spec.layout(validate_grades_cost(self.algebra, self.spec, grades))

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

    def product_executor(
        self,
        request: ProductRequest,
        *,
        cache: bool = True,
    ) -> torch.nn.Module:
        """Return the cached executor for one normalized product request."""
        request.validate(self.spec)
        validate_product_request(self.algebra, request)
        key = self._product_request_cache_key(request)
        executor = self._product_executors.get(key) if cache else None
        if executor is not None:
            return executor
        if executor is None:
            from clifra.core._kernel.providers import product_execution_request

            executor = self.router.execute_plan(
                product_execution_request(self.algebra, request),
                self.policy,
                self.limits,
            )
            if cache:
                self._product_executors[key] = executor
        return executor

    def unary_executor(
        self,
        request: UnaryRequest,
        *,
        cache: bool = True,
    ) -> torch.nn.Module:
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
    ) -> torch.nn.Module:
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
    ) -> torch.nn.Module:
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

    def bivector_exp_executor(self, *, input_layout, output_layout, dtype=None, device=None, cache=True):
        """Return a cached materialized Clifford-exponential executor."""
        dtype = self.algebra.dtype if dtype is None else dtype
        device = self.algebra.device if device is None else device
        input_layout = self._compact_contract(input_layout, "input_layout").layout
        output_layout = self._compact_contract(output_layout, "output_layout").layout
        if input_layout.grades != (2,):
            raise ValueError(f"bivector exp requires grade-2 input layout, got {input_layout.grades}")
        key = (self.spec, str(device), dtype, input_layout.grades, output_layout.grades)
        executor = self._bivector_exp_executors.get(key) if cache else None
        if executor is None:
            from clifra.core._kernel.providers import exp_execution_request

            request = exp_execution_request(self.spec, device, dtype, output_layout, planner=self)
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
    ) -> torch.nn.Module:
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

            from clifra.core._kernel.providers import action_execution_request

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
        from clifra.core._kernel.providers import UnaryExecutionRequest
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
        from clifra.core._kernel.providers import action_execution_request

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
            family,
            grade,
            input_layout.grades,
            None if output_layout is None else output_layout.grades,
            None if parameter_layout is None else parameter_layout.grades,
        )

    def _compact_contract(self, layout: GradeLayout, name: str) -> TensorContract:
        contract = TensorContract.compact(layout)
        return _check_contract_spec(self.spec, contract, name)
