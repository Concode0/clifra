# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Static contracts for materialized bivector exponentials."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from clifra.core._kernel.basis import build_bivector_squared_signs
from clifra.core._kernel.contracts import _check_contract_spec
from clifra.core._kernel.planning.policy import DEFAULT_PLANNING_POLICY, BivectorExpFacts, ProductFacts
from clifra.core._kernel.planning.resources import ResourceRequirements
from clifra.core.layout import AlgebraSpec, GradeLayout
from clifra.core.tensors import TensorContract


@dataclass(frozen=True)
class BivectorExpPlan:
    spec: AlgebraSpec
    input_layout: GradeLayout
    output_layout: GradeLayout
    operator_layout: GradeLayout
    grade4_layout: GradeLayout | None
    route: str
    eps: float
    bivector_squared_signs: torch.Tensor
    output_scalar_mask: torch.Tensor
    bivector_to_output: torch.Tensor
    grade4_to_output: torch.Tensor
    operator_to_output: torch.Tensor
    operator_output_mask: torch.Tensor
    operator_eye: torch.Tensor
    input_contract: TensorContract = field(init=False)
    output_contract: TensorContract = field(init=False)

    def __post_init__(self):
        for role in ("input", "output"):
            contract = TensorContract.compact(getattr(self, f"{role}_layout"))
            object.__setattr__(self, f"{role}_contract", _check_contract_spec(self.spec, contract, role))


def taylor_degree(dtype):
    return 18 if dtype == torch.float64 else 12


def taylor_layouts(spec, output_layout, degree):
    """Horner supports: scalar-reachable grades intersect backward output reachability.

    Each Horner stage injects a scalar. Thus reachability includes all shorter
    paths as well as paths of exactly the current length. Keeping grades within
    two per remaining multiplication is conservative for every signature.
    """
    targets = tuple(g for g in output_layout.grades if g % 2 == 0) or (0,)
    layouts = [spec.layout((0,))]
    for step in range(1, degree + 1):
        remaining = degree - step
        grades = tuple(
            g for g in range(0, min(2 * step, spec.n) + 1, 2) if any(abs(g - out) <= 2 * remaining for out in targets)
        )
        # The last stage also includes requested odd lanes, whose products and
        # scalar masks are identically zero.
        layouts.append(output_layout if step == degree else spec.layout(grades))
    return tuple(layouts)


def build_bivector_exp_plan(
    spec, *, input_layout, output_layout, dtype, device, planning_policy=DEFAULT_PLANNING_POLICY, route_decision=None
):
    _check_contract_spec(spec, TensorContract.compact(input_layout), "input_layout")
    _check_contract_spec(spec, TensorContract.compact(output_layout), "output_layout")
    if input_layout.grades != (2,):
        raise ValueError(f"bivector exp requires grade-2 input layout, got {input_layout.grades}")
    decision = route_decision or select_bivector_exp_route(
        spec, device, dtype=dtype, output_layout=output_layout, policy=planning_policy
    )
    if decision.route == "left_matrix_exp" or (
        decision.route == "closed" and spec.n >= 4 and torch.device(device).type == "mps"
    ):
        device = torch.device("cpu")
    even = spec.layout(range(0, spec.n + 1, 2))
    grade4 = spec.layout((4,)) if decision.route == "closed" and spec.n >= 4 else None
    positions = {index: pos for pos, index in enumerate(even.basis_indices)}
    return BivectorExpPlan(
        spec,
        input_layout,
        output_layout,
        even,
        grade4,
        decision.route,
        torch.finfo(dtype).eps,
        build_bivector_squared_signs(input_layout, dtype=dtype, device=device),
        torch.tensor([float(i == 0) for i in output_layout.basis_indices], dtype=dtype, device=device),
        _layout_map(input_layout, output_layout, dtype=dtype, device=device),
        _layout_map(grade4, output_layout, dtype=dtype, device=device),
        torch.tensor([positions.get(i, 0) for i in output_layout.basis_indices], dtype=torch.long, device=device),
        torch.tensor([float(i in positions) for i in output_layout.basis_indices], dtype=dtype, device=device),
        torch.eye(even.dim, dtype=dtype, device=device)
        if decision.route == "left_matrix_exp"
        else torch.empty(0, dtype=dtype, device=device),
    )


def _layout_map(source, target, *, dtype, device):
    result = torch.zeros((0 if source is None else source.dim, target.dim), dtype=dtype, device=device)
    positions = {index: pos for pos, index in enumerate(target.basis_indices)}
    if source is not None:
        for pos, index in enumerate(source.basis_indices):
            if index in positions:
                result[pos, positions[index]] = 1
    return result


def select_bivector_exp_route(spec, device, *, dtype, output_layout, policy, router=None, limits=None):
    from clifra.core._kernel.planning.resources import DEFAULT_RESOURCE_LIMITS
    from clifra.core._kernel.providers import exp_execution_request
    from clifra.core._kernel.routing import default_router

    request = exp_execution_request(spec, device, dtype, output_layout)
    return (default_router() if router is None else router).select(
        request, policy, DEFAULT_RESOURCE_LIMITS if limits is None else limits
    )


def select_bivector_exp_executor_family(
    spec, device, *, dtype=torch.float32, output_layout=None, planning_policy=DEFAULT_PLANNING_POLICY
):
    return select_bivector_exp_route(
        spec, device, dtype=dtype, output_layout=output_layout, policy=planning_policy
    ).route


def assess_bivector_exp_routes(spec, device, *, dtype, output_layout):
    """Declare route capabilities, resources, and family-owned structural facts."""
    return tuple(
        assess_bivector_exp_route(spec, device, dtype=dtype, output_layout=output_layout, route=route)
        for route in ("closed", "taylor", "left_matrix_exp")
    )


@dataclass(frozen=True)
class BivectorExpRouteAssessment:
    route: str
    facts: BivectorExpFacts
    resources: ResourceRequirements
    unavailable_reason: str | None = None


def _product_facts(spec, left, right, output, op="geometric_product"):
    from clifra.core._kernel.planning.product import count_grade_product_interactions
    from clifra.core._kernel.planning.tree import build_grade_plan_tree

    tree = build_grade_plan_tree(
        spec,
        op=op,
        left_grades=left.grades,
        right_grades=right.grades,
        output_grades=output.grades,
    )
    interactions = count_grade_product_interactions(tree)
    reductions = interactions if output.dim > 1 and interactions > 0 else 0
    return ProductFacts(interactions, reductions)


def _sum_product_facts(parts):
    parts = tuple(parts)
    return (
        sum(part.interactions for part in parts),
        sum(part.indexed_reduction_terms for part in parts),
    )


def assess_bivector_exp_route(spec, device, *, dtype, output_layout, route):
    """Assess one exponential route without constructing Torch plan buffers."""
    even = 1 << max(spec.n - 1, 0)
    bivector_lanes = spec.n * (spec.n - 1) // 2
    width = output_layout.dim if output_layout is not None else 1
    output = spec.layout((0,)) if output_layout is None else output_layout
    inputs = spec.layout((2,)) if spec.n >= 2 else spec.layout(())
    even_layout = spec.layout(range(0, spec.n + 1, 2))
    reason = None
    if route == "closed":
        reason = None if 2 <= spec.n <= 5 else "closed_requires_n_2_through_5"
        pairs = max(width, spec.n**4)
        products = []
        if reason is None and spec.n >= 4:
            grade4 = spec.layout((4,))
            products = [
                _product_facts(spec, inputs, inputs, grade4, "wedge"),
                _product_facts(spec, grade4, grade4, spec.layout((0,))),
                _product_facts(spec, inputs, grade4, output),
            ]
        fixed, reductions = _sum_product_facts(products)
        facts = BivectorExpFacts(fixed, reductions)
    elif route == "taylor":
        reason = None if 2 <= spec.n <= 12 else "materialized_exp_requires_n_2_through_12"
        pairs = even**2

        def polynomial_facts(target):
            layouts = taylor_layouts(spec, target, taylor_degree(dtype))
            return _sum_product_facts(
                _product_facts(spec, inputs, left, right) for left, right in zip(layouts, layouts[1:])
            )

        polynomial, polynomial_reductions = polynomial_facts(output) if reason is None else (0, 0)
        scaled, scaled_reductions = (
            ((polynomial, polynomial_reductions) if output == even_layout else polynomial_facts(even_layout))
            if reason is None
            else (0, 0)
        )
        facts = BivectorExpFacts(
            polynomial_interactions=polynomial,
            polynomial_reduction_terms=polynomial_reductions,
            scaled_polynomial_interactions=scaled,
            scaled_polynomial_reduction_terms=scaled_reductions,
        )
    elif route == "left_matrix_exp":
        reason = None if 2 <= spec.n <= 12 else "materialized_exp_requires_n_2_through_12"
        pairs = max(bivector_lanes, 1) * even**2
        product = _product_facts(spec, inputs, even_layout, even_layout) if reason is None else None
        facts = BivectorExpFacts(
            fixed_product_interactions=0 if product is None else product.interactions,
            fixed_reduction_terms=0 if product is None else product.indexed_reduction_terms,
        )
    else:
        raise ValueError(f"unknown bivector exponential route {route!r}")
    if route != "closed" and dtype not in (torch.float32, torch.float64):
        reason = "general_exp_requires_float32_or_float64"
    if torch.device(device).type == "mps" and dtype == torch.float64:
        reason = "mps_does_not_support_float64_output"
    return BivectorExpRouteAssessment(
        route,
        facts,
        ResourceRequirements(max(width, even), pairs),
        reason,
    )
