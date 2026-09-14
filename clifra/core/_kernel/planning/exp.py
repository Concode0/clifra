# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Static contracts for materialized bivector exponentials."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from clifra.core._kernel.basis import build_bivector_squared_signs
from clifra.core._kernel.contracts import _check_contract_spec
from clifra.core._kernel.planning.policy import BivectorExpFacts
from clifra.core._kernel.planning.resources import ResourceRequirements
from clifra.core._kernel.planning.work import BivectorExpWorkProfile, product_work_profile
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


def build_bivector_exp_plan(spec, *, input_layout, output_layout, dtype, device, route):
    _check_contract_spec(spec, TensorContract.compact(input_layout), "input_layout")
    _check_contract_spec(spec, TensorContract.compact(output_layout), "output_layout")
    if input_layout.grades != (2,):
        raise ValueError(f"bivector exp requires grade-2 input layout, got {input_layout.grades}")
    if route == "left_matrix_exp" or (route == "closed" and spec.n >= 4 and torch.device(device).type == "mps"):
        device = torch.device("cpu")
    even = spec.layout(range(0, spec.n + 1, 2))
    grade4 = spec.layout((4,)) if route == "closed" and spec.n >= 4 else None
    positions = {index: pos for pos, index in enumerate(even.basis_indices)}
    return BivectorExpPlan(
        spec,
        input_layout,
        output_layout,
        even,
        grade4,
        route,
        torch.finfo(dtype).eps,
        build_bivector_squared_signs(input_layout, dtype=dtype, device=device),
        torch.tensor([float(i == 0) for i in output_layout.basis_indices], dtype=dtype, device=device),
        _layout_map(input_layout, output_layout, dtype=dtype, device=device),
        _layout_map(grade4, output_layout, dtype=dtype, device=device),
        torch.tensor([positions.get(i, 0) for i in output_layout.basis_indices], dtype=torch.long, device=device),
        torch.tensor([float(i in positions) for i in output_layout.basis_indices], dtype=dtype, device=device),
        torch.eye(even.dim, dtype=dtype, device=device)
        if route == "left_matrix_exp"
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


@dataclass(frozen=True)
class BivectorExpRouteAssessment:
    route: str
    facts: BivectorExpFacts
    resources: ResourceRequirements
    unavailable_reason: str | None = None


def _sparse_product_profile(spec, left, right, output, op="geometric_product"):
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
    return product_work_profile("sparse", interactions=interactions, output_width=output.dim, full_width=spec.dim)


def assess_bivector_exp_route(spec, device, *, dtype, output_layout, route):
    """Assess one exponential route without constructing Torch plan buffers."""
    even = 1 << max(spec.n - 1, 0)
    bivector_lanes = spec.n * (spec.n - 1) // 2
    width = output_layout.dim if output_layout is not None else 1
    output = spec.layout((0,)) if output_layout is None else output_layout
    inputs = spec.layout((2,)) if spec.n >= 2 else spec.layout(())
    even_layout = spec.layout(range(0, spec.n + 1, 2))
    fixed_products = plain_products = scaled_products = ()
    square_product = None
    plain_widths = scaled_widths = ()
    reason = None
    if route == "closed":
        reason = None if 2 <= spec.n <= 5 else "closed_requires_n_2_through_5"
        pairs = max(width, spec.n**4)
        products = []
        if reason is None and spec.n >= 4:
            grade4 = spec.layout((4,))
            products = [
                _sparse_product_profile(spec, inputs, inputs, grade4, "wedge"),
                _sparse_product_profile(spec, grade4, grade4, spec.layout((0,))),
                _sparse_product_profile(spec, inputs, grade4, output),
            ]
        fixed_products = tuple(products)
    elif route == "taylor":
        reason = None if 2 <= spec.n <= 12 else "materialized_exp_requires_n_2_through_12"
        pairs = even**2

        def polynomial_profiles(target):
            layouts = taylor_layouts(spec, target, taylor_degree(dtype))
            return (
                tuple(_sparse_product_profile(spec, inputs, left, right) for left, right in zip(layouts, layouts[1:])),
                tuple(layout.dim for layout in layouts[1:]),
            )

        if reason is None:
            plain_products, plain_widths = polynomial_profiles(output)
            scaled_products, scaled_widths = (
                (plain_products, plain_widths) if output == even_layout else polynomial_profiles(even_layout)
            )
            square_product = _sparse_product_profile(spec, even_layout, even_layout, even_layout, "symmetric_product")
    elif route == "left_matrix_exp":
        reason = None if 2 <= spec.n <= 12 else "materialized_exp_requires_n_2_through_12"
        pairs = max(bivector_lanes, 1) * even**2
        fixed_products = (_sparse_product_profile(spec, inputs, even_layout, even_layout),) if reason is None else ()
    else:
        raise ValueError(f"unknown bivector exponential route {route!r}")
    if route != "closed" and dtype not in (torch.float32, torch.float64):
        reason = "general_exp_requires_float32_or_float64"
    if torch.device(device).type == "mps" and dtype == torch.float64:
        reason = "mps_does_not_support_float64_output"
    facts = BivectorExpFacts(
        BivectorExpWorkProfile(
            route=route,
            bivector_width=inputs.dim,
            grade4_width=spec.layout((4,)).dim if route == "closed" and spec.n >= 4 else 0,
            even_width=even_layout.dim,
            output_width=output.dim,
            fixed_products=fixed_products,
            plain_products=plain_products,
            scaled_products=scaled_products,
            square_product=square_product,
            plain_stage_widths=plain_widths,
            scaled_stage_widths=scaled_widths,
        )
    )
    return BivectorExpRouteAssessment(
        route,
        facts,
        ResourceRequirements(max(width, even), pairs),
        reason,
    )
