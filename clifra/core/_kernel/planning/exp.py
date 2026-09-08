# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Static contracts for materialized bivector exponentials."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from clifra.core._kernel.contracts import _check_contract_spec
from clifra.core._kernel.planning.policy import DEFAULT_PLANNING_POLICY
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
    if decision.route == "left_matrix_exp":
        device = torch.device("cpu")
    even = spec.layout(range(0, spec.n + 1, 2))
    grade4 = spec.layout((4,)) if decision.route == "closed" and spec.n >= 4 else None
    metric = [1] * spec.p + [-1] * spec.q + [0] * spec.r
    signs = []
    for index in input_layout.basis_indices:
        i, j = [bit for bit in range(spec.n) if index & (1 << bit)]
        signs.append(-metric[i] * metric[j])
    positions = {index: pos for pos, index in enumerate(even.basis_indices)}
    return BivectorExpPlan(
        spec,
        input_layout,
        output_layout,
        even,
        grade4,
        decision.route,
        torch.finfo(dtype).eps,
        torch.tensor(signs, dtype=dtype, device=device),
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
    from clifra.core._kernel.execution.providers import exp_execution_request
    from clifra.core._kernel.planning.resources import DEFAULT_RESOURCE_LIMITS
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
