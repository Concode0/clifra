# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Bivector-exponential capability, policy, and resource boundaries."""

import pytest
import torch

from clifra.core import AlgebraContext, AlgebraSpec
from clifra.core._kernel.basis import build_bivector_squared_signs
from clifra.core._kernel.planning.policy import DEFAULT_PLANNING_POLICY
from clifra.core._kernel.planning.resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits
from clifra.core._kernel.providers import builtin_providers, exp_execution_request
from clifra.core._kernel.routing import ExecutorRouter
from clifra.core.executors import Rejected
from tests.helpers.bivector_exp_oracle import bivector_exp_cpu_reference
from tests.helpers.policy import PreferRoute
from tests.helpers.small_oracle import SmallCliffordOracle


def _select(spec, device="cpu", *, dtype=torch.float32, output=None, policy=None, limits=None):
    request = exp_execution_request(spec, device, dtype, output)
    return ExecutorRouter(builtin_providers()).select(
        request,
        DEFAULT_PLANNING_POLICY if policy is None else policy,
        DEFAULT_RESOURCE_LIMITS if limits is None else limits,
    )


@pytest.mark.parametrize("signature", [(3, 0, 0), (0, 3, 0), (2, 1, 1), (0, 0, 3)])
def test_bivector_square_preparation_matches_independent_products(signature):
    spec = AlgebraSpec(*signature)
    layout = spec.layout((2,))
    basis = layout.full(torch.eye(layout.dim, dtype=torch.float64))
    expected = SmallCliffordOracle(*signature).product(basis, basis)[..., 0]
    torch.testing.assert_close(build_bivector_squared_signs(layout, dtype=torch.float64, device="cpu"), expected)


def test_closed_route_is_preferred_throughout_its_domain():
    for n in range(2, 6):
        assert _select(AlgebraSpec(n)).route == "closed"


def test_general_default_policy_is_device_independent():
    for n in (6, 8, 12):
        routes = {_select(AlgebraSpec(n), device).route for device in ("cpu", "mps", "cuda")}
        assert len(routes) == 1


def test_capability_domain_is_independent_of_default_policy():
    for n in range(2, 13):
        spec = AlgebraSpec(n)
        request = exp_execution_request(spec, "cpu", torch.float32, spec.layout((0,)))
        accepted = {
            provider.identity[1]
            for provider in builtin_providers()
            if provider.identity[0] == "bivector_exp" and not isinstance(provider.assess(request), Rejected)
        }
        assert accepted == ({"closed", "taylor", "left_matrix_exp"} if n <= 5 else {"taylor", "left_matrix_exp"})


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_general_routes_reject_unvalidated_low_precision(dtype):
    with pytest.raises(ValueError, match="float32_or_float64"):
        _select(AlgebraSpec(7), dtype=dtype)


def test_dimension_and_resource_limits_reject_before_construction():
    with pytest.raises(ValueError, match="materialized_exp_requires_n_2_through_12"):
        _select(AlgebraSpec(13))
    algebra = AlgebraContext(8)
    with pytest.raises(ValueError, match="pair/interaction footprint"):
        _select(
            algebra.spec,
            output=algebra.layout((0,)),
            limits=ResourceLimits(max_pairs=1_000),
        )


@pytest.mark.parametrize("n,route", [(3, "taylor"), (4, "left_matrix_exp"), (7, "left_matrix_exp")])
def test_nondefault_routes_match_reference_and_gradients(n, route):
    from clifra.core._kernel.configuration import configured_algebra

    algebra = configured_algebra(n, dtype=torch.float64, planning_policy=PreferRoute("bivector_exp", route))
    layout, output = algebra.layout((2,)), algebra.layout((0, 2))
    operation = algebra.plan_bivector_exp(input=layout, output=output)
    values = (torch.randn(1, layout.dim, dtype=torch.float64) * 0.1).requires_grad_()
    actual = operation(values)
    expected = bivector_exp_cpu_reference(algebra, values, input_layout=layout, output_layout=output)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), values)[0],
        torch.autograd.grad(expected.sum(), values)[0],
        atol=2e-11,
        rtol=2e-11,
    )


def test_matrix_route_resource_rejection_falls_back_before_build():
    spec = AlgebraSpec(10)
    request = exp_execution_request(spec, "cpu", torch.float64, spec.layout((0,)))
    providers = builtin_providers()
    selection = ExecutorRouter(providers).select(
        request,
        PreferRoute("bivector_exp", "left_matrix_exp"),
        DEFAULT_RESOURCE_LIMITS,
    )
    assert selection.route == "taylor"


def test_mps_float64_is_capability_rejection():
    with pytest.raises(ValueError, match="mps_does_not_support_float64_output"):
        _select(AlgebraSpec(6), "mps", dtype=torch.float64)
