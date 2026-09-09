# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from clifra.core._kernel.basis import build_bivector_squared_signs
from clifra.core._kernel.configuration import configured_algebra
from clifra.core._kernel.planning.exp import (
    assess_bivector_exp_routes,
    select_bivector_exp_executor_family,
    select_bivector_exp_route,
    taylor_layouts,
)
from clifra.core._kernel.planning.resources import ResourceLimits
from clifra.core.algebra import AlgebraContext
from clifra.core.layout import AlgebraSpec
from tests.helpers.bivector_exp_oracle import bivector_exp_cpu_reference
from tests.helpers.policy import PreferRoute

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("signature", [(3, 0, 0), (0, 3, 0), (2, 1, 1), (0, 0, 3)])
def test_bivector_square_preparation_matches_independent_products(signature):
    from tests.helpers.small_oracle import SmallCliffordOracle

    spec = AlgebraSpec(*signature)
    layout = spec.layout((2,))
    basis = layout.full(torch.eye(layout.dim, dtype=torch.float64))
    expected = SmallCliffordOracle(*signature).product(basis, basis)[..., 0]
    actual = build_bivector_squared_signs(layout, dtype=torch.float64, device="cpu")
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "n,device,route",
    [
        (3, "cpu", "closed"),
        (5, "mps", "closed"),
        (6, "cpu", "left_matrix_exp"),
        (6, "mps", "left_matrix_exp"),
        (7, "mps", "left_matrix_exp"),
        (7, "cpu", "taylor"),
        (8, "mps", "taylor"),
        (6, "cuda", "taylor"),
        (10, "cpu", "taylor"),
        (12, "cuda", "taylor"),
    ],
)
def test_exp_route_domains(n, device, route):
    assert select_bivector_exp_executor_family(AlgebraSpec(n), device) == route


def test_exp_rejects_dimension_before_product_allocation():
    with pytest.raises(ValueError, match="materialized_exp_requires_n_2_through_12"):
        select_bivector_exp_executor_family(AlgebraSpec(13), "cpu")


def test_exp_respects_intermediate_resource_limit():
    a = configured_algebra(8, resource_limits=ResourceLimits(max_pairs=1000))
    with pytest.raises(ValueError, match="intermediate interactions"):
        a._planner.bivector_exp_executor(input_layout=a.layout((2,)), output_layout=a.layout((0,)))


def test_exp_cache_and_contracts():
    a = AlgebraContext(7)
    kwargs = dict(input_layout=a.layout((2,)), output_layout=a.layout((0, 2)))
    f = a._planner.bivector_exp_executor(**kwargs)
    assert a._planner.bivector_exp_executor(**kwargs) is f
    assert f.input_contract.layout == kwargs["input_layout"]
    assert f.output_contract.layout == kwargs["output_layout"]
    with pytest.raises(ValueError):
        f(torch.zeros(3))


def test_taylor_layouts_keep_return_paths():
    spec = AlgebraSpec(8)
    layouts = taylor_layouts(spec, spec.layout((0,)), 12)
    assert layouts[0].grades == (0,)
    assert layouts[-1].grades == (0,)
    assert layouts[-2].grades == (0, 2)
    assert layouts[6].grades == (0, 2, 4, 6, 8)


def test_taylor_reuses_repeated_product_modules():
    a = configured_algebra(7, planning_policy=PreferRoute("bivector_exp", "taylor"))
    f = a._planner.bivector_exp_executor(input_layout=a.layout((2,)), output_layout=a.layout((0,)))
    products = list(f.polynomial.products)
    assert len({id(x) for x in products}) < len(products)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_general_exp_rejects_unvalidated_low_precision(dtype):
    with pytest.raises(ValueError, match="float32_or_float64"):
        select_bivector_exp_executor_family(AlgebraSpec(7), "cpu", dtype=dtype)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8, 9, 12])
@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_exp_capability_is_independent_of_default_crossover(n, device):
    spec = AlgebraSpec(n)
    routes = assess_bivector_exp_routes(spec, device, dtype=torch.float32, output_layout=spec.layout((0,)))
    accepted = {candidate.route for candidate in routes if candidate.unavailable_reason is None}
    assert accepted == ({"closed", "taylor", "left_matrix_exp"} if n <= 5 else {"taylor", "left_matrix_exp"})


@pytest.mark.parametrize(
    "n,route", [(3, "taylor"), (4, "left_matrix_exp"), (5, "taylor"), (7, "left_matrix_exp"), (8, "left_matrix_exp")]
)
def test_nondefault_exp_routes_execute_overlapping_contracts(n, route):
    a = configured_algebra(n, dtype=torch.float64, planning_policy=PreferRoute("bivector_exp", route))
    layout, output = a.layout((2,)), a.layout((0, 2))
    f = a._planner.bivector_exp_executor(input_layout=layout, output_layout=output)
    assert f.route == route
    values = (torch.randn(1, layout.dim, dtype=torch.float64) * 0.1).requires_grad_()
    actual = f(values)
    expected = bivector_exp_cpu_reference(a, values, input_layout=layout, output_layout=output)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), values)[0],
        torch.autograd.grad(expected.sum(), values)[0],
        atol=2e-11,
        rtol=2e-11,
    )


def test_matrix_column_materialization_is_guarded_before_allocation(monkeypatch):
    policy = PreferRoute("bivector_exp", "left_matrix_exp")
    spec = AlgebraSpec(10)
    candidates = assess_bivector_exp_routes(spec, "cpu", dtype=torch.float64, output_layout=spec.layout((0,)))
    matrix = next(candidate for candidate in candidates if candidate.route == "left_matrix_exp")
    assert matrix.unavailable_reason is None
    assert matrix.facts.resources.pairs == 45 * 512 * 512
    assert matrix.facts.resources.rejection_reason(ResourceLimits()) is not None

    def fail(*args, **kwargs):
        raise AssertionError("route selection must not allocate execution tensors")

    with monkeypatch.context() as check:
        for name in ("zeros", "ones", "eye", "tensor", "empty", "arange"):
            check.setattr(torch, name, fail)
        assert select_bivector_exp_executor_family(spec, "cpu", planning_policy=policy) == "taylor"
        decision = select_bivector_exp_route(
            spec,
            "cpu",
            dtype=torch.float64,
            output_layout=spec.layout((0,)),
            policy=policy,
            limits=ResourceLimits(max_pairs=12_000_000),
        )
        assert decision.route == "left_matrix_exp"


def test_mps_float64_output_is_a_capability_rejection():
    with pytest.raises(ValueError, match="mps_does_not_support_float64_output"):
        select_bivector_exp_executor_family(AlgebraSpec(6), "mps", dtype=torch.float64)
