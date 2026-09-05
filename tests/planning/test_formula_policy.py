# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from clifra.core._kernel.execution.product import FullTableProductExecutor, GradeProductExecutor
from clifra.core._kernel.planning.layouts import ProductRequest
from clifra.core._kernel.planning.policy import (
    BoundaryRegion,
    FormulaConstraint,
    FormulaPolicy,
    PlanCandidate,
    PlanFacts,
    PolicyCoverageError,
    Polynomial,
    PolynomialTerm,
    RouteRule,
    select_policy_route,
)
from clifra.core.config import AlgebraConfig

pytestmark = pytest.mark.unit


def _candidate(route: str, x: float, *, unavailable_reason=None) -> PlanCandidate:
    return PlanCandidate(
        "product",
        route,
        PlanFacts(forward_work=x, extensions={"test.x": x}),
        unavailable_reason,
    )


def test_polynomial_region_supports_nonlinear_inclusive_boundaries():
    circle = Polynomial(
        constant=-4.0,
        terms=(
            PolynomialTerm(1.0, (("test.x", 2),)),
            PolynomialTerm(1.0, (("test.y", 2),)),
        ),
    )
    boundary = FormulaConstraint(circle, "outside_radius_two")

    assert boundary.evaluate({"test.x": 0.0, "test.y": 2.0}) == (True, 0.0)
    accepted, value = boundary.evaluate({"test.x": 2.0, "test.y": 2.0})
    assert not accepted
    assert value == 4.0


@pytest.mark.parametrize(
    "term",
    [
        lambda: PolynomialTerm(float("nan")),
        lambda: PolynomialTerm(1.0, (("x", -1),)),
        lambda: PolynomialTerm(1.0, (("x", 1.5),)),
        lambda: PolynomialTerm(1.0, (("x", 1), ("x", 2))),
    ],
)
def test_polynomial_rejects_invalid_definitions(term):
    with pytest.raises(ValueError):
        term()


def test_overlapping_regions_choose_minimum_score_then_declared_order():
    policy = FormulaPolicy(
        rules=(
            RouteRule("product", "full_table", score=Polynomial.feature("forward_work")),
            RouteRule("product", "sparse", score=Polynomial(constant=-1.0)),
        )
    )
    candidates = (_candidate("full_table", 2.0), _candidate("sparse", 2.0))

    assert select_policy_route(policy, candidates).route == "sparse"
    tied = FormulaPolicy(
        rules=(
            RouteRule("product", "full_table", score=Polynomial.feature("forward_work")),
            RouteRule("product", "sparse", score=Polynomial.feature("forward_work")),
        )
    )
    assert select_policy_route(tied, candidates).route == "full_table"


def test_unsupported_route_cannot_win_and_uncovered_supported_route_raises_diagnostics():
    outside = FormulaConstraint(Polynomial.feature("forward_work", constant=1.0), "work_must_be_nonpositive")
    policy = FormulaPolicy(
        rules=(
            RouteRule("product", "full_table", score=Polynomial(constant=-100.0)),
            RouteRule("product", "sparse", score=Polynomial()),
        )
    )
    decision = select_policy_route(
        policy,
        (_candidate("full_table", 1.0, unavailable_reason="not_implemented"), _candidate("sparse", 1.0)),
    )
    assert decision.route == "sparse"

    uncovered = FormulaPolicy(
        rules=(RouteRule("product", "sparse", (BoundaryRegion((outside,), "negative_x"),), Polynomial()),)
    )
    with pytest.raises(PolicyCoverageError, match="work_must_be_nonpositive"):
        select_policy_route(uncovered, (_candidate("sparse", 1.0),))


def test_extension_facts_are_open_immutable_and_qualified():
    policy = FormulaPolicy(rules=(RouteRule("product", "sparse", score=Polynomial.feature("vendor.machine_score")),))
    candidate = PlanCandidate(
        "product",
        "sparse",
        PlanFacts(extensions={"vendor.machine_score": 3.0}),
    )
    assert policy.evaluate(candidate).score == 3.0
    assert select_policy_route(policy, (candidate,)).route == "sparse"
    with pytest.raises(TypeError):
        candidate.facts.extensions[0] = ("vendor.machine_score", 4.0)

    with pytest.raises(ValueError, match="dot-qualified"):
        Polynomial.feature("missing")


def test_module_apply_clears_policy_dependent_plans_before_dtype_replanning():
    policy = FormulaPolicy(
        rules=(
            RouteRule("product", "full_table", score=Polynomial.feature("dtype.bytes")),
            RouteRule("product", "sparse", score=Polynomial(constant=6.0)),
        )
    )

    algebra = configured_algebra(3, device="cpu", dtype=torch.float32, planning_policy=policy)
    layout = algebra.spec.full_layout()

    def request(dtype):
        return ProductRequest.compact(
            algebra.spec,
            op="geometric_product",
            left_layout=layout,
            right_layout=layout,
            output_layout=layout,
            dtype=dtype,
            device="cpu",
        )

    first = algebra._planner.product_executor(request(torch.float32))
    assert isinstance(first, FullTableProductExecutor)

    algebra._apply(lambda value: value.to(torch.float64))
    assert not algebra._planner._product_executors
    second = algebra._planner.product_executor(request(torch.float64))
    assert isinstance(second, GradeProductExecutor)


def test_config_rejects_unknown_fields_instead_of_silently_ignoring_them():
    with pytest.raises(TypeError, match="unexpected_policy"):
        AlgebraConfig.from_mapping({"p": 3, "unexpected_policy": object()})


from clifra.core._kernel.configuration import configured_algebra
