import warnings
from dataclasses import FrozenInstanceError, dataclass

import pytest
import torch

from clifra.core import AlgebraContext, ResourceLimits, TensorContract
from clifra.core._kernel.configuration import configured_algebra
from clifra.core._kernel.planning.policy import (
    PlanCandidate,
    PolicyCoverageError,
    PolicyEvaluation,
    ProductFacts,
    select_policy_route,
)
from clifra.core.executors import ExecutorRequest
from tests.planning._grade_plan_helpers import select_product_route


@dataclass(frozen=True)
class ScoringPolicy:
    score: object

    def evaluate(self, candidate):
        return PolicyEvaluation(self.score(candidate), "test_rejection")


def candidate(route, interactions=1):
    algebra = AlgebraContext(2)
    contract = TensorContract.compact(algebra.layout())
    request = ExecutorRequest("product", "geometric_product", (contract, contract), contract, torch.float32, "cpu")
    return PlanCandidate("product", route, request, ProductFacts(interactions))


def test_minimum_score_and_registration_order_ties():
    candidates = (candidate("first", 2), candidate("second", 1))
    assert select_policy_route(ScoringPolicy(lambda c: c.facts.interactions), candidates).route == "second"
    assert select_policy_route(ScoringPolicy(lambda c: 0), candidates).route == "first"


def test_uncovered_routes_report_rejection():
    with pytest.raises(PolicyCoverageError, match="test_rejection"):
        select_policy_route(ScoringPolicy(lambda c: None), (candidate("route"),))


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_policy_scores_rejected(score):
    with pytest.raises(ValueError, match="finite"):
        select_policy_route(ScoringPolicy(lambda c: score), (candidate("route"),))


def test_family_facts_are_immutable_and_nonnegative():
    facts = ProductFacts(3, 2)
    with pytest.raises(FrozenInstanceError):
        facts.interactions = 4
    with pytest.raises(ValueError, match="non-negative integer"):
        ProductFacts(-1)


def test_dtype_replanning_uses_request_context():
    policy = ScoringPolicy(lambda c: c.request.dtype.itemsize if c.route == "full_table" else 6)
    algebra = configured_algebra(3, dtype=torch.float32, planning_policy=policy)
    first = algebra.plan_product()._kernel
    algebra.to(dtype=torch.float64)
    second = algebra.plan_product()._kernel
    assert first.route == "full_table"
    assert second.route == "sparse"


def test_default_product_policy_prefers_pruning_and_is_device_independent():
    routes = []
    for device in ("cpu", "mps", "cuda"):
        algebra = AlgebraContext(4, device="cpu")
        full = algebra.layout()
        decision = select_product_route(
            algebra,
            op="wedge",
            left_layout=full,
            right_layout=full,
            output_layout=full,
            dtype=torch.float32,
            device=device,
        )
        routes.append(decision.route)
    assert len(set(routes)) == 1


def test_unselected_large_route_does_not_warn():
    algebra = configured_algebra(
        4,
        resource_limits=ResourceLimits(warn_lanes=1000, warn_pairs=300, max_pairs=1000),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        executor = algebra.plan_product()._kernel

    assert executor.route == "full_table"


def test_resource_rejection_precedes_builtin_policy_ranking():
    seen = []

    class RecordingPolicy:
        def evaluate(self, item):
            seen.append(item.route)
            return PolicyEvaluation(0)

    algebra = configured_algebra(
        4,
        planning_policy=RecordingPolicy(),
        resource_limits=ResourceLimits(max_pairs=300),
    )
    executor = algebra.plan_product()._kernel

    assert executor.route == "full_table"
    assert seen == ["full_table"]
