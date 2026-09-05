from dataclasses import dataclass

import pytest
import torch

from clifra.core._kernel.configuration import configured_algebra
from clifra.core._kernel.planning.policy import (
    PlanCandidate,
    PlanFacts,
    PolicyCoverageError,
    PolicyEvaluation,
    select_policy_route,
)


@dataclass(frozen=True)
class ScoringPolicy:
    score: object

    def evaluate(self, candidate):
        return PolicyEvaluation(self.score(candidate), "test_rejection")


def candidate(route, work=1, reason=None):
    return PlanCandidate("product", route, PlanFacts(forward_work=work), reason)


def test_minimum_score_and_registration_order_ties():
    candidates = (candidate("first", 2), candidate("second", 1))
    assert select_policy_route(ScoringPolicy(lambda c: c.facts.forward_work), candidates).route == "second"
    assert select_policy_route(ScoringPolicy(lambda c: 0), candidates).route == "first"


def test_unsupported_route_never_reaches_policy():
    seen = []

    def score(item):
        seen.append(item.route)
        return 0

    decision = select_policy_route(
        ScoringPolicy(score),
        (candidate("unsupported", reason="domain"), candidate("supported")),
    )
    assert decision.route == "supported"
    assert seen == ["supported"]


def test_uncovered_routes_report_rejection():
    with pytest.raises(PolicyCoverageError, match="test_rejection"):
        select_policy_route(ScoringPolicy(lambda c: None), (candidate("route"),))


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_policy_scores_rejected(score):
    with pytest.raises(ValueError, match="finite"):
        select_policy_route(ScoringPolicy(lambda c: score), (candidate("route"),))


def test_extension_facts_are_immutable_and_qualified():
    facts = PlanFacts(extensions={"vendor.score": 3})
    assert facts["vendor.score"] == 3
    with pytest.raises(TypeError):
        facts.extensions[0] = ("vendor.score", 4)
    with pytest.raises(ValueError, match="dot-qualified"):
        PlanFacts(extensions={"unqualified": 3})


def test_dtype_replanning_uses_fresh_selection():
    policy = ScoringPolicy(lambda c: c.facts["dtype.bytes"] if c.route == "full_table" else 6)
    algebra = configured_algebra(3, dtype=torch.float32, planning_policy=policy)
    first = algebra.plan_product()._kernel
    algebra.to(dtype=torch.float64)
    second = algebra.plan_product()._kernel
    assert first.metadata.route == "full_table"
    assert second.metadata.route == "sparse"
