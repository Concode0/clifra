"""Final built-in policy decisions from authoritative structural profiles."""

from dataclasses import astuple, replace

import pytest
import torch

from clifra.core import AlgebraContext, AlgebraSpec, TensorContract
from clifra.core._kernel.planning.exp import assess_bivector_exp_route
from clifra.core._kernel.planning.policy import DEFAULT_PLANNING_POLICY, PlanCandidate, select_policy_route
from clifra.core._kernel.planning.product import assess_product_routes
from clifra.core._kernel.providers import BuiltinProvider, action_execution_request, exp_execution_request
from clifra.core.executors import ExecutorRequest


def test_product_scores_use_only_bulk_and_output_profile():
    for n, expected in ((5, "sparse"), (8, "full_table")):
        spec = AlgebraSpec(n)
        full = spec.full_layout()
        contract = TensorContract.compact(full)
        request = ExecutorRequest("product", "geometric_product", (contract, contract), contract, torch.float32, "cpu")
        candidates = tuple(
            PlanCandidate("product", assessment.route, request, assessment.facts)
            for assessment in assess_product_routes(
                op="geometric_product", left_layout=full, right_layout=full, output_layout=full
            )
        )
        scores = {item.route: DEFAULT_PLANNING_POLICY.evaluate(item).score for item in candidates}
        profiles = {item.route: item.facts.work_profile for item in candidates}
        assert scores["sparse"] == profiles["sparse"].bulk + profiles["sparse"].output
        assert scores["full_table"] == 0.5 * spec.dim**2 + 32 * spec.dim
        assert select_policy_route(DEFAULT_PLANNING_POLICY, candidates).route == expected


@pytest.mark.parametrize(
    "signature,dtype,expected",
    [
        ((3, 0, 0), torch.float32, "closed"),
        ((4, 0, 0), torch.float32, "left_matrix_exp"),
        ((4, 0, 0), torch.float64, "closed"),
        ((3, 1, 0), torch.float64, "left_matrix_exp"),
        ((5, 0, 0), torch.float32, "closed"),
    ],
)
def test_exp_direct_assessment_has_complete_profile_and_calibrated_route(signature, dtype, expected):
    spec = AlgebraSpec(*signature)
    output = spec.layout(range(0, spec.n + 1, 2))
    request = exp_execution_request(spec, "cpu", dtype, output, planner=None)
    candidates = []
    for route in ("closed", "left_matrix_exp", "taylor"):
        assessed = BuiltinProvider(("bivector_exp", route)).assess(request)
        direct = assess_bivector_exp_route(spec, "cpu", dtype=dtype, output_layout=output, route=route)
        assert assessed.preparation.facts == direct.facts
        profile = assessed.preparation.facts.work_profile
        assert profile.route == route
        if route == "taylor":
            assert len(profile.plain_products) == (18 if dtype == torch.float64 else 12)
            assert profile.square_product is not None
        candidates.append(PlanCandidate("bivector_exp", route, request, direct.facts))
    assert select_policy_route(DEFAULT_PLANNING_POLICY, tuple(candidates)).route == expected
    matrix = next(item for item in candidates if item.route == "left_matrix_exp")
    assert DEFAULT_PLANNING_POLICY.evaluate(matrix).reason == (
        "small_matrix_exp_regime" if expected == "left_matrix_exp" else "structural_exp_work"
    )


def test_action_score_uses_direct_endpoint_and_selected_children_without_device_choice():
    algebra = AlgebraContext(4)
    request = action_execution_request(
        algebra,
        "versor",
        grade=2,
        input_layout=algebra.layout(),
        output_layout=algebra.layout(),
        parameter_layout=algebra.layout((2,)),
    )
    scores = {}
    for route in ("vector_matrix", "rotor_product", "full_action_matrix"):
        assessed = BuiltinProvider(("action", route)).assess(request)
        facts = assessed.preparation.facts
        profile = facts.work_profile
        score = DEFAULT_PLANNING_POLICY.evaluate(PlanCandidate("action", route, request, facts)).score
        scores[route] = score
        if route == "vector_matrix":
            lift = profile.lift.direct()
            assert score == profile.generator_terms + 0.2 * profile.vector_matrix_exp_order + 20 * sum(astuple(lift))
            assert score != profile.generator_terms + 0.2 * profile.vector_matrix_exp_order + 20 * sum(
                astuple(profile.lift.compound())
            )
            mps = replace(request, device=torch.device("mps"))
            mps_facts = BuiltinProvider(("action", route)).assess(mps).preparation.facts
            assert mps_facts == facts
            assert DEFAULT_PLANNING_POLICY.evaluate(PlanCandidate("action", route, mps, mps_facts)).score == score
        elif route == "rotor_product":
            assert len(profile.product_children) == 2
            assert profile.exponential_child.route == "left_matrix_exp"
        else:
            assert profile.full_action_matrix_order == algebra.spec.dim**3
            assert profile.full_action_cells == algebra.spec.dim**2
    assert min(scores, key=scores.get) == "full_action_matrix"


def test_taylor_cpu_partition_proxy_matches_previous_pair_and_scatter_count():
    algebra = AlgebraContext(4, dtype=torch.float32)
    output = algebra.layout((0,))
    request = exp_execution_request(algebra.spec, "cpu", torch.float32, output, planner=algebra._planner)
    assessment = BuiltinProvider(("bivector_exp", "taylor")).assess(request)
    executor = BuiltinProvider(("bivector_exp", "taylor")).build(request, assessment)
    profile = assessment.preparation.facts.work_profile

    def old_proxy(child):
        return child.bulk + (child.bulk if child.route == "sparse" and child.output > 1 and child.bulk > 0 else 0)

    assert executor.taylor_work == (
        sum(map(old_proxy, profile.plain_products)),
        sum(map(old_proxy, profile.scaled_products)),
        old_proxy(profile.square_product),
    )
