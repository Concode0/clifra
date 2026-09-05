from dataclasses import FrozenInstanceError, dataclass, replace

import pytest
import torch
from torch import nn

from clifra.core import AlgebraContext, TensorContract
from clifra.core._kernel.configuration import configured_algebra
from clifra.core._kernel.execution.providers import (
    _exp_child,
    action_execution_request,
    product_execution_request,
)
from clifra.core._kernel.execution.registry import ExecutorRegistry
from clifra.core._kernel.planning.layouts import ProductRequest
from clifra.core._kernel.planning.policy import PolicyEvaluation
from clifra.core._kernel.planning.resources import ResourceLimits
from clifra.core.executors import Assessment, ExecutorRequest, Rejected


class EqualPolicy:
    def evaluate(self, candidate):
        return PolicyEvaluation(0)


@dataclass(frozen=True)
class MockProvider:
    identity: tuple
    result: object

    def assess(self, request):
        return self.result

    def build(self, request, assessment):
        assert assessment is self.result
        return nn.Identity()


def request():
    algebra = AlgebraContext(2)
    contract = TensorContract(algebra.layout((1,)))
    return ExecutorRequest("mock", "identity", (contract,), contract, torch.float32, "cpu")


def accepted(**overrides):
    return Assessment(lanes=2, pairs=2, forward_work=1, exact=True, **overrides)


def test_rejected_has_no_cost_quality_or_preparation():
    result = Rejected("unsupported")
    for name in ("forward_work", "exact", "preparation"):
        assert not hasattr(result, name)
    with pytest.raises(TypeError):
        Rejected("unsupported", forward_work=0)


def test_registry_immutable_duplicate_identity_and_ties():
    first = MockProvider(("mock", "first"), accepted())
    second = MockProvider(("mock", "second"), accepted())
    registry = ExecutorRegistry([first, second])
    assert isinstance(registry.providers, tuple)
    assert registry.select(request(), EqualPolicy()).provider is first
    with pytest.raises(FrozenInstanceError):
        registry.providers = ()
    with pytest.raises(ValueError, match="duplicate"):
        ExecutorRegistry((first, first))


def test_resource_and_capability_rejections_cannot_reach_policy_or_build():
    providers = (
        MockProvider(("mock", "unsupported"), Rejected("domain")),
        MockProvider(("mock", "large"), replace(accepted(), pairs=100)),
        MockProvider(("mock", "valid"), accepted(preparation=("owned",))),
    )

    class Policy:
        def evaluate(self, candidate):
            assert candidate.route == "valid"
            return PolicyEvaluation(0)

    registry = ExecutorRegistry(providers)
    selection = registry.select(request(), Policy(), ResourceLimits(max_pairs=2))
    assert selection.assessment is providers[-1].result
    assert isinstance(selection.build(), nn.Identity)
    with pytest.raises(ValueError, match="original"):
        registry.construct(request(), selection)


@pytest.mark.parametrize(
    "change",
    [
        {"lanes": -1},
        {"pairs": 1.5},
        {"forward_work": float("nan")},
        {"exact": True, "truncated": True},
        {"exact": True, "value_dependent": True},
    ],
)
def test_invalid_assessment_rejected(change):
    with pytest.raises(ValueError):
        replace(accepted(), **change)


@pytest.mark.parametrize("kind", ["product", "exponential", "action"])
def test_assess_allocates_no_execution_buffers_and_build_never_selects(monkeypatch, kind):
    algebra = AlgebraContext(4, dtype=torch.float64)
    spec, planner = algebra.spec, algebra._planner
    vector, bivector, even = spec.layout((1,)), spec.layout((2,)), spec.layout((0, 2, 4))

    def fail(*args, **kwargs):
        raise AssertionError("allocation during assessment or selection during build")

    with monkeypatch.context() as check:
        for name in ("empty", "zeros", "ones", "eye", "tensor", "arange"):
            check.setattr(torch, name, fail)
        if kind == "exponential":
            selection = _exp_child(planner, bivector, even, algebra.dtype, algebra.device)
        else:
            if kind == "product":
                declaration = ProductRequest.compact(
                    spec,
                    op="geometric_product",
                    left_layout=vector,
                    right_layout=vector,
                    output_layout=even,
                    dtype=algebra.dtype,
                    device=algebra.device,
                )
                declaration = product_execution_request(algebra, declaration)
            else:
                declaration = action_execution_request(
                    algebra,
                    "versor",
                    grade=2,
                    input_layout=even,
                    output_layout=even,
                    parameter_layout=bivector,
                )
            selection = planner.registry.select(declaration, planner.policy, planner.limits)
    with monkeypatch.context() as check:
        check.setattr(ExecutorRegistry, "select", fail)
        executor = selection.build()
    assert executor.metadata.route == selection.route
    assert executor.metadata.facts == selection.facts


def test_unavailable_rotor_children_do_not_hide_vector_matrix():
    algebra = configured_algebra(12, resource_limits=ResourceLimits(max_lanes=100, max_pairs=1000))
    vector, bivector = algebra.layout((1,)), algebra.layout((2,))
    operation = algebra.plan_versor_action(grade=2, input=vector, output=vector, parameter=bivector)
    assert operation._kernel.metadata.route == "vector_matrix"
