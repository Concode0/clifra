"""An external-style provider: this file imports no private clifra modules."""

from dataclasses import FrozenInstanceError, dataclass, replace

import pytest
import torch
from torch import nn

from clifra import AlgebraContext, ExecutorRegistry, ResourceLimits, TensorContract, make_algebra
from clifra.core.executors import Assessment, ExecutorRequest, Rejected


@dataclass(frozen=True)
class ScalarPreparation:
    dtype: torch.dtype
    device: torch.device


class ScalarProduct(nn.Module):
    def __init__(self, preparation):
        super().__init__()
        self.register_buffer("one", torch.ones((), dtype=preparation.dtype, device=preparation.device))

    def forward(self, left, right):
        return left * right * self.one


class ParameterScalarProduct(nn.Module):
    def __init__(self, preparation):
        super().__init__()
        self.scale = nn.Parameter(torch.ones((), dtype=preparation.dtype, device=preparation.device))

    def forward(self, left, right):
        return left * right * self.scale


@dataclass(frozen=True)
class ExternalScalarProvider:
    calls: list
    identity: tuple[str, str] = ("product", "external_scalar")
    lanes: int = 1
    pairs: int = 1

    def assess(self, request):
        assert type(request) is ExecutorRequest
        assert set(vars(request)) == {"family", "operation", "inputs", "output", "dtype", "device"}
        if request.operation != "geometric_product" or any(
            contract.layout.grades != (0,) for contract in (*request.inputs, request.output)
        ):
            return Rejected("only scalar geometric products")
        assessment = Assessment(
            lanes=self.lanes,
            pairs=self.pairs,
            preparation=ScalarPreparation(request.dtype, request.device),
        )
        self.calls.append(("assess", request, assessment))
        return assessment

    def build(self, request, assessment):
        assert any(
            event == "assess" and original is request and result is assessment for event, original, result in self.calls
        )
        self.calls.append(("build", request, assessment))
        return ScalarProduct(assessment.preparation)


@dataclass(frozen=True)
class ExternalParameterProvider(ExternalScalarProvider):
    identity: tuple[str, str] = ("product", "external_parameter")

    def build(self, request, assessment):
        self.calls.append(("build", request, assessment))
        return ParameterScalarProduct(assessment.preparation)


def registry_with_external(calls):
    return ExecutorRegistry((ExternalScalarProvider(calls), *ExecutorRegistry.default().providers))


def test_external_provider_uses_only_public_contract_and_preserves_storage_and_gradients():
    calls = []
    registry = registry_with_external(calls)
    algebra = make_algebra(3, dtype=torch.float64, registry=registry)
    assert algebra.registry is registry
    scalar = algebra.layout((0,))
    left = torch.randn(2, 1, dtype=torch.float64, requires_grad=True)
    right = torch.randn(2, 1, dtype=torch.float64, requires_grad=True)
    operation = algebra.plan_product(left=scalar, right=scalar, output=scalar)
    assert [event for event, *_ in calls] == ["assess", "build"]
    actual = operation(left, right)
    torch.testing.assert_close(actual, left * right)
    gradients = torch.autograd.grad(actual.sum(), (left, right))
    torch.testing.assert_close(gradients[0], right)
    torch.testing.assert_close(gradients[1], left)
    assert [event for event, *_ in calls] == ["assess", "build"]

    canonical = TensorContract.canonical(scalar)
    full_operation = algebra.plan_product(left=canonical, right=canonical, output=canonical)
    torch.testing.assert_close(
        full_operation(scalar.full(left), scalar.full(right)),
        scalar.full(left * right),
    )
    # Storage conversion does not require exposing a different executor contract.
    assert all(contract.uses_compact_storage for contract in calls[0][1].inputs)
    moved = operation.to(dtype=torch.float32)
    torch.testing.assert_close(moved(left.float(), right.float()), left.float() * right.float())


def test_parameter_only_cached_external_kernel_moves_independently_between_plans():
    calls = []
    provider = ExternalParameterProvider(calls)
    algebra = AlgebraContext(2, registry=ExecutorRegistry((provider,)))
    scalar = algebra.layout((0,))
    first = algebra.plan_product(left=scalar, right=scalar, output=scalar)
    second = algebra.plan_product(left=scalar, right=scalar, output=scalar)
    assert first._kernel.executor is second._kernel.executor

    first.to(dtype=torch.float64)

    assert first._kernel.executor is not second._kernel.executor
    assert first._kernel.executor.scale.dtype == torch.float64
    assert second._kernel.executor.scale.dtype == torch.float32
    value = torch.tensor([2.0])
    torch.testing.assert_close(first(value.double(), value.double()), torch.tensor([4.0], dtype=torch.float64))
    torch.testing.assert_close(second(value, value), torch.tensor([4.0]))


def test_external_provider_pairwise_and_compile_need_only_forward():
    calls = []
    algebra = AlgebraContext(2, registry=registry_with_external(calls))
    scalar = algebra.layout((0,))
    operation = algebra.plan_product(left=scalar, right=scalar, output=scalar, pairwise=True)
    left, right = torch.randn(2, 3, 1), torch.randn(2, 4, 1)
    expected = left.unsqueeze(-2) * right.unsqueeze(-3)
    compiled = torch.compile(operation, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(left, right), expected)
    assert [event for event, *_ in calls] == ["assess", "build"]


def test_external_rejection_falls_back_and_registries_are_isolated():
    calls = []
    registry = registry_with_external(calls)
    extended, ordinary = AlgebraContext(2, registry=registry), AlgebraContext(2)
    left, right = torch.randn(4), torch.randn(4)
    torch.testing.assert_close(extended.geometric_product(left, right), ordinary.geometric_product(left, right))
    assert not calls
    scalar = extended.layout((0,))
    extended.plan_product(left=scalar, right=scalar, output=scalar)
    assert len(calls) == 2
    ordinary.plan_product(left=scalar, right=scalar, output=scalar)
    assert len(calls) == 2
    with pytest.raises(FrozenInstanceError):
        registry.providers = ()
    with pytest.raises(AttributeError):
        extended.registry = ordinary.registry
    with pytest.raises(ValueError, match="duplicate"):
        ExecutorRegistry((registry.providers[0], registry.providers[0]))


def test_explicit_empty_registry_does_not_silently_restore_defaults():
    algebra = AlgebraContext(2, registry=ExecutorRegistry(()))
    with pytest.raises(ValueError, match="No implemented product"):
        algebra.plan_product()


def test_public_assessment_carries_resource_counts_and_provider_preparation():
    calls = []
    provider = ExternalScalarProvider(calls, lanes=0, pairs=0)
    a = AlgebraContext(3, registry=ExecutorRegistry((provider,)))
    scalar = a.layout((0,))
    operation = a.plan_product(left=scalar, right=scalar, output=scalar)
    request, assessment = calls[0][1:]
    assert request.device == torch.device("cpu")
    assert request.dtype == torch.float32
    assert calls[1][1] is request and calls[1][2] is assessment
    assert isinstance(assessment.preparation, ScalarPreparation)
    torch.testing.assert_close(operation(torch.tensor([2.0]), torch.tensor([3.0])), torch.tensor([6.0]))


@pytest.mark.parametrize("external_first", [True, False])
def test_registry_order_defines_external_precedence_against_builtins(external_first):
    calls = []
    external = ExternalScalarProvider(calls)
    builtins = ExecutorRegistry.default().providers
    providers = (external, *builtins) if external_first else (*builtins, external)
    a = AlgebraContext(3, registry=ExecutorRegistry(providers))
    scalar = a.layout((0,))
    operation = a.plan_product(left=scalar, right=scalar, output=scalar)
    torch.testing.assert_close(operation(torch.tensor([2.0]), torch.tensor([3.0])), torch.tensor([6.0]))
    assert bool(calls) == external_first
    assert any(isinstance(module, ScalarProduct) for module in operation.modules()) == external_first


def test_appended_external_provider_extends_builtin_capability():
    calls = []
    # Built-in tensorized products reject n > 63. Scalar multiplication itself
    # has no such restriction and this external module needs no bitmask tables.
    providers = (*ExecutorRegistry.default().providers, ExternalScalarProvider(calls))
    a = AlgebraContext(64, registry=ExecutorRegistry(providers))
    scalar = a.layout((0,))
    operation = a.plan_product(left=scalar, right=scalar, output=scalar)
    assert [event for event, *_ in calls] == ["assess", "build"]
    torch.testing.assert_close(operation(torch.tensor([2.0]), torch.tensor([3.0])), torch.tensor([6.0]))


def test_external_resource_rejection_falls_through_without_build():
    oversized, first, second = [], [], []
    providers = (
        ExternalScalarProvider(oversized, identity=("product", "oversized"), pairs=10**12),
        ExternalScalarProvider(first, identity=("product", "first")),
        ExternalScalarProvider(second, identity=("product", "second")),
        *ExecutorRegistry.default().providers,
    )
    a = AlgebraContext(3, registry=ExecutorRegistry(providers))
    scalar = a.layout((0,))
    a.plan_product(left=scalar, right=scalar, output=scalar)
    assert [event for event, *_ in oversized] == ["assess"]
    assert [event for event, *_ in first] == ["assess", "build"]
    assert second == []


def test_public_budget_filters_external_assessment_before_build():
    oversized, feasible = [], []
    registry = ExecutorRegistry(
        (
            ExternalScalarProvider(oversized, identity=("product", "too_large"), pairs=11),
            ExternalScalarProvider(feasible, identity=("product", "fits"), pairs=10),
        )
    )
    algebra = make_algebra(3, registry=registry, resource_limits=ResourceLimits(max_pairs=10))
    scalar = algebra.layout((0,))
    product = algebra.plan_product(left=scalar, right=scalar, output=scalar)
    assert [event for event, *_ in oversized] == ["assess"]
    assert [event for event, *_ in feasible] == ["assess", "build"]
    torch.testing.assert_close(product(torch.tensor([2.0]), torch.tensor([3.0])), torch.tensor([6.0]))


@pytest.mark.parametrize("change", [{"lanes": -1}, {"pairs": 1.5}, {"lanes": True}, {"pairs": float("nan")}])
def test_public_assessment_rejects_invalid_resource_counts(change):
    with pytest.raises((TypeError, ValueError)):
        replace(Assessment(), **change)
