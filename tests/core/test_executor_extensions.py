"""An external-style provider: this file imports no private clifra modules."""

from dataclasses import FrozenInstanceError, dataclass

import pytest
import torch
from torch import nn

from clifra import AlgebraContext, ExecutorRegistry, TensorContract, make_algebra
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


@dataclass(frozen=True)
class ExternalScalarProvider:
    calls: list
    identity = ("product", "external_scalar")

    def assess(self, request):
        assert type(request) is ExecutorRequest
        assert set(vars(request)) == {"family", "operation", "inputs", "output", "dtype", "device"}
        if request.operation != "geometric_product" or any(
            contract.layout.grades != (0,) for contract in (*request.inputs, request.output)
        ):
            return Rejected("only scalar geometric products")
        assessment = Assessment(
            lanes=1,
            pairs=1,
            forward_work=0,
            exact=True,
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
