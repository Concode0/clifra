# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Representative public planned-operation compilation contracts."""

import pytest
import torch

from clifra.core import AlgebraContext
from clifra.core._kernel.configuration import configured_algebra
from tests.helpers.policy import PreferRoute

pytestmark = pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")


@pytest.mark.parametrize("route", ["sparse", "full_table"])
def test_product_routes_compile_fullgraph_with_gradients(route):
    algebra = configured_algebra(4, dtype=torch.float64, planning_policy=PreferRoute("product", route))
    layout = algebra.layout() if route == "full_table" else algebra.layout((1, 2))
    operation = algebra.plan_product(left=layout, right=layout, output=layout)
    left = torch.randn(2, 1, layout.dim, dtype=torch.float64, requires_grad=True)
    right = torch.randn(1, 3, layout.dim, dtype=torch.float64, requires_grad=True)
    compiled = torch.compile(operation, backend="aot_eager", fullgraph=True)
    expected, actual = operation(left, right), compiled(left, right)
    torch.testing.assert_close(actual, expected)
    seed = torch.randn_like(actual)
    for got, want in zip(
        torch.autograd.grad(actual, (left, right), seed),
        torch.autograd.grad(expected, (left, right), seed),
    ):
        torch.testing.assert_close(got, want)


def test_pairwise_product_compiles_fullgraph_with_broadcast_gradients():
    algebra = AlgebraContext(5, dtype=torch.float64)
    vector, output = algebra.layout((1,)), algebra.layout((0, 2))
    operation = algebra.plan_product(left=vector, right=vector, output=output, pairwise=True)
    left = torch.randn(2, 3, vector.dim, dtype=torch.float64, requires_grad=True)
    right = torch.randn(1, 4, vector.dim, dtype=torch.float64, requires_grad=True)
    compiled = torch.compile(operation, backend="aot_eager", fullgraph=True)
    expected, actual = operation(left, right), compiled(left, right)
    torch.testing.assert_close(actual, expected)
    for got, want in zip(
        torch.autograd.grad(actual.sum(), (left, right)),
        torch.autograd.grad(expected.sum(), (left, right)),
    ):
        torch.testing.assert_close(got, want)


def test_large_leading_sparse_strategy_compiles_fullgraph_with_gradients():
    algebra = AlgebraContext(8, dtype=torch.float64)
    vector, output = algebra.layout((1,)), algebra.layout((0, 2))
    operation = algebra.plan_product(left=vector, right=vector, output=output)
    left = torch.randn(128, 1, vector.dim, dtype=torch.float64, requires_grad=True)
    right = torch.randn(1, 2, vector.dim, dtype=torch.float64, requires_grad=True)
    compiled = torch.compile(operation, backend="aot_eager", fullgraph=True)

    expected, actual = operation(left, right), compiled(left, right)

    torch.testing.assert_close(actual, expected)
    for got, want in zip(
        torch.autograd.grad(actual.sum(), (left, right)),
        torch.autograd.grad(expected.sum(), (left, right)),
    ):
        torch.testing.assert_close(got, want)


def test_unary_layout_conversion_compiles_fullgraph():
    algebra = AlgebraContext(5)
    source, output = algebra.layout((1, 2)), algebra.layout((2,))
    operation = algebra.plan_unary(op="clifford_conjugation", input=source, output=output)
    values = torch.randn(3, source.dim)
    compiled = torch.compile(operation, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(values), operation(values))


@pytest.mark.parametrize("route", ["vector_matrix", "rotor_product"])
def test_versor_action_routes_compile_fullgraph(route):
    algebra = configured_algebra(3, dtype=torch.float64, planning_policy=PreferRoute("action", route))
    layout, parameter = algebra.layout((1,)), algebra.layout((2,))
    operation = algebra.plan_versor_action(grade=2, input=layout, output=layout, parameter=parameter)
    values = torch.randn(2, 1, layout.dim, dtype=torch.float64, requires_grad=True)
    weights = (0.1 * torch.randn(3, parameter.dim, dtype=torch.float64)).requires_grad_()
    compiled = torch.compile(operation, backend="aot_eager", fullgraph=True)
    expected, actual = operation(values, weights), compiled(values, weights)
    torch.testing.assert_close(actual, expected)
    for got, want in zip(
        torch.autograd.grad(actual.sum(), (values, weights)),
        torch.autograd.grad(expected.sum(), (values, weights)),
    ):
        torch.testing.assert_close(got, want)


def test_bivector_exponential_compiles_fullgraph():
    algebra = AlgebraContext(4)
    bivector, even = algebra.layout((2,)), algebra.layout((0, 2, 4))
    operation = algebra.plan_bivector_exp(input=bivector, output=even)
    values = torch.randn(2, bivector.dim) * 0.1
    compiled = torch.compile(operation, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(values), operation(values))


@pytest.mark.parametrize("kind", ["signature_norm", "pseudoscalar"])
def test_metric_and_permutation_compile_fullgraph(kind):
    algebra = AlgebraContext(4)
    layout = algebra.layout((1, 2))
    operation = (
        algebra.plan_signature_norm_squared(input=layout)
        if kind == "signature_norm"
        else algebra.plan_pseudoscalar_product(input=layout)
    )
    values = torch.randn(2, layout.dim)
    compiled = torch.compile(operation, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(values), operation(values))


def test_compiled_strict_inverse_rejects_singular_input():
    algebra = AlgebraContext(1, 1)
    inverse = torch.compile(
        algebra.plan_strict_blade_inverse(input=algebra.layout((1,))), backend="aot_eager", fullgraph=True
    )
    with pytest.raises(RuntimeError):
        inverse(torch.tensor([1.0, 1.0]))


def test_multigrade_reflection_compiles_fullgraph():
    algebra = AlgebraContext(3)
    values = torch.randn(2, algebra.dim)
    normals = torch.randn(2, 3)
    operation = algebra.plan_reflect(input=algebra.layout(), normal=algebra.layout((1,)))
    compiled = torch.compile(operation, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(values, normals), operation(values, normals))
