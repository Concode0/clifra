"""Mathematical coverage formerly exercised through the attention wrapper."""

import pytest
import torch

from clifra.core._kernel.configuration import configured_algebra
from clifra.core.algebra import AlgebraContext
from tests.helpers.policy import PreferRoute


@pytest.mark.parametrize("grades", [None, (1,)])
def test_pairwise_product_reductions_match_broadcast_reference_and_gradients(grades):
    algebra = AlgebraContext(4, dtype=torch.float64)
    layout = algebra.layout(grades)
    output = algebra.layout((0, 2))
    left = torch.randn(2, 3, layout.dim, dtype=torch.float64, requires_grad=True)
    right = torch.randn(2, 5, layout.dim, dtype=torch.float64, requires_grad=True)
    reverse = algebra.plan_unary(op="reverse", input=layout, output=layout)
    product = algebra.plan_product(left=layout, right=layout, output=output, pairwise=True)
    actual = product(left, reverse(right))
    expected = output.compact(
        algebra.geometric_product(
            layout.full(left).unsqueeze(-2),
            algebra.reverse(layout.full(right)).unsqueeze(-3),
        )
    )

    def reduce(value):
        scalar = value[..., output.positions_for_grades((0,))].sum(-1)
        bivectors = value[..., output.positions_for_grades((2,))]
        return scalar + 0.25 * bivectors.square().sum(-1).sqrt()

    actual, expected = reduce(actual), reduce(expected)
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual.sum(), (left, right), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), (left, right))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_pairwise_product_rejects_foreign_contract():
    algebra = AlgebraContext(3)
    foreign = AlgebraContext(0, 3)
    with pytest.raises(ValueError, match="signature"):
        algebra.plan_product(left=foreign.layout((1,)), pairwise=True)


def test_pairwise_scalar_product_ignores_nonfinite_null_lanes_and_their_gradients():
    algebra = AlgebraContext(1, 0, 1, dtype=torch.float64)
    vector, scalar = algebra.layout((1,)), algebra.layout((0,))
    product = algebra.plan_product(left=vector, right=vector, output=scalar, pairwise=True)
    left = torch.tensor([[1.0, float("inf")]], dtype=torch.float64, requires_grad=True)
    right = torch.tensor([[2.0, float("nan")]], dtype=torch.float64, requires_grad=True)

    actual = product(left, right)

    torch.testing.assert_close(actual, torch.tensor([[[2.0]]], dtype=torch.float64))
    gradients = torch.autograd.grad(actual.sum(), (left, right))
    for actual, expected in zip(gradients, ([[2.0, 0.0]], [[1.0, 0.0]])):
        torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64))


@pytest.mark.parametrize("swap", [False, True])
def test_pairwise_partial_output_preserves_only_participating_nonfinite_terms(swap):
    algebra = AlgebraContext(2, dtype=torch.float64)
    vector, scalar, output = algebra.layout((1,)), algebra.layout((0,)), algebra.layout((0, 1))
    layouts = (scalar, vector) if swap else (vector, scalar)
    product = algebra.plan_product(left=layouts[0], right=layouts[1], output=output, pairwise=True)
    vector_values = torch.tensor([[float("inf"), 3.0]], dtype=torch.float64)
    for scale, expected in ((2.0, [0.0, float("inf"), 6.0]), (0.0, [0.0, float("nan"), 0.0])):
        scalar_values = torch.tensor([[scale]], dtype=torch.float64)
        values = (scalar_values, vector_values) if swap else (vector_values, scalar_values)
        torch.testing.assert_close(product(*values), torch.tensor([[expected]], dtype=torch.float64), equal_nan=True)


def test_pairwise_full_table_excludes_null_interactions():
    algebra = configured_algebra(0, 0, 1, dtype=torch.float64, planning_policy=PreferRoute("product", "full_table"))
    product = algebra.plan_product(pairwise=True)
    left = torch.tensor([[0.0, float("inf")]], dtype=torch.float64)
    right = torch.tensor([[1.0, 0.0]], dtype=torch.float64)

    torch.testing.assert_close(product(left, right), torch.tensor([[[0.0, float("inf")]]], dtype=torch.float64))
