"""Mathematical coverage formerly exercised through the attention wrapper."""

import pytest
import torch

from clifra.core.algebra import AlgebraContext


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
