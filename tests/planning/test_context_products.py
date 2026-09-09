# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Integrated public context planning over explicit layouts."""

import pytest
import torch

from clifra.core import AlgebraContext, PlannedOperation, TensorContract, make_algebra
from tests.helpers.small_oracle import SmallCliffordOracle


def test_projected_product_matches_oracle_and_returns_declared_lanes():
    algebra = AlgebraContext(4, 1, 1, dtype=torch.float64)
    oracle = SmallCliffordOracle(4, 1, 1)
    left_layout, right_layout, output = algebra.layout((1,)), algebra.layout((1, 2)), algebra.layout((0, 1, 2, 3))
    left = torch.randn(2, left_layout.dim, dtype=torch.float64)
    right = torch.randn(2, right_layout.dim, dtype=torch.float64)
    actual = algebra.geometric_product(left, right, left=left_layout, right=right_layout, output=output)
    expected = output.compact(oracle.product(left_layout.full(left), right_layout.full(right)))
    torch.testing.assert_close(actual, expected)


def test_preplanned_product_is_reusable_and_keeps_contracts():
    algebra = AlgebraContext(5)
    vector, output = algebra.layout((1,)), algebra.layout((0, 2))
    operation = algebra.plan_product(left=vector, right=vector, output=output)
    assert isinstance(operation, PlannedOperation)
    assert operation.inputs == (TensorContract.compact(vector), TensorContract.compact(vector))
    values = torch.randn(3, vector.dim)
    torch.testing.assert_close(
        operation(values, values),
        algebra.geometric_product(values, values, left=vector, right=vector, output=output),
    )


@pytest.mark.parametrize("family", ["product", "unary", "metric", "permutation", "exponential", "sandwich"])
def test_planning_rejects_foreign_layouts(family):
    algebra, foreign = AlgebraContext(3), AlgebraContext(0, 3)
    vector, foreign_vector = algebra.layout((1,)), foreign.layout((1,))
    with pytest.raises(ValueError, match="signature"):
        if family == "product":
            algebra.plan_product(left=foreign_vector, right=vector)
        elif family == "unary":
            algebra.plan_unary(op="reverse", input=foreign_vector)
        elif family == "metric":
            algebra.plan_signature_norm_squared(input=foreign_vector)
        elif family == "permutation":
            algebra.plan_pseudoscalar_product(input=foreign_vector)
        elif family == "exponential":
            algebra.plan_bivector_exp(input=foreign.layout((2,)))
        else:
            algebra.plan_sandwich_action(input=foreign.layout())


def test_equal_signature_layouts_are_interchangeable():
    algebra, peer = AlgebraContext(3), AlgebraContext(3)
    vector = peer.layout((1,))
    values = torch.randn(2, vector.dim)
    operation = algebra.plan_unary(op="reverse", input=vector)
    torch.testing.assert_close(operation(values), values)


def test_compact_high_dimensional_product_avoids_canonical_storage():
    algebra = AlgebraContext(32)
    vector, output = algebra.layout((1,)), algebra.layout((0, 2))
    operation = algebra.plan_product(left=vector, right=vector, output=output)
    values = torch.randn(2, vector.dim)
    result = operation(values, values)
    assert result.shape == (2, output.dim)
    assert output.dim < algebra.dim


def test_canonical_operand_contract_is_explicitly_normalized():
    algebra = AlgebraContext(4)
    vector, output = algebra.layout((1,)), algebra.layout((0, 2))
    compact = torch.randn(2, vector.dim)
    operation = algebra.plan_product(
        left=TensorContract.canonical(vector), right=vector, output=TensorContract.canonical(output)
    )
    torch.testing.assert_close(
        operation(vector.full(compact), compact),
        output.full(algebra.geometric_product(compact, compact, left=vector, right=vector, output=output)),
    )


def test_pairwise_product_requires_and_broadcasts_item_axes():
    algebra = AlgebraContext(4)
    vector = algebra.layout((1,))
    operation = algebra.plan_product(left=vector, right=vector, pairwise=True)
    with pytest.raises(ValueError):
        operation(torch.randn(vector.dim), torch.randn(vector.dim))
    left, right = torch.randn(2, 3, vector.dim), torch.randn(1, 5, vector.dim)
    expected = algebra.geometric_product(left.unsqueeze(-2), right.unsqueeze(-3), left=vector, right=vector)
    torch.testing.assert_close(operation(left, right), expected)


def test_context_defaults_to_canonical_full_layout_without_grade_inference():
    algebra = make_algebra(3)
    values = torch.randn(2, algebra.dim)
    torch.testing.assert_close(algebra.reverse(values), algebra.plan_unary(op="reverse")(values))
    with pytest.raises(ValueError, match="last dimension"):
        algebra.reverse(torch.randn(2, algebra.layout((1,)).dim))
