# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from clifra.core.legacy import product_method_entry
from clifra.core.runtime.algebra import AlgebraContext
from tests.helpers.hypothesis_cases import (
    PROPERTY_SETTINGS,
    compact_product_cases,
    full_product_cases,
    signature_strategy,
    tensor_with_shape,
)
from tests.helpers.small_oracle import SmallCliffordOracle

pytestmark = pytest.mark.unit

def _product(algebra: AlgebraContext, op: str, left: torch.Tensor, right: torch.Tensor, **kwargs) -> torch.Tensor:
    return getattr(algebra, product_method_entry(op)[1])(left, right, **kwargs)


@PROPERTY_SETTINGS
@given(case=full_product_cases())
def test_full_lane_products_match_small_oracle(case):
    signature, op, left, right = case
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(*signature)

    actual = _product(algebra, op, left, right)
    expected = oracle.product(left, right, op=op)

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


@PROPERTY_SETTINGS
@given(case=compact_product_cases())
def test_compact_products_match_small_oracle_for_declared_layouts(case):
    signature, op, left_grades, right_grades, output_grades, left, right = case
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(*signature)
    left_layout = algebra.layout(left_grades)
    right_layout = algebra.layout(right_grades)
    output_layout = algebra.layout(output_grades)

    actual = _product(
        algebra,
        op,
        left,
        right,
        left_layout=left_layout,
        right_layout=right_layout,
        output_layout=output_layout,
    )
    expected = oracle.product(
        left,
        right,
        op=op,
        left_indices=left_layout.basis_indices,
        right_indices=right_layout.basis_indices,
        output_indices=output_layout.basis_indices,
    )

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


@PROPERTY_SETTINGS
@given(signature=signature_strategy(max_n=4), data=st.data())
def test_geometric_product_is_associative_against_small_oracle(signature, data):
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(*signature)
    batch = data.draw(st.integers(min_value=1, max_value=2))
    dim = algebra.dim
    left = data.draw(tensor_with_shape((batch, dim)))
    middle = data.draw(tensor_with_shape((batch, dim)))
    right = data.draw(tensor_with_shape((batch, dim)))

    actual = algebra.geometric_product(algebra.geometric_product(left, middle), right)
    expected = oracle.product(oracle.product(left, middle), right)

    assert torch.allclose(actual, expected, atol=1e-9, rtol=1e-9)
    assert torch.allclose(actual, algebra.geometric_product(left, algebra.geometric_product(middle, right)), atol=1e-9)


@PROPERTY_SETTINGS
@given(signature=signature_strategy(min_n=1, max_n=5), data=st.data())
def test_vector_wedge_with_itself_is_zero_and_matches_oracle(signature, data):
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(*signature)
    batch = data.draw(st.integers(min_value=1, max_value=3))
    vector = data.draw(tensor_with_shape((batch, algebra.n)))
    values = algebra.embed_vector(vector)

    actual = algebra.wedge(values, values)
    expected = oracle.product(values, values, op="wedge")

    assert torch.allclose(expected, torch.zeros_like(expected), atol=1e-12, rtol=1e-12)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
