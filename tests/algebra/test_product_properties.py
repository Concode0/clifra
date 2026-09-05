# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from clifra.core._kernel.planning.policy import FormulaPolicy, Polynomial, RouteRule
from clifra.core.algebra import AlgebraContext
from tests.helpers.hypothesis_cases import (
    CORE_NUMERIC_SETTINGS,
    CORE_PROPERTY_SETTINGS,
    compact_product_cases,
    full_product_cases,
    signature_strategy,
    tensor_with_shape,
)
from tests.helpers.small_oracle import SmallCliffordOracle

pytestmark = [pytest.mark.unit, pytest.mark.property]

_PRODUCT_METHODS = {
    "geometric_product": "geometric_product",
    "wedge": "wedge",
    "symmetric_product": "symmetric_product",
    "commutator_product": "commutator_product",
    "anti_commutator_product": "anti_commutator_product",
    "left_contraction": "left_contraction",
    "right_contraction": "right_contraction",
}


def _product(algebra: AlgebraContext, op: str, a: torch.Tensor, b: torch.Tensor, **kwargs) -> torch.Tensor:
    return getattr(algebra, _PRODUCT_METHODS[op])(a, b, **kwargs)


@CORE_PROPERTY_SETTINGS
@given(case=full_product_cases())
def test_full_lane_products_match_small_oracle(case):
    signature, op, left, right = case
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(*signature)

    actual = _product(algebra, op, left, right)
    expected = oracle.product(left, right, op=op)

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


@CORE_PROPERTY_SETTINGS
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
        left=left_layout,
        right=right_layout,
        output=output_layout,
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


@CORE_PROPERTY_SETTINGS
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


@CORE_PROPERTY_SETTINGS
@given(signature=signature_strategy(min_n=1, max_n=5), data=st.data())
def test_vector_wedge_with_itself_is_zero_and_matches_oracle(signature, data):
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(*signature)
    batch = data.draw(st.integers(min_value=1, max_value=3))
    vector = data.draw(tensor_with_shape((batch, algebra.n)))
    values = algebra.layout((1,)).full(vector)

    actual = algebra.wedge(values, values)
    expected = oracle.product(values, values, op="wedge")

    assert torch.allclose(expected, torch.zeros_like(expected), atol=1e-12, rtol=1e-12)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


@CORE_NUMERIC_SETTINGS
@given(
    signature=signature_strategy(max_n=5),
    op=st.sampled_from(tuple(_PRODUCT_METHODS)),
    dtype=st.sampled_from((torch.float32, torch.float64)),
    data=st.data(),
)
def test_full_products_stay_within_a_conservative_roundoff_bound(signature, op, dtype, data):
    algebra = AlgebraContext(*signature, device="cpu", dtype=dtype)
    oracle = SmallCliffordOracle(*signature)
    batch = data.draw(st.integers(min_value=1, max_value=2))
    left = data.draw(tensor_with_shape((batch, algebra.dim), dtype=dtype))
    right = data.draw(tensor_with_shape((batch, algebra.dim), dtype=dtype))

    actual = _product(algebra, op, left, right).to(torch.float64)
    expected = oracle.product_fsum(left, right, op=op)
    term_count = max(2 * algebra.dim**2, 1)
    eps = torch.finfo(dtype).eps
    gamma = term_count * eps / (1.0 - term_count * eps)
    scale = 2.0 * left.abs().sum(dim=-1, keepdim=True) * right.abs().sum(dim=-1, keepdim=True)
    bound = gamma * scale.to(torch.float64)

    assert torch.all((actual - expected).abs() <= bound + torch.finfo(torch.float64).tiny)


@CORE_NUMERIC_SETTINGS
@given(case=full_product_cases(max_n=5))
def test_full_product_routes_match_the_independent_oracle(case):
    signature, op, left, right = case
    oracle = SmallCliffordOracle(*signature)
    expected = oracle.product(left, right, op=op)
    for selected, other in (("full_table", "sparse"), ("sparse", "full_table")):
        policy = FormulaPolicy(
            (
                RouteRule("product", selected),
                RouteRule("product", other, score=Polynomial(constant=1.0)),
            )
        )
        algebra = configured_algebra(*signature, device="cpu", dtype=torch.float64, planning_policy=policy)
        layout = algebra.spec.full_layout()
        product = algebra.plan_product(op=op, left=layout, right=layout, output=layout)

        assert product._kernel.executor_family == selected
        assert torch.allclose(product(left, right), expected, atol=1e-10, rtol=1e-10)


from clifra.core._kernel.configuration import configured_algebra
