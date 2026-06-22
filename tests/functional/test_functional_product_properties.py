# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from clifra.core.runtime.algebra import AlgebraContext
from clifra.functional import (
    anti_commutator,
    clifford_conjugation,
    commutator,
    dual,
    geometric_product,
    grade_involution,
    grade_projection,
    inner_product,
    left_contraction,
    norm_sq,
    product,
    reverse,
    right_contraction,
    wedge,
)
from tests.helpers.hypothesis_cases import (
    PROPERTY_SETTINGS,
    compact_product_cases,
    full_multivector_cases,
    full_product_cases,
)
from tests.helpers.small_oracle import SmallCliffordOracle

pytestmark = pytest.mark.unit

FUNCTIONAL_PRODUCTS = {
    "gp": geometric_product,
    "wedge": wedge,
    "inner": inner_product,
    "commutator": commutator,
    "anti_commutator": anti_commutator,
    "left_contraction": left_contraction,
    "right_contraction": right_contraction,
}


@PROPERTY_SETTINGS
@given(case=full_product_cases())
def test_functional_full_lane_products_match_small_oracle(case):
    signature, op, left, right = case
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(*signature)

    assert torch.allclose(FUNCTIONAL_PRODUCTS[op](algebra, left, right), oracle.product(left, right, op=op))
    assert torch.allclose(product(algebra, left, right, op=op), oracle.product(left, right, op=op))


@PROPERTY_SETTINGS
@given(case=compact_product_cases())
def test_functional_compact_products_match_small_oracle(case):
    signature, op, left_grades, right_grades, output_grades, left, right = case
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(*signature)
    left_layout = algebra.layout(left_grades)
    right_layout = algebra.layout(right_grades)
    output_layout = algebra.layout(output_grades)

    actual = product(
        algebra,
        left,
        right,
        op=op,
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

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


@PROPERTY_SETTINGS
@given(case=full_multivector_cases(), data=st.data())
def test_functional_unary_helpers_match_small_oracle(case, data):
    signature, values = case
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(*signature)
    grade = data.draw(st.integers(min_value=0, max_value=algebra.n))
    grade_layout = algebra.layout((grade,))

    assert torch.allclose(reverse(algebra, values), oracle.reverse(values))
    assert torch.allclose(grade_involution(algebra, values), oracle.grade_involution(values))
    assert torch.allclose(clifford_conjugation(algebra, values), oracle.clifford_conjugation(values))
    assert torch.allclose(dual(algebra, values), oracle.dual(values))
    assert torch.allclose(norm_sq(algebra, values), oracle.norm_sq(values))
    assert torch.allclose(grade_projection(algebra, values, grade), grade_layout.compact(oracle.project(values, (grade,))))


def test_functional_product_rejects_unknown_op():
    algebra = AlgebraContext(3, 0, device="cpu", dtype=torch.float64)
    values = torch.zeros(1, algebra.dim, dtype=torch.float64)

    with pytest.raises(ValueError, match="Unsupported product op"):
        product(algebra, values, values, op="unknown")
