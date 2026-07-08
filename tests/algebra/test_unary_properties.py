# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from hypothesis import assume, given
from hypothesis import strategies as st

from clifra.core.runtime.algebra import AlgebraContext
from tests.helpers.hypothesis_cases import (
    PROPERTY_SETTINGS,
    compact_multivector_cases,
    full_multivector_cases,
    signature_strategy,
    tensor_with_shape,
)
from tests.helpers.small_oracle import SmallCliffordOracle

pytestmark = pytest.mark.unit

UNARY_ORACLES = {
    "reverse": SmallCliffordOracle.reverse,
    "grade_involution": SmallCliffordOracle.grade_involution,
    "clifford_conjugation": SmallCliffordOracle.clifford_conjugation,
}


@PROPERTY_SETTINGS
@given(case=full_multivector_cases())
def test_full_lane_unary_involutions_match_small_oracle(case):
    signature, values = case
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(*signature)

    for op, oracle_method in UNARY_ORACLES.items():
        actual = getattr(algebra, op)(values)
        expected = oracle_method(oracle, values)

        assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
        assert torch.allclose(getattr(algebra, op)(actual), values, atol=1e-12, rtol=1e-12)


@PROPERTY_SETTINGS
@given(case=compact_multivector_cases())
def test_compact_unary_involutions_match_small_oracle(case):
    signature, grades, values = case
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(*signature)
    layout = algebra.layout(grades)

    for op, oracle_method in UNARY_ORACLES.items():
        actual = getattr(algebra, op)(values, input_layout=layout, output_layout=layout)
        expected = oracle_method(oracle, values, layout.basis_indices)

        assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


@PROPERTY_SETTINGS
@given(case=full_multivector_cases(max_n=5), data=st.data())
def test_grade_projection_matches_small_oracle(case, data):
    signature, values = case
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(*signature)
    grade = data.draw(st.integers(min_value=0, max_value=algebra.n))
    output_layout = algebra.layout((grade,))

    actual = algebra.grade_projection(values, grade, output_layout=output_layout)
    expected = output_layout.compact(oracle.project(values, (grade,)))

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


@PROPERTY_SETTINGS
@given(case=compact_multivector_cases())
def test_signature_norm_squared_matches_small_oracle_for_declared_layouts(case):
    signature, grades, values = case
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(*signature)
    layout = algebra.layout(grades)

    actual = algebra.signature_norm_squared(values, input_layout=layout)
    expected = oracle.signature_norm_squared(values, layout.basis_indices)

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


@PROPERTY_SETTINGS
@given(case=compact_multivector_cases())
def test_pseudoscalar_product_matches_small_oracle_for_declared_layouts(case):
    signature, grades, values = case
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(*signature)
    input_layout = algebra.layout(grades)

    actual, output_layout = algebra.pseudoscalar_product(values, input_layout=input_layout, return_layout=True)
    expected = oracle.pseudoscalar_product(
        values,
        input_indices=input_layout.basis_indices,
        output_indices=output_layout.basis_indices,
    )

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


@PROPERTY_SETTINGS
@given(case=compact_multivector_cases())
def test_blade_inverse_matches_small_oracle_for_declared_layouts(case):
    signature, grades, values = case
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(*signature)
    layout = algebra.layout(grades)

    actual = algebra.blade_inverse(values, input_layout=layout)
    expected = oracle.blade_inverse(values, layout.basis_indices)

    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


@PROPERTY_SETTINGS
@given(signature=signature_strategy(min_n=1, max_n=5), data=st.data())
def test_vector_reflection_matches_small_oracle_sandwich(signature, data):
    p, q, _ = signature
    assume(p + q > 0)
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(*signature)
    vector_layout = algebra.layout((1,))
    batch = data.draw(st.integers(min_value=1, max_value=3))
    bit = data.draw(st.integers(min_value=0, max_value=p + q - 1))
    values = data.draw(tensor_with_shape((batch, vector_layout.dim)))
    normal = torch.zeros(batch, vector_layout.dim, dtype=torch.float64)
    normal[:, vector_layout.basis_indices.index(1 << bit)] = 1.0

    actual = algebra.reflect(
        values,
        normal,
        input_layout=vector_layout,
        normal_layout=vector_layout,
        output_layout=vector_layout,
    )
    normal_hat = oracle.grade_involution(normal, vector_layout.basis_indices)
    normal_inv = oracle.blade_inverse(normal, vector_layout.basis_indices)
    middle = oracle.product(
        normal_hat,
        values,
        left_indices=vector_layout.basis_indices,
        right_indices=vector_layout.basis_indices,
    )
    expected = oracle.product(
        middle,
        normal_inv,
        right_indices=vector_layout.basis_indices,
        output_indices=vector_layout.basis_indices,
    )

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
