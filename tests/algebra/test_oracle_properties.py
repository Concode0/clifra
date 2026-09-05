# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from clifra.core._kernel.basis import (
    basis_index_tuple_for_grades,
    basis_product,
    operation_coefficient,
    reverse_sign,
)
from clifra.core.algebra import AlgebraContext
from tests.helpers.hypothesis_cases import (
    CORE_PROPERTY_SETTINGS,
    DEEP_PROPERTY_SETTINGS,
    PRODUCT_OPS,
    SIGNATURE_SWEEP_SETTINGS,
    blade_index_strategy,
    signature_for_null_count,
    signature_strategy,
    small_signatures,
    tensor_with_shape,
)
from tests.helpers.small_oracle import SmallCliffordOracle

pytestmark = pytest.mark.property


@pytest.mark.unit
@CORE_PROPERTY_SETTINGS
@given(signature=signature_strategy(max_n=63), data=st.data())
def test_oracle_satisfies_the_defining_vector_relations(signature, data):
    oracle = SmallCliffordOracle(*signature)
    axis = data.draw(st.integers(min_value=0, max_value=oracle.n - 1))
    expected_square = 1.0 if axis < oracle.p else -1.0 if axis < oracle.p + oracle.q else 0.0
    assert oracle.basis_product(1 << axis, 1 << axis) == (0, expected_square)

    if oracle.n > 1:
        other = data.draw(st.sampled_from(tuple(candidate for candidate in range(oracle.n) if candidate != axis)))
        forward = oracle.basis_product(1 << axis, 1 << other)
        backward = oracle.basis_product(1 << other, 1 << axis)
        assert forward == (backward[0], -backward[1])


@pytest.mark.unit
@pytest.mark.parametrize("signature", small_signatures(max_n=6))
def test_independent_oracle_covers_every_small_signature_grade_and_blade(signature):
    oracle = SmallCliffordOracle(*signature)
    for grade in range(oracle.n + 1):
        assert oracle.indices_for_grades((grade,)) == basis_index_tuple_for_grades(oracle.n, (grade,))

    for left_index in oracle.full_indices:
        assert oracle.reverse_sign(left_index) == reverse_sign(left_index)
        for right_index in oracle.full_indices:
            assert oracle.basis_product(left_index, right_index) == basis_product(left_index, right_index, *signature)
            for op in PRODUCT_OPS:
                assert oracle.operation_coefficient(left_index, right_index, op) == operation_coefficient(
                    left_index,
                    right_index,
                    *signature,
                    op,
                )


@pytest.mark.unit
@pytest.mark.parametrize("signature", small_signatures(max_n=6))
@SIGNATURE_SWEEP_SETTINGS
@given(data=st.data())
def test_hypothesis_exercises_every_small_signature_and_homogeneous_grade(signature, data):
    oracle = SmallCliffordOracle(*signature)
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    for grade in range(oracle.n + 1):
        layout = algebra.layout((grade,))
        values = data.draw(tensor_with_shape((1, layout.dim)), label=f"grade {grade} values")
        assert torch.allclose(
            algebra.reverse(values, input=layout, output=layout),
            oracle.reverse(values, layout.basis_indices),
            atol=1e-12,
            rtol=1e-12,
        )
        assert torch.allclose(
            algebra.signature_norm_squared(values, input=layout),
            oracle.signature_norm_squared(values, layout.basis_indices),
            atol=1e-12,
            rtol=1e-12,
        )


@pytest.mark.parametrize("r", range(64))
@DEEP_PROPERTY_SETTINGS
@given(data=st.data())
def test_high_dimensional_blades_cover_every_null_count(r, data):
    signature = data.draw(signature_for_null_count(r), label="signature")
    oracle = SmallCliffordOracle(*signature)
    left_index = data.draw(blade_index_strategy(oracle.n), label="left blade")
    right_index = data.draw(blade_index_strategy(oracle.n), label="right blade")
    if r and data.draw(st.booleans(), label="force null overlap"):
        null_axis = data.draw(st.integers(min_value=oracle.p + oracle.q, max_value=oracle.n - 1))
        left_index |= 1 << null_axis
        right_index |= 1 << null_axis

    assert oracle.basis_product(left_index, right_index) == basis_product(left_index, right_index, *signature)
    assert oracle.reverse_sign(left_index) == reverse_sign(left_index)
    for op in PRODUCT_OPS:
        assert oracle.operation_coefficient(left_index, right_index, op) == operation_coefficient(
            left_index,
            right_index,
            *signature,
            op,
        )
