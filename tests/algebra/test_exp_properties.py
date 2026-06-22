# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from itertools import combinations

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from clifra.core.runtime.algebra import AlgebraContext
from tests.helpers.bivector_exp_oracle import bivector_exp_cpu_reference
from tests.helpers.hypothesis_cases import QUICK_PROPERTY_SETTINGS, signature_strategy, tensor_with_shape
from tests.helpers.small_oracle import SmallCliffordOracle

pytestmark = pytest.mark.unit


def _even_grade_sets(n: int) -> tuple[tuple[int, ...], ...]:
    grades = tuple(range(0, n + 1, 2))
    return tuple(
        tuple(selection)
        for size in range(1, len(grades) + 1)
        for selection in combinations(grades, size)
    )


@st.composite
def _bivector_exp_cases(draw, *, include_degenerate: bool = True):
    signature = draw(signature_strategy(min_n=2, max_n=4, include_degenerate=include_degenerate))
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    input_layout = algebra.layout((2,))
    output_grades = draw(st.sampled_from(_even_grade_sets(algebra.n)))
    output_layout = algebra.layout(output_grades)
    batch = draw(st.integers(min_value=1, max_value=2))
    values = draw(tensor_with_shape((batch, input_layout.dim)))
    return signature, input_layout, output_layout, values


@st.composite
def _euclidean_bivector_exp_cases(draw):
    n = draw(st.integers(min_value=2, max_value=4))
    algebra = AlgebraContext(n, 0, 0, device="cpu", dtype=torch.float64)
    input_layout = algebra.layout((2,))
    batch = draw(st.integers(min_value=1, max_value=2))
    values = draw(tensor_with_shape((batch, input_layout.dim)))
    return (n, 0, 0), input_layout, values


@QUICK_PROPERTY_SETTINGS
@given(case=_bivector_exp_cases())
def test_bivector_exp_matches_cpu_oracle_for_even_output_layouts(case):
    signature, input_layout, output_layout, values = case
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)

    actual = algebra.exp(values, input_layout=input_layout, output_layout=output_layout)
    expected = bivector_exp_cpu_reference(
        algebra,
        values,
        input_layout=input_layout,
        output_layout=output_layout,
    )

    assert torch.allclose(actual, expected, atol=1e-9, rtol=1e-9)


@QUICK_PROPERTY_SETTINGS
@given(case=_euclidean_bivector_exp_cases())
def test_euclidean_bivector_exp_is_unit_rotor_by_small_oracle(case):
    signature, input_layout, values = case
    p, q, r = signature
    algebra = AlgebraContext(p, q, r, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(p, q, r)
    even_layout = algebra.layout(range(0, algebra.n + 1, 2))

    rotor = algebra.exp(values, input_layout=input_layout, output_layout=even_layout)
    rotor_reverse = oracle.reverse(rotor, even_layout.basis_indices)
    product = oracle.product(
        rotor,
        rotor_reverse,
        left_indices=even_layout.basis_indices,
        right_indices=even_layout.basis_indices,
    )
    expected = torch.zeros_like(product)
    expected[..., 0] = 1.0

    assert torch.allclose(product, expected, atol=1e-9, rtol=1e-9)
