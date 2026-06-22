# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from clifra.core.runtime.algebra import AlgebraContext
from clifra.core.runtime.energy import lane_distance, lane_energy, lane_inner_product, lane_norm
from clifra.core.runtime.forms import (
    conjugate_form_distance_like,
    conjugate_form_magnitude,
    conjugate_scalar_form,
    conjugate_scalar_form_signs,
    signature_norm_squared,
    signature_trace_form,
)
from clifra.core.runtime.metric import scalar_product, signature_magnitude
from tests.helpers.hypothesis_cases import (
    PROPERTY_SETTINGS,
    compact_multivector_cases,
    signature_strategy,
    tensor_with_shape,
)
from tests.helpers.small_oracle import SmallCliffordOracle

pytestmark = pytest.mark.unit


@PROPERTY_SETTINGS
@given(signature=signature_strategy(max_n=4), data=st.data())
def test_full_lane_signed_scalar_forms_match_small_oracle(signature, data):
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(*signature)
    batch = data.draw(st.integers(min_value=1, max_value=3))
    left = data.draw(tensor_with_shape((batch, algebra.dim)))
    right = data.draw(tensor_with_shape((batch, algebra.dim)))

    assert torch.allclose(
        scalar_product(algebra, left, right),
        oracle.scalar_product(left, right),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        conjugate_scalar_form(algebra, left, right),
        oracle.conjugate_scalar_form(left, right),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        signature_trace_form(algebra, left, right),
        oracle.signature_trace_form(left, right),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        signature_norm_squared(algebra, left),
        oracle.signature_trace_form(left, left),
        atol=1e-12,
        rtol=1e-12,
    )


@PROPERTY_SETTINGS
@given(case=compact_multivector_cases(), data=st.data())
def test_compact_signed_scalar_forms_match_small_oracle(case, data):
    signature, grades, left = case
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(*signature)
    layout = algebra.layout(grades)
    right = data.draw(tensor_with_shape(tuple(left.shape)))

    assert torch.allclose(
        scalar_product(algebra, left, right, left_layout=layout, right_layout=layout),
        oracle.scalar_product(left, right, left_indices=layout.basis_indices, right_indices=layout.basis_indices),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        conjugate_scalar_form(algebra, left, right, layout=layout),
        oracle.conjugate_scalar_form(left, right, layout.basis_indices),
        atol=1e-12,
        rtol=1e-12,
    )


@PROPERTY_SETTINGS
@given(signature=signature_strategy(max_n=4), data=st.data())
def test_lane_metrics_are_positive_coefficient_geometry(signature, data):
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    batch = data.draw(st.integers(min_value=1, max_value=3))
    left = data.draw(tensor_with_shape((batch, algebra.dim)))
    middle = data.draw(tensor_with_shape((batch, algebra.dim)))
    right = data.draw(tensor_with_shape((batch, algebra.dim)))

    assert torch.allclose(lane_energy(algebra, left), (left * left).sum(dim=-1, keepdim=True))
    assert torch.allclose(lane_norm(algebra, left), torch.linalg.vector_norm(left, dim=-1, keepdim=True))
    assert torch.allclose(lane_inner_product(algebra, left, right), (left * right).sum(dim=-1, keepdim=True))
    assert torch.all(lane_norm(algebra, left) >= 0)
    assert torch.allclose(lane_distance(algebra, left, left), torch.zeros(batch, 1, dtype=torch.float64))
    assert torch.allclose(lane_distance(algebra, left, right), lane_distance(algebra, right, left))
    assert torch.all(
        lane_distance(algebra, left, right)
        <= lane_distance(algebra, left, middle) + lane_distance(algebra, middle, right) + 1e-12
    )


def test_null_blade_has_positive_lane_energy_but_zero_signed_forms():
    algebra = AlgebraContext(1, 0, 1, device="cpu", dtype=torch.float64)
    values = torch.zeros(algebra.dim, dtype=torch.float64)
    null_index = 1 << algebra.p
    values[null_index] = 1.0

    assert torch.allclose(lane_energy(algebra, values), torch.tensor([1.0], dtype=torch.float64))
    assert torch.allclose(lane_norm(algebra, values), torch.tensor([1.0], dtype=torch.float64))
    assert torch.allclose(conjugate_scalar_form(algebra, values, values), torch.zeros(1, dtype=torch.float64))
    assert torch.allclose(signature_norm_squared(algebra, values), torch.zeros(1, dtype=torch.float64))


def test_conjugate_scalar_form_signs_preserve_null_degeneracy_in_layouts():
    algebra = AlgebraContext(2, 0, 1, device="cpu", dtype=torch.float64)
    vector_layout = algebra.layout((1,))
    signs = conjugate_scalar_form_signs(algebra, layout=vector_layout)
    null_basis_index = 1 << algebra.p
    null_position = vector_layout.basis_indices.index(null_basis_index)

    assert signs.shape == (vector_layout.dim,)
    assert signs[null_position] == 0.0
    assert (signs != 0).any()


def test_signed_magnitudes_are_absolute_forms_not_lane_norms():
    algebra = AlgebraContext(0, 1, 0, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(0, 1, 0)
    values = torch.zeros(1, algebra.dim, dtype=torch.float64)
    values[0, 1] = 2.0

    expected_norm_sq = oracle.signature_trace_form(values, values)

    assert torch.allclose(signature_norm_squared(algebra, values), expected_norm_sq)
    assert torch.allclose(signature_magnitude(algebra, values), torch.sqrt(expected_norm_sq.abs()))
    assert torch.allclose(conjugate_form_magnitude(algebra, values), torch.sqrt(expected_norm_sq.abs()))
    assert torch.allclose(conjugate_form_distance_like(algebra, values, values), torch.zeros(1, 1, dtype=torch.float64))
