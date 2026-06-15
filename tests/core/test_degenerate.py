# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Layout-first tests for degenerate signatures Cl(p, q, r)."""

import pytest
import torch

from clifra.core.runtime.algebra import AlgebraContext

pytestmark = pytest.mark.unit

DEVICE = "cpu"


def test_null_vector_squares_to_zero():
    algebra = AlgebraContext(2, 0, 1, device=DEVICE, dtype=torch.float64)
    vector_layout = algebra.layout((1,))
    scalar_layout = algebra.layout((0,))
    e3 = torch.zeros(1, vector_layout.dim, dtype=torch.float64)
    e3[0, vector_layout.basis_indices.index(4)] = 1.0

    actual = algebra.geometric_product(
        e3,
        e3,
        left_layout=vector_layout,
        right_layout=vector_layout,
        output_layout=scalar_layout,
    )

    assert torch.allclose(actual, torch.zeros_like(actual), atol=1e-12, rtol=1e-12)


def test_positive_vectors_still_square_to_one_with_null_dimension():
    algebra = AlgebraContext(2, 0, 1, device=DEVICE, dtype=torch.float64)
    vector_layout = algebra.layout((1,))
    scalar_layout = algebra.layout((0,))
    e1 = torch.zeros(1, vector_layout.dim, dtype=torch.float64)
    e1[0, vector_layout.basis_indices.index(1)] = 1.0

    actual = algebra.geometric_product(
        e1,
        e1,
        left_layout=vector_layout,
        right_layout=vector_layout,
        output_layout=scalar_layout,
    )

    assert torch.allclose(actual, torch.ones_like(actual), atol=1e-12, rtol=1e-12)


def test_bivector_squared_signs_include_null_components():
    algebra = AlgebraContext(2, 0, 1, device=DEVICE, dtype=torch.float64)
    bivector_layout = algebra.layout((2,))
    signs = algebra.bivector_squared_signs(dtype=torch.float64)
    by_index = dict(zip(bivector_layout.basis_indices, signs.tolist()))

    assert by_index[3] == pytest.approx(-1.0)
    assert by_index[5] == pytest.approx(0.0)
    assert by_index[6] == pytest.approx(0.0)


def test_bivector_exp_null_parabolic_branch():
    algebra = AlgebraContext(2, 0, 1, device=DEVICE, dtype=torch.float64)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout((0, 2))
    values = torch.zeros(1, bivector_layout.dim, dtype=torch.float64)
    values[0, bivector_layout.basis_indices.index(5)] = 0.3

    actual = algebra.exp(values, input_layout=bivector_layout, output_layout=even_layout)
    expected = torch.zeros_like(actual)
    expected[0, even_layout.basis_indices.index(0)] = 1.0
    expected[0, even_layout.basis_indices.index(5)] = 0.3

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_wedge_with_null_vectors_is_nonzero():
    algebra = AlgebraContext(2, 0, 1, device=DEVICE, dtype=torch.float64)
    vector_layout = algebra.layout((1,))
    bivector_layout = algebra.layout((2,))
    e1 = torch.zeros(1, vector_layout.dim, dtype=torch.float64)
    e3 = torch.zeros(1, vector_layout.dim, dtype=torch.float64)
    e1[0, vector_layout.basis_indices.index(1)] = 1.0
    e3[0, vector_layout.basis_indices.index(4)] = 1.0

    actual = algebra.wedge(
        e1,
        e3,
        left_layout=vector_layout,
        right_layout=vector_layout,
        output_layout=bivector_layout,
    )

    assert actual[0, bivector_layout.basis_indices.index(5)] == pytest.approx(1.0)


def test_signature_validation_rejects_negative_counts():
    with pytest.raises(ValueError, match="non-negative"):
        AlgebraContext(2, 0, -1, device=DEVICE)


def test_n4_bivector_exp_full_and_even_outputs_are_consistent():
    algebra = AlgebraContext(3, 0, 1, device=DEVICE, dtype=torch.float64)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout((0, 2, 4))
    full_layout = algebra.layout()
    values = torch.zeros(1, bivector_layout.dim, dtype=torch.float64)
    values[0, 0] = 0.5

    even = algebra.exp(values, input_layout=bivector_layout, output_layout=even_layout)
    full = algebra.exp(bivector_layout.full(values), output_layout=full_layout)

    assert torch.isfinite(even).all()
    assert torch.isfinite(full).all()
    assert torch.allclose(even, even_layout.compact(full), atol=1e-12, rtol=1e-12)
