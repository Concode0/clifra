# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from clifra.core.algebra import AlgebraContext

pytestmark = pytest.mark.unit


def test_projected_product_accepts_string_storage_and_materializes_canonical_output():
    algebra = AlgebraContext(3, 0, 0, device="cpu", dtype=torch.float64)
    vector_layout = algebra.layout((1,))
    bivector_layout = algebra.layout((2,))
    left = torch.randn(5, vector_layout.dim, dtype=torch.float64)
    right = torch.randn(5, vector_layout.dim, dtype=torch.float64)

    actual = algebra.wedge(
        left,
        right,
        left=vector_layout,
        right=vector_layout,
        output=TensorContract(bivector_layout, storage="canonical"),
    )
    expected = bivector_layout.full(
        algebra.wedge(left, right, left=vector_layout, right=vector_layout, output=bivector_layout)
    )

    assert actual.shape == (5, algebra.dim)
    assert torch.allclose(actual, expected)


def test_planned_unary_accepts_string_storage_for_canonical_output():
    algebra = AlgebraContext(4, 0, 0, device="cpu", dtype=torch.float64)
    vector_layout = algebra.layout((1,))
    values = torch.randn(3, vector_layout.dim, dtype=torch.float64)

    actual = algebra.reverse(values, input=vector_layout, output=TensorContract.canonical(vector_layout))
    expected = vector_layout.full(algebra.reverse(values, input=vector_layout))

    assert actual.shape == (3, algebra.dim)
    assert torch.allclose(actual, expected)


def test_pseudoscalar_product_accepts_output_storage_and_materializes_canonical_output():
    algebra = AlgebraContext(3, 0, 0, device="cpu", dtype=torch.float64)
    vector_layout = algebra.layout((1,))
    values = torch.randn(4, vector_layout.dim, dtype=torch.float64)

    output_layout = algebra.layout((algebra.n - 1,))
    compact = algebra.pseudoscalar_product(values, input=vector_layout, output=output_layout)
    actual_layout = output_layout
    actual = algebra.pseudoscalar_product(values, input=vector_layout, output=TensorContract.canonical(output_layout))

    assert actual_layout == output_layout
    assert actual.shape == (4, algebra.dim)
    assert torch.allclose(actual, output_layout.full(compact))


def test_bivector_exp_accepts_output_storage_and_materializes_canonical_output():
    algebra = AlgebraContext(3, 0, 0, device="cpu", dtype=torch.float64)
    bivector_layout = algebra.layout((2,))
    values = torch.randn(2, bivector_layout.dim, dtype=torch.float64) * 0.1

    compact, output_layout = (
        algebra.bivector_exp(values, input=bivector_layout, output=algebra.layout()),
        algebra.layout(),
    )
    actual, actual_layout = (
        algebra.bivector_exp(
            values, input=bivector_layout, output=TensorContract(algebra.layout(), storage="canonical")
        ),
        algebra.layout(),
    )

    assert actual_layout == output_layout
    assert actual.shape == (2, algebra.dim)
    assert torch.allclose(actual, output_layout.full(compact), atol=1e-12, rtol=1e-12)


from clifra.core.tensors import TensorContract
