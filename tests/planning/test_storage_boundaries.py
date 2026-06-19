# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from clifra.core.runtime.algebra import AlgebraContext
from clifra.core.runtime.tensors import LaneStorage

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
        left_layout=vector_layout,
        right_layout=vector_layout,
        output_layout=bivector_layout,
        left_storage="compact",
        right_storage="compact",
        output_storage="canonical",
    )
    expected = bivector_layout.full(
        algebra.wedge(
            left,
            right,
            left_layout=vector_layout,
            right_layout=vector_layout,
            output_layout=bivector_layout,
            left_storage=LaneStorage.COMPACT,
            right_storage=LaneStorage.COMPACT,
            output_storage=LaneStorage.COMPACT,
        )
    )

    assert actual.shape == (5, algebra.dim)
    assert torch.allclose(actual, expected)


def test_planned_unary_accepts_string_storage_for_canonical_output():
    algebra = AlgebraContext(4, 0, 0, device="cpu", dtype=torch.float64)
    vector_layout = algebra.layout((1,))
    values = torch.randn(3, vector_layout.dim, dtype=torch.float64)

    actual = algebra.reverse(
        values,
        input_layout=vector_layout,
        input_storage="compact",
        output_storage="canonical",
    )
    expected = vector_layout.full(algebra.reverse(values, input_layout=vector_layout, input_storage="compact"))

    assert actual.shape == (3, algebra.dim)
    assert torch.allclose(actual, expected)
