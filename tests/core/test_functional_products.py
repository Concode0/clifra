# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from clifra.functional import (
    anti_commutator,
    commutator,
    geometric_product,
    grade_projection,
    inner_product,
    product,
    reverse,
    wedge,
)

pytestmark = pytest.mark.unit


def test_functional_products_match_algebra_full_lanes(algebra_3d):
    left = torch.randn(3, algebra_3d.dim)
    right = torch.randn(3, algebra_3d.dim)

    assert torch.allclose(geometric_product(algebra_3d, left, right), algebra_3d.geometric_product(left, right))
    assert torch.allclose(wedge(algebra_3d, left, right), algebra_3d.wedge(left, right))
    assert torch.allclose(inner_product(algebra_3d, left, right), algebra_3d.inner_product(left, right))
    assert torch.allclose(commutator(algebra_3d, left, right), algebra_3d.commutator(left, right))
    assert torch.allclose(anti_commutator(algebra_3d, left, right), algebra_3d.anti_commutator(left, right))


def test_functional_projected_product_active_output(algebra_3d):
    left = algebra_3d.embed_vector(torch.randn(4, algebra_3d.n))
    right = algebra_3d.embed_vector(torch.randn(4, algebra_3d.n))
    layout = algebra_3d.layout((2,))

    actual = wedge(
        algebra_3d,
        left,
        right,
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(2,),
        active_output=True,
    )
    expected = layout.compact(algebra_3d.wedge(left, right))

    assert actual.shape == (4, layout.dim)
    assert torch.allclose(actual, expected)


def test_functional_unary_helpers(algebra_3d):
    values = torch.randn(2, algebra_3d.dim)

    assert torch.allclose(reverse(algebra_3d, values), algebra_3d.reverse(values))
    assert torch.allclose(grade_projection(algebra_3d, values, 1), algebra_3d.grade_projection(values, 1))


def test_functional_product_rejects_unknown_op(algebra_3d):
    values = torch.randn(2, algebra_3d.dim)

    with pytest.raises(ValueError, match="Unsupported product op"):
        product(algebra_3d, values, values, op="unknown")
