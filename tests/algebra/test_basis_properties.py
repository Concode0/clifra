# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from math import comb

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from clifra.core.foundation.basis import (
    basis_count_for_grades,
    basis_index_tuple_for_grades,
    basis_indices_for_grades,
    expand_output_grades,
    geometric_product_output_grades,
    operation_coefficient,
    product_output_grades,
)
from clifra.core.runtime.algebra import AlgebraContext
from tests.helpers.hypothesis_cases import PRODUCT_OPS, PROPERTY_SETTINGS, compact_multivector_cases, grade_sets

pytestmark = pytest.mark.unit


@PROPERTY_SETTINGS
@given(n=st.integers(min_value=0, max_value=12), data=st.data())
def test_basis_indices_for_grades_are_combinatorial(data, n):
    grades = data.draw(st.sampled_from(grade_sets(n)))

    indices = basis_index_tuple_for_grades(n, grades)

    assert indices == tuple(sorted(set(indices)))
    assert all(index.bit_count() in grades for index in indices)
    assert len(indices) == basis_count_for_grades(n, grades)
    assert len(indices) == sum(comb(n, grade) for grade in grades)


@PROPERTY_SETTINGS
@given(n=st.integers(min_value=1, max_value=6), op=st.sampled_from(PRODUCT_OPS), data=st.data())
def test_product_output_grades_match_basis_pair_support(data, n, op):
    left_grade = data.draw(st.integers(min_value=0, max_value=n))
    right_grade = data.draw(st.integers(min_value=0, max_value=n))
    left_indices = basis_index_tuple_for_grades(n, (left_grade,))
    right_indices = basis_index_tuple_for_grades(n, (right_grade,))

    supported = {
        (left_index ^ right_index).bit_count()
        for left_index in left_indices
        for right_index in right_indices
        if operation_coefficient(left_index, right_index, n, 0, 0, op) != 0.0
    }

    assert tuple(sorted(supported)) == product_output_grades(left_grade, right_grade, n, op=op)


@PROPERTY_SETTINGS
@given(n=st.integers(min_value=1, max_value=6), op=st.sampled_from(PRODUCT_OPS), data=st.data())
def test_expand_output_grades_is_union_of_homogeneous_routes(data, n, op):
    left_grades = data.draw(st.sampled_from(grade_sets(n)))
    right_grades = data.draw(st.sampled_from(grade_sets(n)))
    expected = tuple(
        sorted(
            {
                grade
                for left_grade in left_grades
                for right_grade in right_grades
                for grade in product_output_grades(left_grade, right_grade, n, op=op)
            }
        )
    )

    if not expected:
        with pytest.raises(ValueError, match="Grade expansion is empty"):
            expand_output_grades(left_grades, right_grades, n, op=op)
        return

    assert expand_output_grades(left_grades, right_grades, n, op=op) == expected

    projected = data.draw(st.sampled_from(grade_sets(n)))
    selected = tuple(grade for grade in expected if grade in projected)
    if selected:
        assert expand_output_grades(left_grades, right_grades, n, op=op, project_grades=projected) == selected
    else:
        with pytest.raises(ValueError, match="Grade expansion is empty"):
            expand_output_grades(left_grades, right_grades, n, op=op, project_grades=projected)


@PROPERTY_SETTINGS
@given(case=compact_multivector_cases(max_n=5))
def test_grade_layout_compact_full_round_trip(case):
    signature, grades, values = case
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    layout = algebra.layout(grades)

    materialized = layout.full(values)

    assert materialized.shape == (*values.shape[:-1], algebra.dim)
    assert torch.allclose(layout.compact(materialized), values)
    outside = torch.ones(algebra.dim, dtype=torch.bool)
    outside[layout.indices_tensor()] = False
    assert torch.count_nonzero(materialized[..., outside]) == 0


def test_geometric_product_grade_bounds_for_high_dimensional_routes():
    assert geometric_product_output_grades(1, 1, 16) == (0, 2)
    assert geometric_product_output_grades(2, 1, 16) == (1, 3)
    assert product_output_grades(2, 1, 16, op="wedge") == (3,)
    assert product_output_grades(2, 1, 16, op="commutator") == (1,)
    assert product_output_grades(2, 1, 16, op="anti_commutator") == (3,)
    assert expand_output_grades((0, 2), (1,), 16, op="gp") == (1, 3)
    assert expand_output_grades((1,), (1,), 16, op="wedge") == (2,)
    assert expand_output_grades((1,), (1,), 16, op="gp", project_grades=(0,)) == (0,)


def test_basis_tensorization_reports_int64_bitmask_boundary():
    with pytest.raises(ValueError, match="torch.long basis bitmasks"):
        basis_indices_for_grades(64, (1,))


def test_operation_coefficients_keep_wedge_as_exterior_product():
    assert operation_coefficient(3, 4, 3, 0, 0, "wedge") == 1.0
    assert operation_coefficient(3, 4, 3, 0, 0, "commutator") == 0.0
    assert operation_coefficient(3, 4, 3, 0, 0, "anti_commutator") == 2.0
