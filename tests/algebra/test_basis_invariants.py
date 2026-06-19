# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from tests.planning._grade_plan_helpers import (
    DEVICE,
    AlgebraContext,
    AlgebraSpec,
    DualExecutor,
    FullSandwichActionExecutor,
    FullSandwichActionHandle,
    FullTableProductExecutor,
    GradeFlow,
    GradePlanner,
    GradeProductExecutor,
    LaneStorage,
    MultiVersorActionHandle,
    NormSquaredExecutor,
    PairedBivectorActionHandle,
    PlanningLimits,
    ProductExecutionPolicy,
    ProductPlanHandle,
    SmallCliffordOracle,
    UnaryPlanHandle,
    VersorActionHandle,
    _grade_only_input,
    _mps_available,
    _oracle_for,
    _oracle_sandwich_action_matrices,
    _product_method_name,
    _sparse_pairwise_product_reference,
    apply_graded_linear_action,
    apply_multi_graded_linear_action,
    basis_count_for_grades,
    basis_index_tuple_for_grades,
    basis_indices_for_grades,
    build_grade_plan_tree,
    build_grade_product_plan,
    build_product_request,
    build_unary_request,
    estimate_product_executor_cost,
    expand_output_grades,
    geometric_product_output_grades,
    make_algebra,
    operation_coefficient,
    product_output_grades,
    pytest,
    torch,
)

pytestmark = pytest.mark.unit


def test_grade_expansion_for_common_high_dim_paths():
    assert geometric_product_output_grades(1, 1, 16) == (0, 2)
    assert geometric_product_output_grades(2, 1, 16) == (1, 3)
    assert product_output_grades(2, 1, 16, op="wedge") == (3,)
    assert product_output_grades(2, 1, 16, op="commutator") == (1,)
    assert product_output_grades(2, 1, 16, op="anti_commutator") == (3,)
    assert expand_output_grades((0, 2), (1,), 16, op="gp") == (1, 3)
    assert expand_output_grades((1,), (1,), 16, op="wedge") == (2,)
    assert expand_output_grades((1,), (1,), 16, op="gp", project_grades=(0,)) == (0,)


def test_operation_coefficients_keep_wedge_as_exterior_product():
    # e12 and e3 commute, so the antisymmetric formula would vanish.  The
    # exterior product is instead the grade-sum part of the geometric product.
    assert operation_coefficient(3, 4, 3, 0, 0, "wedge") == 1.0
    assert operation_coefficient(3, 4, 3, 0, 0, "commutator") == 0.0
    assert operation_coefficient(3, 4, 3, 0, 0, "anti_commutator") == 2.0


def test_basis_indices_for_grades_are_combinatorial_and_high_dimensional():
    assert basis_index_tuple_for_grades(4, (1, 2)) == tuple(
        index for index in range(1 << 4) if index.bit_count() in {1, 2}
    )
    high = basis_index_tuple_for_grades(32, (1, 2))
    assert len(high) == basis_count_for_grades(32, (1, 2))
    assert high[0] == 1
    assert high[-1] == (1 << 31) | (1 << 30)


def test_basis_tensorization_reports_int64_bitmask_boundary():
    with pytest.raises(ValueError, match="torch.long basis bitmasks"):
        basis_indices_for_grades(64, (1,))


def test_grade_layout_compact_full_round_trip():
    spec = AlgebraSpec(4, 1, 1)
    layout = spec.layout((0, 2))
    reference = torch.randn(2, spec.dim, dtype=torch.float64, generator=torch.Generator().manual_seed(97))

    values = layout.compact(reference)
    materialized = layout.full(values)

    assert values.shape[-1] == layout.dim
    assert torch.allclose(materialized[..., layout.indices_tensor(device=reference.device)], values)
    outside = torch.ones(spec.dim, dtype=torch.bool)
    outside[layout.indices_tensor()] = False
    assert materialized[..., outside].abs().sum().item() == 0.0


def test_wedge_full_and_planned_paths_are_exterior_product_for_higher_grades():
    algebra = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float64)
    oracle = _oracle_for(algebra)
    spec = AlgebraSpec.from_algebra(algebra)
    layout_2 = spec.layout((2,))
    layout_1 = spec.layout((1,))
    layout_3 = spec.layout((3,))

    e12 = torch.zeros(1, algebra.dim, dtype=torch.float64)
    e12[0, 3] = 1.0
    e3 = torch.zeros(1, algebra.dim, dtype=torch.float64)
    e3[0, 4] = 1.0
    expected = torch.zeros_like(e12)
    expected[0, 7] = 1.0

    full = algebra.wedge(e12, e3)
    compact = algebra.wedge(
        layout_2.compact(e12),
        layout_1.compact(e3),
        left_layout=layout_2,
        right_layout=layout_1,
        output_grades=(3,),
        left_storage=LaneStorage.COMPACT,
        right_storage=LaneStorage.COMPACT,
        output_storage=LaneStorage.COMPACT,
    )

    assert torch.allclose(full, oracle.product(e12, e3, op="wedge"), atol=1e-12, rtol=1e-12)
    assert torch.allclose(full, expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(compact, layout_3.compact(expected), atol=1e-12, rtol=1e-12)


def test_wedge_chains_as_iterative_exterior_product():
    algebra = AlgebraContext(4, 0, 0, device=DEVICE, dtype=torch.float64)
    layout_1 = algebra.layout((1,))
    layout_2 = algebra.layout((2,))
    layout_3 = algebra.layout((3,))
    e1 = torch.zeros(1, algebra.dim, dtype=torch.float64)
    e2 = torch.zeros(1, algebra.dim, dtype=torch.float64)
    e3 = torch.zeros(1, algebra.dim, dtype=torch.float64)
    e1[0, 1] = 1.0
    e2[0, 2] = 1.0
    e3[0, 4] = 1.0
    expected = torch.zeros_like(e1)
    expected[0, 7] = 1.0

    e12 = algebra.wedge(
        layout_1.compact(e1),
        layout_1.compact(e2),
        left_layout=layout_1,
        right_layout=layout_1,
        output_layout=layout_2,
    )
    actual = algebra.wedge(
        e12,
        layout_1.compact(e3),
        left_layout=layout_2,
        right_layout=layout_1,
        output_layout=layout_3,
    )

    assert torch.allclose(actual, layout_3.compact(expected), atol=1e-12, rtol=1e-12)
