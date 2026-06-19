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


def test_grade_plan_tree_groups_routes_without_runtime_partition_backend():
    spec = AlgebraSpec(10, 4, 2)
    tree = build_grade_plan_tree(
        spec,
        left_grades=(1, 2),
        right_grades=(1,),
        output_grades=(0, 2),
        op="gp",
        chunk_pair_limit=128,
    )

    assert tree.output_grades == (0, 2)
    assert [(path.left_grade, path.right_grade, path.output_grades) for path in tree.paths] == [
        (1, 1, (0, 2)),
    ]
    assert tree.path_count == 1
    assert tree.estimated_pairs == 16 * 16
    assert tree.estimated_chunks == 2
    assert tree.path_for_grades(1, 1) is tree.paths[0]
    assert tree.path_for_grades(2, 1) is None


def test_product_request_infers_declared_layouts_and_output_grades():
    spec = AlgebraSpec(10, 4, 2)
    layout = spec.layout((1,))
    left = torch.zeros(2, layout.dim)
    right = torch.zeros(2, layout.dim)

    request = build_product_request(
        spec,
        left,
        right,
        left_grades=(1,),
        right_grades=(1,),
        op="gp",
    )

    assert request.left_grades == (1,)
    assert request.right_grades == (1,)
    assert request.output_grades == (0, 2)
    assert request.left_uses_compact_storage
    assert request.right_uses_compact_storage


def test_product_request_detects_compact_lane_tensors_from_layout_shape():
    spec = AlgebraSpec(6, 0, 0)
    layout = spec.layout((1,))
    left = torch.zeros(2, layout.dim)
    right = torch.zeros(2, layout.dim)

    request = build_product_request(
        spec,
        left,
        right,
        left_layout=layout,
        right_layout=layout,
        output_grades=(0, 2),
        op="gp",
    )

    assert request.left_uses_compact_storage
    assert request.right_uses_compact_storage


def test_unary_request_infers_projection_layout_without_full_layout():
    spec = AlgebraSpec(10, 4, 2)
    layout = spec.layout((1,))
    values = torch.zeros(2, layout.dim)

    request = build_unary_request(
        spec,
        values,
        op="grade_projection",
        output_grades=(1,),
    )

    assert request.input_grades == (1,)
    assert request.output_grades == (1,)
    assert request.input_uses_compact_storage


def test_grade_flow_propagates_embedding_unary_product_and_merge():
    spec = AlgebraSpec(8, 0, 0)
    vector = GradeFlow.vector(spec)
    scalar = GradeFlow.scalar(spec)

    product = vector.product(vector)
    projected = product.project((2,))
    merged = scalar.merge(projected)

    assert vector.grades == (1,)
    assert vector.unary("reverse").grades == (1,)
    assert product.grades == (0, 2)
    assert projected.grades == (2,)
    assert merged.grades == (0, 2)
