# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from tests.planning._grade_plan_helpers import (
    DEVICE,
    AlgebraContext,
    AlgebraSpec,
    FullSandwichActionExecutor,
    FullSandwichActionHandle,
    FullTableProductExecutor,
    GradeFlow,
    GradePlanner,
    GradeProductExecutor,
    LaneStorage,
    MultiVersorActionHandle,
    PairedBivectorActionHandle,
    PlanningLimits,
    ProductExecutionPolicy,
    ProductPlanHandle,
    PseudoscalarProductExecutor,
    SignatureNormSquaredExecutor,
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


def test_context_static_product_cost_limits_raise_before_executor_build():
    limits = PlanningLimits(warn_lanes=512, max_lanes=512, warn_pairs=512, max_pairs=64)
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32, planning_limits=limits)
    layout = algebra.layout((1,))
    left = torch.zeros(1, layout.dim)
    right = torch.zeros(1, layout.dim)

    with pytest.raises(ValueError, match="basis interactions"):
        algebra.projected_geometric_product(
            left,
            right,
            left_layout=layout,
            right_layout=layout,
            output_storage=LaneStorage.COMPACT,
        )


def test_context_static_layout_cost_limit_raises_before_basis_materialization():
    limits = PlanningLimits(warn_lanes=32, max_lanes=64, warn_pairs=512, max_pairs=1024)
    algebra = make_algebra(32, 0, 0, device=DEVICE, dtype=torch.float32, planning_limits=limits)

    with pytest.raises(ValueError, match="compact lanes"):
        algebra.layout((1, 2))


def test_high_dimensional_vector_product_plan_avoids_full_basis_enumeration():
    algebra = make_algebra(
        32,
        0,
        0,
        device=DEVICE,
        dtype=torch.float32,
        planning_limits=PlanningLimits(max_lanes=4096, max_pairs=100_000),
    )
    vector_layout = algebra.layout((1,))
    executor = algebra.planner.product_executor(
        op="gp",
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
        dtype=torch.float32,
        device=DEVICE,
    )

    assert vector_layout.dim == 32
    assert executor.output_dim == 1 + 32 * 31 // 2
    assert executor.pair_count == 32 * 32


def test_high_dimensional_vector_product_plan_avoids_dense_lookup_at_int64_limit():
    algebra = make_algebra(
        63,
        0,
        0,
        device=DEVICE,
        dtype=torch.float32,
        planning_limits=PlanningLimits(max_lanes=4096, max_pairs=100_000),
    )
    executor = algebra.planner.product_executor(
        op="gp",
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
        dtype=torch.float32,
        device=DEVICE,
    )

    assert executor.output_dim == 1 + 63 * 62 // 2
    assert executor.pair_count == 63 * 63


def test_high_dimensional_product_plan_reports_int64_bitmask_boundary():
    algebra = make_algebra(
        64,
        0,
        0,
        device=DEVICE,
        dtype=torch.float32,
        planning_limits=PlanningLimits(max_lanes=4096, max_pairs=100_000),
    )

    with pytest.raises(ValueError, match="Current Torch-backed executors support bitmask tensorization up to n=63"):
        algebra.planner.product_executor(
            op="gp",
            left_grades=(1,),
            right_grades=(1,),
            output_grades=(0, 2),
            dtype=torch.float32,
            device=DEVICE,
        )


def test_context_static_product_cost_warns_near_configured_limits():
    limits = PlanningLimits(warn_lanes=512, max_lanes=512, warn_pairs=128, max_pairs=512)
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32, planning_limits=limits)
    layout = algebra.layout((1,))
    left = torch.zeros(1, layout.dim)
    right = torch.zeros(1, layout.dim)

    with pytest.warns(RuntimeWarning, match="basis interactions"):
        values = algebra.projected_geometric_product(
            left,
            right,
            left_layout=layout,
            right_layout=layout,
            output_storage=LaneStorage.COMPACT,
        )

    assert values.shape[-1] == algebra.layout((0, 2)).dim
