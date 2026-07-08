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


@pytest.mark.parametrize(
    "op",
    [
        "gp",
        "wedge",
        "symmetric_product",
        "commutator_product",
        "anti_commutator_product",
        "left_contraction",
        "right_contraction",
    ],
)
def test_static_grade_product_matches_small_oracle_for_selected_grade_paths(op):
    algebra = SmallCliffordOracle(4, 1, 1)
    left_grades = (1,)
    right_grades = (1, 2)
    output_grades = expand_output_grades(left_grades, right_grades, algebra.n, op=op)
    plan = build_grade_product_plan(
        algebra.p,
        algebra.q,
        algebra.r,
        left_grades=left_grades,
        right_grades=right_grades,
        output_grades=output_grades,
        op=op,
        device=DEVICE,
        dtype=torch.float64,
    )
    product = GradeProductExecutor(plan)
    A = _grade_only_input(algebra, 3, left_grades, seed=101)
    B = _grade_only_input(algebra, 3, right_grades, seed=103)

    expected = algebra.project(algebra.product(A, B, op=op), output_grades)
    actual = product.forward_full(A, B)

    assert product.pair_count < algebra.dim * algebra.dim
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_wedge_plan_prunes_grade_route_pairs_before_coefficients():
    algebra = SmallCliffordOracle(6, 0, 0)
    broad = build_grade_product_plan(
        algebra.p,
        algebra.q,
        algebra.r,
        left_grades=(2,),
        right_grades=(1,),
        output_grades=(1, 3),
        op="wedge",
        device=DEVICE,
        dtype=torch.float64,
    )
    exterior_only = build_grade_product_plan(
        algebra.p,
        algebra.q,
        algebra.r,
        left_grades=(2,),
        right_grades=(1,),
        output_grades=(3,),
        op="wedge",
        device=DEVICE,
        dtype=torch.float64,
    )

    assert broad.pair_count == exterior_only.pair_count
    assert all(int(index).bit_count() == 3 for index in broad.output_indices.tolist())


def test_product_plan_owns_compact_lane_position_buffers():
    algebra = SmallCliffordOracle(4, 1, 0)
    plan = build_grade_product_plan(
        algebra.p,
        algebra.q,
        algebra.r,
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
        op="gp",
        device=DEVICE,
        dtype=torch.float64,
    )
    product = GradeProductExecutor(plan)
    A = _grade_only_input(algebra, 2, (1,), seed=109)
    B = _grade_only_input(algebra, 2, (1,), seed=111)

    left_positions = {index: position for position, index in enumerate(plan.left_layout.basis_indices)}
    right_positions = {index: position for position, index in enumerate(plan.right_layout.basis_indices)}
    expected_left = torch.tensor([left_positions[int(index)] for index in plan.left_indices], dtype=torch.long)
    expected_right = torch.tensor([right_positions[int(index)] for index in plan.right_indices], dtype=torch.long)

    assert torch.equal(plan.left_compact_positions.cpu(), expected_left)
    assert torch.equal(plan.right_compact_positions.cpu(), expected_right)
    assert torch.allclose(
        product.forward_compact(plan.left_layout.compact(A), plan.right_layout.compact(B)),
        product(A, B),
        atol=1e-12,
        rtol=1e-12,
    )


def test_product_executor_compact_forward_supports_different_layout_widths():
    algebra = SmallCliffordOracle(4, 1, 0)
    plan = build_grade_product_plan(
        algebra.p,
        algebra.q,
        algebra.r,
        left_grades=(1,),
        right_grades=(1, 2),
        output_grades=(0, 1, 2, 3),
        op="gp",
        device=DEVICE,
        dtype=torch.float64,
    )
    product = GradeProductExecutor(plan)
    A = _grade_only_input(algebra, 2, (1,), seed=115)
    B = _grade_only_input(algebra, 2, (1, 2), seed=117)

    compact = product.forward_compact(plan.left_layout.compact(A), plan.right_layout.compact(B))
    reference = product(A, B)

    assert plan.left_layout.dim != plan.right_layout.dim
    assert torch.allclose(compact, reference, atol=1e-12, rtol=1e-12)


def test_product_executor_pairwise_uses_factorized_smaller_lane_contraction():
    algebra = SmallCliffordOracle(6, 0, 0)
    plan = build_grade_product_plan(
        algebra.p,
        algebra.q,
        algebra.r,
        left_grades=(2,),
        right_grades=(1,),
        output_grades=(3,),
        op="wedge",
        device=DEVICE,
        dtype=torch.float64,
    )
    product = GradeProductExecutor(plan)
    generator = torch.Generator(device=DEVICE).manual_seed(119)
    left = torch.randn(2, 3, plan.left_layout.dim, dtype=torch.float64, generator=generator)
    right = torch.randn(2, 4, plan.right_layout.dim, dtype=torch.float64, generator=generator)

    actual = product.forward_pairwise_compact(left, right)
    expected = _sparse_pairwise_product_reference(product, left, right)

    assert not plan.pairwise_contract_left
    assert product.pairwise_gather_positions.shape == (plan.right_layout.dim, plan.output_dim)
    assert product.pairwise_coefficients.shape == product.pairwise_gather_positions.shape
    assert product.pairwise_gather_positions.numel() < plan.left_layout.dim * plan.output_dim
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("op", ["gp", "wedge", "symmetric_product", "commutator_product", "anti_commutator_product"])
def test_planner_full_table_executor_matches_small_oracle_full_layout_product(op):
    context = AlgebraContext(4, 1, 0, device=DEVICE, dtype=torch.float64)
    oracle = _oracle_for(context)
    full_layout = context.layout()
    generator = torch.Generator(device=DEVICE).manual_seed(199)
    left = torch.randn(3, context.dim, dtype=torch.float64, generator=generator)
    right = torch.randn(3, context.dim, dtype=torch.float64, generator=generator)

    executor = context.planner.product_executor_for_layouts(
        op=op,
        left_layout=full_layout,
        right_layout=full_layout,
        output_layout=full_layout,
        dtype=torch.float64,
        device=DEVICE,
    )
    actual = getattr(context, _product_method_name(op))(left, right)
    expected = oracle.product(left, right, op=op)

    assert isinstance(executor, FullTableProductExecutor)
    assert executor.executor_family == "full_table"
    assert actual.shape[-1] == context.dim
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_product_executor_policy_selects_sparse_for_pruned_full_layout_wedge():
    context = AlgebraContext(6, 0, 0, device=DEVICE, dtype=torch.float64)
    oracle = _oracle_for(context)
    full_layout = context.layout()
    generator = torch.Generator(device=DEVICE).manual_seed(1019)
    left = torch.randn(2, context.dim, dtype=torch.float64, generator=generator)
    right = torch.randn(2, context.dim, dtype=torch.float64, generator=generator)

    cost = estimate_product_executor_cost(
        context,
        op="wedge",
        left_layout=full_layout,
        right_layout=full_layout,
        output_layout=full_layout,
        dtype=torch.float64,
        device=DEVICE,
    )
    executor = context.planner.product_executor_for_layouts(
        op="wedge",
        left_layout=full_layout,
        right_layout=full_layout,
        output_layout=full_layout,
        dtype=torch.float64,
        device=DEVICE,
    )

    assert cost.executor_family == "sparse"
    assert cost.sparse_score < cost.full_table_score
    assert isinstance(executor, GradeProductExecutor)
    assert torch.allclose(context.wedge(left, right), oracle.product(left, right, op="wedge"), atol=1e-12, rtol=1e-12)


def test_product_executor_policy_override_can_force_full_table_full_layout_wedge():
    policy = ProductExecutionPolicy(cpu_sparse_pair_weight=10.0, cpu_sparse_path_weight=50.0)
    context = AlgebraContext(6, 0, 0, device=DEVICE, dtype=torch.float64, product_execution_policy=policy)
    full_layout = context.layout()

    cost = estimate_product_executor_cost(
        context,
        op="wedge",
        left_layout=full_layout,
        right_layout=full_layout,
        output_layout=full_layout,
        dtype=torch.float64,
        device=DEVICE,
    )
    executor = context.planner.product_executor_for_layouts(
        op="wedge",
        left_layout=full_layout,
        right_layout=full_layout,
        output_layout=full_layout,
        dtype=torch.float64,
        device=DEVICE,
    )

    assert cost.executor_family == "full_table"
    assert cost.full_table_score < cost.sparse_score
    assert isinstance(executor, FullTableProductExecutor)


def test_product_executor_policy_uses_backend_coefficients_without_benchmark_rows():
    context = AlgebraContext(5, 0, 0, device=DEVICE, dtype=torch.float32)
    full_layout = context.layout()

    cpu_cost = estimate_product_executor_cost(
        context,
        op="gp",
        left_layout=full_layout,
        right_layout=full_layout,
        output_layout=full_layout,
        dtype=torch.float32,
        device="cpu",
    )
    mps_cost = estimate_product_executor_cost(
        context,
        op="gp",
        left_layout=full_layout,
        right_layout=full_layout,
        output_layout=full_layout,
        dtype=torch.float32,
        device="mps",
    )

    assert cpu_cost.executor_family == "full_table"
    assert mps_cost.executor_family == "sparse"


def test_direct_product_executor_obeys_static_pair_limits():
    limits = PlanningLimits(warn_lanes=512, max_lanes=512, warn_pairs=512, max_pairs=64)
    algebra = make_algebra(16, 0, 0, device=DEVICE, dtype=torch.float32, planning_limits=limits)

    with pytest.raises(ValueError, match="basis interactions"):
        algebra.planner.product_executor(
            op="gp",
            left_grades=(1,),
            right_grades=(1,),
            output_grades=(0, 2),
            dtype=torch.float32,
            device=DEVICE,
        )
