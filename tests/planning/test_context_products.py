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


def test_algebra_projected_product_matches_small_oracle_and_compact_output():
    algebra = AlgebraContext(4, 1, 1, device=DEVICE, dtype=torch.float64)
    oracle = _oracle_for(algebra)
    vector_layout = algebra.layout((1,))
    output_layout = algebra.layout((0, 2))
    A = _grade_only_input(algebra, 2, (1,), seed=113)
    B = _grade_only_input(algebra, 2, (1,), seed=127)
    A_values = vector_layout.compact(A)
    B_values = vector_layout.compact(B)

    expected = oracle.product(
        A_values,
        B_values,
        left_indices=vector_layout.basis_indices,
        right_indices=vector_layout.basis_indices,
        output_indices=output_layout.basis_indices,
    )
    actual = algebra.projected_geometric_product(
        A_values,
        B_values,
        left_layout=vector_layout,
        right_layout=vector_layout,
        output_layout=output_layout,
    )
    compact_actual = algebra.projected_geometric_product(
        A_values,
        B_values,
        left_layout=vector_layout,
        right_layout=vector_layout,
        output_layout=output_layout,
    )

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert compact_actual.shape[-1] == output_layout.dim


def test_grade_planner_reuses_projected_product_executor():
    algebra = AlgebraContext(4, 1, 1, device=DEVICE, dtype=torch.float64)
    planner = GradePlanner(algebra)

    first = planner.product_executor(
        op="gp",
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
        dtype=torch.float64,
        device=DEVICE,
    )
    second = planner.product_executor(
        op="gp",
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
        dtype=torch.float64,
        device=DEVICE,
    )

    assert first is second


def test_algebra_product_executor_returns_preplanned_runtime_handle():
    algebra = AlgebraContext(6, 0, device=DEVICE, dtype=torch.float32)

    first = algebra.product_executor(
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
    )
    second = algebra.product_executor(
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
    )

    assert first is second
    assert first.left_grades == (1,)
    assert first.right_grades == (1,)
    assert first.output_grades == (0, 2)
    assert first.coefficients.dtype == algebra.dtype


def test_algebra_plan_product_returns_compact_lane_handle():
    algebra = AlgebraContext(6, 0, device=DEVICE, dtype=torch.float32)
    oracle = _oracle_for(algebra)
    vector_layout = algebra.layout((1,))
    output_layout = algebra.layout((0, 2))
    generator = torch.Generator(device=DEVICE).manual_seed(301)
    left = torch.randn(3, vector_layout.dim, dtype=torch.float32, generator=generator)
    right = torch.randn(3, vector_layout.dim, dtype=torch.float32, generator=generator)

    handle = algebra.plan_product(
        op="gp",
        left_layout=vector_layout,
        right_layout=vector_layout,
        output_layout=output_layout,
    )
    expected = oracle.product(
        left,
        right,
        left_indices=vector_layout.basis_indices,
        right_indices=vector_layout.basis_indices,
        output_indices=output_layout.basis_indices,
    )

    assert isinstance(handle, ProductPlanHandle)
    assert handle.executor is algebra.planner.product_executor_for_layouts(
        op="gp",
        left_layout=vector_layout,
        right_layout=vector_layout,
        output_layout=output_layout,
        dtype=torch.float32,
        device=DEVICE,
    )
    assert handle.output_layout == output_layout
    assert torch.allclose(handle(left, right), expected, atol=1e-6, rtol=1e-6)


def test_algebra_plan_unary_norm_dual_and_exp_handles_match_public_routes():
    algebra = AlgebraContext(5, 0, device=DEVICE, dtype=torch.float32)
    vector_layout = algebra.layout((1,))
    dual_layout = algebra.layout((4,))
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout((0, 2, 4))
    generator = torch.Generator(device=DEVICE).manual_seed(307)
    vector = torch.randn(3, vector_layout.dim, dtype=torch.float32, generator=generator)
    bivector = torch.randn(3, bivector_layout.dim, dtype=torch.float32, generator=generator) * 0.1

    reverse = algebra.plan_unary(op="reverse", input_layout=vector_layout)
    norm_sq = algebra.plan_norm_sq(input_layout=vector_layout)
    dual = algebra.plan_dual(input_layout=vector_layout, output_layout=dual_layout)
    exp = algebra.plan_exp(input_layout=bivector_layout, output_layout=even_layout)

    assert isinstance(reverse, UnaryPlanHandle)
    assert reverse.output_layout == vector_layout
    assert torch.allclose(reverse(vector), algebra.reverse(vector, input_layout=vector_layout))
    assert torch.allclose(norm_sq(vector), algebra.norm_sq(vector, input_layout=vector_layout))
    assert torch.allclose(dual(vector), algebra.dual(vector, input_layout=vector_layout, output_layout=dual_layout))
    assert torch.allclose(exp(bivector), algebra.exp(bivector, input_layout=bivector_layout, output_layout=even_layout))


def test_grade_planner_rebuilds_executor_after_dtype_move():
    algebra = AlgebraContext(4, 1, 1, device=DEVICE, dtype=torch.float64)
    executor = algebra.planner.product_executor(
        op="gp",
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
        dtype=algebra.dtype,
        device=DEVICE,
    )

    algebra.to(dtype=torch.float32)
    moved = algebra.planner.product_executor(
        op="gp",
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
        dtype=algebra.dtype,
        device=DEVICE,
    )

    assert moved is not executor
    assert moved.coefficients.dtype == torch.float32


def test_compact_projected_product_returns_declared_output_lanes():
    algebra = AlgebraContext(4, 1, 1, device=DEVICE, dtype=torch.float64)
    oracle = _oracle_for(algebra)
    vector_layout = algebra.layout((1,))
    output_layout = algebra.layout((0, 2))
    A_full = _grade_only_input(algebra, 2, (1,), seed=131)
    B_full = _grade_only_input(algebra, 2, (1,), seed=137)
    A = vector_layout.compact(A_full)
    B = vector_layout.compact(B_full)

    result = algebra.projected_product(
        A,
        B,
        left_layout=vector_layout,
        right_layout=vector_layout,
        output_layout=output_layout,
    )
    expected = oracle.product(
        A,
        B,
        left_indices=vector_layout.basis_indices,
        right_indices=vector_layout.basis_indices,
        output_indices=output_layout.basis_indices,
    )

    assert result.shape[-1] == output_layout.dim
    assert torch.allclose(result, expected, atol=1e-12, rtol=1e-12)


def test_declared_layout_full_operand_is_normalized_by_plan():
    algebra = AlgebraContext(4, 1, 1, device=DEVICE, dtype=torch.float64)
    oracle = _oracle_for(algebra)
    vector_layout = algebra.layout((1,))
    output_layout = algebra.layout((0, 2))
    A = vector_layout.compact(_grade_only_input(algebra, 2, (1,), seed=139))
    B_full = _grade_only_input(algebra, 2, (1,), seed=149)
    expected = oracle.product(
        A,
        vector_layout.compact(B_full),
        left_indices=vector_layout.basis_indices,
        right_indices=vector_layout.basis_indices,
        output_indices=output_layout.basis_indices,
    )

    actual = algebra.projected_product(A, B_full, left_layout=vector_layout, right_layout=vector_layout)

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_make_algebra_returns_planner_context_by_default():
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32)
    small = make_algebra(3, 0, 0, device=DEVICE, dtype=torch.float32)

    assert isinstance(algebra, AlgebraContext)
    assert algebra.n == 16
    assert isinstance(small, AlgebraContext)
    assert small.n == 3


def test_product_methods_accept_shared_planned_operation_kwargs():
    algebra = AlgebraContext(4, 1, 1, device=DEVICE, dtype=torch.float64)
    vector_layout = algebra.layout((1,))
    output_layout = algebra.layout((0, 2))
    A_full = _grade_only_input(algebra, 2, (1,), seed=191)
    B_full = _grade_only_input(algebra, 2, (1,), seed=193)
    A = vector_layout.compact(A_full)
    B = vector_layout.compact(B_full)

    actual = algebra.geometric_product(
        A,
        B,
        left_layout=vector_layout,
        right_layout=vector_layout,
        output_layout=output_layout,
    )
    expected = algebra.projected_geometric_product(
        A,
        B,
        left_layout=vector_layout,
        right_layout=vector_layout,
        output_layout=output_layout,
    )

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_context_projected_product_handles_high_dim_vector_product():
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32)
    vector_layout = algebra.layout((1,))
    A = torch.zeros(1, vector_layout.dim)
    B = torch.zeros(1, vector_layout.dim)
    A[0, 0] = 1.0
    B[0, 0] = 1.0
    B[0, 1] = 1.0

    values, layout = algebra.projected_geometric_product(
        A,
        B,
        left_layout=vector_layout,
        right_layout=vector_layout,
        output_grades=(0, 2),
        return_layout=True,
    )

    scalar_pos = layout.basis_indices.index(0)
    bivector_pos = layout.basis_indices.index(3)
    assert values.shape[-1] == layout.dim
    assert torch.allclose(values[0, scalar_pos], torch.tensor(1.0))
    assert torch.allclose(values[0, bivector_pos], torch.tensor(1.0))


def test_context_planned_unary_projection_and_reverse_avoid_full_layout():
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32)
    vector_layout = algebra.layout((1,))
    bivector_layout = algebra.layout((2,))
    vector = torch.zeros(1, vector_layout.dim)
    bivector = torch.zeros(1, bivector_layout.dim)
    vector[0, 0] = 2.0
    bivector[0, 0] = 5.0

    projected, projected_layout = algebra.grade_projection(
        vector,
        1,
        input_layout=vector_layout,
        return_layout=True,
    )
    reversed_bivector = algebra.reverse(
        bivector,
        input_layout=bivector_layout,
    )
    vector_pos = projected_layout.basis_indices.index(1)
    bivector_pos = bivector_layout.basis_indices.index(3)

    assert torch.allclose(projected[0, vector_pos], torch.tensor(2.0))
    assert torch.allclose(reversed_bivector[0, bivector_pos], torch.tensor(-5.0))


def test_context_planned_unary_compact_reverse():
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32)
    layout = algebra.layout((2,))
    values = torch.arange(layout.dim, dtype=torch.float32).unsqueeze(0)

    actual, output_layout = algebra.reverse(
        values,
        input_layout=layout,
        input_storage=LaneStorage.COMPACT,
        output_storage=LaneStorage.COMPACT,
        return_layout=True,
    )

    assert output_layout == layout
    assert torch.allclose(actual, -values)


def test_planned_norm_sq_matches_small_oracle_for_full_and_compact_layouts():
    context = AlgebraContext(3, 1, 1, device=DEVICE, dtype=torch.float64)
    oracle = _oracle_for(context)
    bivector_layout = context.layout((2,))
    generator = torch.Generator(device=DEVICE).manual_seed(211)
    full = torch.randn(3, context.dim, dtype=torch.float64, generator=generator)
    compact = torch.randn(3, bivector_layout.dim, dtype=torch.float64, generator=generator)

    full_executor = context.planner.norm_sq_executor_for_layout(
        input_layout=context.layout(),
        dtype=torch.float64,
        device=DEVICE,
    )
    compact_executor = context.planner.norm_sq_executor_for_layout(
        input_layout=bivector_layout,
        dtype=torch.float64,
        device=DEVICE,
    )

    assert isinstance(full_executor, NormSquaredExecutor)
    assert full_executor.executor_family == "metric_diagonal"
    assert compact_executor.input_layout == bivector_layout
    assert torch.allclose(context.norm_sq(full), oracle.norm_sq(full), atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        context.norm_sq(compact, input_layout=bivector_layout),
        oracle.norm_sq(compact, bivector_layout.basis_indices),
        atol=1e-12,
        rtol=1e-12,
    )


def test_planned_dual_matches_small_oracle_for_full_and_compact_layouts():
    context = AlgebraContext(3, 1, 0, device=DEVICE, dtype=torch.float64)
    oracle = _oracle_for(context)
    vector_layout = context.layout((1,))
    trivector_layout = context.layout((context.n - 1,))
    generator = torch.Generator(device=DEVICE).manual_seed(227)
    full = torch.randn(3, context.dim, dtype=torch.float64, generator=generator)
    compact = torch.randn(3, vector_layout.dim, dtype=torch.float64, generator=generator)

    full_executor = context.planner.dual_executor_for_layout(
        input_layout=context.layout(),
        dtype=torch.float64,
        device=DEVICE,
    )
    compact_executor = context.planner.dual_executor_for_layout(
        input_layout=vector_layout,
        output_layout=trivector_layout,
        dtype=torch.float64,
        device=DEVICE,
    )
    compact_actual, compact_layout = context.dual(compact, input_layout=vector_layout, return_layout=True)
    compact_expected = oracle.dual(
        compact,
        input_indices=vector_layout.basis_indices,
        output_indices=trivector_layout.basis_indices,
    )

    assert isinstance(full_executor, DualExecutor)
    assert full_executor.executor_family == "unary_permutation"
    assert compact_executor.output_layout == trivector_layout
    assert torch.allclose(context.dual(full), oracle.dual(full), atol=1e-12, rtol=1e-12)
    assert compact_layout == trivector_layout
    assert torch.allclose(compact_actual, compact_expected, atol=1e-12, rtol=1e-12)


def test_planned_contractions_match_small_oracle_for_full_and_compact_layouts():
    context = AlgebraContext(3, 1, 0, device=DEVICE, dtype=torch.float64)
    oracle = _oracle_for(context)
    vector_layout = context.layout((1,))
    bivector_layout = context.layout((2,))
    generator = torch.Generator(device=DEVICE).manual_seed(239)
    full_left = torch.randn(3, context.dim, dtype=torch.float64, generator=generator)
    full_right = torch.randn(3, context.dim, dtype=torch.float64, generator=generator)
    vector_values = torch.randn(3, vector_layout.dim, dtype=torch.float64, generator=generator)
    bivector_values = torch.randn(3, bivector_layout.dim, dtype=torch.float64, generator=generator)

    full_executor = context.planner.product_executor_for_layouts(
        op="left_contraction",
        left_layout=context.layout(),
        right_layout=context.layout(),
        output_layout=context.layout(),
        dtype=torch.float64,
        device=DEVICE,
    )
    compact_executor = context.planner.product_executor_for_layouts(
        op="right_contraction",
        left_layout=bivector_layout,
        right_layout=vector_layout,
        output_layout=vector_layout,
        dtype=torch.float64,
        device=DEVICE,
    )

    expected_left = oracle.product(
        vector_values,
        bivector_values,
        op="left_contraction",
        left_indices=vector_layout.basis_indices,
        right_indices=bivector_layout.basis_indices,
        output_indices=vector_layout.basis_indices,
    )
    expected_right = oracle.product(
        bivector_values,
        vector_values,
        op="right_contraction",
        left_indices=bivector_layout.basis_indices,
        right_indices=vector_layout.basis_indices,
        output_indices=vector_layout.basis_indices,
    )

    assert isinstance(full_executor, FullTableProductExecutor)
    assert isinstance(compact_executor, GradeProductExecutor)
    assert torch.allclose(
        context.left_contraction(full_left, full_right),
        oracle.product(full_left, full_right, op="left_contraction"),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        context.left_contraction(
            vector_values,
            bivector_values,
            left_layout=vector_layout,
            right_layout=bivector_layout,
            output_layout=vector_layout,
        ),
        expected_left,
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        context.right_contraction(
            bivector_values,
            vector_values,
            left_layout=bivector_layout,
            right_layout=vector_layout,
            output_layout=vector_layout,
        ),
        expected_right,
        atol=1e-12,
        rtol=1e-12,
    )


def test_planned_contraction_blade_signs_for_compact_layouts():
    context = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float64)
    vector_layout = context.layout((1,))
    bivector_layout = context.layout((2,))
    e1 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    e2 = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    e12 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)

    assert torch.allclose(
        context.left_contraction(
            e1,
            e12,
            left_layout=vector_layout,
            right_layout=bivector_layout,
            output_layout=vector_layout,
        ),
        e2,
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        context.left_contraction(
            e2,
            e12,
            left_layout=vector_layout,
            right_layout=bivector_layout,
            output_layout=vector_layout,
        ),
        -e1,
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        context.right_contraction(
            e12,
            e1,
            left_layout=bivector_layout,
            right_layout=vector_layout,
            output_layout=vector_layout,
        ),
        -e2,
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        context.right_contraction(
            e12,
            e2,
            left_layout=bivector_layout,
            right_layout=vector_layout,
            output_layout=vector_layout,
        ),
        e1,
        atol=1e-12,
        rtol=1e-12,
    )


def test_planned_blade_inverse_matches_small_oracle_for_full_and_compact_layouts():
    context = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float64)
    oracle = _oracle_for(context)
    vector_layout = context.layout((1,))
    generator = torch.Generator(device=DEVICE).manual_seed(251)
    full = torch.randn(3, context.dim, dtype=torch.float64, generator=generator) * 0.1
    full[..., 1] += 1.0
    compact = torch.randn(3, vector_layout.dim, dtype=torch.float64, generator=generator) * 0.1
    compact[..., 0] += 1.0

    compact_actual, compact_layout = context.blade_inverse(
        compact,
        input_layout=vector_layout,
        return_layout=True,
    )
    compact_expected = oracle.blade_inverse(compact, vector_layout.basis_indices)

    assert torch.allclose(context.blade_inverse(full), oracle.blade_inverse(full), atol=1e-12, rtol=1e-12)
    assert compact_layout == vector_layout
    assert torch.allclose(compact_actual, compact_expected, atol=1e-12, rtol=1e-12)


def test_planned_blade_project_and_reject_exact_for_compact_vectors():
    context = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float64)
    vector_layout = context.layout((1,))
    values = torch.tensor([[2.0, 3.0, 0.0]], dtype=torch.float64)
    blade = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    expected_project = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64)
    expected_reject = torch.tensor([[0.0, 3.0, 0.0]], dtype=torch.float64)

    project, project_layout = context.blade_project(
        values,
        blade,
        input_layout=vector_layout,
        blade_layout=vector_layout,
        return_layout=True,
    )
    reject, reject_layout = context.blade_reject(
        values,
        blade,
        input_layout=vector_layout,
        blade_layout=vector_layout,
        return_layout=True,
    )

    assert project_layout == vector_layout
    assert reject_layout == vector_layout
    assert torch.allclose(project, expected_project, atol=1e-12, rtol=1e-12)
    assert torch.allclose(reject, expected_reject, atol=1e-12, rtol=1e-12)


def test_planned_reflect_and_versor_product_exact_for_compact_vectors():
    context = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float64)
    vector_layout = context.layout((1,))
    values = torch.tensor([[2.0, 3.0, 0.0]], dtype=torch.float64)
    normal = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    expected = torch.tensor([[-2.0, 3.0, 0.0]], dtype=torch.float64)

    reflected, reflected_layout = context.reflect(
        values,
        normal,
        input_layout=vector_layout,
        normal_layout=vector_layout,
        return_layout=True,
    )
    versor = context.versor_product(
        normal,
        values,
        versor_layout=vector_layout,
        input_layout=vector_layout,
    )

    assert reflected_layout == vector_layout
    assert torch.allclose(reflected, expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(versor, expected, atol=1e-12, rtol=1e-12)


def test_planner_unary_handles_compact_layouts():
    algebra = AlgebraContext(6, 0, 0, device=DEVICE, dtype=torch.float32)
    layout = algebra.layout((2,))
    values = torch.arange(layout.dim, dtype=torch.float32).unsqueeze(0)

    actual, output_layout = algebra.reverse(
        values,
        input_layout=layout,
        input_storage=LaneStorage.COMPACT,
        output_storage=LaneStorage.COMPACT,
        return_layout=True,
    )

    assert output_layout == layout
    assert torch.allclose(actual, -values)


def test_compact_geometric_product_stays_compact_in_high_dimensions():
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32)
    vector_layout = algebra.layout((1,))
    left = torch.zeros(1, vector_layout.dim)
    right = torch.zeros(1, vector_layout.dim)
    left[0, 0] = 1.0
    right[0, 0] = 1.0

    result, result_layout = algebra.geometric_product(
        left,
        right,
        left_layout=vector_layout,
        right_layout=vector_layout,
        output_storage=LaneStorage.COMPACT,
        return_layout=True,
    )

    assert result_layout.grades == (0, 2)
    scalar_pos = result_layout.basis_indices.index(0)
    assert torch.allclose(result[0, scalar_pos], torch.tensor(1.0))


def test_compact_binary_products_do_not_unwrap_full_tensors():
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32)
    bivector_layout = algebra.layout((2,))
    vector_layout = algebra.layout((1,))
    bivector = torch.randn(2, bivector_layout.dim)
    vector = torch.randn(2, vector_layout.dim)

    results = [
        (algebra.geometric_product(bivector, vector, left_layout=bivector_layout, right_layout=vector_layout), (1, 3)),
        (algebra.wedge(bivector, vector, left_layout=bivector_layout, right_layout=vector_layout), (3,)),
        (algebra.inner_product(bivector, vector, left_layout=bivector_layout, right_layout=vector_layout), (3,)),
        (algebra.commutator(bivector, vector, left_layout=bivector_layout, right_layout=vector_layout), (1,)),
        (algebra.anti_commutator(bivector, vector, left_layout=bivector_layout, right_layout=vector_layout), (3,)),
    ]

    for values, expected_grades in results:
        assert values.shape[-1] == algebra.layout(expected_grades).dim


def test_layout_conversion_merges_values_without_full_materialization():
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32)
    vector_layout = algebra.layout((1,))
    bivector_layout = algebra.layout((2,))
    vector = torch.ones(1, vector_layout.dim)
    bivector = 2.0 * torch.ones(1, bivector_layout.dim)
    merged_layout = algebra.layout((1, 2))

    result = merged_layout.convert(vector, vector_layout) + merged_layout.convert(bivector, bivector_layout)

    assert merged_layout.grades == (1, 2)
    assert result.shape[-1] == merged_layout.dim


def test_context_default_grades_drive_compact_product_without_callsite_metadata():
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32, default_grades=(1,))
    vector_layout = algebra.layout()
    left = torch.zeros(1, vector_layout.dim)
    right = torch.zeros(1, vector_layout.dim)
    left[0, 0] = 1.0
    right[0, 0] = 1.0

    values, output_layout = algebra.geometric_product(
        left,
        right,
        output_storage=LaneStorage.COMPACT,
        return_layout=True,
    )

    assert vector_layout.grades == (1,)
    assert output_layout.grades == (0, 2)
    assert torch.allclose(values[0, output_layout.basis_indices.index(0)], torch.tensor(1.0))


def test_context_declared_grades_infer_compact_operand_shapes():
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32)
    vector_layout = algebra.layout((1,))
    left = torch.zeros(1, vector_layout.dim)
    right = torch.zeros(1, vector_layout.dim)
    left[0, 0] = 1.0
    right[0, 0] = 1.0

    values, output_layout = algebra.projected_geometric_product(
        left,
        right,
        left_grades=(1,),
        right_grades=(1,),
        output_storage=LaneStorage.COMPACT,
        return_layout=True,
    )

    assert output_layout.grades == (0, 2)
    assert torch.allclose(values[0, output_layout.basis_indices.index(0)], torch.tensor(1.0))


def test_context_projected_product_pairwise_mixed_compact_widths():
    algebra = make_algebra(16, 0, 0, device=DEVICE, dtype=torch.float32)
    left_layout = algebra.layout((2,))
    right_layout = algebra.layout((1,))
    left = torch.randn(3, left_layout.dim)
    right = torch.randn(4, right_layout.dim)

    values, output_layout = algebra.projected_wedge(
        left,
        right,
        left_layout=left_layout,
        right_layout=right_layout,
        output_grades=(3,),
        pairwise=True,
        output_storage=LaneStorage.COMPACT,
        return_layout=True,
    )
    executor = algebra.product_executor(
        op="wedge",
        left_grades=(2,),
        right_grades=(1,),
        output_grades=(3,),
        dtype=torch.float32,
        device=DEVICE,
    )

    assert output_layout.grades == (3,)
    assert values.shape == (3, 4, output_layout.dim)
    assert torch.allclose(values, executor.forward_pairwise_compact(left, right), atol=1e-6, rtol=1e-6)


def test_context_projected_product_suggests_pairwise_for_mismatched_item_axes():
    algebra = make_algebra(16, 0, 0, device=DEVICE, dtype=torch.float32)
    left_layout = algebra.layout((2,))
    right_layout = algebra.layout((1,))
    left = torch.randn(3, left_layout.dim)
    right = torch.randn(4, right_layout.dim)

    with pytest.raises(ValueError, match="Use pairwise=True"):
        algebra.projected_wedge(
            left,
            right,
            left_layout=left_layout,
            right_layout=right_layout,
            output_grades=(3,),
            output_storage=LaneStorage.COMPACT,
        )


def test_context_pairwise_projected_product_requires_item_axes():
    algebra = make_algebra(16, 0, 0, device=DEVICE, dtype=torch.float32)
    left_layout = algebra.layout((2,))
    right_layout = algebra.layout((1,))

    with pytest.raises(ValueError, match="explicit item axes"):
        algebra.projected_wedge(
            torch.randn(left_layout.dim),
            torch.randn(right_layout.dim),
            left_layout=left_layout,
            right_layout=right_layout,
            output_grades=(3,),
            pairwise=True,
            output_storage=LaneStorage.COMPACT,
        )


def test_context_declared_product_returns_compact_output_without_full_materialization():
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32)
    vector_layout = algebra.layout((1,))
    left = torch.zeros(1, vector_layout.dim)
    right = torch.zeros(1, vector_layout.dim)

    values, layout = algebra.projected_geometric_product(
        left,
        right,
        left_layout=vector_layout,
        right_layout=vector_layout,
        return_layout=True,
    )

    assert layout.grades == (0, 2)
    assert values.shape[-1] == layout.dim


def test_high_dim_context_requires_declared_layout_for_products():
    algebra = make_algebra(13, 0, 0, device=DEVICE, dtype=torch.float32)
    A = torch.zeros(1, algebra.dim)
    B = torch.zeros(1, algebra.dim)

    with pytest.raises(ValueError, match="too large"):
        algebra.geometric_product(A, B)

    with pytest.raises(ValueError, match="too large"):
        algebra.reverse(A)


def test_context_defaults_to_full_layout_when_no_grades_are_declared():
    context = make_algebra(4, 0, 0, device=DEVICE, dtype=torch.float64)

    assert context.layout().grades == tuple(range(context.n + 1))


def test_context_full_layout_is_canonical_not_warning_fallback():
    context = make_algebra(
        9,
        0,
        0,
        device=DEVICE,
        dtype=torch.float32,
    )

    layout = context.layout()

    assert layout.grades == tuple(range(context.n + 1))


def test_low_dim_context_can_use_declared_full_layout():
    context = make_algebra(
        4,
        0,
        0,
        device=DEVICE,
        dtype=torch.float64,
    )
    oracle = _oracle_for(context)
    A = _grade_only_input(context, 2, (1,), seed=163)
    B = _grade_only_input(context, 2, (1,), seed=167)

    actual = context.geometric_product(A, B)
    expected = oracle.product(A, B)

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
