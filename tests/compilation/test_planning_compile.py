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


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_static_grade_product_compiles_fullgraph_with_aot_eager():
    algebra = AlgebraContext(5, 1, 0, device=DEVICE, dtype=torch.float32)
    plan = build_grade_product_plan(
        algebra.p,
        algebra.q,
        algebra.r,
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
        op="gp",
        device=DEVICE,
        dtype=torch.float32,
    )
    product = GradeProductExecutor(plan)
    A = _grade_only_input(algebra, 2, (1,), seed=107).to(dtype=torch.float32)
    B = _grade_only_input(algebra, 2, (1,), seed=109).to(dtype=torch.float32)

    compiled = torch.compile(product, backend="aot_eager", fullgraph=True)

    expected = product(A, B)
    actual = compiled(A, B)

    assert actual.shape[-1] == product.output_dim
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_static_grade_product_pairwise_compact_compiles_fullgraph_with_aot_eager():
    algebra = AlgebraContext(6, 0, 0, device=DEVICE, dtype=torch.float32)
    plan = build_grade_product_plan(
        algebra.p,
        algebra.q,
        algebra.r,
        left_grades=(2,),
        right_grades=(1,),
        output_grades=(3,),
        op="wedge",
        device=DEVICE,
        dtype=torch.float32,
    )
    product = GradeProductExecutor(plan)
    generator = torch.Generator(device=DEVICE).manual_seed(123)
    left = torch.randn(2, 3, plan.left_layout.dim, dtype=torch.float32, generator=generator)
    right = torch.randn(2, 4, plan.right_layout.dim, dtype=torch.float32, generator=generator)

    compiled = torch.compile(product.forward_pairwise_compact, backend="aot_eager", fullgraph=True)

    expected = product.forward_pairwise_compact(left, right)
    actual = compiled(left, right)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_planned_unary_compiles_fullgraph_with_aot_eager():
    algebra = make_algebra(6, 0, 0, device=DEVICE, dtype=torch.float32)
    executor = algebra.planner.unary_executor(
        op="reverse",
        input_grades=(2,),
        dtype=torch.float32,
        device=DEVICE,
    )
    values = _grade_only_input(algebra, 2, (2,), seed=173).to(dtype=torch.float32)

    compiled = torch.compile(executor, backend="aot_eager", fullgraph=True)

    expected = executor(values)
    actual = compiled(values)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_algebra_projected_product_compiles_fullgraph_after_cache_warm():
    algebra = AlgebraContext(5, 1, 0, device=DEVICE, dtype=torch.float32)
    vector_layout = algebra.layout((1,))
    A = vector_layout.compact(_grade_only_input(algebra, 2, (1,), seed=151)).to(dtype=torch.float32)
    B = vector_layout.compact(_grade_only_input(algebra, 2, (1,), seed=157)).to(dtype=torch.float32)

    def product(x, y):
        return algebra.projected_geometric_product(
            x,
            y,
            left_layout=vector_layout,
            right_layout=vector_layout,
            output_grades=(0, 2),
        )

    expected = product(A, B)
    compiled = torch.compile(product, backend="aot_eager", fullgraph=True)
    actual = compiled(A, B)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_context_projected_product_compiles_fullgraph_from_cold_planner_cache():
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()
    algebra = AlgebraContext(6, 0, device=DEVICE, dtype=torch.float32)
    generator = torch.Generator(device=DEVICE).manual_seed(181)
    left = torch.randn(2, algebra.layout((1,)).dim, dtype=torch.float32, generator=generator)
    right = torch.randn(2, algebra.layout((1,)).dim, dtype=torch.float32, generator=generator)

    def product(x, y):
        return algebra.geometric_product(
            x,
            y,
            left_grades=(1,),
            right_grades=(1,),
            output_grades=(0, 2),
            left_storage=LaneStorage.COMPACT,
            right_storage=LaneStorage.COMPACT,
            output_storage=LaneStorage.COMPACT,
        )

    assert not algebra.planner._product_executors
    compiled = torch.compile(product, backend="aot_eager", fullgraph=True)
    actual = compiled(left, right)
    expected = product(left, right)

    assert len(algebra.planner._product_executors) == 1
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_full_table_product_executor_compiles_fullgraph_with_aot_eager():
    algebra = AlgebraContext(4, 0, device=DEVICE, dtype=torch.float32)
    full_layout = algebra.layout()
    executor = algebra.planner.product_executor_for_layouts(
        op="gp",
        left_layout=full_layout,
        right_layout=full_layout,
        output_layout=full_layout,
        dtype=torch.float32,
        device=DEVICE,
    )
    generator = torch.Generator(device=DEVICE).manual_seed(197)
    left = torch.randn(4, algebra.dim, dtype=torch.float32, generator=generator)
    right = torch.randn(4, algebra.dim, dtype=torch.float32, generator=generator)

    compiled = torch.compile(executor.forward_compact, backend="aot_eager", fullgraph=True)

    expected = executor.forward_compact(left, right)
    actual = compiled(left, right)

    assert isinstance(executor, FullTableProductExecutor)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_plan_product_handle_compiles_fullgraph_without_cache_mutation():
    algebra = AlgebraContext(6, 0, device=DEVICE, dtype=torch.float32)
    vector_layout = algebra.layout((1,))
    output_layout = algebra.layout((0, 2))
    handle = algebra.plan_product(
        op="gp",
        left_layout=vector_layout,
        right_layout=vector_layout,
        output_layout=output_layout,
    )
    generator = torch.Generator(device=DEVICE).manual_seed(303)
    left = torch.randn(4, vector_layout.dim, dtype=torch.float32, generator=generator)
    right = torch.randn(4, vector_layout.dim, dtype=torch.float32, generator=generator)
    cache_size = len(algebra.planner._product_executors)

    compiled = torch.compile(handle, backend="aot_eager", fullgraph=True)

    expected = handle(left, right)
    actual = compiled(left, right)

    assert cache_size == 1
    assert len(algebra.planner._product_executors) == cache_size
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_plan_unary_handle_compiles_fullgraph_without_cache_mutation():
    algebra = AlgebraContext(6, 0, device=DEVICE, dtype=torch.float32)
    bivector_layout = algebra.layout((2,))
    handle = algebra.plan_unary(op="reverse", input_layout=bivector_layout)
    values = torch.randn(
        4,
        bivector_layout.dim,
        dtype=torch.float32,
        generator=torch.Generator(device=DEVICE).manual_seed(305),
    )
    cache_size = len(algebra.planner._unary_executors)

    compiled = torch.compile(handle, backend="aot_eager", fullgraph=True)

    expected = handle(values)
    actual = compiled(values)

    assert cache_size == 1
    assert len(algebra.planner._unary_executors) == cache_size
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
@pytest.mark.parametrize("op", ["gp", "wedge", "inner", "commutator", "anti_commutator"])
def test_clifford_public_full_layout_product_compiles_fullgraph_after_cache_warm(op):
    algebra = AlgebraContext(4, 0, 0, device=DEVICE, dtype=torch.float32)
    generator = torch.Generator(device=DEVICE).manual_seed(293)
    left = torch.randn(4, algebra.dim, dtype=torch.float32, generator=generator)
    right = torch.randn(4, algebra.dim, dtype=torch.float32, generator=generator)
    method = getattr(algebra, _product_method_name(op))

    def product(x, y):
        return method(x, y)

    expected = product(left, right)
    compiled = torch.compile(product, backend="aot_eager", fullgraph=True)
    actual = compiled(left, right)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_full_sandwich_action_executor_compiles_fullgraph_with_aot_eager():
    algebra = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float32)
    layout = algebra.layout()
    bivector_layout = algebra.layout((2,))
    executor = FullSandwichActionExecutor.from_layout(layout, device=DEVICE, dtype=torch.float32)
    generator = torch.Generator(device=DEVICE).manual_seed(271)
    bivectors = torch.randn(4, bivector_layout.dim, dtype=torch.float32, generator=generator) * 0.1
    left = algebra.exp(-0.5 * bivectors, input_layout=bivector_layout, output_layout=layout)
    right = algebra.reverse(left, input_layout=layout, output_layout=layout)
    values = torch.randn(2, 4, algebra.dim, dtype=torch.float32, generator=generator)

    compiled = torch.compile(executor.per_channel, backend="aot_eager", fullgraph=True)

    expected = executor.per_channel(left, values, right)
    actual = compiled(left, values, right)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_planned_public_per_channel_sandwich_compiles_fullgraph_after_cache_warm():
    algebra = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float32)
    layout = algebra.layout()
    bivector_layout = algebra.layout((2,))
    generator = torch.Generator(device=DEVICE).manual_seed(299)
    bivectors = torch.randn(4, bivector_layout.dim, dtype=torch.float32, generator=generator) * 0.1
    left = algebra.exp(-0.5 * bivectors, input_layout=bivector_layout, output_layout=layout)
    right = algebra.reverse(left, input_layout=layout, output_layout=layout)
    values = torch.randn(2, 4, algebra.dim, dtype=torch.float32, generator=generator)

    def sandwich(left_arg, values_arg, right_arg):
        return algebra.per_channel_sandwich(left_arg, values_arg, right_arg)

    expected = sandwich(left, values, right)
    compiled = torch.compile(sandwich, backend="aot_eager", fullgraph=True)
    actual = compiled(left, values, right)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_plan_sandwich_action_handle_compiles_fullgraph_without_cache_mutation():
    algebra = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float32)
    layout = algebra.layout()
    bivector_layout = algebra.layout((2,))
    handle = algebra.plan_sandwich_action(layout=layout, dtype=torch.float32, device=DEVICE)
    generator = torch.Generator(device=DEVICE).manual_seed(311)
    bivectors = torch.randn(4, bivector_layout.dim, dtype=torch.float32, generator=generator) * 0.1
    left = algebra.exp(-0.5 * bivectors, input_layout=bivector_layout, output_layout=layout)
    right = algebra.reverse(left, input_layout=layout, output_layout=layout)
    values = torch.randn(2, 4, algebra.dim, dtype=torch.float32, generator=generator)
    cache_size = len(algebra.planner._full_sandwich_action_executors)

    compiled = torch.compile(handle, backend="aot_eager", fullgraph=True)

    expected = handle(left, values, right)
    actual = compiled(left, values, right)

    assert cache_size == 1
    assert len(algebra.planner._full_sandwich_action_executors) == cache_size
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_plan_versor_action_handle_compiles_fullgraph_with_aot_eager():
    algebra = AlgebraContext(5, 0, 0, device=DEVICE, dtype=torch.float32)
    input_layout = algebra.layout((1,))
    parameter_layout = algebra.layout((2,))
    handle = algebra.plan_versor_action(
        grade=2,
        input_layout=input_layout,
        output_layout=input_layout,
        parameter_layout=parameter_layout,
    )
    generator = torch.Generator(device=DEVICE).manual_seed(315)
    values = torch.randn(2, 4, input_layout.dim, dtype=torch.float32, generator=generator)
    weights = torch.randn(4, parameter_layout.dim, dtype=torch.float32, generator=generator) * 0.1
    cache_sizes = (
        len(algebra.planner._product_executors),
        len(algebra.planner._unary_executors),
        len(algebra.planner._bivector_exp_executors),
        len(algebra.planner._full_sandwich_action_executors),
    )

    compiled = torch.compile(handle, backend="aot_eager", fullgraph=True)

    expected = handle(values, weights)
    actual = compiled(values, weights)

    assert (
        len(algebra.planner._product_executors),
        len(algebra.planner._unary_executors),
        len(algebra.planner._bivector_exp_executors),
        len(algebra.planner._full_sandwich_action_executors),
    ) == cache_sizes
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_plan_multi_versor_action_handle_compiles_fullgraph_without_cache_mutation():
    algebra = AlgebraContext(5, 0, 0, device=DEVICE, dtype=torch.float32)
    input_layout = algebra.layout((1,))
    parameter_layout = algebra.layout((2,))
    handle = algebra.plan_multi_versor_action(
        grade=2,
        input_layout=input_layout,
        output_layout=input_layout,
        parameter_layout=parameter_layout,
    )
    generator = torch.Generator(device=DEVICE).manual_seed(317)
    values = torch.randn(2, 4, input_layout.dim, dtype=torch.float32, generator=generator)
    weights = torch.randn(5, parameter_layout.dim, dtype=torch.float32, generator=generator) * 0.1
    mix = torch.randn(4, 5, dtype=torch.float32, generator=generator)
    cache_sizes = (
        len(algebra.planner._product_executors),
        len(algebra.planner._unary_executors),
        len(algebra.planner._bivector_exp_executors),
        len(algebra.planner._full_sandwich_action_executors),
    )

    compiled = torch.compile(handle, backend="aot_eager", fullgraph=True)

    expected = handle(values, weights, mix)
    actual = compiled(values, weights, mix)

    assert (
        len(algebra.planner._product_executors),
        len(algebra.planner._unary_executors),
        len(algebra.planner._bivector_exp_executors),
        len(algebra.planner._full_sandwich_action_executors),
    ) == cache_sizes
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_plan_full_versor_action_handle_compiles_fullgraph_without_cache_mutation():
    algebra = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float32)
    full_layout = algebra.layout()
    parameter_layout = algebra.layout((2,))
    handle = algebra.plan_versor_action(
        grade=2,
        input_layout=full_layout,
        output_layout=full_layout,
        parameter_layout=parameter_layout,
    )
    generator = torch.Generator(device=DEVICE).manual_seed(319)
    values = torch.randn(2, 4, full_layout.dim, dtype=torch.float32, generator=generator)
    weights = torch.randn(4, parameter_layout.dim, dtype=torch.float32, generator=generator) * 0.1
    cache_sizes = (
        len(algebra.planner._product_executors),
        len(algebra.planner._unary_executors),
        len(algebra.planner._bivector_exp_executors),
        len(algebra.planner._full_sandwich_action_executors),
    )

    compiled = torch.compile(handle, backend="aot_eager", fullgraph=True)

    expected = handle(values, weights)
    actual = compiled(values, weights)

    assert (
        len(algebra.planner._product_executors),
        len(algebra.planner._unary_executors),
        len(algebra.planner._bivector_exp_executors),
        len(algebra.planner._full_sandwich_action_executors),
    ) == cache_sizes
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_plan_paired_bivector_action_handle_compiles_fullgraph_without_cache_mutation():
    algebra = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float32)
    vector_layout = algebra.layout((1,))
    parameter_layout = algebra.layout((2,))
    handle = algebra.plan_paired_bivector_action(
        input_layout=vector_layout,
        output_layout=vector_layout,
        parameter_layout=parameter_layout,
    )
    generator = torch.Generator(device=DEVICE).manual_seed(323)
    values = torch.randn(2, 4, vector_layout.dim, dtype=torch.float32, generator=generator)
    left_weights = torch.randn(3, parameter_layout.dim, dtype=torch.float32, generator=generator) * 0.1
    right_weights = torch.randn(3, parameter_layout.dim, dtype=torch.float32, generator=generator) * 0.1
    channel_to_pair = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    cache_sizes = (
        len(algebra.planner._product_executors),
        len(algebra.planner._unary_executors),
        len(algebra.planner._bivector_exp_executors),
        len(algebra.planner._full_sandwich_action_executors),
    )

    compiled = torch.compile(handle, backend="aot_eager", fullgraph=True)

    expected = handle(values, left_weights, right_weights, channel_to_pair)
    actual = compiled(values, left_weights, right_weights, channel_to_pair)

    assert (
        len(algebra.planner._product_executors),
        len(algebra.planner._unary_executors),
        len(algebra.planner._bivector_exp_executors),
        len(algebra.planner._full_sandwich_action_executors),
    ) == cache_sizes
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_norm_sq_executor_compiles_fullgraph_with_aot_eager():
    algebra = AlgebraContext(5, 0, device=DEVICE, dtype=torch.float32)
    layout = algebra.layout((2,))
    executor = algebra.planner.norm_sq_executor_for_layout(
        input_layout=layout,
        dtype=torch.float32,
        device=DEVICE,
    )
    values = torch.randn(4, layout.dim, dtype=torch.float32, generator=torch.Generator(device=DEVICE).manual_seed(223))

    compiled = torch.compile(executor, backend="aot_eager", fullgraph=True)

    expected = executor(values)
    actual = compiled(values)

    assert isinstance(executor, NormSquaredExecutor)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_planned_blade_inverse_compiles_fullgraph_after_cache_warm():
    algebra = AlgebraContext(5, 0, device=DEVICE, dtype=torch.float32)
    layout = algebra.layout((1,))
    values = torch.randn(4, layout.dim, dtype=torch.float32, generator=torch.Generator(device=DEVICE).manual_seed(257))
    values[..., 0] += 1.0

    def inverse(x):
        return algebra.blade_inverse(x, input_layout=layout)

    expected = inverse(values)
    compiled = torch.compile(inverse, backend="aot_eager", fullgraph=True)
    actual = compiled(values)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_planned_reflect_compiles_fullgraph_after_cache_warm():
    algebra = AlgebraContext(5, 0, device=DEVICE, dtype=torch.float32)
    layout = algebra.layout((1,))
    generator = torch.Generator(device=DEVICE).manual_seed(277)
    values = torch.randn(4, layout.dim, dtype=torch.float32, generator=generator)
    normals = torch.randn(4, layout.dim, dtype=torch.float32, generator=generator)
    normals[..., 0] += 1.0

    def reflect(values_arg, normals_arg):
        return algebra.reflect(values_arg, normals_arg, input_layout=layout, normal_layout=layout)

    expected = reflect(values, normals)
    compiled = torch.compile(reflect, backend="aot_eager", fullgraph=True)
    actual = compiled(values, normals)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_dual_executor_compiles_fullgraph_with_aot_eager():
    algebra = AlgebraContext(5, 0, device=DEVICE, dtype=torch.float32)
    input_layout = algebra.layout((1,))
    output_layout = algebra.layout((4,))
    executor = algebra.planner.dual_executor_for_layout(
        input_layout=input_layout,
        output_layout=output_layout,
        dtype=torch.float32,
        device=DEVICE,
    )
    values = torch.randn(
        4,
        input_layout.dim,
        dtype=torch.float32,
        generator=torch.Generator(device=DEVICE).manual_seed(229),
    )

    compiled = torch.compile(executor, backend="aot_eager", fullgraph=True)

    expected = executor(values)
    actual = compiled(values)

    assert isinstance(executor, DualExecutor)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
@pytest.mark.parametrize(
    ("op", "left_grades", "right_grades", "output_grades"),
    [
        ("left_contraction", (1,), (2,), (1,)),
        ("right_contraction", (2,), (1,), (1,)),
    ],
)
def test_contraction_executor_compiles_fullgraph_with_aot_eager(op, left_grades, right_grades, output_grades):
    algebra = AlgebraContext(6, 0, device=DEVICE, dtype=torch.float32)
    left_layout = algebra.layout(left_grades)
    right_layout = algebra.layout(right_grades)
    output_layout = algebra.layout(output_grades)
    executor = algebra.planner.product_executor_for_layouts(
        op=op,
        left_layout=left_layout,
        right_layout=right_layout,
        output_layout=output_layout,
        dtype=torch.float32,
        device=DEVICE,
    )
    generator = torch.Generator(device=DEVICE).manual_seed(241)
    left = torch.randn(4, left_layout.dim, dtype=torch.float32, generator=generator)
    right = torch.randn(4, right_layout.dim, dtype=torch.float32, generator=generator)

    compiled = torch.compile(executor.forward_compact, backend="aot_eager", fullgraph=True)

    expected = executor.forward_compact(left, right)
    actual = compiled(left, right)

    assert isinstance(executor, GradeProductExecutor)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
