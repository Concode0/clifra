# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from tests.planning._grade_plan_helpers import (
    DEVICE,
    AlgebraContext,
    FullSandwichActionExecutor,
    FullTableProductExecutor,
    GradeProductExecutor,
    PseudoscalarProductExecutor,
    SignatureNormSquaredExecutor,
    _grade_only_input,
    _product_method_name,
    build_grade_product_plan,
    make_algebra,
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
        op="geometric_product",
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
    executor = algebra.plan_unary(op="reverse", input=algebra.layout((2,)))._kernel
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
        return algebra.geometric_product(x, y, left=vector_layout, right=vector_layout, output=algebra.layout((0, 2)))

    expected = product(A, B)
    compiled = torch.compile(product, backend="aot_eager", fullgraph=True)
    actual = compiled(A, B)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
@pytest.mark.parametrize("output_grades", [(0, 2), (0,)])
def test_context_projected_product_compiles_fullgraph_from_cold_planner_cache(output_grades):
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()
    algebra = AlgebraContext(6, 0, device=DEVICE, dtype=torch.float32)
    generator = torch.Generator(device=DEVICE).manual_seed(181)
    left = torch.randn(2, algebra.layout((1,)).dim, dtype=torch.float32, generator=generator)
    right = torch.randn(2, algebra.layout((1,)).dim, dtype=torch.float32, generator=generator)

    def product(x, y):
        return algebra.geometric_product(
            x, y, left=algebra.layout((1,)), right=algebra.layout((1,)), output=algebra.layout(output_grades)
        )

    assert not algebra._planner._product_executors
    compiled = torch.compile(product, backend="aot_eager", fullgraph=True)
    actual = compiled(left, right)
    expected = product(left, right)

    assert len(algebra._planner._product_executors) == 1
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_full_table_product_executor_compiles_fullgraph_with_aot_eager():
    algebra = AlgebraContext(4, 0, device=DEVICE, dtype=torch.float32)
    full_layout = algebra.layout()
    executor = algebra.plan_product(
        op="geometric_product", left=full_layout, right=full_layout, output=full_layout
    )._kernel
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
    handle = algebra.plan_product(op="geometric_product", left=vector_layout, right=vector_layout, output=output_layout)
    generator = torch.Generator(device=DEVICE).manual_seed(303)
    left = torch.randn(4, vector_layout.dim, dtype=torch.float32, generator=generator)
    right = torch.randn(4, vector_layout.dim, dtype=torch.float32, generator=generator)
    cache_size = len(algebra._planner._product_executors)

    compiled = torch.compile(handle, backend="aot_eager", fullgraph=True)

    expected = handle(left, right)
    actual = compiled(left, right)

    assert cache_size == 1
    assert len(algebra._planner._product_executors) == cache_size
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_plan_unary_handle_compiles_fullgraph_without_cache_mutation():
    algebra = AlgebraContext(6, 0, device=DEVICE, dtype=torch.float32)
    bivector_layout = algebra.layout((2,))
    handle = algebra.plan_unary(op="reverse", input=bivector_layout)
    values = torch.randn(
        4,
        bivector_layout.dim,
        dtype=torch.float32,
        generator=torch.Generator(device=DEVICE).manual_seed(305),
    )
    cache_size = len(algebra._planner._unary_executors)

    compiled = torch.compile(handle, backend="aot_eager", fullgraph=True)

    expected = handle(values)
    actual = compiled(values)

    assert cache_size == 1
    assert len(algebra._planner._unary_executors) == cache_size
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
@pytest.mark.parametrize(
    "op", ["geometric_product", "wedge", "symmetric_product", "commutator_product", "anti_commutator_product"]
)
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
    left = algebra.bivector_exp(-0.5 * bivectors, input=bivector_layout, output=layout)
    right = algebra.reverse(left, input=layout, output=layout)
    values = torch.randn(2, 4, algebra.dim, dtype=torch.float32, generator=generator)

    compiled = torch.compile(executor.forward, backend="aot_eager", fullgraph=True)

    expected = executor.forward(left, values, right)
    actual = compiled(left, values, right)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_planned_public_per_channel_sandwich_compiles_fullgraph_after_cache_warm():
    algebra = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float32)
    layout = algebra.layout()
    bivector_layout = algebra.layout((2,))
    generator = torch.Generator(device=DEVICE).manual_seed(299)
    bivectors = torch.randn(4, bivector_layout.dim, dtype=torch.float32, generator=generator) * 0.1
    left = algebra.bivector_exp(-0.5 * bivectors, input=bivector_layout, output=layout)
    right = algebra.reverse(left, input=layout, output=layout)
    values = torch.randn(2, 4, algebra.dim, dtype=torch.float32, generator=generator)

    def sandwich(left_arg, values_arg, right_arg):
        return action_helpers.per_channel_sandwich(algebra, left_arg, values_arg, right_arg)

    expected = sandwich(left, values, right)
    compiled = torch.compile(sandwich, backend="aot_eager", fullgraph=True)
    actual = compiled(left, values, right)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_plan_sandwich_action_handle_compiles_fullgraph_without_cache_mutation():
    algebra = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float32)
    layout = algebra.layout()
    bivector_layout = algebra.layout((2,))
    handle = action_helpers.plan_sandwich_action(algebra, layout=layout, dtype=torch.float32, device=DEVICE)
    generator = torch.Generator(device=DEVICE).manual_seed(311)
    bivectors = torch.randn(4, bivector_layout.dim, dtype=torch.float32, generator=generator) * 0.1
    left = algebra.bivector_exp(-0.5 * bivectors, input=bivector_layout, output=layout)
    right = algebra.reverse(left, input=layout, output=layout)
    values = torch.randn(2, 4, algebra.dim, dtype=torch.float32, generator=generator)
    cache_size = len(algebra._planner._full_sandwich_action_executors)

    compiled = torch.compile(handle.forward, backend="aot_eager", fullgraph=True)

    expected = handle.forward(left, values, right)
    actual = compiled(left, values, right)

    assert cache_size == 1
    assert len(algebra._planner._full_sandwich_action_executors) == cache_size
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_plan_versor_action_handle_compiles_fullgraph_with_aot_eager():
    algebra = AlgebraContext(5, 0, 0, device=DEVICE, dtype=torch.float32)
    input_layout = algebra.layout((1,))
    parameter_layout = algebra.layout((2,))
    handle = algebra.plan_versor_action(grade=2, input=input_layout, output=input_layout, parameter=parameter_layout)
    generator = torch.Generator(device=DEVICE).manual_seed(315)
    values = torch.randn(2, 4, input_layout.dim, dtype=torch.float32, generator=generator)
    weights = torch.randn(4, parameter_layout.dim, dtype=torch.float32, generator=generator) * 0.1
    cache_sizes = (
        len(algebra._planner._product_executors),
        len(algebra._planner._unary_executors),
        len(algebra._planner._bivector_exp_executors),
        len(algebra._planner._full_sandwich_action_executors),
    )

    compiled = torch.compile(handle, backend="aot_eager", fullgraph=True)

    expected = handle(values, weights)
    actual = compiled(values, weights)

    assert (
        len(algebra._planner._product_executors),
        len(algebra._planner._unary_executors),
        len(algebra._planner._bivector_exp_executors),
        len(algebra._planner._full_sandwich_action_executors),
    ) == cache_sizes
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_plan_full_versor_action_handle_compiles_fullgraph_without_cache_mutation():
    algebra = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float32)
    full_layout = algebra.layout()
    parameter_layout = algebra.layout((2,))
    handle = algebra.plan_versor_action(grade=2, input=full_layout, output=full_layout, parameter=parameter_layout)
    generator = torch.Generator(device=DEVICE).manual_seed(319)
    values = torch.randn(2, 4, full_layout.dim, dtype=torch.float32, generator=generator)
    weights = torch.randn(4, parameter_layout.dim, dtype=torch.float32, generator=generator) * 0.1
    cache_sizes = (
        len(algebra._planner._product_executors),
        len(algebra._planner._unary_executors),
        len(algebra._planner._bivector_exp_executors),
        len(algebra._planner._full_sandwich_action_executors),
    )

    compiled = torch.compile(handle, backend="aot_eager", fullgraph=True)

    expected = handle(values, weights)
    actual = compiled(values, weights)

    assert (
        len(algebra._planner._product_executors),
        len(algebra._planner._unary_executors),
        len(algebra._planner._bivector_exp_executors),
        len(algebra._planner._full_sandwich_action_executors),
    ) == cache_sizes
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_signature_norm_squared_executor_compiles_fullgraph_with_aot_eager():
    algebra = AlgebraContext(5, 0, device=DEVICE, dtype=torch.float32)
    layout = algebra.layout((2,))
    executor = algebra._planner.signature_norm_squared_executor(
        input_layout=layout,
        dtype=torch.float32,
        device=DEVICE,
    )
    values = torch.randn(4, layout.dim, dtype=torch.float32, generator=torch.Generator(device=DEVICE).manual_seed(223))

    compiled = torch.compile(executor, backend="aot_eager", fullgraph=True)

    expected = executor(values)
    actual = compiled(values)

    assert isinstance(executor, SignatureNormSquaredExecutor)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_planned_blade_inverse_compiles_fullgraph_after_cache_warm():
    algebra = AlgebraContext(5, 0, device=DEVICE, dtype=torch.float32)
    layout = algebra.layout((1,))
    values = torch.randn(4, layout.dim, dtype=torch.float32, generator=torch.Generator(device=DEVICE).manual_seed(257))
    values[..., 0] += 1.0

    def inverse(x):
        return algebra.blade_inverse(x, input=layout)

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
        return algebra.reflect(values_arg, normals_arg, input=layout, normal=layout)

    expected = reflect(values, normals)
    compiled = torch.compile(reflect, backend="aot_eager", fullgraph=True)
    actual = compiled(values, normals)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_pseudoscalar_product_executor_compiles_fullgraph_with_aot_eager():
    algebra = AlgebraContext(5, 0, device=DEVICE, dtype=torch.float32)
    input_layout = algebra.layout((1,))
    output_layout = algebra.layout((4,))
    executor = algebra._planner.pseudoscalar_product_executor(
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

    assert isinstance(executor, PseudoscalarProductExecutor)
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
    executor = algebra.plan_product(op=op, left=left_layout, right=right_layout, output=output_layout)._kernel
    generator = torch.Generator(device=DEVICE).manual_seed(241)
    left = torch.randn(4, left_layout.dim, dtype=torch.float32, generator=generator)
    right = torch.randn(4, right_layout.dim, dtype=torch.float32, generator=generator)

    compiled = torch.compile(executor.forward_compact, backend="aot_eager", fullgraph=True)

    expected = executor.forward_compact(left, right)
    actual = compiled(left, right)

    assert isinstance(executor, GradeProductExecutor)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


from tests.helpers import action as action_helpers
