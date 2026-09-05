# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from clifra.core._kernel.planning.layouts import ProductRequest
from tests.planning._grade_plan_helpers import (
    DEVICE,
    AlgebraContext,
    FullTableProductExecutor,
    GradePlanner,
    GradeProductExecutor,
    PlannedOperation,
    PseudoscalarProductExecutor,
    SignatureNormSquaredExecutor,
    _grade_only_input,
    _oracle_for,
    make_algebra,
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
    actual = algebra.geometric_product(
        A_values, B_values, left=vector_layout, right=vector_layout, output=output_layout
    )
    compact_actual = algebra.geometric_product(
        A_values, B_values, left=vector_layout, right=vector_layout, output=output_layout
    )

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert compact_actual.shape[-1] == output_layout.dim


def test_grade_planner_reuses_projected_product_executor():
    algebra = AlgebraContext(4, 1, 1, device=DEVICE, dtype=torch.float64)
    planner = GradePlanner(algebra)

    request = ProductRequest.compact(
        planner.spec,
        op="geometric_product",
        left_layout=algebra.layout((1,)),
        right_layout=algebra.layout((1,)),
        output_layout=algebra.layout((0, 2)),
        dtype=torch.float64,
        device=DEVICE,
    )
    first = planner.product_executor(request)
    second = planner.product_executor(request)

    assert first is second


def test_algebra_plan_product_reuses_preplanned_executor():
    algebra = AlgebraContext(6, 0, device=DEVICE, dtype=torch.float32)

    first = algebra.plan_product(left=algebra.layout((1,)), right=algebra.layout((1,)), output=algebra.layout((0, 2)))
    second = algebra.plan_product(left=algebra.layout((1,)), right=algebra.layout((1,)), output=algebra.layout((0, 2)))

    assert first._kernel is second._kernel
    assert first.inputs[0].grades == (1,)
    assert first.inputs[1].grades == (1,)
    assert first.output.grades == (0, 2)
    assert first._kernel.coefficients.dtype == algebra.dtype


def test_algebra_plan_product_returns_compact_lane_handle():
    algebra = AlgebraContext(6, 0, device=DEVICE, dtype=torch.float32)
    oracle = _oracle_for(algebra)
    vector_layout = algebra.layout((1,))
    output_layout = algebra.layout((0, 2))
    generator = torch.Generator(device=DEVICE).manual_seed(301)
    left = torch.randn(3, vector_layout.dim, dtype=torch.float32, generator=generator)
    right = torch.randn(3, vector_layout.dim, dtype=torch.float32, generator=generator)

    handle = algebra.plan_product(op="geometric_product", left=vector_layout, right=vector_layout, output=output_layout)
    expected = oracle.product(
        left,
        right,
        left_indices=vector_layout.basis_indices,
        right_indices=vector_layout.basis_indices,
        output_indices=output_layout.basis_indices,
    )

    assert isinstance(handle, PlannedOperation)
    assert (
        handle._kernel
        is algebra.plan_product(
            op="geometric_product", left=vector_layout, right=vector_layout, output=output_layout
        )._kernel
    )
    assert handle.output.layout == output_layout
    assert torch.allclose(handle(left, right), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("warm_cache", [False, True])
@pytest.mark.parametrize("route", ["projected_product", "plan_product"])
def test_product_routes_reject_foreign_signatures_with_cold_and_warm_caches(route, warm_cache):
    algebra = make_algebra(3, 0, 0, device=DEVICE, dtype=torch.float32)
    foreign = make_algebra(0, 3, 0, device=DEVICE, dtype=torch.float32)
    own_layouts = {
        "left": algebra.layout((1,)),
        "right": algebra.layout((1,)),
        "output": algebra.layout((0,)),
    }
    foreign_layouts = {
        "left": foreign.layout((1,)),
        "right": foreign.layout((1,)),
        "output": foreign.layout((0,)),
    }
    if warm_cache:
        algebra.plan_product(**own_layouts)
    cache_before = tuple(algebra._planner._product_executors.items())

    with pytest.raises(ValueError, match="signature.*does not match"):
        if route == "projected_product":
            values = torch.tensor([[1.0, 0.0, 0.0]])
            algebra.product(values, values, **foreign_layouts)
        else:
            algebra.plan_product(**foreign_layouts)

    assert tuple(algebra._planner._product_executors.items()) == cache_before


@pytest.mark.parametrize("warm_cache", [False, True])
def test_plan_unary_rejects_foreign_signatures_with_cold_and_warm_caches(warm_cache):
    algebra = make_algebra(3, 0, 0, device=DEVICE, dtype=torch.float32)
    foreign = make_algebra(0, 3, 0, device=DEVICE, dtype=torch.float32)
    if warm_cache:
        algebra.plan_unary(op="reverse", input=algebra.layout((1,)), output=algebra.layout((1,)))
    cache_before = tuple(algebra._planner._unary_executors.items())

    with pytest.raises(ValueError, match="signature.*does not match"):
        algebra.plan_unary(op="reverse", input=foreign.layout((1,)), output=foreign.layout((1,)))

    assert tuple(algebra._planner._unary_executors.items()) == cache_before


@pytest.mark.parametrize("cache", [False, True])
def test_unary_executor_rejects_foreign_request_before_cache_lookup(cache):
    algebra = make_algebra(3, 0, 0, device=DEVICE, dtype=torch.float32)
    foreign = make_algebra(0, 3, 0, device=DEVICE, dtype=torch.float32)
    values = torch.zeros(1, 3)
    foreign_request = foreign._planner.unary_request(
        values,
        op="reverse",
        input_layout=foreign.layout((1,)),
        output_layout=foreign.layout((1,)),
    )
    algebra.plan_unary(op="reverse", input=algebra.layout((1,)), output=algebra.layout((1,)))
    cache_before = tuple(algebra._planner._unary_executors.items())

    with pytest.raises(ValueError, match="request signature .* does not match algebra signature"):
        algebra._planner.unary_executor(foreign_request, cache=cache)

    assert tuple(algebra._planner._unary_executors.items()) == cache_before


def test_module_apply_discards_foreign_cached_executor_without_rekeying():
    algebra = make_algebra(3, 0, 0, device=DEVICE, dtype=torch.float32)
    foreign = make_algebra(4, 0, 0, device=DEVICE, dtype=torch.float32)
    foreign_executor = foreign.plan_unary(op="reverse", input=foreign.layout((1,)), output=foreign.layout((1,)))._kernel
    algebra._planner._unary_executors[("foreign",)] = foreign_executor

    algebra._apply(lambda tensor: tensor)

    assert not algebra._planner._unary_executors


def test_plan_unary_accepts_layouts_from_equal_signature_algebra():
    algebra = make_algebra(3, 0, 0, device=DEVICE, dtype=torch.float32)
    peer = make_algebra(3, 0, 0, device=DEVICE, dtype=torch.float32)
    own = algebra.plan_unary(op="reverse", input=algebra.layout((1,)), output=algebra.layout((1,)))
    shared = algebra.plan_unary(op="reverse", input=peer.layout((1,)), output=peer.layout((1,)))

    assert shared._kernel is own._kernel


@pytest.mark.parametrize("warm_cache", [False, True])
@pytest.mark.parametrize(
    "route,cache_name,error_role",
    [
        ("signature_norm", "_signature_norm_squared_executors", "input_layout"),
        ("pseudoscalar", "_pseudoscalar_product_executors", "input_layout"),
        ("bivector_exp", "_bivector_exp_executors", "input_layout"),
        ("sandwich", "_full_sandwich_action_executors", "layout"),
    ],
)
def test_cached_nonproduct_plans_reject_foreign_contracts_before_lookup(route, cache_name, error_role, warm_cache):
    algebra = make_algebra(3, 0, 0, device=DEVICE, dtype=torch.float32)
    foreign = make_algebra(0, 3, 0, device=DEVICE, dtype=torch.float32)

    def plan(source):
        if route == "signature_norm":
            return algebra.plan_signature_norm_squared(input=source.layout((1,)))
        if route == "pseudoscalar":
            return algebra.plan_pseudoscalar_product(input=source.layout((1,)), output=source.layout((2,)))
        if route == "bivector_exp":
            return algebra.plan_bivector_exp(input=source.layout((2,)), output=source.layout((0, 2)))
        return action_helpers.plan_sandwich_action(algebra, layout=source.layout())

    if warm_cache:
        plan(algebra)
    cache = getattr(algebra._planner, cache_name)
    cache_before = tuple(cache.items())

    with pytest.raises(ValueError, match="signature.*does not match"):
        plan(foreign)

    assert tuple(cache.items()) == cache_before


def test_algebra_plan_unary_signature_pseudoscalar_and_bivector_exp_handles_match_public_routes():
    algebra = AlgebraContext(5, 0, device=DEVICE, dtype=torch.float32)
    vector_layout = algebra.layout((1,))
    pseudoscalar_layout = algebra.layout((4,))
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout((0, 2, 4))
    generator = torch.Generator(device=DEVICE).manual_seed(307)
    vector = torch.randn(3, vector_layout.dim, dtype=torch.float32, generator=generator)
    bivector = torch.randn(3, bivector_layout.dim, dtype=torch.float32, generator=generator) * 0.1

    reverse = algebra.plan_unary(op="reverse", input=vector_layout)
    signature_norm_squared = algebra.plan_signature_norm_squared(input=vector_layout)
    pseudoscalar_product = algebra.plan_pseudoscalar_product(input=vector_layout, output=pseudoscalar_layout)
    bivector_exp = algebra.plan_bivector_exp(input=bivector_layout, output=even_layout)

    assert isinstance(reverse, PlannedOperation)
    assert reverse.output.layout == vector_layout
    assert torch.allclose(reverse(vector), algebra.reverse(vector, input=vector_layout))
    assert torch.allclose(signature_norm_squared(vector), algebra.signature_norm_squared(vector, input=vector_layout))
    assert torch.allclose(
        pseudoscalar_product(vector),
        algebra.pseudoscalar_product(vector, input=vector_layout, output=pseudoscalar_layout),
    )
    assert torch.allclose(
        bivector_exp(bivector), algebra.bivector_exp(bivector, input=bivector_layout, output=even_layout)
    )


def test_grade_planner_rebuilds_executor_after_dtype_move():
    algebra = AlgebraContext(4, 1, 1, device=DEVICE, dtype=torch.float64)
    executor = algebra.plan_product(
        op="geometric_product", left=algebra.layout((1,)), right=algebra.layout((1,)), output=algebra.layout((0, 2))
    )._kernel

    algebra.to(dtype=torch.float32)
    moved = algebra.plan_product(
        op="geometric_product", left=algebra.layout((1,)), right=algebra.layout((1,)), output=algebra.layout((0, 2))
    )._kernel

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

    result = algebra.product(A, B, left=vector_layout, right=vector_layout, output=output_layout)
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

    actual = algebra.product(A, B_full, left=vector_layout, right=TensorContract.canonical(vector_layout))

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

    actual = algebra.geometric_product(A, B, left=vector_layout, right=vector_layout, output=output_layout)
    expected = algebra.geometric_product(A, B, left=vector_layout, right=vector_layout, output=output_layout)

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_context_projected_product_handles_high_dim_vector_product():
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32)
    vector_layout = algebra.layout((1,))
    A = torch.zeros(1, vector_layout.dim)
    B = torch.zeros(1, vector_layout.dim)
    A[0, 0] = 1.0
    B[0, 0] = 1.0
    B[0, 1] = 1.0

    values, layout = (
        algebra.geometric_product(A, B, left=vector_layout, right=vector_layout, output=algebra.layout((0, 2))),
        algebra.layout((0, 2)),
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

    projected, projected_layout = (
        algebra.grade_projection(vector, input=vector_layout, output=algebra.layout((1,))),
        algebra.layout((1,)),
    )
    reversed_bivector = algebra.reverse(bivector, input=bivector_layout)
    vector_pos = projected_layout.basis_indices.index(1)
    bivector_pos = bivector_layout.basis_indices.index(3)

    assert torch.allclose(projected[0, vector_pos], torch.tensor(2.0))
    assert torch.allclose(reversed_bivector[0, bivector_pos], torch.tensor(-5.0))


def test_context_planned_unary_compact_reverse():
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32)
    layout = algebra.layout((2,))
    values = torch.arange(layout.dim, dtype=torch.float32).unsqueeze(0)

    actual, output_layout = (algebra.reverse(values, input=layout), layout)

    assert output_layout == layout
    assert torch.allclose(actual, -values)


def test_planned_signature_norm_squared_matches_small_oracle_for_full_and_compact_layouts():
    context = AlgebraContext(3, 1, 1, device=DEVICE, dtype=torch.float64)
    oracle = _oracle_for(context)
    bivector_layout = context.layout((2,))
    generator = torch.Generator(device=DEVICE).manual_seed(211)
    full = torch.randn(3, context.dim, dtype=torch.float64, generator=generator)
    compact = torch.randn(3, bivector_layout.dim, dtype=torch.float64, generator=generator)

    full_executor = context._planner.signature_norm_squared_executor(
        input_layout=context.layout(),
        dtype=torch.float64,
        device=DEVICE,
    )
    compact_executor = context._planner.signature_norm_squared_executor(
        input_layout=bivector_layout,
        dtype=torch.float64,
        device=DEVICE,
    )

    assert isinstance(full_executor, SignatureNormSquaredExecutor)
    assert full_executor.route == "diagonal"
    assert compact_executor.input_layout == bivector_layout
    assert torch.allclose(
        context.signature_norm_squared(full), oracle.signature_norm_squared(full), atol=1e-12, rtol=1e-12
    )
    assert torch.allclose(
        context.signature_norm_squared(compact, input=bivector_layout),
        oracle.signature_norm_squared(compact, bivector_layout.basis_indices),
        atol=1e-12,
        rtol=1e-12,
    )


def test_planned_pseudoscalar_product_matches_small_oracle_for_full_and_compact_layouts():
    context = AlgebraContext(3, 1, 0, device=DEVICE, dtype=torch.float64)
    oracle = _oracle_for(context)
    vector_layout = context.layout((1,))
    trivector_layout = context.layout((context.n - 1,))
    generator = torch.Generator(device=DEVICE).manual_seed(227)
    full = torch.randn(3, context.dim, dtype=torch.float64, generator=generator)
    compact = torch.randn(3, vector_layout.dim, dtype=torch.float64, generator=generator)

    full_executor = context._planner.pseudoscalar_product_executor(
        input_layout=context.layout(),
        dtype=torch.float64,
        device=DEVICE,
    )
    compact_executor = context._planner.pseudoscalar_product_executor(
        input_layout=vector_layout,
        output_layout=trivector_layout,
        dtype=torch.float64,
        device=DEVICE,
    )
    compact_actual = context.pseudoscalar_product(compact, input=vector_layout)
    compact_layout = trivector_layout
    compact_expected = oracle.pseudoscalar_product(
        compact,
        input_indices=vector_layout.basis_indices,
        output_indices=trivector_layout.basis_indices,
    )

    assert isinstance(full_executor, PseudoscalarProductExecutor)
    assert full_executor.route == "pseudoscalar"
    assert compact_executor.output_layout == trivector_layout
    assert torch.allclose(context.pseudoscalar_product(full), oracle.pseudoscalar_product(full), atol=1e-12, rtol=1e-12)
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

    full_executor = context.plan_product(
        op="left_contraction", left=context.layout(), right=context.layout(), output=context.layout()
    )._kernel
    compact_executor = context.plan_product(
        op="right_contraction", left=bivector_layout, right=vector_layout, output=vector_layout
    )._kernel

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
            vector_values, bivector_values, left=vector_layout, right=bivector_layout, output=vector_layout
        ),
        expected_left,
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        context.right_contraction(
            bivector_values, vector_values, left=bivector_layout, right=vector_layout, output=vector_layout
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
        context.left_contraction(e1, e12, left=vector_layout, right=bivector_layout, output=vector_layout),
        e2,
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        context.left_contraction(e2, e12, left=vector_layout, right=bivector_layout, output=vector_layout),
        -e1,
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        context.right_contraction(e12, e1, left=bivector_layout, right=vector_layout, output=vector_layout),
        -e2,
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        context.right_contraction(e12, e2, left=bivector_layout, right=vector_layout, output=vector_layout),
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

    compact_actual = context.blade_inverse(compact, input=vector_layout)
    compact_layout = vector_layout
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

    project = context.blade_project(values, blade, input=vector_layout, blade=vector_layout)
    project_layout = vector_layout
    reject = context.blade_reject(values, blade, input=vector_layout, blade=vector_layout)
    reject_layout = vector_layout

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

    reflected = context.reflect(values, normal, input=vector_layout, normal=vector_layout)
    reflected_layout = vector_layout
    versor = context.versor_product(normal, values, input=vector_layout, versor=vector_layout)

    assert reflected_layout == vector_layout
    assert torch.allclose(reflected, expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(versor, expected, atol=1e-12, rtol=1e-12)


def test_planner_unary_handles_compact_layouts():
    algebra = AlgebraContext(6, 0, 0, device=DEVICE, dtype=torch.float32)
    layout = algebra.layout((2,))
    values = torch.arange(layout.dim, dtype=torch.float32).unsqueeze(0)

    actual, output_layout = (algebra.reverse(values, input=layout), layout)

    assert output_layout == layout
    assert torch.allclose(actual, -values)


def test_compact_geometric_product_stays_compact_in_high_dimensions():
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32)
    vector_layout = algebra.layout((1,))
    left = torch.zeros(1, vector_layout.dim)
    right = torch.zeros(1, vector_layout.dim)
    left[0, 0] = 1.0
    right[0, 0] = 1.0

    result, result_layout = (
        algebra.geometric_product(left, right, left=vector_layout, right=vector_layout),
        algebra.layout((0, 2)),
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
        (algebra.geometric_product(bivector, vector, left=bivector_layout, right=vector_layout), (1, 3)),
        (algebra.wedge(bivector, vector, left=bivector_layout, right=vector_layout), (3,)),
        (algebra.symmetric_product(bivector, vector, left=bivector_layout, right=vector_layout), (3,)),
        (algebra.commutator_product(bivector, vector, left=bivector_layout, right=vector_layout), (1,)),
        (
            algebra.anti_commutator_product(bivector, vector, left=bivector_layout, right=vector_layout),
            (3,),
        ),
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


def test_explicit_layouts_drive_compact_product():
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32)
    vector_layout = algebra.layout((1,))
    left = torch.zeros(1, vector_layout.dim)
    right = torch.zeros(1, vector_layout.dim)
    left[0, 0] = 1.0
    right[0, 0] = 1.0

    values, output_layout = (
        algebra.geometric_product(left, right, left=vector_layout, right=vector_layout),
        algebra.layout((0, 2)),
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

    values, output_layout = (
        algebra.geometric_product(left, right, left=vector_layout, right=vector_layout),
        algebra.layout((0, 2)),
    )

    assert output_layout.grades == (0, 2)
    assert torch.allclose(values[0, output_layout.basis_indices.index(0)], torch.tensor(1.0))


def test_context_projected_product_pairwise_mixed_compact_widths():
    algebra = make_algebra(16, 0, 0, device=DEVICE, dtype=torch.float32)
    left_layout = algebra.layout((2,))
    right_layout = algebra.layout((1,))
    left = torch.randn(3, left_layout.dim)
    right = torch.randn(4, right_layout.dim)

    values, output_layout = (
        algebra.wedge(left, right, pairwise=True, left=left_layout, right=right_layout, output=algebra.layout((3,))),
        algebra.layout((3,)),
    )
    handle = algebra.plan_product(
        op="wedge", left=algebra.layout((2,)), right=algebra.layout((1,)), output=algebra.layout((3,))
    )

    assert output_layout.grades == (3,)
    assert values.shape == (3, 4, output_layout.dim)
    assert torch.allclose(values, handle._kernel.forward_pairwise_compact(left, right), atol=1e-6, rtol=1e-6)


def test_context_projected_product_suggests_pairwise_for_mismatched_item_axes():
    algebra = make_algebra(16, 0, 0, device=DEVICE, dtype=torch.float32)
    left_layout = algebra.layout((2,))
    right_layout = algebra.layout((1,))
    left = torch.randn(3, left_layout.dim)
    right = torch.randn(4, right_layout.dim)

    with pytest.raises(ValueError, match="Use pairwise=True"):
        algebra.wedge(left, right, left=left_layout, right=right_layout, output=algebra.layout((3,)))


def test_context_pairwise_projected_product_requires_item_axes():
    algebra = make_algebra(16, 0, 0, device=DEVICE, dtype=torch.float32)
    left_layout = algebra.layout((2,))
    right_layout = algebra.layout((1,))

    with pytest.raises(ValueError, match="explicit item axes"):
        algebra.wedge(
            torch.randn(left_layout.dim),
            torch.randn(right_layout.dim),
            pairwise=True,
            left=left_layout,
            right=right_layout,
            output=algebra.layout((3,)),
        )


def test_context_declared_product_returns_compact_output_without_full_materialization():
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32)
    vector_layout = algebra.layout((1,))
    left = torch.zeros(1, vector_layout.dim)
    right = torch.zeros(1, vector_layout.dim)

    values = algebra.geometric_product(left, right, left=vector_layout, right=vector_layout)
    layout = algebra.layout((0, 2))

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


from clifra.core.tensors import TensorContract
from tests.helpers import action as action_helpers
