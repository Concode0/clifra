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


def test_multi_graded_linear_action_matches_stacked_single_actions():
    algebra = AlgebraContext(4, 0, 0, device=DEVICE, dtype=torch.float64)
    layout = algebra.layout((0, 1, 2))
    values = torch.randn(2, 3, layout.dim, dtype=torch.float64)
    matrices = torch.randn(5, algebra.n, algebra.n, dtype=torch.float64)

    actual = apply_multi_graded_linear_action(values, matrices, input_layout=layout, output_layout=layout)
    expected = torch.stack(
        [
            apply_graded_linear_action(
                values,
                matrix.unsqueeze(0).expand(values.shape[-2], -1, -1),
                input_layout=layout,
                output_layout=layout,
            )
            for matrix in matrices
        ],
        dim=-2,
    )

    assert actual.shape == (2, 3, 5, layout.dim)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_full_sandwich_action_executor_matches_small_oracle_action_matrices():
    algebra = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float64)
    oracle = _oracle_for(algebra)
    layout = algebra.layout()
    bivector_layout = algebra.layout((2,))
    executor = FullSandwichActionExecutor.from_layout(layout, device=DEVICE, dtype=torch.float64)
    generator = torch.Generator(device=DEVICE).manual_seed(263)
    bivectors = torch.randn(4, bivector_layout.dim, dtype=torch.float64, generator=generator) * 0.1
    left = algebra.bivector_exp(-0.5 * bivectors, input_layout=bivector_layout, output_layout=layout)
    right = algebra.reverse(left, input_layout=layout, output_layout=layout)
    values = torch.randn(2, 4, algebra.dim, dtype=torch.float64, generator=generator)

    expected_matrices = _oracle_sandwich_action_matrices(oracle, left, right)
    expected_values = torch.einsum("...cj,ckj->...ck", values, expected_matrices)

    assert executor.executor_family == "action_matrix"
    assert torch.allclose(executor.action_matrices(left, right), expected_matrices, atol=1e-12, rtol=1e-12)
    assert torch.allclose(executor.per_channel(left, values, right), expected_values, atol=1e-12, rtol=1e-12)


def test_context_sandwich_helpers_use_planner_full_action_executor():
    algebra = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float64)
    full_layout = algebra.layout()
    bivector_layout = algebra.layout((2,))
    generator = torch.Generator(device=DEVICE).manual_seed(295)
    bivectors = torch.randn(4, bivector_layout.dim, dtype=torch.float64, generator=generator) * 0.1
    left = algebra.bivector_exp(-0.5 * bivectors, input_layout=bivector_layout, output_layout=full_layout)
    right = algebra.reverse(left, input_layout=full_layout, output_layout=full_layout)
    values = torch.randn(2, 4, algebra.dim, dtype=torch.float64, generator=generator)

    actual = algebra.per_channel_sandwich(left, values, right)
    expected = algebra.geometric_product(
        algebra.geometric_product(
            left,
            values,
            left_layout=full_layout,
            right_layout=full_layout,
            output_layout=full_layout,
        ),
        right,
        left_layout=full_layout,
        right_layout=full_layout,
        output_layout=full_layout,
    )
    executors = list(algebra.planner._full_sandwich_action_executors.values())

    assert len(executors) == 1
    assert isinstance(executors[0], FullSandwichActionExecutor)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_context_sandwich_product_and_multi_rotor_sandwich_match_sequential_products():
    algebra = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float64)
    full_layout = algebra.layout()
    bivector_layout = algebra.layout((2,))
    generator = torch.Generator(device=DEVICE).manual_seed(297)
    batch_bivectors = torch.randn(3, bivector_layout.dim, dtype=torch.float64, generator=generator) * 0.1
    rotor_bivectors = torch.randn(5, bivector_layout.dim, dtype=torch.float64, generator=generator) * 0.1
    batch_left = algebra.bivector_exp(-0.5 * batch_bivectors, input_layout=bivector_layout, output_layout=full_layout)
    batch_right = algebra.reverse(batch_left, input_layout=full_layout, output_layout=full_layout)
    rotor_left = algebra.bivector_exp(-0.5 * rotor_bivectors, input_layout=bivector_layout, output_layout=full_layout)
    rotor_right = algebra.reverse(rotor_left, input_layout=full_layout, output_layout=full_layout)
    values = torch.randn(3, 4, algebra.dim, dtype=torch.float64, generator=generator)

    batched_actual = algebra.sandwich_product(batch_left, values, batch_right)
    batched_expected = algebra.geometric_product(
        algebra.geometric_product(
            batch_left.unsqueeze(-2),
            values,
            left_layout=full_layout,
            right_layout=full_layout,
            output_layout=full_layout,
        ),
        batch_right.unsqueeze(-2),
        left_layout=full_layout,
        right_layout=full_layout,
        output_layout=full_layout,
    )
    multi_actual = algebra.multi_rotor_sandwich(rotor_left, values, rotor_right)
    multi_expected = torch.stack(
        [
            algebra.geometric_product(
                algebra.geometric_product(
                    rotor_left[index],
                    values,
                    left_layout=full_layout,
                    right_layout=full_layout,
                    output_layout=full_layout,
                ),
                rotor_right[index],
                left_layout=full_layout,
                right_layout=full_layout,
                output_layout=full_layout,
            )
            for index in range(rotor_left.shape[0])
        ],
        dim=-2,
    )

    assert torch.allclose(batched_actual, batched_expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(multi_actual, multi_expected, atol=1e-12, rtol=1e-12)


def test_plan_sandwich_action_handle_covers_public_full_action_helpers():
    algebra = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float64)
    full_layout = algebra.layout()
    bivector_layout = algebra.layout((2,))
    generator = torch.Generator(device=DEVICE).manual_seed(309)
    left_bivectors = torch.randn(4, bivector_layout.dim, dtype=torch.float64, generator=generator) * 0.1
    batch_bivectors = torch.randn(3, bivector_layout.dim, dtype=torch.float64, generator=generator) * 0.1
    left = algebra.bivector_exp(-0.5 * left_bivectors, input_layout=bivector_layout, output_layout=full_layout)
    right = algebra.reverse(left, input_layout=full_layout, output_layout=full_layout)
    batch_left = algebra.bivector_exp(-0.5 * batch_bivectors, input_layout=bivector_layout, output_layout=full_layout)
    batch_right = algebra.reverse(batch_left, input_layout=full_layout, output_layout=full_layout)
    values = torch.randn(3, 4, algebra.dim, dtype=torch.float64, generator=generator)

    handle = algebra.plan_sandwich_action(layout=full_layout, dtype=torch.float64, device=DEVICE)

    assert isinstance(handle, FullSandwichActionHandle)
    assert handle.executor is algebra.planner.full_sandwich_action_executor_for_layout(
        layout=full_layout,
        dtype=torch.float64,
        device=DEVICE,
    )
    assert torch.allclose(handle.action_matrices(left, right), algebra.sandwich_action_matrices(left, right))
    assert torch.allclose(
        handle.batched(batch_left, values, batch_right), algebra.sandwich_product(batch_left, values, batch_right)
    )
    assert torch.allclose(handle.per_channel(left, values, right), algebra.per_channel_sandwich(left, values, right))
    assert torch.allclose(handle.multi(left, values, right), algebra.multi_rotor_sandwich(left, values, right))


def test_context_full_layout_versor_action_uses_static_action_matrix_executor():
    context = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float64)
    oracle = _oracle_for(context)
    full_layout = context.layout()
    parameter_layout = context.layout((2,))
    generator = torch.Generator(device=DEVICE).manual_seed(269)
    weights = torch.randn(4, parameter_layout.dim, dtype=torch.float64, generator=generator) * 0.1
    values = torch.randn(2, 4, context.dim, dtype=torch.float64, generator=generator)

    left = context.bivector_exp(-0.5 * weights, input_layout=parameter_layout, output_layout=full_layout)
    right = context.reverse(left, input_layout=full_layout, output_layout=full_layout)
    matrices = _oracle_sandwich_action_matrices(oracle, left, right)
    expected = torch.einsum("...cj,ckj->...ck", values, matrices)
    actual = context.versor_action(
        values,
        weights,
        grade=2,
        input_layout=full_layout,
        output_layout=full_layout,
        parameter_layout=parameter_layout,
    )

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_action_plan_handles_match_public_versor_helpers():
    context = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float64)
    full_layout = context.layout()
    parameter_layout = context.layout((2,))
    generator = torch.Generator(device=DEVICE).manual_seed(313)
    values = torch.randn(2, 4, context.dim, dtype=torch.float64, generator=generator)
    weights = torch.randn(4, parameter_layout.dim, dtype=torch.float64, generator=generator) * 0.1
    multi_weights = torch.randn(5, parameter_layout.dim, dtype=torch.float64, generator=generator) * 0.1
    mix = torch.randn(4, 5, dtype=torch.float64, generator=generator)
    left_weights = torch.randn(3, parameter_layout.dim, dtype=torch.float64, generator=generator) * 0.1
    right_weights = torch.randn(3, parameter_layout.dim, dtype=torch.float64, generator=generator) * 0.1
    channel_to_pair = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    versor = context.plan_versor_action(
        grade=2,
        input_layout=full_layout,
        output_layout=full_layout,
        parameter_layout=parameter_layout,
    )
    multi = context.plan_multi_versor_action(
        grade=2,
        input_layout=full_layout,
        output_layout=full_layout,
        parameter_layout=parameter_layout,
    )
    paired = context.plan_paired_bivector_action(
        input_layout=full_layout,
        output_layout=full_layout,
        parameter_layout=parameter_layout,
    )

    assert isinstance(versor, VersorActionHandle)
    assert isinstance(multi, MultiVersorActionHandle)
    assert isinstance(paired, PairedBivectorActionHandle)
    assert versor.executor.bivector_exp is not None
    assert versor.executor.rotor_reverse is not None
    assert multi.executor.bivector_exp is not None
    assert multi.executor.rotor_reverse is not None
    assert paired.executor.bivector_exp is not None
    assert paired.executor.rotor_reverse is not None
    assert torch.allclose(
        versor(values, weights),
        context.versor_action(
            values,
            weights,
            grade=2,
            input_layout=full_layout,
            output_layout=full_layout,
            parameter_layout=parameter_layout,
        ),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        multi(values, multi_weights, mix),
        context.multi_versor_action(
            values,
            multi_weights,
            mix,
            grade=2,
            input_layout=full_layout,
            output_layout=full_layout,
            parameter_layout=parameter_layout,
        ),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        paired(values, left_weights, right_weights, channel_to_pair),
        context.paired_bivector_action(
            values,
            left_weights,
            right_weights,
            channel_to_pair,
            input_layout=full_layout,
            output_layout=full_layout,
            parameter_layout=parameter_layout,
        ),
        atol=1e-12,
        rtol=1e-12,
    )


def test_compact_paired_bivector_action_handle_preplans_factor_products():
    context = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float64)
    vector_layout = context.layout((1,))
    parameter_layout = context.layout((2,))
    handle = context.plan_paired_bivector_action(
        input_layout=vector_layout,
        output_layout=vector_layout,
        parameter_layout=parameter_layout,
    )

    assert isinstance(handle, PairedBivectorActionHandle)
    assert handle.executor.bivector_exp is not None
    assert handle.executor.rotor_reverse is not None
    assert handle.executor.left_product is not None
    assert handle.executor.right_product is not None
    cached_products = set(context.planner._product_executors.values())
    assert handle.executor.left_product.executor in cached_products
    assert handle.executor.right_product.executor in cached_products


def test_compact_versor_action_routes_vector_actions_without_full_rotor_layouts():
    context = AlgebraContext(5, 0, 0, device=DEVICE, dtype=torch.float64)
    vector_layout = context.layout((1,))
    mixed_layout = context.layout((1, 2))
    bivector_layout = context.layout((2,))

    vector_rotor = context.plan_versor_action(
        grade=2,
        input_layout=vector_layout,
        output_layout=vector_layout,
        parameter_layout=bivector_layout,
    )
    vector_multi = context.plan_multi_versor_action(
        grade=2,
        input_layout=vector_layout,
        output_layout=vector_layout,
        parameter_layout=bivector_layout,
    )
    mixed_rotor = context.plan_versor_action(
        grade=2,
        input_layout=mixed_layout,
        output_layout=mixed_layout,
        parameter_layout=bivector_layout,
    )
    reflection = context.plan_versor_action(
        grade=1,
        input_layout=vector_layout,
        output_layout=vector_layout,
        parameter_layout=vector_layout,
    )

    assert not vector_rotor.executor.use_rotor_product_action
    assert vector_rotor.executor.vector_matrix is not None
    assert vector_rotor.executor.action is not None
    assert vector_rotor.executor.bivector_exp is None
    assert vector_rotor.executor.rotor_reverse is None
    assert vector_rotor.executor.left_product is None
    assert vector_rotor.executor.right_product is None
    assert vector_rotor.executor.rotor_layout is None
    assert not vector_multi.executor.use_rotor_product_action
    assert vector_multi.executor.vector_matrix is not None
    assert vector_multi.executor.action is not None
    assert vector_multi.executor.bivector_exp is None
    assert vector_multi.executor.rotor_reverse is None
    assert vector_multi.executor.left_product is None
    assert vector_multi.executor.right_product is None
    assert vector_multi.executor.rotor_layout is None
    assert mixed_rotor.executor.use_rotor_product_action
    assert mixed_rotor.executor.vector_matrix is None
    assert mixed_rotor.executor.action is None
    assert mixed_rotor.executor.bivector_exp is not None
    assert mixed_rotor.executor.rotor_reverse is not None
    assert mixed_rotor.executor.left_product is not None
    assert mixed_rotor.executor.right_product is not None
    assert reflection.executor.vector_matrix.metric_signs.numel() == vector_layout.dim
    assert reflection.executor.action.flat_positions_1.numel() == vector_layout.dim * vector_layout.dim


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_compact_vector_bivector_action_uses_vector_matrix_fullgraph():
    context = AlgebraContext(5, 0, 0, device=DEVICE, dtype=torch.float32)
    vector_layout = context.layout((1,))
    bivector_layout = context.layout((2,))
    handle = context.plan_versor_action(
        grade=2,
        input_layout=vector_layout,
        output_layout=vector_layout,
        parameter_layout=bivector_layout,
    )
    values = torch.randn(
        2,
        3,
        vector_layout.dim,
        dtype=torch.float32,
        generator=torch.Generator(device="cpu").manual_seed(331),
    )
    weights = (
        torch.randn(
            3,
            bivector_layout.dim,
            dtype=torch.float32,
            generator=torch.Generator(device="cpu").manual_seed(337),
        )
        * 0.1
    )

    compiled = torch.compile(handle, backend="aot_eager", fullgraph=True)

    expected = handle(values, weights)
    actual = compiled(values, weights)

    assert not handle.executor.use_rotor_product_action
    assert handle.executor.vector_matrix is not None
    assert handle.executor.bivector_exp is None
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_action_plan_handles_split_checked_validation_from_fast_forward():
    context = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float64)
    vector_layout = context.layout((1,))
    bivector_layout = context.layout((2,))
    handle = context.plan_versor_action(
        grade=2,
        input_layout=vector_layout,
        output_layout=vector_layout,
        parameter_layout=bivector_layout,
    )
    generator = torch.Generator(device=DEVICE).manual_seed(337)
    values = torch.randn(2, 4, vector_layout.dim, dtype=torch.float64, generator=generator)
    weights = torch.randn(4, bivector_layout.dim, dtype=torch.float64, generator=generator) * 0.1

    assert torch.allclose(handle(values, weights), handle.executor.execute(values, weights), atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError, match="expected 3 channels"):
        handle.checked(values, weights[:3])
