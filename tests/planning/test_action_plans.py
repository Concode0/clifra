# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from clifra.core._kernel.providers import action_execution_request
from tests.planning._grade_plan_helpers import (
    DEVICE,
    AlgebraContext,
    FullSandwichActionExecutor,
    GradedLinearActionExecutor,
    _oracle_for,
    _oracle_sandwich_action_matrices,
    pytest,
    torch,
)

pytestmark = pytest.mark.unit


def test_policy_selected_action_plans_are_cached_by_static_contract():
    algebra = AlgebraContext(4, 0, 0, device=DEVICE, dtype=torch.float32)
    vector_layout = algebra.layout((1,))
    bivector_layout = algebra.layout((2,))

    first = algebra._planner.versor_action_plan(
        grade=2,
        input_layout=vector_layout,
        output_layout=vector_layout,
        parameter_layout=bivector_layout,
    )
    second = algebra._planner.versor_action_plan(
        grade=2,
        input_layout=vector_layout,
        output_layout=vector_layout,
        parameter_layout=bivector_layout,
    )

    assert second is first
    assert len(algebra._planner._versor_action_plans) == 1


def test_high_dimensional_action_route_selection_uses_only_static_layout_facts(monkeypatch):
    algebra = AlgebraContext(24, 0, 1, device=DEVICE, dtype=torch.float64)
    vector_layout = algebra.layout((1,))
    bivector_layout = algebra.layout((2,))

    def reject_materialization(*args, **kwargs):
        raise AssertionError("route selection must not materialize basis data or tensors")

    monkeypatch.setattr("clifra.core.layout.basis_index_tuple_for_grades", reject_materialization)
    for factory in ("arange", "empty", "eye", "ones", "tensor", "zeros"):
        monkeypatch.setattr(torch, factory, reject_materialization)

    plan = algebra._planner.versor_action_plan(
        grade=2,
        input_layout=vector_layout,
        output_layout=vector_layout,
        parameter_layout=bivector_layout,
    )

    assert plan.route == "vector_matrix"


def test_default_full_layout_action_is_not_forced_to_full_matrix_route():
    algebra = AlgebraContext(3, device=DEVICE, dtype=torch.float32)
    full = algebra.layout()
    operation = algebra.plan_versor_action(
        grade=2,
        input=full,
        output=full,
        parameter=algebra.layout((2,)),
    )

    assert operation._kernel.metadata.route in {"vector_matrix", "rotor_product"}


def test_default_action_policy_is_device_independent():
    algebra = AlgebraContext(4, device=DEVICE, dtype=torch.float32)
    layout = algebra.layout((0, 2, 4))
    request = action_execution_request(
        algebra,
        "versor",
        grade=2,
        input_layout=layout,
        output_layout=layout,
        parameter_layout=algebra.layout((2,)),
    )
    routes = {
        algebra._planner.router.select(
            replace(request, device=torch.device(device)),
            algebra._planner.policy,
            algebra._planner.limits,
        ).route
        for device in ("cpu", "mps", "cuda")
    }

    assert len(routes) == 1


@pytest.mark.parametrize("foreign_side", ["input", "output", "parameter"])
def test_action_plans_reject_foreign_contracts_before_executor_construction(foreign_side):
    algebra = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float32)
    foreign = AlgebraContext(0, 3, 0, device=DEVICE, dtype=torch.float32)
    layouts = {
        "input_layout": algebra.layout((1,)),
        "output_layout": algebra.layout((1,)),
        "parameter_layout": algebra.layout((2,)),
    }
    layouts[f"{foreign_side}_layout"] = foreign.layout((2,) if foreign_side == "parameter" else (1,))
    cache_sizes = (
        len(algebra._planner._product_executors),
        len(algebra._planner._unary_executors),
        len(algebra._planner._bivector_exp_executors),
    )

    with pytest.raises(ValueError, match=rf"{foreign_side}_layout signature .* does not match algebra signature"):
        action_helpers.plan_versor_action(algebra, grade=2, **layouts)

    assert (
        len(algebra._planner._product_executors),
        len(algebra._planner._unary_executors),
        len(algebra._planner._bivector_exp_executors),
    ) == cache_sizes


def test_action_plan_accepts_contracts_from_equal_signature_algebra():
    algebra = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float32)
    peer = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float32)

    handle = algebra.plan_versor_action(
        grade=2, input=peer.layout((1,)), output=peer.layout((1,)), parameter=peer.layout((2,))
    )

    assert handle.inputs[0].layout.spec == algebra._planner.spec
    assert handle.output.layout.spec == algebra._planner.spec
    assert handle.inputs[1].layout.spec == algebra._planner.spec

    plan = algebra._planner.versor_action_plan(
        grade=2,
        input_layout=peer.layout((1,)),
        output_layout=peer.layout((1,)),
        parameter_layout=peer.layout((2,)),
    )
    assert plan.input_contract.spec == algebra._planner.spec
    assert plan.output_contract.spec == algebra._planner.spec
    assert plan.parameter_contract.spec == algebra._planner.spec


def test_multi_graded_linear_action_matches_stacked_single_actions():
    algebra = AlgebraContext(4, 0, 0, device=DEVICE, dtype=torch.float64)
    layout = algebra.layout((0, 1, 2))
    values = torch.randn(2, 3, layout.dim, dtype=torch.float64)
    matrices = torch.randn(5, algebra.n, algebra.n, dtype=torch.float64)

    executor = GradedLinearActionExecutor(input_layout=layout, output_layout=layout)
    actual = executor(values.unsqueeze(-2), matrices)
    expected = torch.stack(
        [executor(values, matrix.unsqueeze(0).expand(values.shape[-2], -1, -1)) for matrix in matrices],
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
    left = algebra.bivector_exp(-0.5 * bivectors, input=bivector_layout, output=layout)
    right = algebra.reverse(left, input=layout, output=layout)
    values = torch.randn(2, 4, algebra.dim, dtype=torch.float64, generator=generator)

    expected_matrices = _oracle_sandwich_action_matrices(oracle, left, right)
    expected_values = torch.einsum("...cj,ckj->...ck", values, expected_matrices)

    assert executor.route == "full_action_matrix"
    assert torch.allclose(executor.action_matrices(left, right), expected_matrices, atol=1e-12, rtol=1e-12)
    assert torch.allclose(executor.forward(left, values, right), expected_values, atol=1e-12, rtol=1e-12)


def test_context_sandwich_helpers_use_planner_full_action_executor():
    algebra = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float64)
    full_layout = algebra.layout()
    bivector_layout = algebra.layout((2,))
    generator = torch.Generator(device=DEVICE).manual_seed(295)
    bivectors = torch.randn(4, bivector_layout.dim, dtype=torch.float64, generator=generator) * 0.1
    left = algebra.bivector_exp(-0.5 * bivectors, input=bivector_layout, output=full_layout)
    right = algebra.reverse(left, input=full_layout, output=full_layout)
    values = torch.randn(2, 4, algebra.dim, dtype=torch.float64, generator=generator)

    actual = action_helpers.per_channel_sandwich(algebra, left, values, right)
    expected = algebra.geometric_product(
        algebra.geometric_product(left, values, left=full_layout, right=full_layout, output=full_layout),
        right,
        left=full_layout,
        right=full_layout,
        output=full_layout,
    )
    executors = list(algebra._planner._full_sandwich_action_executors.values())

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
    batch_left = algebra.bivector_exp(-0.5 * batch_bivectors, input=bivector_layout, output=full_layout)
    batch_right = algebra.reverse(batch_left, input=full_layout, output=full_layout)
    rotor_left = algebra.bivector_exp(-0.5 * rotor_bivectors, input=bivector_layout, output=full_layout)
    rotor_right = algebra.reverse(rotor_left, input=full_layout, output=full_layout)
    values = torch.randn(3, 4, algebra.dim, dtype=torch.float64, generator=generator)

    batched_actual = action_helpers.sandwich_product(algebra, batch_left, values, batch_right)
    batched_expected = algebra.geometric_product(
        algebra.geometric_product(
            batch_left.unsqueeze(-2), values, left=full_layout, right=full_layout, output=full_layout
        ),
        batch_right.unsqueeze(-2),
        left=full_layout,
        right=full_layout,
        output=full_layout,
    )
    multi_actual = action_helpers.multi_rotor_sandwich(algebra, rotor_left, values, rotor_right)
    multi_expected = torch.stack(
        [
            algebra.geometric_product(
                algebra.geometric_product(
                    rotor_left[index], values, left=full_layout, right=full_layout, output=full_layout
                ),
                rotor_right[index],
                left=full_layout,
                right=full_layout,
                output=full_layout,
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
    left = algebra.bivector_exp(-0.5 * left_bivectors, input=bivector_layout, output=full_layout)
    right = algebra.reverse(left, input=full_layout, output=full_layout)
    batch_left = algebra.bivector_exp(-0.5 * batch_bivectors, input=bivector_layout, output=full_layout)
    batch_right = algebra.reverse(batch_left, input=full_layout, output=full_layout)
    values = torch.randn(3, 4, algebra.dim, dtype=torch.float64, generator=generator)

    handle = action_helpers.plan_sandwich_action(algebra, layout=full_layout, dtype=torch.float64, device=DEVICE)

    assert isinstance(handle, FullSandwichActionExecutor)
    assert handle is algebra._planner.full_sandwich_action_executor(
        layout=full_layout,
        dtype=torch.float64,
        device=DEVICE,
    )
    assert torch.allclose(
        handle.action_matrices(left, right), action_helpers.sandwich_action_matrices(algebra, left, right)
    )
    assert torch.allclose(
        handle(batch_left.unsqueeze(-2), values, batch_right.unsqueeze(-2)),
        action_helpers.sandwich_product(algebra, batch_left, values, batch_right),
    )
    assert torch.allclose(
        handle.forward(left, values, right), action_helpers.per_channel_sandwich(algebra, left, values, right)
    )
    assert torch.allclose(
        handle(left, values.unsqueeze(-2), right), action_helpers.multi_rotor_sandwich(algebra, left, values, right)
    )


def test_context_full_layout_versor_action_uses_static_action_matrix_executor():
    context = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float64)
    oracle = _oracle_for(context)
    full_layout = context.layout()
    parameter_layout = context.layout((2,))
    generator = torch.Generator(device=DEVICE).manual_seed(269)
    weights = torch.randn(4, parameter_layout.dim, dtype=torch.float64, generator=generator) * 0.1
    values = torch.randn(2, 4, context.dim, dtype=torch.float64, generator=generator)

    left = context.bivector_exp(-0.5 * weights, input=parameter_layout, output=full_layout)
    right = context.reverse(left, input=full_layout, output=full_layout)
    matrices = _oracle_sandwich_action_matrices(oracle, left, right)
    expected = torch.einsum("...cj,ckj->...ck", values, matrices)
    actual = context.versor_action(
        values, weights, grade=2, input=full_layout, output=full_layout, parameter=parameter_layout
    )

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_compact_versor_action_routes_vector_actions_without_full_rotor_layouts():
    context = AlgebraContext(5, 0, 0, device=DEVICE, dtype=torch.float64)
    vector_layout = context.layout((1,))
    mixed_layout = context.layout((1, 2))
    bivector_layout = context.layout((2,))

    vector_rotor = context.plan_versor_action(
        grade=2, input=vector_layout, output=vector_layout, parameter=bivector_layout
    )
    mixed_rotor = context.plan_versor_action(
        grade=2, input=mixed_layout, output=mixed_layout, parameter=bivector_layout
    )
    reflection = context.plan_versor_action(grade=1, input=vector_layout, output=vector_layout, parameter=vector_layout)

    assert not vector_rotor._kernel.use_rotor_product_action
    assert vector_rotor._kernel.vector_matrix is not None
    assert vector_rotor._kernel.action is not None
    assert vector_rotor._kernel.bivector_exp is None
    assert vector_rotor._kernel.rotor_reverse is None
    assert vector_rotor._kernel.left_product is None
    assert vector_rotor._kernel.right_product is None
    assert vector_rotor._kernel.rotor_layout is None
    assert mixed_rotor._kernel.route == "vector_matrix"
    assert mixed_rotor._kernel.bivector_exp is None
    assert mixed_rotor._kernel.rotor_layout is None
    assert reflection._kernel.vector_matrix.metric_signs.numel() == vector_layout.dim
    assert not hasattr(reflection._kernel.action, "flat_positions_1")


def test_compact_vector_bivector_action_has_nonzero_infinitesimal_gradient_at_identity():
    context = AlgebraContext(2, 0, 0, device=DEVICE, dtype=torch.float64)
    vector_layout = context.layout((1,))
    action = context.plan_versor_action(
        grade=2, input=vector_layout, output=vector_layout, parameter=context.layout((2,))
    )
    values = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64, device=DEVICE)
    weights = torch.zeros(1, 1, dtype=torch.float64, device=DEVICE, requires_grad=True)

    output = action(values, weights)
    output[0, 0, 1].backward()

    assert torch.allclose(weights.grad, torch.ones_like(weights), atol=1e-12, rtol=1e-12)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_compact_vector_bivector_identity_gradient_compiles_fullgraph():
    context = AlgebraContext(2, 0, 0, device=DEVICE, dtype=torch.float64)
    vector_layout = context.layout((1,))
    action = context.plan_versor_action(
        grade=2, input=vector_layout, output=vector_layout, parameter=context.layout((2,))
    )
    values = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64, device=DEVICE)
    weights = torch.zeros(1, 1, dtype=torch.float64, device=DEVICE, requires_grad=True)
    compiled = torch.compile(action, backend="aot_eager", fullgraph=True)

    output = compiled(values, weights)
    output[0, 0, 1].backward()

    assert torch.allclose(weights.grad, torch.ones_like(weights), atol=1e-12, rtol=1e-12)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_compact_vector_bivector_action_uses_vector_matrix_fullgraph():
    context = AlgebraContext(5, 0, 0, device=DEVICE, dtype=torch.float32)
    vector_layout = context.layout((1,))
    bivector_layout = context.layout((2,))
    handle = context.plan_versor_action(grade=2, input=vector_layout, output=vector_layout, parameter=bivector_layout)
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

    assert not handle._kernel.use_rotor_product_action
    assert handle._kernel.vector_matrix is not None
    assert handle._kernel.bivector_exp is None
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_action_plan_handle_validates_inputs_through_executor_forward():
    context = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float64)
    vector_layout = context.layout((1,))
    bivector_layout = context.layout((2,))
    handle = context.plan_versor_action(grade=2, input=vector_layout, output=vector_layout, parameter=bivector_layout)
    generator = torch.Generator(device=DEVICE).manual_seed(337)
    values = torch.randn(2, 4, vector_layout.dim, dtype=torch.float64, generator=generator)
    weights = torch.randn(4, bivector_layout.dim, dtype=torch.float64, generator=generator) * 0.1

    assert torch.allclose(handle(values, weights), handle._kernel(values, weights), atol=1e-12, rtol=1e-12)
    with pytest.raises(RuntimeError):
        handle(values, weights[:3])


from tests.helpers import action as action_helpers
