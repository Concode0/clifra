# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from clifra.core._kernel.planning.layouts import ProductRequest
from clifra.core.tensors import TensorContract
from tests.planning._grade_plan_helpers import (
    DEVICE,
    AlgebraContext,
    FullTableProductExecutor,
    GradeProductExecutor,
    PreferRoute,
    ResourceLimits,
    SmallCliffordOracle,
    _grade_only_input,
    _oracle_for,
    _product_method_name,
    _sparse_pairwise_product_reference,
    build_grade_product_plan,
    expand_output_grades,
    pytest,
    select_product_route,
    torch,
)

pytestmark = pytest.mark.unit


def _compact_product_request(algebra, *, op, left_layout, right_layout, output_layout):
    return ProductRequest.compact(
        algebra._planner.spec,
        op=op,
        left_layout=left_layout,
        right_layout=right_layout,
        output_layout=output_layout,
        dtype=algebra.dtype,
        device=algebra.device,
    )


@pytest.mark.parametrize("warm_cache", [False, True])
@pytest.mark.parametrize("foreign_side", ["left", "right", "output"])
def test_product_request_rejects_foreign_layouts_before_cache_lookup(
    foreign_side,
    warm_cache,
):
    algebra = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float32)
    foreign = AlgebraContext(0, 3, 0, device=DEVICE, dtype=torch.float32)
    layouts = {
        "left_layout": algebra.layout((1,)),
        "right_layout": algebra.layout((1,)),
        "output_layout": algebra.layout((0,)),
    }
    if warm_cache:
        algebra._planner.product_executor(_compact_product_request(algebra, op="geometric_product", **layouts))
    cache_before = tuple(algebra._planner._product_executors.items())
    layouts[f"{foreign_side}_layout"] = foreign.layout((0,) if foreign_side == "output" else (1,))

    with pytest.raises(ValueError, match=rf"{foreign_side}_layout signature .* does not match algebra signature"):
        algebra._planner.product_executor(_compact_product_request(algebra, op="geometric_product", **layouts))

    assert tuple(algebra._planner._product_executors.items()) == cache_before


def test_product_request_rejects_foreign_tensor_contracts_at_construction():
    algebra = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float32)
    foreign = AlgebraContext(0, 3, 0, device=DEVICE, dtype=torch.float32)
    own_layouts = {
        "left_layout": algebra.layout((1,)),
        "right_layout": algebra.layout((1,)),
        "output_layout": algebra.layout((0,)),
    }
    algebra._planner.product_executor(_compact_product_request(algebra, op="geometric_product", **own_layouts))
    cache_before = tuple(algebra._planner._product_executors.items())
    with pytest.raises(ValueError, match="left_layout signature .* does not match algebra signature"):
        ProductRequest(
            spec=algebra._planner.spec,
            op="geometric_product",
            left=TensorContract.compact(foreign.layout((1,))),
            right=TensorContract.compact(foreign.layout((1,))),
            output=TensorContract.compact(foreign.layout((0,))),
            dtype=torch.float32,
            device=torch.device(DEVICE),
        )

    assert tuple(algebra._planner._product_executors.items()) == cache_before


@pytest.mark.parametrize("cache", [False, True])
def test_product_executor_rejects_foreign_request_signature(cache):
    algebra = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float32)
    foreign = AlgebraContext(0, 3, 0, device=DEVICE, dtype=torch.float32)
    values = torch.zeros(1, 3)
    request = foreign._planner.product_request(
        values,
        values,
        left_layout=foreign.layout((1,)),
        right_layout=foreign.layout((1,)),
        output_layout=foreign.layout((0,)),
    )

    cache_before = tuple(algebra._planner._product_executors.items())

    with pytest.raises(ValueError, match="request signature .* does not match algebra signature"):
        algebra._planner.product_executor(request, cache=cache)

    assert tuple(algebra._planner._product_executors.items()) == cache_before


def test_product_executor_accepts_layouts_from_equal_signature_algebra():
    algebra = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float32)
    peer = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float32)
    own = algebra._planner.product_executor(
        _compact_product_request(
            algebra,
            op="geometric_product",
            left_layout=algebra.layout((1,)),
            right_layout=algebra.layout((1,)),
            output_layout=algebra.layout((0,)),
        )
    )

    shared = algebra._planner.product_executor(
        _compact_product_request(
            algebra,
            op="geometric_product",
            left_layout=peer.layout((1,)),
            right_layout=peer.layout((1,)),
            output_layout=peer.layout((0,)),
        )
    )

    assert shared is own


def test_planner_conversion_rejects_mutually_consistent_foreign_layouts():
    algebra = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float32)
    foreign = AlgebraContext(0, 3, 0, device=DEVICE, dtype=torch.float32)
    values = torch.zeros(2, foreign.layout((1,)).dim)

    with pytest.raises(ValueError, match="source_layout signature .* does not match algebra signature"):
        algebra._planner.convert_values(
            values,
            source_layout=foreign.layout((1,)),
            target_layout=foreign.layout((1, 2)),
        )


@pytest.mark.parametrize(
    "op",
    [
        "geometric_product",
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
        op="geometric_product",
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
        op="geometric_product",
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


@pytest.mark.parametrize(
    "op", ["geometric_product", "wedge", "symmetric_product", "commutator_product", "anti_commutator_product"]
)
def test_planner_full_table_executor_matches_small_oracle_full_layout_product(op):
    context = AlgebraContext(4, 1, 0, device=DEVICE, dtype=torch.float64)
    oracle = _oracle_for(context)
    full_layout = context.layout()
    generator = torch.Generator(device=DEVICE).manual_seed(199)
    left = torch.randn(3, context.dim, dtype=torch.float64, generator=generator)
    right = torch.randn(3, context.dim, dtype=torch.float64, generator=generator)

    executor = context._planner.product_executor(
        _compact_product_request(
            context,
            op=op,
            left_layout=full_layout,
            right_layout=full_layout,
            output_layout=full_layout,
        )
    )
    actual = getattr(context, _product_method_name(op))(left, right)
    expected = oracle.product(left, right, op=op)

    assert isinstance(executor, FullTableProductExecutor)
    assert executor.route == "full_table"
    assert actual.shape[-1] == context.dim
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_product_executor_policy_selects_sparse_for_pruned_full_layout_wedge():
    context = AlgebraContext(6, 0, 0, device=DEVICE, dtype=torch.float64)
    oracle = _oracle_for(context)
    full_layout = context.layout()
    generator = torch.Generator(device=DEVICE).manual_seed(1019)
    left = torch.randn(2, context.dim, dtype=torch.float64, generator=generator)
    right = torch.randn(2, context.dim, dtype=torch.float64, generator=generator)

    decision = select_product_route(
        context,
        op="wedge",
        left_layout=full_layout,
        right_layout=full_layout,
        output_layout=full_layout,
        dtype=torch.float64,
        device=DEVICE,
    )
    executor = context._planner.product_executor(
        _compact_product_request(
            context,
            op="wedge",
            left_layout=full_layout,
            right_layout=full_layout,
            output_layout=full_layout,
        )
    )

    assert decision.route == "sparse"
    assert isinstance(executor, GradeProductExecutor)
    assert torch.allclose(context.wedge(left, right), oracle.product(left, right, op="wedge"), atol=1e-12, rtol=1e-12)


def test_product_executor_policy_override_can_force_full_table_full_layout_wedge():
    policy = PreferRoute("product", "full_table")
    context = configured_algebra(6, 0, 0, device=DEVICE, dtype=torch.float64, planning_policy=policy)
    full_layout = context.layout()

    decision = select_product_route(
        context,
        op="wedge",
        left_layout=full_layout,
        right_layout=full_layout,
        output_layout=full_layout,
        dtype=torch.float64,
        device=DEVICE,
    )
    executor = context._planner.product_executor(
        _compact_product_request(
            context,
            op="wedge",
            left_layout=full_layout,
            right_layout=full_layout,
            output_layout=full_layout,
        )
    )

    assert decision.route == "full_table"
    assert isinstance(executor, FullTableProductExecutor)


def test_product_executor_policy_uses_backend_coefficients_without_benchmark_rows():
    context = AlgebraContext(5, 0, 0, device=DEVICE, dtype=torch.float32)
    full_layout = context.layout()

    cpu_decision = select_product_route(
        context,
        op="geometric_product",
        left_layout=full_layout,
        right_layout=full_layout,
        output_layout=full_layout,
        dtype=torch.float32,
        device="cpu",
    )
    mps_decision = select_product_route(
        context,
        op="geometric_product",
        left_layout=full_layout,
        right_layout=full_layout,
        output_layout=full_layout,
        dtype=torch.float32,
        device="mps",
    )

    assert cpu_decision.route == "full_table"
    assert mps_decision.route == "sparse"


def test_direct_product_executor_obeys_static_pair_limits():
    limits = ResourceLimits(warn_lanes=512, max_lanes=512, warn_pairs=512, max_pairs=64)
    algebra = configured_algebra(16, 0, 0, device=DEVICE, dtype=torch.float32, resource_limits=limits)

    with pytest.raises(ValueError, match="basis interactions"):
        algebra.plan_product(
            op="geometric_product", left=algebra.layout((1,)), right=algebra.layout((1,)), output=algebra.layout((0, 2))
        )


from clifra.core._kernel.configuration import configured_algebra
