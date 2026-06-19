# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from clifra.core.config import make_algebra
from clifra.core.execution.action import (
    FullSandwichActionExecutor,
    apply_graded_linear_action,
    apply_multi_graded_linear_action,
)
from clifra.core.execution.handles import (
    FullSandwichActionHandle,
    MultiVersorActionHandle,
    PairedBivectorActionHandle,
    ProductPlanHandle,
    UnaryPlanHandle,
    VersorActionHandle,
)
from clifra.core.execution.metric import NormSquaredExecutor
from clifra.core.execution.permutation import DualExecutor
from clifra.core.execution.product import FullTableProductExecutor, GradeProductExecutor
from clifra.core.foundation.basis import (
    basis_count_for_grades,
    basis_index_tuple_for_grades,
    basis_indices_for_grades,
    expand_output_grades,
    geometric_product_output_grades,
    operation_coefficient,
    product_output_grades,
)
from clifra.core.foundation.layout import AlgebraSpec
from clifra.core.planning.flow import GradeFlow
from clifra.core.planning.layouts import build_product_request
from clifra.core.planning.planner import GradePlanner
from clifra.core.planning.policy import PlanningLimits, ProductExecutionPolicy, estimate_product_executor_cost
from clifra.core.planning.product import build_grade_product_plan
from clifra.core.planning.tree import build_grade_plan_tree
from clifra.core.planning.unary import build_unary_request
from clifra.core.runtime.algebra import AlgebraContext
from clifra.core.runtime.tensors import LaneStorage
from tests.helpers.small_oracle import SmallCliffordOracle

pytestmark = pytest.mark.unit

DEVICE = "cpu"


def _mps_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def _oracle_for(algebra) -> SmallCliffordOracle:
    return SmallCliffordOracle(algebra.p, algebra.q, algebra.r)


def _oracle_sandwich_action_matrices(
    oracle: SmallCliffordOracle,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    basis = torch.eye(oracle.dim, dtype=left.dtype, device=left.device)
    matrices = []
    for item in range(left.shape[0]):
        left_values = left[item].expand(oracle.dim, oracle.dim)
        right_values = right[item].expand(oracle.dim, oracle.dim)
        transformed = oracle.product(oracle.product(left_values, basis), right_values)
        matrices.append(transformed.transpose(0, 1))
    return torch.stack(matrices)


def _grade_only_input(algebra, batch: int, grades: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    mv = torch.zeros(batch, algebra.dim, dtype=torch.float64)
    indices = basis_indices_for_grades(algebra.n, grades, device=DEVICE)
    mv[:, indices] = torch.randn(batch, indices.numel(), dtype=torch.float64, generator=generator) * 0.1
    return mv


def _sparse_pairwise_product_reference(
    executor: GradeProductExecutor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    prefix = torch.broadcast_shapes(left.shape[:-2], right.shape[:-2])
    left = left.expand(*prefix, *left.shape[-2:])
    right = right.expand(*prefix, *right.shape[-2:])
    left_terms = torch.index_select(left, -1, executor.left_compact_positions)
    right_terms = torch.index_select(right, -1, executor.right_compact_positions)
    terms = left_terms.unsqueeze(-2) * right_terms.unsqueeze(-3) * executor.coefficients
    output = terms.new_zeros(*terms.shape[:-1], executor.output_dim)
    return output.index_add(-1, executor.output_positions, terms)


def test_grade_expansion_for_common_high_dim_paths():
    assert geometric_product_output_grades(1, 1, 16) == (0, 2)
    assert geometric_product_output_grades(2, 1, 16) == (1, 3)
    assert product_output_grades(2, 1, 16, op="wedge") == (3,)
    assert product_output_grades(2, 1, 16, op="commutator") == (1,)
    assert product_output_grades(2, 1, 16, op="anti_commutator") == (3,)
    assert expand_output_grades((0, 2), (1,), 16, op="gp") == (1, 3)
    assert expand_output_grades((1,), (1,), 16, op="wedge") == (2,)
    assert expand_output_grades((1,), (1,), 16, op="gp", project_grades=(0,)) == (0,)


def test_operation_coefficients_keep_wedge_as_exterior_product():
    # e12 and e3 commute, so the antisymmetric formula would vanish.  The
    # exterior product is instead the grade-sum part of the geometric product.
    assert operation_coefficient(3, 4, 3, 0, 0, "wedge") == 1.0
    assert operation_coefficient(3, 4, 3, 0, 0, "commutator") == 0.0
    assert operation_coefficient(3, 4, 3, 0, 0, "anti_commutator") == 2.0


def test_basis_indices_for_grades_are_combinatorial_and_high_dimensional():
    assert basis_index_tuple_for_grades(4, (1, 2)) == tuple(
        index for index in range(1 << 4) if index.bit_count() in {1, 2}
    )
    high = basis_index_tuple_for_grades(32, (1, 2))
    assert len(high) == basis_count_for_grades(32, (1, 2))
    assert high[0] == 1
    assert high[-1] == (1 << 31) | (1 << 30)


def test_basis_tensorization_reports_int64_bitmask_boundary():
    with pytest.raises(ValueError, match="torch.long basis bitmasks"):
        basis_indices_for_grades(64, (1,))


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


def test_product_request_detects_active_lane_tensors_from_layout_shape():
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


def test_grade_layout_compact_full_round_trip():
    spec = AlgebraSpec(4, 1, 1)
    layout = spec.layout((0, 2))
    reference = torch.randn(2, spec.dim, dtype=torch.float64, generator=torch.Generator().manual_seed(97))

    values = layout.compact(reference)
    materialized = layout.full(values)

    assert values.shape[-1] == layout.dim
    assert torch.allclose(materialized[..., layout.indices_tensor(device=reference.device)], values)
    outside = torch.ones(spec.dim, dtype=torch.bool)
    outside[layout.indices_tensor()] = False
    assert materialized[..., outside].abs().sum().item() == 0.0


@pytest.mark.parametrize(
    "op",
    ["gp", "wedge", "inner", "commutator", "anti_commutator", "left_contraction", "right_contraction"],
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


def test_wedge_full_and_planned_paths_are_exterior_product_for_higher_grades():
    algebra = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float64)
    oracle = _oracle_for(algebra)
    spec = AlgebraSpec.from_algebra(algebra)
    layout_2 = spec.layout((2,))
    layout_1 = spec.layout((1,))
    layout_3 = spec.layout((3,))

    e12 = torch.zeros(1, algebra.dim, dtype=torch.float64)
    e12[0, 3] = 1.0
    e3 = torch.zeros(1, algebra.dim, dtype=torch.float64)
    e3[0, 4] = 1.0
    expected = torch.zeros_like(e12)
    expected[0, 7] = 1.0

    full = algebra.wedge(e12, e3)
    compact = algebra.wedge(
        layout_2.compact(e12),
        layout_1.compact(e3),
        left_layout=layout_2,
        right_layout=layout_1,
        output_grades=(3,),
        left_storage=LaneStorage.COMPACT,
        right_storage=LaneStorage.COMPACT,
        output_storage=LaneStorage.COMPACT,
    )

    assert torch.allclose(full, oracle.product(e12, e3, op="wedge"), atol=1e-12, rtol=1e-12)
    assert torch.allclose(full, expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(compact, layout_3.compact(expected), atol=1e-12, rtol=1e-12)


def test_wedge_chains_as_iterative_exterior_product():
    algebra = AlgebraContext(4, 0, 0, device=DEVICE, dtype=torch.float64)
    layout_1 = algebra.layout((1,))
    layout_2 = algebra.layout((2,))
    layout_3 = algebra.layout((3,))
    e1 = torch.zeros(1, algebra.dim, dtype=torch.float64)
    e2 = torch.zeros(1, algebra.dim, dtype=torch.float64)
    e3 = torch.zeros(1, algebra.dim, dtype=torch.float64)
    e1[0, 1] = 1.0
    e2[0, 2] = 1.0
    e3[0, 4] = 1.0
    expected = torch.zeros_like(e1)
    expected[0, 7] = 1.0

    e12 = algebra.wedge(
        layout_1.compact(e1),
        layout_1.compact(e2),
        left_layout=layout_1,
        right_layout=layout_1,
        output_layout=layout_2,
    )
    actual = algebra.wedge(
        e12,
        layout_1.compact(e3),
        left_layout=layout_2,
        right_layout=layout_1,
        output_layout=layout_3,
    )

    assert torch.allclose(actual, layout_3.compact(expected), atol=1e-12, rtol=1e-12)


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


def test_product_plan_owns_active_lane_position_buffers():
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
    left = algebra.exp(-0.5 * bivectors, input_layout=bivector_layout, output_layout=layout)
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
    left = algebra.exp(-0.5 * bivectors, input_layout=bivector_layout, output_layout=full_layout)
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
    batch_left = algebra.exp(-0.5 * batch_bivectors, input_layout=bivector_layout, output_layout=full_layout)
    batch_right = algebra.reverse(batch_left, input_layout=full_layout, output_layout=full_layout)
    rotor_left = algebra.exp(-0.5 * rotor_bivectors, input_layout=bivector_layout, output_layout=full_layout)
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
    left = algebra.exp(-0.5 * left_bivectors, input_layout=bivector_layout, output_layout=full_layout)
    right = algebra.reverse(left, input_layout=full_layout, output_layout=full_layout)
    batch_left = algebra.exp(-0.5 * batch_bivectors, input_layout=bivector_layout, output_layout=full_layout)
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

    left = context.exp(-0.5 * weights, input_layout=parameter_layout, output_layout=full_layout)
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


def test_algebra_plan_product_returns_active_lane_handle():
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


def test_active_projected_product_returns_declared_output_lanes():
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


@pytest.mark.parametrize("op", ["gp", "wedge", "inner", "commutator", "anti_commutator"])
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


def test_active_geometric_product_stays_active_in_high_dimensions():
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


def test_active_binary_products_do_not_unwrap_full_tensors():
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


def _product_method_name(op: str) -> str:
    if op == "gp":
        return "geometric_product"
    if op == "inner":
        return "inner_product"
    if op == "anti_commutator":
        return "anti_commutator"
    if op == "left_contraction":
        return "left_contraction"
    if op == "right_contraction":
        return "right_contraction"
    return op
