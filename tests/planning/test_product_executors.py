# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Structural product-planning invariants and selected-route behavior."""

import pytest
import torch

from clifra.core import AlgebraContext
from clifra.core._kernel.configuration import configured_algebra
from clifra.core._kernel.planning.product import (
    assess_product_routes,
    build_grade_product_plan_from_tree,
    count_grade_product_interactions,
)
from clifra.core._kernel.planning.resources import ResourceLimits
from clifra.core._kernel.planning.tree import build_grade_plan_tree
from tests.helpers.policy import PreferRoute
from tests.helpers.small_oracle import SmallCliffordOracle
from tests.planning._grade_plan_helpers import select_product_route

PRODUCT_OPS = (
    "geometric_product",
    "wedge",
    "left_contraction",
    "right_contraction",
    "symmetric_product",
    "commutator_product",
    "anti_commutator_product",
)


@pytest.mark.parametrize("op", PRODUCT_OPS)
def test_combinatorial_interaction_count_matches_lowering(op):
    for signature in ((4, 0, 0), (0, 4, 0), (2, 2, 0), (1, 1, 2), (0, 0, 4)):
        spec = AlgebraContext(*signature).spec
        tree = build_grade_plan_tree(
            spec,
            op=op,
            left_grades=range(5),
            right_grades=range(5),
            output_grades=range(5),
        )
        plan = build_grade_product_plan_from_tree(tree, dtype=torch.float64)
        assert count_grade_product_interactions(tree) == plan.pair_count


def test_product_assessment_uses_combinatorics_without_enumeration(monkeypatch):
    algebra = AlgebraContext(16)

    def fail(*args, **kwargs):
        raise AssertionError("assessment must not enumerate basis interactions")

    monkeypatch.setattr("clifra.core._kernel.planning.product.operation_coefficient", fail)
    assessments = assess_product_routes(
        op="geometric_product",
        left_layout=algebra.layout((1,)),
        right_layout=algebra.layout((2,)),
        output_layout=algebra.layout((1, 3)),
    )
    sparse = next(item for item in assessments if item.route == "sparse")
    assert sparse.facts.work_profile.bulk > 0


@pytest.mark.parametrize(
    "n,op,forbidden_builder",
    [
        (8, "geometric_product", "build_grade_product_plan_from_tree"),
        (4, "wedge", "build_full_table_product_plan_from_request"),
    ],
)
def test_only_selected_product_route_constructs_buffers(n, op, forbidden_builder, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("unselected route constructed buffers")

    monkeypatch.setattr(f"clifra.core._kernel.planning.product.{forbidden_builder}", fail)
    AlgebraContext(n).plan_product(op=op)


def test_sparse_route_survives_infeasible_dense_cartesian_size():
    limits = ResourceLimits(max_lanes=100, max_pairs=2_200)
    algebra = configured_algebra(8, dtype=torch.float64, resource_limits=limits)
    layout, output = algebra.layout((3,)), algebra.layout((6,))
    operation = algebra.plan_product(op="wedge", left=layout, right=layout, output=output)
    assert layout.dim**2 > limits.max_pairs
    assert operation._kernel.route == "sparse"
    assert operation._kernel.pair_count < limits.max_pairs


@pytest.mark.parametrize("role", ["left", "right", "output"])
def test_product_rejects_foreign_layouts(role):
    algebra, foreign = AlgebraContext(3), AlgebraContext(0, 3)
    layouts = {
        "left": algebra.layout((1,)),
        "right": algebra.layout((1,)),
        "output": algebra.layout((0, 2)),
    }
    layouts[role] = foreign.layout((0, 2) if role == "output" else (1,))
    with pytest.raises(ValueError, match="signature"):
        algebra.plan_product(**layouts)


def test_product_accepts_layouts_from_equal_signature_context():
    algebra, peer = AlgebraContext(3), AlgebraContext(3)
    operation = algebra.plan_product(left=peer.layout((1,)), right=peer.layout((1,)), output=peer.layout((0, 2)))
    values = torch.randn(2, 3)
    torch.testing.assert_close(
        operation(values, values),
        algebra.geometric_product(
            values,
            values,
            left=algebra.layout((1,)),
            right=algebra.layout((1,)),
            output=algebra.layout((0, 2)),
        ),
    )


@pytest.mark.parametrize("op", PRODUCT_OPS)
def test_selected_product_route_matches_independent_oracle(op):
    algebra = AlgebraContext(3, 1, 1, dtype=torch.float64)
    oracle = SmallCliffordOracle(3, 1, 1)
    left = torch.randn(2, algebra.dim, dtype=torch.float64)
    right = torch.randn(2, algebra.dim, dtype=torch.float64)
    actual = getattr(algebra, op)(left, right)
    torch.testing.assert_close(actual, oracle.product(left, right, op=op))


def test_sparse_cpu_float64_gather_preserves_broadcast_compact_canonical_and_gradients():
    algebra = configured_algebra(3, 1, 1, dtype=torch.float64, planning_policy=PreferRoute("product", "sparse"))
    left_layout, right_layout, output_layout = (
        algebra.layout((1,)),
        algebra.layout((2,)),
        algebra.layout((1, 3)),
    )
    operation = algebra.plan_product(left=left_layout, right=right_layout, output=output_layout)
    oracle = SmallCliffordOracle(3, 1, 1)
    left = torch.randn(4, 1, left_layout.dim, dtype=torch.float64, requires_grad=True)
    right = torch.randn(1, 3, right_layout.dim, dtype=torch.float64, requires_grad=True)
    expected = oracle.product(
        left,
        right,
        left_indices=left_layout.basis_indices,
        right_indices=right_layout.basis_indices,
        output_indices=output_layout.basis_indices,
    )
    methods = (
        lambda: operation(left, right),
        lambda: operation._kernel.forward(left_layout.full(left), right_layout.full(right)),
    )
    weights = torch.randn_like(expected)
    expected_gradient = torch.autograd.grad((expected * weights).sum(), (left, right))
    for method in methods:
        actual = method()
        torch.testing.assert_close(actual, expected)
        actual_gradient = torch.autograd.grad((actual * weights).sum(), (left, right))
        for found, reference in zip(actual_gradient, expected_gradient):
            torch.testing.assert_close(found, reference)


def test_product_policy_prefers_structural_pruning_and_is_device_independent():
    algebra = AlgebraContext(5)
    layout = algebra.layout()
    routes = {
        select_product_route(
            algebra,
            op="wedge",
            left_layout=layout,
            right_layout=layout,
            output_layout=layout,
            dtype=torch.float32,
            device=device,
        ).route
        for device in ("cpu", "mps", "cuda")
    }
    assert routes == {"sparse"}


def test_product_policy_override_selects_requested_feasible_route():
    algebra = configured_algebra(5, planning_policy=PreferRoute("product", "full_table"))
    assert algebra.plan_product(op="wedge")._kernel.route == "full_table"


def test_product_resource_limit_rejects_structurally_oversized_route():
    algebra = configured_algebra(16, resource_limits=ResourceLimits(max_pairs=64))
    with pytest.raises(ValueError, match="pair/interaction footprint"):
        algebra.plan_product(left=algebra.layout((1,)), right=algebra.layout((1,)), output=algebra.layout((0, 2)))


def test_layout_declaration_is_independent_of_route_limits():
    limits = ResourceLimits(max_lanes=64, max_pairs=1_024)
    algebra = configured_algebra(32, resource_limits=limits)
    layout = algebra.layout((1, 2))
    assert layout.dim > limits.max_lanes
    with pytest.raises(ValueError, match="intermediate lanes"):
        algebra.plan_unary(op="reverse", input=layout)


@pytest.mark.parametrize("n", [32, 63])
def test_high_dimensional_vector_product_uses_declared_grades(n):
    algebra = configured_algebra(n, resource_limits=ResourceLimits(max_lanes=4_096, max_pairs=200_000))
    vector, output = algebra.layout((1,)), algebra.layout((0, 2))
    operation = algebra.plan_product(left=vector, right=vector, output=output)
    assert operation._kernel.output_dim == 1 + n * (n - 1) // 2
    assert operation._kernel.pair_count == n * n


def test_product_reports_torch_bitmask_boundary():
    algebra = configured_algebra(64, resource_limits=ResourceLimits(max_lanes=4_096, max_pairs=200_000))
    with pytest.raises(ValueError, match="up to n=63"):
        algebra.plan_product(left=algebra.layout((1,)), right=algebra.layout((1,)), output=algebra.layout((0, 2)))


def test_selected_product_route_warns_near_limit():
    limits = ResourceLimits(warn_pairs=128, max_pairs=4_096)
    algebra = configured_algebra(10, 4, 2, resource_limits=limits)
    vector = algebra.layout((1,))
    with pytest.warns(RuntimeWarning):
        algebra.plan_product(left=vector, right=vector)
