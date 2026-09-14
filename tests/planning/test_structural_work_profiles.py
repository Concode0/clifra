"""Planning-visible work structure without a calibrated route score."""

import pytest
import torch

from clifra.core import AlgebraContext, AlgebraSpec
from clifra.core._kernel.planning.action import _direct_grade_counts
from clifra.core._kernel.planning.exp import taylor_layouts
from clifra.core._kernel.planning.product import assess_product_routes, count_grade_product_interactions
from clifra.core._kernel.planning.tree import build_grade_plan_tree
from clifra.core._kernel.planning.work import action_lift_profile
from clifra.core._kernel.providers import BuiltinProvider, action_execution_request, exp_execution_request


def test_product_basis_keeps_bulk_and_output_separate():
    spec = AlgebraSpec(4)
    full = spec.full_layout()
    routes = {
        item.route: item
        for item in assess_product_routes(op="wedge", left_layout=full, right_layout=full, output_layout=full)
    }
    sparse, dense = routes["sparse"].facts, routes["full_table"].facts
    tree = build_grade_plan_tree(
        spec, op="wedge", left_grades=full.grades, right_grades=full.grades, output_grades=full.grades
    )
    assert sparse.work_profile.bulk == count_grade_product_interactions(tree)
    assert sparse.work_profile.output == full.dim
    assert dense.work_profile.bulk == spec.dim**2
    assert dense.work_profile.output == spec.dim
    assert sparse.work_profile.bulk < dense.work_profile.bulk

    vector, scalar = spec.layout((1,)), spec.layout((0,))
    scalar_facts = assess_product_routes(
        op="geometric_product", left_layout=vector, right_layout=vector, output_layout=scalar
    )[1].facts
    assert (scalar_facts.work_profile.bulk, scalar_facts.work_profile.output) == (4, 1)


def test_exp_profiles_keep_selected_products_and_taylor_branches():
    algebra = AlgebraContext(4)
    spec = algebra.spec
    output = spec.layout((0, 2, 4))
    request = exp_execution_request(spec, "cpu", torch.float32, output, planner=algebra._planner)

    closed = BuiltinProvider(("bivector_exp", "closed")).assess(request).preparation.facts.work_profile
    assert len(closed.closed().products) == 3
    assert all(call.profile.route == "sparse" for call in closed.closed().products)

    matrix = BuiltinProvider(("bivector_exp", "left_matrix_exp")).assess(request).preparation.facts.work_profile
    equation = matrix.left_matrix_exp()
    assert equation.products[0].calls == 1 << (spec.n - 1)
    assert equation.matrix_exp_order == (1 << (spec.n - 1)) ** 3

    taylor = BuiltinProvider(("bivector_exp", "taylor")).assess(request).preparation.facts.work_profile
    plain, scaled = taylor.taylor_plain(), taylor.taylor_scaled(3)
    assert len(plain.products) == len(taylor_layouts(spec, output, 12)) - 1
    assert len(scaled.products) == len(plain.products) + 1
    assert scaled.products[-1].calls == 3
    assert scaled.products[-1].profile.route == "sparse"
    assert scaled.elementwise_cells == sum(taylor.scaled_stage_widths) + 3 * taylor.even_width + output.dim
    with pytest.raises(ValueError, match="runtime integer"):
        taylor.taylor_scaled(0)


def test_action_lift_equations_do_not_select_an_implementation():
    profile = action_lift_profile(4, (0, 2, 4), (0, 2, 4))
    compound, hybrid, direct = profile.compound(), profile.hybrid((2,)), profile.direct()
    assert compound.full_matrix_cells == compound.dense_matvec_cells == 8**2
    assert compound.direct_terms == 0
    assert hybrid.full_matrix_cells == hybrid.dense_matvec_cells == 0
    assert hybrid.direct_terms == _direct_grade_counts(4, 2)[0]
    assert hybrid.direct_reductions == 50
    assert hybrid.block_matvec_cells == 1  # remaining grade-4 block
    assert direct.compound_minor_work == direct.block_matvec_cells == 0
    with pytest.raises(ValueError, match="unique available"):
        profile.hybrid(())
    vector_only = action_lift_profile(4, (1,), (1,)).compound()
    assert vector_only.full_matrix_cells == 0
    assert vector_only.dense_matvec_cells == 4**2


def test_action_profiles_retain_children_and_dense_full_work():
    algebra = AlgebraContext(4)
    full, parameter = algebra.layout(), algebra.layout((2,))
    request = action_execution_request(
        algebra, "versor", grade=2, input_layout=full, output_layout=full, parameter_layout=parameter
    )
    vector = BuiltinProvider(("action", "vector_matrix")).assess(request).preparation.facts.work_profile
    rotor = BuiltinProvider(("action", "rotor_product")).assess(request).preparation.facts.work_profile
    dense = BuiltinProvider(("action", "full_action_matrix")).assess(request).preparation.facts.work_profile

    assert vector.generator_terms == 4 * 3
    assert vector.vector_matrix_exp_order == 4**3
    assert vector.lift is not None
    assert len(rotor.product_children) == 2
    assert all(child.route == "sparse" for child in rotor.product_children)
    assert rotor.exponential_child.route == dense.exponential_child.route == "left_matrix_exp"
    assert dense.full_action_matrix_order == algebra.spec.dim**3
    assert dense.full_action_cells == algebra.spec.dim**2
