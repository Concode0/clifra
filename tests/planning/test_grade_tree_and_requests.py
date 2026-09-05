# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from tests.planning._grade_plan_helpers import (
    AlgebraSpec,
    build_grade_plan_tree,
    build_product_request,
    build_unary_request,
    pytest,
    torch,
)

pytestmark = pytest.mark.unit


def test_grade_plan_tree_groups_routes_without_runtime_partition_backend():
    spec = AlgebraSpec(10, 4, 2)
    tree = build_grade_plan_tree(
        spec,
        left_grades=(1, 2),
        right_grades=(1,),
        output_grades=(0, 2),
        op="geometric_product",
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
        op="geometric_product",
    )

    assert request.left_grades == (1,)
    assert request.right_grades == (1,)
    assert request.output_grades == (0, 2)
    assert request.left_uses_compact_storage
    assert request.right_uses_compact_storage


def test_product_request_detects_compact_lane_tensors_from_layout_shape():
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
        op="geometric_product",
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
