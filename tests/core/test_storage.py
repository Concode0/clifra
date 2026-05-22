import pytest
import torch

from clifra.core.foundation.layout import AlgebraSpec
from clifra.core.planning.layouts import build_product_request
from clifra.core.storage import ExecutorPath, LaneFormat, resolve_output_boundary, resolve_value_layout

pytestmark = pytest.mark.unit


def test_value_layout_distinguishes_logical_layout_from_lane_format():
    spec = AlgebraSpec(5, 0, 0)
    vector_layout = spec.layout((1,))
    full = torch.zeros(2, spec.dim)
    active = torch.zeros(2, vector_layout.dim)

    full_value = resolve_value_layout(spec, full, layout=vector_layout, side="value")
    active_value = resolve_value_layout(spec, active, layout=vector_layout, side="value")

    assert full_value.layout is vector_layout
    assert full_value.lane_format is LaneFormat.FULL
    assert full_value.lane_dim == spec.dim
    assert active_value.layout is vector_layout
    assert active_value.lane_format is LaneFormat.ACTIVE
    assert active_value.lane_dim == vector_layout.dim


def test_grade_layout_returns_compact_positions_for_grades():
    spec = AlgebraSpec(4, 0, 0)
    layout = spec.layout((0, 2))

    scalar_positions = layout.positions_for_grades((0,))
    bivector_positions = layout.positions_for_grades((2,))

    assert scalar_positions.tolist() == [0]
    assert len(bivector_positions) == 6
    assert set(bivector_positions.tolist()).isdisjoint(scalar_positions.tolist())


def test_product_request_carries_resolved_operand_layouts():
    spec = AlgebraSpec(6, 0, 0)
    vector_layout = spec.layout((1,))
    bivector_layout = spec.layout((2,))
    left = torch.zeros(2, vector_layout.dim)
    right = torch.zeros(2, spec.dim)

    request = build_product_request(
        spec,
        left,
        right,
        left_layout=vector_layout,
        right_layout=bivector_layout,
        op="gp",
    )

    assert request.left_value.lane_format is LaneFormat.ACTIVE
    assert request.right_value.lane_format is LaneFormat.FULL
    assert request.left_grades == (1,)
    assert request.right_grades == (2,)
    assert request.output_value.lane_format is LaneFormat.ACTIVE
    assert request.output_grades == (1, 3)


def test_planned_boundary_resolves_active_or_full_output_lanes():
    spec = AlgebraSpec(4, 0, 0)
    vector_layout = spec.layout((1,))
    values = torch.zeros(2, vector_layout.dim)
    request = build_product_request(
        spec,
        values,
        values,
        left_layout=vector_layout,
        right_layout=vector_layout,
        op="gp",
    )

    active_boundary = resolve_output_boundary(request, active_output=True)
    full_boundary = resolve_output_boundary(request, active_output=False)

    assert active_boundary.path is ExecutorPath.PLANNED_ACTIVE
    assert active_boundary.output_value.uses_active_lanes
    assert not active_boundary.materializes_full
    assert full_boundary.path is ExecutorPath.PLANNED_FULL
    assert full_boundary.output_value.uses_full_lanes
    assert full_boundary.materializes_full
