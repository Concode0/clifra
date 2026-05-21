import pytest
import torch

from clifra.core.foundation.layout import AlgebraSpec
from clifra.core.planning.layouts import build_product_request
from clifra.core.storage import DispatchPath, StorageMode, resolve_planned_dispatch, resolve_tensor_storage

pytestmark = pytest.mark.unit


def test_tensor_storage_distinguishes_logical_layout_from_physical_mode():
    spec = AlgebraSpec(5, 0, 0)
    vector_layout = spec.layout((1,))
    dense = torch.zeros(2, spec.dim)
    compact = torch.zeros(2, vector_layout.dim)

    dense_storage = resolve_tensor_storage(spec, dense, layout=vector_layout, side="value")
    compact_storage = resolve_tensor_storage(spec, compact, layout=vector_layout, side="value")

    assert dense_storage.layout is vector_layout
    assert dense_storage.mode is StorageMode.DENSE
    assert dense_storage.lane_dim == spec.dim
    assert compact_storage.layout is vector_layout
    assert compact_storage.mode is StorageMode.COMPACT
    assert compact_storage.lane_dim == vector_layout.dim


def test_grade_layout_returns_compact_positions_for_grades():
    spec = AlgebraSpec(4, 0, 0)
    layout = spec.layout((0, 2))

    scalar_positions = layout.positions_for_grades((0,))
    bivector_positions = layout.positions_for_grades((2,))

    assert scalar_positions.tolist() == [0]
    assert len(bivector_positions) == 6
    assert set(bivector_positions.tolist()).isdisjoint(scalar_positions.tolist())


def test_product_request_carries_resolved_operand_storage():
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

    assert request.left_storage.mode is StorageMode.COMPACT
    assert request.right_storage.mode is StorageMode.DENSE
    assert request.left_grades == (1,)
    assert request.right_grades == (2,)
    assert request.output_storage.mode is StorageMode.COMPACT
    assert request.output_grades == (1, 3)


def test_planned_dispatch_resolves_compact_or_dense_output_boundary():
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

    compact_dispatch = resolve_planned_dispatch(request, compact_output=True)
    dense_dispatch = resolve_planned_dispatch(request, compact_output=False)

    assert compact_dispatch.path is DispatchPath.PLANNED_COMPACT
    assert compact_dispatch.output_storage.is_compact
    assert not compact_dispatch.materializes_dense
    assert dense_dispatch.path is DispatchPath.PLANNED_DENSE_OUTPUT
    assert dense_dispatch.output_storage.is_dense
    assert dense_dispatch.materializes_dense
