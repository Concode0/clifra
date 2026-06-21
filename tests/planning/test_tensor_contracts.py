# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from clifra.core.foundation.layout import AlgebraSpec
from clifra.core.planning.layouts import build_product_request
from clifra.core.runtime.tensors import LaneStorage, TensorContract, compact_pair_values, infer_contract

pytestmark = pytest.mark.unit


def test_tensor_contract_records_declared_layout_and_storage():
    spec = AlgebraSpec(5, 0, 0)
    vector_layout = spec.layout((1,))
    compact = TensorContract.compact(spec, vector_layout)
    canonical = TensorContract.canonical(spec, vector_layout)

    assert compact.layout is vector_layout
    assert compact.storage is LaneStorage.COMPACT
    assert compact.lane_dim == vector_layout.dim
    assert canonical.layout is vector_layout
    assert canonical.storage is LaneStorage.CANONICAL
    assert canonical.lane_dim == spec.dim


def test_tensor_contract_converts_between_compact_and_canonical():
    spec = AlgebraSpec(4, 0, 0)
    layout = spec.layout((0, 2))
    compact_contract = TensorContract.compact(spec, layout)
    canonical_contract = TensorContract.canonical(spec, layout)
    compact_values = torch.randn(3, layout.dim)
    canonical_values = layout.full(compact_values)

    assert torch.allclose(compact_contract.to_canonical(compact_values), canonical_values)
    assert torch.allclose(canonical_contract.to_compact(canonical_values), compact_values)


def test_infer_contract_detects_compact_and_canonical_storage():
    spec = AlgebraSpec(6, 0, 0)
    layout = spec.layout((1,))
    compact = torch.zeros(2, layout.dim)
    canonical = torch.zeros(2, spec.dim)

    compact_contract = infer_contract(spec, compact, layout=layout, side="value")
    canonical_contract = infer_contract(spec, canonical, layout=layout, side="value")

    assert compact_contract.storage is LaneStorage.COMPACT
    assert canonical_contract.storage is LaneStorage.CANONICAL


def test_grade_layout_returns_compact_positions_for_grades():
    spec = AlgebraSpec(4, 0, 0)
    layout = spec.layout((0, 2))

    scalar_positions = layout.positions_for_grades((0,))
    bivector_positions = layout.positions_for_grades((2,))

    assert scalar_positions.tolist() == [0]
    assert len(bivector_positions) == 6
    assert set(bivector_positions.tolist()).isdisjoint(scalar_positions.tolist())


def test_product_request_carries_resolved_tensor_contracts():
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
        right_storage=LaneStorage.CANONICAL,
        op="gp",
    )

    assert request.left.storage is LaneStorage.COMPACT
    assert request.right.storage is LaneStorage.CANONICAL
    assert request.left_grades == (1,)
    assert request.right_grades == (2,)
    assert request.output.storage is LaneStorage.COMPACT
    assert request.output_grades == (1, 3)


def test_product_request_can_declare_canonical_output_storage():
    spec = AlgebraSpec(4, 0, 0)
    vector_layout = spec.layout((1,))
    values = torch.zeros(2, vector_layout.dim)
    request = build_product_request(
        spec,
        values,
        values,
        left_layout=vector_layout,
        right_layout=vector_layout,
        output_storage=LaneStorage.CANONICAL,
        op="gp",
    )

    assert request.output.uses_canonical_storage
    assert request.output.lane_dim == spec.dim


def test_compact_pair_values_aligns_to_grade_union_without_full_materialization():
    spec = AlgebraSpec(4, 0, 0)
    vector_layout = spec.layout((1,))
    bivector_layout = spec.layout((2,))
    resolved_layout = spec.layout((1, 2))
    vector_values = torch.ones(2, vector_layout.dim)
    bivector_values = torch.full((2, bivector_layout.dim), 2.0)

    aligned_vector, aligned_bivector, resolved = compact_pair_values(
        spec,
        vector_values,
        bivector_values,
        left_layout=vector_layout,
        right_layout=bivector_layout,
    )

    assert resolved == resolved_layout
    assert aligned_vector.shape[-1] == resolved_layout.dim
    assert aligned_bivector.shape[-1] == resolved_layout.dim
    vector_positions = resolved_layout.positions_for_grades((1,))
    bivector_positions = resolved_layout.positions_for_grades((2,))
    assert torch.allclose(torch.index_select(aligned_vector, -1, vector_positions), vector_values)
    assert torch.allclose(torch.index_select(aligned_vector, -1, bivector_positions), torch.zeros_like(bivector_values))
    assert torch.allclose(torch.index_select(aligned_bivector, -1, vector_positions), torch.zeros_like(vector_values))
    assert torch.allclose(torch.index_select(aligned_bivector, -1, bivector_positions), bivector_values)
