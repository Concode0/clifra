# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from clifra.core import AlgebraContext
from clifra.core.layout import AlgebraSpec
from clifra.core.tensors import LaneStorage, TensorContract

pytestmark = pytest.mark.unit


def test_explicit_canonical_product_contracts_preserve_narrow_semantics():
    algebra = AlgebraContext(4)
    vector = algebra.layout((1,))
    output = algebra.layout((0, 2))
    canonical = TensorContract.canonical(vector)
    operation = algebra.plan_product(left=canonical, right=vector, output=TensorContract.canonical(output))
    left = torch.randn(2, algebra.dim)
    right = torch.randn(2, vector.dim)
    expected = algebra.geometric_product(vector.compact(left), right, left=vector, right=vector, output=output)
    torch.testing.assert_close(operation(left, right), output.full(expected))


def test_layout_does_not_infer_canonical_storage_from_width():
    algebra = AlgebraContext(4)
    vector = algebra.layout((1,))
    values = torch.randn(2, algebra.dim)
    with pytest.raises(ValueError, match="compact last dimension must be 4"):
        algebra.reverse(values, input=vector)
    actual = algebra.reverse(values, input=TensorContract.canonical(vector), output=vector)
    torch.testing.assert_close(actual, vector.compact(values))


def test_equal_width_layouts_keep_distinct_algebraic_meanings():
    algebra = AlgebraContext(3)
    values = torch.tensor([1.0, 0.0, 0.0])
    vector, bivector = algebra.layout((1,)), algebra.layout((2,))
    scalar = algebra.layout((0,))
    assert vector.dim == bivector.dim
    torch.testing.assert_close(
        algebra.geometric_product(values, values, left=vector, right=vector, output=scalar), torch.tensor([1.0])
    )
    torch.testing.assert_close(
        algebra.geometric_product(values, values, left=bivector, right=bivector, output=scalar), torch.tensor([-1.0])
    )
    with pytest.raises(ValueError, match="last dimension must be 8"):
        algebra.reverse(values)


def test_tensor_contract_records_declared_layout_and_storage():
    spec = AlgebraSpec(5, 0, 0)
    vector_layout = spec.layout((1,))
    compact = TensorContract.compact(vector_layout)
    canonical = TensorContract.canonical(vector_layout)

    assert compact.layout is vector_layout
    assert compact.storage is LaneStorage.COMPACT
    assert compact.lane_dim == vector_layout.dim
    assert canonical.layout is vector_layout
    assert canonical.storage is LaneStorage.CANONICAL
    assert canonical.lane_dim == spec.dim


def test_tensor_contract_converts_between_compact_and_canonical():
    spec = AlgebraSpec(4, 0, 0)
    layout = spec.layout((0, 2))
    compact_contract = TensorContract.compact(layout)
    canonical_contract = TensorContract.canonical(layout)
    compact_values = torch.randn(3, layout.dim)
    canonical_values = layout.full(compact_values)

    assert torch.allclose(compact_contract.to_canonical(compact_values), canonical_values)
    assert torch.allclose(canonical_contract.to_compact(canonical_values), compact_values)


def test_grade_layout_returns_compact_positions_for_grades():
    spec = AlgebraSpec(4, 0, 0)
    layout = spec.layout((0, 2))

    scalar_positions = layout.positions_for_grades((0,))
    bivector_positions = layout.positions_for_grades((2,))

    assert scalar_positions.tolist() == [0]
    assert len(bivector_positions) == 6
    assert set(bivector_positions.tolist()).isdisjoint(scalar_positions.tolist())


def test_canonical_output_storage_is_consistent_across_operation_families():
    algebra = AlgebraContext(4, dtype=torch.float64)
    vector, bivector = algebra.layout((1,)), algebra.layout((2,))
    vectors = torch.randn(3, vector.dim, dtype=torch.float64)
    bivectors = torch.randn(3, bivector.dim, dtype=torch.float64) * 0.1
    cases = (
        (
            lambda output: algebra.wedge(vectors, vectors, left=vector, right=vector, output=output),
            bivector,
        ),
        (lambda output: algebra.reverse(vectors, input=vector, output=output), vector),
        (
            lambda output: algebra.pseudoscalar_product(vectors, input=vector, output=output),
            algebra.layout((3,)),
        ),
        (lambda output: algebra.bivector_exp(bivectors, input=bivector, output=output), algebra.layout()),
    )
    for operation, layout in cases:
        compact = operation(layout)
        canonical = operation(TensorContract.canonical(layout))
        assert canonical.shape[-1] == algebra.dim
        torch.testing.assert_close(canonical, layout.full(compact))
