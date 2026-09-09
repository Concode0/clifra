# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from clifra.core import AlgebraContext
from clifra.core._kernel.contracts import resolve_contract
from clifra.core._kernel.planning.unary import UnaryRequest
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


def test_contract_resolution_ignores_obsolete_grade_defaults():
    algebra = AlgebraContext(3)
    algebra._default_grades = (1,)
    algebra.default_layout = algebra.layout((2,))
    assert resolve_contract(algebra).layout == algebra.layout()


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


def test_resolve_contract_reports_role_and_grade_disagreement():
    spec = AlgebraSpec(3, 0, 0)
    foreign_spec = AlgebraSpec(0, 3, 0)

    with pytest.raises(ValueError, match="input_layout signature .* does not match algebra signature"):
        resolve_contract(spec, layout=foreign_spec.layout((1,)), name="input_layout")
    with pytest.raises(ValueError, match="input_layout and input_grades disagree"):
        resolve_contract(spec, layout=spec.layout((1,)), grades=(2,), name="input_layout")


def test_unary_request_rejects_inconsistent_contracts_at_construction():
    spec = AlgebraSpec(3, 0, 0)
    foreign_spec = AlgebraSpec(0, 3, 0)
    foreign = TensorContract.compact(foreign_spec.layout((1,)))

    with pytest.raises(ValueError, match="input_layout signature .* does not match algebra signature"):
        UnaryRequest(
            spec=spec,
            op="reverse",
            input=foreign,
            output=foreign,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )


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
