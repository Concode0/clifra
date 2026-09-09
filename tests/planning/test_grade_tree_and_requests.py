# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import torch

from clifra.core._kernel.planning import product
from tests.planning._grade_plan_helpers import (
    AlgebraSpec,
    build_grade_plan_tree,
    pytest,
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
    )

    assert tree.output_grades == (0, 2)
    assert [(path.left_grade, path.right_grade, path.output_grades) for path in tree.paths] == [
        (1, 1, (0, 2)),
    ]
    assert tree.path_count == 1
    assert tree.estimated_pairs == 16 * 16


def test_product_lowering_chunks_temporary_pairs_without_changing_buffers(monkeypatch):
    tree = build_grade_plan_tree(AlgebraSpec(4, 1, 1), left_grades=(2,), right_grades=(2,))
    expected = product.build_grade_product_plan_from_tree(tree, dtype=torch.float64)
    original = product._operation_coefficients
    temporary_pairs = []

    def record(left, right, output_indices, **kwargs):
        temporary_pairs.append(output_indices.numel())
        return original(left, right, output_indices, **kwargs)

    monkeypatch.setattr(product, "_PRODUCT_CHUNK_PAIRS", 30)
    monkeypatch.setattr(product, "_operation_coefficients", record)
    actual = product.build_grade_product_plan_from_tree(tree, dtype=torch.float64)
    assert len(temporary_pairs) > 1
    assert max(temporary_pairs) <= 30
    assert sum(temporary_pairs) == tree.estimated_pairs
    for name, value in vars(expected).items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(getattr(actual, name), value)
