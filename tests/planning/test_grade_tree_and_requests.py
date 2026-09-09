# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from clifra.core import AlgebraSpec
from clifra.core._kernel.execution.product import GradeProductExecutor
from clifra.core._kernel.planning import product
from clifra.core._kernel.planning.tree import build_grade_plan_tree

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
    assert len(tree.paths) == 1


def test_product_lowering_chunks_temporary_pairs_without_changing_buffers(monkeypatch):
    tree = build_grade_plan_tree(AlgebraSpec(4, 1, 1), left_grades=(2,), right_grades=(2,))
    expected = product.build_grade_product_plan_from_tree(tree, dtype=torch.float64)
    original = product._cat_long_chunks
    chunk_sizes = []

    def record(chunks):
        chunk_sizes.extend(chunk.numel() for chunk in chunks)
        return original(chunks)

    monkeypatch.setattr(product, "_PRODUCT_CHUNK_PAIRS", 30)
    monkeypatch.setattr(product, "_cat_long_chunks", record)
    actual = product.build_grade_product_plan_from_tree(tree, dtype=torch.float64)
    assert len(chunk_sizes) > 1
    assert max(chunk_sizes) <= 30
    left = torch.randn(3, expected.left_layout.dim, dtype=torch.float64)
    right = torch.randn(3, expected.right_layout.dim, dtype=torch.float64)
    torch.testing.assert_close(
        GradeProductExecutor(actual).forward_compact(left, right),
        GradeProductExecutor(expected).forward_compact(left, right),
    )
