# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F401

import pytest
import torch

from clifra.core._kernel.basis import (
    basis_indices_for_grades,
    expand_output_grades,
)
from clifra.core._kernel.execution.action import (
    FullSandwichActionExecutor,
    GradedLinearActionExecutor,
    MultiVersorActionExecutor,
    PairedBivectorActionExecutor,
)
from clifra.core._kernel.execution.metric import SignatureNormSquaredExecutor
from clifra.core._kernel.execution.permutation import PseudoscalarProductExecutor
from clifra.core._kernel.execution.product import FullTableProductExecutor, GradeProductExecutor
from clifra.core._kernel.planning.planner import GradePlanner
from clifra.core._kernel.planning.product import build_grade_product_plan, select_product_route
from clifra.core._kernel.planning.resources import ResourceLimits
from clifra.core._kernel.planning.tree import build_grade_plan_tree
from clifra.core.algebra import AlgebraContext
from clifra.core.config import make_algebra
from clifra.core.layout import AlgebraSpec
from clifra.core.operation import PlannedOperation
from clifra.core.tensors import LaneStorage
from tests.helpers.policy import PreferRoute
from tests.helpers.small_oracle import SmallCliffordOracle

DEVICE = "cpu"
_PRODUCT_METHODS = {
    "geometric_product": "geometric_product",
    "wedge": "wedge",
    "symmetric_product": "symmetric_product",
    "commutator_product": "commutator_product",
    "anti_commutator_product": "anti_commutator_product",
    "left_contraction": "left_contraction",
    "right_contraction": "right_contraction",
}


def _mps_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def _oracle_for(algebra) -> SmallCliffordOracle:
    return SmallCliffordOracle(algebra.p, algebra.q, algebra.r)


def _oracle_sandwich_action_matrices(
    oracle: SmallCliffordOracle,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    basis = torch.eye(oracle.dim, dtype=left.dtype, device=left.device)
    matrices = []
    for item in range(left.shape[0]):
        left_values = left[item].expand(oracle.dim, oracle.dim)
        right_values = right[item].expand(oracle.dim, oracle.dim)
        transformed = oracle.product(oracle.product(left_values, basis), right_values)
        matrices.append(transformed.transpose(0, 1))
    return torch.stack(matrices)


def _grade_only_input(algebra, batch: int, grades: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    mv = torch.zeros(batch, algebra.dim, dtype=torch.float64)
    indices = basis_indices_for_grades(algebra.n, grades, device=DEVICE)
    mv[:, indices] = torch.randn(batch, indices.numel(), dtype=torch.float64, generator=generator) * 0.1
    return mv


def _sparse_pairwise_product_reference(
    executor: GradeProductExecutor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    prefix = torch.broadcast_shapes(left.shape[:-2], right.shape[:-2])
    left = left.expand(*prefix, *left.shape[-2:])
    right = right.expand(*prefix, *right.shape[-2:])
    left_terms = torch.index_select(left, -1, executor.left_compact_positions)
    right_terms = torch.index_select(right, -1, executor.right_compact_positions)
    terms = left_terms.unsqueeze(-2) * right_terms.unsqueeze(-3) * executor.coefficients
    output = terms.new_zeros(*terms.shape[:-1], executor.output_dim)
    return output.index_add(-1, executor.output_positions, terms)


def _product_method_name(op: str) -> str:
    return _PRODUCT_METHODS[op]
