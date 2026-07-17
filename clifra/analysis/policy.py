# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Static analysis policy.

Analysis routines sometimes need optional full-lane matrices or broad products
that are not part of normal model forward paths. This module keeps those
decisions equation-based and separate from benchmark observations while using
the same static metadata style as planner policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch

from clifra.core.foundation.layout import GradeLayout
from clifra.core.planning.policy import estimate_product_executor_cost


@dataclass(frozen=True)
class AnalysisCostPolicy:
    """Equation weights for optional analysis materialization decisions."""

    matrix_entry_weight: float = 1.0
    matrix_memory_weight: float = 0.0
    product_pair_weight: float = 1.0
    product_path_weight: float = 0.0
    product_output_weight: float = 0.0
    product_memory_weight: float = 0.0
    memory_cost_unit_bytes: int = 4096


@dataclass(frozen=True)
class AnalysisFeasibility:
    """Static cost verdict for optional analysis materialization."""

    supported: bool
    reason: str
    details: Mapping[str, object]

    def __bool__(self) -> bool:
        """Allow direct use in guards."""
        return self.supported


@dataclass(frozen=True)
class MatrixAnalysisCost:
    """Static cost summary for an explicit analysis matrix."""

    role: str
    matrix_kind: str
    matrix_dim: int
    max_entries: int
    dtype: torch.dtype
    policy: AnalysisCostPolicy = AnalysisCostPolicy()

    @property
    def matrix_entries(self) -> int:
        """Return square matrix element count."""
        return self.matrix_dim * self.matrix_dim

    @property
    def estimated_bytes(self) -> int:
        """Return estimated matrix storage bytes."""
        return self.matrix_entries * _dtype_bytes(self.dtype)

    @property
    def score(self) -> float:
        """Return the weighted analysis score."""
        return _analysis_score(
            units=self.matrix_entries,
            paths=0,
            output_lanes=self.matrix_dim,
            estimated_bytes=self.estimated_bytes,
            unit_weight=self.policy.matrix_entry_weight,
            path_weight=0.0,
            output_weight=0.0,
            memory_weight=self.policy.matrix_memory_weight,
            memory_cost_unit_bytes=self.policy.memory_cost_unit_bytes,
        )

    @property
    def limit_score(self) -> float:
        """Return the weighted score limit."""
        return float(self.max_entries) * float(self.policy.matrix_entry_weight)

    def details(self) -> dict[str, object]:
        """Return JSON-like metadata for diagnostics."""
        return {
            "role": self.role,
            "matrix_kind": self.matrix_kind,
            "matrix_dim": self.matrix_dim,
            "matrix_entries": self.matrix_entries,
            "max_entries": self.max_entries,
            "estimated_bytes": self.estimated_bytes,
            "analysis_score": self.score,
            "analysis_limit_score": self.limit_score,
            "dtype": str(self.dtype).removeprefix("torch."),
        }


@dataclass(frozen=True)
class ProductAnalysisCost:
    """Static cost summary for an optional analysis product."""

    role: str
    op: str
    left_layout: GradeLayout
    right_layout: GradeLayout
    output_layout: GradeLayout
    max_pairs: int
    executor_family: str
    pair_count: int
    estimated_pairs: int
    estimated_bytes: int
    path_count: int
    backend: str
    dtype: torch.dtype
    policy: AnalysisCostPolicy = AnalysisCostPolicy()

    @property
    def score(self) -> float:
        """Return the weighted analysis score."""
        return _analysis_score(
            units=self.pair_count,
            paths=self.path_count,
            output_lanes=self.output_layout.dim,
            estimated_bytes=self.estimated_bytes,
            unit_weight=self.policy.product_pair_weight,
            path_weight=self.policy.product_path_weight,
            output_weight=self.policy.product_output_weight,
            memory_weight=self.policy.product_memory_weight,
            memory_cost_unit_bytes=self.policy.memory_cost_unit_bytes,
        )

    @property
    def limit_score(self) -> float:
        """Return the weighted score limit."""
        return float(self.max_pairs) * float(self.policy.product_pair_weight)

    def details(self) -> dict[str, object]:
        """Return JSON-like metadata for diagnostics."""
        return {
            "role": self.role,
            "op": self.op,
            "n": self.left_layout.spec.n,
            "left_grades": self.left_layout.grades,
            "right_grades": self.right_layout.grades,
            "output_grades": self.output_layout.grades,
            "left_lanes": self.left_layout.dim,
            "right_lanes": self.right_layout.dim,
            "output_lanes": self.output_layout.dim,
            "estimated_pairs": self.estimated_pairs,
            "pair_count": self.pair_count,
            "max_pairs": self.max_pairs,
            "path_count": self.path_count,
            "executor_family": self.executor_family,
            "backend": self.backend,
            "estimated_bytes": self.estimated_bytes,
            "analysis_score": self.score,
            "analysis_limit_score": self.limit_score,
            "dtype": str(self.dtype).removeprefix("torch."),
        }


DEFAULT_ANALYSIS_COST_POLICY = AnalysisCostPolicy()


def feasibility_record(feasibility: AnalysisFeasibility) -> dict[str, object]:
    """Return JSON-like metadata for a feasibility verdict."""
    return {
        "reason": feasibility.reason,
        "details": dict(feasibility.details),
    }


def analysis_cost_policy_for(algebra, policy: Optional[AnalysisCostPolicy] = None) -> AnalysisCostPolicy:
    """Return an analysis cost policy for an algebra-like object."""
    if policy is not None:
        return policy
    return getattr(algebra, "analysis_cost_policy", DEFAULT_ANALYSIS_COST_POLICY)


def evaluate_matrix_cost(cost: MatrixAnalysisCost) -> AnalysisFeasibility:
    """Return a feasibility verdict for a matrix cost."""
    if cost.score > cost.limit_score:
        return AnalysisFeasibility(False, f"{cost.matrix_kind}_matrix_cap", cost.details())
    return AnalysisFeasibility(True, "ok", cost.details())


def build_product_analysis_cost(
    algebra,
    *,
    role: str,
    op: str,
    left_layout: GradeLayout,
    right_layout: GradeLayout,
    output_layout: GradeLayout,
    max_pairs: int,
    dtype: Optional[torch.dtype] = None,
    device=None,
    policy: Optional[AnalysisCostPolicy] = None,
) -> ProductAnalysisCost:
    """Build static product cost metadata using planner policy estimates."""
    resolved_policy = analysis_cost_policy_for(algebra, policy)
    resolved_dtype = getattr(algebra, "dtype", torch.float32) if dtype is None else dtype
    resolved_device = getattr(algebra, "device", "cpu") if device is None else device
    executor_cost = estimate_product_executor_cost(
        algebra,
        op=op,
        left_layout=left_layout,
        right_layout=right_layout,
        output_layout=output_layout,
        dtype=resolved_dtype,
        device=resolved_device,
    )
    if executor_cost.executor_family == "full_table":
        pair_count = executor_cost.full_table_pair_count
        estimated_bytes = executor_cost.full_table_estimated_bytes
    else:
        pair_count = executor_cost.sparse_estimated_pairs
        estimated_bytes = executor_cost.sparse_estimated_bytes
    return ProductAnalysisCost(
        role=str(role),
        op=str(op),
        left_layout=left_layout,
        right_layout=right_layout,
        output_layout=output_layout,
        max_pairs=int(max_pairs),
        executor_family=executor_cost.executor_family,
        pair_count=int(pair_count),
        estimated_pairs=int(left_layout.dim) * int(right_layout.dim),
        estimated_bytes=int(estimated_bytes),
        path_count=int(executor_cost.path_count),
        backend=executor_cost.backend,
        dtype=resolved_dtype,
        policy=resolved_policy,
    )


def evaluate_product_cost(cost: ProductAnalysisCost) -> AnalysisFeasibility:
    """Return a feasibility verdict for a product cost."""
    if cost.score > cost.limit_score:
        return AnalysisFeasibility(False, "product_pair_cap", cost.details())
    return AnalysisFeasibility(True, "ok", cost.details())


def _analysis_score(
    *,
    units: int,
    paths: int,
    output_lanes: int,
    estimated_bytes: int,
    unit_weight: float,
    path_weight: float,
    output_weight: float,
    memory_weight: float,
    memory_cost_unit_bytes: int,
) -> float:
    memory_units = int(estimated_bytes) / max(int(memory_cost_unit_bytes), 1)
    return (
        float(units) * float(unit_weight)
        + float(paths) * float(path_weight)
        + float(output_lanes) * float(output_weight)
        + float(memory_units) * float(memory_weight)
    )


def _dtype_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()
