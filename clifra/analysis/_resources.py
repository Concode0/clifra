# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Static resource budgets for optional analysis operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from clifra.core._kernel.planning.resources import ResourceLimits
from clifra.core.layout import GradeLayout


@dataclass(frozen=True)
class AnalysisBudgetCheck:
    """Static lane/entry budget check; does not establish backend support."""

    within_limits: bool
    reason: str
    details: Mapping[str, object]

    def __bool__(self) -> bool:
        return self.within_limits


def budget_check_record(check: AnalysisBudgetCheck) -> dict[str, object]:
    return {"reason": check.reason, "details": dict(check.details)}


def check_matrix_budget(
    *,
    role: str,
    matrix_dim: int,
    limits: ResourceLimits,
    matrix_kind: str,
    dtype: torch.dtype = torch.float32,
) -> AnalysisBudgetCheck:
    """Check an explicit square matrix against static resource limits."""
    matrix_dim = int(matrix_dim)
    matrix_entries = matrix_dim * matrix_dim
    details = {
        "role": str(role),
        "matrix_kind": str(matrix_kind),
        "matrix_dim": matrix_dim,
        "matrix_entries": matrix_entries,
        "max_lanes": limits.max_lanes,
        "max_entries": limits.max_pairs,
        "estimated_bytes": matrix_entries * (torch.finfo(dtype).bits // 8),
        "dtype": str(dtype).removeprefix("torch."),
    }
    if matrix_dim > limits.max_lanes or matrix_entries > limits.max_pairs:
        return AnalysisBudgetCheck(False, f"{matrix_kind}_matrix_cap", details)
    return AnalysisBudgetCheck(True, "ok", details)


def check_full_matrix_budget(
    algebra,
    *,
    role: str,
    limits: ResourceLimits,
    matrix_kind: str,
    dtype: torch.dtype | None = None,
) -> AnalysisBudgetCheck:
    """Check a full-layout square matrix materialization."""
    layout = algebra.spec.full_layout()
    verdict = check_matrix_budget(
        role=role,
        matrix_dim=layout.dim,
        limits=limits,
        matrix_kind=matrix_kind,
        dtype=algebra.dtype if dtype is None else dtype,
    )
    details = dict(verdict.details)
    details.update({"n": layout.spec.n, "full_lanes": layout.dim})
    return AnalysisBudgetCheck(verdict.within_limits, verdict.reason, details)


def check_product_budget(
    *,
    role: str,
    op: str,
    left_layout: GradeLayout,
    right_layout: GradeLayout,
    output_layout: GradeLayout,
    limits: ResourceLimits,
) -> AnalysisBudgetCheck:
    """Check a product's declared layouts against static resource limits."""
    max_lanes = max(left_layout.dim, right_layout.dim, output_layout.dim)
    estimated_pairs = int(left_layout.dim) * int(right_layout.dim)
    details = {
        "role": str(role),
        "op": str(op),
        "n": left_layout.spec.n,
        "left_grades": left_layout.grades,
        "right_grades": right_layout.grades,
        "output_grades": output_layout.grades,
        "left_lanes": left_layout.dim,
        "right_lanes": right_layout.dim,
        "output_lanes": output_layout.dim,
        "estimated_pairs": estimated_pairs,
        "max_lanes": limits.max_lanes,
        "max_pairs": limits.max_pairs,
    }
    if max_lanes > limits.max_lanes:
        return AnalysisBudgetCheck(False, "product_lane_cap", details)
    if estimated_pairs > limits.max_pairs:
        return AnalysisBudgetCheck(False, "product_pair_cap", details)
    return AnalysisBudgetCheck(True, "ok", details)


def check_full_product_budget(
    algebra,
    *,
    role: str,
    op: str,
    limits: ResourceLimits,
) -> AnalysisBudgetCheck:
    """Check a full-layout product against static resource limits."""
    layout = algebra.spec.full_layout()
    return check_product_budget(
        role=role,
        op=op,
        left_layout=layout,
        right_layout=layout,
        output_layout=layout,
        limits=limits,
    )


def first_skip_reason(*checks) -> str:
    return next((check.reason for check in checks if not check), "ok")
