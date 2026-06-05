# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Internal utilities for the analysis toolkit."""

from __future__ import annotations

from typing import Iterable

import torch

from clifra.core.foundation.device import resolve_dtype
from clifra.core.foundation.layout import AlgebraSpec, GradeLayout

from .policy import (
    AnalysisFeasibility,
    MatrixAnalysisCost,
    analysis_cost_policy_for,
    build_product_analysis_cost,
    evaluate_matrix_cost,
    evaluate_product_cost,
    feasibility_record,
)


def analysis_dtype(dtype=None) -> torch.dtype:
    """Resolve the floating-point dtype used by analysis routines."""
    return resolve_dtype(torch.float32 if dtype is None else dtype)


def as_analysis_tensor(data: torch.Tensor, *, device, dtype=None) -> torch.Tensor:
    """Move data to the requested analysis device and floating dtype."""
    resolved = analysis_dtype(dtype if dtype is not None else data.dtype)
    if not resolved.is_floating_point:
        resolved = torch.float32
    return data.to(device=device, dtype=resolved)


def full_grades(algebra) -> tuple[int, ...]:
    """Return all grades for explicit full-layout planned calls."""
    return tuple(range(int(algebra.n) + 1))


def analysis_spec(algebra) -> AlgebraSpec:
    """Return immutable signature metadata for an algebra-like object."""
    spec = getattr(algebra, "spec", None)
    if isinstance(spec, AlgebraSpec):
        return spec
    return AlgebraSpec.from_algebra(algebra)


def full_layout_for_analysis(algebra) -> GradeLayout:
    """Return a full-lane layout without applying planner allocation policy."""
    return analysis_spec(algebra).full_layout()


def grade_layout_for_analysis(algebra, grades: Iterable[int]) -> GradeLayout:
    """Return a compact grade layout without applying planner allocation policy."""
    return analysis_spec(algebra).layout(grades)


def matrix_feasibility(
    *,
    role: str,
    matrix_dim: int,
    max_entries: int,
    matrix_kind: str,
    dtype: torch.dtype = torch.float32,
    policy=None,
) -> AnalysisFeasibility:
    """Check whether an explicit square matrix is within analysis policy."""
    return evaluate_matrix_cost(
        MatrixAnalysisCost(
            role=str(role),
            matrix_kind=str(matrix_kind),
            matrix_dim=int(matrix_dim),
            max_entries=int(max_entries),
            dtype=dtype,
            policy=policy if policy is not None else analysis_cost_policy_for(None),
        )
    )


def full_matrix_feasibility(
    algebra,
    *,
    role: str,
    max_entries: int,
    matrix_kind: str,
) -> AnalysisFeasibility:
    """Check a full-layout square matrix materialization."""
    layout = full_layout_for_analysis(algebra)
    verdict = matrix_feasibility(
        role=role,
        matrix_dim=layout.dim,
        max_entries=max_entries,
        matrix_kind=matrix_kind,
        dtype=getattr(algebra, "dtype", torch.float32),
        policy=analysis_cost_policy_for(algebra),
    )
    details = dict(verdict.details)
    details.update({"n": layout.spec.n, "full_lanes": layout.dim})
    return AnalysisFeasibility(verdict.supported, verdict.reason, details)


def product_feasibility(
    algebra,
    *,
    role: str,
    op: str,
    left_layout: GradeLayout,
    right_layout: GradeLayout,
    output_layout: GradeLayout,
    max_pairs: int,
) -> AnalysisFeasibility:
    """Check a planned product using static executor cost metadata."""
    try:
        cost = build_product_analysis_cost(
            algebra,
            role=role,
            op=op,
            left_layout=left_layout,
            right_layout=right_layout,
            output_layout=output_layout,
            max_pairs=max_pairs,
            dtype=getattr(algebra, "dtype", torch.float32),
            device=getattr(algebra, "device", "cpu"),
            policy=analysis_cost_policy_for(algebra),
        )
    except ValueError as exc:
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
            "estimated_pairs": int(left_layout.dim) * int(right_layout.dim),
            "max_pairs": int(max_pairs),
        }
        details["error"] = str(exc)
        return AnalysisFeasibility(False, "planning_limit", details)
    return evaluate_product_cost(cost)


def full_product_feasibility(
    algebra,
    *,
    role: str,
    op: str,
    max_pairs: int,
) -> AnalysisFeasibility:
    """Check a full-layout product used by an optional analysis report."""
    layout = full_layout_for_analysis(algebra)
    return product_feasibility(
        algebra,
        role=role,
        op=op,
        left_layout=layout,
        right_layout=layout,
        output_layout=layout,
        max_pairs=max_pairs,
    )


def action_matrix_feasibility_for_spec(
    spec: AlgebraSpec,
    *,
    role: str,
    max_entries: int,
) -> AnalysisFeasibility:
    """Check a full-layout action matrix before constructing an algebra host."""
    layout = spec.full_layout()
    verdict = matrix_feasibility(
        role=role,
        matrix_dim=layout.dim,
        max_entries=max_entries,
        matrix_kind="action",
        dtype=torch.float32,
    )
    details = dict(verdict.details)
    details.update({"n": spec.n, "full_lanes": layout.dim})
    return AnalysisFeasibility(verdict.supported, verdict.reason, details)


def declared_full_product_kwargs(algebra) -> dict[str, Iterable[int]]:
    """Return explicit full-grade metadata for planned full-lane products."""
    grades = full_grades(algebra)
    return {
        "left_grades": grades,
        "right_grades": grades,
        "output_grades": grades,
        "active_output": True,
    }
