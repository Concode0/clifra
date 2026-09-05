# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Internal utilities for the analysis toolkit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch

from clifra.core._kernel.device import resolve_dtype
from clifra.core._kernel.planning.resources import ResourceLimits
from clifra.core.layout import AlgebraSpec, GradeLayout


@dataclass(frozen=True)
class AnalysisFeasibility:
    """Resource-limit verdict for one optional analysis operation."""

    supported: bool
    reason: str
    details: Mapping[str, object]

    def __bool__(self) -> bool:
        return self.supported


def feasibility_record(feasibility: AnalysisFeasibility) -> dict[str, object]:
    return {"reason": feasibility.reason, "details": dict(feasibility.details)}


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
    limits: ResourceLimits,
    matrix_kind: str,
    dtype: torch.dtype = torch.float32,
) -> AnalysisFeasibility:
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
        return AnalysisFeasibility(False, f"{matrix_kind}_matrix_cap", details)
    return AnalysisFeasibility(True, "ok", details)


def full_matrix_feasibility(
    algebra,
    *,
    role: str,
    limits: ResourceLimits,
    matrix_kind: str,
) -> AnalysisFeasibility:
    """Check a full-layout square matrix materialization."""
    layout = full_layout_for_analysis(algebra)
    verdict = matrix_feasibility(
        role=role,
        matrix_dim=layout.dim,
        limits=limits,
        matrix_kind=matrix_kind,
        dtype=getattr(algebra, "dtype", torch.float32),
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
    limits: ResourceLimits,
) -> AnalysisFeasibility:
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
        return AnalysisFeasibility(False, "product_lane_cap", details)
    if estimated_pairs > limits.max_pairs:
        return AnalysisFeasibility(False, "product_pair_cap", details)
    return AnalysisFeasibility(True, "ok", details)


def full_product_feasibility(
    algebra,
    *,
    role: str,
    op: str,
    limits: ResourceLimits,
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
        limits=limits,
    )


def action_matrix_feasibility_for_spec(
    spec: AlgebraSpec,
    *,
    role: str,
    limits: ResourceLimits,
) -> AnalysisFeasibility:
    """Check a full-layout action matrix before constructing an algebra host."""
    layout = spec.full_layout()
    verdict = matrix_feasibility(
        role=role,
        matrix_dim=layout.dim,
        limits=limits,
        matrix_kind="action",
        dtype=torch.float32,
    )
    details = dict(verdict.details)
    details.update({"n": spec.n, "full_lanes": layout.dim})
    return AnalysisFeasibility(verdict.supported, verdict.reason, details)


def declared_full_product_kwargs(algebra) -> dict[str, object]:
    """Return explicit full-grade metadata for planned compact product outputs."""
    layout = algebra.layout()
    return {
        "left": layout,
        "right": layout,
        "output": layout,
    }
