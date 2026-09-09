# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Static allocation limits, independent of route-selection policy."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from clifra.core._kernel.basis import basis_count_for_grades, expand_output_grades, normalize_grades
from clifra.core.layout import AlgebraSpec


class ResourceLimitError(ValueError):
    """A conservative static request exceeds its resource budget."""


@dataclass(frozen=True)
class ResourceLimits:
    """User-configurable static allocation boundaries."""

    warn_lanes: int = 2048
    max_lanes: int = 4096
    warn_pairs: int = 1_000_000
    max_pairs: int = 8_000_000

    def __post_init__(self) -> None:
        values = (self.warn_lanes, self.max_lanes, self.warn_pairs, self.max_pairs)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("resource limits must be non-negative integers")


DEFAULT_RESOURCE_LIMITS = ResourceLimits()


@dataclass(frozen=True)
class ResourceRequirements:
    """Conservative width and static pair/interaction footprint.

    ``pairs`` adds route-owned resident structures and child footprints that
    coexist. A route may also include its larger known fixed-shape temporary.
    Caller-controlled batch dimensions are excluded.
    """

    lanes: int = 0
    pairs: int = 0

    def __post_init__(self):
        for value in (self.lanes, self.pairs):
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError("resource requirements must be non-negative integers")

    def rejection_reason(self, limits: ResourceLimits) -> str | None:
        if self.lanes > limits.max_lanes:
            return f"intermediate lanes {self.lanes} exceed max_lanes={limits.max_lanes}"
        if self.pairs > limits.max_pairs:
            return f"static pair/interaction footprint {self.pairs} exceeds max_pairs={limits.max_pairs}"
        return None

    def warning_reason(self, limits: ResourceLimits) -> str | None:
        warnings = []
        if self.lanes >= limits.warn_lanes:
            warnings.append(f"intermediate lanes {self.lanes} are near max_lanes={limits.max_lanes}")
        if self.pairs >= limits.warn_pairs:
            warnings.append(f"static pair/interaction footprint {self.pairs} is near max_pairs={limits.max_pairs}")
        return "; ".join(warnings) or None


@dataclass(frozen=True)
class _PlanCost:
    spec: AlgebraSpec
    kind: str
    op: str
    left_lanes: int = 0
    right_lanes: int = 0
    output_lanes: int = 0
    pair_count: int = 0
    input_grades: tuple[int, ...] = ()
    left_grades: tuple[int, ...] = ()
    right_grades: tuple[int, ...] = ()
    output_grades: tuple[int, ...] = ()

    @property
    def max_lanes(self) -> int:
        return max(self.left_lanes, self.right_lanes, self.output_lanes)


_WARNED_PLAN_COSTS: set[tuple[object, ...]] = set()


def validate_grades_cost(algebra, spec: AlgebraSpec, grades, *, role: str = "layout") -> tuple[int, ...]:
    normalized = normalize_grades(grades, spec.n)
    validate_plan_cost(
        algebra,
        _PlanCost(
            spec=spec,
            kind="layout",
            op=role,
            output_lanes=basis_count_for_grades(spec.n, normalized),
            output_grades=normalized,
        ),
    )
    return normalized


def validate_product_grades_cost(
    algebra,
    spec: AlgebraSpec,
    *,
    op: str,
    left_grades,
    right_grades,
    output_grades=None,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    left = normalize_grades(left_grades, spec.n, name="left_grades")
    right = normalize_grades(right_grades, spec.n, name="right_grades")
    output = (
        expand_output_grades(left, right, spec.n, op=op)
        if output_grades is None
        else normalize_grades(output_grades, spec.n, name="output_grades")
    )
    left_lanes = basis_count_for_grades(spec.n, left)
    right_lanes = basis_count_for_grades(spec.n, right)
    output_lanes = basis_count_for_grades(spec.n, output)
    validate_plan_cost(
        algebra,
        _PlanCost(
            spec=spec,
            kind="product",
            op=str(op),
            left_lanes=left_lanes,
            right_lanes=right_lanes,
            output_lanes=output_lanes,
            left_grades=left,
            right_grades=right,
            output_grades=output,
        ),
    )
    return left, right, output


def product_plan_cost(request) -> _PlanCost:
    return _PlanCost(
        spec=request.spec,
        kind="product",
        op=request.op,
        left_lanes=request.left_layout.dim,
        right_lanes=request.right_layout.dim,
        output_lanes=request.output_layout.dim,
        left_grades=request.left_grades,
        right_grades=request.right_grades,
        output_grades=request.output_grades,
    )


def unary_plan_cost(request) -> _PlanCost:
    return _PlanCost(
        spec=request.spec,
        kind="unary",
        op=request.op,
        left_lanes=request.input_layout.dim,
        output_lanes=request.output_layout.dim,
        input_grades=request.input_grades,
        output_grades=request.output_grades,
    )


def validate_product_request(algebra, request) -> None:
    validate_plan_cost(algebra, product_plan_cost(request))


def validate_unary_request(algebra, request) -> None:
    validate_plan_cost(algebra, unary_plan_cost(request))


def validate_plan_cost(algebra, cost: _PlanCost, *, limits: ResourceLimits | None = None) -> None:
    limits = algebra._resource_limits if limits is None else limits
    errors = []
    if cost.max_lanes > limits.max_lanes:
        errors.append(f"compact lanes {cost.max_lanes} exceed max_lanes={limits.max_lanes}")
    if cost.pair_count > limits.max_pairs:
        errors.append(f"basis interactions {cost.pair_count} exceed max_pairs={limits.max_pairs}")
    if errors:
        detail = "; ".join(errors)
        raise ResourceLimitError(f"Static {cost.kind} plan for {cost.op} is too large: {detail}. Declare fewer grades.")

    warnings_to_emit = []
    if cost.max_lanes >= limits.warn_lanes:
        warnings_to_emit.append(f"compact lanes {cost.max_lanes} are near max_lanes={limits.max_lanes}")
    if cost.pair_count >= limits.warn_pairs:
        warnings_to_emit.append(f"basis interactions {cost.pair_count} are near max_pairs={limits.max_pairs}")
    if warnings_to_emit:
        _warn_plan_cost_once(cost, "; ".join(warnings_to_emit))


def _warn_plan_cost_once(cost: _PlanCost, detail: str) -> None:
    key = (
        cost.spec,
        cost.kind,
        cost.op,
        cost.input_grades,
        cost.left_grades,
        cost.right_grades,
        cost.output_grades,
        cost.max_lanes,
        cost.pair_count,
    )
    if key in _WARNED_PLAN_COSTS:
        return
    _WARNED_PLAN_COSTS.add(key)
    warnings.warn(f"Static {cost.kind} plan for {cost.op} is large: {detail}.", RuntimeWarning, stacklevel=3)
