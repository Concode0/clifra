# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Family-owned route facts and the private planning-policy boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol

import torch

from clifra.core._kernel.planning.work import (
    ActionWorkProfile,
    BivectorExpWorkProfile,
    ProductWorkProfile,
    SandwichWorkProfile,
)
from clifra.core.executors import ExecutorRequest


def _count(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"planning fact {name!r} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class ProductFacts:
    """Authoritative structural profile for one product route."""

    work_profile: ProductWorkProfile

    def __post_init__(self) -> None:
        _count(self.work_profile.bulk, "product.bulk")
        _count(self.work_profile.output, "product.output")


@dataclass(frozen=True)
class BivectorExpFacts:
    """Selected child structure and equations for one exponential route."""

    work_profile: BivectorExpWorkProfile


@dataclass(frozen=True)
class ActionFacts:
    """Authoritative structural profile for one versor-action route."""

    work_profile: ActionWorkProfile


@dataclass(frozen=True)
class SandwichFacts:
    """Authoritative structural profile for one generic sandwich route."""

    work_profile: SandwichWorkProfile

    def __post_init__(self) -> None:
        if not isinstance(self.work_profile, SandwichWorkProfile):
            raise TypeError("sandwich facts require a SandwichWorkProfile")


PlanningFacts = ProductFacts | ActionFacts | SandwichFacts | BivectorExpFacts | None


@dataclass(frozen=True)
class PlanCandidate:
    """One feasible built-in route offered to the policy selector."""

    family: str
    route: str
    request: ExecutorRequest
    facts: PlanningFacts = None

    def __post_init__(self) -> None:
        if not self.family or not self.route:
            raise ValueError("candidate family and route must be non-empty")
        expected = {"product": ProductFacts, "bivector_exp": BivectorExpFacts}.get(self.family)
        if expected is not None and not isinstance(self.facts, expected):
            raise TypeError(f"{self.family} candidates require {expected.__name__}")
        if self.family == "action" and self.request.operation == "versor":
            if not isinstance(self.facts, ActionFacts):
                raise TypeError("versor action candidates require ActionFacts")
        elif self.family == "action" and self.request.operation == "sandwich":
            if not isinstance(self.facts, SandwichFacts):
                raise TypeError("sandwich action candidates require SandwichFacts")
        elif self.family not in {"product", "bivector_exp"} and self.facts is not None:
            raise TypeError(f"{self.family} candidates do not use route facts")


@dataclass(frozen=True)
class PolicyEvaluation:
    """A finite score accepts a candidate; ``None`` rejects it."""

    score: float | None
    reason: str = "rejected_by_policy"
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    @property
    def accepted(self) -> bool:
        return self.score is not None


class PlanningPolicy(Protocol):
    """Structural DI contract for deterministic static route selection."""

    def evaluate(self, candidate: PlanCandidate) -> PolicyEvaluation: ...


def _product_child_work(profile: ProductWorkProfile) -> int:
    return profile.bulk + profile.output


def _equation_work(equation) -> float:
    return (
        sum(call.calls * _product_child_work(call.profile) for call in equation.products)
        + equation.elementwise_cells
        + equation.matrix_exp_order
    )


def _exp_profile_score(profile: BivectorExpWorkProfile) -> float:
    if profile.route == "closed":
        return _equation_work(profile.closed())
    if profile.route == "left_matrix_exp":
        return _equation_work(profile.left_matrix_exp())
    if profile.route == "taylor":
        if profile.square_product is None:
            raise ValueError("Taylor work requires a selected square product")
        plain = _equation_work(profile.taylor_plain())
        scaled_base = (
            sum(_product_child_work(child) for child in profile.scaled_products)
            + sum(profile.scaled_stage_widths)
            + profile.output_width
        )
        square = _product_child_work(profile.square_product) + profile.even_width
        # One square-work basis charge; neither branch nor square count is observed by planning.
        return 0.5 * plain + 0.5 * scaled_base + square
    raise ValueError(f"unknown bivector exponential route {profile.route!r}")


def _small_matrix_exp_regime(request, profile: BivectorExpWorkProfile) -> bool:
    spec = request.output.spec
    stabilized_closed = spec.r > 0 or spec.p > 0 and spec.q > 0
    return profile.even_width == 8 and (request.dtype == torch.float32 or stabilized_closed)


def _action_profile_score(profile: ActionWorkProfile, request: ExecutorRequest) -> float:
    products = profile.product_children
    score = (
        profile.generator_terms
        + profile.reflection_cells
        + 0.2 * (profile.vector_matrix_exp_order + profile.full_action_matrix_order)
        + 100 * sum(child.bulk for child in products)
        + sum(child.output for child in products)
        + profile.full_action_cells
    )
    if profile.exponential_child is not None:
        score += _exp_profile_score(profile.exponential_child)
    if profile.lift is not None:
        # This endpoint is a static score basis, not the executor's device-selected grade set.
        lift = profile.lift.direct() if profile.lift.grades else profile.lift.compound()
        score += 20 * (
            lift.compound_minor_work
            + lift.direct_terms
            + lift.direct_reductions
            + lift.full_matrix_cells
            + lift.dense_matvec_cells
            + lift.block_matvec_cells
            + lift.grade1_matvec_cells
            + lift.output_assembly_cells
        )
    if profile.route == "rotor_product":
        score += profile.exponential_child.even_width + products[0].output + request.output.layout.dim
    elif profile.route == "full_action_matrix":
        score += request.output.spec.dim + request.inputs[0].layout.dim
    return score


@dataclass(frozen=True)
class DefaultPolicy:
    """Small device-independent structural heuristics for built-in routes."""

    def evaluate(self, candidate: PlanCandidate) -> PolicyEvaluation:
        family, route, request, facts = candidate.family, candidate.route, candidate.request, candidate.facts
        if family == "product":
            profile = facts.work_profile
            score = profile.bulk + profile.output if route == "sparse" else 0.5 * profile.bulk + 32 * profile.output
            return PolicyEvaluation(score, "structural_product_work")
        if family == "bivector_exp":
            profile = facts.work_profile
            if route == "left_matrix_exp" and _small_matrix_exp_regime(request, profile):
                # Explicit selection preference at the first biquadratic/small-matrix crossover.
                return PolicyEvaluation(0.0, "small_matrix_exp_regime")
            return PolicyEvaluation(_exp_profile_score(profile), "structural_exp_work")
        if family == "action" and request.operation == "versor":
            if route not in {"vector_matrix", "rotor_product", "full_action_matrix"}:
                return PolicyEvaluation(None, "unknown_builtin_route")
            return PolicyEvaluation(_action_profile_score(facts.work_profile, request), "structural_action_work")
        if family == "action" and request.operation == "sandwich":
            scores = {"composed_products": 0.0, "full_action_matrix": 1.0}
            score = scores.get(route)
            return PolicyEvaluation(
                score,
                "unknown_builtin_route" if score is None else "uncalibrated_composed_sandwich_default",
            )
        scores = {
            ("action", "graded_linear"): 0.0,
            ("action", "full_action_matrix"): 0.0,
            ("unary", "grade_map"): 0.0,
            ("metric", "diagonal"): 0.0,
            ("permutation", "pseudoscalar"): 0.0,
        }
        score = scores.get((family, route))
        return PolicyEvaluation(score, "unknown_builtin_route" if score is None else "only_builtin_route")


DEFAULT_PLANNING_POLICY = DefaultPolicy()


@dataclass(frozen=True)
class RouteDecision:
    """Selected route and its family-owned facts."""

    route: str
    facts: PlanningFacts
    family: str = ""


class NoAvailableRouteError(ValueError):
    """No registered implementation satisfies capability/resource constraints."""


class PolicyCoverageError(ValueError):
    """Raised when executable candidates fall outside every policy region."""


def _finite(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"planning score {name!r} must be finite, got {result}")
    return result


def select_policy_route(policy: PlanningPolicy, candidates: tuple[PlanCandidate, ...]) -> RouteDecision:
    """Select the minimum-score feasible built-in candidate."""
    keys = [(candidate.family, candidate.route) for candidate in candidates]
    if len(keys) != len(set(keys)):
        raise ValueError("route candidates may contain only one candidate per family and route")
    diagnostics: list[Mapping[str, object]] = []
    matches: list[tuple[float, int, PlanCandidate]] = []
    for order, candidate in enumerate(candidates):
        evaluation = policy.evaluate(candidate)
        if not evaluation.accepted:
            diagnostics.append({"route": candidate.route, "status": evaluation.reason, **dict(evaluation.details)})
            continue
        score = _finite(evaluation.score, f"score.{candidate.family}.{candidate.route}")
        matches.append((score, order, candidate))
    if not matches:
        if candidates:
            raise PolicyCoverageError(
                f"Planning policy does not cover candidates {[candidate.route for candidate in candidates]!r}; "
                f"diagnostics={diagnostics!r}"
            )
        raise NoAvailableRouteError("No implemented operation route is available")
    best = matches[0]
    for item in matches[1:]:
        if item[:2] < best[:2]:
            best = item
    _, _, candidate = best
    return RouteDecision(candidate.route, candidate.facts, candidate.family)
