# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Family-owned route facts and the private planning-policy boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol

from clifra.core.executors import ExecutorRequest


def _count(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"planning fact {name!r} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class ProductFacts:
    """Structural execution counts for one product route."""

    interactions: int
    indexed_reduction_terms: int = 0

    def __post_init__(self) -> None:
        for name in ("interactions", "indexed_reduction_terms"):
            object.__setattr__(self, name, _count(getattr(self, name), name))


@dataclass(frozen=True)
class BivectorExpFacts:
    """Product structure used by one bivector-exponential regime."""

    fixed_product_interactions: int = 0
    fixed_reduction_terms: int = 0
    polynomial_interactions: int = 0
    polynomial_reduction_terms: int = 0
    scaled_polynomial_interactions: int = 0
    scaled_polynomial_reduction_terms: int = 0

    def __post_init__(self) -> None:
        for name in (
            "fixed_product_interactions",
            "fixed_reduction_terms",
            "polynomial_interactions",
            "polynomial_reduction_terms",
            "scaled_polynomial_interactions",
            "scaled_polynomial_reduction_terms",
        ):
            object.__setattr__(self, name, _count(getattr(self, name), name))


@dataclass(frozen=True)
class ActionFacts:
    """Representation and child-product structure for one action route."""

    generator_terms: int = 0
    lifted_coefficients: int = 0
    minor_entries: int = 0
    determinant_work: int = 0
    product_interactions: int = 0
    indexed_reduction_terms: int = 0
    exponential_route: str | None = None
    exponential_facts: BivectorExpFacts | None = None

    def __post_init__(self) -> None:
        for name in (
            "generator_terms",
            "lifted_coefficients",
            "minor_entries",
            "determinant_work",
            "product_interactions",
            "indexed_reduction_terms",
        ):
            object.__setattr__(self, name, _count(getattr(self, name), name))
        if (self.exponential_route is None) != (self.exponential_facts is None):
            raise ValueError("action exponential route and facts must be provided together")
        if self.exponential_route is not None and not self.exponential_route:
            raise ValueError("action exponential route must be non-empty")


PlanningFacts = ProductFacts | ActionFacts | BivectorExpFacts | None


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


def _exp_work(route: str, request, facts: BivectorExpFacts) -> int:
    """Return one deliberately coarse structural comparison within exp routes."""
    if route == "closed":
        return facts.fixed_product_interactions + facts.fixed_reduction_terms
    if route == "left_matrix_exp":
        order = 1 << max(request.output.spec.n - 1, 0)
        return facts.fixed_product_interactions + facts.fixed_reduction_terms + order**3
    if route == "taylor":
        # Scaling is input-dependent. Its fixed 16-step envelope is a safety
        # bound, not an assumption that every call performs sixteen squarings.
        return max(
            facts.polynomial_interactions + facts.polynomial_reduction_terms,
            facts.scaled_polynomial_interactions + facts.scaled_polynomial_reduction_terms,
        )
    raise ValueError(f"unknown bivector exponential route {route!r}")


@dataclass(frozen=True)
class DefaultPolicy:
    """Small device-independent structural heuristics for built-in routes."""

    def evaluate(self, candidate: PlanCandidate) -> PolicyEvaluation:
        family, route, request, facts = candidate.family, candidate.route, candidate.request, candidate.facts
        if family == "product":
            score = facts.interactions + facts.indexed_reduction_terms
            return PolicyEvaluation(score, "structural_product_work")
        if family == "bivector_exp":
            if route == "closed":
                return PolicyEvaluation(0.0, "closed_domain")
            return PolicyEvaluation(1.0 + _exp_work(route, request, facts), "structural_exp_work")
        if family == "action" and request.operation == "versor":
            spec = request.output.spec
            inputs, output = request.inputs[0].layout, request.output.layout
            exp_work = (
                0
                if facts.exponential_facts is None
                else _exp_work(facts.exponential_route, request, facts.exponential_facts)
            )
            if route == "vector_matrix":
                matrix_work = spec.n**3 if request.grade == 2 else spec.n**2
                score = (
                    facts.generator_terms
                    + matrix_work
                    + facts.lifted_coefficients
                    + facts.minor_entries
                    + facts.determinant_work
                    + inputs.dim * output.dim
                )
            elif route == "rotor_product":
                score = exp_work + facts.product_interactions + facts.indexed_reduction_terms
            elif route == "full_action_matrix":
                score = exp_work + spec.dim**3 + 3 * spec.dim**2
            else:
                return PolicyEvaluation(None, "unknown_builtin_route")
            return PolicyEvaluation(score, "structural_action_work")
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
