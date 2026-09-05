# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Static route facts and the minimal injected planning-policy boundary."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

_COMMON_FACT_NAMES = frozenset(
    {
        "forward_work",
        "backward_work",
        "peak_bytes",
        "compile_work",
        "quality_exact",
        "quality_truncated",
        "quality_value_dependent",
    }
)
_COMMON_FACT_ORDER = tuple(sorted(_COMMON_FACT_NAMES))


def _finite(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"planning fact {name!r} must be finite, got {result}")
    return result


def _qualified(name: str) -> str:
    name = str(name)
    parts = name.split(".")
    if len(parts) < 2 or any(not part.isidentifier() for part in parts):
        raise ValueError(
            f"extension attribute {name!r} must be a dot-qualified identifier, for example 'vendor.machine_score'"
        )
    return name


def _normalize_extensions(values) -> tuple[tuple[str, float], ...]:
    source = values.items() if isinstance(values, Mapping) else values
    normalized: dict[str, float] = {}
    for raw_name, raw_value in source:
        name = _qualified(raw_name)
        if name in normalized:
            raise ValueError(f"duplicate extension attribute {name!r}")
        normalized[name] = _finite(raw_value, name)
    return tuple(sorted(normalized.items()))


@dataclass(frozen=True)
class PlanFacts(Mapping[str, float]):
    """Common route facts plus operation-owned extension attributes."""

    forward_work: float = 0.0
    backward_work: float = 0.0
    peak_bytes: int = 0
    compile_work: float = 0.0
    exact: bool = True
    truncated: bool = False
    value_dependent: bool = False
    extensions: Mapping[str, float] | Iterable[tuple[str, float]] = ()

    def __post_init__(self) -> None:
        for name in ("forward_work", "backward_work", "compile_work"):
            value = _finite(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"planning fact {name!r} must be non-negative")
            object.__setattr__(self, name, value)
        peak_bytes = int(self.peak_bytes)
        if peak_bytes < 0:
            raise ValueError("planning fact 'peak_bytes' must be non-negative")
        object.__setattr__(self, "peak_bytes", peak_bytes)
        if self.exact and self.truncated:
            raise ValueError("a route cannot be both exact and truncated")
        if self.exact and self.value_dependent:
            raise ValueError("an exact guarantee cannot be value-dependent")
        object.__setattr__(self, "extensions", _normalize_extensions(self.extensions))

    def __getitem__(self, name: str) -> float:
        if name == "forward_work":
            return self.forward_work
        if name == "backward_work":
            return self.backward_work
        if name == "peak_bytes":
            return float(self.peak_bytes)
        if name == "compile_work":
            return self.compile_work
        if name == "quality_exact":
            return float(self.exact)
        if name == "quality_truncated":
            return float(self.truncated)
        if name == "quality_value_dependent":
            return float(self.value_dependent)
        for key, value in self.extensions:
            if key == name:
                return value
        raise KeyError(name)

    def __iter__(self) -> Iterator[str]:
        return iter((*_COMMON_FACT_ORDER, *(name for name, _ in self.extensions)))

    def __len__(self) -> int:
        return len(_COMMON_FACT_NAMES) + len(self.extensions)


def environment_extensions(spec, backend: str, dtype_bytes: int) -> dict[str, float]:
    """Return the shared static environment coordinates for a route."""
    return {
        "algebra.p": spec.p,
        "algebra.q": spec.q,
        "algebra.r": spec.r,
        "algebra.n": spec.n,
        "backend.cpu": backend == "cpu",
        "backend.mps": backend == "mps",
        "backend.other": backend not in {"cpu", "mps"},
        "dtype.bytes": dtype_bytes,
    }


def compose_plan_facts(
    *parts: PlanFacts,
    peak_bytes: int = 0,
    extensions: Mapping[str, float] | Iterable[tuple[str, float]] = (),
) -> PlanFacts:
    """Compose sequential facts without guessing storage or error propagation."""
    return PlanFacts(
        forward_work=sum(part.forward_work for part in parts),
        backward_work=sum(part.backward_work for part in parts),
        peak_bytes=max(peak_bytes, *(part.peak_bytes for part in parts)),
        compile_work=sum(part.compile_work for part in parts),
        exact=all(part.exact for part in parts),
        truncated=any(part.truncated for part in parts),
        value_dependent=any(part.value_dependent for part in parts),
        extensions=extensions,
    )


@dataclass(frozen=True)
class PlanCandidate:
    """One operation-owned route offered to the policy selector."""

    family: str
    route: str
    facts: PlanFacts
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.family or not self.route:
            raise ValueError("candidate family and route must be non-empty")
        if not isinstance(self.facts, PlanFacts):
            raise TypeError("candidate facts must use PlanFacts")


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


class ScalarFormula(Protocol):
    """Structural contract for a deterministic scalar formula."""

    @property
    def feature_names(self) -> frozenset[str]: ...

    def evaluate(self, facts: Mapping[str, float]) -> float: ...


@dataclass(frozen=True)
class PolynomialTerm:
    """One sparse monomial ``coefficient * product(fact ** exponent)``."""

    coefficient: float
    powers: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficient", _finite(self.coefficient, "coefficient"))
        normalized: list[tuple[str, int]] = []
        seen: set[str] = set()
        for raw_name, raw_exponent in self.powers:
            name = str(raw_name)
            if name not in _COMMON_FACT_NAMES:
                _qualified(name)
            exponent = int(raw_exponent)
            if exponent != raw_exponent or exponent < 0:
                raise ValueError("polynomial exponents must be non-negative integers")
            if name in seen:
                raise ValueError(f"duplicate polynomial fact {name!r} in one term")
            seen.add(name)
            if exponent:
                normalized.append((name, exponent))
        object.__setattr__(self, "powers", tuple(sorted(normalized)))

    def evaluate(self, facts: Mapping[str, float]) -> float:
        value = self.coefficient
        for name, exponent in self.powers:
            try:
                fact = _finite(facts[name], name)
            except KeyError as error:
                raise ValueError(f"formula references unavailable fact {name!r}") from error
            value *= fact**exponent
        return value


@dataclass(frozen=True)
class Polynomial:
    """Immutable sparse multivariate polynomial."""

    constant: float = 0.0
    terms: tuple[PolynomialTerm, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "constant", _finite(self.constant, "constant"))
        object.__setattr__(self, "terms", tuple(self.terms))

    @classmethod
    def feature(cls, name: str, coefficient: float = 1.0, *, constant: float = 0.0) -> "Polynomial":
        return cls(constant, (PolynomialTerm(coefficient, ((name, 1),)),))

    @property
    def feature_names(self) -> frozenset[str]:
        return frozenset(name for term in self.terms for name, _ in term.powers)

    def evaluate(self, facts: Mapping[str, float]) -> float:
        value = self.constant + sum(term.evaluate(facts) for term in self.terms)
        if not math.isfinite(value):
            raise ValueError(f"polynomial evaluation must be finite, got {value}")
        return value


@dataclass(frozen=True)
class FormulaConstraint:
    """Inclusive boundary in normalized form ``formula(facts) <= 0``."""

    formula: ScalarFormula
    reason: str = "outside_region"

    def evaluate(self, facts: Mapping[str, float]) -> tuple[bool, float]:
        value = _finite(self.formula.evaluate(facts), "constraint")
        return value <= 0.0, value


@dataclass(frozen=True)
class BoundaryRegion:
    """Intersection of formula constraints; an empty region is unbounded."""

    constraints: tuple[FormulaConstraint, ...] = ()
    name: str = "default"

    def __post_init__(self) -> None:
        object.__setattr__(self, "constraints", tuple(self.constraints))


@dataclass(frozen=True)
class RouteRule:
    """Formula regions and score for one operation-owned route."""

    family: str
    route: str
    regions: tuple[BoundaryRegion, ...] = (BoundaryRegion(),)
    score: ScalarFormula = Polynomial()

    def __post_init__(self) -> None:
        if not self.family or not self.route:
            raise ValueError("rule family and route must be non-empty")
        if not self.regions:
            raise ValueError(f"route {self.route!r} must declare at least one boundary region")
        object.__setattr__(self, "regions", tuple(self.regions))


@dataclass(frozen=True)
class FormulaPolicy:
    """Declarative formula implementation of :class:`PlanningPolicy`."""

    rules: tuple[RouteRule, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rules", tuple(self.rules))
        keys = [(rule.family, rule.route) for rule in self.rules]
        if len(keys) != len(set(keys)):
            raise ValueError("formula policy may declare only one rule per family and route")
        for rule in self.rules:
            formulas = [rule.score]
            formulas.extend(constraint.formula for region in rule.regions for constraint in region.constraints)
            for formula in formulas:
                for name in formula.feature_names:
                    if name not in _COMMON_FACT_NAMES:
                        _qualified(name)

    def evaluate(self, candidate: PlanCandidate) -> PolicyEvaluation:
        rule = next(
            (rule for rule in self.rules if rule.family == candidate.family and rule.route == candidate.route),
            None,
        )
        if rule is None:
            return PolicyEvaluation(None, "no_policy_rule")

        region_failures: list[Mapping[str, object]] = []
        for region in rule.regions:
            failures = []
            for constraint in region.constraints:
                accepted, value = constraint.evaluate(candidate.facts)
                if not accepted:
                    failures.append({"reason": constraint.reason, "value": value})
            if not failures:
                return PolicyEvaluation(rule.score.evaluate(candidate.facts), "eligible")
            region_failures.append({"region": region.name, "failures": tuple(failures)})
        return PolicyEvaluation(
            None,
            "outside_region",
            details={"regions": tuple(region_failures)},
        )


def _term(coefficient: float, **powers: int) -> PolynomialTerm:
    return PolynomialTerm(coefficient, tuple(powers.items()))


def _default_rules() -> tuple[RouteRule, ...]:
    full_table_score = Polynomial(
        terms=(
            _term(1.0, **{"backend.cpu": 1, "forward_work": 1}),
            _term(0.05, **{"backend.cpu": 1, "layout.output_lanes": 1}),
            _term(1.2, **{"backend.mps": 1, "forward_work": 1}),
            _term(0.03, **{"backend.mps": 1, "layout.output_lanes": 1}),
            _term(1.0, **{"backend.other": 1, "forward_work": 1}),
            _term(0.05, **{"backend.other": 1, "layout.output_lanes": 1}),
            _term(1.0 / 4096.0, peak_bytes=1),
        )
    )
    sparse_score = Polynomial(
        terms=(
            _term(1.5, **{"backend.cpu": 1, "forward_work": 1}),
            _term(5.0, **{"backend.cpu": 1, "compile_work": 1}),
            _term(0.05, **{"backend.cpu": 1, "layout.output_lanes": 1}),
            _term(0.9, **{"backend.mps": 1, "forward_work": 1}),
            _term(1.0, **{"backend.mps": 1, "compile_work": 1}),
            _term(0.03, **{"backend.mps": 1, "layout.output_lanes": 1}),
            _term(1.25, **{"backend.other": 1, "forward_work": 1}),
            _term(3.0, **{"backend.other": 1, "compile_work": 1}),
            _term(0.05, **{"backend.other": 1, "layout.output_lanes": 1}),
            _term(1.0 / 4096.0, peak_bytes=1),
        )
    )
    return (
        RouteRule("product", "full_table", score=full_table_score),
        RouteRule("product", "sparse", score=sparse_score),
        RouteRule("bivector_exp", "closed_simple"),
        RouteRule("bivector_exp", "closed_biquadratic"),
        RouteRule(
            "bivector_exp",
            "spectral_local",
            score=Polynomial.feature("algebra.n", coefficient=-1.0, constant=10.0),
        ),
        RouteRule("bivector_exp", "left_matrix_exp"),
        RouteRule("bivector_exp", "cpu_matrix_exp", score=Polynomial(constant=1.0)),
        RouteRule("action", "vector_matrix", score=Polynomial(constant=-1.0)),
        RouteRule("action", "rotor_product"),
        RouteRule("action", "full_action_matrix", score=Polynomial(constant=-2.0)),
        RouteRule("action", "paired_rotor_product"),
    )


DEFAULT_PLANNING_POLICY = FormulaPolicy(_default_rules())


@dataclass(frozen=True)
class RouteDecision:
    """Selected route and the facts needed by composed planners."""

    route: str
    facts: PlanFacts


class PolicyCoverageError(ValueError):
    """Raised when executable candidates fall outside every policy region."""


def select_policy_route(
    policy: PlanningPolicy,
    candidates: tuple[PlanCandidate, ...],
) -> RouteDecision:
    """Select the minimum-score executable candidate accepted by ``policy``."""
    keys = [(candidate.family, candidate.route) for candidate in candidates]
    if len(keys) != len(set(keys)):
        raise ValueError("route candidates may contain only one candidate per family and route")
    diagnostics: list[Mapping[str, object]] = []
    matches: list[tuple[float, int, PlanCandidate]] = []

    for order, candidate in enumerate(candidates):
        if candidate.unavailable_reason is not None:
            diagnostics.append(
                {"route": candidate.route, "status": "unavailable", "reason": candidate.unavailable_reason}
            )
            continue
        evaluation = policy.evaluate(candidate)
        if not evaluation.accepted:
            diagnostics.append({"route": candidate.route, "status": evaluation.reason, **dict(evaluation.details)})
            continue
        score = _finite(evaluation.score, f"score.{candidate.family}.{candidate.route}")
        matches.append((score, order, candidate))

    if not matches:
        executable = [candidate.route for candidate in candidates if candidate.unavailable_reason is None]
        if executable:
            raise PolicyCoverageError(
                f"Planning policy does not cover candidates {executable!r}; diagnostics={diagnostics!r}"
            )
        reasons = {candidate.route: candidate.unavailable_reason for candidate in candidates}
        family = candidates[0].family if candidates else "operation"
        raise ValueError(f"No implemented {family} route is available: {reasons!r}")

    best = matches[0]
    for item in matches[1:]:
        if item[:2] < best[:2]:
            best = item
    _, _, candidate = best
    return RouteDecision(candidate.route, candidate.facts)
