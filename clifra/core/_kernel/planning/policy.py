# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Static route facts and the minimal injected planning-policy boundary."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

from .resources import ResourceRequirements

_COMMON_FACT_NAMES = frozenset(
    {
        "forward_work",
        "backward_work",
        "peak_bytes",
        "compile_work",
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
    """Private built-in costs, resources, and operation-owned coordinates.

    These estimates are not part of the external provider contract. All accepted
    routes implement their declared operation; there is no approximation-quality
    flag. Input-dependent work (such as Taylor scaling) does not change semantics.
    """

    forward_work: float = 0.0
    backward_work: float = 0.0
    peak_bytes: int = 0
    compile_work: float = 0.0
    extensions: Mapping[str, float] | Iterable[tuple[str, float]] = ()
    resources: ResourceRequirements | None = None

    def __post_init__(self) -> None:
        if self.resources is None:
            object.__setattr__(self, "resources", ResourceRequirements())
        elif not isinstance(self.resources, ResourceRequirements):
            raise TypeError("route resources must use ResourceRequirements")
        for name in ("forward_work", "backward_work", "compile_work"):
            value = _finite(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"planning fact {name!r} must be non-negative")
            object.__setattr__(self, name, value)
        peak_bytes = int(self.peak_bytes)
        if peak_bytes < 0:
            raise ValueError("planning fact 'peak_bytes' must be non-negative")
        object.__setattr__(self, "peak_bytes", peak_bytes)
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
        extensions=extensions,
        resources=ResourceRequirements(
            max((part.resources.lanes for part in parts), default=0),
            max((part.resources.pairs for part in parts), default=0),
        ),
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


@dataclass(frozen=True)
class DefaultPolicy:
    """Lightweight regime selector; constants preserve the established routing."""

    def evaluate(self, candidate: PlanCandidate) -> PolicyEvaluation:
        family, route, facts = candidate.family, candidate.route, candidate.facts
        if family == "bivector_exp":
            if route == "closed":
                return PolicyEvaluation(0.0, "closed_domain")
            if route == "taylor":
                return PolicyEvaluation(1.0, "general_clifford_execution")
            if route == "left_matrix_exp":
                n = facts["algebra.n"]
                preferred = (facts["backend.cpu"] and n == 6) or (facts["backend.mps"] and 6 <= n <= 7)
                return PolicyEvaluation(0.0 if preferred else 2.0, "measured_matrix_crossover")
        if family == "product" and route in {"full_table", "sparse"}:
            backend = "mps" if facts["backend.mps"] else ("cpu" if facts["backend.cpu"] else "other")
            if route == "full_table":
                work, compile_weight, lanes = (1.2, 0.0, 0.03) if backend == "mps" else (1.0, 0.0, 0.05)
            else:
                work, compile_weight, lanes = {
                    "cpu": (1.5, 5.0, 0.05),
                    "mps": (0.9, 1.0, 0.03),
                    "other": (1.25, 3.0, 0.05),
                }[backend]
            score = (
                work * facts.forward_work
                + compile_weight * facts.compile_work
                + lanes * facts["layout.output_lanes"]
                + facts.peak_bytes / 4096.0
            )
            return PolicyEvaluation(score, "eligible")
        scores = {
            ("action", "vector_matrix"): -1.0,
            ("action", "rotor_product"): 0.0,
            ("action", "full_action_matrix"): -2.0,
            ("action", "graded_linear"): 0.0,
            ("unary", "grade_map"): 0.0,
            ("metric", "diagonal"): 0.0,
            ("permutation", "pseudoscalar"): 0.0,
        }
        score = scores.get((family, route))
        return PolicyEvaluation(score, "unknown_builtin_route" if score is None else "eligible")


DEFAULT_PLANNING_POLICY = DefaultPolicy()


@dataclass(frozen=True)
class RouteDecision:
    """Selected route and the facts needed by composed planners."""

    route: str
    facts: PlanFacts
    family: str = ""


class NoAvailableRouteError(ValueError):
    """No registered implementation satisfies capability/resource constraints."""


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
        raise NoAvailableRouteError(f"No implemented {family} route is available: {reasons!r}")

    best = matches[0]
    for item in matches[1:]:
        if item[:2] < best[:2]:
            best = item
    _, _, candidate = best
    return RouteDecision(candidate.route, candidate.facts, candidate.family)
