"""Selection retains the exact assessed request, provider, and preparation."""

import warnings
from dataclasses import dataclass

from torch import nn

from clifra.core._kernel.planning.policy import (
    ActionFacts,
    BivectorExpFacts,
    NoAvailableRouteError,
    PlanCandidate,
    ProductFacts,
    SandwichFacts,
    select_policy_route,
)
from clifra.core._kernel.planning.resources import DEFAULT_RESOURCE_LIMITS, ResourceRequirements
from clifra.core.executors import Assessment, ExecutorProvider, ExecutorRequest, Rejected

PlanningFacts = ProductFacts | ActionFacts | SandwichFacts | BivectorExpFacts | None


@dataclass(frozen=True)
class Selection:
    provider: ExecutorProvider
    request: ExecutorRequest
    assessment: Assessment
    facts: PlanningFacts = None

    @property
    def family(self):
        return self.provider.identity[0]

    @property
    def route(self):
        return self.provider.identity[1]

    def build(self):
        executor = self.provider.build(self.request, self.assessment)
        if not isinstance(executor, nn.Module):
            raise TypeError("provider.build must return an nn.Module")
        from .providers import BuiltinProvider

        if not isinstance(self.provider, BuiltinProvider) and self.family in {"product", "unary"}:
            executor = CompactExecutorAdapter(executor)
        return executor


@dataclass(frozen=True)
class ExecutorRouter:
    providers: tuple[ExecutorProvider, ...]

    def __post_init__(self):
        object.__setattr__(self, "providers", tuple(self.providers))
        keys = tuple(provider.identity for provider in self.providers)
        if any(
            not isinstance(key, tuple) or len(key) != 2 or any(not isinstance(value, str) or not value for value in key)
            for key in keys
        ):
            raise ValueError("provider identity must be a non-empty (family, route) pair")
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate executor family/route")

    def select(self, request, policy, limits=DEFAULT_RESOURCE_LIMITS, *, warn_selected=True):
        from .providers import BuiltinPreparation, BuiltinProvider

        candidates, selections, rejected = [], [], []
        requirements_by_route = {}
        for provider in self.providers:
            family, route = provider.identity
            if family != request.family:
                continue
            builtin = isinstance(provider, BuiltinProvider)
            # An eligible built-in establishes precedence over later external
            # providers. Policy still compares all eligible built-in routes.
            if selections and not builtin:
                continue
            provider_request = (
                request
                if builtin
                else ExecutorRequest(
                    request.family,
                    request.operation,
                    request.inputs,
                    request.output,
                    request.dtype,
                    request.device,
                )
            )
            assessment = provider.assess(provider_request)
            if isinstance(assessment, Rejected):
                rejected.append((route, assessment.reason))
                continue
            if not isinstance(assessment, Assessment):
                raise TypeError("provider.assess must return Assessment or Rejected")
            if builtin:
                if not isinstance(assessment.preparation, BuiltinPreparation):
                    raise TypeError("built-in assessment requires private planning facts")
                facts = assessment.preparation.facts
            else:
                facts = None
            contract_lanes = max(
                (
                    request.output.layout.dim,
                    *(contract.layout.dim for contract in request.inputs if contract is not None),
                )
            )
            resources = ResourceRequirements(
                max(contract_lanes, assessment.lanes),
                assessment.pairs,
            )
            reason = resources.rejection_reason(limits)
            if reason:
                rejected.append((route, reason))
                continue
            selection = Selection(provider, provider_request, assessment, facts)
            if not builtin:
                if warn_selected:
                    self._warn_requirements(request, route, resources, limits)
                return selection
            candidates.append(PlanCandidate(family, route, request, facts))
            selections.append(selection)
            requirements_by_route[route] = resources
        if not candidates:
            raise NoAvailableRouteError(f"No implemented {request.family} route is available: {rejected!r}")
        decision = select_policy_route(policy, tuple(candidates))
        selected = next(selection for selection in selections if selection.route == decision.route)
        if warn_selected:
            self._warn_requirements(request, selected.route, requirements_by_route[selected.route], limits)
        return selected

    @staticmethod
    def _warn_requirements(request, route, requirements, limits):
        reason = requirements.warning_reason(limits)
        if reason:
            warnings.warn(
                f"Static {request.family} route {route} is large: {reason}.",
                RuntimeWarning,
                stacklevel=3,
            )

    def execute_plan(self, request, policy, limits=DEFAULT_RESOURCE_LIMITS):
        return self.select(request, policy, limits).build()


class CompactExecutorAdapter(nn.Module):
    """Keep historical built-in entrypoints out of the public provider protocol."""

    def __init__(self, executor):
        super().__init__()
        self.executor = executor

    def forward(self, *values):
        return self.executor(*values)

    def forward_compact(self, *values):
        return self.executor(*values)

    def forward_pairwise_compact(self, left, right):
        return self.executor(left.unsqueeze(-2), right.unsqueeze(-3))
