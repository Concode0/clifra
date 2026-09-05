"""Selection retains the exact assessed request, provider, and preparation."""

from dataclasses import dataclass, replace

import torch
from torch import nn

from clifra.core._kernel.planning.policy import (
    NoAvailableRouteError,
    PlanCandidate,
    PlanFacts,
    select_policy_route,
)
from clifra.core._kernel.planning.resources import DEFAULT_RESOURCE_LIMITS, ResourceRequirements

from .interface import Assessment, ExecutorProvider, ExecutorRequest, Rejected


@dataclass(frozen=True)
class Selection:
    provider: ExecutorProvider
    request: ExecutorRequest
    assessment: Assessment
    facts: PlanFacts

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
        executor.metadata = ExecutorMetadata(
            self.family,
            self.route,
            self.request.inputs,
            self.request.output,
            self.request.dtype,
            self.request.device,
            self.facts,
        )
        return executor


@dataclass(frozen=True)
class ExecutorMetadata:
    family: str
    route: str
    inputs: tuple
    output: object
    dtype: torch.dtype
    device: torch.device
    facts: PlanFacts


def assessment_facts(assessment):
    return PlanFacts(
        assessment.forward_work,
        assessment.backward_work,
        assessment.peak_bytes,
        assessment.compile_work,
        exact=assessment.exact,
        truncated=assessment.truncated,
        value_dependent=assessment.value_dependent,
        resources=ResourceRequirements(assessment.lanes, assessment.pairs),
    )


@dataclass(frozen=True)
class ExecutorRegistry:
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

    def select(self, request, policy, limits=DEFAULT_RESOURCE_LIMITS):
        candidates, selections, rejected = [], [], []
        for provider in self.providers:
            family, route = provider.identity
            if family != request.family:
                continue
            assessment = provider.assess(request)
            if isinstance(assessment, Rejected):
                rejected.append((route, assessment.reason))
                continue
            if not isinstance(assessment, Assessment):
                raise TypeError("provider.assess must return Assessment or Rejected")
            facts = assessment_facts(assessment)
            # Built-in preparation may supply private, operation-owned policy coordinates.
            from .providers import BuiltinPreparation

            if isinstance(assessment.preparation, BuiltinPreparation):
                facts = replace(facts, extensions=assessment.preparation.facts.extensions)
            reason = facts.resources.rejection_reason(limits)
            if reason:
                rejected.append((route, reason))
                continue
            candidates.append(PlanCandidate(family, route, facts))
            selections.append(Selection(provider, request, assessment, facts))
        if not candidates:
            raise NoAvailableRouteError(f"No implemented {request.family} route is available: {rejected!r}")
        decision = select_policy_route(policy, tuple(candidates))
        return next(selection for selection in selections if selection.route == decision.route)

    def construct(self, request, selection):
        if selection.request is not request:
            raise ValueError("construction requires the original assessed request")
        return selection.build()

    def execute_plan(self, request, policy, limits=DEFAULT_RESOURCE_LIMITS):
        return self.select(request, policy, limits).build()


def default_registry():
    from .providers import builtin_providers

    return ExecutorRegistry(builtin_providers())
