"""Selection retains the exact assessed request, provider, and preparation."""

from dataclasses import dataclass, replace

import torch
from torch import nn

from clifra.core._kernel.planning.policy import (
    NoAvailableRouteError,
    PlanCandidate,
    PlanFacts,
    environment_extensions,
    select_policy_route,
)
from clifra.core._kernel.planning.resources import DEFAULT_RESOURCE_LIMITS, ResourceRequirements
from clifra.core.executors import Assessment, ExecutorProvider, ExecutorRequest, Rejected


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
        from .providers import BuiltinProvider

        if not isinstance(self.provider, BuiltinProvider) and self.family in {"product", "unary"}:
            executor = CompactExecutorAdapter(executor, self.request)
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

    def select(self, request, policy, limits=DEFAULT_RESOURCE_LIMITS):
        from .providers import BuiltinPreparation, BuiltinProvider

        candidates, selections, rejected = [], [], []
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
                facts = PlanFacts()
            contract_lanes = max(
                (
                    request.output.layout.dim,
                    *(contract.layout.dim for contract in request.inputs if contract is not None),
                )
            )
            resources = ResourceRequirements(
                max(contract_lanes, assessment.lanes, facts.resources.lanes),
                max(assessment.pairs, facts.resources.pairs),
            )
            extensions = {
                **environment_extensions(request.output.spec, request.device.type, request.dtype.itemsize),
                "layout.output_lanes": request.output.layout.dim,
                "layout.left_lanes": request.inputs[0].layout.dim
                if request.inputs and request.inputs[0] is not None
                else 0,
                "layout.right_lanes": request.inputs[1].layout.dim
                if len(request.inputs) > 1 and request.inputs[1] is not None
                else 0,
            }
            facts = replace(facts, resources=resources, extensions={**extensions, **dict(facts.extensions)})
            reason = facts.resources.rejection_reason(limits)
            if reason:
                rejected.append((route, reason))
                continue
            selection = Selection(provider, provider_request, assessment, facts)
            if not builtin:
                return selection
            candidates.append(PlanCandidate(family, route, facts))
            selections.append(selection)
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


def default_router():
    from .providers import builtin_providers

    return ExecutorRouter(builtin_providers())


class CompactExecutorAdapter(nn.Module):
    """Keep historical built-in entrypoints out of the public provider protocol."""

    def __init__(self, executor, request):
        super().__init__()
        self.executor = executor
        self.inputs = request.inputs
        self.output = request.output

    def forward(self, *values):
        return self.executor(*values)

    def forward_compact(self, *values):
        return self.executor(*values)

    def forward_pairwise_compact(self, left, right):
        return self.executor(left.unsqueeze(-2), right.unsqueeze(-3))

    def forward_pairwise_compact_right_signed(self, left, right, signs):
        return self.forward_pairwise_compact(left, right * signs)

    def forward_full(self, *values):
        compact = tuple(contract.layout.compact(value) for contract, value in zip(self.inputs, values))
        return self.output.layout.full(self.executor(*compact))
