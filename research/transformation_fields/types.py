# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Shared contracts for transformation field policies, criteria, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Mapping, Protocol

import torch

if TYPE_CHECKING:
    from .inputs import CoordinateFieldInput, CoordinateLike

MetricValue = torch.Tensor | float | int | bool


@dataclass(frozen=True)
class TransformationState:
    """A transformation state produced from a coordinate field input."""

    input_coordinates: torch.Tensor
    transformed_coordinates: torch.Tensor
    input_multivectors: torch.Tensor
    transformed_multivectors: torch.Tensor
    generator_weights: torch.Tensor
    domain_shape: tuple[int, ...]
    batch_shape: tuple[int, ...]
    field_input: CoordinateFieldInput | None = None
    latent_coordinates: torch.Tensor | None = None

    @property
    def coordinate_dim(self) -> int:
        """Return the exposed coordinate lane count."""
        return int(self.input_coordinates.shape[-1])

    def inverse_input(self) -> CoordinateFieldInput | torch.Tensor:
        """Return transformed values paired with the original sample identity."""
        if self.field_input is None:
            return self.transformed_coordinates
        return self.field_input.with_coordinates(self.transformed_coordinates)


@dataclass(frozen=True)
class TransformationRollout:
    """Intermediate states of one forward path and its exact reverse path.

    By default the leading axis has ``path_steps + 1`` entries. With action
    subdivision it has ``path_steps * substeps_per_step + 1`` entries, sampled
    at canonical generator fractions from each step's starting state. The
    forward trajectory starts at the input. The inverse trajectory starts at
    the forward endpoint and applies the same sampled generators in reverse
    order with opposite signs.
    """

    forward_coordinates: torch.Tensor
    inverse_coordinates: torch.Tensor
    forward_multivectors: torch.Tensor
    inverse_multivectors: torch.Tensor
    generator_weights: torch.Tensor
    latent_coordinates: torch.Tensor | None
    field_input: CoordinateFieldInput | None = None
    includes_initial: bool = True
    path_steps: int | None = None
    substeps_per_step: int = 1

    @property
    def forward(self) -> torch.Tensor:
        """Alias for the forward coordinate trajectory."""
        return self.forward_coordinates

    @property
    def inverse(self) -> torch.Tensor:
        """Alias for the inverse coordinate trajectory."""
        return self.inverse_coordinates

    @property
    def coordinates(self) -> torch.Tensor:
        """Return the complete forward-then-inverse round-trip trajectory."""
        inverse_start = 1 if self.includes_initial else 0
        return torch.cat((self.forward_coordinates, self.inverse_coordinates[inverse_start:]), dim=0)

    @property
    def steps(self) -> int:
        """Return the number of stored generator steps in each direction."""
        if self.path_steps is not None:
            return int(self.path_steps)
        initial_offset = 1 if self.includes_initial else 0
        return int(self.forward_coordinates.shape[0] - initial_offset) // int(self.substeps_per_step)

    @property
    def action_steps(self) -> int:
        """Return the number of sampled fractional action intervals per direction."""
        return self.steps * int(self.substeps_per_step)


@dataclass(frozen=True)
class CriterionResult:
    """Differentiable target score plus scalar diagnostics."""

    name: str
    loss: torch.Tensor
    metrics: Mapping[str, MetricValue] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyResult:
    """Differentiable geometric constraint penalty plus strictness diagnostics."""

    name: str
    loss: torch.Tensor
    weight: float = 1.0
    metrics: Mapping[str, MetricValue] = field(default_factory=dict)
    violations: Mapping[str, torch.Tensor] = field(default_factory=dict)
    strict_tolerance: float = 1e-6

    @property
    def weighted_loss(self) -> torch.Tensor:
        """Return the contribution this policy adds to the optimization objective."""
        return self.loss * float(self.weight)


@dataclass(frozen=True)
class TransformationEvaluation:
    """Complete loss decomposition for one transformation field evaluation."""

    state: TransformationState
    loss: torch.Tensor
    target: CriterionResult
    policies: tuple[PolicyResult, ...]
    diagnostics: Mapping[str, MetricValue]
    target_weight: MetricValue = 1.0
    policy_weights: Mapping[str, MetricValue] = field(default_factory=dict)

    def detached_metrics(self) -> dict[str, MetricValue]:
        """Return detached metric values without forcing host scalar synchronization."""
        target_weight = _detach_metric(self.target_weight)
        metrics: dict[str, MetricValue] = {
            "loss/total": _detach_metric(self.loss),
            f"loss/target/{self.target.name}": _detach_metric(self.target.loss),
            f"loss/target_weighted/{self.target.name}": _detach_metric(self.target.loss * self.target_weight),
            f"weight/target/{self.target.name}": target_weight,
        }
        for key, value in self.target.metrics.items():
            metrics[f"target/{self.target.name}/{key}"] = _detach_metric(value)
        for policy in self.policies:
            policy_weight = self.policy_weights.get(policy.name, policy.weight)
            metrics[f"loss/policy/{policy.name}"] = _detach_metric(policy.loss)
            metrics[f"loss/policy_weighted/{policy.name}"] = _detach_metric(policy.loss * policy_weight)
            metrics[f"weight/policy/{policy.name}"] = _detach_metric(policy_weight)
            for key, value in policy.metrics.items():
                metrics[f"policy/{policy.name}/{key}"] = _detach_metric(value)
            for key, value in policy.violations.items():
                metrics[f"constraint/{policy.name}/{key}"] = _detach_metric(value)
        for key, value in self.diagnostics.items():
            metrics[f"diagnostic/{key}"] = _detach_metric(value)
        return metrics


class TargetCriterion(Protocol):
    """Differentiable target scoring function injected into a transformation field engine."""

    def __call__(self, engine, state: TransformationState) -> CriterionResult:
        """Return a target loss for the current transformation state."""
        ...


class CoordinateTransformationField(Protocol):
    """Minimal optimizer-facing contract for a coordinate transformation field."""

    algebra: object

    def __call__(self, coordinates: CoordinateLike) -> torch.Tensor:
        """Transform coordinate values."""
        ...

    def state(self, coordinates: CoordinateLike) -> TransformationState:
        """Return transformed values and generator diagnostics."""
        ...

    def inverse(self, coordinates: CoordinateLike) -> torch.Tensor:
        """Evaluate the field's declared inverse contract."""
        ...

    def rollout(
        self,
        coordinates: CoordinateLike,
        *,
        include_initial: bool = True,
        substeps_per_step: int = 1,
    ) -> TransformationRollout:
        """Expose one field application's forward path and reverse retracing."""
        ...

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        """Yield trainable field parameters."""
        ...


class GeometricPolicy(Protocol):
    """Mathematical constraint injected into a transformation field engine."""

    def __call__(self, engine, state: TransformationState) -> PolicyResult:
        """Return a constraint loss and strictness diagnostics."""
        ...


def zero_criterion(state: TransformationState, *, name: str = "none") -> CriterionResult:
    """Return a zero target result on the state's device and dtype."""
    return CriterionResult(name=name, loss=state.input_coordinates.new_zeros(()))


def _detach_metric(value: MetricValue) -> MetricValue:
    if isinstance(value, (bool, float, int)):
        return value
    if isinstance(value, torch.Tensor):
        detached = value.detach()
        if detached.numel() == 0:
            return float("nan")
        if detached.numel() > 1:
            detached = detached.float().mean()
        return detached
    return float(value)
