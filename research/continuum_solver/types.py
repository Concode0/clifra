# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Shared contracts for continuum solver policies, criteria, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Mapping, Protocol

import torch

if TYPE_CHECKING:
    from .inputs import CoordinateFieldInput, CoordinateLike

MetricValue = torch.Tensor | float | int | bool


@dataclass(frozen=True)
class ContinuumState:
    """A transformation state produced from a coordinate field input."""

    reference_coordinates: torch.Tensor
    deformed_coordinates: torch.Tensor
    reference_multivectors: torch.Tensor
    deformed_multivectors: torch.Tensor
    bivector_weights: torch.Tensor
    spatial_shape: tuple[int, ...]
    batch_shape: tuple[int, ...]
    field_input: CoordinateFieldInput | None = None

    @property
    def coordinate_dim(self) -> int:
        """Return the Euclidean coordinate lane count."""
        return int(self.reference_coordinates.shape[-1])

    @property
    def generator_weights(self) -> torch.Tensor:
        """Return sampled generators under a field-generic name."""
        return self.bivector_weights

    def inverse_input(self) -> CoordinateFieldInput | torch.Tensor:
        """Return deformed values paired with the original sample identity."""
        if self.field_input is None:
            return self.deformed_coordinates
        return self.field_input.with_coordinates(self.deformed_coordinates)


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
class SolverEvaluation:
    """Complete loss decomposition for one solver evaluation."""

    state: ContinuumState
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
    """Differentiable target scoring function injected into a solver engine."""

    def __call__(self, engine, state: ContinuumState) -> CriterionResult:
        """Return a target loss for the current deformation state."""
        ...


class CoordinateTransformationField(Protocol):
    """Minimal optimizer-facing contract for a coordinate transformation field."""

    algebra: object

    def __call__(self, coordinates: CoordinateLike) -> torch.Tensor:
        """Transform coordinate values."""
        ...

    def state(self, coordinates: CoordinateLike) -> ContinuumState:
        """Return transformed values and generator diagnostics."""
        ...

    def inverse(self, coordinates: CoordinateLike) -> torch.Tensor:
        """Evaluate the field's declared inverse contract."""
        ...

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        """Yield trainable field parameters."""
        ...


class GeometricPolicy(Protocol):
    """Mathematical constraint injected into a solver engine."""

    def __call__(self, engine, state: ContinuumState) -> PolicyResult:
        """Return a constraint loss and strictness diagnostics."""
        ...


def zero_criterion(state: ContinuumState, *, name: str = "none") -> CriterionResult:
    """Return a zero target result on the state's device and dtype."""
    return CriterionResult(name=name, loss=state.reference_coordinates.new_zeros(()))


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
