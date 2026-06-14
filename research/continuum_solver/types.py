# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Shared contracts for continuum solver policies, criteria, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

import torch

MetricValue = torch.Tensor | float | int | bool


@dataclass(frozen=True)
class ContinuumState:
    """A deformation state produced from direct coordinate tensors."""

    reference_coordinates: torch.Tensor
    deformed_coordinates: torch.Tensor
    reference_multivectors: torch.Tensor
    deformed_multivectors: torch.Tensor
    bivector_weights: torch.Tensor
    spatial_shape: tuple[int, ...]
    batch_shape: tuple[int, ...]

    @property
    def coordinate_dim(self) -> int:
        """Return the Euclidean coordinate lane count."""
        return int(self.reference_coordinates.shape[-1])


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

    def detached_metrics(self) -> dict[str, MetricValue]:
        """Return detached metric values without forcing host scalar synchronization."""
        metrics: dict[str, MetricValue] = {
            "loss/total": _detach_metric(self.loss),
            f"loss/target/{self.target.name}": _detach_metric(self.target.loss),
        }
        for key, value in self.target.metrics.items():
            metrics[f"target/{self.target.name}/{key}"] = _detach_metric(value)
        for policy in self.policies:
            metrics[f"loss/policy/{policy.name}"] = _detach_metric(policy.loss)
            metrics[f"loss/policy_weighted/{policy.name}"] = _detach_metric(policy.weighted_loss)
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
