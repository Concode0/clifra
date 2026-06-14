# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Optimizer-coupled continuum solver engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn as nn

from clifra.core.foundation.module import CliffordModule

from .field import InvertibleBivectorField
from .logging import MetricLogger
from .types import ContinuumState, GeometricPolicy, SolverEvaluation, TargetCriterion, zero_criterion


@dataclass(frozen=True)
class SolverRun:
    """Result object returned by ``ContinuumSolverEngine.fit``."""

    output: torch.Tensor
    evaluation: SolverEvaluation
    history: MetricLogger
    optimizer: torch.optim.Optimizer


class ContinuumSolverEngine(CliffordModule):
    """Fit an invertible bivector field against injected targets and policies."""

    def __init__(
        self,
        field: InvertibleBivectorField,
        *,
        target_criterion: TargetCriterion | None = None,
        geometric_policies: Sequence[GeometricPolicy] = (),
    ):
        super().__init__(field.algebra)
        self.field = field
        self.target_criterion = target_criterion
        self.geometric_policies = tuple(geometric_policies)
        self.fit_step = 0
        self.fit_steps = 1
        self.fit_progress = 1.0

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Return deformed coordinates."""
        return self.field(coordinates)

    def evaluate(self, coordinates: torch.Tensor) -> SolverEvaluation:
        """Evaluate total loss, target loss, geometric policies, and diagnostics."""
        state = self.field.state(coordinates)
        target = self.target_criterion(self, state) if self.target_criterion is not None else zero_criterion(state)
        policies = tuple(policy(self, state) for policy in self.geometric_policies)
        total = target.loss
        for policy in policies:
            total = total + policy.weighted_loss
        diagnostics = self._diagnostics(state, policies)
        return SolverEvaluation(state=state, loss=total, target=target, policies=policies, diagnostics=diagnostics)

    def fit(
        self,
        coordinates: torch.Tensor,
        *,
        steps: int = 100,
        optimizer: torch.optim.Optimizer | None = None,
        optimizer_factory: Callable[[object], torch.optim.Optimizer] | None = None,
        lr: float = 1e-2,
        log_every: int = 1,
        clip_grad_norm: float | None = None,
        compile_step: bool = False,
        compile_backend: str | None = None,
        compile_mode: str | None = None,
        compile_fullgraph: bool = False,
    ) -> SolverRun:
        """Optimize the field parameters and return the final deformation."""
        steps = int(steps)
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        log_every = int(log_every)
        if log_every <= 0:
            raise ValueError(f"log_every must be positive, got {log_every}")
        if optimizer is not None and optimizer_factory is not None:
            raise ValueError("pass either optimizer or optimizer_factory, not both")
        if optimizer is None:
            optimizer = optimizer_factory(self.parameters()) if optimizer_factory is not None else torch.optim.Adam(self.parameters(), lr=lr)

        loss_fn = self._loss_for_fit
        if compile_step:
            compile_kwargs = {"fullgraph": bool(compile_fullgraph)}
            if compile_backend:
                compile_kwargs["backend"] = compile_backend
            if compile_mode:
                compile_kwargs["mode"] = compile_mode
            loss_fn = torch.compile(loss_fn, **compile_kwargs)

        history = MetricLogger()
        for step in range(steps):
            self.fit_step = step
            self.fit_steps = steps
            self.fit_progress = (step + 1) / max(steps, 1)
            optimizer.zero_grad(set_to_none=True)
            should_log = step % log_every == 0 or step == steps - 1
            if compile_step:
                loss = loss_fn(coordinates)
                loss.backward()
                if should_log:
                    with torch.no_grad():
                        evaluation = self.evaluate(coordinates)
                else:
                    evaluation = None
            else:
                evaluation = self.evaluate(coordinates)
                evaluation.loss.backward()
            if clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.parameters(), float(clip_grad_norm))
            optimizer.step()

            if should_log and evaluation is not None:
                history.append(step, evaluation.detached_metrics())

        self.fit_progress = 1.0
        final = self.evaluate(coordinates)
        return SolverRun(
            output=final.state.deformed_coordinates.detach(),
            evaluation=final,
            history=history,
            optimizer=optimizer,
        )

    def _loss_for_fit(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.evaluate(coordinates).loss

    def _diagnostics(self, state: ContinuumState, policies):
        reconstructed = self.field.inverse(state.deformed_coordinates)
        path_residual = reconstructed - state.reference_coordinates
        norms = torch.linalg.vector_norm(state.bivector_weights, dim=-1)
        max_violation = state.reference_coordinates.new_zeros(())
        strict_observed = torch.ones((), device=state.reference_coordinates.device, dtype=torch.bool)
        for policy in policies:
            policy_violation = _policy_max_violation(policy)
            max_violation = torch.maximum(max_violation, policy_violation)
            tolerance = torch.as_tensor(policy.strict_tolerance, device=policy_violation.device, dtype=policy_violation.dtype)
            strict_observed = torch.logical_and(strict_observed, policy_violation <= tolerance)
        return {
            "invertible_path/mse": path_residual.square().mean(),
            "invertible_path/rmse": path_residual.square().mean().sqrt(),
            "invertible_path/max_abs": path_residual.abs().amax(),
            "bivector/max_norm": norms.amax(),
            "bivector/mean_norm": norms.mean(),
            "bivector/path_steps": float(self.field.path_steps),
            "strict_constraints/max_violation": max_violation,
            "strict_constraints/observed": strict_observed,
        }


def _policy_max_violation(policy) -> torch.Tensor:
    if not policy.violations:
        return policy.loss.detach().new_zeros(())
    values = [value.detach().abs().amax() for value in policy.violations.values()]
    return torch.stack(values).amax()
