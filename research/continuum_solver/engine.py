# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Optimizer-coupled continuum solver engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn as nn

from clifra.core.foundation.module import CliffordModule

from .curriculum import ConstantCurriculum, LossWeightSchedule
from .field import InvertibleBivectorField
from .logging import MetricLogger
from .types import ContinuumState, GeometricPolicy, SolverEvaluation, TargetCriterion, zero_criterion

OptimizerLike = object
OptimizerFactory = Callable[[object], OptimizerLike]
OptimizerStepper = Callable[["OptimizationStepContext"], object]


@dataclass
class OptimizationStepContext:
    """Context passed to injected optimizer steppers.

    Custom steppers can call :meth:`closure` as many times as needed, which is
    the contract required by LBFGS-style optimizers and higher-order methods.
    """

    engine: "ContinuumSolverEngine"
    optimizer: OptimizerLike
    coordinates: torch.Tensor
    loss_fn: Callable[[torch.Tensor], torch.Tensor]
    step: int
    steps: int
    clip_grad_norm: float | None = None
    zero_grad_set_to_none: bool = True
    backward_create_graph: bool = False
    backward_retain_graph: bool | None = None

    def closure(
        self,
        *,
        backward: bool = True,
        create_graph: bool | None = None,
        retain_graph: bool | None = None,
        zero_grad: bool = True,
    ) -> torch.Tensor:
        """Evaluate the current objective and optionally backpropagate it."""
        if zero_grad:
            self.optimizer.zero_grad(set_to_none=bool(self.zero_grad_set_to_none))
        loss = self.loss_fn(self.coordinates)
        if backward:
            loss.backward(
                create_graph=self.backward_create_graph if create_graph is None else bool(create_graph),
                retain_graph=self.backward_retain_graph if retain_graph is None else retain_graph,
            )
            if self.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.engine.parameters(), float(self.clip_grad_norm))
        return loss

    def step_optimizer(self) -> torch.Tensor:
        """Run the standard first-order optimizer step."""
        loss = self.closure()
        self.optimizer.step()
        return loss


@dataclass(frozen=True)
class SolverRun:
    """Result object returned by ``ContinuumSolverEngine.fit``."""

    output: torch.Tensor
    evaluation: SolverEvaluation
    history: MetricLogger
    optimizer: OptimizerLike


class ContinuumSolverEngine(CliffordModule):
    """Fit an invertible bivector field against injected targets and policies."""

    def __init__(
        self,
        field: InvertibleBivectorField,
        *,
        target_criterion: TargetCriterion | None = None,
        geometric_policies: Sequence[GeometricPolicy] = (),
        curriculum: LossWeightSchedule | None = None,
    ):
        super().__init__(field.algebra)
        self.field = field
        self.target_criterion = target_criterion
        self.geometric_policies = tuple(geometric_policies)
        self.curriculum = ConstantCurriculum() if curriculum is None else curriculum
        self.fit_step = 0
        self.fit_steps = 1
        self.fit_progress = 1.0
        progress_device = field.bivectors.device
        progress_dtype = field.bivectors.dtype
        self.register_buffer("_fit_step_tensor", torch.zeros((), device=progress_device, dtype=progress_dtype), persistent=False)
        self.register_buffer("_fit_steps_tensor", torch.ones((), device=progress_device, dtype=progress_dtype), persistent=False)
        self.register_buffer("_fit_progress_tensor", torch.ones((), device=progress_device, dtype=progress_dtype), persistent=False)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Return deformed coordinates."""
        return self.field(coordinates)

    def evaluate(self, coordinates: torch.Tensor) -> SolverEvaluation:
        """Evaluate total loss, target loss, geometric policies, and diagnostics."""
        state = self.field.state(coordinates)
        target = self.target_criterion(self, state) if self.target_criterion is not None else zero_criterion(state)
        policies = tuple(policy(self, state) for policy in self.geometric_policies)
        target_weight = self._term_weight("target", target.name, state.reference_coordinates, 1.0)
        total = target.loss * target_weight
        policy_weights = {}
        for policy in policies:
            weight = self._term_weight("policy", policy.name, state.reference_coordinates, policy.weight)
            policy_weights[policy.name] = weight
            total = total + policy.loss * weight
        diagnostics = self._diagnostics(state, policies)
        return SolverEvaluation(
            state=state,
            loss=total,
            target=target,
            policies=policies,
            diagnostics=diagnostics,
            target_weight=target_weight,
            policy_weights=policy_weights,
        )

    def fit(
        self,
        coordinates: torch.Tensor,
        *,
        steps: int = 100,
        optimizer: OptimizerLike | None = None,
        optimizer_factory: OptimizerFactory | None = None,
        optimizer_step: OptimizerStepper | None = None,
        lr: float = 1e-2,
        log_every: int = 1,
        clip_grad_norm: float | None = None,
        backward_create_graph: bool = False,
        backward_retain_graph: bool | None = None,
        zero_grad_set_to_none: bool = True,
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
        stepper = _default_optimizer_step if optimizer_step is None else optimizer_step

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
            self._set_fit_state(step, steps)
            should_log = step % log_every == 0 or step == steps - 1
            context = OptimizationStepContext(
                engine=self,
                optimizer=optimizer,
                coordinates=coordinates,
                loss_fn=loss_fn,
                step=step,
                steps=steps,
                clip_grad_norm=clip_grad_norm,
                zero_grad_set_to_none=zero_grad_set_to_none,
                backward_create_graph=backward_create_graph,
                backward_retain_graph=backward_retain_graph,
            )
            stepper(context)
            if should_log:
                evaluation = self.evaluate(coordinates)
                history.append(step, evaluation.detached_metrics())

        self._set_fit_state(steps - 1, steps)
        final = self.evaluate(coordinates)
        return SolverRun(
            output=final.state.deformed_coordinates.detach(),
            evaluation=final,
            history=history,
            optimizer=optimizer,
        )

    def _loss_for_fit(self, coordinates: torch.Tensor) -> torch.Tensor:
        state = self.field.state(coordinates)
        target = self.target_criterion(self, state) if self.target_criterion is not None else zero_criterion(state)
        target_weight = self._term_weight("target", target.name, state.reference_coordinates, 1.0)
        total = target.loss * target_weight
        for policy in self.geometric_policies:
            result = policy(self, state)
            weight = self._term_weight("policy", result.name, state.reference_coordinates, result.weight)
            total = total + result.loss * weight
        return total

    def _term_weight(self, kind: str, name: str, reference: torch.Tensor, base_weight: float | torch.Tensor) -> torch.Tensor:
        aliases = (f"{kind}:{name}", name, kind, "*")
        return self.curriculum.weight(self, aliases, reference, base_weight=base_weight)

    def fit_step_like(self, values: torch.Tensor) -> torch.Tensor:
        """Return the current fit step as a tensor matching ``values``.

        Criteria should use this helper inside compiled objectives instead of
        reading the Python ``fit_step`` attribute, which changes every step and
        can otherwise force repeated TorchDynamo recompilation.
        """
        return self._fit_step_tensor.to(device=values.device, dtype=values.dtype)

    def fit_steps_like(self, values: torch.Tensor) -> torch.Tensor:
        """Return total fit steps as a tensor matching ``values``."""
        return self._fit_steps_tensor.to(device=values.device, dtype=values.dtype)

    def fit_progress_like(self, values: torch.Tensor) -> torch.Tensor:
        """Return normalized fit progress as a tensor matching ``values``."""
        return self._fit_progress_tensor.to(device=values.device, dtype=values.dtype)

    def _set_fit_state(self, step: int, steps: int) -> None:
        step = int(step)
        steps = max(int(steps), 1)
        progress = (step + 1) / float(steps)
        self.fit_step = step
        self.fit_steps = steps
        self.fit_progress = progress
        with torch.no_grad():
            self._fit_step_tensor.fill_(float(step))
            self._fit_steps_tensor.fill_(float(steps))
            self._fit_progress_tensor.fill_(float(progress))

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


def _default_optimizer_step(context: OptimizationStepContext) -> torch.Tensor:
    return context.step_optimizer()
