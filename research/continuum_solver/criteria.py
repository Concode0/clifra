# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Small generic target criteria for continuum deformations.

Domain-specific criteria belong in the example or application that injects
them into :class:`~research.continuum_solver.engine.ContinuumSolverEngine`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .types import ContinuumState, CriterionResult


@dataclass(frozen=True)
class TargetFieldCriterion:
    """Fit the deformed field to a target coordinate tensor."""

    target_coordinates: torch.Tensor
    weight: float = 1.0
    name: str = "target_field"

    def __call__(self, engine, state: ContinuumState) -> CriterionResult:
        target = self.target_coordinates.to(device=state.deformed_coordinates.device, dtype=state.deformed_coordinates.dtype)
        deformed, target = torch.broadcast_tensors(state.deformed_coordinates, target)
        residual = deformed - target
        mse = residual.square().mean()
        return CriterionResult(
            name=self.name,
            loss=mse * float(self.weight),
            metrics={
                "mse": mse,
                "rmse": mse.sqrt(),
                "max_abs": residual.abs().amax(),
                "weight": float(self.weight),
            },
        )
