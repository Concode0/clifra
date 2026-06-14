# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Small generic geometric policies for continuum solver constraints."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .types import ContinuumState, PolicyResult


@dataclass(frozen=True)
class BivectorNormPolicy:
    """Keep local path generators inside a requested coefficient norm bound."""

    max_norm: float = 10.0
    weight: float = 1e-3
    strict_tolerance: float = 1e-6
    name: str = "bivector_norm"

    def __call__(self, engine, state: ContinuumState) -> PolicyResult:
        norms = torch.linalg.vector_norm(state.bivector_weights, dim=-1)
        violation = F.relu(norms - float(self.max_norm))
        return PolicyResult(
            name=self.name,
            loss=violation.square().mean(),
            weight=self.weight,
            strict_tolerance=self.strict_tolerance,
            metrics={
                "max_norm": norms.amax(),
                "mean_norm": norms.mean(),
                "bound": float(self.max_norm),
            },
            violations={"max_excess": violation.amax()},
        )


@dataclass(frozen=True)
class InvertiblePathConsistencyPolicy:
    """Penalize numerical drift after forward deformation followed by inverse path."""

    weight: float = 1.0
    strict_tolerance: float = 1e-5
    name: str = "invertible_path_consistency"

    def __call__(self, engine, state: ContinuumState) -> PolicyResult:
        reconstructed = engine.field.inverse(state.deformed_coordinates)
        residual = reconstructed - state.reference_coordinates
        mse = residual.square().mean()
        max_abs = residual.abs().amax()
        return PolicyResult(
            name=self.name,
            loss=mse,
            weight=self.weight,
            strict_tolerance=self.strict_tolerance,
            metrics={
                "mse": mse,
                "rmse": mse.sqrt(),
                "max_abs": max_abs,
            },
            violations={"max_abs": max_abs},
        )
