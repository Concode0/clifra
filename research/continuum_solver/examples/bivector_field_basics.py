# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Learn a local bivector field on unordered 2D points."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


def _bootstrap_repo_root(file: str) -> None:
    root = Path(file).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_bootstrap_repo_root(__file__)

from clifra.core.runtime.algebra import AlgebraContext
from research.continuum_solver import (
    CoordinateFieldInput,
    InvertibleBivectorField,
    RBFGeneratorSampler,
    TargetFieldCriterion,
    TransformationFieldEngine,
)


def main() -> None:
    torch.manual_seed(0)
    dtype = torch.float64
    angles = torch.linspace(0.0, 2.0 * math.pi, 25, dtype=dtype)[:-1]
    radii = 0.65 + 0.25 * torch.cos(3.0 * angles)
    coordinates = radii.unsqueeze(-1) * torch.stack((angles.cos(), angles.sin()), dim=-1)

    local_angle = 0.55 * torch.tanh(1.5 * coordinates[:, 0]) - 0.20 * coordinates[:, 1]
    cos_angle = local_angle.cos()
    sin_angle = local_angle.sin()
    target = torch.stack(
        (
            cos_angle * coordinates[:, 0] - sin_angle * coordinates[:, 1],
            sin_angle * coordinates[:, 0] + cos_angle * coordinates[:, 1],
        ),
        dim=-1,
    )

    control_points = torch.tensor(
        [[-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [1.0, 1.0]],
        dtype=dtype,
    )
    algebra = AlgebraContext(2, 0, 0, device="cpu", dtype=dtype)
    field = InvertibleBivectorField(
        algebra,
        coordinate_dim=2,
        generator_sampler=RBFGeneratorSampler(control_points, length_scale=0.75),
        init_scale=1e-2,
    )
    engine = TransformationFieldEngine(field, target_criterion=TargetFieldCriterion(target))
    field_input = CoordinateFieldInput(coordinates, sample_coordinates=coordinates)

    initial_rmse = engine.evaluate(field_input).target.metrics["rmse"].detach()
    run = engine.fit(field_input, steps=160, lr=0.06, log_every=159)
    state = run.evaluation.state
    reconstructed = field.inverse(state.inverse_input())

    permutation = torch.arange(coordinates.shape[0] - 1, -1, -1)
    permuted = field(
        CoordinateFieldInput(
            coordinates[permutation],
            sample_coordinates=coordinates[permutation],
        )
    )
    norm_error = (state.deformed_coordinates.norm(dim=-1) - coordinates.norm(dim=-1)).abs().amax()
    permutation_error = (permuted - state.deformed_coordinates[permutation]).abs().amax()

    print(f"initial_rmse: {initial_rmse.item():.6f}")
    print(f"final_rmse: {state.deformed_coordinates.sub(target).square().mean().sqrt().item():.6f}")
    print(f"norm_max_abs: {norm_error.item():.3e}")
    print(f"roundtrip_max_abs: {(reconstructed - coordinates).abs().amax().item():.3e}")
    print(f"permutation_max_abs: {permutation_error.item():.3e}")


if __name__ == "__main__":
    main()
