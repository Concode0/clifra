# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from clifra.core.runtime.algebra import AlgebraContext
from research.continuum_solver import (
    ContinuumSolverEngine,
    CoordinateFieldInput,
    InvertibleBivectorField,
    InvertiblePathConsistencyPolicy,
    RBFGeneratorSampler,
    TargetFieldCriterion,
)

pytestmark = pytest.mark.unit


def test_engine_uses_material_labels_for_coordinate_sampled_inverse_diagnostics():
    algebra = AlgebraContext(2, 0, 0, device="cpu", dtype=torch.float64)
    labels = torch.linspace(-1.0, 1.0, 7, dtype=torch.float64).unsqueeze(-1)
    coordinates = torch.stack((labels[:, 0], labels[:, 0].square()), dim=-1)
    sampler = RBFGeneratorSampler(torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64), length_scale=0.4)
    field = InvertibleBivectorField(algebra, 2, generator_sampler=sampler, path_steps=2, init_scale=0.3)
    field_input = CoordinateFieldInput(coordinates, sample_coordinates=labels)
    engine = ContinuumSolverEngine(field, geometric_policies=(InvertiblePathConsistencyPolicy(),))

    evaluation = engine.evaluate(field_input)

    assert evaluation.diagnostics["invertible_path/max_abs"] < 1e-10
    assert evaluation.policies[0].metrics["max_abs"] < 1e-10


def test_engine_fits_a_small_target_and_returns_tensor_output():
    torch.manual_seed(13)
    algebra = AlgebraContext(2, 0, 0, device="cpu", dtype=torch.float64)
    coordinates = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    target = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64)
    field = InvertibleBivectorField(algebra, 2, init_scale=0.1)
    engine = ContinuumSolverEngine(field, target_criterion=TargetFieldCriterion(target))
    initial_loss = engine.evaluate(coordinates).loss.detach()

    run = engine.fit(coordinates, steps=25, lr=0.1, log_every=24)

    assert run.output.shape == coordinates.shape
    assert run.evaluation.loss.detach() < initial_loss
    assert len(run.history.records) == 2
