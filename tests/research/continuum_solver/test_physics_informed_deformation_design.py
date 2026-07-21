# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from research.continuum_solver.examples.physics_informed_deformation_design import (
    GradedResponseCriterion,
    ResponseBounds,
    _select_snapshot_frames,
    build_problem,
    coordinate_grid_3d,
    validate_response,
)

pytestmark = pytest.mark.unit


def test_physics_problem_starts_from_the_identity_field():
    args = SimpleNamespace(grid_size=4, control_size=3, path_steps=2)

    coordinates, engine, _ = build_problem(args, device="cpu", dtype=torch.float64)
    initial = engine(coordinates)

    assert torch.count_nonzero(engine.field.bivectors) == 0
    assert torch.allclose(initial, coordinates, atol=1e-12, rtol=1e-12)


def test_physics_objective_has_a_nonzero_gradient_at_identity():
    args = SimpleNamespace(grid_size=4, control_size=3, path_steps=2)
    coordinates, engine, _ = build_problem(args, device="cpu", dtype=torch.float64)
    engine._set_fit_state(0, 60)

    loss = engine.evaluate(coordinates).loss
    loss.backward()
    gradient = engine.field.bivectors.grad

    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.linalg.vector_norm(gradient) > 0


def test_response_report_contains_only_optimized_response_metrics():
    reference = coordinate_grid_3d(4, 4, 4, device="cpu", dtype=torch.float64)
    final = reference.clone()
    final[..., 0] = final[..., 0] * math.exp(0.052)

    response = validate_response(
        reference,
        final,
        target=GradedResponseCriterion(),
        bounds=ResponseBounds(),
    )
    report = response.as_report()

    assert set(report) == {"metrics", "strict_pass", "failures"}
    assert "mean_axial_log_strain" in response.metrics
    assert "before_fit" not in report
    assert "after_fit" not in report
    assert "delta" not in report


def test_trajectory_frames_select_recorded_states_without_interpolation():
    snapshots = tuple(torch.tensor([value], dtype=torch.float64) for value in (0.0, 1.0, 4.0, 9.0))

    all_frames = _select_snapshot_frames(snapshots, frame_count=8)
    reduced_frames = _select_snapshot_frames(snapshots, frame_count=2)

    assert [frame.item() for frame in all_frames] == [0.0, 1.0, 4.0, 9.0]
    assert [frame.item() for frame in reduced_frames] == [0.0, 9.0]
