# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest
import torch

from clifra import AlgebraContext
from research.transformation_fields import CoordinateFieldInput
from research.transformation_fields.examples.pga_lidar_deskewing import (
    AnalyticTrajectory,
    Config,
    PGAGeometry,
    PGAPointChart,
    Scan,
    acquire_scan,
    build_field,
    build_scene,
)

pytestmark = pytest.mark.unit


def _pga():
    algebra = AlgebraContext(3, 0, 1, device="cpu", dtype=torch.float64)
    chart = PGAPointChart(algebra)
    return algebra, chart, PGAGeometry(algebra, chart)


def _skew(vectors: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(*vectors.shape[:-1], 3, 3, dtype=vectors.dtype, device=vectors.device)
    result[..., 0, 1] = -vectors[..., 2]
    result[..., 0, 2] = vectors[..., 1]
    result[..., 1, 0] = vectors[..., 2]
    result[..., 1, 2] = -vectors[..., 0]
    result[..., 2, 0] = -vectors[..., 1]
    result[..., 2, 1] = vectors[..., 0]
    return result


def _pga_convention_checks(field_model, pga, scan):
    dtype, device = scan.measured_points.dtype, scan.measured_points.device
    points = torch.tensor(
        [[1.25, -0.4, 0.7], [-0.8, 1.1, -0.3], [0.35, 0.25, 1.4], [1.7, -1.2, 0.15]],
        dtype=dtype,
        device=device,
    )
    labels = torch.linspace(0.13, 0.87, points.shape[0], dtype=dtype, device=device).unsqueeze(-1)
    field_input = CoordinateFieldInput(points, sample_coordinates=labels)
    with torch.no_grad():
        moved = field_model(field_input)
        round_trip = field_model.inverse(field_input.with_coordinates(moved))
    plane_normals = torch.tensor(
        [[1.0, 0.0, 0.0], [0.3, -0.8, 0.5], [-0.4, 0.2, 0.9], [0.2, 0.7, -0.6]],
        dtype=dtype,
        device=device,
    )
    plane_normals = plane_normals / torch.linalg.vector_norm(plane_normals, dim=-1, keepdim=True)
    plane_origins = torch.tensor(
        [[0.1, 0.2, -0.3], [-0.4, 0.5, 0.2], [0.7, -0.1, 0.4], [0.2, -0.5, 0.8]],
        dtype=dtype,
        device=device,
    )
    planes = torch.stack([pga.plane(n, o) for n, o in zip(plane_normals, plane_origins)])
    pga_plane = pga.point_plane_signed_distance(pga.chart.embed(points), planes)
    euclidean_plane = ((points - plane_origins) * plane_normals).sum(dim=-1)
    line_starts = torch.tensor(
        [[0.0, 0.0, -1.0], [-0.5, 0.4, 0.1], [0.8, -0.6, 0.2], [-0.3, -0.7, 0.9]],
        dtype=dtype,
        device=device,
    )
    line_ends = torch.tensor(
        [[0.0, 0.0, 1.0], [1.2, 0.9, -0.4], [1.4, 0.7, 1.1], [0.6, -0.2, -0.8]],
        dtype=dtype,
        device=device,
    )
    lines = torch.stack([pga.line_from_points(a, b) for a, b in zip(line_starts, line_ends)])
    directions = line_ends - line_starts
    directions = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    euclidean_line = torch.linalg.vector_norm(torch.linalg.cross(points - line_starts, directions, dim=-1), dim=-1)
    pga_line = pga.point_line_distance(pga.chart.embed(points), lines)
    motor_latent = torch.tensor(
        [
            [0.18, -0.11, 0.27, 0.30, -0.20, 0.10],
            [-0.22, 0.16, 0.09, -0.12, 0.25, 0.18],
            [0.07, 0.21, -0.19, 0.15, 0.08, -0.24],
            [-0.13, -0.08, 0.14, -0.20, -0.12, 0.22],
        ],
        dtype=dtype,
        device=device,
    )
    generator = torch.zeros(points.shape[0], 4, 4, dtype=dtype, device=device)
    generator[:, :3, :3] = _skew(motor_latent[:, :3])
    generator[:, :3, 3] = motor_latent[:, 3:]
    euclidean_motor = torch.matrix_exp(generator)
    homogeneous = torch.cat((points, torch.ones(points.shape[0], 1, dtype=dtype, device=device)), dim=-1)
    matrix_moved = torch.einsum("...ij,...j->...i", euclidean_motor, homogeneous)[..., :3]
    pga_generator = field_model.generator_subspace(motor_latent)
    pga_moved_mv = field_model.action(pga.chart.embed(points), pga_generator)
    pga_moved = pga.chart.extract(pga_moved_mv)
    recovered = pga.chart.extract(field_model.action(pga_moved_mv, -pga_generator))
    return {
        "field_inverse_round_trip_max": float((round_trip - points).abs().max()),
        "point_plane_reference_max_error": float((pga_plane - euclidean_plane).abs().max()),
        "point_line_reference_max_error": float((pga_line - euclidean_line).abs().max()),
        "se3_motor_matrix_reference_max_error": float((pga_moved - matrix_moved).abs().max()),
        "se3_motor_inverse_max_error": float((recovered - points).abs().max()),
        "cases_per_check": points.shape[0],
    }


def test_dual_pga_incidence_residuals_are_euclidean_distances():
    _, chart, pga = _pga()
    coordinates = torch.tensor([[1.25, -0.4, 0.7]], dtype=torch.float64)
    point = chart.embed(coordinates)
    x_zero = pga.plane(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), torch.zeros(3, dtype=torch.float64))
    z_axis = pga.line_from_points(
        torch.tensor([0.0, 0.0, -2.0], dtype=torch.float64),
        torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64),
    )

    plane_distance = pga.point_plane_signed_distance(point, x_zero.unsqueeze(0))
    line_distance = pga.point_line_distance(point, z_axis.unsqueeze(0))

    assert torch.allclose(plane_distance, torch.tensor([1.25], dtype=torch.float64), atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        line_distance,
        torch.tensor([math.hypot(1.25, -0.4)], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )


def test_pga_motor_translation_sign_and_persistent_time_identity():
    algebra, chart, _ = _pga()
    config = Config(rbf_controls=3, rbf_length_scale=0.12)
    field_model = build_field(config, algebra=algebra, chart=chart)
    with torch.no_grad():
        field_model.latent_coordinates[0, :, 3] = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    coordinates = torch.zeros(2, 3, dtype=torch.float64)
    times = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    field_input = CoordinateFieldInput(coordinates, sample_coordinates=times)

    state = field_model.state(field_input)
    reconstructed = field_model.inverse(state.inverse_input())

    assert state.field_input is not None
    assert state.field_input.sample_coordinates is times
    assert state.transformed_coordinates[1, 0] > state.transformed_coordinates[0, 0] + 0.9
    assert torch.allclose(reconstructed, coordinates, atol=1e-12, rtol=1e-12)


def test_pga_geometry_and_mixed_motors_match_euclidean_matrix_references():
    algebra, chart, pga = _pga()
    field_model = build_field(Config(rbf_controls=3), algebra=algebra, chart=chart)
    coordinates = torch.zeros(4, 3, dtype=torch.float64)
    scan = Scan(
        measured_points=coordinates,
        world_points=coordinates,
        acquisition_times=torch.linspace(0.1, 0.9, 4, dtype=torch.float64).unsqueeze(-1),
        plane_ids=torch.full((4,), -1, dtype=torch.long),
        line_ids=torch.full((4,), -1, dtype=torch.long),
        colors=torch.empty(0).numpy(),
    )

    checks = _pga_convention_checks(field_model, pga, scan)

    assert checks["cases_per_check"] == 4
    assert checks["point_plane_reference_max_error"] < 1e-12
    assert checks["point_line_reference_max_error"] < 1e-12
    assert checks["se3_motor_matrix_reference_max_error"] < 1e-12
    assert checks["se3_motor_inverse_max_error"] < 1e-12


def test_analytic_truth_is_matrix_defined_and_inverts_without_an_estimator():
    truth = AnalyticTrajectory()
    times = torch.tensor([[0.07], [0.41], [0.88]], dtype=torch.float64)
    sensor_points = torch.tensor([[1.2, -0.3, 0.7], [3.1, 0.8, -0.4], [0.2, -1.1, 2.0]], dtype=torch.float64)

    world_points = truth.sensor_to_world(sensor_points, times)
    reconstructed = truth.world_to_sensor(world_points, times)

    assert not isinstance(truth, torch.nn.Module)
    assert torch.allclose(reconstructed, sensor_points, atol=1e-12, rtol=1e-12)


def test_heldout_scan_resamples_geometry_and_time_without_retraining_state():
    algebra, _, pga = _pga()
    config = Config(
        plane_samples=(6, 6, 6, 6),
        line_samples=(4, 4, 4, 4, 4),
        measurement_noise=0.0,
    )
    primary_scene = build_scene(config, pga, device=torch.device("cpu"), dtype=torch.float64)
    heldout_scene = build_scene(
        config,
        pga,
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed_offset=101,
    )
    truth = AnalyticTrajectory()
    primary = acquire_scan(config, primary_scene, truth)
    heldout = acquire_scan(config, heldout_scene, truth, seed_offset=101, time_exponent=0.93)

    assert not torch.equal(primary.world_points, heldout.world_points)
    assert not torch.equal(primary.acquisition_times, heldout.acquisition_times)
    assert torch.allclose(
        truth.sensor_to_world(heldout.measured_points, heldout.acquisition_times),
        heldout.world_points,
        atol=1e-12,
        rtol=1e-12,
    )


def test_start_regularizer_must_sample_normalized_rbf_field_not_first_control():
    algebra, chart, _ = _pga()
    field_model = build_field(Config(rbf_controls=3, rbf_length_scale=0.4), algebra=algebra, chart=chart)
    with torch.no_grad():
        field_model.latent_coordinates.zero_()
        field_model.latent_coordinates[0, 0, 3] = 1.0
    start_input = CoordinateFieldInput(
        torch.zeros(1, 3, dtype=torch.float64), sample_coordinates=torch.zeros(1, 1, dtype=torch.float64)
    )

    sampled_start = field_model.generator_sampler.sample(field_model.latent_coordinates, start_input).weights[0, 0]

    assert 0.0 < sampled_start[3] < field_model.latent_coordinates[0, 0, 3]
