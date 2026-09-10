# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from clifra import AlgebraContext
from research.transformation_fields import CoordinateFieldInput
from research.transformation_fields.examples.relativistic_beamline_design import (
    Config,
    Trajectory,
    _crossings_at_x,
    build_beamline,
    build_field,
    build_initial_beam,
    minkowski_norm_squared,
    run_refinement_study,
    simulate,
)

pytestmark = pytest.mark.unit


def _lorentz_convention_checks(algebra, field_model, dtype):
    action = algebra.plan_versor_action(
        grade=2,
        input=algebra.layout((1,)),
        parameter=algebra.layout((2,)),
        output=algebra.layout((1,)),
    )
    spatial = torch.tensor(
        [[0.0, 0.0, 0.0], [0.31, -0.12, 0.08], [0.42, 0.17, -0.21], [0.18, -0.29, 0.11]],
        device=algebra.device,
        dtype=dtype,
    )
    values = torch.cat((torch.sqrt(1.0 + spatial.square().sum(dim=-1, keepdim=True)), spatial), dim=-1)
    generators = torch.tensor(
        [
            [0.40, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.12, -0.08, 0.05, 0.17, -0.11, 0.09],
            [-0.16, 0.13, 0.07, -0.10, 0.18, -0.14],
            [0.06, 0.11, -0.15, 0.08, 0.04, 0.19],
        ],
        device=algebra.device,
        dtype=dtype,
    )
    rotor_values = action(values, generators)
    f01, f02, f12, f03, f13, f23 = generators.unbind(dim=-1)
    zero = torch.zeros_like(f01)
    matrix_generator = torch.stack(
        (
            zero,
            f01,
            f02,
            f03,
            f01,
            zero,
            f12,
            f13,
            f02,
            -f12,
            zero,
            f23,
            f03,
            -f13,
            -f23,
            zero,
        ),
        dim=-1,
    ).reshape(-1, 4, 4)
    matrix_values = torch.einsum("...ij,...j->...i", torch.matrix_exp(matrix_generator), values)
    clifra_norm = algebra.signature_norm_squared(rotor_values, input=algebra.layout((1,)))[..., 0]
    direct_norm = minkowski_norm_squared(rotor_values)
    field_input = CoordinateFieldInput(
        values, sample_coordinates=torch.zeros(values.shape[0], 3, device=algebra.device, dtype=dtype)
    )
    reconstructed = field_model.inverse(field_model.state(field_input).inverse_input())
    return {
        "positive_e01_spatial_component": float(rotor_values[0, 1]),
        "lorentz_matrix_exponential_max_error": float((rotor_values - matrix_values).abs().max()),
        "boost_minkowski_norm_error": float((direct_norm - 1.0).abs().max()),
        "clifra_signature_norm_agreement": float((clifra_norm - direct_norm).abs().max()),
        "field_inverse_error": float((reconstructed - values).abs().max().detach()),
    }


def test_cl13_action_uses_documented_boost_sign_and_minkowski_norm():
    algebra = AlgebraContext(1, 3, device="cpu", dtype=torch.float64)
    config = replace(Config(), particles_per_axis=3, proper_time_steps=8)
    field_model = build_field(config, algebra=algebra)

    checks = _lorentz_convention_checks(algebra, field_model, torch.float64)

    assert checks["positive_e01_spatial_component"] > 0.0
    assert checks["boost_minkowski_norm_error"] < 1e-12
    assert checks["clifra_signature_norm_agreement"] < 1e-12
    assert checks["field_inverse_error"] < 1e-12
    assert checks["lorentz_matrix_exponential_max_error"] < 1e-12


def test_proper_time_rollout_is_differentiable_and_preserves_mass_shell():
    algebra = AlgebraContext(1, 3, device="cpu", dtype=torch.float64)
    config = replace(Config(), particles_per_axis=3, proper_time_steps=24)
    beam = build_initial_beam(config, device=torch.device("cpu"), dtype=torch.float64)
    field_model = build_field(config, algebra=algebra)
    with torch.no_grad():
        field_model.latent_coordinates[..., 0] = 0.12
        field_model.latent_coordinates[..., 3] = -0.18
        field_model.latent_coordinates[..., 5] = 0.08

    trajectory = simulate(config, field_model, beam)
    loss = trajectory.worldlines[-1, ..., 2:].square().mean()
    loss.backward()

    gradient = field_model.latent_coordinates.grad
    invariant_error = (minkowski_norm_squared(trajectory.four_velocities) - 1.0).abs().max()
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.linalg.vector_norm(gradient) > 1e-8
    assert invariant_error < 1e-11


def test_crossing_selection_handles_nonmonotone_worldlines_and_uses_fractional_rotor():
    algebra = AlgebraContext(1, 3, device="cpu", dtype=torch.float64)
    config = replace(Config(), particles_per_axis=1, proper_time_steps=3, proper_time_step=0.2)
    field_model = build_field(config, algebra=algebra)
    worldlines = torch.zeros(4, 1, 4, dtype=torch.float64)
    worldlines[:, 0, 1] = torch.tensor([0.0, 2.0, 1.0, 3.0], dtype=torch.float64)
    velocities = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]] * 4, dtype=torch.float64)
    field_rates = torch.zeros(3, 1, 6, dtype=torch.float64)
    field_rates[0, 0, 0] = 0.7
    trajectory = Trajectory(worldlines, velocities, field_rates)

    crossing, crossing_velocity = _crossings_at_x(trajectory, torch.tensor([1.5], dtype=torch.float64), field_model)
    expected = field_model.action.action(
        velocities[0], torch.tensor([[0.105, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    )

    assert crossing[0, 0, 1] == pytest.approx(1.5)
    assert torch.allclose(crossing_velocity[0], expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        minkowski_norm_squared(crossing_velocity),
        torch.ones(1, 1, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )


def test_refined_rollout_has_same_horizon_and_zero_field_solution():
    algebra = AlgebraContext(1, 3, device="cpu", dtype=torch.float64)
    coarse_config = replace(Config(), particles_per_axis=3, proper_time_steps=12, proper_time_step=0.04)
    refined_config = replace(coarse_config, proper_time_steps=24, proper_time_step=0.02)
    beam = build_initial_beam(coarse_config, device=torch.device("cpu"), dtype=torch.float64)
    coarse = simulate(coarse_config, build_field(coarse_config, algebra=algebra), beam)
    refined = simulate(refined_config, build_field(refined_config, algebra=algebra), beam)

    assert coarse_config.proper_time_steps * coarse_config.proper_time_step == pytest.approx(
        refined_config.proper_time_steps * refined_config.proper_time_step
    )
    assert torch.allclose(coarse.worldlines[-1], refined.worldlines[-1], atol=1e-12, rtol=1e-12)
    assert torch.allclose(coarse.four_velocities[-1], refined.four_velocities[-1], atol=1e-12, rtol=1e-12)


def test_three_level_refinement_reuses_field_and_heldout_bunch_without_retraining():
    algebra = AlgebraContext(1, 3, device="cpu", dtype=torch.float64)
    config = replace(
        Config(),
        particles_per_axis=3,
        heldout_particles_per_axis=3,
        proper_time_steps=96,
        proper_time_step=0.063,
    )
    beam = build_initial_beam(
        config,
        device=torch.device("cpu"),
        dtype=torch.float64,
        normalized_grid_shift=(0.019, -0.013),
    )
    field_model = build_field(config, algebra=algebra)
    with torch.no_grad():
        sites = torch.linspace(-0.2, 0.2, config.control_sites, dtype=torch.float64)
        field_model.latent_coordinates[0, :, 0] = 0.15
        field_model.latent_coordinates[0, :, 3] = sites
        field_model.latent_coordinates[0, :, 5] = -0.5 * sites

    report, states = run_refinement_study(config, algebra, field_model, build_beamline(), beam)

    assert report["uses_optimizer"] is False
    assert report["same_restored_field"] is True
    assert report["same_heldout_bunch"] is True
    assert report["formal_order_claimed"] is False
    assert set(states) == {"h", "h_over_2", "h_over_4"}
    for factor, label in ((1, "h"), (2, "h_over_2"), (4, "h_over_4")):
        level = report["levels"][label]
        assert level["proper_time_steps"] == factor * config.proper_time_steps
        assert level["proper_time_step"] == pytest.approx(config.proper_time_step / factor)
        assert level["total_proper_time"] == pytest.approx(report["total_proper_time"])
        assert level["metrics"]["mass_shell_max_error"] < 1e-11
    for pair in report["pairwise"].values():
        assert pair["gate_crossing_rms_difference"] > 0.0
        assert pair["target_crossing_rms_difference"] > 0.0
        assert pair["target_centroid_shift"] >= 0.0
        assert pair["target_spread_absolute_difference"] >= 0.0
        assert pair["target_gamma_rms_difference"] >= 0.0
        assert pair["mass_shell_max_error_fine"] < 1e-11


def test_dense_validation_bunch_is_off_the_optimization_grid():
    config = replace(Config(), particles_per_axis=5)
    heldout_config = replace(config, particles_per_axis=9)
    training = build_initial_beam(config, device=torch.device("cpu"), dtype=torch.float64)
    heldout = build_initial_beam(
        heldout_config,
        device=torch.device("cpu"),
        dtype=torch.float64,
        normalized_grid_shift=(0.019, -0.013),
    )

    distances = torch.cdist(training.position[:, 2:], heldout.position[:, 2:])

    assert heldout.position.shape[0] > training.position.shape[0]
    assert distances.min() > 1e-5
