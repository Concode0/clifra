# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Optimize a relativistic particle beamline through a Lorentz bivector field."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from clifra import AlgebraContext
from research.transformation_fields import (
    CoordinateFieldInput,
    GeneratorFieldSample,
    InvertibleBivectorField,
)


@dataclass(frozen=True)
class Config:
    particles_per_axis: int = 5
    heldout_particles_per_axis: int = 9
    proper_time_steps: int = 144
    proper_time_step: float = 0.042
    initial_speed: float = 0.76
    initial_radius: float = 0.085
    initial_divergence: float = 0.032
    control_sites: int = 11
    field_start: float = 0.35
    field_end: float = 6.35
    quadrupole_radius: float = 0.18
    gate_aperture_soft_limit: float = 0.92
    target_region_soft_limit: float = 0.88
    gate_aperture_slope: float = 7.0
    target_region_slope: float = 8.0
    reach_margin: float = 0.04
    monotonic_step_floor: float = 0.014
    optimization_steps: int = 520
    learning_rate: float = 0.055
    minimum_learning_rate_fraction: float = 0.10
    gradient_clip: float = 5.0
    log_every: int = 65
    live_every: int = 20
    target_gamma: float = 1.65
    output_dir: Path = Path("outputs/relativistic_beamline_design")
    live: bool = False
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "gate_aperture": 350.0,
            "target_region": 25.0,
            "target_spread": 180.0,
            "target_direction": 10.0,
            "target_energy": 7.0,
            "energy_spread": 4.0,
            "reach": 30.0,
            "monotonic": 250.0,
            "field_strength": 0.018,
            "field_smoothness": 0.055,
            "field_edges": 0.10,
        }
    )


@dataclass(frozen=True)
class Aperture:
    x: float
    center_yz: tuple[float, float]
    radii_yz: tuple[float, float]
    color: str


@dataclass(frozen=True)
class Beamline:
    apertures: tuple[Aperture, ...]
    target_x: float
    target_center_yz: tuple[float, float]
    target_radii_yz: tuple[float, float]


@dataclass(frozen=True)
class Beam:
    position: torch.Tensor
    four_velocity: torch.Tensor


@dataclass(frozen=True)
class Trajectory:
    worldlines: torch.Tensor
    four_velocities: torch.Tensor
    field_rates: torch.Tensor


@dataclass
class ObjectiveState:
    loss: torch.Tensor
    components: dict[str, torch.Tensor]
    trajectory: Trajectory
    gate_crossings: torch.Tensor
    target_crossing: torch.Tensor
    target_velocity: torch.Tensor


@dataclass(frozen=True)
class OptimizationResult:
    history: list[float]
    best_step: int
    best_loss: float
    final_history_loss: float


class ProperTimeLorentzAction(nn.Module):
    """Scale a field rate by ``dtau`` before clifra's rotor action."""

    def __init__(self, action: nn.Module, proper_time_step: float):
        super().__init__()
        self.action = action
        self.proper_time_step = float(proper_time_step)

    def forward(self, values: torch.Tensor, field_rates: torch.Tensor) -> torch.Tensor:
        return self.action(values, self.proper_time_step * field_rates)


class BeamlineFieldSampler(nn.Module):
    """Map local cubic beamline controls to Cl(1,3) bivector field rates.

    Channels are (Ex, Ey, Ez, steer_xy, steer_xz, quadrupole).
    """

    def __init__(self, config: Config, *, device: torch.device, dtype: torch.dtype):
        super().__init__()
        control_x = torch.linspace(
            config.field_start, config.field_end, config.control_sites, device=device, dtype=dtype
        )
        # Channel scales keep electric, steering, and alternating-gradient controls distinct.
        scales = torch.tensor([0.24, 0.16, 0.16, 0.62, 0.62, 1.10], device=device, dtype=dtype)
        self.register_buffer("control_x", control_x)
        self.register_buffer("scales", scales)
        self.field_start = float(config.field_start)
        self.field_end = float(config.field_end)
        self.quadrupole_radius = float(config.quadrupole_radius)

    def parameter_shape(self, path_steps: int, generator_dim: int) -> tuple[int, ...]:
        if int(path_steps) != 1 or int(generator_dim) != 6:
            raise ValueError("BeamlineFieldSampler requires one path step and the six Cl(1,3) bivector lanes")
        return 1, int(self.control_x.numel()), 6

    def control_values(self, parameters: torch.Tensor) -> torch.Tensor:
        return self.scales * torch.tanh(parameters[0])

    def sample(self, parameters: torch.Tensor, field_input: CoordinateFieldInput) -> GeneratorFieldSample:
        expected = self.parameter_shape(parameters.shape[0], parameters.shape[-1])
        if tuple(parameters.shape) != expected:
            raise ValueError(f"generator parameters must have shape {expected}, got {tuple(parameters.shape)}")
        xyz = field_input.sampling_coordinates
        if xyz.shape[-1] != 3:
            raise ValueError("beamline field sampling coordinates must be spatial (x,y,z) positions")
        xyz = xyz.to(device=parameters.device, dtype=parameters.dtype)
        x, y, z = xyz.unbind(dim=-1)
        # Each sample depends on two neighboring sites; C1 smoothstep interpolation
        # keeps the longitudinal controls local and directly interpretable.
        right = torch.searchsorted(self.control_x, x.detach().contiguous()).clamp(1, self.control_x.numel() - 1)
        left = right - 1
        amount = ((x - self.control_x[left]) / (self.control_x[right] - self.control_x[left])).clamp(0.0, 1.0)
        smooth_amount = amount.square() * (3.0 - 2.0 * amount)
        controls = self.control_values(parameters)
        channels = controls[left] + smooth_amount.unsqueeze(-1) * (controls[right] - controls[left])
        # A smooth finite beamline envelope suppresses fields before the first
        # magnet and after the last magnet without hard spatial branches.
        edge_width = 0.16
        envelope = torch.sigmoid((x - self.field_start) / edge_width) * torch.sigmoid((self.field_end - x) / edge_width)
        channels = channels * envelope.unsqueeze(-1)
        ex, ey, ez, steer_xy, steer_xz, quadrupole = channels.unbind(dim=-1)
        generator = torch.stack(
            (
                ex,
                ey,
                steer_xy - quadrupole * y / self.quadrupole_radius,
                ez,
                steer_xz + quadrupole * z / self.quadrupole_radius,
                torch.zeros_like(ex),
            ),
            dim=-1,
        )
        domain_shape, batch_shape = field_input.topology_shapes()
        return GeneratorFieldSample(
            weights=generator.unsqueeze(0),
            domain_shape=domain_shape,
            batch_shape=batch_shape,
            latent_coordinates=channels.unsqueeze(0),
        )


def build_beamline() -> Beamline:
    """Three offset elliptical apertures force a coupled S-bend and refocus."""
    return Beamline(
        apertures=(
            Aperture(1.65, (0.20, 0.10), (0.22, 0.18), "#f4a261"),
            Aperture(3.35, (0.62, -0.22), (0.19, 0.16), "#e76f51"),
            Aperture(5.05, (0.36, 0.20), (0.16, 0.13), "#9b5de5"),
        ),
        target_x=6.35,
        target_center_yz=(0.78, 0.02),
        target_radii_yz=(0.095, 0.075),
    )


def build_initial_beam(
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
    normalized_grid_shift: tuple[float, float] = (0.0, 0.0),
) -> Beam:
    """Construct a transverse phase-space grid with coupled divergence."""
    grid = torch.linspace(-1.0, 1.0, config.particles_per_axis, device=device, dtype=dtype)
    gy, gz = torch.meshgrid(grid, grid, indexing="ij")
    gy = gy + normalized_grid_shift[0]
    gz = gz + normalized_grid_shift[1]
    y = config.initial_radius * gy.reshape(-1)
    z = config.initial_radius * gz.reshape(-1)
    n = y.numel()
    position = torch.zeros(n, 4, device=device, dtype=dtype)
    position[:, 1] = 0.0
    position[:, 2] = y
    position[:, 3] = z

    phase = torch.arange(n, device=device, dtype=dtype)
    vy = config.initial_divergence * (0.72 * gy.reshape(-1) + 0.28 * torch.sin(1.7 * phase))
    vz = config.initial_divergence * (-0.62 * gz.reshape(-1) + 0.30 * torch.cos(1.3 * phase))
    transverse_sq = vy.square() + vz.square()
    vx = torch.sqrt(torch.full_like(transverse_sq, config.initial_speed**2) - transverse_sq)
    gamma = 1.0 / math.sqrt(1.0 - config.initial_speed**2)
    four_velocity = gamma * torch.stack((torch.ones_like(vx), vx, vy, vz), dim=-1)
    return Beam(position=position, four_velocity=four_velocity)


def build_field(config: Config, *, algebra: AlgebraContext) -> InvertibleBivectorField:
    sampler = BeamlineFieldSampler(config, device=algebra.device, dtype=algebra.dtype)
    vector_layout = algebra.layout((1,))
    bivector_layout = algebra.layout((2,))
    base_action = algebra.plan_versor_action(
        grade=2,
        input=vector_layout,
        parameter=bivector_layout,
        output=vector_layout,
    )
    field_model = InvertibleBivectorField(
        algebra,
        coordinate_dim=4,
        path_steps=1,
        generator_sampler=sampler,
        init_scale=0.0,
        action=ProperTimeLorentzAction(base_action, config.proper_time_step),
    )
    with torch.no_grad():
        field_model.latent_coordinates.zero_()
    return field_model


def simulate(config: Config, field_model: InvertibleBivectorField, beam: Beam) -> Trajectory:
    """Advance four-velocities by Lorentz rotors and integrate worldlines."""
    position = beam.position
    four_velocity = beam.four_velocity
    positions = [position]
    velocities = [four_velocity]
    field_rates = []
    for _ in range(config.proper_time_steps):
        field_input = CoordinateFieldInput(four_velocity, sample_coordinates=position[..., 1:])
        state = field_model.state(field_input)
        next_velocity = state.transformed_coordinates
        position = position + 0.5 * config.proper_time_step * (four_velocity + next_velocity)
        four_velocity = next_velocity
        positions.append(position)
        velocities.append(four_velocity)
        field_rates.append(state.generator_weights[0])
    return Trajectory(
        worldlines=torch.stack(positions),
        four_velocities=torch.stack(velocities),
        field_rates=torch.stack(field_rates),
    )


def _crossings_at_x(
    trajectory: Trajectory,
    x_planes: torch.Tensor,
    field_model: InvertibleBivectorField,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the first genuine forward crossing of each fixed-x plane.

    Segment selection is discrete and detached; position interpolation and the
    fractional rotor within the selected step remain differentiable. If a
    trajectory never crosses, its furthest-reaching segment is used so the
    reach penalty can provide a corrective gradient.
    """
    worldlines = trajectory.worldlines
    x0 = worldlines[:-1, ..., 1].transpose(0, 1).contiguous()
    x1 = worldlines[1:, ..., 1].transpose(0, 1).contiguous()
    particles, _ = x0.shape
    query = x_planes.reshape(1, -1).expand(particles, -1)
    eps = 32.0 * torch.finfo(x0.dtype).eps
    forward_crossing = (
        (x0.detach()[:, None, :] <= query[..., None])
        & (x1.detach()[:, None, :] >= query[..., None])
        & ((x1 - x0).detach()[:, None, :] > eps)
    )
    valid = forward_crossing.any(dim=-1)
    first = forward_crossing.to(torch.int64).argmax(dim=-1)
    fallback = x1.detach().argmax(dim=-1, keepdim=True).expand_as(first)
    left = torch.where(valid, first, fallback)
    right = left + 1
    particle = torch.arange(particles, device=x0.device).unsqueeze(-1).expand_as(left)
    transposed_worldlines = worldlines.transpose(0, 1)
    p0 = transposed_worldlines[particle, left]
    p1 = transposed_worldlines[particle, right]
    denominator = p1[..., 1] - p0[..., 1]
    safe_denominator = torch.where(
        denominator.abs() < eps,
        torch.where(denominator < 0.0, -eps, eps),
        denominator,
    )

    amount = ((query - p0[..., 1]) / safe_denominator).clamp(0.0, 1.0)
    crossing = p0 + amount.unsqueeze(-1) * (p1 - p0)

    velocities = trajectory.four_velocities.transpose(0, 1)
    u0 = velocities[particle, left]
    rates = trajectory.field_rates.transpose(0, 1)[particle, left]
    action = field_model.action
    if not isinstance(action, ProperTimeLorentzAction):
        raise TypeError("beamline crossings require ProperTimeLorentzAction")
    crossing_velocity = action.action(u0, amount.unsqueeze(-1) * action.proper_time_step * rates)
    return crossing.transpose(0, 1), crossing_velocity.transpose(0, 1)


def _elliptical_radius(points_yz: torch.Tensor, center: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm((points_yz - center) / radii, dim=-1)


class BeamlineObjective:
    def __init__(self, config: Config, beamline: Beamline, beam: Beam):
        self.config = config
        self.beamline = beamline
        self.beam = beam
        reference = beam.position
        self.gate_x = reference.new_tensor([gate.x for gate in beamline.apertures])
        self.gate_centers = reference.new_tensor([gate.center_yz for gate in beamline.apertures])
        self.gate_radii = reference.new_tensor([gate.radii_yz for gate in beamline.apertures])
        self.target_x = reference.new_tensor([beamline.target_x])
        self.target_center = reference.new_tensor(beamline.target_center_yz)
        self.target_radii = reference.new_tensor(beamline.target_radii_yz)

    def __call__(self, field_model: InvertibleBivectorField) -> ObjectiveState:
        trajectory = simulate(self.config, field_model, self.beam)
        gate_crossings, _ = _crossings_at_x(trajectory, self.gate_x, field_model)
        target_crossing, target_velocity = _crossings_at_x(trajectory, self.target_x, field_model)
        gate_yz = gate_crossings[..., 2:4]
        target_yz = target_crossing[0, ..., 2:4]
        final_u = target_velocity[0]
        gate_radius = _elliptical_radius(gate_yz, self.gate_centers[:, None, :], self.gate_radii[:, None, :])
        target_spread = target_yz - target_yz.mean(dim=0)
        target_radius = _elliptical_radius(target_yz, self.target_center, self.target_radii)
        transverse_slope = final_u[..., 2:4] / final_u[..., 1:2].clamp_min(0.2)
        delta_x = trajectory.worldlines[1:, ..., 1] - trajectory.worldlines[:-1, ..., 1]
        downstream_x = trajectory.worldlines[-1, ..., 1]
        sampler = field_model.generator_sampler
        if not isinstance(sampler, BeamlineFieldSampler):
            raise TypeError("BeamlineObjective requires BeamlineFieldSampler")
        controls = sampler.control_values(field_model.latent_coordinates)
        gate_excess = self.config.gate_aperture_slope * (gate_radius - self.config.gate_aperture_soft_limit)
        target_excess = self.config.target_region_slope * (target_radius - self.config.target_region_soft_limit)
        components = {
            "gate_aperture": F.softplus(gate_excess).square().mean() / self.config.gate_aperture_slope**2,
            "target_region": F.softplus(target_excess).square().mean() / self.config.target_region_slope**2,
            "target_spread": target_spread.square().mean(),
            "target_direction": transverse_slope.square().mean(),
            "target_energy": (final_u[..., 0].mean() - self.config.target_gamma).square(),
            "energy_spread": final_u[..., 0].var(unbiased=False),
            "reach": F.relu(self.beamline.target_x + self.config.reach_margin - downstream_x).square().mean(),
            "monotonic": F.relu(self.config.monotonic_step_floor - delta_x).square().mean(),
            "field_strength": controls.square().mean(),
            "field_smoothness": (controls[1:] - controls[:-1]).square().mean(),
            "field_edges": controls[[0, -1]].square().mean(),
        }
        loss = sum(self.config.weights[name] * value for name, value in components.items())
        return ObjectiveState(
            loss=loss,
            components=components,
            trajectory=trajectory,
            gate_crossings=gate_crossings,
            target_crossing=target_crossing[0],
            target_velocity=final_u,
        )


def optimize(
    config: Config,
    field_model: InvertibleBivectorField,
    objective: BeamlineObjective,
    initial_state: ObjectiveState,
    view: LiveView | None = None,
) -> OptimizationResult:
    optimizer = torch.optim.Adam(field_model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.optimization_steps,
        eta_min=config.minimum_learning_rate_fraction * config.learning_rate,
    )
    history = [float(initial_state.loss)]
    best_loss = history[0]
    best_step = 0
    best_parameters = field_model.latent_coordinates.detach().clone()
    print(
        f"[{0:4d}/{config.optimization_steps}] loss={initial_state.loss.item():.4e} "
        f"aperture={math.sqrt(initial_state.components['gate_aperture'].item()):.4f} "
        f"focus={math.sqrt(initial_state.components['target_spread'].item()):.4f} "
        f"gamma={initial_state.target_velocity[:, 0].mean().item():.3f}"
    )
    if view is not None:
        view.update(0, initial_state)
    for step in range(1, config.optimization_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        state = objective(field_model)
        if not torch.isfinite(state.loss):
            raise RuntimeError(f"non-finite loss at optimization step {step}")
        state.loss.backward()
        torch.nn.utils.clip_grad_norm_(field_model.parameters(), config.gradient_clip)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            state = objective(field_model)
        detached_loss = float(state.loss)
        history.append(detached_loss)
        if detached_loss < best_loss:
            best_loss = detached_loss
            best_step = step
            best_parameters.copy_(field_model.latent_coordinates.detach())
        if view is not None:
            view.pump_events()
            if step % config.live_every == 0 or step == config.optimization_steps:
                view.update(step, state)
        if step % config.log_every == 0 or step == config.optimization_steps:
            target_rms = math.sqrt(float(state.components["target_spread"].detach()))
            print(
                f"[{step:4d}/{config.optimization_steps}] loss={state.loss.item():.4e} "
                f"aperture={math.sqrt(state.components['gate_aperture'].item()):.4f} "
                f"focus={target_rms:.4f} gamma={state.target_velocity[:, 0].mean().item():.3f}"
            )
    with torch.no_grad():
        field_model.latent_coordinates.copy_(best_parameters)
    return OptimizationResult(history, best_step, best_loss, history[-1])


def minkowski_norm_squared(four_velocity: torch.Tensor) -> torch.Tensor:
    return four_velocity[..., 0].square() - four_velocity[..., 1:].square().sum(dim=-1)


def _trajectory_metrics(
    objective: BeamlineObjective,
    state: ObjectiveState,
) -> dict[str, Any]:
    gate_yz = state.gate_crossings[..., 2:4]
    gate_radius = _elliptical_radius(
        gate_yz,
        objective.gate_centers[:, None, :],
        objective.gate_radii[:, None, :],
    )
    target_yz = state.target_crossing[..., 2:4]
    target_elliptical_radius = _elliptical_radius(target_yz, objective.target_center, objective.target_radii)
    centered = target_yz - target_yz.mean(dim=0)
    spread_rms = torch.sqrt(centered.square().sum(dim=-1).mean())
    invariant = minkowski_norm_squared(state.trajectory.four_velocities)
    invariant_error = (invariant - 1.0).abs()
    slopes = state.target_velocity[..., 2:4] / state.target_velocity[..., 1:2]
    crossing_invariant_error = (minkowski_norm_squared(state.target_velocity) - 1.0).abs()
    return {
        "gate_centroid_errors": torch.linalg.vector_norm(gate_yz.mean(dim=1) - objective.gate_centers, dim=-1),
        "gate_max_elliptical_radius": gate_radius.max(dim=1).values,
        "gate_pass_fraction": (gate_radius <= 1.0).to(gate_radius.dtype).mean(dim=1),
        "target_centroid_error": torch.linalg.vector_norm(target_yz.mean(dim=0) - objective.target_center),
        "target_rms_spread": spread_rms,
        "target_max_elliptical_radius": target_elliptical_radius.max(),
        "target_pass_fraction": (target_elliptical_radius <= 1.0).to(target_yz.dtype).mean(),
        "target_transverse_slope_rms": torch.sqrt(slopes.square().sum(dim=-1).mean()),
        "target_gamma_mean": state.target_velocity[..., 0].mean(),
        "target_gamma_std": state.target_velocity[..., 0].std(unbiased=False),
        "mass_shell_max_error": invariant_error.max(),
        "mass_shell_rms_error": torch.sqrt(invariant_error.square().mean()),
        "target_crossing_mass_shell_max_error": crossing_invariant_error.max(),
        "minimum_longitudinal_step": (
            state.trajectory.worldlines[1:, ..., 1] - state.trajectory.worldlines[:-1, ..., 1]
        ).min(),
    }


def _to_python(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach())
        return value.detach().cpu().tolist()
    return value


def _refinement_pair(
    coarse_state: ObjectiveState,
    fine_state: ObjectiveState,
    coarse_metrics: dict[str, Any],
    fine_metrics: dict[str, Any],
) -> dict[str, Any]:
    gate_delta = torch.linalg.vector_norm(
        coarse_state.gate_crossings[..., 2:4] - fine_state.gate_crossings[..., 2:4], dim=-1
    )
    target_delta = torch.linalg.vector_norm(
        coarse_state.target_crossing[..., 2:4] - fine_state.target_crossing[..., 2:4], dim=-1
    )
    coarse_centroid = coarse_state.target_crossing[..., 2:4].mean(dim=0)
    fine_centroid = fine_state.target_crossing[..., 2:4].mean(dim=0)
    gamma_delta = coarse_state.target_velocity[..., 0] - fine_state.target_velocity[..., 0]
    return {
        "gate_crossing_rms_difference": float(torch.sqrt(gate_delta.square().mean())),
        "gate_crossing_max_difference": float(gate_delta.max()),
        "gate_crossing_rms_difference_per_gate": _to_python(torch.sqrt(gate_delta.square().mean(dim=1))),
        "target_crossing_rms_difference": float(torch.sqrt(target_delta.square().mean())),
        "target_crossing_max_difference": float(target_delta.max()),
        "target_centroid_shift": float(torch.linalg.vector_norm(coarse_centroid - fine_centroid)),
        "target_spread_absolute_difference": abs(
            float(coarse_metrics["target_rms_spread"]) - float(fine_metrics["target_rms_spread"])
        ),
        "target_gamma_mean_absolute_difference": abs(
            float(coarse_metrics["target_gamma_mean"]) - float(fine_metrics["target_gamma_mean"])
        ),
        "target_gamma_rms_difference": float(torch.sqrt(gamma_delta.square().mean())),
        "mass_shell_max_error_coarse": float(coarse_metrics["mass_shell_max_error"]),
        "mass_shell_max_error_fine": float(fine_metrics["mass_shell_max_error"]),
        "mass_shell_max_error_absolute_difference": abs(
            float(coarse_metrics["mass_shell_max_error"]) - float(fine_metrics["mass_shell_max_error"])
        ),
    }


def run_refinement_study(
    config: Config,
    algebra: AlgebraContext,
    restored_field: InvertibleBivectorField,
    beamline: Beamline,
    heldout_beam: Beam,
) -> tuple[dict[str, Any], dict[str, ObjectiveState]]:
    """Evaluate one restored field and held-out bunch at h, h/2, and h/4."""
    states: dict[str, ObjectiveState] = {}
    metrics: dict[str, dict[str, Any]] = {}
    levels: dict[str, Any] = {}
    total_proper_time = config.proper_time_step * config.proper_time_steps
    with torch.no_grad():
        for label, factor in (("h", 1), ("h_over_2", 2), ("h_over_4", 4)):
            level_config = replace(
                config,
                particles_per_axis=config.heldout_particles_per_axis,
                proper_time_step=config.proper_time_step / factor,
                proper_time_steps=config.proper_time_steps * factor,
            )
            level_field = build_field(level_config, algebra=algebra)
            level_field.latent_coordinates.copy_(restored_field.latent_coordinates)
            level_objective = BeamlineObjective(level_config, beamline, heldout_beam)
            state = level_objective(level_field)
            level_metrics = {
                name: _to_python(value) for name, value in _trajectory_metrics(level_objective, state).items()
            }
            states[label] = state
            metrics[label] = level_metrics
            levels[label] = {
                "proper_time_step": level_config.proper_time_step,
                "proper_time_steps": level_config.proper_time_steps,
                "total_proper_time": level_config.proper_time_step * level_config.proper_time_steps,
                "metrics": level_metrics,
            }

    coarse_to_half = _refinement_pair(states["h"], states["h_over_2"], metrics["h"], metrics["h_over_2"])
    half_to_quarter = _refinement_pair(states["h_over_2"], states["h_over_4"], metrics["h_over_2"], metrics["h_over_4"])
    target_ratio = half_to_quarter["target_crossing_rms_difference"] / max(
        coarse_to_half["target_crossing_rms_difference"], torch.finfo(algebra.dtype).tiny
    )
    gate_ratio = half_to_quarter["gate_crossing_rms_difference"] / max(
        coarse_to_half["gate_crossing_rms_difference"], torch.finfo(algebra.dtype).tiny
    )
    meaningful_trend = target_ratio < 0.8 and gate_ratio < 0.8
    if meaningful_trend:
        characterization = (
            "Successive gate- and target-crossing changes decrease under refinement; "
            "this is observed convergence behavior, not a formal order estimate."
        )
    else:
        characterization = (
            "Successive crossing changes do not both decrease clearly; results show bounded "
            "discretization sensitivity but not convincing convergence."
        )
    return (
        {
            "uses_optimizer": False,
            "same_restored_field": True,
            "same_heldout_bunch": True,
            "total_proper_time": total_proper_time,
            "levels": levels,
            "pairwise": {
                "h_to_h_over_2": coarse_to_half,
                "h_over_2_to_h_over_4": half_to_quarter,
            },
            "successive_target_crossing_rms_ratio": target_ratio,
            "successive_gate_crossing_rms_ratio": gate_ratio,
            "meaningful_observed_convergence_trend": meaningful_trend,
            "characterization": characterization,
            "formal_order_claimed": False,
        },
        states,
    )


def evaluate(
    config: Config,
    algebra: AlgebraContext,
    field_model: InvertibleBivectorField,
    objective: BeamlineObjective,
    initial_state: ObjectiveState,
    optimized_state: ObjectiveState,
    optimization: OptimizationResult,
    heldout_objective: BeamlineObjective,
    refinement: dict[str, Any],
    refinement_states: dict[str, ObjectiveState],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    initial_metrics = {name: _to_python(value) for name, value in _trajectory_metrics(objective, initial_state).items()}
    final_metrics = {name: _to_python(value) for name, value in _trajectory_metrics(objective, optimized_state).items()}
    heldout_metrics = refinement["levels"]["h"]["metrics"]
    heldout_state = refinement_states["h"]
    sampler = field_model.generator_sampler
    if not isinstance(sampler, BeamlineFieldSampler):
        raise TypeError("expected BeamlineFieldSampler")
    controls = sampler.control_values(field_model.latent_coordinates).detach()
    report: dict[str, Any] = {
        "experiment": "relativistic_beamline_design",
        "algebra": "Cl(1,3)",
        "signature": "+---",
        "problem": {
            "natural_units": "c=q/m=1",
            "optimization_particles": int(objective.beam.position.shape[0]),
            "proper_time_steps": config.proper_time_steps,
            "proper_time_step": config.proper_time_step,
            "total_proper_time": config.proper_time_steps * config.proper_time_step,
            "field_control_sites": config.control_sites,
            "field_parameterization": "C1 compact two-site cubic interpolation along x",
            "crossing_strategy": "first forward crossing with a fractional in-step Lorentz rotor",
            "intermediate_gate_centroid_equality_objective": False,
            "intermediate_gate_objective": "softened elliptical aperture penalty; centroid errors are diagnostics",
            "target_centroid_equality_objective": False,
            "target_objective": "softened target-region penalty and focus; centroid error is diagnostic",
            "gate_aperture_soft_limit": config.gate_aperture_soft_limit,
            "target_region_soft_limit": config.target_region_soft_limit,
            "target_gamma": config.target_gamma,
        },
        "optimization": {
            "algorithm": "Adam with cosine learning-rate decay",
            "steps": config.optimization_steps,
            "initial_loss": optimization.history[0],
            "best_step": optimization.best_step,
            "best_loss": optimization.best_loss,
            "final_history_loss": optimization.final_history_loss,
            "history": optimization.history,
        },
        "primary": {"initial": initial_metrics, "optimized": final_metrics},
        "validation": {
            "heldout_dense_bunch": {
                "particles": int(heldout_objective.beam.position.shape[0]),
                "uses_optimizer": False,
                "sampling": (
                    f"off-grid shifted {config.heldout_particles_per_axis}x{config.heldout_particles_per_axis} "
                    "transverse phase-space grid with no training particles"
                ),
                **heldout_metrics,
            },
            "integration_refinement": refinement,
        },
        "field": {
            "control_x": sampler.control_x.detach().cpu().tolist(),
            "optimized_controls": controls.cpu().tolist(),
            "control_channels": ["Ex", "Ey", "Ez", "steer_e12", "steer_e13", "quadrupole"],
            "control_rms": float(torch.sqrt(controls.square().mean())),
            "control_max": float(controls.abs().max()),
            "control_smoothness_rms": float(torch.sqrt((controls[1:] - controls[:-1]).square().mean())),
            "maximum_sampled_field_rate": float(
                torch.linalg.vector_norm(optimized_state.trajectory.field_rates, dim=-1).max()
            ),
        },
        "limitations": [
            "The prescribed compact field is a synthetic control model rather than a Maxwell or fringe-field solution.",
            "Three resolutions demonstrate observed sensitivity but do not establish a formal convergence order.",
            "Apertures and target are idealized geometric regions in natural units; optimization uses smooth boundary penalties.",
        ],
    }
    report["checks"] = {
        "optimized beamline is feasible": (
            min(final_metrics["gate_pass_fraction"]) >= 0.999
            and final_metrics["target_rms_spread"] < 0.060
            and final_metrics["target_pass_fraction"] >= 0.999
            and abs(final_metrics["target_gamma_mean"] - config.target_gamma) < 0.035
            and final_metrics["minimum_longitudinal_step"] > 0.0
        ),
        "off-grid held-out bunch remains feasible and focused": (
            min(heldout_metrics["gate_pass_fraction"]) >= 0.99
            and heldout_metrics["target_pass_fraction"] >= 0.98
            and heldout_metrics["target_rms_spread"] < 0.070
        ),
        "Lorentz and mass-shell invariants are preserved": (
            final_metrics["mass_shell_max_error"] < 2e-10
            and final_metrics["target_crossing_mass_shell_max_error"] < 2e-10
            and heldout_metrics["mass_shell_max_error"] < 2e-10
            and max(level["metrics"]["mass_shell_max_error"] for level in refinement["levels"].values()) < 2e-10
        ),
        "successive crossing differences decrease under integration refinement": refinement[
            "meaningful_observed_convergence_trend"
        ],
    }
    arrays = {
        "initial_worldlines": initial_state.trajectory.worldlines.detach(),
        "heldout_worldlines": heldout_state.trajectory.worldlines.detach(),
    }
    return report, arrays


def _np(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy()


def _draw_aperture(axis: Any, aperture: Aperture, *, alpha: float = 0.95) -> None:
    angle = np.linspace(0.0, 2.0 * np.pi, 96)
    tube_angle = np.linspace(0.0, 2.0 * np.pi, 14)
    aa, tt = np.meshgrid(angle, tube_angle, indexing="ij")
    thickness = 0.035
    x = aperture.x + thickness * np.sin(tt)
    y = aperture.center_yz[0] + (aperture.radii_yz[0] + thickness * np.cos(tt)) * np.cos(aa)
    z = aperture.center_yz[1] + (aperture.radii_yz[1] + thickness * np.cos(tt)) * np.sin(aa)
    axis.plot_surface(
        x,
        y,
        z,
        color=aperture.color,
        alpha=0.84 * alpha,
        linewidth=0.12,
        edgecolor=(0.15, 0.15, 0.15, 0.18),
        shade=True,
        antialiased=True,
    )


def _draw_target(axis: Any, beamline: Beamline) -> None:
    target = Aperture(
        beamline.target_x,
        beamline.target_center_yz,
        beamline.target_radii_yz,
        "#2a9d8f",
    )
    _draw_aperture(axis, target, alpha=1.0)
    axis.scatter(
        [beamline.target_x],
        [beamline.target_center_yz[0]],
        [beamline.target_center_yz[1]],
        marker="*",
        s=120,
        color="#006d77",
        depthshade=False,
    )


def _configure_bundle_axis(axis: Any, title: str, beamline: Beamline) -> None:
    for aperture in beamline.apertures:
        _draw_aperture(axis, aperture)
    _draw_target(axis, beamline)
    axis.set_title(title, fontweight="bold")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_xlim(-0.1, 6.65)
    axis.set_ylim(-0.25, 1.02)
    axis.set_zlim(-0.48, 0.46)
    axis.set_box_aspect((2.0, 0.75, 0.62))
    axis.view_init(elev=21, azim=-67)
    axis.set_proj_type("persp", focal_length=0.82)


class LiveView:
    """Persistent beamline geometry with dynamic particle worldlines."""

    def __init__(self, beamline: Beamline, particle_count: int):
        import matplotlib.pyplot as plt

        self.plt = plt
        self._last_event_time = 0.0
        self.closed = False
        self.figure = plt.figure(figsize=(11.5, 7.4))
        self.interactive = self.figure.canvas.required_interactive_framework is not None
        if self.figure.canvas.manager is not None:
            self.figure.canvas.manager.set_window_title("Relativistic beamline design")
        self.axis = self.figure.add_subplot(111, projection="3d")
        self.figure.canvas.mpl_connect("close_event", self._mark_closed)
        _configure_bundle_axis(self.axis, "Current particle bundle", beamline)
        colors = plt.colormaps["viridis"](np.linspace(0.08, 0.92, particle_count))
        self.lines = [self.axis.plot([], [], [], color=color, alpha=0.76, linewidth=1.0)[0] for color in colors]
        self.status = self.figure.text(0.5, 0.025, "initializing", ha="center", family="monospace", fontsize=9)
        self.figure.suptitle("Relativistic beamline inverse design", fontsize=15, fontweight="bold")
        self.figure.subplots_adjust(left=0.03, right=0.97, bottom=0.08, top=0.90)
        if self.interactive:
            plt.show(block=False)
        self.figure.canvas.draw_idle()
        self.pump_events()

    def pump_events(self) -> None:
        if self.closed or not self.interactive:
            return
        now = time.monotonic()
        if now - self._last_event_time < 0.05:
            return
        self._last_event_time = now
        try:
            self.figure.canvas.flush_events()
        except (RuntimeError, SystemError):
            self.closed = True

    def update(self, step: int | str, state: ObjectiveState) -> None:
        if self.closed:
            return
        worldlines = _np(state.trajectory.worldlines)[..., 1:]
        for line, particle in zip(self.lines, worldlines.transpose(1, 0, 2)):
            line.set_data_3d(particle[:, 0], particle[:, 1], particle[:, 2])
        label = f"step {step:4d}" if isinstance(step, int) else str(step)
        self.status.set_text(f"{label} · loss {state.loss.item():.3e}")
        try:
            self.figure.canvas.draw_idle()
        except (RuntimeError, SystemError):
            self.closed = True

    def _mark_closed(self, _event: Any) -> None:
        self.closed = True

    def keep_open(self) -> None:
        if self.interactive and not self.closed:
            self.plt.show(block=True)


def save_result(
    config: Config,
    beamline: Beamline,
    arrays: dict[str, torch.Tensor],
) -> Path:
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(14.8, 7.2), constrained_layout=True)
    FigureCanvasAgg(figure)
    initial_axis = figure.add_subplot(121, projection="3d")
    optimized_axis = figure.add_subplot(122, projection="3d")
    initial = _np(arrays["initial_worldlines"])[..., 1:]
    optimized = _np(arrays["heldout_worldlines"])[..., 1:]
    for particle, color in enumerate(colormaps["viridis"](np.linspace(0.08, 0.92, initial.shape[1]))):
        initial_axis.plot(*initial[:, particle].T, color=color, alpha=0.46, linewidth=0.8)
    for particle, color in enumerate(colormaps["viridis"](np.linspace(0.08, 0.92, optimized.shape[1]))):
        optimized_axis.plot(*optimized[:, particle].T, color=color, alpha=0.70, linewidth=0.9)
    _configure_bundle_axis(initial_axis, "Initial zero-field bunch", beamline)
    _configure_bundle_axis(optimized_axis, "Held-out optimized dense bunch", beamline)
    figure.suptitle("Relativistic beamline inverse design", fontsize=17, fontweight="bold")
    figure.text(
        0.5,
        0.94,
        "Matched view · three offset apertures · target at x = 6.35 · no held-out retraining",
        ha="center",
        color="#4b5563",
        fontsize=10,
    )
    figure.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.90))
    output_path = config.output_dir / "result.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=185)
    return output_path


def save_diagnostics(
    config: Config,
    optimization: OptimizationResult,
    report: dict[str, Any],
) -> Path:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(14.8, 4.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    history_axis, aperture_axis, refinement_axis = axes
    history_axis.semilogy(optimization.history, color="#264653", linewidth=2.0)
    history_axis.scatter(
        [optimization.best_step], [optimization.best_loss], color="#e76f51", s=26, zorder=3, label="restored best"
    )
    history_axis.set_title("Optimization", fontweight="bold")
    history_axis.set_xlabel("Adam step")
    history_axis.set_ylabel("objective")
    history_axis.grid(alpha=0.22)
    history_axis.legend(fontsize=8, frameon=False)

    initial = report["primary"]["initial"]
    heldout = report["validation"]["heldout_dense_bunch"]
    x = np.arange(4)
    width = 0.34
    initial_radii = np.array((*initial["gate_max_elliptical_radius"], initial["target_max_elliptical_radius"]))
    heldout_radii = np.array((*heldout["gate_max_elliptical_radius"], heldout["target_max_elliptical_radius"]))
    aperture_axis.bar(x - width / 2, initial_radii, width, color="#adb5bd", label="initial")
    aperture_axis.bar(x + width / 2, heldout_radii, width, color="#2a9d8f", label="held-out")
    aperture_axis.axhline(1.0, color="#111827", linestyle="--", linewidth=1.3, label="region edge")
    aperture_axis.set_xticks(x, ("gate 1", "gate 2", "gate 3", "target"), rotation=15)
    aperture_axis.set_ylabel("max normalized radius")
    aperture_axis.set_title("Held-out geometric margins", fontweight="bold")
    aperture_axis.grid(axis="y", alpha=0.22)
    aperture_axis.legend(fontsize=8, frameon=False)

    refinement = report["validation"]["integration_refinement"]
    pairwise = refinement["pairwise"]
    step_sizes = np.array((config.proper_time_step, 0.5 * config.proper_time_step))
    target_mm = 1000.0 * np.array(
        (
            pairwise["h_to_h_over_2"]["target_crossing_rms_difference"],
            pairwise["h_over_2_to_h_over_4"]["target_crossing_rms_difference"],
        )
    )
    gate_mm = 1000.0 * np.array(
        (
            pairwise["h_to_h_over_2"]["gate_crossing_rms_difference"],
            pairwise["h_over_2_to_h_over_4"]["gate_crossing_rms_difference"],
        )
    )
    refinement_axis.semilogy(step_sizes, gate_mm, "o-", color="#457b9d", linewidth=2.0, label="gate crossings")
    refinement_axis.semilogy(step_sizes, target_mm, "o-", color="#e76f51", linewidth=2.0, label="target crossings")
    refinement_axis.invert_xaxis()
    refinement_axis.set_xticks(step_sizes, ("h", "h/2"))
    refinement_axis.set_xlabel("coarser step in pair")
    refinement_axis.set_ylabel("successive RMS difference [mm]")
    refinement_axis.set_title("Integration refinement", fontweight="bold")
    refinement_axis.grid(alpha=0.22, which="both")
    refinement_axis.legend(fontsize=8, frameon=False)

    figure.suptitle("Relativistic beamline diagnostics", fontsize=16, fontweight="bold")
    fine_pair = pairwise["h_over_2_to_h_over_4"]
    figure.text(
        0.5,
        0.015,
        f"{heldout['particles']} held-out particles · {heldout['target_pass_fraction']:.0%} target pass · "
        f"h/2↔h/4 target ΔRMS {1000 * fine_pair['target_crossing_rms_difference']:.2f} mm",
        ha="center",
        color="#4b5563",
        fontsize=9,
    )
    figure.get_layout_engine().set(rect=(0.0, 0.06, 1.0, 0.92))
    output_path = config.output_dir / "diagnostics.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=185)
    return output_path


def print_report(report: dict[str, Any]) -> None:
    initial, final = report["primary"]["initial"], report["primary"]["optimized"]
    heldout = report["validation"]["heldout_dense_bunch"]
    refinement = report["validation"]["integration_refinement"]
    optimization = report["optimization"]
    print("\nRELATIVISTIC CLIFFORD BEAMLINE DESIGN\n")
    print(
        "primary      aperture radii "
        + ", ".join(
            f"{before:.2f}→{after:.2f}"
            for before, after in zip(initial["gate_max_elliptical_radius"], final["gate_max_elliptical_radius"])
        )
    )
    print(
        f"validation   {heldout['particles']} held-out particles | target {heldout['target_pass_fraction']:.1%} | "
        f"spread {1000 * heldout['target_rms_spread']:.2f} mm | centroid {1000 * heldout['target_centroid_error']:.2f} mm"
    )
    pair_a = refinement["pairwise"]["h_to_h_over_2"]
    pair_b = refinement["pairwise"]["h_over_2_to_h_over_4"]
    print(
        f"refinement   target ΔRMS {1000 * pair_a['target_crossing_rms_difference']:.2f}→"
        f"{1000 * pair_b['target_crossing_rms_difference']:.2f} mm | gate ΔRMS "
        f"{1000 * pair_a['gate_crossing_rms_difference']:.2f}→"
        f"{1000 * pair_b['gate_crossing_rms_difference']:.2f} mm"
    )
    print(
        f"numerical    gamma {heldout['target_gamma_mean']:.4f} | "
        f"mass shell {heldout['mass_shell_max_error']:.1e} | "
        f"h/2↔h/4 mass-shell {refinement['levels']['h_over_4']['metrics']['mass_shell_max_error']:.1e}"
    )
    print(
        f"optimization best {optimization['best_loss']:.3e} at {optimization['best_step']} | "
        f"final {optimization['final_history_loss']:.3e}\n"
    )
    for name, passed in report["checks"].items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")


def run(config: Config = Config()) -> dict[str, Any]:
    device, dtype = torch.device("cpu"), torch.float64
    algebra = AlgebraContext(1, 3, device=device, dtype=dtype)
    beamline = build_beamline()
    beam = build_initial_beam(config, device=device, dtype=dtype)
    field_model = build_field(config, algebra=algebra)
    objective = BeamlineObjective(config, beamline, beam)

    with torch.no_grad():
        initial_state = objective(field_model)
    print(
        f"setup         {config.particles_per_axis**2} optimization particles | {config.control_sites} field sites | "
        f"{config.proper_time_steps} steps at h={config.proper_time_step:g}"
    )
    view = LiveView(beamline, beam.position.shape[0]) if config.live else None
    optimization = optimize(config, field_model, objective, initial_state, view)
    with torch.no_grad():
        optimized_state = objective(field_model)
        heldout_config = replace(config, particles_per_axis=config.heldout_particles_per_axis)
        heldout_beam = build_initial_beam(
            heldout_config,
            device=device,
            dtype=dtype,
            normalized_grid_shift=(0.019, -0.013),
        )
        heldout_objective = BeamlineObjective(config, beamline, heldout_beam)
    refinement, refinement_states = run_refinement_study(config, algebra, field_model, beamline, heldout_beam)
    report, arrays = evaluate(
        config,
        algebra,
        field_model,
        objective,
        initial_state,
        optimized_state,
        optimization,
        heldout_objective,
        refinement,
        refinement_states,
    )
    if view is not None:
        view.update("restored best", optimized_state)
    result_path = save_result(config, beamline, arrays)
    diagnostics_path = save_diagnostics(config, optimization, report)
    report_path = config.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print_report(report)
    print("\nARTIFACTS")
    print(f"result       {result_path}")
    print(f"diagnostics  {diagnostics_path}")
    print(f"report       {report_path}")
    if view is not None:
        view.keep_open()
    return report


def _parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=Config.optimization_steps)
    parser.add_argument("--output-dir", type=Path, default=Config.output_dir)
    parser.add_argument("--live", action="store_true", help="show the particle bundle evolving during optimization")
    args = parser.parse_args()
    return replace(Config(), optimization_steps=args.steps, output_dir=args.output_dir, live=args.live)


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
