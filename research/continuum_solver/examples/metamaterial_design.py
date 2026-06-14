# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""PGA metamaterial design with local criteria injected into the solver."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    from _common import add_runtime_arguments, bootstrap_repo_root, configure_runtime, coordinate_grid_3d, print_latest, resolve_runtime
else:
    from ._common import add_runtime_arguments, bootstrap_repo_root, configure_runtime, coordinate_grid_3d, print_latest, resolve_runtime

bootstrap_repo_root(__file__)

from clifra.core.runtime.algebra import AlgebraContext
from research.continuum_solver import (
    BivectorNormPolicy,
    ContinuumSolverEngine,
    CriterionResult,
    InvertibleBivectorField,
    InvertiblePathConsistencyPolicy,
    PolicyResult,
)


@dataclass(frozen=True)
class PhaseInversionPoissonCriterion:
    """Drive a core-to-rim Poisson-ratio inversion using log strain ratios."""

    core_target: float = -0.6
    rim_target: float = 0.5
    core_radius: float = 0.40
    rim_radius: float = 0.78
    axial_log_strain: float = 0.035
    poisson_weight: float = 1.0
    axial_weight: float = 0.2
    ramp_steps: int = 12
    ramp_start: float = 0.15
    name: str = "phase_inversion_poisson"

    def __call__(self, engine, state) -> CriterionResult:
        log_sx, log_sy, log_sz = _cell_log_stretches(state.reference_coordinates, state.deformed_coordinates)
        centers = _cell_centers(state.reference_coordinates)
        radius = _normalized_radius(centers)
        target = _phase_target(
            radius,
            core_target=self.core_target,
            rim_target=self.rim_target,
            core_radius=self.core_radius,
            rim_radius=self.rim_radius,
        )
        poisson = -0.5 * (log_sy + log_sz) / _signed_clamp(log_sx, 1e-4)
        poisson_loss = (poisson - target).square().mean()
        axial_target = torch.as_tensor(self.axial_log_strain, device=log_sx.device, dtype=log_sx.dtype)
        axial_loss = (log_sx - axial_target).square().mean()
        ramp = _ramp_multiplier(engine, log_sx, steps=self.ramp_steps, start=self.ramp_start)
        loss = ramp * (float(self.poisson_weight) * poisson_loss + float(self.axial_weight) * axial_loss)

        core = (radius <= self.core_radius).to(poisson.dtype)
        transition = ((radius > self.core_radius) & (radius < self.rim_radius)).to(poisson.dtype)
        rim = (radius >= self.rim_radius).to(poisson.dtype)
        return CriterionResult(
            name=self.name,
            loss=loss,
            metrics={
                "core_poisson": _masked_mean(poisson, core),
                "transition_poisson": _masked_mean(poisson, transition),
                "rim_poisson": _masked_mean(poisson, rim),
                "core_target": float(self.core_target),
                "rim_target": float(self.rim_target),
                "mean_axial_log_strain": log_sx.mean(),
                "poisson_rmse": poisson_loss.sqrt(),
                "weight_ramp": ramp,
            },
        )


@dataclass(frozen=True)
class TransitionReversibilityPolicy:
    """Keep the phase transition reversible and smooth."""

    core_radius: float = 0.40
    rim_radius: float = 0.78
    reverse_weight: float = 1.0
    smoothness_weight: float = 0.1
    weight: float = 1.0
    strict_tolerance: float = 5e-4
    name: str = "transition_reversibility"

    def __call__(self, engine, state) -> PolicyResult:
        point_radius = _normalized_radius(state.reference_coordinates)
        point_mask = ((point_radius > self.core_radius) & (point_radius < self.rim_radius)).to(state.reference_coordinates.dtype)
        reconstructed = engine.field.inverse(state.deformed_coordinates)
        residual = torch.linalg.vector_norm(reconstructed - state.reference_coordinates, dim=-1)
        reverse_loss = _masked_mean(residual.square(), point_mask)

        log_sx, log_sy, log_sz = _cell_log_stretches(state.reference_coordinates, state.deformed_coordinates)
        poisson = -0.5 * (log_sy + log_sz) / _signed_clamp(log_sx, 1e-4)
        smoothness = _gradient_energy3_scalar(poisson)
        loss = float(self.reverse_weight) * reverse_loss + float(self.smoothness_weight) * smoothness
        max_residual = (residual * point_mask).amax()
        return PolicyResult(
            name=self.name,
            loss=loss,
            weight=self.weight,
            strict_tolerance=self.strict_tolerance,
            metrics={
                "transition_reverse_rmse": reverse_loss.sqrt(),
                "poisson_gradient_energy": smoothness,
                "max_transition_reverse_residual": max_residual,
            },
            violations={"max_transition_reverse_residual": max_residual},
        )


@dataclass(frozen=True)
class AlternatingMicroVortexPolicy:
    """Enforce alternating local chirality while clamping the outer boundary."""

    target_vorticity: float = 0.08
    clamp_weight: float = 30.0
    vortex_weight: float = 1.0
    ramp_steps: int = 12
    ramp_start: float = 0.15
    weight: float = 1.0
    strict_tolerance: float = 2e-3
    name: str = "alternating_micro_vortex"

    def __call__(self, engine, state) -> PolicyResult:
        displacement = state.deformed_coordinates - state.reference_coordinates
        curl_z = _curl_z(displacement, state.reference_coordinates)
        target = float(self.target_vorticity) * _checkerboard(curl_z.shape, device=curl_z.device, dtype=curl_z.dtype)
        interior = _interior_cell_mask(curl_z.shape, device=curl_z.device, dtype=curl_z.dtype)
        vortex_delta = (curl_z - target) * interior
        vortex_loss = _masked_mean(vortex_delta.square(), interior)

        boundary = _boundary_point_mask(state.reference_coordinates.shape[:-1], device=displacement.device, dtype=displacement.dtype)
        displacement_norm_sq = displacement.square().sum(dim=-1)
        clamp_loss = _masked_mean(displacement_norm_sq, boundary)
        ramp = _ramp_multiplier(engine, displacement, steps=self.ramp_steps, start=self.ramp_start)
        loss = ramp * float(self.vortex_weight) * vortex_loss + float(self.clamp_weight) * clamp_loss
        max_vortex_error = (vortex_delta.abs() * interior).amax()
        max_clamp = (displacement_norm_sq.sqrt() * boundary).amax()
        max_violation = torch.maximum(max_vortex_error, max_clamp)
        return PolicyResult(
            name=self.name,
            loss=loss,
            weight=self.weight,
            strict_tolerance=self.strict_tolerance,
            metrics={
                "mean_abs_vorticity": _masked_mean(curl_z.abs(), interior),
                "target_abs_vorticity": float(abs(self.target_vorticity)),
                "clamp_rmse": clamp_loss.sqrt(),
                "max_clamp_displacement": max_clamp,
                "vortex_weight_ramp": ramp,
            },
            violations={"max_vortex_or_clamp_error": max_violation},
        )


@dataclass(frozen=True)
class BucklingSubtractionPolicy:
    """Suppress folds and high-frequency buckling through volume and Laplacian energy."""

    min_jacobian: float = 0.08
    max_laplacian_energy: float = 6e-3
    fold_weight: float = 80.0
    laplacian_weight: float = 1.0
    weight: float = 1.0
    strict_tolerance: float = 1e-4
    name: str = "buckling_subtraction"

    def __call__(self, engine, state) -> PolicyResult:
        displacement = state.deformed_coordinates - state.reference_coordinates
        laplacian = _laplacian3(displacement)
        curvature = torch.linalg.vector_norm(laplacian, dim=-1)
        laplacian_energy = curvature.square().mean()

        jacobian = _volume_jacobian(state.reference_coordinates, state.deformed_coordinates)
        jacobian_shortfall = F.relu(float(self.min_jacobian) - jacobian)
        energy_excess = F.relu(laplacian_energy - float(self.max_laplacian_energy))
        loss = (
            float(self.fold_weight) * jacobian_shortfall.square().mean()
            + float(self.laplacian_weight) * laplacian_energy
            + energy_excess.square()
        )
        max_violation = torch.maximum(jacobian_shortfall.amax(), energy_excess)
        return PolicyResult(
            name=self.name,
            loss=loss,
            weight=self.weight,
            strict_tolerance=self.strict_tolerance,
            metrics={
                "laplacian_energy": laplacian_energy,
                "max_curvature": curvature.amax(),
                "min_jacobian": jacobian.amin(),
                "mean_jacobian": jacobian.mean(),
                "folded_cell_fraction": (jacobian <= 0).to(jacobian.dtype).mean(),
            },
            violations={"max_fold_or_energy_excess": max_violation},
        )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--path-steps", type=int, default=3)
    parser.add_argument("--control-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.2e-2)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--ramp-steps", type=int, default=12)
    parser.add_argument("--ramp-start", type=float, default=0.15)
    parser.add_argument("--frames", type=int, default=18)
    add_runtime_arguments(parser)
    args = parser.parse_args()
    runtime = resolve_runtime(args)
    configure_runtime(seed=31, max_threads=args.max_threads)

    size = int(args.grid_size)
    coords = coordinate_grid_3d(size, size, size, device=runtime.device, dtype=runtime.dtype)
    algebra = AlgebraContext(p=3, q=0, r=1, device=runtime.device, dtype=runtime.dtype)
    field = InvertibleBivectorField(
        algebra,
        coordinate_dim=3,
        projective=True,
        path_steps=args.path_steps,
        control_shape=(args.control_size, args.control_size, args.control_size),
        init_scale=1e-2,
    )
    engine = ContinuumSolverEngine(
        field,
        target_criterion=PhaseInversionPoissonCriterion(ramp_steps=args.ramp_steps, ramp_start=args.ramp_start),
        geometric_policies=(
            TransitionReversibilityPolicy(),
            AlternatingMicroVortexPolicy(ramp_steps=args.ramp_steps, ramp_start=args.ramp_start),
            BucklingSubtractionPolicy(),
            InvertiblePathConsistencyPolicy(weight=0.5),
            BivectorNormPolicy(max_norm=1.4, weight=1e-2),
        ),
    )

    run = engine.fit(
        coords,
        steps=args.steps,
        lr=args.lr,
        log_every=args.log_every or max(1, args.steps // 10),
        clip_grad_norm=args.clip_grad_norm,
        compile_step=runtime.compile_step,
        compile_backend=runtime.compile_backend,
        compile_mode=runtime.compile_mode,
        compile_fullgraph=runtime.compile_fullgraph,
    )
    print_latest(run.history, title="PGA Metamaterial Design", limit=36)


def _cell_edges_3d(coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if coordinates.ndim < 4 or coordinates.shape[-1] != 3:
        raise ValueError(f"expected a 3D coordinate grid with shape [..., z, y, x, 3], got {tuple(coordinates.shape)}")
    z_axis = coordinates.ndim - 4
    y_axis = coordinates.ndim - 3
    x_axis = coordinates.ndim - 2
    depth = coordinates.shape[z_axis]
    height = coordinates.shape[y_axis]
    width = coordinates.shape[x_axis]
    dx = coordinates.diff(dim=x_axis).narrow(z_axis, 0, depth - 1).narrow(y_axis, 0, height - 1)
    dy = coordinates.diff(dim=y_axis).narrow(z_axis, 0, depth - 1).narrow(x_axis, 0, width - 1)
    dz = coordinates.diff(dim=z_axis).narrow(y_axis, 0, height - 1).narrow(x_axis, 0, width - 1)
    return dx, dy, dz


def _cell_log_stretches(reference: torch.Tensor, deformed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ref_edges = _cell_edges_3d(reference)
    def_edges = _cell_edges_3d(deformed)
    logs = []
    for ref_edge, def_edge in zip(ref_edges, def_edges):
        ref_length = torch.linalg.vector_norm(ref_edge, dim=-1).clamp_min(1e-8)
        def_length = torch.linalg.vector_norm(def_edge, dim=-1).clamp_min(1e-8)
        logs.append(torch.log(def_length / ref_length))
    return logs[0], logs[1], logs[2]


def _cell_centers(coordinates: torch.Tensor) -> torch.Tensor:
    z_axis = coordinates.ndim - 4
    y_axis = coordinates.ndim - 3
    x_axis = coordinates.ndim - 2
    depth = coordinates.shape[z_axis]
    height = coordinates.shape[y_axis]
    width = coordinates.shape[x_axis]
    return 0.125 * (
        coordinates.narrow(z_axis, 0, depth - 1).narrow(y_axis, 0, height - 1).narrow(x_axis, 0, width - 1)
        + coordinates.narrow(z_axis, 1, depth - 1).narrow(y_axis, 0, height - 1).narrow(x_axis, 0, width - 1)
        + coordinates.narrow(z_axis, 0, depth - 1).narrow(y_axis, 1, height - 1).narrow(x_axis, 0, width - 1)
        + coordinates.narrow(z_axis, 0, depth - 1).narrow(y_axis, 0, height - 1).narrow(x_axis, 1, width - 1)
        + coordinates.narrow(z_axis, 1, depth - 1).narrow(y_axis, 1, height - 1).narrow(x_axis, 0, width - 1)
        + coordinates.narrow(z_axis, 1, depth - 1).narrow(y_axis, 0, height - 1).narrow(x_axis, 1, width - 1)
        + coordinates.narrow(z_axis, 0, depth - 1).narrow(y_axis, 1, height - 1).narrow(x_axis, 1, width - 1)
        + coordinates.narrow(z_axis, 1, depth - 1).narrow(y_axis, 1, height - 1).narrow(x_axis, 1, width - 1)
    )


def _normalized_radius(coordinates: torch.Tensor) -> torch.Tensor:
    radius = torch.linalg.vector_norm(coordinates, dim=-1)
    return radius / radius.detach().amax().clamp_min(1e-8)


def _phase_target(
    radius: torch.Tensor,
    *,
    core_target: float,
    rim_target: float,
    core_radius: float,
    rim_radius: float,
) -> torch.Tensor:
    t = ((radius - float(core_radius)) / max(float(rim_radius) - float(core_radius), 1e-8)).clamp(0.0, 1.0)
    smooth = t.square() * (3.0 - 2.0 * t)
    return float(core_target) + (float(rim_target) - float(core_target)) * smooth


def _curl_z(displacement: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    dx_u, dy_u, _ = _cell_edges_3d(displacement)
    dx_ref, dy_ref, _ = _cell_edges_3d(reference)
    step_x = torch.linalg.vector_norm(dx_ref, dim=-1).clamp_min(1e-8)
    step_y = torch.linalg.vector_norm(dy_ref, dim=-1).clamp_min(1e-8)
    return dx_u[..., 1] / step_x - dy_u[..., 0] / step_y


def _volume_jacobian(reference: torch.Tensor, deformed: torch.Tensor) -> torch.Tensor:
    dx_ref, dy_ref, dz_ref = _cell_edges_3d(reference)
    dx_def, dy_def, dz_def = _cell_edges_3d(deformed)
    ref_matrix = torch.stack((dx_ref, dy_ref, dz_ref), dim=-1)
    def_matrix = torch.stack((dx_def, dy_def, dz_def), dim=-1)
    det_ref = torch.linalg.det(ref_matrix)
    det_def = torch.linalg.det(def_matrix)
    return det_def / _signed_clamp(det_ref, 1e-8)


def _gradient_energy3_scalar(values: torch.Tensor) -> torch.Tensor:
    diffs = []
    for dim in range(values.ndim):
        if values.shape[dim] > 1:
            diffs.append(values.diff(dim=dim).square().mean())
    return torch.stack(diffs).mean() if diffs else values.new_zeros(())


def _laplacian3(values: torch.Tensor) -> torch.Tensor:
    z_axis = values.ndim - 4
    y_axis = values.ndim - 3
    x_axis = values.ndim - 2
    center = values.narrow(z_axis, 1, values.shape[z_axis] - 2).narrow(y_axis, 1, values.shape[y_axis] - 2).narrow(x_axis, 1, values.shape[x_axis] - 2)
    x_plus = values.narrow(z_axis, 1, values.shape[z_axis] - 2).narrow(y_axis, 1, values.shape[y_axis] - 2).narrow(x_axis, 2, values.shape[x_axis] - 2)
    x_minus = values.narrow(z_axis, 1, values.shape[z_axis] - 2).narrow(y_axis, 1, values.shape[y_axis] - 2).narrow(x_axis, 0, values.shape[x_axis] - 2)
    y_plus = values.narrow(z_axis, 1, values.shape[z_axis] - 2).narrow(y_axis, 2, values.shape[y_axis] - 2).narrow(x_axis, 1, values.shape[x_axis] - 2)
    y_minus = values.narrow(z_axis, 1, values.shape[z_axis] - 2).narrow(y_axis, 0, values.shape[y_axis] - 2).narrow(x_axis, 1, values.shape[x_axis] - 2)
    z_plus = values.narrow(z_axis, 2, values.shape[z_axis] - 2).narrow(y_axis, 1, values.shape[y_axis] - 2).narrow(x_axis, 1, values.shape[x_axis] - 2)
    z_minus = values.narrow(z_axis, 0, values.shape[z_axis] - 2).narrow(y_axis, 1, values.shape[y_axis] - 2).narrow(x_axis, 1, values.shape[x_axis] - 2)
    return x_plus + x_minus + y_plus + y_minus + z_plus + z_minus - 6.0 * center


def _checkerboard(shape: torch.Size, *, device, dtype) -> torch.Tensor:
    axes = torch.meshgrid(*(torch.arange(size, device=device) for size in shape), indexing="ij")
    parity = sum(axes).remainder(2)
    return torch.where(parity == 0, torch.ones((), device=device, dtype=dtype), -torch.ones((), device=device, dtype=dtype))


def _interior_cell_mask(shape: torch.Size, *, device, dtype) -> torch.Tensor:
    mask = torch.ones(tuple(shape), device=device, dtype=dtype)
    if len(shape) >= 3 and min(shape) > 2:
        mask[0, :, :] = 0.0
        mask[-1, :, :] = 0.0
        mask[:, 0, :] = 0.0
        mask[:, -1, :] = 0.0
        mask[:, :, 0] = 0.0
        mask[:, :, -1] = 0.0
    return mask


def _boundary_point_mask(shape: tuple[int, ...], *, device, dtype) -> torch.Tensor:
    mask = torch.zeros(shape, device=device, dtype=dtype)
    if len(shape) != 3:
        raise ValueError(f"expected a 3D point shape, got {shape}")
    mask[0, :, :] = 1.0
    mask[-1, :, :] = 1.0
    mask[:, 0, :] = 1.0
    mask[:, -1, :] = 1.0
    mask[:, :, 0] = 1.0
    mask[:, :, -1] = 1.0
    return mask


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _signed_clamp(values: torch.Tensor, eps: float) -> torch.Tensor:
    magnitude = values.abs().clamp_min(float(eps))
    return torch.where(values < 0, -magnitude, magnitude)


def _ramp_multiplier(engine, values: torch.Tensor, *, steps: int, start: float) -> torch.Tensor:
    steps = int(steps)
    if steps <= 0:
        return values.new_ones(())
    progress = min(1.0, max(0.0, (getattr(engine, "fit_step", 0) + 1) / float(steps)))
    start = min(1.0, max(0.0, float(start)))
    return values.new_tensor(start + (1.0 - start) * progress)


if __name__ == "__main__":
    main()
