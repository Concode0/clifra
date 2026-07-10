# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Strict PGA metamaterial design example with VTK visualization export.

The continuum solver stays generic; this script owns the metamaterial target,
physical policies, guarded optimization loop, validation, and visualization.
"""

from __future__ import annotations

import copy
import json
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


def _bootstrap_repo_root(file: str) -> None:
    root = Path(file).resolve().parents[3]
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


_bootstrap_repo_root(__file__)

from clifra.core.runtime.algebra import AlgebraContext
from research.continuum_solver import (
    BivectorNormPolicy,
    ContinuumSolverEngine,
    CriterionResult,
    InvertibleBivectorField,
    InvertiblePathConsistencyPolicy,
    PhaseCurriculum,
    PolicyResult,
)


DTYPE_CHOICES = ("float64", "float32")
DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")


@dataclass(frozen=True)
class MetamaterialTarget:
    """A difficult but physically admissible auxetic-to-positive phase target."""

    core_poisson: float = -0.65
    rim_poisson: float = 0.40
    core_radius: float = 0.34
    rim_radius: float = 0.84
    axial_log_strain: float = 0.052
    target_twist: float = 0.055
    poisson_weight: float = 3.0
    axial_weight: float = 1.5
    volume_weight: float = 1.0
    curl_weight: float = 0.8
    name: str = "phase_metamaterial"

    def __call__(self, engine: ContinuumSolverEngine, state) -> CriterionResult:
        log_sx, log_sy, log_sz = _cell_log_stretches(state.reference_coordinates, state.deformed_coordinates)
        centers = _cell_centers(state.reference_coordinates)
        radius = _normalized_radius(centers)
        phase = _smoothstep(engine.fit_progress_like(log_sx))
        poisson_target = phase * _phase_profile(
            radius,
            core_value=self.core_poisson,
            rim_value=self.rim_poisson,
            core_radius=self.core_radius,
            rim_radius=self.rim_radius,
        )
        axial_target = log_sx.new_tensor(float(self.axial_log_strain)) * phase
        log_volume_target = axial_target * (1.0 - 2.0 * poisson_target)

        poisson_residual = (log_sy + log_sz) + 2.0 * poisson_target * log_sx
        axial_residual = log_sx - axial_target
        volume_residual = (log_sx + log_sy + log_sz) - log_volume_target

        displacement = state.deformed_coordinates - state.reference_coordinates
        curl = _curl3(displacement, state.reference_coordinates)
        target_curl = torch.zeros_like(curl)
        target_curl[..., 2] = (
            phase
            * float(self.target_twist)
            * _checkerboard(curl.shape[:-1], device=curl.device, dtype=curl.dtype)
            * _interior_cell_mask(curl.shape[:-1], device=curl.device, dtype=curl.dtype)
        )
        curl_residual = curl - target_curl

        poisson_loss = poisson_residual.square().mean()
        axial_loss = axial_residual.square().mean()
        volume_loss = volume_residual.square().mean()
        curl_loss = curl_residual.square().sum(dim=-1).mean()
        total = (
            float(self.poisson_weight) * poisson_loss
            + float(self.axial_weight) * axial_loss
            + float(self.volume_weight) * volume_loss
            + float(self.curl_weight) * curl_loss
        )
        poisson = _poisson_ratio(log_sx, log_sy, log_sz)
        core = (radius <= self.core_radius).to(poisson.dtype)
        transition = ((radius > self.core_radius) & (radius < self.rim_radius)).to(poisson.dtype)
        rim = (radius >= self.rim_radius).to(poisson.dtype)
        return CriterionResult(
            name=self.name,
            loss=total,
            metrics={
                "phase": phase,
                "poisson_rmse": poisson_loss.sqrt(),
                "axial_rmse": axial_loss.sqrt(),
                "volume_rmse": volume_loss.sqrt(),
                "curl_rmse": curl_loss.sqrt(),
                "core_poisson": _masked_mean(poisson, core),
                "transition_poisson": _masked_mean(poisson, transition),
                "rim_poisson": _masked_mean(poisson, rim),
                "target_core_poisson": phase * float(self.core_poisson),
                "target_rim_poisson": phase * float(self.rim_poisson),
                "mean_axial_log_strain": log_sx.mean(),
                "mean_log_volume": (log_sx + log_sy + log_sz).mean(),
            },
        )


@dataclass(frozen=True)
class StrictPhysicalPolicy:
    """Soft loss counterpart to hard physical guards."""

    min_jacobian: float = 0.42
    max_jacobian: float = 2.35
    min_edge_stretch: float = 0.55
    max_edge_stretch: float = 1.85
    max_displacement: float = 0.82
    max_boundary_displacement: float = 0.13
    barrier_width: float = 0.025
    weight: float = 1.0
    strict_tolerance: float = 1e-4
    name: str = "strict_physical_realism"

    def __call__(self, engine: ContinuumSolverEngine, state) -> PolicyResult:
        del engine
        displacement = state.deformed_coordinates - state.reference_coordinates
        displacement_norm = torch.linalg.vector_norm(displacement, dim=-1)
        boundary = _boundary_point_mask(state.reference_coordinates.shape[:-1], device=displacement.device, dtype=displacement.dtype)
        boundary_displacement = displacement_norm * boundary

        jacobian = _volume_jacobian(state.reference_coordinates, state.deformed_coordinates)
        stretches = _edge_stretches(state.reference_coordinates, state.deformed_coordinates)
        short_j = _soft_shortfall(jacobian, self.min_jacobian, self.barrier_width)
        high_j = _soft_excess(jacobian, self.max_jacobian, self.barrier_width)
        short_edge = _soft_shortfall(stretches, self.min_edge_stretch, self.barrier_width)
        high_edge = _soft_excess(stretches, self.max_edge_stretch, self.barrier_width)
        high_disp = _soft_excess(displacement_norm, self.max_displacement, self.barrier_width)
        high_boundary = _soft_excess(boundary_displacement, self.max_boundary_displacement, self.barrier_width)

        loss = (
            20.0 * short_j.square().mean()
            + 8.0 * high_j.square().mean()
            + 6.0 * short_edge.square().mean()
            + 3.0 * high_edge.square().mean()
            + 2.0 * high_disp.square().mean()
            + 20.0 * _masked_mean(high_boundary.square(), boundary)
        )
        violation = torch.stack(
            (
                F.relu(float(self.min_jacobian) - jacobian).amax(),
                F.relu(jacobian - float(self.max_jacobian)).amax(),
                F.relu(float(self.min_edge_stretch) - stretches).amax(),
                F.relu(stretches - float(self.max_edge_stretch)).amax(),
                F.relu(displacement_norm - float(self.max_displacement)).amax(),
                F.relu(boundary_displacement - float(self.max_boundary_displacement)).amax(),
            )
        ).amax()
        return PolicyResult(
            name=self.name,
            loss=loss,
            weight=self.weight,
            strict_tolerance=self.strict_tolerance,
            metrics={
                "min_jacobian": jacobian.amin(),
                "max_jacobian": jacobian.amax(),
                "min_edge_stretch": stretches.amin(),
                "max_edge_stretch": stretches.amax(),
                "max_displacement": displacement_norm.amax(),
                "max_boundary_displacement": boundary_displacement.amax(),
                "folded_cell_fraction": (jacobian <= 0).to(jacobian.dtype).mean(),
            },
            violations={"physical_bound_violation": violation},
        )


@dataclass(frozen=True)
class SmoothManufacturabilityPolicy:
    """Favor smooth, printable, low-buckling lattice deformations."""

    laplacian_weight: float = 1.0
    strain_smoothness_weight: float = 0.7
    off_axis_curl_weight: float = 0.5
    weight: float = 1.0
    strict_tolerance: float = 2e-3
    name: str = "smooth_manufacturability"

    def __call__(self, engine: ContinuumSolverEngine, state) -> PolicyResult:
        del engine
        displacement = state.deformed_coordinates - state.reference_coordinates
        laplacian = _laplacian3(displacement)
        laplacian_energy = laplacian.square().sum(dim=-1).mean()
        log_sx, log_sy, log_sz = _cell_log_stretches(state.reference_coordinates, state.deformed_coordinates)
        strain_smoothness = _gradient_energy3_scalar(log_sx) + _gradient_energy3_scalar(log_sy) + _gradient_energy3_scalar(log_sz)
        curl = _curl3(displacement, state.reference_coordinates)
        off_axis_curl = curl[..., :2].square().sum(dim=-1).mean()
        loss = (
            float(self.laplacian_weight) * laplacian_energy
            + float(self.strain_smoothness_weight) * strain_smoothness
            + float(self.off_axis_curl_weight) * off_axis_curl
        )
        violation = torch.stack((laplacian.abs().amax(), strain_smoothness.sqrt(), off_axis_curl.sqrt())).amax()
        return PolicyResult(
            name=self.name,
            loss=loss,
            weight=self.weight,
            strict_tolerance=self.strict_tolerance,
            metrics={
                "laplacian_energy": laplacian_energy,
                "strain_smoothness": strain_smoothness,
                "off_axis_curl_rmse": off_axis_curl.sqrt(),
                "max_laplacian": laplacian.abs().amax(),
            },
            violations={"manufacturability_violation": violation},
        )


@dataclass(frozen=True)
class ValidationBounds:
    min_jacobian: float = 0.35
    max_jacobian: float = 2.60
    min_edge_stretch: float = 0.48
    max_edge_stretch: float = 2.05
    max_displacement: float = 0.90
    max_boundary_displacement: float = 0.16
    max_inverse_rmse: float = 2e-3
    max_inverse_abs: float = 8e-3


@dataclass(frozen=True)
class ValidationReport:
    strict_pass: bool
    metrics: dict[str, float | int | bool]
    failures: tuple[str, ...]

    def raise_if_failed(self) -> None:
        if self.strict_pass:
            return
        joined = "; ".join(self.failures)
        raise RuntimeError(f"physical validation failed: {joined}")


@dataclass(frozen=True)
class VisualizationArtifacts:
    vtk_grid: Path
    glyph_points: Path
    formation_gif: Path

    def as_report(self) -> dict[str, str]:
        return {
            "vtk_grid": str(self.vtk_grid),
            "glyph_points": str(self.glyph_points),
            "formation_gif": str(self.formation_gif),
        }


class GuardedOptimizerStep:
    """Rollback optimizer steps that violate physical validation bounds."""

    def __init__(self, bounds: ValidationBounds, *, max_retries: int = 4, backtracking: float = 0.35):
        self.bounds = bounds
        self.max_retries = int(max_retries)
        self.backtracking = float(backtracking)
        self.rejections = 0

    def __call__(self, context) -> torch.Tensor:
        parameters = tuple(context.engine.parameters())
        parameter_snapshot = _snapshot_parameters(parameters)
        optimizer_snapshot = copy.deepcopy(context.optimizer.state_dict())
        base_lrs = [float(group.get("lr", 1.0)) for group in context.optimizer.param_groups]
        last_report = None
        last_loss = None
        for retry in range(self.max_retries + 1):
            _restore_parameters(parameters, parameter_snapshot)
            context.optimizer.load_state_dict(optimizer_snapshot)
            _scale_learning_rates(context.optimizer, base_lrs, self.backtracking**retry)
            last_loss = _optimizer_step(context)
            report = validate_grid(
                context.coordinates,
                context.engine.field(context.coordinates),
                field=context.engine.field,
                bounds=self.bounds,
            )
            if report.strict_pass:
                return last_loss
            last_report = report
            self.rejections += 1
        _restore_parameters(parameters, parameter_snapshot)
        context.optimizer.load_state_dict(optimizer_snapshot)
        if last_report is not None:
            last_report.raise_if_failed()
        raise RuntimeError("optimizer step failed physical validation")


class FormationRecorder:
    """Capture sparse deformation snapshots for post-fit visualization."""

    def __init__(self, *, max_snapshots: int):
        self.max_snapshots = max(2, int(max_snapshots))
        self._snapshots: list[torch.Tensor] = []

    def capture(self, grid: torch.Tensor) -> None:
        self._snapshots.append(grid.detach().cpu().clone())

    def snapshots(self) -> tuple[torch.Tensor, ...]:
        if len(self._snapshots) <= self.max_snapshots:
            return tuple(self._snapshots)
        last = len(self._snapshots) - 1
        indices = sorted({round(i * last / float(self.max_snapshots - 1)) for i in range(self.max_snapshots)})
        return tuple(self._snapshots[index] for index in indices)


class RecordingOptimizerStep:
    """Wrap an optimizer stepper and record accepted grid states."""

    def __init__(self, wrapped, recorder: FormationRecorder, *, capture_every: int):
        self.wrapped = wrapped
        self.recorder = recorder
        self.capture_every = max(1, int(capture_every))
        self.steps = 0

    def __call__(self, context) -> torch.Tensor:
        loss = self.wrapped(context)
        self.steps += 1
        should_capture = self.steps % self.capture_every == 0 or context.step == context.steps - 1
        if should_capture:
            with torch.no_grad():
                self.recorder.capture(context.engine(context.coordinates))
        return loss


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/metamaterial_design"))
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--optimizer", choices=("adamw", "lbfgs", "hybrid"), default="hybrid")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lbfgs-steps", type=int, default=40)
    parser.add_argument("--lbfgs-max-iter", type=int, default=20)
    parser.add_argument("--max-total-steps", type=int, default=220)
    parser.add_argument("--target-loss", type=float, default=1e-2)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--control-size", type=int, default=10)
    parser.add_argument("--path-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-3)
    parser.add_argument("--lbfgs-lr", type=float, default=0.22)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad-norm", type=float, default=0.45)
    parser.add_argument("--guard-retries", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="cpu")
    parser.add_argument("--dtype", choices=DTYPE_CHOICES, default="float64")
    parser.add_argument("--max-threads", type=int, default=4)
    parser.add_argument("--formation-gif", type=Path, default=None)
    parser.add_argument("--viz-frames", type=int, default=48)
    parser.add_argument("--viz-final-hold", type=int, default=12)
    parser.add_argument("--viz-sample-stride", type=int, default=2)
    args = parser.parse_args()

    _guard_problem_size(args)
    runtime = _resolve_runtime(args.device, args.dtype)
    _configure_runtime(seed=67, max_threads=args.max_threads)
    coords, engine, bounds = build_problem(args, device=runtime[0], dtype=runtime[1])
    guard = GuardedOptimizerStep(bounds, max_retries=args.guard_retries)
    recorder = FormationRecorder(max_snapshots=args.viz_frames)
    recorder.capture(coords)
    optimizer_step = RecordingOptimizerStep(guard, recorder, capture_every=_visualization_capture_every(args))

    run = None
    if args.optimizer in {"adamw", "hybrid"} and args.steps > 0:
        optimizer = torch.optim.AdamW(engine.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
        run = engine.fit(
            coords,
            steps=args.steps,
            optimizer=optimizer,
            optimizer_step=optimizer_step,
            log_every=args.log_every or max(1, args.steps // 8),
            clip_grad_norm=args.clip_grad_norm,
        )
        _print_latest(run.history, "AdamW guarded stage")

    if args.optimizer in {"lbfgs", "hybrid"} and args.lbfgs_steps > 0:
        optimizer = torch.optim.LBFGS(
            engine.parameters(),
            lr=args.lbfgs_lr,
            max_iter=args.lbfgs_max_iter,
            tolerance_grad=1e-12,
            tolerance_change=1e-14,
            line_search_fn="strong_wolfe",
        )
        run = engine.fit(
            coords,
            steps=args.lbfgs_steps,
            optimizer=optimizer,
            optimizer_step=optimizer_step,
            log_every=args.log_every or max(1, args.lbfgs_steps // 6),
            clip_grad_norm=args.clip_grad_norm,
        )
        _print_latest(run.history, "LBFGS guarded polish")

    if run is None:
        run = engine.fit(coords, steps=1, lr=0.0, log_every=1)

    final_grid = engine(coords).detach()
    recorder.capture(final_grid)
    validation = validate_grid(coords, final_grid, field=engine.field, bounds=bounds)
    validation.raise_if_failed()

    output_dir = args.output_dir
    report_path = args.report_json or output_dir / "validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    formation_gif = args.formation_gif or output_dir / "formation.gif"
    visualization = write_vtk_visualizations(
        coords,
        final_grid,
        field=engine.field,
        snapshots=recorder.snapshots(),
        output_dir=output_dir,
        formation_gif=formation_gif,
        frame_count=args.viz_frames,
        final_hold=args.viz_final_hold,
        sample_stride=args.viz_sample_stride,
    )
    report = {
        "output_dir": str(output_dir),
        "final_loss": _to_float(run.evaluation.loss),
        "target_loss": float(args.target_loss),
        "target_reached": bool(_to_float(run.evaluation.loss) <= float(args.target_loss)),
        "guard_rejections": int(guard.rejections),
        "visualization": visualization.as_report(),
        "validation": validation.metrics,
        "validation_failures": list(validation.failures),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"output_dir: {output_dir}")
    print(f"validation: {report_path}")
    print(f"vtk_grid: {visualization.vtk_grid}")
    print(f"glyph_points: {visualization.glyph_points}")
    print(f"formation_gif: {visualization.formation_gif}")
    print(f"final_loss: {report['final_loss']:.12g}")
    print(f"target_reached: {report['target_reached']}")
    print(f"guard_rejections: {guard.rejections}")


def build_problem(args, *, device, dtype: torch.dtype) -> tuple[torch.Tensor, ContinuumSolverEngine, ValidationBounds]:
    coords = coordinate_grid_3d(args.grid_size, args.grid_size, args.grid_size, device=device, dtype=dtype)
    algebra = AlgebraContext(p=3, q=0, r=1, device=device, dtype=dtype)
    field = InvertibleBivectorField(
        algebra,
        coordinate_dim=3,
        projective=True,
        path_steps=args.path_steps,
        control_shape=(args.control_size, args.control_size, args.control_size),
        init_scale=4e-3,
    )
    curriculum = PhaseCurriculum(
        (
            (0.0, {"target": 1.00, "policy": 0.1, "policy:strict_physical_realism": 0.1, "policy:smooth_manufacturability": 0.05}),
            (0.5, {"target": 1.00, "policy": 1.0, "policy:strict_physical_realism": 1.5, "policy:smooth_manufacturability": 0.5}),
            (1.0, {"target": 1.00, "policy": 2.5, "policy:strict_physical_realism": 4.0, "policy:smooth_manufacturability": 2.0}),
        )
    )
    engine = ContinuumSolverEngine(
        field,
        target_criterion=MetamaterialTarget(),
        geometric_policies=(
            StrictPhysicalPolicy(),
            SmoothManufacturabilityPolicy(),
            InvertiblePathConsistencyPolicy(weight=1.0),
            BivectorNormPolicy(max_norm=0.90, weight=0.08),
        ),
        curriculum=curriculum,
    )
    return coords, engine, ValidationBounds()


def write_vtk_visualizations(
    reference: torch.Tensor,
    final_grid: torch.Tensor,
    *,
    field: InvertibleBivectorField,
    snapshots: Sequence[torch.Tensor],
    output_dir: Path,
    formation_gif: Path,
    frame_count: int,
    final_hold: int,
    sample_stride: int,
) -> VisualizationArtifacts:
    vtk, numpy_support, np, imageio = _import_visualization_dependencies()
    output_dir.mkdir(parents=True, exist_ok=True)
    formation_gif.parent.mkdir(parents=True, exist_ok=True)

    vtk_grid_path = output_dir / "metamaterial_grid.vtu"
    glyph_points_path = output_dir / "metamaterial_response_points.vtp"

    reference_cpu = reference.detach().cpu()
    final_cpu = final_grid.detach().cpu()
    reference_np = _tensor_grid_to_numpy(reference_cpu, np)
    final_np = _tensor_grid_to_numpy(final_cpu, np)
    displacement_np = final_np - reference_np
    weights = field.weights_for_shape(tuple(reference.shape[:-1]), device=reference.device, dtype=reference.dtype).detach().cpu()
    response = _bivector_response_vectors(
        weights,
        field.bivector_layout.basis_indices,
        coordinate_dim=field.coordinate_dim,
        homogeneous_bit=field.algebra.p + field.algebra.q if field.projective else None,
    )
    response_np = _tensor_grid_to_numpy(response, np)

    jacobian = _volume_jacobian(reference_cpu, final_cpu).numpy()
    curl = _curl3(final_cpu - reference_cpu, reference_cpu)
    curl_magnitude = torch.linalg.vector_norm(curl, dim=-1).numpy()
    cell_response = _cell_average_values(response)
    cell_response_np = _tensor_grid_to_numpy(cell_response, np)
    cell_response_magnitude = np.linalg.norm(cell_response_np, axis=-1)
    cell_displacement = _cell_average_values(final_cpu - reference_cpu)
    cell_displacement_np = _tensor_grid_to_numpy(cell_displacement, np)
    cell_displacement_magnitude = np.linalg.norm(cell_displacement_np, axis=-1)
    cell_centers_np = _tensor_grid_to_numpy(_cell_centers(final_cpu), np)
    grid = _vtk_hexahedral_grid(
        vtk,
        numpy_support,
        np,
        final_np,
        point_data={
            "displacement": displacement_np,
            "displacement_magnitude": np.linalg.norm(displacement_np, axis=-1),
            "bivector_response": response_np,
            "bivector_response_magnitude": np.linalg.norm(response_np, axis=-1),
        },
        cell_data={
            "cell_displacement": cell_displacement_np,
            "cell_displacement_magnitude": cell_displacement_magnitude,
            "cell_bivector_response": cell_response_np,
            "cell_bivector_response_magnitude": cell_response_magnitude,
            "jacobian": jacobian,
            "curl_magnitude": curl_magnitude,
        },
    )
    _write_vtu(vtk, grid, vtk_grid_path)
    glyph_points = _vtk_glyph_points(
        vtk,
        numpy_support,
        np,
        cell_centers_np,
        point_data={
            "bivector_response": cell_response_np,
            "bivector_response_magnitude": cell_response_magnitude,
            "displacement": cell_displacement_np,
            "displacement_magnitude": cell_displacement_magnitude,
            "jacobian": jacobian,
            "curl_magnitude": curl_magnitude,
        },
    )
    _write_vtp(vtk, glyph_points, glyph_points_path)

    _render_formation_gif(
        vtk,
        numpy_support,
        np,
        imageio,
        snapshots=snapshots or (reference_cpu, final_cpu),
        gif_path=formation_gif,
        frame_count=max(2, int(frame_count)),
        final_hold=max(0, int(final_hold)),
        sample_stride=max(1, int(sample_stride)),
    )
    return VisualizationArtifacts(
        vtk_grid=vtk_grid_path,
        glyph_points=glyph_points_path,
        formation_gif=formation_gif,
    )


def validate_grid(
    reference: torch.Tensor,
    deformed: torch.Tensor,
    *,
    field: InvertibleBivectorField,
    bounds: ValidationBounds,
) -> ValidationReport:
    failures: list[str] = []
    metrics: dict[str, float | int | bool] = {}
    finite = bool(torch.isfinite(deformed).all().detach().cpu())
    metrics["finite_grid"] = finite
    if not finite:
        failures.append("grid contains non-finite coordinates")

    jacobian = _volume_jacobian(reference, deformed)
    stretches = _edge_stretches(reference, deformed)
    displacement = deformed - reference
    displacement_norm = torch.linalg.vector_norm(displacement, dim=-1)
    boundary = _boundary_point_mask(reference.shape[:-1], device=reference.device, dtype=reference.dtype)
    boundary_displacement = displacement_norm * boundary
    reconstructed = field.inverse(deformed)
    inverse_residual = reconstructed - reference

    metrics.update(
        {
            "min_jacobian": _to_float(jacobian.amin()),
            "max_jacobian": _to_float(jacobian.amax()),
            "folded_cell_fraction": _to_float((jacobian <= 0).to(jacobian.dtype).mean()),
            "min_edge_stretch": _to_float(stretches.amin()),
            "max_edge_stretch": _to_float(stretches.amax()),
            "max_displacement": _to_float(displacement_norm.amax()),
            "max_boundary_displacement": _to_float(boundary_displacement.amax()),
            "inverse_rmse": _to_float(inverse_residual.square().mean().sqrt()),
            "inverse_max_abs": _to_float(inverse_residual.abs().amax()),
        }
    )
    _require(metrics["min_jacobian"] >= bounds.min_jacobian, "min_jacobian below strict bound", failures)
    _require(metrics["max_jacobian"] <= bounds.max_jacobian, "max_jacobian above strict bound", failures)
    _require(metrics["folded_cell_fraction"] == 0.0, "one or more cells are folded", failures)
    _require(metrics["min_edge_stretch"] >= bounds.min_edge_stretch, "edge stretch below strict bound", failures)
    _require(metrics["max_edge_stretch"] <= bounds.max_edge_stretch, "edge stretch above strict bound", failures)
    _require(metrics["max_displacement"] <= bounds.max_displacement, "displacement above strict bound", failures)
    _require(
        metrics["max_boundary_displacement"] <= bounds.max_boundary_displacement,
        "boundary displacement above strict bound",
        failures,
    )
    _require(metrics["inverse_rmse"] <= bounds.max_inverse_rmse, "inverse rmse above strict bound", failures)
    _require(metrics["inverse_max_abs"] <= bounds.max_inverse_abs, "inverse max abs above strict bound", failures)

    return ValidationReport(strict_pass=not failures, metrics=metrics, failures=tuple(failures))


def _import_visualization_dependencies():
    try:
        import imageio.v2 as imageio
        import numpy as np
        import vtk
        from vtk.util import numpy_support # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "visualization requires optional dependencies; install them with `uv sync --group viz`"
        ) from exc
    return vtk, numpy_support, np, imageio


def _tensor_grid_to_numpy(tensor: torch.Tensor, np_module):
    return tensor.detach().cpu().to(torch.float64).contiguous().numpy().astype(np_module.float64, copy=False)


def _vtk_hexahedral_grid(vtk, numpy_support, np_module, grid_np, *, point_data, cell_data):
    depth, height, width = grid_np.shape[:3]
    point_shape = (depth, height, width)
    cell_shape = (depth - 1, height - 1, width - 1)
    unstructured = vtk.vtkUnstructuredGrid()
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(grid_np.reshape(-1, 3), deep=True))
    unstructured.SetPoints(points)

    def idx(z: int, y: int, x: int) -> int:
        return (z * height + y) * width + x

    for z in range(depth - 1):
        for y in range(height - 1):
            for x in range(width - 1):
                hexahedron = vtk.vtkHexahedron()
                point_ids = (
                    idx(z, y, x),
                    idx(z, y, x + 1),
                    idx(z, y + 1, x + 1),
                    idx(z, y + 1, x),
                    idx(z + 1, y, x),
                    idx(z + 1, y, x + 1),
                    idx(z + 1, y + 1, x + 1),
                    idx(z + 1, y + 1, x),
                )
                for local_id, point_id in enumerate(point_ids):
                    hexahedron.GetPointIds().SetId(local_id, int(point_id))
                unstructured.InsertNextCell(hexahedron.GetCellType(), hexahedron.GetPointIds())

    for name, values in point_data.items():
        value_array = np_module.asarray(values, dtype=np_module.float64)
        if value_array.shape[:-1] == point_shape and value_array.shape[-1] in {2, 3}:
            array = numpy_support.numpy_to_vtk(value_array.reshape(-1, value_array.shape[-1]), deep=True)
        else:
            array = numpy_support.numpy_to_vtk(value_array.reshape(-1), deep=True)
        array.SetName(str(name))
        unstructured.GetPointData().AddArray(array)
        if name == "bivector_response":
            unstructured.GetPointData().SetVectors(array)
        elif name == "bivector_response_magnitude":
            unstructured.GetPointData().SetScalars(array)
    for name, values in cell_data.items():
        value_array = np_module.asarray(values, dtype=np_module.float64)
        if value_array.shape[:-1] == cell_shape and value_array.shape[-1] in {2, 3}:
            array = numpy_support.numpy_to_vtk(value_array.reshape(-1, value_array.shape[-1]), deep=True)
        else:
            array = numpy_support.numpy_to_vtk(value_array.reshape(-1), deep=True)
        array.SetName(str(name))
        unstructured.GetCellData().AddArray(array)
        if name == "cell_bivector_response":
            unstructured.GetCellData().SetVectors(array)
        elif name == "cell_bivector_response_magnitude":
            unstructured.GetCellData().SetScalars(array)
    return unstructured


def _vtk_glyph_points(vtk, numpy_support, np_module, points_np, *, point_data):
    point_shape = points_np.shape[:-1]
    flat_points = points_np.reshape(-1, 3)
    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(flat_points, deep=True))
    polydata.SetPoints(points)

    vertices = vtk.vtkCellArray()
    for point_id in range(flat_points.shape[0]):
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, int(point_id))
        vertices.InsertNextCell(vertex)
    polydata.SetVerts(vertices)

    for name, values in point_data.items():
        value_array = np_module.asarray(values, dtype=np_module.float64)
        if value_array.shape[:-1] == point_shape and value_array.shape[-1] in {2, 3}:
            array = numpy_support.numpy_to_vtk(value_array.reshape(-1, value_array.shape[-1]), deep=True)
        else:
            array = numpy_support.numpy_to_vtk(value_array.reshape(-1), deep=True)
        array.SetName(str(name))
        polydata.GetPointData().AddArray(array)
        if name == "bivector_response":
            polydata.GetPointData().SetVectors(array)
        elif name == "bivector_response_magnitude":
            polydata.GetPointData().SetScalars(array)
    return polydata


def _write_vtu(vtk, grid, path: Path) -> None:
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(grid)
    writer.SetCompressorTypeToZLib()
    if writer.Write() != 1:
        raise RuntimeError(f"failed to write VTK grid to {path}")


def _write_vtp(vtk, polydata, path: Path) -> None:
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    writer.SetCompressorTypeToZLib()
    if writer.Write() != 1:
        raise RuntimeError(f"failed to write VTK glyph points to {path}")


def _render_formation_gif(
    vtk,
    numpy_support,
    np_module,
    imageio,
    *,
    snapshots: Sequence[torch.Tensor],
    gif_path: Path,
    frame_count: int,
    final_hold: int,
    sample_stride: int,
) -> None:
    frames = _interpolate_snapshot_frames(snapshots, frame_count=frame_count)
    if not frames:
        raise RuntimeError("no formation snapshots were captured")

    final_frame = frames[-1]
    frames = [*frames, *([final_frame] * final_hold)]
    renderer, window = _new_renderer(vtk, width=960, height=720)
    mapper = vtk.vtkPolyDataMapper()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.05, 0.08, 0.11)
    actor.GetProperty().SetLineWidth(1.6)
    renderer.AddActor(actor)

    surface_mapper = vtk.vtkDataSetMapper()
    surface_actor = vtk.vtkActor()
    surface_actor.SetMapper(surface_mapper)
    surface_actor.GetProperty().SetColor(0.77, 0.86, 0.91)
    surface_actor.GetProperty().SetOpacity(0.14)
    renderer.AddActor(surface_actor)

    images = []
    total = max(1, len(frames) - 1)
    for index, frame in enumerate(frames):
        frame_np = _tensor_grid_to_numpy(frame, np_module)
        mapper.SetInputData(_vtk_lattice_polydata(vtk, numpy_support, frame_np, sample_stride=sample_stride))
        surface = _vtk_hexahedral_grid(vtk, numpy_support, np_module, frame_np, point_data={}, cell_data={})
        surface_mapper.SetInputData(surface)
        _set_camera(vtk, renderer, _grid_bounds(np_module, frame_np), angle_degrees=-18.0 + 30.0 * index / total)
        images.append(_capture_window(vtk, numpy_support, np_module, window))

    imageio.mimsave(gif_path, images, duration=0.085, loop=0)


def _new_renderer(vtk, *, width: int, height: int):
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetMultiSamples(0)
    window.AddRenderer(renderer)
    window.SetSize(int(width), int(height))
    return renderer, window


def _vtk_lattice_polydata(vtk, numpy_support, grid_np, *, sample_stride: int):
    depth, height, width = grid_np.shape[:3]
    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(grid_np.reshape(-1, 3), deep=True))
    polydata.SetPoints(points)
    lines = vtk.vtkCellArray()

    def idx(z: int, y: int, x: int) -> int:
        return (z * height + y) * width + x

    def add_line(point_ids: Sequence[int]) -> None:
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(point_ids))
        for local_id, point_id in enumerate(point_ids):
            line.GetPointIds().SetId(local_id, int(point_id))
        lines.InsertNextCell(line)

    z_indices = _sample_indices(depth, sample_stride)
    y_indices = _sample_indices(height, sample_stride)
    x_indices = _sample_indices(width, sample_stride)
    for z in z_indices:
        for y in y_indices:
            add_line([idx(z, y, x) for x in range(width)])
    for z in z_indices:
        for x in x_indices:
            add_line([idx(z, y, x) for y in range(height)])
    for y in y_indices:
        for x in x_indices:
            add_line([idx(z, y, x) for z in range(depth)])
    polydata.SetLines(lines)
    return polydata


def _interpolate_snapshot_frames(snapshots: Sequence[torch.Tensor], *, frame_count: int) -> list[torch.Tensor]:
    snapshots = tuple(snapshot.detach().cpu() for snapshot in snapshots)
    if not snapshots:
        return []
    if len(snapshots) == 1:
        return [snapshots[0].clone() for _ in range(max(1, frame_count))]
    frame_count = max(2, int(frame_count))
    frames = []
    last = len(snapshots) - 1
    for index in range(frame_count):
        position = index * last / float(frame_count - 1)
        low = int(position)
        high = min(low + 1, last)
        weight = position - low
        frames.append(snapshots[low].lerp(snapshots[high], float(weight)))
    return frames


def _capture_window(vtk, numpy_support, np_module, window):
    window.Render()
    window_to_image = vtk.vtkWindowToImageFilter()
    window_to_image.SetInput(window)
    window_to_image.SetInputBufferTypeToRGB()
    window_to_image.ReadFrontBufferOff()
    window_to_image.Update()
    image = window_to_image.GetOutput()
    width, height, _ = image.GetDimensions()
    scalars = image.GetPointData().GetScalars()
    components = scalars.GetNumberOfComponents()
    pixels = numpy_support.vtk_to_numpy(scalars).reshape(height, width, components)
    return np_module.flipud(pixels[..., :3]).copy()


def _set_camera(vtk, renderer, bounds: tuple[float, float, float, float, float, float], *, angle_degrees: float) -> None:
    import math

    center = (
        0.5 * (bounds[0] + bounds[1]),
        0.5 * (bounds[2] + bounds[3]),
        0.5 * (bounds[4] + bounds[5]),
    )
    span = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4], 1e-6)
    radius = 3.2 * span
    azimuth = math.radians(42.0 + float(angle_degrees))
    elevation = math.radians(25.0)
    camera = vtk.vtkCamera()
    camera.SetFocalPoint(*center)
    camera.SetPosition(
        center[0] + radius * math.cos(elevation) * math.cos(azimuth),
        center[1] + radius * math.cos(elevation) * math.sin(azimuth),
        center[2] + radius * math.sin(elevation),
    )
    camera.SetViewUp(0.0, 0.0, 1.0)
    renderer.SetActiveCamera(camera)
    renderer.ResetCameraClippingRange()


def _grid_bounds(np_module, grid_np) -> tuple[float, float, float, float, float, float]:
    mins = np_module.min(grid_np.reshape(-1, 3), axis=0)
    maxes = np_module.max(grid_np.reshape(-1, 3), axis=0)
    return (
        float(mins[0]),
        float(maxes[0]),
        float(mins[1]),
        float(maxes[1]),
        float(mins[2]),
        float(maxes[2]),
    )


def _sample_indices(size: int, stride: int) -> list[int]:
    stride = max(1, int(stride))
    indices = list(range(0, int(size), stride))
    if not indices or indices[-1] != int(size) - 1:
        indices.append(int(size) - 1)
    return indices


def _bivector_response_vectors(
    weights: torch.Tensor,
    basis_indices: Sequence[int],
    *,
    coordinate_dim: int,
    homogeneous_bit: int | None,
) -> torch.Tensor:
    mean_weights = weights.mean(dim=0)
    response = mean_weights.new_zeros(*mean_weights.shape[:-1], min(int(coordinate_dim), 3))
    for lane, blade_index in enumerate(basis_indices):
        bits = tuple(bit for bit in range(8) if int(blade_index) & (1 << bit))
        coefficient = mean_weights[..., lane]
        if bits == (1, 2) and response.shape[-1] >= 1:
            response[..., 0] += coefficient
        elif bits == (0, 2) and response.shape[-1] >= 2:
            response[..., 1] -= coefficient
        elif bits == (0, 1) and response.shape[-1] >= 3:
            response[..., 2] += coefficient
        elif homogeneous_bit is not None and homogeneous_bit in bits:
            coordinate_bits = [bit for bit in bits if bit != homogeneous_bit and bit < response.shape[-1]]
            if coordinate_bits:
                response[..., coordinate_bits[0]] += 0.5 * coefficient
    if response.shape[-1] == 3:
        return response
    padded = mean_weights.new_zeros(*mean_weights.shape[:-1], 3)
    padded[..., : response.shape[-1]] = response
    return padded


def _cell_average_values(values: torch.Tensor) -> torch.Tensor:
    z_axis = values.ndim - 4
    y_axis = values.ndim - 3
    x_axis = values.ndim - 2
    depth = values.shape[z_axis]
    height = values.shape[y_axis]
    width = values.shape[x_axis]
    return 0.125 * (
        values.narrow(z_axis, 0, depth - 1).narrow(y_axis, 0, height - 1).narrow(x_axis, 0, width - 1)
        + values.narrow(z_axis, 1, depth - 1).narrow(y_axis, 0, height - 1).narrow(x_axis, 0, width - 1)
        + values.narrow(z_axis, 0, depth - 1).narrow(y_axis, 1, height - 1).narrow(x_axis, 0, width - 1)
        + values.narrow(z_axis, 0, depth - 1).narrow(y_axis, 0, height - 1).narrow(x_axis, 1, width - 1)
        + values.narrow(z_axis, 1, depth - 1).narrow(y_axis, 1, height - 1).narrow(x_axis, 0, width - 1)
        + values.narrow(z_axis, 1, depth - 1).narrow(y_axis, 0, height - 1).narrow(x_axis, 1, width - 1)
        + values.narrow(z_axis, 0, depth - 1).narrow(y_axis, 1, height - 1).narrow(x_axis, 1, width - 1)
        + values.narrow(z_axis, 1, depth - 1).narrow(y_axis, 1, height - 1).narrow(x_axis, 1, width - 1)
    )


def _optimizer_step(context) -> torch.Tensor:
    if isinstance(context.optimizer, torch.optim.LBFGS):
        return context.optimizer.step(context.closure)
    return context.step_optimizer()


def _snapshot_parameters(parameters: Iterable[torch.nn.Parameter]) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().clone() for parameter in parameters)


def _restore_parameters(parameters: Iterable[torch.nn.Parameter], snapshot: tuple[torch.Tensor, ...]) -> None:
    with torch.no_grad():
        for parameter, value in zip(parameters, snapshot):
            parameter.copy_(value)


def _scale_learning_rates(optimizer: torch.optim.Optimizer, base_lrs: list[float], factor: float) -> None:
    for group, lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = float(lr) * float(factor)


def coordinate_grid_3d(depth: int, height: int, width: int, *, device, dtype: torch.dtype, extent: float = 1.0) -> torch.Tensor:
    z = torch.linspace(-float(extent), float(extent), int(depth), device=device, dtype=dtype)
    y = torch.linspace(-float(extent), float(extent), int(height), device=device, dtype=dtype)
    x = torch.linspace(-float(extent), float(extent), int(width), device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack((xx, yy, zz), dim=-1)


def _cell_edges_3d(coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return tuple(torch.log(stretch.clamp_min(1e-8)) for stretch in _edge_stretches(reference, deformed).unbind(dim=0))


def _edge_stretches(reference: torch.Tensor, deformed: torch.Tensor) -> torch.Tensor:
    ref_edges = _cell_edges_3d(reference)
    def_edges = _cell_edges_3d(deformed)
    stretches = []
    for ref_edge, def_edge in zip(ref_edges, def_edges):
        ref_length = torch.linalg.vector_norm(ref_edge, dim=-1).clamp_min(1e-8)
        def_length = torch.linalg.vector_norm(def_edge, dim=-1).clamp_min(1e-8)
        stretches.append(def_length / ref_length)
    return torch.stack(stretches, dim=0)


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


def _volume_jacobian(reference: torch.Tensor, deformed: torch.Tensor) -> torch.Tensor:
    dx_ref, dy_ref, dz_ref = _cell_edges_3d(reference)
    dx_def, dy_def, dz_def = _cell_edges_3d(deformed)
    det_ref = torch.linalg.det(torch.stack((dx_ref, dy_ref, dz_ref), dim=-1))
    det_def = torch.linalg.det(torch.stack((dx_def, dy_def, dz_def), dim=-1))
    return det_def / _signed_clamp(det_ref, 1e-8)


def _curl3(displacement: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    dx_u, dy_u, dz_u = _cell_edges_3d(displacement)
    dx_ref, dy_ref, dz_ref = _cell_edges_3d(reference)
    step_x = torch.linalg.vector_norm(dx_ref, dim=-1).clamp_min(1e-8)
    step_y = torch.linalg.vector_norm(dy_ref, dim=-1).clamp_min(1e-8)
    step_z = torch.linalg.vector_norm(dz_ref, dim=-1).clamp_min(1e-8)
    curl_x = dy_u[..., 2] / step_y - dz_u[..., 1] / step_z
    curl_y = dz_u[..., 0] / step_z - dx_u[..., 2] / step_x
    curl_z = dx_u[..., 1] / step_x - dy_u[..., 0] / step_y
    return torch.stack((curl_x, curl_y, curl_z), dim=-1)


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


def _gradient_energy3_scalar(values: torch.Tensor) -> torch.Tensor:
    terms = [values.diff(dim=dim).square().mean() for dim in range(values.ndim) if values.shape[dim] > 1]
    return torch.stack(terms).mean() if terms else values.new_zeros(())


def _boundary_point_mask(shape: tuple[int, ...], *, device, dtype: torch.dtype) -> torch.Tensor:
    if len(shape) != 3:
        raise ValueError(f"expected 3D grid shape, got {shape}")
    mask = torch.zeros(shape, device=device, dtype=dtype)
    mask[0, :, :] = 1.0
    mask[-1, :, :] = 1.0
    mask[:, 0, :] = 1.0
    mask[:, -1, :] = 1.0
    mask[:, :, 0] = 1.0
    mask[:, :, -1] = 1.0
    return mask


def _interior_cell_mask(shape: torch.Size | tuple[int, ...], *, device, dtype: torch.dtype) -> torch.Tensor:
    shape = tuple(int(v) for v in shape)
    mask = torch.ones(shape, device=device, dtype=dtype)
    if len(shape) != 3 or min(shape) <= 2:
        return torch.zeros(shape, device=device, dtype=dtype)
    mask[0, :, :] = 0.0
    mask[-1, :, :] = 0.0
    mask[:, 0, :] = 0.0
    mask[:, -1, :] = 0.0
    mask[:, :, 0] = 0.0
    mask[:, :, -1] = 0.0
    return mask


def _checkerboard(shape: torch.Size | tuple[int, ...], *, device, dtype: torch.dtype) -> torch.Tensor:
    axes = torch.meshgrid(*(torch.arange(int(size), device=device) for size in shape), indexing="ij")
    parity = sum(axes).remainder(2)
    return torch.where(parity == 0, torch.ones((), device=device, dtype=dtype), -torch.ones((), device=device, dtype=dtype))


def _phase_profile(radius: torch.Tensor, *, core_value: float, rim_value: float, core_radius: float, rim_radius: float) -> torch.Tensor:
    t = ((radius - float(core_radius)) / max(float(rim_radius) - float(core_radius), 1e-8)).clamp(0.0, 1.0)
    smooth = _smoothstep(t)
    return float(core_value) + (float(rim_value) - float(core_value)) * smooth


def _normalized_radius(coordinates: torch.Tensor) -> torch.Tensor:
    radius = torch.linalg.vector_norm(coordinates, dim=-1)
    return radius / radius.detach().amax().clamp_min(1e-8)


def _poisson_ratio(log_sx: torch.Tensor, log_sy: torch.Tensor, log_sz: torch.Tensor) -> torch.Tensor:
    return -0.5 * (log_sy + log_sz) / _signed_clamp(log_sx, 7e-4)


def _smoothstep(values: torch.Tensor) -> torch.Tensor:
    t = values.clamp(0.0, 1.0)
    return t.square() * (3.0 - 2.0 * t)


def _soft_shortfall(values: torch.Tensor, lower: float, width: float) -> torch.Tensor:
    width = max(float(width), 1e-8)
    return width * F.softplus((float(lower) - values) / width)


def _soft_excess(values: torch.Tensor, upper: float, width: float) -> torch.Tensor:
    width = max(float(width), 1e-8)
    return width * F.softplus((values - float(upper)) / width)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _signed_clamp(values: torch.Tensor, eps: float) -> torch.Tensor:
    magnitude = values.abs().clamp_min(float(eps))
    return torch.where(values < 0, -magnitude, magnitude)


def _guard_problem_size(args) -> None:
    total_steps = _total_optimizer_steps(args)
    if total_steps > int(args.max_total_steps):
        raise ValueError(f"requested {total_steps} optimizer steps, exceeding --max-total-steps={args.max_total_steps}")
    if args.grid_size < 4:
        raise ValueError("--grid-size must be at least 4 for 3D curl and validation")
    if args.control_size < 2:
        raise ValueError("--control-size must be at least 2")
    if args.viz_frames < 2:
        raise ValueError("--viz-frames must be at least 2")
    if args.viz_sample_stride < 1:
        raise ValueError("--viz-sample-stride must be at least 1")


def _total_optimizer_steps(args) -> int:
    return (args.steps if args.optimizer in {"adamw", "hybrid"} else 0) + (
        args.lbfgs_steps if args.optimizer in {"lbfgs", "hybrid"} else 0
    )


def _visualization_capture_every(args) -> int:
    return max(1, _total_optimizer_steps(args) // max(1, int(args.viz_frames) - 2))


def _resolve_runtime(device_name: str, dtype_name: str) -> tuple[torch.device, torch.dtype]:
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda"), dtype
        if dtype == torch.float32 and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps"), dtype
        return torch.device("cpu"), dtype
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("MPS was requested but is not available")
        if dtype != torch.float32:
            raise ValueError("MPS requires --dtype float32 for this example")
    return device, dtype


def _configure_runtime(*, seed: int, max_threads: int) -> None:
    torch.manual_seed(int(seed))
    if max_threads > 0:
        torch.set_num_threads(max(1, min(int(max_threads), torch.get_num_threads())))


def _print_latest(history, title: str) -> None:
    latest = history.latest()
    if latest is None:
        print(f"\n== {title} ==\nno metrics")
        return
    print(f"\n== {title} ==\nstep: {latest.step}")
    for index, (key, value) in enumerate(latest.metrics.items()):
        if index >= 28:
            print("...")
            break
        print(f"{key}: {_to_float(value) if isinstance(value, torch.Tensor) else value}")


def _require(condition: bool, message: str, failures: list[str]) -> None:
    if not bool(condition):
        failures.append(message)


def _to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(()))
    return float(value)


if __name__ == "__main__":
    main()
