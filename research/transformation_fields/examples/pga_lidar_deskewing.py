# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Recover continuous ego-motion from sparse geometric constraints in 3-D PGA."""

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

from clifra import AlgebraContext
from research.transformation_fields import (
    CoordinateFieldInput,
    GeneratorSubspace,
    InvertibleBivectorField,
    RBFGeneratorSampler,
)


@dataclass(frozen=True)
class Config:
    seed: int = 29
    plane_samples: tuple[int, ...] = (310, 250, 170, 170)
    line_samples: tuple[int, ...] = (55, 55, 45, 50, 55)
    heldout_plane_samples: tuple[int, ...] = (180, 150, 110, 110)
    heldout_line_samples: tuple[int, ...] = (36, 36, 30, 32, 36)
    plane_constraints_per_patch: int = 26
    line_constraints_per_feature: int = 12
    anchor_constraints: int = 24
    rbf_controls: int = 10
    rbf_length_scale: float = 0.16
    optimization_steps: int = 700
    learning_rate: float = 0.035
    lbfgs_steps: int = 45
    measurement_noise: float = 0.003
    dense_times: int = 241
    log_every: int = 100
    live_every: int = 20
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "plane": 1.0,
            "line": 1.4,
            "anchor": 12.0,
            "coefficient_second_difference": 0.035,
            "coefficient_first_difference": 0.0015,
            "start_field": 0.08,
        }
    )
    output_dir: Path = Path("outputs/pga_lidar_deskewing")
    live: bool = False


@dataclass(frozen=True)
class PlanePatch:
    name: str
    origin: torch.Tensor
    axis_u: torch.Tensor
    axis_v: torch.Tensor
    extent_u: float
    extent_v: float
    plane: torch.Tensor
    color: str


@dataclass(frozen=True)
class LineFeature:
    name: str
    start: torch.Tensor
    end: torch.Tensor
    line: torch.Tensor
    color: str


@dataclass(frozen=True)
class Scene:
    planes: tuple[PlanePatch, ...]
    lines: tuple[LineFeature, ...]
    world_points: torch.Tensor
    plane_ids: torch.Tensor
    line_ids: torch.Tensor
    colors: np.ndarray


@dataclass(frozen=True)
class Scan:
    measured_points: torch.Tensor
    world_points: torch.Tensor
    acquisition_times: torch.Tensor
    plane_ids: torch.Tensor
    line_ids: torch.Tensor
    colors: np.ndarray

    def field_input(self, indices: torch.Tensor | None = None) -> CoordinateFieldInput:
        if indices is None:
            return CoordinateFieldInput(self.measured_points, sample_coordinates=self.acquisition_times)
        return CoordinateFieldInput(
            self.measured_points[indices],
            sample_coordinates=self.acquisition_times[indices],
        )


@dataclass(frozen=True)
class SparseConstraints:
    indices: torch.Tensor
    plane_rows: torch.Tensor
    line_rows: torch.Tensor
    anchor_rows: torch.Tensor
    plane_ids: torch.Tensor
    line_ids: torch.Tensor
    anchor_targets: torch.Tensor


@dataclass
class FitState:
    loss: torch.Tensor
    components: dict[str, torch.Tensor]


@dataclass(frozen=True)
class OptimizationResult:
    history: list[float]
    best_step: int
    best_loss: float
    final_history_loss: float


class AnalyticTrajectory:
    """Independent analytic SE(3) trajectory used for scan synthesis and evaluation."""

    def pose(self, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = times.squeeze(-1)
        rotation_vector = torch.stack(
            (
                0.10 * torch.sin(2.0 * math.pi * t) + 0.07 * t + 0.018 * torch.sin(5.0 * math.pi * t),
                -0.14 * torch.sin(math.pi * t) - 0.05 * t + 0.025 * torch.sin(3.0 * math.pi * t),
                0.58 * t + 0.07 * torch.sin(2.0 * math.pi * t) + 0.035 * t.square(),
            ),
            dim=-1,
        )
        position = torch.stack(
            (
                0.74 * t + 0.09 * torch.sin(math.pi * t) + 0.025 * torch.sin(3.0 * math.pi * t),
                0.15 * torch.sin(2.0 * math.pi * t) + 0.11 * t.square(),
                0.18 * t + 0.065 * torch.sin(math.pi * t) - 0.025 * torch.sin(4.0 * math.pi * t),
            ),
            dim=-1,
        )
        rotation = torch.matrix_exp(_skew(rotation_vector))
        return position, rotation

    def sensor_to_world(self, points: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        position, rotation = self.pose(times)
        return torch.einsum("...ij,...j->...i", rotation, points) + position

    def world_to_sensor(self, points: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        position, rotation = self.pose(times)
        return torch.einsum("...ji,...j->...i", rotation, points - position)


def _skew(vectors: torch.Tensor) -> torch.Tensor:
    x, y, z = vectors.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(*vectors.shape[:-1], 3, 3)


class PGAPointChart:
    """A correct dual-PGA Euclidean point chart for ``Cl(3,0,1)``.

    With basis order ``(e1,e2,e3,e0)``, the intersection of the coordinate
    planes is

    ``P(x,y,z) = e123 - z e120 + y e130 - x e230``.
    """

    coordinate_dim = 3
    homogeneous_position = 0

    def __init__(self, algebra: AlgebraContext):
        if (algebra.p, algebra.q, algebra.r) != (3, 0, 1):
            raise ValueError("PGAPointChart requires Cl(3,0,1)")
        self.algebra = algebra
        self.layout = algebra.layout((3,))
        if self.layout.basis_indices != (0b0111, 0b1011, 0b1101, 0b1110):
            raise RuntimeError("unexpected Cl(3,0,1) grade-3 basis order")

    def embed(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.ndim < 1 or coordinates.shape[-1] != 3:
            raise ValueError(f"coordinates must have shape [..., 3], got {tuple(coordinates.shape)}")
        points = coordinates.new_empty(*coordinates.shape[:-1], 4)
        points[..., 0] = 1.0
        points[..., 1] = -coordinates[..., 2]
        points[..., 2] = coordinates[..., 1]
        points[..., 3] = -coordinates[..., 0]
        return points

    def extract(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim < 1 or points.shape[-1] != 4:
            raise ValueError(f"PGA points must have shape [..., 4], got {tuple(points.shape)}")
        weight = points[..., 0:1]
        eps = torch.finfo(points.dtype).eps
        safe_weight = torch.where(weight.abs() < eps, weight.sign().masked_fill(weight == 0, 1.0) * eps, weight)
        normalized = points / safe_weight
        return torch.stack((-normalized[..., 3], normalized[..., 2], -normalized[..., 1]), dim=-1)

    def metric_signs(self, *, device=None, dtype=None) -> torch.Tensor:
        return torch.ones(3, device=device, dtype=self.algebra.dtype if dtype is None else dtype)


class PGAGeometry:
    """Local PGA construction and incidence operations backed by fixed plans."""

    def __init__(self, algebra: AlgebraContext, chart: PGAPointChart):
        self.algebra = algebra
        self.chart = chart
        self.plane_layout = algebra.layout((1,))
        self.line_layout = algebra.layout((2,))
        self.point_layout = algebra.layout((3,))
        self.volume_layout = algebra.layout((4,))
        self.meet_planes = algebra.plan_product(
            op="wedge", left=self.plane_layout, right=self.plane_layout, output=self.line_layout
        )
        self.incidence = algebra.plan_product(
            op="wedge", left=self.plane_layout, right=self.point_layout, output=self.volume_layout
        )
        # The only product needed by P v L = D^-1(D(P) ^ D(L)).
        self.dual_meet = algebra.plan_product(
            op="wedge", left=self.plane_layout, right=self.line_layout, output=self.point_layout
        )

    def plane(self, normal: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        normal = normal / torch.linalg.vector_norm(normal).clamp_min(torch.finfo(normal.dtype).eps)
        result = normal.new_empty(4)
        result[:3] = normal
        result[3] = -torch.dot(normal, point)
        return result

    def line_from_points(self, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        direction = end - start
        direction = direction / torch.linalg.vector_norm(direction).clamp_min(torch.finfo(direction.dtype).eps)
        seed = direction.new_tensor([0.0, 0.0, 1.0])
        if torch.abs(torch.dot(direction, seed)) > 0.88:
            seed = direction.new_tensor([0.0, 1.0, 0.0])
        normal_a = torch.linalg.cross(direction, seed, dim=-1)
        normal_a = normal_a / torch.linalg.vector_norm(normal_a)
        normal_b = torch.linalg.cross(direction, normal_a, dim=-1)
        plane_a = self.plane(normal_a, start)
        plane_b = self.plane(normal_b, start)
        return self.meet_planes(plane_a, plane_b)

    def point_plane_signed_distance(self, points: torch.Tensor, planes: torch.Tensor) -> torch.Tensor:
        # The dual-chart pseudoscalar orientation is opposite the ordinary
        # n·(x-p) convention, hence the explicit minus sign.
        return -self.incidence(planes, points)[..., 0]

    def point_line_distance(self, points: torch.Tensor, lines: torch.Tensor) -> torch.Tensor:
        # A regressive product joins each point to its target line.  Because the
        # line is the meet of orthonormal planes, the Euclidean norm of the
        # resulting plane normal is exactly the point--line distance.
        dual_points = self._dual(points, self.point_layout, self.plane_layout)
        dual_lines = self._dual(lines, self.line_layout, self.line_layout)
        dual_join = self.dual_meet(dual_points, dual_lines)
        join_plane = self._undual(dual_join, self.point_layout, self.plane_layout)
        return torch.linalg.vector_norm(join_plane[..., :3], dim=-1)

    @staticmethod
    def _wedge_sign(left_blade: int, right_blade: int) -> float:
        sign = 1.0
        for bit in range(4):
            if left_blade & (1 << bit):
                if (right_blade & ((1 << bit) - 1)).bit_count() % 2:
                    sign = -sign
        return sign

    def _dual(self, values: torch.Tensor, source: Any, target: Any) -> torch.Tensor:
        result = values.new_zeros(*values.shape[:-1], target.dim)
        for source_lane, blade in enumerate(source.basis_indices):
            complement = 0b1111 ^ blade
            target_lane = target.basis_indices.index(complement)
            result[..., target_lane] = self._wedge_sign(blade, complement) * values[..., source_lane]
        return result

    def _undual(self, values: torch.Tensor, source: Any, target: Any) -> torch.Tensor:
        result = values.new_zeros(*values.shape[:-1], target.dim)
        for target_lane, blade in enumerate(target.basis_indices):
            complement = 0b1111 ^ blade
            source_lane = source.basis_indices.index(complement)
            result[..., target_lane] = self._wedge_sign(blade, complement) * values[..., source_lane]
        return result


def _bivector_lane(layout: Any, axis_a: int, axis_b: int) -> int:
    return int(layout.basis_indices.index((1 << axis_a) | (1 << axis_b)))


def build_se3_subspace(algebra: AlgebraContext) -> GeneratorSubspace:
    """Map ``[rx,ry,rz,tx,ty,tz]`` to dual-PGA motor generators."""
    if (algebra.p, algebra.q, algebra.r) != (3, 0, 1):
        raise ValueError("the PGA SE(3) subspace requires Cl(3,0,1)")
    layout = algebra.layout((2,))
    mapping = torch.zeros(layout.dim, 6, device=algebra.device, dtype=algebra.dtype)
    mapping[_bivector_lane(layout, 1, 2), 0] = 1.0
    mapping[_bivector_lane(layout, 0, 2), 1] = -1.0
    mapping[_bivector_lane(layout, 0, 1), 2] = 1.0
    # clifra applies exp(-B/2) P exp(+B/2).  On point trivectors, -e_i0
    # therefore produces a positive translation along coordinate i.
    for axis in range(3):
        mapping[_bivector_lane(layout, axis, 3), 3 + axis] = -1.0
    return GeneratorSubspace(mapping)


def build_field(config: Config, *, algebra: AlgebraContext, chart: PGAPointChart) -> InvertibleBivectorField:
    controls = torch.linspace(0.0, 1.0, config.rbf_controls, device=algebra.device, dtype=algebra.dtype).unsqueeze(-1)
    action = algebra.plan_versor_action(
        grade=2,
        input=chart.layout,
        output=chart.layout,
        parameter=algebra.layout((2,)),
    )
    return InvertibleBivectorField(
        algebra,
        3,
        path_steps=1,
        projective=True,
        chart=chart,
        action=action,
        generator_sampler=RBFGeneratorSampler(controls, length_scale=config.rbf_length_scale),
        generator_subspace=build_se3_subspace(algebra),
        init_scale=0.0,
    )


def _sample_patch(patch: PlanePatch, count: int, generator: torch.Generator) -> torch.Tensor:
    uv = 2.0 * torch.rand(count, 2, generator=generator, device=patch.origin.device, dtype=patch.origin.dtype) - 1.0
    return patch.origin + uv[:, :1] * patch.extent_u * patch.axis_u + uv[:, 1:] * patch.extent_v * patch.axis_v


def _sample_line(feature: LineFeature, count: int, generator: torch.Generator) -> torch.Tensor:
    t = 0.02 + 0.96 * torch.rand(count, 1, generator=generator, device=feature.start.device, dtype=feature.start.dtype)
    t = torch.sort(t.squeeze(-1)).values.unsqueeze(-1)
    return feature.start + t * (feature.end - feature.start)


def build_scene(
    config: Config,
    pga: PGAGeometry,
    *,
    device: torch.device,
    dtype: torch.dtype,
    plane_samples: tuple[int, ...] | None = None,
    line_samples: tuple[int, ...] | None = None,
    seed_offset: int = 0,
) -> Scene:
    def v(values: list[float]) -> torch.Tensor:
        return torch.tensor(values, device=device, dtype=dtype)

    plane_specs = (
        ("ground", v([4.5, 0.0, -1.25]), v([1.0, 0.0, 0.0]), v([0.0, 1.0, 0.0]), 3.3, 3.1, "#78909c"),
        ("front wall", v([7.6, 0.0, 0.55]), v([0.0, 1.0, 0.0]), v([0.0, 0.0, 1.0]), 3.0, 1.8, "#7e8aa2"),
        ("side wall", v([4.5, -3.15, 0.45]), v([1.0, 0.0, 0.0]), v([0.0, 0.0, 1.0]), 3.0, 1.7, "#8d99ae"),
        (
            "ramp",
            v([4.15, 1.80, -0.88]),
            v([1.0, 0.0, 0.38]) / math.sqrt(1.0 + 0.38**2),
            v([0.0, 1.0, 0.0]),
            1.35,
            0.80,
            "#c48a5a",
        ),
    )
    planes: list[PlanePatch] = []
    for name, origin, axis_u, axis_v, extent_u, extent_v, color in plane_specs:
        normal = torch.linalg.cross(axis_u, axis_v, dim=-1)
        planes.append(PlanePatch(name, origin, axis_u, axis_v, extent_u, extent_v, pga.plane(normal, origin), color))

    line_specs = (
        ("door left", v([7.52, -0.90, -1.22]), v([7.52, -0.90, 1.35]), "#e76f51"),
        ("door right", v([7.52, 0.90, -1.22]), v([7.52, 0.90, 1.35]), "#e76f51"),
        ("door lintel", v([7.52, -0.90, 1.35]), v([7.52, 0.90, 1.35]), "#e76f51"),
        ("pole", v([4.25, -1.05, -1.22]), v([4.25, -1.05, 2.15]), "#2a9d8f"),
        ("sloped beam", v([2.25, -2.65, 1.25]), v([5.75, -2.65, 2.12]), "#457b9d"),
    )
    lines = tuple(
        LineFeature(name, start, end, pga.line_from_points(start, end), color) for name, start, end, color in line_specs
    )

    plane_samples = config.plane_samples if plane_samples is None else plane_samples
    line_samples = config.line_samples if line_samples is None else line_samples
    generator = torch.Generator(device=device).manual_seed(config.seed + seed_offset)
    point_chunks: list[torch.Tensor] = []
    plane_id_chunks: list[torch.Tensor] = []
    line_id_chunks: list[torch.Tensor] = []
    color_chunks: list[np.ndarray] = []
    for index, (patch, count) in enumerate(zip(planes, plane_samples)):
        points = _sample_patch(patch, count, generator)
        # Leave a real doorway in the front-wall samples.
        if patch.name == "front wall":
            local_y = points[:, 1]
            doorway = (local_y.abs() < 0.88) & (points[:, 2] < 1.32)
            rejected = int(doorway.sum())
            if rejected:
                replacement = _sample_patch(patch, rejected * 3, generator)
                replacement = replacement[~((replacement[:, 1].abs() < 0.88) & (replacement[:, 2] < 1.32))]
                points[doorway] = replacement[:rejected]
        point_chunks.append(points)
        plane_id_chunks.append(torch.full((count,), index, device=device, dtype=torch.long))
        line_id_chunks.append(torch.full((count,), -1, device=device, dtype=torch.long))
        color_chunks.append(np.repeat(np.array([patch.color], dtype=object), count))
    for index, (feature, count) in enumerate(zip(lines, line_samples)):
        point_chunks.append(_sample_line(feature, count, generator))
        plane_id_chunks.append(torch.full((count,), -1, device=device, dtype=torch.long))
        line_id_chunks.append(torch.full((count,), index, device=device, dtype=torch.long))
        color_chunks.append(np.repeat(np.array([feature.color], dtype=object), count))
    return Scene(
        planes=tuple(planes),
        lines=lines,
        world_points=torch.cat(point_chunks),
        plane_ids=torch.cat(plane_id_chunks),
        line_ids=torch.cat(line_id_chunks),
        colors=np.concatenate(color_chunks),
    )


def acquire_scan(
    config: Config,
    scene: Scene,
    truth: AnalyticTrajectory,
    *,
    seed_offset: int = 0,
    time_exponent: float = 1.0,
) -> Scan:
    # A spinning sweep visits targets in increasing initial-frame azimuth.  Each
    # target is then expressed in its instantaneous moving sensor frame.
    azimuth = torch.atan2(scene.world_points[:, 1], scene.world_points[:, 0])
    elevation = torch.atan2(scene.world_points[:, 2], torch.linalg.vector_norm(scene.world_points[:, :2], dim=-1))
    order = torch.argsort(azimuth + 0.035 * elevation)
    world = scene.world_points[order]
    count = world.shape[0]
    phase = (torch.arange(count, device=world.device, dtype=world.dtype) + 0.5) / count
    times = (0.002 + 0.996 * phase.pow(time_exponent)).unsqueeze(-1)
    with torch.no_grad():
        measured = truth.world_to_sensor(world, times)
        noise_generator = torch.Generator(device=world.device).manual_seed(config.seed + 1 + seed_offset)
        measured = measured + config.measurement_noise * torch.randn(
            measured.shape, generator=noise_generator, device=measured.device, dtype=measured.dtype
        )
    return Scan(
        measured_points=measured,
        world_points=world,
        acquisition_times=times,
        plane_ids=scene.plane_ids[order],
        line_ids=scene.line_ids[order],
        colors=scene.colors[order.cpu().numpy()],
    )


def _stratified_indices(candidates: torch.Tensor, count: int) -> torch.Tensor:
    if candidates.numel() <= count:
        return candidates
    positions = torch.linspace(0, candidates.numel() - 1, count, device=candidates.device).round().long()
    return candidates[positions]


def build_sparse_constraints(config: Config, scan: Scan) -> SparseConstraints:
    selected_planes = []
    for plane_id in range(len(config.plane_samples)):
        candidates = torch.nonzero(scan.plane_ids == plane_id, as_tuple=False).squeeze(-1)
        selected_planes.append(_stratified_indices(candidates, config.plane_constraints_per_patch))
    selected_lines = []
    for line_id in range(len(config.line_samples)):
        candidates = torch.nonzero(scan.line_ids == line_id, as_tuple=False).squeeze(-1)
        selected_lines.append(_stratified_indices(candidates, config.line_constraints_per_feature))
    plane_indices = torch.cat(selected_planes)
    line_indices = torch.cat(selected_lines)
    all_candidates = torch.arange(scan.measured_points.shape[0], device=scan.measured_points.device)
    anchor_indices = _stratified_indices(all_candidates, config.anchor_constraints)
    indices = torch.unique(torch.cat((plane_indices, line_indices, anchor_indices)), sorted=True)
    row_for_scan = torch.full((scan.measured_points.shape[0],), -1, device=indices.device, dtype=torch.long)
    row_for_scan[indices] = torch.arange(indices.numel(), device=indices.device)
    return SparseConstraints(
        indices=indices,
        plane_rows=row_for_scan[plane_indices],
        line_rows=row_for_scan[line_indices],
        anchor_rows=row_for_scan[anchor_indices],
        plane_ids=scan.plane_ids[plane_indices],
        line_ids=scan.line_ids[line_indices],
        # Only these surveyed points enter the estimator; the full world scan
        # and the analytic trajectory remain evaluation/synthesis data.
        anchor_targets=scan.world_points[anchor_indices],
    )


class DeskewObjective:
    def __init__(
        self,
        config: Config,
        scan: Scan,
        constraints: SparseConstraints,
        scene: Scene,
        pga: PGAGeometry,
    ):
        self.config = config
        self.input = scan.field_input(constraints.indices)
        self.constraints = constraints
        self.pga = pga
        self.planes = torch.stack([patch.plane for patch in scene.planes])
        self.lines = torch.stack([feature.line for feature in scene.lines])

    def __call__(self, field_model: InvertibleBivectorField) -> FitState:
        state = field_model.state(self.input)
        corrected = state.transformed_coordinates
        points = state.transformed_multivectors
        c = self.constraints
        plane_residual = self.pga.point_plane_signed_distance(points[c.plane_rows], self.planes[c.plane_ids])
        line_residual = self.pga.point_line_distance(points[c.line_rows], self.lines[c.line_ids])
        anchor_residual = corrected[c.anchor_rows] - c.anchor_targets
        controls = field_model.latent_coordinates[0]
        first_difference = controls[1:] - controls[:-1]
        second_difference = controls[2:] - 2.0 * controls[1:-1] + controls[:-2]
        start_input = CoordinateFieldInput(corrected.new_zeros(1, 3), sample_coordinates=corrected.new_zeros(1, 1))
        start_field = field_model.generator_sampler.sample(field_model.latent_coordinates, start_input).weights[0, 0]
        components = {
            "plane": plane_residual.square().mean(),
            "line": line_residual.square().mean(),
            "anchor": anchor_residual.square().mean(),
            "coefficient_second_difference": second_difference.square().mean(),
            "coefficient_first_difference": first_difference.square().mean(),
            "start_field": start_field.square().mean(),
        }
        loss = sum(self.config.weights[name] * value for name, value in components.items())
        return FitState(loss, components)


class LiveView:
    """Persistent scene whose corrected scan is updated during optimization."""

    def __init__(self, scene: Scene, scan: Scan):
        import matplotlib.pyplot as plt

        self.plt = plt
        self._last_event_time = 0.0
        self.closed = False
        self.scan = scan
        self.figure = plt.figure(figsize=(11.5, 7.4))
        self.interactive = self.figure.canvas.required_interactive_framework is not None
        if self.figure.canvas.manager is not None:
            self.figure.canvas.manager.set_window_title("PGA LiDAR deskewing")
        self.axis = self.figure.add_subplot(111, projection="3d")
        self.figure.canvas.mpl_connect("close_event", self._mark_closed)
        _configure_scan_axis(self.axis, "Current PGA-deskewed scan", scene)
        raw = _np(scan.measured_points)
        self.axis.scatter(*raw.T, c=scan.colors, s=3.0, alpha=0.14, linewidths=0, label="raw")
        self.current = self.axis.scatter(*raw.T, c=scan.colors, s=5.0, alpha=0.82, linewidths=0, label="current")
        self.axis.legend(loc="upper left", fontsize=8, frameon=False)
        self.status = self.figure.text(0.5, 0.025, "initializing", ha="center", family="monospace", fontsize=9)
        self.figure.suptitle("Continuous-time PGA LiDAR deskewing", fontsize=15, fontweight="bold")
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

    def update(self, step: int | str, field_model: InvertibleBivectorField, state: FitState) -> None:
        if self.closed:
            return
        with torch.no_grad():
            corrected = field_model(self.scan.field_input())
        values = _np(corrected)
        self.current._offsets3d = (values[:, 0], values[:, 1], values[:, 2])
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


def optimize(
    config: Config,
    field_model: InvertibleBivectorField,
    objective: DeskewObjective,
    initial_state: FitState,
    view: LiveView | None = None,
) -> OptimizationResult:
    optimizer = torch.optim.Adam(field_model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.optimization_steps, eta_min=0.12 * config.learning_rate
    )
    history = [float(initial_state.loss)]
    best_step = 0
    best_loss = history[0]
    best_parameters = field_model.latent_coordinates.detach().clone()
    print(
        f"[{0:4d}/{config.optimization_steps}] loss={initial_state.loss.item():.4e} "
        f"plane={math.sqrt(initial_state.components['plane'].item()):.4f} "
        f"line={math.sqrt(initial_state.components['line'].item()):.4f}"
    )
    if view is not None:
        view.update(0, field_model, initial_state)
    for step in range(1, config.optimization_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        state = objective(field_model)
        if not torch.isfinite(state.loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        state.loss.backward()
        torch.nn.utils.clip_grad_norm_(field_model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            state = objective(field_model)
        value = float(state.loss)
        history.append(value)
        if value < best_loss:
            best_step = len(history) - 1
            best_loss = value
            best_parameters.copy_(field_model.latent_coordinates.detach())
        if view is not None:
            view.pump_events()
            if step % config.live_every == 0 or step == config.optimization_steps:
                view.update(step, field_model, state)
        if step % config.log_every == 0 or step == config.optimization_steps:
            print(
                f"[{step:4d}/{config.optimization_steps}] loss={state.loss.item():.4e} "
                f"plane={math.sqrt(state.components['plane'].item()):.4f} "
                f"line={math.sqrt(state.components['line'].item()):.4f} "
                f"anchor={math.sqrt(state.components['anchor'].item()):.4f}"
            )

    if config.lbfgs_steps:
        lbfgs = torch.optim.LBFGS(
            field_model.parameters(),
            lr=0.65,
            max_iter=config.lbfgs_steps,
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            loss = objective(field_model).loss
            nonlocal best_step, best_loss
            value = float(loss.detach())
            history.append(value)
            if value < best_loss:
                best_step = len(history) - 1
                best_loss = value
                best_parameters.copy_(field_model.latent_coordinates.detach())
            loss.backward()
            return loss

        lbfgs.step(closure)
        final_loss = float(objective(field_model).loss.detach())
        history.append(final_loss)
        if final_loss < best_loss:
            best_step = len(history) - 1
            best_loss = final_loss
            best_parameters.copy_(field_model.latent_coordinates.detach())
        print(f"[L-BFGS/{config.lbfgs_steps}] loss={final_loss:.4e}")
    final_history_loss = history[-1]
    with torch.no_grad():
        field_model.latent_coordinates.copy_(best_parameters)
    return OptimizationResult(history, best_step, best_loss, final_history_loss)


def evaluate_pose_field(field_model: InvertibleBivectorField, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probes = times.new_zeros(times.shape[0], 4, 3)
    probes[:, 1, 0] = 1.0
    probes[:, 2, 1] = 1.0
    probes[:, 3, 2] = 1.0
    labels = times[:, None, :].expand(times.shape[0], 4, 1)
    transformed = field_model(CoordinateFieldInput(probes, sample_coordinates=labels, domain_shape=(4,)))
    origins = transformed[:, 0]
    rotations = (transformed[:, 1:] - origins[:, None, :]).transpose(-1, -2)
    return origins, rotations


def rotation_errors(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    relative = actual.transpose(-1, -2) @ expected
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.acos(cosine)


def _geometry_residuals(
    pga: PGAGeometry, scene: Scene, scan: Scan, coordinates: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    points = pga.chart.embed(coordinates)
    plane_mask = scan.plane_ids >= 0
    line_mask = scan.line_ids >= 0
    planes = torch.stack([patch.plane for patch in scene.planes])
    lines = torch.stack([feature.line for feature in scene.lines])
    plane = pga.point_plane_signed_distance(points[plane_mask], planes[scan.plane_ids[plane_mask]])
    line = pga.point_line_distance(points[line_mask], lines[scan.line_ids[line_mask]])
    return plane, line


def _scan_evaluation(
    learned: InvertibleBivectorField,
    truth: AnalyticTrajectory,
    pga: PGAGeometry,
    scene: Scene,
    scan: Scan,
) -> tuple[dict[str, float], torch.Tensor]:
    corrected = learned(scan.field_input())
    oracle_corrected = truth.sensor_to_world(scan.measured_points, scan.acquisition_times)
    raw_plane, raw_line = _geometry_residuals(pga, scene, scan, scan.measured_points)
    corrected_plane, corrected_line = _geometry_residuals(pga, scene, scan, corrected)
    oracle_plane, oracle_line = _geometry_residuals(pga, scene, scan, oracle_corrected)
    metrics = {
        "returns": int(scan.measured_points.shape[0]),
        "raw_plane_rmse": float(raw_plane.square().mean().sqrt()),
        "corrected_plane_rmse": float(corrected_plane.square().mean().sqrt()),
        "oracle_plane_rmse": float(oracle_plane.square().mean().sqrt()),
        "raw_line_rmse": float(raw_line.square().mean().sqrt()),
        "corrected_line_rmse": float(corrected_line.square().mean().sqrt()),
        "oracle_line_rmse": float(oracle_line.square().mean().sqrt()),
        "raw_world_rmse": float((scan.measured_points - scan.world_points).square().sum(dim=-1).mean().sqrt()),
        "corrected_world_rmse": float((corrected - scan.world_points).square().sum(dim=-1).mean().sqrt()),
        "oracle_world_rmse": float((oracle_corrected - scan.world_points).square().sum(dim=-1).mean().sqrt()),
    }
    return metrics, corrected


def evaluate(
    config: Config,
    learned: InvertibleBivectorField,
    truth: AnalyticTrajectory,
    pga: PGAGeometry,
    scene: Scene,
    scan: Scan,
    constraints: SparseConstraints,
    heldout_scene: Scene,
    heldout_scan: Scan,
    optimization: OptimizationResult,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    dense_t = torch.linspace(
        0.0, 1.0, config.dense_times, device=scan.measured_points.device, dtype=scan.measured_points.dtype
    ).unsqueeze(-1)
    with torch.no_grad():
        primary, corrected = _scan_evaluation(learned, truth, pga, scene, scan)
        heldout, _ = _scan_evaluation(learned, truth, pga, heldout_scene, heldout_scan)
        learned_position, learned_rotation = evaluate_pose_field(learned, dense_t)
        true_position, true_rotation = truth.pose(dense_t)
        translation_error = torch.linalg.vector_norm(learned_position - true_position, dim=-1)
        rotation_error = rotation_errors(learned_rotation, true_rotation)
        reconstructed = learned.inverse(CoordinateFieldInput(corrected, sample_coordinates=scan.acquisition_times))
    oracle = {
        "available_to_optimizer": False,
        "purpose": "simulation noise-floor and trajectory evaluation only",
        "primary_plane_rmse": primary.pop("oracle_plane_rmse"),
        "primary_line_rmse": primary.pop("oracle_line_rmse"),
        "primary_world_rmse": primary.pop("oracle_world_rmse"),
        "heldout_plane_rmse": heldout.pop("oracle_plane_rmse"),
        "heldout_line_rmse": heldout.pop("oracle_line_rmse"),
        "heldout_world_rmse": heldout.pop("oracle_world_rmse"),
    }
    primary.pop("returns")
    pose = {
        "samples": config.dense_times,
        "translation_rmse": float(translation_error.square().mean().sqrt()),
        "translation_max": float(translation_error.max()),
        "rotation_rmse_degrees": math.degrees(float(rotation_error.square().mean().sqrt())),
        "rotation_max_degrees": math.degrees(float(rotation_error.max())),
    }
    report: dict[str, Any] = {
        "experiment": "pga_lidar_deskewing",
        "algebra": "Cl(3,0,1)",
        "signature": "+++0",
        "problem": {
            "scan_returns": int(scan.measured_points.shape[0]),
            "optimization_returns": int(constraints.indices.numel()),
            "plane_correspondences": int(constraints.plane_rows.numel()),
            "line_correspondences": int(constraints.line_rows.numel()),
            "surveyed_point_anchors": int(constraints.anchor_rows.numel()),
            "truth_representation": "independent analytic matrix trajectory",
            "estimator_representation": "normalized Gaussian-RBF PGA motor field",
            "analytic_trajectory_available_to_optimizer": False,
            "surveyed_anchor_targets_available_to_optimizer": True,
            "map_feature_correspondences_supplied": True,
        },
        "optimization": {
            "algorithm": "Adam with cosine learning-rate decay, then L-BFGS",
            "adam_steps": config.optimization_steps,
            "lbfgs_max_iterations": config.lbfgs_steps,
            "initial_loss": optimization.history[0],
            "best_step": optimization.best_step,
            "best_loss": optimization.best_loss,
            "final_history_loss": optimization.final_history_loss,
            "history": optimization.history,
            "regularizer_semantics": {
                "coefficient_first_difference": "adjacent stored RBF coefficients; not physical velocity",
                "coefficient_second_difference": "second difference of stored RBF coefficients; not physical acceleration",
                "start_field": "normalized-RBF field evaluated at t=0",
            },
        },
        "primary": primary,
        "validation": {
            "uses_optimizer": False,
            "heldout_scan": heldout,
            "dense_pose": pose,
            "simulation_only_oracle": oracle,
            "inverse_round_trip_max": float((reconstructed - scan.measured_points).abs().max()),
        },
        "limitations": [
            "Map-feature correspondences and surveyed anchors are supplied in the synthetic scene.",
            "Plane/line incidences alone leave the late scan locally pose-underdetermined; anchors support dense trajectory recovery.",
            "Sensor noise is independent Gaussian noise and the environment is static.",
            "The RBF estimator approximates, but does not share, the analytic truth representation.",
        ],
    }
    arrays = {
        "corrected": corrected,
        "dense_t": dense_t.squeeze(-1),
        "translation_error": translation_error,
        "rotation_error": rotation_error,
    }
    return report, arrays


def _np(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy()


def _draw_map(axis: Any, scene: Scene, *, alpha: float = 0.15) -> None:
    for patch in scene.planes:
        corners = []
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            corners.append(_np(patch.origin + su * patch.extent_u * patch.axis_u + sv * patch.extent_v * patch.axis_v))
        corners.append(corners[0])
        corners_array = np.stack(corners)
        axis.plot(*corners_array.T, color=patch.color, alpha=0.55, linewidth=1.0)
        surface = np.stack((corners[0], corners[3], corners[1], corners[2])).reshape(2, 2, 3)
        axis.plot_surface(
            surface[:, :, 0], surface[:, :, 1], surface[:, :, 2], color=patch.color, alpha=alpha, shade=False
        )
    for feature in scene.lines:
        values = np.stack((_np(feature.start), _np(feature.end)))
        axis.plot(*values.T, color=feature.color, linewidth=3.0, alpha=0.85)


def _configure_scan_axis(axis: Any, title: str, scene: Scene) -> None:
    axis.set_title(title, fontsize=12, fontweight="bold")
    axis.set_xlim(0.0, 8.2)
    axis.set_ylim(-3.6, 3.6)
    axis.set_zlim(-1.5, 2.6)
    axis.set_box_aspect((1.35, 1.0, 0.72))
    axis.view_init(elev=22, azim=-118)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    _draw_map(axis, scene)


def save_result(config: Config, scene: Scene, scan: Scan, corrected: torch.Tensor) -> Path:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(14.8, 7.2), constrained_layout=True)
    FigureCanvasAgg(figure)
    raw_axis = figure.add_subplot(121, projection="3d")
    corrected_axis = figure.add_subplot(122, projection="3d")
    raw_axis.scatter(*_np(scan.measured_points).T, c=scan.colors, s=5.0, alpha=0.62, linewidths=0)
    corrected_axis.scatter(*_np(corrected).T, c=scan.colors, s=5.0, alpha=0.84, linewidths=0)
    _configure_scan_axis(raw_axis, "Raw moving scan", scene)
    _configure_scan_axis(corrected_axis, "PGA-deskewed scan", scene)
    figure.suptitle("Continuous-time PGA LiDAR deskewing", fontsize=17, fontweight="bold")
    figure.text(
        0.5,
        0.94,
        "Matched view · sparse map constraints · continuous motor field",
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
    arrays: dict[str, torch.Tensor],
) -> Path:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(14.8, 4.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    history_axis, pose_axis, residual_axis = axes
    history_axis.semilogy(optimization.history, color="#264653", linewidth=2.0)
    history_axis.scatter(
        [optimization.best_step], [optimization.best_loss], color="#e76f51", s=26, zorder=3, label="restored best"
    )
    history_axis.set_title("Optimization", fontweight="bold")
    history_axis.set_xlabel("objective evaluation")
    history_axis.set_ylabel("objective")
    history_axis.grid(alpha=0.22)
    history_axis.legend(fontsize=8, frameon=False)

    time = _np(arrays["dense_t"])
    pose_axis.plot(time, 100.0 * _np(arrays["translation_error"]), color="#e76f51", linewidth=2.0)
    pose_axis.set_xlabel("acquisition time")
    pose_axis.set_ylabel("translation error [cm]", color="#e76f51")
    pose_axis.tick_params(axis="y", labelcolor="#e76f51")
    rotation_axis = pose_axis.twinx()
    rotation_axis.plot(time, np.degrees(_np(arrays["rotation_error"])), color="#457b9d", linewidth=1.8)
    rotation_axis.set_ylabel("rotation error [deg]", color="#457b9d")
    rotation_axis.tick_params(axis="y", labelcolor="#457b9d")
    pose_axis.set_title("Dense pose error", fontweight="bold")
    pose_axis.grid(alpha=0.22)

    primary = report["primary"]
    heldout = report["validation"]["heldout_scan"]
    labels = ("plane", "line", "world")
    raw = np.array([primary[f"raw_{name}_rmse"] for name in labels])
    corrected = np.array([primary[f"corrected_{name}_rmse"] for name in labels])
    heldout_corrected = np.array([heldout[f"corrected_{name}_rmse"] for name in labels])
    x = np.arange(3)
    width = 0.25
    residual_axis.bar(x - width, raw, width, color="#adb5bd", label="raw")
    residual_axis.bar(x, corrected, width, color="#2a9d8f", label="corrected")
    residual_axis.bar(x + width, heldout_corrected, width, color="#457b9d", label="held-out")
    residual_axis.set_yscale("log")
    residual_axis.set_xticks(x, labels)
    residual_axis.set_ylabel("RMSE [m]")
    residual_axis.set_title("Geometric residuals", fontweight="bold")
    residual_axis.grid(axis="y", alpha=0.22)
    residual_axis.legend(fontsize=8, frameon=False)

    figure.suptitle("PGA LiDAR diagnostics", fontsize=16, fontweight="bold")
    figure.text(
        0.5,
        0.015,
        f"{heldout['returns']} held-out returns · no retraining · "
        f"{100 * heldout['corrected_world_rmse']:.1f} cm world RMSE",
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
    problem = report["problem"]
    primary = report["primary"]
    validation = report["validation"]
    heldout = validation["heldout_scan"]
    pose = validation["dense_pose"]
    optimization = report["optimization"]
    print("\nPGA CONTINUOUS-TIME LIDAR DESKEWING\n")
    print(
        f"primary      plane {primary['raw_plane_rmse']:.4f}→{primary['corrected_plane_rmse']:.4f} m | "
        f"line {primary['raw_line_rmse']:.4f}→{primary['corrected_line_rmse']:.4f} m | "
        f"world {primary['raw_world_rmse']:.4f}→{primary['corrected_world_rmse']:.4f} m"
    )
    print(
        f"validation   {heldout['returns']} held-out returns | plane {heldout['corrected_plane_rmse']:.4f} m | "
        f"line {heldout['corrected_line_rmse']:.4f} m | world {heldout['corrected_world_rmse']:.4f} m"
    )
    print(
        f"numerical    pose {100 * pose['translation_rmse']:.2f} cm / {pose['rotation_rmse_degrees']:.3f}° | "
        f"inverse {validation['inverse_round_trip_max']:.1e}"
    )
    print(
        f"optimization best {optimization['best_loss']:.3e} at {optimization['best_step']} | "
        f"final {optimization['final_history_loss']:.3e} | {problem['optimization_returns']} selected returns"
    )
    print()
    for name, passed in report["checks"].items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")


def acceptance_checks(report: dict[str, Any]) -> dict[str, bool]:
    primary = report["primary"]
    validation = report["validation"]
    heldout = validation["heldout_scan"]
    pose = validation["dense_pose"]
    return {
        "primary deskewing improves point-plane and point-line residuals": (
            primary["corrected_plane_rmse"] < 0.1 * primary["raw_plane_rmse"]
            and primary["corrected_line_rmse"] < 0.125 * primary["raw_line_rmse"]
        ),
        "independent held-out scan improves": (
            heldout["corrected_plane_rmse"] < 0.16 * heldout["raw_plane_rmse"]
            and heldout["corrected_line_rmse"] < 0.18 * heldout["raw_line_rmse"]
            and heldout["corrected_world_rmse"] < 0.10
        ),
        "dense analytic trajectory is recovered": pose["translation_rmse"] < 0.06
        and pose["rotation_rmse_degrees"] < 3.0,
        "PGA path is invertible": validation["inverse_round_trip_max"] < 2e-10,
    }


def run(config: Config = Config()) -> dict[str, Any]:
    device, dtype = torch.device("cpu"), torch.float64
    algebra = AlgebraContext(3, 0, 1, device=device, dtype=dtype)
    chart = PGAPointChart(algebra)
    pga = PGAGeometry(algebra, chart)
    truth = AnalyticTrajectory()
    scene = build_scene(config, pga, device=device, dtype=dtype)
    scan = acquire_scan(config, scene, truth)
    heldout_scene = build_scene(
        config,
        pga,
        device=device,
        dtype=dtype,
        plane_samples=config.heldout_plane_samples,
        line_samples=config.heldout_line_samples,
        seed_offset=101,
    )
    heldout_scan = acquire_scan(config, heldout_scene, truth, seed_offset=101, time_exponent=0.93)
    constraints = build_sparse_constraints(config, scan)
    learned = build_field(config, algebra=algebra, chart=chart)
    objective = DeskewObjective(config, scan, constraints, scene, pga)
    with torch.no_grad():
        initial_state = objective(learned)
    print(
        f"setup         {scan.measured_points.shape[0]} timed returns | {constraints.indices.numel()} selected returns | "
        f"{config.rbf_controls} RBF controls | seed {config.seed}"
    )
    view = LiveView(scene, scan) if config.live else None
    optimization = optimize(config, learned, objective, initial_state, view)
    report, arrays = evaluate(
        config,
        learned,
        truth,
        pga,
        scene,
        scan,
        constraints,
        heldout_scene,
        heldout_scan,
        optimization,
    )
    with torch.no_grad():
        final_state = objective(learned)
    if view is not None:
        view.update("restored best", learned, final_state)
    report["checks"] = acceptance_checks(report)
    result_path = save_result(config, scene, scan, arrays["corrected"])
    diagnostics_path = save_diagnostics(config, optimization, report, arrays)
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
    parser.add_argument("--lbfgs-steps", type=int, default=Config.lbfgs_steps)
    parser.add_argument("--output-dir", type=Path, default=Config.output_dir)
    parser.add_argument("--live", action="store_true", help="show the scan evolving during optimization")
    args = parser.parse_args()
    return replace(
        Config(),
        optimization_steps=args.steps,
        lbfgs_steps=args.lbfgs_steps,
        output_dir=args.output_dir,
        live=args.live,
    )


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
