# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Sparse geometric constraints induce a transferable continuum deformation."""

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
import torch.nn.functional as F

from clifra import AlgebraContext
from research.transformation_fields import (
    CoordinateFieldInput,
    GeneratorSubspace,
    InvertibleBivectorField,
    RBFGeneratorSampler,
)


@dataclass(frozen=True)
class Config:
    seed: int = 17
    optimization_sections: int = 42
    surface_samples: int = 4
    dense_sections: int = 180
    dense_surface_samples: int = 12
    # Fixed undeformed rod length, independent of any gate-to-material assignment.
    robot_length: float = 1.4829312322619825
    robot_radius: float = 0.014
    rbf_controls: int = 14
    rbf_length_scale: float = 0.115
    path_steps: int = 10
    init_scale: float = 1e-3
    optimization_steps: int = 800
    learning_rate: float = 0.0125
    minimum_learning_rate_fraction: float = 0.12
    gradient_clip: float = 8.0
    log_every: int = 100
    live_every: int = 20
    ring_major_radius: float = 0.105
    ring_tube_radius: float = 0.021
    clearance_margin: float = 0.004
    collision_temperature: float = 0.006
    strain_soft_limit: float = 0.055
    max_curvature: float = 22.0

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "gate": 320.0,
            "gate_alignment": 10.0,
            "tip": 420.0,
            "orientation": 7.0,
            "collision": 0.0936,
            "strain": 32.0,
            "curvature": 0.012,
            "base": 600.0,
            "generator": 0.002,
            "material_smoothness": 0.045,
            "stage_smoothness": 0.035,
        }
    )
    output_dir: Path = Path("outputs/sparse_continuum_threading")
    live: bool = False


@dataclass(frozen=True)
class RingObstacle:
    center: torch.Tensor
    normal: torch.Tensor
    major_radius: float
    tube_radius: float
    color: str


@dataclass(frozen=True)
class Scene:
    """Sparse gate, tip, and obstacle geometry."""

    rings: tuple[RingObstacle, ...]
    gate_centers: torch.Tensor
    gate_normals: torch.Tensor
    tip_position: torch.Tensor
    tip_frame: torch.Tensor
    robot_length: float


@dataclass(frozen=True)
class RobotSamples:
    """A straight rod whose persistent material label is independent of xyz."""

    xyz: torch.Tensor
    material_s: torch.Tensor
    section_s: torch.Tensor
    radius: float

    @property
    def field_input(self) -> CoordinateFieldInput:
        return CoordinateFieldInput(
            coordinates=self.xyz,
            sample_coordinates=self.material_s,
            domain_shape=tuple(self.xyz.shape[:-1]),
        )

    @property
    def point_count(self) -> int:
        return int(self.xyz[..., 0].numel())

    @property
    def reference_segment_lengths(self) -> torch.Tensor:
        return torch.linalg.vector_norm(self.xyz[1:, 0] - self.xyz[:-1, 0], dim=-1)


@dataclass
class FitState:
    loss: torch.Tensor
    final_coordinates: torch.Tensor
    metrics: dict[str, torch.Tensor]


@dataclass(frozen=True)
class OptimizationResult:
    history: list[float]
    best_step: int
    best_loss: float
    final_history_loss: float


_RING_COLORS = ("#ef8354", "#8f6ccf", "#35a7a0")


def _normalize(values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return values / torch.linalg.vector_norm(values, dim=-1, keepdim=True).clamp_min(eps)


def _linear_interp(samples_s: torch.Tensor, samples: torch.Tensor, query_s: torch.Tensor) -> torch.Tensor:
    query = query_s.to(device=samples_s.device, dtype=samples_s.dtype).clamp(samples_s[0], samples_s[-1])
    right = torch.searchsorted(samples_s, query, right=True).clamp(1, samples_s.numel() - 1)
    left = right - 1
    amount = (query - samples_s[left]) / (samples_s[right] - samples_s[left]).clamp_min(1e-12)
    return samples[left] + amount.unsqueeze(-1) * (samples[right] - samples[left])


def build_scene(config: Config, *, device: torch.device, dtype: torch.dtype) -> Scene:
    """Construct the sparse gate/tip problem geometry."""

    knots = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [0.28, 0.16, 0.03],
            [0.55, -0.13, 0.20],
            [0.82, 0.11, 0.36],
            [1.08, 0.01, 0.45],
        ],
        device=device,
        dtype=dtype,
    )
    directions = _normalize(
        torch.tensor(
            [
                [1.00, 0.00, 0.00],
                [0.78, 0.38, 0.22],
                [0.76, -0.36, 0.48],
                [0.70, 0.31, 0.62],
                [0.90, -0.25, 0.35],
            ],
            device=device,
            dtype=dtype,
        )
    )

    tip_x = directions[-1]
    world_z = tip_x.new_tensor([0.0, 0.0, 1.0])
    tip_y0 = _normalize(torch.linalg.cross(world_z, tip_x, dim=-1))
    tip_z0 = _normalize(torch.linalg.cross(tip_x, tip_y0, dim=-1))
    roll = 0.95 * math.pi
    tip_y = math.cos(roll) * tip_y0 + math.sin(roll) * tip_z0
    tip_z = -math.sin(roll) * tip_y0 + math.cos(roll) * tip_z0
    tip_frame = torch.stack((tip_x, _normalize(tip_y), _normalize(tip_z)))

    rings = tuple(
        RingObstacle(
            center=knots[index + 1],
            normal=directions[index + 1],
            major_radius=config.ring_major_radius,
            tube_radius=config.ring_tube_radius,
            color=_RING_COLORS[index],
        )
        for index in range(3)
    )
    return Scene(
        rings=rings,
        gate_centers=knots[1:-1],
        gate_normals=directions[1:-1],
        tip_position=knots[-1],
        tip_frame=tip_frame,
        robot_length=config.robot_length,
    )


def build_robot(section_count: int, surface_count: int, scene: Scene, config: Config) -> RobotSamples:
    """Sample a cylinder; every point on a section shares one material s."""
    s = torch.linspace(0.0, 1.0, section_count, device=scene.gate_centers.device, dtype=scene.gate_centers.dtype)
    centers = torch.stack((scene.robot_length * s, torch.zeros_like(s), torch.zeros_like(s)), dim=-1)
    angle = torch.arange(surface_count, device=s.device, dtype=s.dtype) * (2.0 * math.pi / surface_count)
    offsets = torch.stack(
        (
            torch.zeros_like(angle),
            config.robot_radius * torch.cos(angle),
            config.robot_radius * torch.sin(angle),
        ),
        dim=-1,
    )
    surface = centers[:, None, :] + offsets[None, :, :]
    xyz = torch.cat((centers[:, None, :], surface), dim=1)
    material_s = s[:, None, None].expand(section_count, surface_count + 1, 1).clone()
    return RobotSamples(xyz=xyz, material_s=material_s, section_s=s, radius=config.robot_radius)


def torus_clearance(points: torch.Tensor, scene: Scene, robot_radius: float) -> torch.Tensor:
    """Conservative centerline-to-solid clearance for oriented torus gates."""
    centers = torch.stack([ring.center for ring in scene.rings])
    normals = torch.stack([ring.normal for ring in scene.rings])
    major = points.new_tensor([ring.major_radius for ring in scene.rings])
    tube = points.new_tensor([ring.tube_radius for ring in scene.rings])
    relative = points.unsqueeze(-2) - centers
    axial = (relative * normals).sum(dim=-1)
    radial = torch.sqrt((relative.square().sum(dim=-1) - axial.square()).clamp_min(torch.finfo(points.dtype).eps))
    distance_to_circle = torch.sqrt((radial - major).square() + axial.square() + torch.finfo(points.dtype).eps)
    return distance_to_circle - tube - float(robot_radius)


def axial_strain(points: torch.Tensor, robot: RobotSamples) -> torch.Tensor:
    lengths = torch.linalg.vector_norm(points[..., 1:, :] - points[..., :-1, :], dim=-1)
    return lengths / robot.reference_segment_lengths.clamp_min(torch.finfo(points.dtype).eps) - 1.0


def discrete_curvature(points: torch.Tensor) -> torch.Tensor:
    segments = points[..., 1:, :] - points[..., :-1, :]
    length = torch.linalg.vector_norm(segments, dim=-1).clamp_min(torch.finfo(points.dtype).eps)
    tangent = segments / length.unsqueeze(-1)
    turn = torch.linalg.vector_norm(tangent[..., 1:, :] - tangent[..., :-1, :], dim=-1)
    return turn / (0.5 * (length[..., 1:] + length[..., :-1])).clamp_min(torch.finfo(points.dtype).eps)


def section_frame(coordinates: torch.Tensor) -> torch.Tensor:
    """Recover the carried xyz frame from two points on each rigid section."""
    center = coordinates[..., 0, :]
    frame_y = _normalize(coordinates[..., 1, :] - center)
    quarter_index = max(2, 1 + (coordinates.shape[-2] - 1) // 4)
    raw_z = coordinates[..., quarter_index, :] - center
    frame_z = _normalize(raw_z - (raw_z * frame_y).sum(dim=-1, keepdim=True) * frame_y)
    frame_x = _normalize(torch.linalg.cross(frame_y, frame_z, dim=-1))
    return torch.stack((frame_x, frame_y, frame_z), dim=-2)


def _bivector_lane(layout: Any, axis_a: int, axis_b: int) -> int:
    blade = (1 << axis_a) | (1 << axis_b)
    position = {basis_index: lane for lane, basis_index in enumerate(layout.basis_indices)}
    return int(position[blade])


def build_se3_generator_subspace(algebra: AlgebraContext) -> GeneratorSubspace:
    """Map [wx, wy, wz, tx, ty, tz] twists into Cl(4,1) bivectors."""
    if (algebra.p, algebra.q, algebra.r) != (4, 1, 0):
        raise ValueError("the 3-D conformal SE(3) map requires Cl(4,1,0)")
    layout = algebra.layout((2,))
    mapping = torch.zeros(layout.dim, 6, device=algebra.device, dtype=algebra.dtype)
    mapping[_bivector_lane(layout, 1, 2), 0] = 1.0
    mapping[_bivector_lane(layout, 0, 2), 1] = -1.0
    mapping[_bivector_lane(layout, 0, 1), 2] = 1.0
    for axis in range(3):
        # e_inf = e- + e+; exp(-B/2) then gives the desired translation sign.
        mapping[_bivector_lane(layout, axis, 3), 3 + axis] = 1.0
        mapping[_bivector_lane(layout, axis, 4), 3 + axis] = 1.0
    return GeneratorSubspace(mapping)


def build_field(config: Config, *, device: torch.device, dtype: torch.dtype) -> InvertibleBivectorField:
    algebra = AlgebraContext(4, 1, 0, device=device, dtype=dtype)
    subspace = build_se3_generator_subspace(algebra)
    controls = torch.linspace(0.0, 1.0, config.rbf_controls, device=device, dtype=dtype).unsqueeze(-1)
    return InvertibleBivectorField(
        algebra,
        3,
        path_steps=config.path_steps,
        conformal=True,
        generator_sampler=RBFGeneratorSampler(controls, length_scale=config.rbf_length_scale),
        generator_subspace=subspace,
        init_scale=config.init_scale,
    )


def current_gate_crossings(
    centers: torch.Tensor, section_s: torch.Tensor, scene: Scene
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find the first forward centerline crossing of each physical gate plane.

    Segment selection is detached. Position and material-coordinate interpolation
    within the selected segment remain differentiable.
    """
    relative = centers[:, None, :] - scene.gate_centers[None, :, :]
    signed = (relative * scene.gate_normals[None, :, :]).sum(dim=-1).transpose(0, 1)
    before, after = signed[:, :-1], signed[:, 1:]
    eps = 32.0 * torch.finfo(centers.dtype).eps
    forward = (before.detach() <= 0.0) & (after.detach() >= 0.0) & ((after - before).detach() > eps)
    valid = forward.any(dim=-1)
    first = forward.long().argmax(dim=-1)
    fallback = torch.minimum(before.detach().abs(), after.detach().abs()).argmin(dim=-1)
    left = torch.where(valid, first, fallback)

    p0, p1 = centers[left], centers[left + 1]
    d0 = ((p0 - scene.gate_centers) * scene.gate_normals).sum(dim=-1)
    d1 = ((p1 - scene.gate_centers) * scene.gate_normals).sum(dim=-1)
    denominator = d1 - d0
    safe_denominator = torch.where(
        denominator.abs() < eps,
        torch.where(denominator < 0.0, -eps, eps),
        denominator,
    )
    fraction = (-d0 / safe_denominator).clamp(0.0, 1.0)
    crossing = p0 + fraction[:, None] * (p1 - p0)
    tangent = _normalize(p1 - p0)
    material_s = section_s[left] + fraction * (section_s[left + 1] - section_s[left])
    return crossing, tangent, material_s, valid


def geometry_metrics(
    final: torch.Tensor,
    robot: RobotSamples,
    scene: Scene,
) -> dict[str, torch.Tensor]:
    centers = final[..., 0, :]
    frames = section_frame(final)
    tip_dots = (frames[-1] * scene.tip_frame).sum(dim=-1).clamp(-1.0, 1.0)
    gate_crossing, gate_tangent, _, _ = current_gate_crossings(centers, robot.section_s, scene)
    gate_dot = (gate_tangent * scene.gate_normals).sum(dim=-1).clamp(-1.0, 1.0)
    strain = axial_strain(centers, robot)
    curvature = discrete_curvature(centers)
    return {
        "tip_position_error": torch.linalg.vector_norm(centers[-1] - scene.tip_position),
        "tip_orientation_error": torch.acos(tip_dots).amax(),
        "gate_offset": torch.linalg.vector_norm(gate_crossing - scene.gate_centers, dim=-1).amax(),
        "gate_orientation_error": torch.acos(gate_dot).amax(),
        "minimum_clearance": torus_clearance(centers, scene, robot.radius).amin(),
        "maximum_axial_strain": strain.abs().amax(),
        "maximum_curvature": curvature.amax(),
        "base_drift": torch.linalg.vector_norm(final[0] - robot.xyz[0], dim=-1).amax(),
    }


class SparseThreadingObjective:
    def __init__(self, config: Config, robot: RobotSamples, scene: Scene):
        self.config = config
        self.robot = robot
        self.scene = scene

    def __call__(self, field_model: InvertibleBivectorField) -> FitState:
        # A shared material s gives every point on a section the same composed action.
        final = field_model(self.robot.field_input)
        centers = final[..., 0, :]
        frames = section_frame(final)
        gate_centers, gate_tangent, _, _ = current_gate_crossings(centers, self.robot.section_s, self.scene)
        gate_dot = (gate_tangent * self.scene.gate_normals).sum(dim=-1).clamp(-1.0, 1.0)
        tip_dots = (frames[-1] * self.scene.tip_frame).sum(dim=-1).clamp(-1.0, 1.0)

        clearance = torus_clearance(centers, self.scene, self.robot.radius)
        violation = F.softplus((self.config.clearance_margin - clearance) / self.config.collision_temperature)
        strain_values = axial_strain(centers, self.robot)
        strain_excess = F.relu(strain_values.abs() - self.config.strain_soft_limit)
        curvature_values = discrete_curvature(centers)
        curvature_excess = F.relu(curvature_values - self.config.max_curvature)

        controls = field_model.latent_coordinates
        material_smoothness = (controls[:, 1:] - controls[:, :-1]).square().mean()
        if controls.shape[1] > 2:
            material_smoothness = (
                material_smoothness
                + 0.35 * (controls[:, 2:] - 2.0 * controls[:, 1:-1] + controls[:, :-2]).square().mean()
            )

        components = {
            "gate": (gate_centers - self.scene.gate_centers).square().sum(dim=-1).mean(),
            "gate_alignment": (1.0 - gate_dot).square().mean(),
            "tip": (centers[-1] - self.scene.tip_position).square().sum(),
            "orientation": (1.0 - tip_dots).square().sum(),
            "collision": violation.square().mean() + 0.5 * violation.amax().square(),
            "strain": (
                0.2 * strain_values.square().mean() + strain_excess.square().mean() + strain_excess.amax().square()
            ),
            "curvature": curvature_excess.square().mean() + 0.25 * curvature_excess.amax().square(),
            "base": (final[0] - self.robot.xyz[0]).square().mean()
            + 0.5 * (final[0] - self.robot.xyz[0]).square().sum(dim=-1).amax(),
            "generator": controls.square().mean(),
            "material_smoothness": material_smoothness,
            "stage_smoothness": (controls[1:] - controls[:-1]).square().mean(),
        }
        loss = sum(components[name] * self.config.weights[name] for name in components)
        metrics = geometry_metrics(final, self.robot, self.scene)
        metrics["loss"] = loss
        return FitState(loss=loss, final_coordinates=final, metrics=metrics)


def cross_section_rigidity_error(coordinates: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Maximum relative intra-section distance error; never used as a loss."""
    reference_distances = torch.cdist(reference, reference)
    observed = torch.cdist(coordinates, coordinates)
    mask = reference_distances > 10.0 * torch.finfo(reference.dtype).eps
    relative = torch.abs(observed - reference_distances) / reference_distances.clamp_min(
        10.0 * torch.finfo(reference.dtype).eps
    )
    return relative[mask].amax()


def verify_final(
    field_model: InvertibleBivectorField,
    robot: RobotSamples,
    scene: Scene,
) -> tuple[dict[str, Any], torch.Tensor]:
    with torch.no_grad():
        final = field_model(robot.field_input)
        reconstructed = field_model.inverse(robot.field_input.with_coordinates(final))
        report = {name: float(value.item()) for name, value in geometry_metrics(final, robot, scene).items()}
        crossings, tangents, material_s, valid = current_gate_crossings(final[:, 0], robot.section_s, scene)
        report["gate_crossing_position_errors"] = torch.linalg.vector_norm(
            crossings - scene.gate_centers, dim=-1
        ).tolist()
        report["gate_crossing_orientation_errors"] = torch.acos(
            (tangents * scene.gate_normals).sum(dim=-1).clamp(-1.0, 1.0)
        ).tolist()
        report["gate_crossing_material_s"] = material_s.tolist()
        report["gate_forward_crossings"] = valid.tolist()
        report["cross_section_rigidity_error"] = float(cross_section_rigidity_error(final, robot.xyz).item())
        report["inverse_reconstruction_error"] = float((reconstructed - robot.xyz).abs().max().item())
    return report, final


def coarse_dense_shape_error(
    coarse: RobotSamples,
    coarse_final: torch.Tensor,
    dense: RobotSamples,
    dense_final: torch.Tensor,
) -> float:
    predicted = _linear_interp(coarse.section_s, coarse_final[:, 0], dense.section_s)
    return float(torch.linalg.vector_norm(predicted - dense_final[:, 0], dim=-1).max().item())


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _orthogonal_ring_axes(normal: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = _to_numpy(_normalize(normal))
    seed = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.85 else np.array([0.0, 1.0, 0.0])
    axis_u = np.cross(n, seed)
    axis_u /= np.linalg.norm(axis_u)
    return n, axis_u, np.cross(n, axis_u)


def torus_mesh(ring: RingObstacle, major_samples: int = 36, tube_samples: int = 8) -> tuple[np.ndarray, ...]:
    normal, axis_u, axis_v = _orthogonal_ring_axes(ring.normal)
    major_angle = np.linspace(0.0, 2.0 * np.pi, major_samples)
    tube_angle = np.linspace(0.0, 2.0 * np.pi, tube_samples)
    uu, vv = np.meshgrid(major_angle, tube_angle, indexing="ij")
    radial = np.cos(uu)[..., None] * axis_u + np.sin(uu)[..., None] * axis_v
    points = (
        _to_numpy(ring.center)
        + (ring.major_radius + ring.tube_radius * np.cos(vv))[..., None] * radial
        + ring.tube_radius * np.sin(vv)[..., None] * normal
    )
    return points[..., 0], points[..., 1], points[..., 2]


def robot_wire(coordinates: torch.Tensor, max_sections: int = 60) -> np.ndarray:
    values = _to_numpy(coordinates)
    pieces = [values[:, 0], np.full((1, 3), np.nan), values[:, 1]]
    stride = max(1, values.shape[0] // max_sections)
    for index in range(0, values.shape[0], stride):
        ring = np.concatenate((values[index, 1:], values[index, 1:2]), axis=0)
        pieces.extend((np.full((1, 3), np.nan), ring))
    return np.concatenate(pieces)


def _set_line3d(line: Any, values: np.ndarray) -> None:
    line.set_data_3d(values[:, 0], values[:, 1], values[:, 2])


class LiveView:
    """One persistent 3-D view with cheap centerline updates during fitting."""

    def __init__(self, scene: Scene, robot: RobotSamples):
        import matplotlib.pyplot as plt

        self.plt = plt
        self._last_event_time = 0.0
        self.closed = False
        self.figure = plt.figure(figsize=(11.5, 7.4))
        self.interactive = self.figure.canvas.required_interactive_framework is not None
        if self.figure.canvas.manager is not None:
            self.figure.canvas.manager.set_window_title("Sparse-Constraint Continuum Threading")
        self.axis = self.figure.add_subplot(111, projection="3d")
        self.figure.canvas.mpl_connect("close_event", self._mark_closed)
        for ring in scene.rings:
            x, y, z = torus_mesh(ring)
            self.axis.plot_surface(x, y, z, color=ring.color, alpha=0.68, linewidth=0.1, shade=True)
        self.axis.plot(
            *_to_numpy(robot.xyz[:, 0]).T,
            color="#778899",
            alpha=0.48,
            linestyle="--",
            linewidth=1.4,
            label="straight initial rod",
        )
        self.current_line = self.axis.plot([], [], [], color="#173f5f", linewidth=2.8, label="current configuration")[0]
        self.axis.scatter([0.0], [0.0], [0.0], s=48, color="black", marker="s", label="base")
        self.axis.scatter(
            *_to_numpy(scene.gate_centers).T,
            color=[ring.color for ring in scene.rings],
            s=28,
            depthshade=False,
            label="gate constraints",
        )
        frame_colors = ("#e63946", "#2a9d8f", "#457b9d")
        origin = _to_numpy(scene.tip_position)
        for index, (direction, color) in enumerate(zip(_to_numpy(scene.tip_frame), frame_colors)):
            endpoint = origin + 0.075 * direction
            self.axis.plot(
                *np.stack((origin, endpoint)).T,
                color=color,
                linewidth=2.1,
                label="tip pose" if index == 0 else None,
            )

        self.axis.set_xlabel("x")
        self.axis.set_ylabel("y")
        self.axis.set_zlabel("z")
        self.axis.view_init(elev=23, azim=-105)
        self.axis.set_xlim(-0.06, max(scene.robot_length, 1.15) + 0.04)
        self.axis.set_ylim(-0.42, 0.42)
        self.axis.set_zlim(-0.17, 0.62)
        self.axis.set_box_aspect((1.55, 0.85, 0.8))
        self.axis.legend(loc="upper left", fontsize=8)
        self.figure.suptitle(
            "Three gates + one tip pose → a full continuum configuration",
            fontsize=15,
        )
        self.figure.text(
            0.5,
            0.925,
            "Material-space SE(3) field · no target-curve supervision",
            ha="center",
            color="#4b5563",
            fontsize=10,
        )
        self.status = self.figure.text(
            0.5,
            0.018,
            "initializing",
            ha="center",
            family="monospace",
            fontsize=9,
        )
        self.figure.subplots_adjust(left=0.03, right=0.97, bottom=0.075, top=0.89)
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

    def update(self, step: int | str, state: FitState) -> None:
        if self.closed:
            return
        _set_line3d(self.current_line, _to_numpy(state.final_coordinates[:, 0]))
        metrics = state.metrics
        label = f"step {step:4d}" if isinstance(step, int) else str(step)
        self.status.set_text(
            f"{label} · loss {metrics['loss'].item():.2e} · "
            f"gate {metrics['gate_offset'].item():.4f} · tip {metrics['tip_position_error'].item():.4f} · "
            f"orientation {math.degrees(metrics['tip_orientation_error'].item()):.1f}° · "
            f"clearance {metrics['minimum_clearance'].item():+.4f}"
        )
        try:
            self.figure.canvas.draw_idle()
        except (RuntimeError, SystemError):
            self.closed = True

    def _mark_closed(self, _event: Any) -> None:
        self.closed = True

    def keep_open(self) -> None:
        if self.interactive and not self.closed:
            self.plt.show(block=True)


def _progress_line(step: int, steps: int, state: FitState) -> str:
    metrics = state.metrics
    return (
        f"[{step:4d}/{steps}] loss={metrics['loss'].item():.3e} "
        f"gate={metrics['gate_offset'].item():.4f} tip={metrics['tip_position_error'].item():.4f} "
        f"orient={math.degrees(metrics['tip_orientation_error'].item()):.1f}° "
        f"clear={metrics['minimum_clearance'].item():+.4f} "
        f"strain={metrics['maximum_axial_strain'].item():.3f}"
    )


def optimize_field(
    field_model: InvertibleBivectorField,
    objective: SparseThreadingObjective,
    config: Config,
    initial_state: FitState,
    view: LiveView | None = None,
) -> OptimizationResult:
    optimizer = torch.optim.Adam(field_model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.optimization_steps,
        eta_min=config.learning_rate * config.minimum_learning_rate_fraction,
    )
    state = initial_state
    print(_progress_line(0, config.optimization_steps, initial_state))
    if view is not None:
        view.update(0, state)
    history = [float(state.loss)]
    best_step = 0
    best_loss = history[0]
    best_parameters = field_model.latent_coordinates.detach().clone()

    for step in range(1, config.optimization_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        state = objective(field_model)
        if not torch.isfinite(state.loss):
            raise RuntimeError(f"non-finite objective at optimization step {step}")
        state.loss.backward()
        torch.nn.utils.clip_grad_norm_(field_model.parameters(), config.gradient_clip)
        optimizer.step()
        scheduler.step()

        if view is not None:
            view.pump_events()
        should_report = step % config.log_every == 0 or step == config.optimization_steps
        should_draw = view is not None and (step % config.live_every == 0 or step == config.optimization_steps)
        if should_report or should_draw:
            with torch.no_grad():
                state = objective(field_model)
            value = float(state.loss)
            if should_report:
                print(_progress_line(step, config.optimization_steps, state))
            if should_draw:
                assert view is not None
                view.update(step, state)

        if not (should_report or should_draw):
            with torch.no_grad():
                state = objective(field_model)
            value = float(state.loss)
        history.append(value)
        if value < best_loss:
            best_step = step
            best_loss = value
            best_parameters.copy_(field_model.latent_coordinates.detach())

    with torch.no_grad():
        final_history_loss = history[-1]
        field_model.latent_coordinates.copy_(best_parameters)
    return OptimizationResult(history, best_step, best_loss, final_history_loss)


def acceptance_checks(report: dict[str, Any]) -> dict[str, bool]:
    coarse = report["primary"]
    dense = report["validation"]["dense"]
    transfer = report["validation"]["dense_transfer"]
    numerical = 1e-12
    return {
        "sparse threading": (
            all(coarse["gate_forward_crossings"])
            and all(dense["gate_forward_crossings"])
            and coarse["gate_offset"] < 0.035
            and dense["gate_offset"] < 0.035
            and coarse["gate_orientation_error"] < 0.35
            and dense["gate_orientation_error"] < 0.35
            and coarse["tip_position_error"] < 0.035
            and dense["tip_position_error"] < 0.035
            and coarse["tip_orientation_error"] < 0.25
            and dense["tip_orientation_error"] < 0.25
        ),
        "sampled clearance and geometric validity": (
            coarse["minimum_clearance"] >= 0.0
            and dense["minimum_clearance"] >= 0.0
            and coarse["maximum_axial_strain"] < 0.11
            and dense["maximum_axial_strain"] < 0.11
            and coarse["maximum_curvature"] < 30.0
            and dense["maximum_curvature"] < 30.0
            and coarse["base_drift"] < 0.012
            and dense["base_drift"] < 0.012
        ),
        "analytic inverse and rigidity": (
            coarse["inverse_reconstruction_error"] < numerical
            and dense["inverse_reconstruction_error"] < numerical
            and coarse["cross_section_rigidity_error"] < numerical
            and dense["cross_section_rigidity_error"] < numerical
        ),
        "zero-shot dense transfer": transfer["coarse_dense_centerline_discrepancy"] < 0.03,
    }


def print_report(report: dict[str, Any]) -> None:
    coarse = report["primary"]
    validation = report["validation"]
    dense = validation["dense"]
    transfer = validation["dense_transfer"]
    optimization = report["optimization"]
    print("\nSPARSE CONTINUUM THREADING\n")
    print(
        f"primary      gate {coarse['gate_offset']:.4f} m | tip {coarse['tip_position_error']:.4f} m | "
        f"clearance {coarse['minimum_clearance']:+.4f} m"
    )
    print(
        f"validation   {validation['optimization_sample_count']}→{validation['dense_sample_count']} points | "
        f"gate {dense['gate_offset']:.4f} m | transfer Δ {transfer['coarse_dense_centerline_discrepancy']:.4f} m"
    )
    print("gate s        " + ", ".join(f"{value:.4f}" for value in coarse["gate_crossing_material_s"]))
    print(
        f"numerical    inverse {dense['inverse_reconstruction_error']:.1e} | "
        f"rigidity {dense['cross_section_rigidity_error']:.1e} | strain {dense['maximum_axial_strain']:.3f}"
    )
    print(
        f"optimization best {optimization['best_loss']:.3e} at {optimization['best_step']} | "
        f"final {optimization['final_history_loss']:.3e}\n"
    )
    for name, passed in report["checks"].items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")


def save_result(config: Config, scene: Scene, dense_final: torch.Tensor) -> Path:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(14.8, 7.2))
    FigureCanvasAgg(figure)
    geometry_axis = figure.add_axes((0.025, 0.015, 0.95, 0.86), projection="3d")
    final_wire = robot_wire(dense_final, max_sections=72)
    geometry_axis.plot(*final_wire.T, color="#174a6e", linewidth=1.35, alpha=0.98, label="dense configuration")
    for ring in scene.rings:
        x, y, z = torus_mesh(ring, major_samples=72, tube_samples=14)
        geometry_axis.plot_surface(
            x,
            y,
            z,
            color=ring.color,
            alpha=0.88,
            linewidth=0.12,
            edgecolor=(0.18, 0.18, 0.18, 0.18),
            shade=True,
            antialiased=True,
        )
    geometry_axis.scatter([0.0], [0.0], [0.0], s=45, color="#111827", marker="s", depthshade=True)
    geometry_axis.set_xlabel("x [m]")
    geometry_axis.set_ylabel("y [m]")
    geometry_axis.set_zlabel("z [m]")
    geometry_axis.set_xlim(-0.05, 1.18)
    geometry_axis.set_ylim(-0.34, 0.34)
    geometry_axis.set_zlim(-0.10, 0.57)
    geometry_axis.set_box_aspect((1.55, 0.92, 0.9))
    geometry_axis.view_init(elev=19, azim=-111)
    geometry_axis.set_proj_type("persp", focal_length=0.82)
    geometry_axis.legend(loc="upper left", fontsize=9, frameon=False)
    figure.suptitle("Sparse-constraint continuum threading", fontsize=17, fontweight="bold", y=0.985)
    figure.text(
        0.5,
        0.925,
        "Zero-shot dense material-field evaluation through three solid gates",
        ha="center",
        color="#4b5563",
        fontsize=10,
    )
    output_path = config.output_dir / "result.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=185)
    return output_path


def save_diagnostics(config: Config, optimization: OptimizationResult, report: dict[str, Any]) -> Path:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(14.8, 4.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    history_axis, transfer_axis, integrity_axis = axes
    history = np.asarray(optimization.history)
    history_axis.semilogy(history, color="#264653", linewidth=2.0)
    history_axis.scatter(
        [optimization.best_step], [optimization.best_loss], color="#e76f51", s=26, zorder=3, label="restored best"
    )
    history_axis.set_title("Optimization", fontweight="bold")
    history_axis.set_xlabel("Adam step")
    history_axis.set_ylabel("objective")
    history_axis.grid(alpha=0.22)
    history_axis.legend(fontsize=8, frameon=False)

    dense = report["validation"]["dense"]
    transfer = report["validation"]["dense_transfer"]
    labels = ("gate", "tip", "coarse→dense", "base")
    values_mm = 1000.0 * np.array(
        (
            dense["gate_offset"],
            dense["tip_position_error"],
            transfer["coarse_dense_centerline_discrepancy"],
            dense["base_drift"],
        )
    )
    transfer_axis.bar(labels, values_mm, color=("#e76f51", "#2a9d8f", "#8f6ccf", "#94a3b8"))
    transfer_axis.set_title("Dense spatial errors", fontweight="bold")
    transfer_axis.set_ylabel("distance [mm]")
    transfer_axis.tick_params(axis="x", rotation=18)
    transfer_axis.grid(axis="y", alpha=0.22)

    normalized = np.array(
        (
            dense["gate_orientation_error"] / 0.35,
            dense["tip_orientation_error"] / 0.25,
            dense["maximum_axial_strain"] / 0.11,
            dense["maximum_curvature"] / 30.0,
            dense["base_drift"] / 0.012,
        )
    )
    integrity_axis.barh(
        ("gate angle", "tip angle", "strain", "curvature", "base drift"),
        normalized,
        color="#35a7a0",
    )
    integrity_axis.axvline(1.0, color="#111827", linestyle="--", linewidth=1.3, label="acceptance limit")
    integrity_axis.set_title("Dense acceptance margins", fontweight="bold")
    integrity_axis.set_xlabel("fraction of limit")
    integrity_axis.grid(axis="x", alpha=0.22)
    integrity_axis.legend(fontsize=8, frameon=False)

    figure.suptitle("Continuum threading diagnostics", fontsize=16, fontweight="bold")
    figure.text(
        0.5,
        0.015,
        f"{report['validation']['optimization_sample_count']}→{report['validation']['dense_sample_count']} points · "
        "no retraining · analytic inverse · rigid cross-sections",
        ha="center",
        color="#4b5563",
        fontsize=9,
    )
    figure.get_layout_engine().set(rect=(0.0, 0.06, 1.0, 0.92))
    output_path = config.output_dir / "diagnostics.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=185)
    return output_path


def run(config: Config = Config()) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    device = torch.device("cpu")
    dtype = torch.float64

    scene = build_scene(config, device=device, dtype=dtype)
    robot = build_robot(config.optimization_sections, config.surface_samples, scene, config)
    field_model = build_field(config, device=device, dtype=dtype)
    objective = SparseThreadingObjective(config, robot, scene)
    with torch.no_grad():
        initial_state = objective(field_model)
    print(
        f"setup         3 gates + 1 tip pose | {config.rbf_controls} RBF controls | "
        f"{robot.point_count} optimization points | seed {config.seed}"
    )
    view = LiveView(scene, robot) if config.live else None
    optimization = optimize_field(field_model, objective, config, initial_state, view)
    with torch.no_grad():
        restored_state = objective(field_model)
    if view is not None:
        view.update("restored best", restored_state)
    coarse_report, coarse_final = verify_final(field_model, robot, scene)

    # The learned material field is evaluated at new s values and denser
    # cross-sections directly. No field parameter or optimizer is touched.
    dense_robot = build_robot(config.dense_sections, config.dense_surface_samples, scene, config)
    dense_report, dense_final = verify_final(field_model, dense_robot, scene)
    transfer = {
        "coarse_dense_centerline_discrepancy": coarse_dense_shape_error(robot, coarse_final, dense_robot, dense_final),
    }
    report: dict[str, Any] = {
        "experiment": "sparse_continuum_threading",
        "algebra": "Cl(4,1)",
        "signature": "++++-",
        "problem": {
            "constraints": "three gate-center/alignment targets evaluated at predicted crossings, plus one tip pose",
            "field": "material-coordinate RBF SE(3) generator field",
            "robot_length": config.robot_length,
            "optimization_sections": config.optimization_sections,
            "optimization_surface_samples": config.surface_samples,
            "target_centerline_supplied": False,
            "material_to_gate_correspondence_supplied": False,
            "material_grid": "uniform",
            "gate_crossing_strategy": "first forward predicted centerline crossing; detached segment selection and differentiable interpolation",
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
        "primary": coarse_report,
        "validation": {
            "kind": "zero-shot dense material and cross-section resampling",
            "uses_optimizer": False,
            "optimization_sample_count": robot.point_count,
            "dense_sample_count": dense_robot.point_count,
            "dense": dense_report,
            "dense_transfer": transfer,
        },
        "limitations": [
            "The model is kinematic; strain and curvature are geometric penalties and diagnostics, not constitutive mechanics.",
            "Clearance and curvature are sampled on centerline sections, not proven over every continuous segment.",
            "The sparse gate and tip poses are prescribed exactly in a synthetic scene.",
        ],
    }
    report["checks"] = acceptance_checks(report)
    report_path = config.output_dir / "report.json"
    result_path = save_result(config, scene, dense_final)
    diagnostics_path = save_diagnostics(config, optimization, report)
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
    parser.add_argument("--live", action="store_true", help="show an interactive optimization view")
    args = parser.parse_args()
    return replace(Config(), optimization_steps=args.steps, output_dir=args.output_dir, live=args.live)


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
