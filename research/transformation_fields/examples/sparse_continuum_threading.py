# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Sparse-Constraint Continuum Threading with Transformation Fields."""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.nn.functional as F

from clifra.core.algebra import AlgebraContext
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


@dataclass(frozen=True)
class RingObstacle:
    center: torch.Tensor
    normal: torch.Tensor
    major_radius: float
    tube_radius: float
    color: str


@dataclass(frozen=True)
class Scene:
    """Only sparse constraints survive scene construction."""

    rings: tuple[RingObstacle, ...]
    gate_s: torch.Tensor
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
    def sections(self) -> int:
        return int(self.xyz.shape[0])

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
    """Derive three gate poses, a tip pose, and rod length from a local sketch."""

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

    # This Hermite sketch is discarded here. It only assigns material gate
    # identities and total rod length; no sampled curve reaches optimization.
    pieces: list[torch.Tensor] = []
    for index in range(knots.shape[0] - 1):
        p0, p1 = knots[index], knots[index + 1]
        t = torch.linspace(0.0, 1.0, 321, device=device, dtype=dtype)
        chord = torch.linalg.vector_norm(p1 - p0)
        m0 = 0.72 * chord * directions[index]
        m1 = 0.72 * chord * directions[index + 1]
        t2, t3 = t.square(), t.pow(3)
        xyz = (
            (2.0 * t3 - 3.0 * t2 + 1.0)[:, None] * p0
            + (t3 - 2.0 * t2 + t)[:, None] * m0
            + (-2.0 * t3 + 3.0 * t2)[:, None] * p1
            + (t3 - t2)[:, None] * m1
        )
        pieces.append(xyz if index == 0 else xyz[1:])
    construction_curve = torch.cat(pieces)
    segment_length = torch.linalg.vector_norm(construction_curve[1:] - construction_curve[:-1], dim=-1)
    cumulative = torch.cat((segment_length.new_zeros(1), torch.cumsum(segment_length, dim=0)))
    robot_length = float(cumulative[-1].item())
    construction_s = cumulative / cumulative[-1]
    gate_s = torch.stack(
        [construction_s[torch.linalg.vector_norm(construction_curve - point, dim=-1).argmin()] for point in knots[1:-1]]
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
        gate_s=gate_s,
        gate_centers=knots[1:-1],
        gate_normals=directions[1:-1],
        tip_position=knots[-1],
        tip_frame=tip_frame,
        robot_length=robot_length,
    )


def _material_grid(section_count: int, gate_s: torch.Tensor) -> torch.Tensor:
    values = torch.linspace(0.0, 1.0, section_count, device=gate_s.device, dtype=gate_s.dtype)
    for gate in gate_s:
        values[torch.argmin(torch.abs(values - gate))] = gate
    return torch.sort(values).values


def build_robot(section_count: int, surface_count: int, scene: Scene, config: Config) -> RobotSamples:
    """Sample a cylinder; every point on a section shares one material s."""
    s = _material_grid(section_count, scene.gate_s)
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


def centerline_tangents(points: torch.Tensor) -> torch.Tensor:
    tangent = torch.empty_like(points)
    tangent[..., 0, :] = points[..., 1, :] - points[..., 0, :]
    tangent[..., -1, :] = points[..., -1, :] - points[..., -2, :]
    tangent[..., 1:-1, :] = points[..., 2:, :] - points[..., :-2, :]
    return _normalize(tangent)


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


def geometry_metrics(
    final: torch.Tensor,
    robot: RobotSamples,
    scene: Scene,
    gate_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    centers = final[..., 0, :]
    frames = section_frame(final)
    tip_dots = (frames[-1] * scene.tip_frame).sum(dim=-1).clamp(-1.0, 1.0)
    gate_tangent = centerline_tangents(centers)[gate_indices]
    gate_dot = (gate_tangent * scene.gate_normals).sum(dim=-1).clamp(-1.0, 1.0)
    strain = axial_strain(centers, robot)
    curvature = discrete_curvature(centers)
    return {
        "tip_position_error": torch.linalg.vector_norm(centers[-1] - scene.tip_position),
        "tip_orientation_error": torch.acos(tip_dots).amax(),
        "gate_offset": torch.linalg.vector_norm(centers[gate_indices] - scene.gate_centers, dim=-1).amax(),
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
        self.gate_indices = torch.stack([torch.argmin(torch.abs(robot.section_s - value)) for value in scene.gate_s])

    def __call__(self, field_model: InvertibleBivectorField) -> FitState:
        # Every point on one cross-section has the same s and therefore receives
        # the same composed SE(3) action. Nothing here fits individual vertices.
        final = field_model(self.robot.field_input)
        centers = final[..., 0, :]
        frames = section_frame(final)
        gate_centers = centers[self.gate_indices]
        gate_tangent = centerline_tangents(centers)[self.gate_indices]
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
        metrics = geometry_metrics(final, self.robot, self.scene, self.gate_indices)
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
) -> tuple[dict[str, float], torch.Tensor]:
    gate_indices = torch.stack([torch.argmin(torch.abs(robot.section_s - value)) for value in scene.gate_s])
    with torch.no_grad():
        final = field_model(robot.field_input)
        reconstructed = field_model.inverse(robot.field_input.with_coordinates(final))
        report = {
            name: float(value.item()) for name, value in geometry_metrics(final, robot, scene, gate_indices).items()
        }
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
        plt.ion()
        self.figure = plt.figure(figsize=(11.5, 7.4))
        self.figure.canvas.manager.set_window_title("Sparse-Constraint Continuum Threading")
        self.axis = self.figure.add_subplot(111, projection="3d")
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
        self.title = self.figure.suptitle(
            "Three gates + one tip pose → a full continuum configuration",
            fontsize=15,
        )
        self.subtitle = self.figure.text(
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
        plt.show(block=False)
        self.figure.canvas.draw_idle()
        self.pump_events()

    def pump_events(self) -> None:
        self.figure.canvas.flush_events()
        self.plt.pause(0.001)

    def update(self, step: int, state: FitState) -> None:
        _set_line3d(self.current_line, _to_numpy(state.final_coordinates[:, 0]))
        metrics = state.metrics
        self.status.set_text(
            f"step {step:4d} · loss {metrics['loss'].item():.2e} · "
            f"gate {metrics['gate_offset'].item():.4f} · tip {metrics['tip_position_error'].item():.4f} · "
            f"orientation {math.degrees(metrics['tip_orientation_error'].item()):.1f}° · "
            f"clearance {metrics['minimum_clearance'].item():+.4f}"
        )
        self.figure.canvas.draw_idle()

    def finalize(self, dense_final: torch.Tensor, report: dict[str, Any], path: Path) -> None:
        _set_line3d(self.current_line, robot_wire(dense_final))
        self.current_line.set_linewidth(1.35)
        self.current_line.set_label("dense field evaluation")
        self.axis.legend(loc="upper left", fontsize=8)
        self.title.set_text("Threaded configuration discovered from sparse constraints")
        self.subtitle.set_text("Zero-shot dense resampling · analytic inverse · structural cross-section rigidity")
        coarse, dense = report["coarse"], report["dense"]
        self.status.set_text(
            f"3 gates + 1 tip pose · clearance {coarse['minimum_clearance']:+.4f} coarse / "
            f"{dense['minimum_clearance']:+.4f} dense · {report['optimization_sample_count']} → "
            f"{report['dense_sample_count']} points · 0 retraining"
        )
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(path, dpi=190)

    def keep_open(self) -> None:
        self.plt.ioff()
        self.plt.show()


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
    view: LiveView,
) -> FitState:
    optimizer = torch.optim.Adam(field_model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.optimization_steps,
        eta_min=config.learning_rate * config.minimum_learning_rate_fraction,
    )
    with torch.no_grad():
        state = objective(field_model)
    print(_progress_line(0, config.optimization_steps, state))
    view.update(0, state)

    for step in range(1, config.optimization_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        state = objective(field_model)
        if not torch.isfinite(state.loss):
            raise RuntimeError(f"non-finite objective at optimization step {step}")
        state.loss.backward()
        torch.nn.utils.clip_grad_norm_(field_model.parameters(), config.gradient_clip)
        optimizer.step()
        scheduler.step()

        # GUI input is serviced every iteration; geometry changes less often.
        view.pump_events()
        should_report = step % config.log_every == 0 or step == config.optimization_steps
        should_draw = step % config.live_every == 0 or step == config.optimization_steps
        if should_report or should_draw:
            with torch.no_grad():
                state = objective(field_model)
            if should_report:
                print(_progress_line(step, config.optimization_steps, state))
            if should_draw:
                view.update(step, state)

    with torch.no_grad():
        return objective(field_model)


def acceptance_checks(report: dict[str, Any]) -> dict[str, bool]:
    coarse, dense = report["coarse"], report["dense"]
    numerical = 1e-12
    return {
        "sparse threading": (
            coarse["gate_offset"] < 0.035
            and dense["gate_offset"] < 0.035
            and coarse["gate_orientation_error"] < 0.35
            and dense["gate_orientation_error"] < 0.35
            and coarse["tip_position_error"] < 0.035
            and dense["tip_position_error"] < 0.035
            and coarse["tip_orientation_error"] < 0.25
            and dense["tip_orientation_error"] < 0.25
        ),
        "collision-free final configuration": (
            coarse["minimum_clearance"] >= 0.0 and dense["minimum_clearance"] >= 0.0
        ),
        "rod geometry": (
            coarse["maximum_axial_strain"] < 0.11
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
        "zero-shot dense transfer": (
            report["additional_optimization_steps"] == 0 and report["coarse_dense_centerline_discrepancy"] < 0.03
        ),
    }


def print_report(report: dict[str, Any]) -> None:
    coarse, dense = report["coarse"], report["dense"]
    print("\nSPARSE CONTINUUM THREADING\n")
    print("constraints    3 gates + 1 tip pose")
    print(
        f"fit            gate {coarse['gate_offset']:.4f} | tip {coarse['tip_position_error']:.4f} | "
        f"orientation {math.degrees(coarse['tip_orientation_error']):.1f}°"
    )
    print(
        f"dense fit      gate {dense['gate_offset']:.4f} | tip {dense['tip_position_error']:.4f} | "
        f"orientation {math.degrees(dense['tip_orientation_error']):.1f}°"
    )
    print(f"clearance      {coarse['minimum_clearance']:+.4f} coarse | {dense['minimum_clearance']:+.4f} dense")
    print(
        f"rod            strain {coarse['maximum_axial_strain']:.3f} | "
        f"curvature {coarse['maximum_curvature']:.2f} | base {coarse['base_drift']:.4f}"
    )
    print(
        f"structure      rigidity {dense['cross_section_rigidity_error']:.1e} | "
        f"inverse {dense['inverse_reconstruction_error']:.1e}"
    )
    print(
        f"resampling     {report['optimization_sample_count']} → {report['dense_sample_count']} points | "
        f"0 retraining | discrepancy {report['coarse_dense_centerline_discrepancy']:.4f}\n"
    )
    for name, passed in report["checks"].items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\nRESULT: {'SUCCESS' if all(report['checks'].values()) else 'FAILED CHECKS'}")


def run(config: Config) -> dict[str, Any]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cpu")
    dtype = torch.float64

    scene = build_scene(config, device=device, dtype=dtype)
    robot = build_robot(config.optimization_sections, config.surface_samples, scene, config)
    field_model = build_field(config, device=device, dtype=dtype)
    objective = SparseThreadingObjective(config, robot, scene)
    view = LiveView(scene, robot)

    print("optimizing a material-space SE(3) field from 3 gates + 1 tip pose")
    print(
        f"seed {config.seed} · init scale {config.init_scale:.0e} · {config.rbf_controls} RBF controls · "
        f"{config.path_steps} composition stages · {robot.point_count} optimization points"
    )
    optimize_field(field_model, objective, config, view)
    coarse_report, coarse_final = verify_final(field_model, robot, scene)

    # The learned material field is evaluated at new s values and denser
    # cross-sections directly. No field parameter or optimizer is touched.
    dense_robot = build_robot(config.dense_sections, config.dense_surface_samples, scene, config)
    dense_report, dense_final = verify_final(field_model, dense_robot, scene)
    report: dict[str, Any] = {
        "coarse": coarse_report,
        "dense": dense_report,
        "optimization_sample_count": robot.point_count,
        "dense_sample_count": dense_robot.point_count,
        "additional_optimization_steps": 0,
        "coarse_dense_centerline_discrepancy": coarse_dense_shape_error(robot, coarse_final, dense_robot, dense_final),
    }
    report["checks"] = acceptance_checks(report)
    result_path = config.output_dir / "result.png"
    view.finalize(dense_final, report, result_path)
    print_report(report)
    print(f"\nfigure: {result_path}")
    view.keep_open()
    return report


def main() -> None:
    run(Config())


if __name__ == "__main__":
    main()
