# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Invertible bivector fields over direct coordinate tensors."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Sequence

import torch
import torch.nn as nn

from clifra.core.module import CliffordModule
from clifra.core.algebra import AlgebraContext
from clifra.core._kernel.numerics import signed_clamp_min

from .inputs import CoordinateFieldInput, as_coordinate_field_input
from .sampling import (
    BroadcastGeneratorSampler,
    GeneratorFieldSample,
    GeneratorFieldSampler,
    RegularGridGeneratorSampler,
)
from .types import TransformationRollout, TransformationState


@dataclass(frozen=True)
class CoordinateChart:
    """Embed and extract coordinate tensors through a grade-1 clifra layout."""

    algebra: AlgebraContext
    coordinate_dim: int
    layout: object
    coordinate_positions: tuple[int, ...]
    homogeneous_position: int | None = None

    @classmethod
    def direct(cls, algebra: AlgebraContext, coordinate_dim: int) -> "CoordinateChart":
        """Use the first ``coordinate_dim`` grade-1 basis vectors as coordinates."""
        d = _positive_int(coordinate_dim, "coordinate_dim")
        if d > algebra.n:
            raise ValueError(f"coordinate_dim={d} exceeds algebra basis dimension n={algebra.n}")
        layout = algebra.layout((1,))
        positions = tuple(layout.positions_for_basis(1 << bit for bit in range(d)).tolist())
        return cls(algebra=algebra, coordinate_dim=d, layout=layout, coordinate_positions=positions)

    @classmethod
    def conformal(cls, algebra: AlgebraContext, coordinate_dim: int) -> "ConformalChart":
        """Use a conformal null embedding as a Euclidean coordinate chart."""
        return ConformalChart(algebra, coordinate_dim)

    @classmethod
    def projective(cls, algebra: AlgebraContext, coordinate_dim: int) -> "CoordinateChart":
        """Use a PGA-style homogeneous grade-1 chart with the first null basis vector as e0."""
        d = _positive_int(coordinate_dim, "coordinate_dim")
        non_null = int(algebra.p) + int(algebra.q)
        if algebra.r < 1:
            raise ValueError(f"projective coordinates require at least one null basis vector, got r={algebra.r}")
        if d > non_null:
            raise ValueError(
                f"projective coordinate_dim={d} requires at least {d} non-null basis vectors, got p+q={non_null}"
            )
        layout = algebra.layout((1,))
        coordinate_positions = tuple(layout.positions_for_basis(1 << bit for bit in range(d)).tolist())
        homogeneous_position = layout.positions_for_basis((1 << non_null,)).item()
        return cls(
            algebra=algebra,
            coordinate_dim=d,
            layout=layout,
            coordinate_positions=coordinate_positions,
            homogeneous_position=homogeneous_position,
        )

    def embed(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Embed ``[..., coordinate_dim]`` coordinates into grade-1 compact lanes."""
        if coordinates.ndim < 1 or coordinates.shape[-1] != self.coordinate_dim:
            raise ValueError(
                f"coordinates last dimension must be {self.coordinate_dim}, got shape {tuple(coordinates.shape)}"
            )
        output = coordinates.new_zeros(*coordinates.shape[:-1], self.layout.dim)
        positions = self._coordinate_position_tensor(coordinates.device)
        index = positions.view(*((1,) * (coordinates.ndim - 1)), self.coordinate_dim)
        output.scatter_(-1, index.expand_as(coordinates).long(), coordinates)
        if self.homogeneous_position is not None:
            output[..., int(self.homogeneous_position)] = 1.0
        return output

    def extract(self, values: torch.Tensor) -> torch.Tensor:
        """Extract coordinate lanes from grade-1 active values."""
        if values.ndim < 1 or values.shape[-1] != self.layout.dim:
            raise ValueError(f"values last dimension must be {self.layout.dim}, got shape {tuple(values.shape)}")
        normalized = values
        if self.homogeneous_position is not None:
            pos = int(self.homogeneous_position)
            homogeneous = signed_clamp_min(values[..., pos : pos + 1], self.algebra.eps)
            normalized = values / homogeneous
        positions = self._coordinate_position_tensor(values.device)
        index = positions.view(*((1,) * (values.ndim - 1)), self.coordinate_dim)
        return torch.gather(normalized, -1, index.expand(*values.shape[:-1], self.coordinate_dim).long())

    def metric_signs(self, *, device=None, dtype=None) -> torch.Tensor:
        """Return metric signs for the exposed coordinate axes."""
        dtype = self.algebra.dtype if dtype is None else dtype
        signs = []
        for bit in range(self.coordinate_dim):
            if bit < self.algebra.p:
                signs.append(1.0)
            elif bit < self.algebra.p + self.algebra.q:
                signs.append(-1.0)
            else:
                signs.append(0.0)
        return torch.tensor(signs, device=device, dtype=dtype)

    def _coordinate_position_tensor(self, device) -> torch.Tensor:
        return torch.tensor(self.coordinate_positions, dtype=torch.long, device=device)


class ConformalChart(CliffordModule):
    """Euclidean chart using a compact grade-1 conformal embedding.

    Coordinates in ``R^d`` are lifted to null grade-1 points in
    ``Cl(d + 1, 1)``. Rotor paths can therefore represent conformal motions,
    including translations and dilations, while callers continue to work with
    ordinary Euclidean coordinate tensors.
    """

    homogeneous_position = None

    def __init__(self, algebra: AlgebraContext, coordinate_dim: int):
        super().__init__(algebra)
        self.coordinate_dim = _positive_int(coordinate_dim, "coordinate_dim")
        self.layout = algebra.layout((1,))
        d = self.coordinate_dim
        if algebra.p < d + 1 or algebra.q < 1:
            raise ValueError(
                f"Conformal embedding needs Cl(>={d + 1}, >=1), got Cl({algebra.p},{algebra.q},{algebra.r})"
            )
        positions = tuple(self.layout.positions_for_basis(1 << bit for bit in range(d)).tolist())
        self.register_buffer("_coordinate_positions", torch.tensor(positions, device=algebra.device))
        ep, em = self.layout.positions_for_basis((1 << d, 1 << algebra.p)).tolist()
        e_inf = torch.zeros(self.layout.dim, device=algebra.device, dtype=algebra.dtype)
        e_inf[ep] = e_inf[em] = 1.0
        e_o = torch.zeros_like(e_inf)
        e_o[ep], e_o[em] = -0.5, 0.5
        self.register_buffer("_e_inf", e_inf)
        self.register_buffer("_e_o", e_o)
        self.scalar_layout = algebra.layout((0,))

    def embed(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Lift Euclidean coordinates to conformal null points."""
        if coordinates.ndim < 1 or coordinates.shape[-1] != self.coordinate_dim:
            raise ValueError(
                f"coordinates last dimension must be {self.coordinate_dim}, got shape {tuple(coordinates.shape)}"
            )
        values = coordinates.new_zeros(*coordinates.shape[:-1], self.layout.dim)
        values.scatter_(-1, self._coordinate_positions.expand_as(coordinates), coordinates)
        return values + 0.5 * coordinates.square().sum(dim=-1, keepdim=True) * self._e_inf + self._e_o

    def extract(self, values: torch.Tensor) -> torch.Tensor:
        """Normalize conformal points and expose Euclidean coordinates."""
        if values.ndim < 1 or values.shape[-1] != self.layout.dim:
            raise ValueError(f"values last dimension must be {self.layout.dim}, got shape {tuple(values.shape)}")
        inner = self.algebra.geometric_product(
            values, self._e_inf.expand_as(values), left=self.layout, right=self.layout, output=self.scalar_layout
        )
        normalized = values / signed_clamp_min(-inner, self.algebra.eps)
        return normalized.index_select(-1, self._coordinate_positions)

    def metric_signs(self, *, device=None, dtype=None) -> torch.Tensor:
        """Return the positive Euclidean metric of the exposed coordinates."""
        dtype = self.algebra.dtype if dtype is None else dtype
        return torch.ones(self.coordinate_dim, device=device, dtype=dtype)


class GeneratorSubspace(nn.Module):
    """Fixed linear map from latent coordinates to bivector coefficients.

    ``mapping`` has shape ``[generator_dim, latent_dim]`` and realizes
    ``B = mapping @ z``. The leading dimensions of ``z`` are unrestricted.
    A selector subspace can be created with :meth:`from_lanes`.
    """

    def __init__(self, mapping: torch.Tensor):
        super().__init__()
        if not isinstance(mapping, torch.Tensor):
            mapping = torch.as_tensor(mapping, dtype=torch.get_default_dtype())
        if mapping.ndim != 2 or mapping.shape[0] < 1 or mapping.shape[1] < 1:
            raise ValueError("generator subspace mapping must have shape [generator_dim, latent_dim]")
        if not mapping.dtype.is_floating_point:
            mapping = mapping.to(dtype=torch.get_default_dtype())
        if not torch.isfinite(mapping).all():
            raise ValueError("generator subspace mapping must contain only finite values")
        self.register_buffer("mapping", mapping.detach().clone())

    @classmethod
    def from_lanes(
        cls,
        generator_dim: int,
        lanes: Sequence[int],
        *,
        device=None,
        dtype=None,
    ) -> "GeneratorSubspace":
        """Select compact bivector lane positions as latent generators."""
        generator_dim = _positive_int(generator_dim, "generator_dim")
        lane_tuple = tuple(int(lane) for lane in lanes)
        if not lane_tuple:
            raise ValueError("lanes must contain at least one generator lane")
        if len(set(lane_tuple)) != len(lane_tuple):
            raise ValueError("lanes must not contain duplicates")
        invalid = [lane for lane in lane_tuple if lane < 0 or lane >= generator_dim]
        if invalid:
            raise ValueError(f"generator lanes must be in [0, {generator_dim}), got {invalid}")
        dtype = torch.get_default_dtype() if dtype is None else dtype
        mapping = torch.zeros(generator_dim, len(lane_tuple), device=device, dtype=dtype)
        mapping[torch.tensor(lane_tuple, device=device), torch.arange(len(lane_tuple), device=device)] = 1.0
        return cls(mapping)

    @classmethod
    def from_basis_indices(
        cls, layout, basis_indices: Sequence[int], *, device=None, dtype=None
    ) -> "GeneratorSubspace":
        """Select generators by canonical bivector basis-blade indices."""
        positions = tuple(layout.positions_for_basis(int(index) for index in basis_indices).tolist())
        return cls.from_lanes(layout.dim, positions, device=device, dtype=dtype)

    @property
    def generator_dim(self) -> int:
        return int(self.mapping.shape[0])

    @property
    def latent_dim(self) -> int:
        return int(self.mapping.shape[1])

    def forward(self, latent_coordinates: torch.Tensor) -> torch.Tensor:
        """Map ``[..., latent_dim]`` coordinates into generator lanes."""
        if latent_coordinates.shape[-1] != self.latent_dim:
            raise ValueError(
                f"latent coordinate dimension must be {self.latent_dim}, got {latent_coordinates.shape[-1]}"
            )
        mapping = self.mapping.to(device=latent_coordinates.device, dtype=latent_coordinates.dtype)
        return torch.matmul(latent_coordinates, mapping.transpose(0, 1))

    def encode(self, generator_weights: torch.Tensor) -> torch.Tensor:
        """Return least-squares latent coordinates for generator weights."""
        if generator_weights.shape[-1] != self.generator_dim:
            raise ValueError(f"generator dimension must be {self.generator_dim}, got {generator_weights.shape[-1]}")
        mapping = self.mapping.to(device=generator_weights.device, dtype=generator_weights.dtype)
        return torch.matmul(generator_weights, torch.linalg.pinv(mapping).transpose(0, 1))


GeneratorSubspaceMap = GeneratorSubspace


class InvertibleBivectorField(CliffordModule):
    """Parameterized coordinate transformation built from invertible rotor paths.

    The field accepts tensors or :class:`CoordinateFieldInput` objects. Coordinate
    values are embedded as grade-1 multivectors, transformed by a sequence of
    exponentiated bivectors, and extracted back to coordinates. A pluggable
    ``generator_sampler`` decides how stored bivectors are evaluated over the
    input domain.

    The legacy ``control_shape`` convenience selects one of two samplers:
    - ``None``: one global bivector path is broadcast to every coordinate.
    - ``(m, n, ...)``: a control lattice of bivectors is interpolated to the
      incoming grid resolution, then broadcast across leading batch axes.

    Passing ``generator_sampler`` enables other input organizations, such as
    coordinate-driven RBF sampling for unordered points. ``control_shape`` and
    ``generator_sampler`` are mutually exclusive.
    """

    def __init__(
        self,
        algebra: AlgebraContext,
        coordinate_dim: int,
        *,
        path_steps: int = 1,
        control_shape: Sequence[int] | None = None,
        projective: bool = False,
        conformal: bool = False,
        init_scale: float = 1e-3,
        generator_sampler: GeneratorFieldSampler | nn.Module | None = None,
        generator_subspace: GeneratorSubspace | torch.Tensor | Sequence[int] | None = None,
        chart: CoordinateChart | ConformalChart | None = None,
        action: nn.Module | None = None,
    ):
        super().__init__(algebra)
        if algebra.n < 2:
            raise ValueError("InvertibleBivectorField requires an algebra with at least two basis vectors")
        self.coordinate_dim = _positive_int(coordinate_dim, "coordinate_dim")
        self.path_steps = _positive_int(path_steps, "path_steps")
        if projective and conformal:
            raise ValueError("projective and conformal charts are mutually exclusive")
        if chart is not None:
            if chart.algebra.spec != algebra.spec:
                raise ValueError("chart and field algebra signatures must match")
            if chart.coordinate_dim != self.coordinate_dim:
                raise ValueError(
                    f"chart coordinate_dim={chart.coordinate_dim} does not match field coordinate_dim={self.coordinate_dim}"
                )
            homogeneous_position = getattr(chart, "homogeneous_position", None)
            if projective and homogeneous_position is None:
                raise ValueError("projective=True requires a chart with a homogeneous coordinate")
            if conformal and not isinstance(chart, ConformalChart):
                raise ValueError("conformal=True requires a ConformalChart")
            self.chart = chart
            self.projective = homogeneous_position is not None
            self.conformal = isinstance(chart, ConformalChart)
        else:
            self.projective = bool(projective)
            self.conformal = bool(conformal)
            if self.projective:
                self.chart = CoordinateChart.projective(algebra, self.coordinate_dim)
            elif self.conformal:
                self.chart = CoordinateChart.conformal(algebra, self.coordinate_dim)
            else:
                self.chart = CoordinateChart.direct(algebra, self.coordinate_dim)
        self.vector_layout = self.chart.layout
        self.bivector_layout = algebra.layout((2,))
        self.num_bivectors = self.bivector_layout.dim
        self.generator_subspace = self._resolve_generator_subspace(generator_subspace)
        self.latent_dim = self.num_bivectors if self.generator_subspace is None else self.generator_subspace.latent_dim
        if action is None:
            action = algebra.plan_versor_action(
                grade=2, input=self.vector_layout, output=self.vector_layout, parameter=self.bivector_layout
            )
        if not isinstance(action, nn.Module):
            raise TypeError("action must be a torch.nn.Module implementing action(values, generator_weights)")
        self.action = action

        if generator_sampler is not None and control_shape is not None:
            raise ValueError("pass either control_shape or generator_sampler, not both")
        if generator_sampler is None:
            generator_sampler = (
                BroadcastGeneratorSampler() if control_shape is None else RegularGridGeneratorSampler(control_shape)
            )
        if not isinstance(generator_sampler, nn.Module):
            raise TypeError("generator_sampler must be a torch.nn.Module implementing the sampler contract")
        if not callable(getattr(generator_sampler, "parameter_shape", None)) or not callable(
            getattr(generator_sampler, "sample", None)
        ):
            raise TypeError("generator_sampler must define parameter_shape() and sample()")
        self.generator_sampler = generator_sampler.to(device=algebra.device, dtype=algebra.dtype)
        self.control_shape = (
            self.generator_sampler.control_shape
            if isinstance(self.generator_sampler, RegularGridGeneratorSampler)
            else None
        )
        parameter_shape = self.generator_sampler.parameter_shape(self.path_steps, self.latent_dim)
        self._latent_coordinates = nn.Parameter(
            torch.empty(parameter_shape, device=algebra.device, dtype=algebra.dtype)
        )
        nn.init.normal_(self._latent_coordinates, mean=0.0, std=float(init_scale))

    @property
    def latent_coordinates(self) -> nn.Parameter:
        """Return trainable coordinates in the declared generator subspace."""
        return self._latent_coordinates

    @property
    def bivectors(self) -> torch.Tensor:
        """Return stored latent parameters mapped into compact bivector lanes."""
        return self._map_generators(self._latent_coordinates)

    def forward(
        self,
        coordinates: torch.Tensor | CoordinateFieldInput,
        *,
        sample_coordinates: torch.Tensor | None = None,
        domain_shape: Sequence[int] | None = None,
        return_state: bool = False,
    ):
        """Transform coordinates and optionally return the full transformation state."""
        state = self.state(
            coordinates,
            sample_coordinates=sample_coordinates,
            domain_shape=domain_shape,
        )
        return state if return_state else state.transformed_coordinates

    def state(
        self,
        coordinates: torch.Tensor | CoordinateFieldInput,
        *,
        sample_coordinates: torch.Tensor | None = None,
        domain_shape: Sequence[int] | None = None,
    ) -> TransformationState:
        """Return the transformation state for the supplied coordinate input."""
        field_input = as_coordinate_field_input(
            coordinates,
            sample_coordinates=sample_coordinates,
            domain_shape=domain_shape,
        )
        self._check_coordinates(field_input.coordinates)
        input_mv = self.chart.embed(field_input.coordinates)
        transformed_mv, sampled = self._apply_path(input_mv, field_input=field_input, inverse=False)
        return TransformationState(
            input_coordinates=field_input.coordinates,
            transformed_coordinates=self.chart.extract(transformed_mv),
            input_multivectors=input_mv,
            transformed_multivectors=transformed_mv,
            generator_weights=sampled.weights,
            domain_shape=sampled.domain_shape,
            batch_shape=sampled.batch_shape,
            field_input=field_input.retain_sample_identity(),
            latent_coordinates=sampled.latent_coordinates,
        )

    def inverse(
        self,
        coordinates: torch.Tensor | CoordinateFieldInput,
        *,
        sample_coordinates: torch.Tensor | None = None,
        domain_shape: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Apply the reverse rotor path using the supplied sample identity."""
        field_input = as_coordinate_field_input(
            coordinates,
            sample_coordinates=sample_coordinates,
            domain_shape=domain_shape,
        )
        self._check_coordinates(field_input.coordinates)
        values = self.chart.embed(field_input.coordinates)
        reconstructed, _ = self._apply_path(values, field_input=field_input, inverse=True)
        return self.chart.extract(reconstructed)

    def inverse_state(
        self,
        coordinates: torch.Tensor | CoordinateFieldInput,
        *,
        sample_coordinates: torch.Tensor | None = None,
        domain_shape: Sequence[int] | None = None,
    ) -> TransformationState:
        """Return state metadata for the inverse path."""
        field_input = as_coordinate_field_input(
            coordinates,
            sample_coordinates=sample_coordinates,
            domain_shape=domain_shape,
        )
        self._check_coordinates(field_input.coordinates)
        input_mv = self.chart.embed(field_input.coordinates)
        inverse_mv, sampled = self._apply_path(input_mv, field_input=field_input, inverse=True)
        return TransformationState(
            input_coordinates=field_input.coordinates,
            transformed_coordinates=self.chart.extract(inverse_mv),
            input_multivectors=input_mv,
            transformed_multivectors=inverse_mv,
            generator_weights=sampled.weights,
            domain_shape=sampled.domain_shape,
            batch_shape=sampled.batch_shape,
            field_input=field_input.retain_sample_identity(),
            latent_coordinates=sampled.latent_coordinates,
        )

    def rollout(
        self,
        coordinates: torch.Tensor | CoordinateFieldInput,
        *,
        sample_coordinates: torch.Tensor | None = None,
        domain_shape: Sequence[int] | None = None,
        include_initial: bool = True,
        substeps_per_step: int = 1,
    ) -> TransformationRollout:
        """Expose every internal path state and its exact inverse retracing.

        ``substeps_per_step`` samples each stored generator ``B`` at
        ``alpha = 0, 1/m, ..., 1``. Every intermediate frame is evaluated from
        that step's starting state by the genuine ``exp(-alpha B / 2)`` action,
        rather than by coordinate interpolation or accumulated fractional
        actions. Sampling occurs once at the persistent input identity.

        When ``include_initial`` is false, the starting state is omitted from
        each directional trajectory. The default preserves the original
        ``path_steps + 1`` rollout contract.
        """
        substeps_per_step = _positive_int(substeps_per_step, "substeps_per_step")
        field_input = as_coordinate_field_input(
            coordinates,
            sample_coordinates=sample_coordinates,
            domain_shape=domain_shape,
        )
        self._check_coordinates(field_input.coordinates)
        persistent_input = field_input.retain_sample_identity()
        initial_mv = self.chart.embed(field_input.coordinates)
        sampled = self._sample_for_values(initial_mv, persistent_input)
        flat, flat_weights = self._execution_view(initial_mv, sampled.weights, sampled)
        prefix_shape = tuple(initial_mv.shape[:-1])

        forward_multivectors = [initial_mv]
        for step in range(self.path_steps):
            step_start = flat
            for substep in range(1, substeps_per_step + 1):
                alpha = float(substep) / float(substeps_per_step)
                flat = self.action(step_start, alpha * flat_weights[step])
                forward_multivectors.append(flat.reshape(*prefix_shape, self.vector_layout.dim))

        inverse_multivectors = [forward_multivectors[-1]]
        for step in range(self.path_steps - 1, -1, -1):
            step_start = flat
            for substep in range(1, substeps_per_step + 1):
                alpha = float(substep) / float(substeps_per_step)
                flat = self.action(step_start, -alpha * flat_weights[step])
                inverse_multivectors.append(flat.reshape(*prefix_shape, self.vector_layout.dim))

        forward_mv = torch.stack(forward_multivectors, dim=0)
        inverse_mv = torch.stack(inverse_multivectors, dim=0)
        if not include_initial:
            forward_mv = forward_mv[1:]
            inverse_mv = inverse_mv[1:]
        return TransformationRollout(
            forward_coordinates=self.chart.extract(forward_mv),
            inverse_coordinates=self.chart.extract(inverse_mv),
            forward_multivectors=forward_mv,
            inverse_multivectors=inverse_mv,
            generator_weights=sampled.weights,
            latent_coordinates=sampled.latent_coordinates,
            field_input=persistent_input,
            includes_initial=bool(include_initial),
            path_steps=self.path_steps,
            substeps_per_step=substeps_per_step,
        )

    def weights_for_input(
        self,
        coordinates: torch.Tensor | CoordinateFieldInput,
        *,
        sample_coordinates: torch.Tensor | None = None,
        domain_shape: Sequence[int] | None = None,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        """Evaluate bivector weights for an explicit input domain."""
        field_input = as_coordinate_field_input(
            coordinates,
            sample_coordinates=sample_coordinates,
            domain_shape=domain_shape,
        )
        self._check_coordinates(field_input.coordinates)
        weights = self._sample_generators(field_input).weights
        return weights.to(device=device, dtype=dtype) if device is not None or dtype is not None else weights

    def latent_for_input(
        self,
        coordinates: torch.Tensor | CoordinateFieldInput,
        *,
        sample_coordinates: torch.Tensor | None = None,
        domain_shape: Sequence[int] | None = None,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        """Evaluate sampled latent generator coordinates for an input domain."""
        field_input = as_coordinate_field_input(
            coordinates,
            sample_coordinates=sample_coordinates,
            domain_shape=domain_shape,
        )
        self._check_coordinates(field_input.coordinates)
        latent = self.generator_sampler.sample(self.latent_coordinates, field_input).weights
        return latent.to(device=device, dtype=dtype) if device is not None or dtype is not None else latent

    latent_coordinates_for_input = latent_for_input

    def weights_for_shape(self, prefix_shape: Sequence[int], *, device=None, dtype=None) -> torch.Tensor:
        """Return weights for shape-only samplers retained by the legacy API.

        Coordinate-driven samplers must use :meth:`weights_for_input` because a
        shape alone does not identify their sampling positions.
        """
        prefix_shape = tuple(int(v) for v in prefix_shape)
        sample_shape = getattr(self.generator_sampler, "sample_shape", None)
        if not callable(sample_shape):
            raise ValueError("this generator sampler requires coordinates; use weights_for_input()")
        latent = sample_shape(self.latent_coordinates, prefix_shape).weights
        weights = self._map_generators(latent)
        if device is not None or dtype is not None:
            weights = weights.to(device=device, dtype=dtype)
        return weights

    def latent_for_shape(self, prefix_shape: Sequence[int], *, device=None, dtype=None) -> torch.Tensor:
        """Return sampled latent coordinates for a shape-only sampler."""
        prefix_shape = tuple(int(v) for v in prefix_shape)
        sample_shape = getattr(self.generator_sampler, "sample_shape", None)
        if not callable(sample_shape):
            raise ValueError("this generator sampler requires coordinates; use latent_for_input()")
        latent = sample_shape(self.latent_coordinates, prefix_shape).weights
        return latent.to(device=device, dtype=dtype) if device is not None or dtype is not None else latent

    latent_coordinates_for_shape = latent_for_shape

    def mean_bivector(self) -> torch.Tensor:
        """Return the mean path bivector coefficients over steps and control sites."""
        if self.bivectors.ndim == 2:
            return self.bivectors.mean(dim=0)
        reduce_dims = tuple(range(self.bivectors.ndim - 1))
        return self.bivectors.mean(dim=reduce_dims)

    def rotor_path(self, prefix_shape: Sequence[int] = ()) -> torch.Tensor:
        """Return explicit rotors for a shape-only sampler."""
        weights = self.weights_for_shape(prefix_shape)
        return self._rotors_from_weights(weights)

    def rotors_for_input(
        self,
        coordinates: torch.Tensor | CoordinateFieldInput,
        *,
        sample_coordinates: torch.Tensor | None = None,
        domain_shape: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Return explicit even-grade rotors evaluated on an input domain."""
        weights = self.weights_for_input(
            coordinates,
            sample_coordinates=sample_coordinates,
            domain_shape=domain_shape,
        )
        return self._rotors_from_weights(weights)

    def _rotors_from_weights(self, weights: torch.Tensor) -> torch.Tensor:
        return self.algebra.bivector_exp(
            -0.5 * weights, input=self.bivector_layout, output=self.algebra.layout(range(0, self.algebra.n + 1, 2))
        )

    def _apply_path(
        self,
        values: torch.Tensor,
        *,
        field_input: CoordinateFieldInput,
        inverse: bool,
    ) -> tuple[torch.Tensor, GeneratorFieldSample]:
        prefix_shape = tuple(values.shape[:-1])
        sampled = self._sample_for_values(values, field_input)
        flat, flat_weights = self._execution_view(values, sampled.weights, sampled)
        step_indices = range(self.path_steps - 1, -1, -1) if inverse else range(self.path_steps)
        for step in step_indices:
            step_weights = -flat_weights[step] if inverse else flat_weights[step]
            flat = self.action(flat, step_weights)
        output = flat.reshape(*prefix_shape, self.vector_layout.dim)
        return output, sampled

    def _sample_generators(self, field_input: CoordinateFieldInput) -> GeneratorFieldSample:
        latent_sample = self.generator_sampler.sample(self.latent_coordinates, field_input)
        return GeneratorFieldSample(
            weights=self._map_generators(latent_sample.weights),
            domain_shape=latent_sample.domain_shape,
            batch_shape=latent_sample.batch_shape,
            latent_coordinates=latent_sample.weights,
        )

    def _sample_for_values(
        self,
        values: torch.Tensor,
        field_input: CoordinateFieldInput,
    ) -> GeneratorFieldSample:
        sampled = self._sample_generators(field_input)
        expected_shape = (self.path_steps, *values.shape[:-1], self.num_bivectors)
        if tuple(sampled.weights.shape) != expected_shape:
            raise ValueError(
                f"sampled bivector weights must have shape {expected_shape}, got {tuple(sampled.weights.shape)}"
            )
        if sampled.weights.device == values.device and sampled.weights.dtype == values.dtype:
            return sampled
        latent_coordinates = sampled.latent_coordinates
        if latent_coordinates is not None:
            latent_coordinates = latent_coordinates.to(device=values.device, dtype=values.dtype)
        return GeneratorFieldSample(
            weights=sampled.weights.to(device=values.device, dtype=values.dtype),
            domain_shape=sampled.domain_shape,
            batch_shape=sampled.batch_shape,
            latent_coordinates=latent_coordinates,
        )

    def _map_generators(self, latent_coordinates: torch.Tensor) -> torch.Tensor:
        if self.generator_subspace is None:
            return latent_coordinates
        return self.generator_subspace(latent_coordinates)

    def _resolve_generator_subspace(
        self,
        subspace: GeneratorSubspace | torch.Tensor | Sequence[int] | None,
    ) -> GeneratorSubspace | None:
        if subspace is None:
            return None
        if isinstance(subspace, GeneratorSubspace):
            resolved = subspace
        else:
            value = subspace if isinstance(subspace, torch.Tensor) else torch.as_tensor(subspace)
            if value.ndim == 1:
                if value.dtype.is_floating_point:
                    raise ValueError("one-dimensional generator_subspace values must be integer lane positions")
                resolved = GeneratorSubspace.from_lanes(
                    self.num_bivectors,
                    value.tolist(),
                    device=self.algebra.device,
                    dtype=self.algebra.dtype,
                )
            elif value.ndim == 2:
                # Accept both the standard [generator, latent] convention and
                # basis-row [latent, generator] tensors at this field boundary.
                if value.shape[0] == self.num_bivectors:
                    mapping = value
                elif value.shape[1] == self.num_bivectors:
                    mapping = value.transpose(0, 1)
                else:
                    raise ValueError(
                        "generator_subspace must have one axis equal to the compact bivector dimension "
                        f"{self.num_bivectors}, got {tuple(value.shape)}"
                    )
                resolved = GeneratorSubspace(mapping)
            else:
                raise ValueError("generator_subspace must be lane positions or a rank-2 mapping")
        if resolved.generator_dim != self.num_bivectors:
            raise ValueError(
                f"generator subspace output dimension must be {self.num_bivectors}, got {resolved.generator_dim}"
            )
        return resolved.to(device=self.algebra.device, dtype=self.algebra.dtype)

    def _execution_view(
        self,
        values: torch.Tensor,
        weights: torch.Tensor,
        sampled: GeneratorFieldSample,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Avoid exponentiating generators repeated only by broadcasting."""
        if isinstance(self.generator_sampler, BroadcastGeneratorSampler):
            prefix_rank = values.ndim - 1
            shared_weights = weights[(slice(None), *((0,) * prefix_rank))].unsqueeze(1)
            return values.reshape(-1, 1, self.vector_layout.dim), shared_weights

        if isinstance(self.generator_sampler, RegularGridGeneratorSampler) and sampled.batch_shape:
            batch_rank = len(sampled.batch_shape)
            shared_weights = weights[(slice(None), *((0,) * batch_rank))]
            return (
                values.reshape(prod(sampled.batch_shape), prod(sampled.domain_shape), self.vector_layout.dim),
                shared_weights.reshape(self.path_steps, prod(sampled.domain_shape), self.num_bivectors),
            )

        sample_count = values[..., 0].numel()
        return (
            values.reshape(1, sample_count, self.vector_layout.dim),
            weights.reshape(self.path_steps, sample_count, self.num_bivectors),
        )

    def _check_coordinates(self, coordinates: torch.Tensor) -> None:
        if coordinates.ndim < 1 or coordinates.shape[-1] != self.coordinate_dim:
            raise ValueError(
                f"coordinates must have shape [..., {self.coordinate_dim}], got {tuple(coordinates.shape)}"
            )


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value
