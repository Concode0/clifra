# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Invertible bivector fields over direct coordinate tensors."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Sequence

import torch
import torch.nn as nn

from clifra.core.foundation.manifold import MANIFOLD_SPIN, tag_manifold
from clifra.core.foundation.module import AlgebraLike, CliffordModule
from clifra.core.foundation.numerics import signed_clamp_min

from .inputs import CoordinateFieldInput, as_coordinate_field_input
from .sampling import (
    BroadcastGeneratorSampler,
    GeneratorFieldSample,
    GeneratorFieldSampler,
    RegularGridGeneratorSampler,
)
from .types import ContinuumState


@dataclass(frozen=True)
class CoordinateChart:
    """Embed and extract coordinate tensors through a grade-1 clifra layout."""

    algebra: AlgebraLike
    coordinate_dim: int
    layout: object
    coordinate_positions: tuple[int, ...]
    homogeneous_position: int | None = None

    @classmethod
    def direct(cls, algebra: AlgebraLike, coordinate_dim: int) -> "CoordinateChart":
        """Use the first ``coordinate_dim`` grade-1 basis vectors as coordinates."""
        d = _positive_int(coordinate_dim, "coordinate_dim")
        if d > algebra.n:
            raise ValueError(f"coordinate_dim={d} exceeds algebra basis dimension n={algebra.n}")
        layout = algebra.layout((1,))
        positions = _basis_positions(layout, tuple(1 << bit for bit in range(d)))
        return cls(algebra=algebra, coordinate_dim=d, layout=layout, coordinate_positions=positions)

    @classmethod
    def projective(cls, algebra: AlgebraLike, coordinate_dim: int) -> "CoordinateChart":
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
        coordinate_positions = _basis_positions(layout, tuple(1 << bit for bit in range(d)))
        homogeneous_position = _basis_positions(layout, (1 << non_null,))[0]
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
        algebra: AlgebraLike,
        coordinate_dim: int,
        *,
        path_steps: int = 1,
        control_shape: Sequence[int] | None = None,
        projective: bool = False,
        init_scale: float = 1e-3,
        generator_sampler: GeneratorFieldSampler | nn.Module | None = None,
        chart: CoordinateChart | None = None,
        action: nn.Module | None = None,
    ):
        super().__init__(algebra)
        if algebra.n < 2:
            raise ValueError("InvertibleBivectorField requires an algebra with at least two basis vectors")
        self.coordinate_dim = _positive_int(coordinate_dim, "coordinate_dim")
        self.path_steps = _positive_int(path_steps, "path_steps")
        if chart is not None:
            if chart.algebra.spec != algebra.spec:
                raise ValueError("chart and field algebra signatures must match")
            if chart.coordinate_dim != self.coordinate_dim:
                raise ValueError(
                    f"chart coordinate_dim={chart.coordinate_dim} does not match field coordinate_dim={self.coordinate_dim}"
                )
            if projective and chart.homogeneous_position is None:
                raise ValueError("projective=True requires a chart with a homogeneous coordinate")
            self.chart = chart
            self.projective = chart.homogeneous_position is not None
        else:
            self.projective = bool(projective)
            self.chart = (
                CoordinateChart.projective(algebra, self.coordinate_dim)
                if self.projective
                else CoordinateChart.direct(algebra, self.coordinate_dim)
            )
        self.vector_layout = self.chart.layout
        self.bivector_layout = algebra.layout((2,))
        self.num_bivectors = self.bivector_layout.dim
        if action is None:
            action = algebra.plan_versor_action(
                grade=2,
                input_layout=self.vector_layout,
                output_layout=self.vector_layout,
                parameter_layout=self.bivector_layout,
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
        parameter_shape = self.generator_sampler.parameter_shape(self.path_steps, self.num_bivectors)
        self.bivectors = nn.Parameter(torch.empty(parameter_shape, device=algebra.device, dtype=algebra.dtype))
        tag_manifold(self.bivectors, MANIFOLD_SPIN)
        nn.init.normal_(self.bivectors, mean=0.0, std=float(init_scale))

    def forward(
        self,
        coordinates: torch.Tensor | CoordinateFieldInput,
        *,
        sample_coordinates: torch.Tensor | None = None,
        spatial_shape: Sequence[int] | None = None,
        return_state: bool = False,
    ):
        """Deform coordinates and optionally return the full continuum state."""
        state = self.state(
            coordinates,
            sample_coordinates=sample_coordinates,
            spatial_shape=spatial_shape,
        )
        return state if return_state else state.deformed_coordinates

    def state(
        self,
        coordinates: torch.Tensor | CoordinateFieldInput,
        *,
        sample_coordinates: torch.Tensor | None = None,
        spatial_shape: Sequence[int] | None = None,
    ) -> ContinuumState:
        """Return a full deformation state for direct coordinate input."""
        field_input = as_coordinate_field_input(
            coordinates,
            sample_coordinates=sample_coordinates,
            spatial_shape=spatial_shape,
        )
        self._check_coordinates(field_input.coordinates)
        reference_mv = self.chart.embed(field_input.coordinates)
        deformed_mv, sampled = self._apply_path(reference_mv, field_input=field_input, inverse=False)
        return ContinuumState(
            reference_coordinates=field_input.coordinates,
            deformed_coordinates=self.chart.extract(deformed_mv),
            reference_multivectors=reference_mv,
            deformed_multivectors=deformed_mv,
            bivector_weights=sampled.weights,
            spatial_shape=sampled.spatial_shape,
            batch_shape=sampled.batch_shape,
            field_input=field_input.retain_sample_identity(),
        )

    def inverse(
        self,
        coordinates: torch.Tensor | CoordinateFieldInput,
        *,
        sample_coordinates: torch.Tensor | None = None,
        spatial_shape: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Apply the reverse rotor path using the supplied sample identity."""
        field_input = as_coordinate_field_input(
            coordinates,
            sample_coordinates=sample_coordinates,
            spatial_shape=spatial_shape,
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
        spatial_shape: Sequence[int] | None = None,
    ) -> ContinuumState:
        """Return state metadata for the inverse path."""
        field_input = as_coordinate_field_input(
            coordinates,
            sample_coordinates=sample_coordinates,
            spatial_shape=spatial_shape,
        )
        self._check_coordinates(field_input.coordinates)
        reference_mv = self.chart.embed(field_input.coordinates)
        inverse_mv, sampled = self._apply_path(reference_mv, field_input=field_input, inverse=True)
        return ContinuumState(
            reference_coordinates=field_input.coordinates,
            deformed_coordinates=self.chart.extract(inverse_mv),
            reference_multivectors=reference_mv,
            deformed_multivectors=inverse_mv,
            bivector_weights=sampled.weights,
            spatial_shape=sampled.spatial_shape,
            batch_shape=sampled.batch_shape,
            field_input=field_input.retain_sample_identity(),
        )

    def weights_for_input(
        self,
        coordinates: torch.Tensor | CoordinateFieldInput,
        *,
        sample_coordinates: torch.Tensor | None = None,
        spatial_shape: Sequence[int] | None = None,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        """Evaluate bivector weights for an explicit input domain."""
        field_input = as_coordinate_field_input(
            coordinates,
            sample_coordinates=sample_coordinates,
            spatial_shape=spatial_shape,
        )
        self._check_coordinates(field_input.coordinates)
        weights = self.generator_sampler.sample(self.bivectors, field_input).weights
        return weights.to(device=device, dtype=dtype) if device is not None or dtype is not None else weights

    def weights_for_shape(self, prefix_shape: Sequence[int], *, device=None, dtype=None) -> torch.Tensor:
        """Return weights for shape-only samplers retained by the legacy API.

        Coordinate-driven samplers must use :meth:`weights_for_input` because a
        shape alone does not identify their sampling positions.
        """
        prefix_shape = tuple(int(v) for v in prefix_shape)
        sample_shape = getattr(self.generator_sampler, "sample_shape", None)
        if not callable(sample_shape):
            raise ValueError("this generator sampler requires coordinates; use weights_for_input()")
        weights = sample_shape(self.bivectors, prefix_shape).weights
        if device is not None or dtype is not None:
            weights = weights.to(device=device, dtype=dtype)
        return weights

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
        spatial_shape: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Return explicit even-grade rotors evaluated on an input domain."""
        weights = self.weights_for_input(
            coordinates,
            sample_coordinates=sample_coordinates,
            spatial_shape=spatial_shape,
        )
        return self._rotors_from_weights(weights)

    def _rotors_from_weights(self, weights: torch.Tensor) -> torch.Tensor:
        return self.algebra.bivector_exp(
            -0.5 * weights,
            input_layout=self.bivector_layout,
            output_layout=self.algebra.layout(range(0, self.algebra.n + 1, 2)),
        )

    def _apply_path(
        self,
        values: torch.Tensor,
        *,
        field_input: CoordinateFieldInput,
        inverse: bool,
    ) -> tuple[torch.Tensor, GeneratorFieldSample]:
        prefix_shape = tuple(values.shape[:-1])
        sampled = self.generator_sampler.sample(self.bivectors, field_input)
        weights = sampled.weights
        expected_shape = (self.path_steps, *prefix_shape, self.num_bivectors)
        if tuple(weights.shape) != expected_shape:
            raise ValueError(f"sampled bivector weights must have shape {expected_shape}, got {tuple(weights.shape)}")
        if weights.device != values.device or weights.dtype != values.dtype:
            weights = weights.to(device=values.device, dtype=values.dtype)
            sampled = GeneratorFieldSample(
                weights=weights,
                spatial_shape=sampled.spatial_shape,
                batch_shape=sampled.batch_shape,
            )

        flat, flat_weights = self._execution_view(values, weights, sampled)
        step_indices = range(self.path_steps - 1, -1, -1) if inverse else range(self.path_steps)
        for step in step_indices:
            step_weights = -flat_weights[step] if inverse else flat_weights[step]
            flat = self.action(flat, step_weights)
        output = flat.reshape(*prefix_shape, self.vector_layout.dim)
        return output, sampled

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
                values.reshape(prod(sampled.batch_shape), prod(sampled.spatial_shape), self.vector_layout.dim),
                shared_weights.reshape(self.path_steps, prod(sampled.spatial_shape), self.num_bivectors),
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


def _basis_positions(layout, basis_indices: tuple[int, ...]) -> tuple[int, ...]:
    position_by_index = {index: position for position, index in enumerate(layout.basis_indices)}
    missing = [index for index in basis_indices if index not in position_by_index]
    if missing:
        raise ValueError(f"layout {layout.grades} does not contain basis indices {missing}")
    return tuple(position_by_index[index] for index in basis_indices)


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value
