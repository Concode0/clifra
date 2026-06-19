# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Invertible bivector fields over direct coordinate tensors."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from clifra.core.foundation.manifold import MANIFOLD_SPIN, tag_manifold
from clifra.core.foundation.module import AlgebraLike, CliffordModule
from clifra.core.foundation.numerics import signed_clamp_min

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
                f"projective coordinate_dim={d} requires at least {d} non-null basis vectors, "
                f"got p+q={non_null}"
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
    """Parameterized continuum deformation field built from invertible rotor paths.

    The field accepts coordinate tensors directly. A coordinate tensor with shape
    ``[*batch, *grid, coordinate_dim]`` is embedded as grade-1 values, deformed
    by a sequence of exponentiated bivectors, and extracted back to coordinates.

    ``control_shape`` determines how local the deformation is:
    - ``None``: one global bivector path is broadcast to every coordinate.
    - ``(m, n, ...)``: a control lattice of bivectors is interpolated to the
      incoming grid resolution, then broadcast across leading batch axes.
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
    ):
        super().__init__(algebra)
        if algebra.n < 2:
            raise ValueError("InvertibleBivectorField requires an algebra with at least two basis vectors")
        self.coordinate_dim = _positive_int(coordinate_dim, "coordinate_dim")
        self.path_steps = _positive_int(path_steps, "path_steps")
        self.control_shape = None if control_shape is None else tuple(_positive_int(v, "control_shape") for v in control_shape)
        self.projective = bool(projective)
        self.chart = (
            CoordinateChart.projective(algebra, self.coordinate_dim)
            if self.projective
            else CoordinateChart.direct(algebra, self.coordinate_dim)
        )
        self.vector_layout = self.chart.layout
        self.bivector_layout = algebra.layout((2,))
        self.num_bivectors = self.bivector_layout.dim
        self.action = algebra.plan_versor_action(
            grade=2,
            input_layout=self.vector_layout,
            output_layout=self.vector_layout,
            parameter_layout=self.bivector_layout,
        )

        parameter_shape = (self.path_steps, self.num_bivectors)
        if self.control_shape is not None:
            parameter_shape = (self.path_steps, *self.control_shape, self.num_bivectors)
        self.bivectors = nn.Parameter(torch.empty(parameter_shape, device=algebra.device, dtype=algebra.dtype))
        tag_manifold(self.bivectors, MANIFOLD_SPIN)
        nn.init.normal_(self.bivectors, mean=0.0, std=float(init_scale))

    def forward(self, coordinates: torch.Tensor, *, return_state: bool = False):
        """Deform coordinates and optionally return the full continuum state."""
        state = self.state(coordinates)
        return state if return_state else state.deformed_coordinates

    def state(self, coordinates: torch.Tensor) -> ContinuumState:
        """Return a full deformation state for direct coordinate input."""
        self._check_coordinates(coordinates)
        reference_mv = self.chart.embed(coordinates)
        deformed_mv, weights, spatial_shape, batch_shape = self._apply_path(reference_mv, inverse=False)
        return ContinuumState(
            reference_coordinates=coordinates,
            deformed_coordinates=self.chart.extract(deformed_mv),
            reference_multivectors=reference_mv,
            deformed_multivectors=deformed_mv,
            bivector_weights=weights,
            spatial_shape=spatial_shape,
            batch_shape=batch_shape,
        )

    def inverse(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Apply the reverse rotor path to deformed coordinates."""
        self._check_coordinates(coordinates)
        values = self.chart.embed(coordinates)
        reconstructed, _, _, _ = self._apply_path(values, inverse=True)
        return self.chart.extract(reconstructed)

    def inverse_state(self, coordinates: torch.Tensor) -> ContinuumState:
        """Return state metadata for the inverse path."""
        self._check_coordinates(coordinates)
        reference_mv = self.chart.embed(coordinates)
        inverse_mv, weights, spatial_shape, batch_shape = self._apply_path(reference_mv, inverse=True)
        return ContinuumState(
            reference_coordinates=coordinates,
            deformed_coordinates=self.chart.extract(inverse_mv),
            reference_multivectors=reference_mv,
            deformed_multivectors=inverse_mv,
            bivector_weights=weights,
            spatial_shape=spatial_shape,
            batch_shape=batch_shape,
        )

    def weights_for_shape(self, prefix_shape: Sequence[int], *, device=None, dtype=None) -> torch.Tensor:
        """Return bivector weights broadcast/interpolated to a coordinate prefix shape."""
        prefix_shape = tuple(int(v) for v in prefix_shape)
        weights, _, _ = self._weights_for_prefix(prefix_shape)
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
        """Return the explicit even-grade rotors for the current weights."""
        weights = self.weights_for_shape(prefix_shape)
        return self.algebra.exp(
            -0.5 * weights,
            input_layout=self.bivector_layout,
            output_layout=self.algebra.layout(range(0, self.algebra.n + 1, 2)),
        )

    def _apply_path(self, values: torch.Tensor, *, inverse: bool) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...]]:
        prefix_shape = tuple(values.shape[:-1])
        weights, spatial_shape, batch_shape = self._weights_for_prefix(prefix_shape)
        if weights.device != values.device or weights.dtype != values.dtype:
            weights = weights.to(device=values.device, dtype=values.dtype)

        flat = values.reshape(1, _numel(prefix_shape), self.vector_layout.dim)
        flat_weights = weights.reshape(self.path_steps, _numel(prefix_shape), self.num_bivectors)
        step_indices = range(self.path_steps - 1, -1, -1) if inverse else range(self.path_steps)
        for step in step_indices:
            step_weights = -flat_weights[step] if inverse else flat_weights[step]
            flat = self.action(flat, step_weights)
        output = flat.reshape(*prefix_shape, self.vector_layout.dim)
        return output, weights, spatial_shape, batch_shape

    def _weights_for_prefix(self, prefix_shape: tuple[int, ...]) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...]]:
        if self.control_shape is None:
            view_shape = (self.path_steps, *([1] * len(prefix_shape)), self.num_bivectors)
            weights = self.bivectors.reshape(view_shape).expand(self.path_steps, *prefix_shape, self.num_bivectors)
            return weights, prefix_shape, ()

        rank = len(self.control_shape)
        if len(prefix_shape) < rank:
            raise ValueError(
                f"coordinate prefix shape {prefix_shape} has fewer grid axes than control_shape={self.control_shape}"
            )
        batch_shape = prefix_shape[:-rank] if rank > 0 else prefix_shape
        spatial_shape = prefix_shape[-rank:] if rank > 0 else ()
        grid_weights = self._resized_control_weights(spatial_shape)
        if batch_shape:
            view_shape = (self.path_steps, *([1] * len(batch_shape)), *spatial_shape, self.num_bivectors)
            grid_weights = grid_weights.reshape(view_shape).expand(
                self.path_steps,
                *batch_shape,
                *spatial_shape,
                self.num_bivectors,
            )
        return grid_weights, spatial_shape, batch_shape

    def _resized_control_weights(self, spatial_shape: tuple[int, ...]) -> torch.Tensor:
        if spatial_shape == self.control_shape:
            return self.bivectors
        rank = len(self.control_shape or ())
        if rank == 0:
            return self.bivectors
        if rank > 3:
            raise ValueError("control_shape interpolation supports 1D, 2D, or 3D grids; use a matching grid above 3D")
        mode = {1: "linear", 2: "bilinear", 3: "trilinear"}[rank]
        order = (0, rank + 1, *range(1, rank + 1))
        source = self.bivectors.permute(order).reshape(1, self.path_steps * self.num_bivectors, *self.control_shape)
        resized = F.interpolate(source, size=spatial_shape, mode=mode, align_corners=True)
        resized = resized.reshape(self.path_steps, self.num_bivectors, *spatial_shape)
        return resized.permute(0, *range(2, rank + 2), 1).contiguous()

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


def _numel(shape: Sequence[int]) -> int:
    if not shape:
        return 1
    return int(reduce(mul, shape, 1))
