# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Sampling policies for fields of Clifford generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .inputs import CoordinateFieldInput


@dataclass(frozen=True)
class GeneratorFieldSample:
    """Generator weights evaluated at every input sample."""

    weights: torch.Tensor
    spatial_shape: tuple[int, ...]
    batch_shape: tuple[int, ...]


class GeneratorFieldSampler(Protocol):
    """Map stored generator parameters to an input domain."""

    def parameter_shape(self, path_steps: int, generator_dim: int) -> tuple[int, ...]:
        """Return the trainable parameter shape required by this sampler."""
        ...

    def sample(self, parameters: torch.Tensor, field_input: CoordinateFieldInput) -> GeneratorFieldSample:
        """Evaluate generator parameters for every input coordinate."""
        ...


class BroadcastGeneratorSampler(nn.Module):
    """Broadcast one generator per path step over every input sample."""

    def parameter_shape(self, path_steps: int, generator_dim: int) -> tuple[int, ...]:
        return int(path_steps), int(generator_dim)

    def sample(self, parameters: torch.Tensor, field_input: CoordinateFieldInput) -> GeneratorFieldSample:
        prefix_shape = field_input.prefix_shape
        _check_parameters(parameters, self.parameter_shape(parameters.shape[0], parameters.shape[-1]))
        view_shape = (parameters.shape[0], *([1] * len(prefix_shape)), parameters.shape[-1])
        weights = parameters.reshape(view_shape).expand(parameters.shape[0], *prefix_shape, parameters.shape[-1])
        spatial_shape, batch_shape = field_input.topology_shapes()
        return GeneratorFieldSample(weights=weights, spatial_shape=spatial_shape, batch_shape=batch_shape)

    def sample_shape(self, parameters: torch.Tensor, prefix_shape: Sequence[int]) -> GeneratorFieldSample:
        coordinates = parameters.new_zeros(*tuple(prefix_shape), 1)
        return self.sample(parameters, CoordinateFieldInput(coordinates))


class RegularGridGeneratorSampler(nn.Module):
    """Interpolate a 1D, 2D, or 3D control lattice over structured inputs."""

    def __init__(self, control_shape: Sequence[int]):
        super().__init__()
        self.control_shape = tuple(_positive_int(value, "control_shape") for value in control_shape)
        if not self.control_shape:
            raise ValueError("control_shape must contain at least one spatial axis")
        if len(self.control_shape) > 3:
            raise ValueError("regular-grid interpolation supports 1D, 2D, or 3D control lattices")

    def parameter_shape(self, path_steps: int, generator_dim: int) -> tuple[int, ...]:
        return int(path_steps), *self.control_shape, int(generator_dim)

    def sample(self, parameters: torch.Tensor, field_input: CoordinateFieldInput) -> GeneratorFieldSample:
        expected = self.parameter_shape(parameters.shape[0], parameters.shape[-1])
        _check_parameters(parameters, expected)
        spatial_shape, batch_shape = field_input.topology_shapes(default_spatial_rank=len(self.control_shape))
        grid_weights = self._resize(parameters, spatial_shape)
        if batch_shape:
            view_shape = (parameters.shape[0], *([1] * len(batch_shape)), *spatial_shape, parameters.shape[-1])
            grid_weights = grid_weights.reshape(view_shape).expand(
                parameters.shape[0], *batch_shape, *spatial_shape, parameters.shape[-1]
            )
        return GeneratorFieldSample(weights=grid_weights, spatial_shape=spatial_shape, batch_shape=batch_shape)

    def sample_shape(self, parameters: torch.Tensor, prefix_shape: Sequence[int]) -> GeneratorFieldSample:
        coordinates = parameters.new_zeros(*tuple(prefix_shape), 1)
        return self.sample(parameters, CoordinateFieldInput(coordinates))

    def _resize(self, parameters: torch.Tensor, spatial_shape: tuple[int, ...]) -> torch.Tensor:
        if spatial_shape == self.control_shape:
            return parameters
        rank = len(self.control_shape)
        mode = {1: "linear", 2: "bilinear", 3: "trilinear"}[rank]
        order = (0, rank + 1, *range(1, rank + 1))
        source = parameters.permute(order).reshape(1, parameters.shape[0] * parameters.shape[-1], *self.control_shape)
        resized = F.interpolate(source, size=spatial_shape, mode=mode, align_corners=True)
        resized = resized.reshape(parameters.shape[0], parameters.shape[-1], *spatial_shape)
        return resized.permute(0, *range(2, rank + 2), 1).contiguous()


class RBFGeneratorSampler(nn.Module):
    """Sample generators at arbitrary coordinates with normalized Gaussian RBFs.

    This sampler removes regular-grid ordering assumptions. For an exactly
    reversible indexed path, pass persistent material coordinates through
    :class:`CoordinateFieldInput` during both forward and inverse evaluation.
    """

    def __init__(self, control_points: torch.Tensor, *, length_scale: float = 1.0):
        super().__init__()
        if not isinstance(control_points, torch.Tensor):
            raise TypeError("control_points must be a torch.Tensor")
        if control_points.ndim != 2 or control_points.shape[0] < 1 or control_points.shape[1] < 1:
            raise ValueError("control_points must have shape [num_controls, sample_dim]")
        if not control_points.dtype.is_floating_point:
            control_points = control_points.to(dtype=torch.get_default_dtype())
        if float(length_scale) <= 0.0:
            raise ValueError(f"length_scale must be positive, got {length_scale}")
        self.register_buffer("control_points", control_points.detach().clone())
        self.length_scale = float(length_scale)

    @property
    def num_controls(self) -> int:
        return int(self.control_points.shape[0])

    def parameter_shape(self, path_steps: int, generator_dim: int) -> tuple[int, ...]:
        return int(path_steps), self.num_controls, int(generator_dim)

    def sample(self, parameters: torch.Tensor, field_input: CoordinateFieldInput) -> GeneratorFieldSample:
        expected = self.parameter_shape(parameters.shape[0], parameters.shape[-1])
        _check_parameters(parameters, expected)
        query = field_input.sampling_coordinates
        if query.shape[-1] != self.control_points.shape[-1]:
            raise ValueError(
                f"sampling coordinate dimension must be {self.control_points.shape[-1]}, got {query.shape[-1]}"
            )
        query_values = query.to(device=parameters.device, dtype=parameters.dtype)
        control_points = self.control_points.to(device=parameters.device, dtype=parameters.dtype)
        squared_distance = (query_values.unsqueeze(-2) - control_points).square().sum(dim=-1)
        basis = torch.softmax(-0.5 * squared_distance / (self.length_scale**2), dim=-1)
        sampled = torch.einsum("...c,scg->s...g", basis, parameters)
        spatial_shape, batch_shape = field_input.topology_shapes()
        return GeneratorFieldSample(weights=sampled, spatial_shape=spatial_shape, batch_shape=batch_shape)


def _check_parameters(parameters: torch.Tensor, expected_shape: tuple[int, ...]) -> None:
    if tuple(parameters.shape) != expected_shape:
        raise ValueError(f"generator parameters must have shape {expected_shape}, got {tuple(parameters.shape)}")


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} entries must be positive, got {value}")
    return value
