# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from research.continuum_solver import (
    BroadcastGeneratorSampler,
    CoordinateFieldInput,
    RBFGeneratorSampler,
    RegularGridGeneratorSampler,
)

pytestmark = pytest.mark.unit


def test_coordinate_field_input_separates_values_labels_and_topology():
    coordinates = torch.randn(2, 3, 4, 3)
    labels = torch.randn(2, 3, 4, 2)

    field_input = CoordinateFieldInput(coordinates, sample_coordinates=labels, spatial_shape=(3, 4))

    assert field_input.prefix_shape == (2, 3, 4)
    assert field_input.sampling_coordinates is labels
    assert field_input.topology_shapes() == ((3, 4), (2,))
    assert field_input.with_coordinates(torch.zeros_like(coordinates)).sample_coordinates is labels


def test_coordinate_field_input_supports_batch_only_topology_and_retains_implicit_identity():
    coordinates = torch.randn(5, 3)
    field_input = CoordinateFieldInput(coordinates, spatial_shape=())

    retained = field_input.retain_sample_identity()

    assert field_input.topology_shapes() == ((), (5,))
    assert retained.sample_coordinates is coordinates


@pytest.mark.parametrize(
    ("coordinates_shape", "labels_shape", "spatial_shape", "message"),
    [
        ((2, 3), (3, 1), None, "prefix shape"),
        ((2, 3, 2), None, (2,), "must be a suffix"),
    ],
)
def test_coordinate_field_input_rejects_inconsistent_metadata(coordinates_shape, labels_shape, spatial_shape, message):
    coordinates = torch.zeros(coordinates_shape)
    labels = None if labels_shape is None else torch.zeros(labels_shape)

    with pytest.raises(ValueError, match=message):
        CoordinateFieldInput(coordinates, sample_coordinates=labels, spatial_shape=spatial_shape)


def test_broadcast_sampler_preserves_explicit_batch_and_spatial_metadata():
    parameters = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    field_input = CoordinateFieldInput(torch.zeros(5, 3, 4, 2), spatial_shape=(3, 4))

    sampled = BroadcastGeneratorSampler().sample(parameters, field_input)

    assert sampled.weights.shape == (2, 5, 3, 4, 2)
    assert sampled.spatial_shape == (3, 4)
    assert sampled.batch_shape == (5,)
    assert torch.equal(sampled.weights[:, 0, 0, 0], parameters)


def test_regular_grid_sampler_interpolates_parameters_not_coordinates():
    sampler = RegularGridGeneratorSampler((2, 2))
    parameters = torch.tensor([[[[0.0], [1.0]], [[2.0], [3.0]]]])
    first_coordinates = torch.randn(3, 4, 7)
    second_coordinates = torch.randn(3, 4, 7) * 100.0

    first = sampler.sample(parameters, CoordinateFieldInput(first_coordinates)).weights
    second = sampler.sample(parameters, CoordinateFieldInput(second_coordinates)).weights

    assert first.shape == (1, 3, 4, 1)
    assert torch.equal(first, second)
    assert torch.equal(first[0, (0, -1), :, 0][:, (0, -1)], parameters[0, :, :, 0])


def test_regular_grid_sampler_broadcasts_over_explicit_batch_axes():
    sampler = RegularGridGeneratorSampler((2, 2))
    parameters = torch.randn(2, 2, 2, 3)
    field_input = CoordinateFieldInput(torch.randn(4, 5, 6, 2), spatial_shape=(5, 6))

    sampled = sampler.sample(parameters, field_input)

    assert sampled.weights.shape == (2, 4, 5, 6, 3)
    assert sampled.spatial_shape == (5, 6)
    assert sampled.batch_shape == (4,)
    assert torch.equal(sampled.weights[:, 0], sampled.weights[:, -1])


def test_rbf_sampler_is_equivariant_to_point_reordering():
    control_points = torch.tensor([[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    sampler = RBFGeneratorSampler(control_points, length_scale=0.5)
    parameters = torch.arange(12, dtype=torch.float64).reshape(2, 3, 2)
    points = torch.tensor([[0.8, 0.1], [-0.5, 0.2], [0.1, 0.9], [0.0, -0.5]], dtype=torch.float64)
    permutation = torch.tensor([2, 0, 3, 1])

    original = sampler.sample(parameters, CoordinateFieldInput(points)).weights
    reordered = sampler.sample(parameters, CoordinateFieldInput(points[permutation])).weights

    assert torch.allclose(reordered, original[:, permutation], atol=1e-12, rtol=1e-12)


def test_rbf_sampler_uses_material_labels_instead_of_transformed_values():
    control_points = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
    sampler = RBFGeneratorSampler(control_points, length_scale=0.25)
    parameters = torch.tensor([[[1.0], [5.0]]], dtype=torch.float64)
    labels = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
    first_values = torch.zeros(2, 3, dtype=torch.float64)
    second_values = torch.full((2, 3), 100.0, dtype=torch.float64)

    first = sampler.sample(parameters, CoordinateFieldInput(first_values, sample_coordinates=labels)).weights
    second = sampler.sample(parameters, CoordinateFieldInput(second_values, sample_coordinates=labels)).weights

    assert torch.equal(first, second)
    assert first[0, 0, 0] < first[0, 1, 0]


@pytest.mark.parametrize("control_shape", [(), (2, 2, 2, 2), (2, 0)])
def test_regular_grid_sampler_rejects_unsupported_control_shapes(control_shape):
    with pytest.raises(ValueError):
        RegularGridGeneratorSampler(control_shape)
