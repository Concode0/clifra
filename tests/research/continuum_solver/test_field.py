# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from clifra.core.runtime.algebra import AlgebraContext
from research.continuum_solver import (
    CoordinateChart,
    CoordinateFieldInput,
    InvertibleBivectorField,
    RBFGeneratorSampler,
)

pytestmark = pytest.mark.unit


class _CountingAction(nn.Module):
    def __init__(self, action):
        super().__init__()
        self.action = action
        self.weight_rows: list[int] = []

    def forward(self, values, weights):
        self.weight_rows.append(int(weights.shape[0]))
        return self.action(values, weights)


@pytest.mark.parametrize("projective", [False, True])
def test_zero_bivectors_are_identity(projective):
    algebra = AlgebraContext(2, 0, int(projective), device="cpu", dtype=torch.float64)
    field = InvertibleBivectorField(algebra, 2, projective=projective, path_steps=3, init_scale=0.0)
    coordinates = torch.randn(2, 5, 2, dtype=torch.float64)

    actual = field(coordinates)

    assert actual.dtype == coordinates.dtype
    assert actual.shape == coordinates.shape
    assert torch.allclose(actual, coordinates, atol=1e-12, rtol=1e-12)


def test_projective_bivector_field_applies_expected_rotation():
    algebra = AlgebraContext(2, 0, 1, device="cpu", dtype=torch.float64)
    field = InvertibleBivectorField(algebra, 2, projective=True, init_scale=0.0)
    rotation_lane = field.bivector_layout.basis_indices.index(0b011)
    with torch.no_grad():
        field.bivectors[0, rotation_lane] = math.pi / 2.0
    coordinates = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)

    actual = field(coordinates)
    expected = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64)

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_regular_grid_path_round_trips_and_reports_shapes():
    torch.manual_seed(4)
    algebra = AlgebraContext(3, 0, 0, device="cpu", dtype=torch.float64)
    field = InvertibleBivectorField(
        algebra,
        3,
        path_steps=3,
        control_shape=(2, 3),
        init_scale=0.15,
    )
    coordinates = torch.randn(4, 5, 7, 3, dtype=torch.float64)
    field_input = CoordinateFieldInput(coordinates, spatial_shape=(5, 7))

    state = field.state(field_input)
    reconstructed = field.inverse(state.inverse_input())

    assert state.bivector_weights.shape == (3, 4, 5, 7, 3)
    assert state.spatial_shape == (5, 7)
    assert state.batch_shape == (4,)
    assert torch.allclose(reconstructed, coordinates, atol=2e-10, rtol=2e-10)


def test_coordinate_sampled_path_round_trips_with_persistent_material_labels():
    torch.manual_seed(7)
    algebra = AlgebraContext(2, 0, 0, device="cpu", dtype=torch.float64)
    control_points = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)
    field = InvertibleBivectorField(
        algebra,
        2,
        path_steps=2,
        generator_sampler=RBFGeneratorSampler(control_points, length_scale=0.35),
        init_scale=0.4,
    )
    coordinates = torch.randn(8, 2, dtype=torch.float64)
    material_labels = torch.linspace(-1.0, 1.0, 8, dtype=torch.float64).unsqueeze(-1)
    field_input = CoordinateFieldInput(coordinates, sample_coordinates=material_labels)

    state = field.state(field_input)
    reconstructed = field.inverse(state.inverse_input())

    assert state.bivector_weights.shape == (2, 8, 1)
    assert torch.allclose(reconstructed, coordinates, atol=2e-10, rtol=2e-10)
    with pytest.raises(ValueError, match="requires coordinates"):
        field.weights_for_shape((8,))


def test_state_retains_implicit_coordinate_sample_identity_for_inverse():
    algebra = AlgebraContext(2, 0, 0, device="cpu", dtype=torch.float64)
    control_points = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    field = InvertibleBivectorField(
        algebra,
        2,
        generator_sampler=RBFGeneratorSampler(control_points, length_scale=0.4),
        init_scale=0.0,
    )
    with torch.no_grad():
        field.bivectors[0, :, 0] = torch.tensor([-0.8, 0.8], dtype=torch.float64)
    coordinates = torch.tensor([[-0.8, 0.3], [0.7, -0.2]], dtype=torch.float64)

    state = field.state(coordinates)
    reconstructed = field.inverse(state.inverse_input())

    assert state.field_input is not None
    assert state.field_input.sample_coordinates is coordinates
    assert torch.allclose(reconstructed, coordinates, atol=1e-10, rtol=1e-10)


def test_field_preserves_gradients_through_coordinates_and_generators():
    torch.manual_seed(11)
    algebra = AlgebraContext(3, 0, 0, device="cpu", dtype=torch.float64)
    field = InvertibleBivectorField(algebra, 3, control_shape=(2, 2), init_scale=0.1)
    coordinates = torch.randn(3, 4, 3, dtype=torch.float64, requires_grad=True)

    output = field(coordinates)
    loss = output.square().sum()
    loss.backward()

    assert output.dtype == torch.float64
    assert coordinates.grad is not None
    assert torch.isfinite(coordinates.grad).all()
    assert field.bivectors.grad is not None
    assert torch.isfinite(field.bivectors.grad).all()


def test_local_rotor_inversion_does_not_claim_global_injectivity():
    algebra = AlgebraContext(2, 0, 0, device="cpu", dtype=torch.float64)
    field = InvertibleBivectorField(algebra, 2, control_shape=(2,), init_scale=0.0)
    with torch.no_grad():
        field.bivectors[0, 1, 0] = math.pi
    coordinates = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64)

    state = field.state(coordinates)
    reconstructed = field.inverse(state.inverse_input())

    assert torch.allclose(state.deformed_coordinates[0], state.deformed_coordinates[1], atol=1e-10, rtol=1e-10)
    assert torch.allclose(reconstructed, coordinates, atol=1e-10, rtol=1e-10)


def test_field_rejects_ambiguous_sampler_configuration():
    algebra = AlgebraContext(2, 0, 0, device="cpu", dtype=torch.float64)
    sampler = RBFGeneratorSampler(torch.tensor([[0.0], [1.0]], dtype=torch.float64))

    with pytest.raises(ValueError, match="either control_shape or generator_sampler"):
        InvertibleBivectorField(algebra, 2, control_shape=(2,), generator_sampler=sampler)


def test_field_accepts_injected_chart_and_action_components():
    algebra = AlgebraContext(2, 0, 0, device="cpu", dtype=torch.float64)
    chart = CoordinateChart.direct(algebra, 2)
    bivector_layout = algebra.layout((2,))
    action = algebra.plan_versor_action(
        grade=2,
        input_layout=chart.layout,
        output_layout=chart.layout,
        parameter_layout=bivector_layout,
    )
    field = InvertibleBivectorField(
        algebra,
        2,
        chart=chart,
        action=action,
        init_scale=0.0,
    )
    coordinates = torch.randn(4, 2, dtype=torch.float64)

    assert field.chart is chart
    assert field.action is action
    assert torch.equal(field(coordinates), coordinates)


def test_broadcast_field_exponentiates_one_generator_per_path_step():
    algebra = AlgebraContext(3, 0, 0, device="cpu", dtype=torch.float64)
    chart = CoordinateChart.direct(algebra, 3)
    planned_action = algebra.plan_versor_action(
        grade=2,
        input_layout=chart.layout,
        output_layout=chart.layout,
        parameter_layout=algebra.layout((2,)),
    )
    action = _CountingAction(planned_action)
    field = InvertibleBivectorField(algebra, 3, path_steps=3, chart=chart, action=action)

    field(torch.randn(4, 5, 3, dtype=torch.float64))

    assert action.weight_rows == [1, 1, 1]


def test_regular_grid_field_does_not_repeat_exponentials_across_batches():
    algebra = AlgebraContext(2, 0, 0, device="cpu", dtype=torch.float64)
    chart = CoordinateChart.direct(algebra, 2)
    planned_action = algebra.plan_versor_action(
        grade=2,
        input_layout=chart.layout,
        output_layout=chart.layout,
        parameter_layout=algebra.layout((2,)),
    )
    action = _CountingAction(planned_action)
    field = InvertibleBivectorField(
        algebra,
        2,
        path_steps=2,
        control_shape=(2, 2),
        chart=chart,
        action=action,
    )
    field_input = CoordinateFieldInput(torch.randn(6, 4, 5, 2, dtype=torch.float64), spatial_shape=(4, 5))

    field(field_input)

    assert action.weight_rows == [20, 20]
