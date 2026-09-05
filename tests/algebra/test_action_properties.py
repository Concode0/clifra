# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from clifra.core.algebra import AlgebraContext
from tests.helpers.hypothesis_cases import (
    CORE_NUMERIC_SETTINGS,
    QUICK_PROPERTY_SETTINGS,
    signature_strategy,
    tensor_with_shape,
)
from tests.helpers.policy import PreferRoute
from tests.helpers.small_oracle import SmallCliffordOracle

pytestmark = [pytest.mark.unit, pytest.mark.property]


def _force_action_route(route: str):
    return PreferRoute("action", route)


def _rotor_action_reference(algebra, oracle, values, weights, layout):
    parameter_layout = algebra.layout((2,))
    rotor_layout = algebra.layout(range(0, algebra.n + 1, 2))
    rotor = rotor_layout.full(algebra.bivector_exp(-0.5 * weights, input=parameter_layout, output=rotor_layout))
    full_values = layout.full(values)
    return layout.compact(oracle.product(oracle.product(rotor, full_values), oracle.reverse(rotor)))


@CORE_NUMERIC_SETTINGS
@given(signature=signature_strategy(max_n=4), data=st.data())
def test_full_sandwich_execution_modes_match_independent_products(signature, data):
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(*signature)
    batch = data.draw(st.integers(min_value=1, max_value=2))
    channels = data.draw(st.integers(min_value=1, max_value=3))
    values = data.draw(tensor_with_shape((batch, channels, algebra.dim)))
    batch_left = data.draw(tensor_with_shape((batch, algebra.dim)))
    batch_right = data.draw(tensor_with_shape((batch, algebra.dim)))
    channel_left = data.draw(tensor_with_shape((channels, algebra.dim)))
    channel_right = data.draw(tensor_with_shape((channels, algebra.dim)))

    expected_batch = oracle.product(
        oracle.product(batch_left.unsqueeze(-2), values),
        batch_right.unsqueeze(-2),
    )
    expected_channels = oracle.product(oracle.product(channel_left, values), channel_right)

    assert torch.allclose(
        action_helpers.sandwich_product(algebra, batch_left, values, batch_right),
        expected_batch,
        atol=1e-10,
        rtol=1e-10,
    )
    assert torch.allclose(
        action_helpers.per_channel_sandwich(algebra, channel_left, values, channel_right),
        expected_channels,
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.parametrize("route", ("vector_matrix", "rotor_product"))
@QUICK_PROPERTY_SETTINGS
@given(signature=signature_strategy(min_n=2, max_n=4), data=st.data())
def test_forced_compact_versor_action_routes_match_independent_products(route, signature, data):
    algebra = configured_algebra(
        *signature, device="cpu", dtype=torch.float64, planning_policy=_force_action_route(route)
    )
    oracle = SmallCliffordOracle(*signature)
    layout = algebra.layout((1,))
    channels = data.draw(st.integers(1, 3))
    values = data.draw(tensor_with_shape((data.draw(st.integers(1, 2)), channels, layout.dim)))
    weights = 0.1 * data.draw(tensor_with_shape((channels, algebra.layout((2,)).dim)))
    action = algebra.plan_versor_action(grade=2, input=layout, output=layout, parameter=algebra.layout((2,)))

    assert action._kernel.route == route
    assert torch.allclose(
        action(values, weights), _rotor_action_reference(algebra, oracle, values, weights, layout), atol=1e-9, rtol=1e-9
    )


@pytest.mark.parametrize("route", ("full_action_matrix", "rotor_product"))
@QUICK_PROPERTY_SETTINGS
@given(signature=signature_strategy(min_n=2, max_n=4), data=st.data())
def test_forced_full_versor_action_routes_match_independent_products(route, signature, data):
    algebra = configured_algebra(
        *signature, device="cpu", dtype=torch.float64, planning_policy=_force_action_route(route)
    )
    oracle = SmallCliffordOracle(*signature)
    layout = algebra.layout(range(algebra.n + 1))
    channels = data.draw(st.integers(1, 2))
    values = data.draw(tensor_with_shape((1, channels, layout.dim)))
    weights = 0.1 * data.draw(tensor_with_shape((channels, algebra.layout((2,)).dim)))
    action = algebra.plan_versor_action(grade=2, input=layout, output=layout, parameter=algebra.layout((2,)))

    assert action._kernel.route == route
    assert torch.allclose(
        action(values, weights), _rotor_action_reference(algebra, oracle, values, weights, layout), atol=1e-9, rtol=1e-9
    )


@pytest.mark.parametrize("route", ("full_action_matrix", "paired_rotor_product"))
@QUICK_PROPERTY_SETTINGS
@given(signature=signature_strategy(min_n=2, max_n=4), data=st.data())
def test_forced_paired_action_routes_match_independent_products(route, signature, data):
    algebra = configured_algebra(
        *signature, device="cpu", dtype=torch.float64, planning_policy=_force_action_route(route)
    )
    oracle = SmallCliffordOracle(*signature)
    layout = algebra.layout(range(algebra.n + 1))
    parameter_layout = algebra.layout((2,))
    rotor_layout = algebra.layout(range(0, algebra.n + 1, 2))
    channels = data.draw(st.integers(1, 3))
    pairs = data.draw(st.integers(1, 3))
    values = data.draw(tensor_with_shape((1, channels, layout.dim)))
    left_weights = 0.1 * data.draw(tensor_with_shape((pairs, parameter_layout.dim)))
    right_weights = 0.1 * data.draw(tensor_with_shape((pairs, parameter_layout.dim)))
    channel_to_pair = torch.tensor([index % pairs for index in range(channels)], dtype=torch.long)
    action = action_helpers.plan_paired_bivector_action(
        algebra, input_layout=layout, output_layout=layout, parameter_layout=parameter_layout
    )
    left = rotor_layout.full(algebra.bivector_exp(-0.5 * left_weights, input=parameter_layout, output=rotor_layout))
    right = oracle.reverse(
        rotor_layout.full(algebra.bivector_exp(-0.5 * right_weights, input=parameter_layout, output=rotor_layout))
    )
    expected = torch.stack(
        [
            oracle.product(oracle.product(left[channel_to_pair[c]], values[:, c]), right[channel_to_pair[c]])
            for c in range(channels)
        ],
        dim=1,
    )

    assert action.route == route
    assert torch.allclose(action(values, left_weights, right_weights, channel_to_pair), expected, atol=1e-9, rtol=1e-9)


from clifra.core._kernel.configuration import configured_algebra
from tests.helpers import action as action_helpers
