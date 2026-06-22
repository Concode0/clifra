# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from hypothesis import assume, given
from hypothesis import strategies as st

from clifra.core.runtime.algebra import AlgebraContext
from clifra.functional.activation import geometric_gelu, geometric_square, grade_swish
from clifra.layers import (
    BladeSelector,
    CliffordLayerNorm,
    CliffordLinear,
    GeometricGELU,
    GeometricNeutralizer,
    GeometricSquare,
    GradeSwish,
    MultiVersorLayer,
    ProductLayer,
    ReflectionLayer,
    RotorGadget,
    VersorLayer,
)
from tests.helpers.hypothesis_cases import (
    PROPERTY_SETTINGS,
    QUICK_PROPERTY_SETTINGS,
    compact_product_cases,
    grade_sets,
    signature_strategy,
    tensor_with_shape,
)
from tests.helpers.small_oracle import SmallCliffordOracle

pytestmark = pytest.mark.unit


@PROPERTY_SETTINGS
@given(signature=signature_strategy(max_n=4), data=st.data())
def test_clifford_linear_traditional_matches_channel_einsum_for_declared_layouts(signature, data):
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    grades = data.draw(st.sampled_from(grade_sets(algebra.n)))
    layout = algebra.layout(grades)
    batch = data.draw(st.integers(min_value=1, max_value=3))
    extra = data.draw(st.booleans())
    in_channels = data.draw(st.integers(min_value=1, max_value=4))
    out_channels = data.draw(st.integers(min_value=1, max_value=4))
    prefix = (batch, 2) if extra else (batch,)
    values = data.draw(tensor_with_shape((*prefix, in_channels, layout.dim)))
    weights = data.draw(tensor_with_shape((out_channels, in_channels)))
    bias = data.draw(tensor_with_shape((out_channels, layout.dim)))
    layer = CliffordLinear(algebra, in_channels, out_channels, layout=layout).to(dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(weights)
        layer.bias.copy_(bias)

    actual = layer(values)
    expected = torch.einsum("oi,...id->...od", weights, values)
    expected = expected + bias.view((1,) * len(prefix) + (out_channels, layout.dim))

    assert actual.shape == (*prefix, out_channels, layout.dim)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


@PROPERTY_SETTINGS
@given(signature=signature_strategy(max_n=4), data=st.data())
def test_layer_norm_initialization_outputs_unit_lane_norm_for_nonzero_inputs(signature, data):
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    grades = data.draw(st.sampled_from(grade_sets(algebra.n)))
    layout = algebra.layout(grades)
    batch = data.draw(st.integers(min_value=1, max_value=3))
    channels = data.draw(st.integers(min_value=1, max_value=4))
    values = data.draw(tensor_with_shape((batch, channels, layout.dim)))
    layer = CliffordLayerNorm(algebra, channels, layout=layout).to(dtype=torch.float64)

    actual = layer(values)
    norms = values.norm(dim=-1, keepdim=True).clamp_min(layer.eps)
    expected = values / norms

    assert actual.shape == values.shape
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    well_scaled = values.norm(dim=-1) >= layer.eps
    if bool(well_scaled.any()):
        assert torch.allclose(
            actual.norm(dim=-1)[well_scaled],
            torch.ones_like(actual.norm(dim=-1)[well_scaled]),
            atol=1e-10,
        )


@PROPERTY_SETTINGS
@given(signature=signature_strategy(max_n=4), data=st.data())
def test_blade_selector_uses_declared_gate_logits_componentwise(signature, data):
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    grades = data.draw(st.sampled_from(grade_sets(algebra.n)))
    layout = algebra.layout(grades)
    batch = data.draw(st.integers(min_value=1, max_value=3))
    channels = data.draw(st.integers(min_value=1, max_value=4))
    values = data.draw(tensor_with_shape((batch, channels, layout.dim)))
    logits = data.draw(tensor_with_shape((channels, layout.dim)))
    layer = BladeSelector(algebra, channels, layout=layout).to(dtype=torch.float64)
    with torch.no_grad():
        layer.weights.copy_(logits)

    actual = layer(values)
    expected = values * (2.0 * torch.sigmoid(logits)).view(1, channels, layout.dim)

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


@PROPERTY_SETTINGS
@given(signature=signature_strategy(max_n=4), data=st.data())
def test_activation_layers_match_stateless_activation_formulas(signature, data):
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    grades = data.draw(st.sampled_from(grade_sets(algebra.n)))
    layout = algebra.layout(grades)
    batch = data.draw(st.integers(min_value=1, max_value=3))
    channels = data.draw(st.integers(min_value=1, max_value=4))
    values = data.draw(tensor_with_shape((batch, channels, layout.dim)))
    gelu_bias = data.draw(tensor_with_shape((channels,)))
    grade_weights = data.draw(tensor_with_shape((algebra.n + 1,)))
    grade_biases = data.draw(tensor_with_shape((algebra.n + 1,)))

    gelu = GeometricGELU(algebra, channels, layout=layout).to(dtype=torch.float64)
    swish = GradeSwish(algebra, channels, layout=layout).to(dtype=torch.float64)
    with torch.no_grad():
        gelu.bias.copy_(gelu_bias)
        swish.grade_weights.copy_(grade_weights)
        swish.grade_biases.copy_(grade_biases)

    assert torch.allclose(gelu(values), geometric_gelu(values, bias=gelu_bias), atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        swish(values),
        grade_swish(
            values,
            grade_index=layout.grade_indices_tensor(device=values.device),
            grade_weights=grade_weights,
            grade_biases=grade_biases,
            n_grades=algebra.n + 1,
        ),
        atol=1e-12,
        rtol=1e-12,
    )


@PROPERTY_SETTINGS
@given(signature=signature_strategy(max_n=4), data=st.data())
def test_geometric_square_layer_matches_small_oracle_full_layout(signature, data):
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(*signature)
    batch = data.draw(st.integers(min_value=1, max_value=3))
    channels = data.draw(st.integers(min_value=1, max_value=4))
    values = data.draw(tensor_with_shape((batch, channels, algebra.dim)))
    gate_logit = data.draw(tensor_with_shape((channels,)))
    layer = GeometricSquare(algebra, channels).to(dtype=torch.float64)
    with torch.no_grad():
        layer.gate_logit.copy_(gate_logit)

    actual = layer(values)
    expected = values + torch.sigmoid(gate_logit).view(1, channels, 1) * oracle.product(values, values)

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert torch.allclose(actual, geometric_square(algebra, values, gate=torch.sigmoid(gate_logit)), atol=1e-10, rtol=1e-10)


@PROPERTY_SETTINGS
@given(case=compact_product_cases())
def test_product_layer_declared_layouts_match_small_oracle(case):
    signature, op, left_grades, right_grades, output_grades, left, right = case
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(*signature)
    left_layout = algebra.layout(left_grades)
    right_layout = algebra.layout(right_grades)
    output_layout = algebra.layout(output_grades)
    layer = ProductLayer(
        algebra,
        op=op,
        left_layout=left_layout,
        right_layout=right_layout,
        output_layout=output_layout,
    )

    actual = layer(left, right)
    expected = oracle.product(
        left,
        right,
        op=op,
        left_indices=left_layout.basis_indices,
        right_indices=right_layout.basis_indices,
        output_indices=output_layout.basis_indices,
    )

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


@QUICK_PROPERTY_SETTINGS
@given(n=st.integers(min_value=2, max_value=4), data=st.data())
def test_versor_layer_bivector_action_matches_small_oracle_on_vectors(n, data):
    algebra = AlgebraContext(n, 0, 0, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(n, 0, 0)
    vector_layout = algebra.layout((1,))
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout(range(0, n + 1, 2))
    batch = data.draw(st.integers(min_value=1, max_value=3))
    channels = data.draw(st.integers(min_value=1, max_value=3))
    values = data.draw(tensor_with_shape((batch, channels, vector_layout.dim)))
    weights = 0.25 * data.draw(tensor_with_shape((channels, bivector_layout.dim)))
    layer = VersorLayer(algebra, channels, input_layout=vector_layout).to(dtype=torch.float64)
    with torch.no_grad():
        layer.grade_weights.copy_(weights)

    rotor = even_layout.full(algebra.exp(-0.5 * weights, input_layout=bivector_layout, output_layout=even_layout))
    rotor_reverse = oracle.reverse(rotor)
    full_values = vector_layout.full(values)
    expected_full = oracle.product(oracle.product(rotor, full_values), rotor_reverse)

    actual = layer(values)
    expected = vector_layout.compact(expected_full)

    assert torch.allclose(actual, expected, atol=1e-9, rtol=1e-9)


@QUICK_PROPERTY_SETTINGS
@given(n=st.integers(min_value=1, max_value=4), data=st.data())
def test_reflection_layer_matches_small_oracle_normalized_vector_action(n, data):
    algebra = AlgebraContext(n, 0, 0, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(n, 0, 0)
    vector_layout = algebra.layout((1,))
    batch = data.draw(st.integers(min_value=1, max_value=3))
    channels = data.draw(st.integers(min_value=1, max_value=3))
    values = data.draw(tensor_with_shape((batch, channels, vector_layout.dim)))
    normals = data.draw(tensor_with_shape((channels, vector_layout.dim)))
    assume(bool((normals.norm(dim=-1) > 1e-8).all()))
    unit_normals = normals / normals.norm(dim=-1, keepdim=True)
    layer = ReflectionLayer(algebra, channels, input_layout=vector_layout).to(dtype=torch.float64)
    with torch.no_grad():
        layer.vector_weights.copy_(normals)

    normal_hat = oracle.grade_involution(unit_normals, vector_layout.basis_indices)
    normal_inv = oracle.blade_inverse(unit_normals, vector_layout.basis_indices)
    full_values = vector_layout.full(values)
    expected_full = oracle.product(
        oracle.product(
            normal_hat,
            full_values,
            left_indices=vector_layout.basis_indices,
        ),
        normal_inv,
        right_indices=vector_layout.basis_indices,
    )

    actual = layer(values)
    expected = vector_layout.compact(expected_full)

    assert torch.allclose(actual, expected, atol=1e-9, rtol=1e-9)


@QUICK_PROPERTY_SETTINGS
@given(n=st.integers(min_value=2, max_value=4), data=st.data())
def test_multi_versor_layer_matches_weighted_small_oracle_rotor_sum(n, data):
    algebra = AlgebraContext(n, 0, 0, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(n, 0, 0)
    vector_layout = algebra.layout((1,))
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout(range(0, n + 1, 2))
    batch = data.draw(st.integers(min_value=1, max_value=2))
    channels = data.draw(st.integers(min_value=1, max_value=3))
    num_versors = data.draw(st.integers(min_value=1, max_value=3))
    values = data.draw(tensor_with_shape((batch, channels, vector_layout.dim)))
    grade_weights = 0.25 * data.draw(tensor_with_shape((num_versors, bivector_layout.dim)))
    mix_weights = data.draw(tensor_with_shape((channels, num_versors)))
    layer = MultiVersorLayer(algebra, channels, num_versors=num_versors, input_layout=vector_layout).to(dtype=torch.float64)
    with torch.no_grad():
        layer.grade_weights.copy_(grade_weights)
        layer.weights.copy_(mix_weights)

    rotors = even_layout.full(algebra.exp(-0.5 * grade_weights, input_layout=bivector_layout, output_layout=even_layout))
    rotor_reverses = oracle.reverse(rotors)
    full_values = vector_layout.full(values)
    expected_full = torch.zeros_like(full_values)
    for channel in range(channels):
        channel_values = full_values[:, channel, :]
        for item in range(num_versors):
            transformed = oracle.product(
                oracle.product(rotors[item : item + 1], channel_values),
                rotor_reverses[item : item + 1],
            )
            expected_full[:, channel, :] = expected_full[:, channel, :] + mix_weights[channel, item] * transformed

    actual = layer(values)
    expected = vector_layout.compact(expected_full)

    assert torch.allclose(actual, expected, atol=1e-9, rtol=1e-9)


@QUICK_PROPERTY_SETTINGS
@given(
    n=st.integers(min_value=2, max_value=4),
    in_channels=st.integers(min_value=1, max_value=5),
    out_channels=st.integers(min_value=1, max_value=5),
    aggregation=st.sampled_from(("mean", "sum")),
    data=st.data(),
)
def test_rotor_gadget_identity_parameters_reduce_to_declared_channel_mix(n, in_channels, out_channels, aggregation, data):
    algebra = AlgebraContext(n, 0, 0, device="cpu", dtype=torch.float64)
    batch = data.draw(st.integers(min_value=1, max_value=3))
    values = data.draw(tensor_with_shape((batch, in_channels, algebra.dim)))
    layer = RotorGadget(
        algebra,
        in_channels=in_channels,
        out_channels=out_channels,
        num_rotor_pairs=data.draw(st.integers(min_value=1, max_value=4)),
        aggregation=aggregation,
    ).to(dtype=torch.float64)
    with torch.no_grad():
        layer.bivector_left.zero_()
        layer.bivector_right.zero_()

    mix = layer._channel_mix_sum if aggregation == "sum" else layer._channel_mix_mean
    expected = torch.einsum("oi,...id->...od", mix.to(dtype=values.dtype), values)

    assert torch.allclose(layer(values), expected, atol=1e-10, rtol=1e-10)


@QUICK_PROPERTY_SETTINGS
@given(data=st.data())
def test_geometric_neutralizer_removes_linear_bivector_scalar_covariance(data):
    algebra = AlgebraContext(3, 0, 0, device="cpu", dtype=torch.float64)
    layout = algebra.layout((0, 2))
    neutralizer = GeometricNeutralizer(algebra, channels=1, momentum=1.0, layout=layout).to(dtype=torch.float64)
    coefficients = data.draw(tensor_with_shape((layout.positions_for_grades((2,)).numel(), 1)))
    offset = data.draw(tensor_with_shape((1,)))
    bivec = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0],
        ],
        dtype=torch.float64,
    )
    scalar = bivec @ coefficients + offset
    values = torch.zeros(4, 1, layout.dim, dtype=torch.float64)
    values[:, 0, layout.positions_for_grades((0,))] = scalar
    values[:, 0, layout.positions_for_grades((2,))] = bivec

    actual = neutralizer(values)

    assert torch.allclose(actual[:, 0, layout.positions_for_grades((0,))], offset.expand(4, 1), atol=1e-10, rtol=1e-10)
    assert torch.allclose(actual[:, 0, layout.positions_for_grades((2,))], bivec, atol=1e-12, rtol=1e-12)
