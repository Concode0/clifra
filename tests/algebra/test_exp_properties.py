# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from itertools import combinations

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from clifra.core._kernel.planning.policy import DEFAULT_PLANNING_POLICY, FormulaPolicy, Polynomial
from clifra.core.algebra import AlgebraContext
from tests.helpers.bivector_exp_oracle import bivector_exp_cpu_reference
from tests.helpers.hypothesis_cases import (
    CORE_NUMERIC_SETTINGS,
    DEEP_NUMERIC_SETTINGS,
    signature_strategy,
    tensor_with_shape,
)
from tests.helpers.small_oracle import SmallCliffordOracle

pytestmark = [pytest.mark.unit, pytest.mark.property]


def _force_exp_route(route: str) -> FormulaPolicy:
    return FormulaPolicy(
        tuple(
            replace(rule, score=Polynomial(constant=-100.0))
            if (rule.family, rule.route) == ("bivector_exp", route)
            else rule
            for rule in DEFAULT_PLANNING_POLICY.rules
        )
    )


@st.composite
def _well_conditioned_spectral_bivectors(draw):
    angles = torch.tensor(
        [
            draw(st.floats(0.05, 0.08, allow_nan=False, allow_infinity=False)),
            draw(st.floats(0.12, 0.15, allow_nan=False, allow_infinity=False)),
            draw(st.floats(0.20, 0.23, allow_nan=False, allow_infinity=False)),
        ],
        dtype=torch.float64,
    )
    generator = torch.zeros(6, 6, dtype=torch.float64)
    for plane, angle in enumerate(angles):
        x, y = 2 * plane, 2 * plane + 1
        generator[y, x] = angle
        generator[x, y] = -angle
    rotation, _ = torch.linalg.qr(torch.randn(6, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(421)))
    generator = rotation @ generator @ rotation.T
    pairs = sorted(((1 << i) | (1 << j), i, j) for i in range(6) for j in range(i + 1, 6))
    return torch.tensor(
        [generator[j, i] for _, i, j in pairs],
        dtype=torch.float64,
    ).unsqueeze(0)


def _even_grade_sets(n: int) -> tuple[tuple[int, ...], ...]:
    grades = tuple(range(0, n + 1, 2))
    return tuple(tuple(selection) for size in range(1, len(grades) + 1) for selection in combinations(grades, size))


@st.composite
def _bivector_exp_cases(draw, *, include_degenerate: bool = True):
    signature = draw(signature_strategy(min_n=2, max_n=4, include_degenerate=include_degenerate))
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    input_layout = algebra.layout((2,))
    output_grades = draw(st.sampled_from(_even_grade_sets(algebra.n)))
    output_layout = algebra.layout(output_grades)
    batch = draw(st.integers(min_value=1, max_value=2))
    values = draw(tensor_with_shape((batch, input_layout.dim)))
    return signature, input_layout, output_layout, values


@st.composite
def _euclidean_bivector_exp_cases(draw):
    n = draw(st.integers(min_value=2, max_value=4))
    algebra = AlgebraContext(n, 0, 0, device="cpu", dtype=torch.float64)
    input_layout = algebra.layout((2,))
    batch = draw(st.integers(min_value=1, max_value=2))
    values = draw(tensor_with_shape((batch, input_layout.dim)))
    return (n, 0, 0), input_layout, values


@CORE_NUMERIC_SETTINGS
@given(case=_bivector_exp_cases())
def test_bivector_exp_matches_cpu_oracle_for_even_output_layouts(case):
    signature, input_layout, output_layout, values = case
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)

    actual = algebra.bivector_exp(values, input=input_layout, output=output_layout)
    expected = bivector_exp_cpu_reference(
        algebra,
        values,
        input_layout=input_layout,
        output_layout=output_layout,
    )

    assert torch.allclose(actual, expected, atol=1e-9, rtol=1e-9)


@CORE_NUMERIC_SETTINGS
@given(case=_euclidean_bivector_exp_cases())
def test_euclidean_bivector_exp_is_unit_rotor_by_small_oracle(case):
    signature, input_layout, values = case
    p, q, r = signature
    algebra = AlgebraContext(p, q, r, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(p, q, r)
    even_layout = algebra.layout(range(0, algebra.n + 1, 2))

    rotor = algebra.bivector_exp(values, input=input_layout, output=even_layout)
    rotor_reverse = oracle.reverse(rotor, even_layout.basis_indices)
    product = oracle.product(
        rotor,
        rotor_reverse,
        left_indices=even_layout.basis_indices,
        right_indices=even_layout.basis_indices,
    )
    expected = torch.zeros_like(product)
    expected[..., 0] = 1.0

    assert torch.allclose(product, expected, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("exponent", range(1, 4))
@CORE_NUMERIC_SETTINGS
@given(data=st.data())
def test_disjoint_null_bivectors_match_projected_nilpotent_closed_form(exponent, data):
    r = 1 << exponent
    algebra = AlgebraContext(0, 0, r, device="cpu", dtype=torch.float64)
    bivector_layout = algebra.layout((2,))
    output_layout = algebra.layout((0, 2))
    batch = data.draw(st.integers(min_value=1, max_value=2))
    coefficients = 0.25 * data.draw(tensor_with_shape((batch, r // 2)))
    values = torch.zeros(batch, bivector_layout.dim, dtype=torch.float64)
    for pair in range(r // 2):
        blade = (1 << (2 * pair)) | (1 << (2 * pair + 1))
        values[:, bivector_layout.basis_indices.index(blade)] = coefficients[:, pair]

    actual = algebra.bivector_exp(values, input=bivector_layout, output=output_layout)
    expected = torch.cat((torch.ones(batch, 1, dtype=torch.float64), values), dim=-1)

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_mixed_signature_scalar_exp_gradient_at_zero_real_invariant():
    algebra = AlgebraContext(1, 3, 0, device="cpu", dtype=torch.float64)
    input_layout = algebra.layout((2,))
    output_layout = algebra.layout((0,))
    values = torch.zeros(1, input_layout.dim, dtype=torch.float64)
    boost = input_layout.basis_indices.index((1 << 0) | (1 << 1))
    rotation = input_layout.basis_indices.index((1 << 2) | (1 << 3))
    values[0, boost] = 1.0
    values[0, rotation] = 1.0
    values.requires_grad_(True)

    scalar = algebra.bivector_exp(values, input=input_layout, output=output_layout)
    gradient = torch.autograd.grad(scalar, values)[0]
    expected = torch.zeros_like(values)
    expected[0, boost] = torch.sinh(values[0, boost]) * torch.cos(values[0, rotation])
    expected[0, rotation] = -torch.cosh(values[0, boost]) * torch.sin(values[0, rotation])

    assert torch.allclose(gradient, expected, atol=1e-12, rtol=1e-12)


@DEEP_NUMERIC_SETTINGS
@given(case=_bivector_exp_cases())
def test_bivector_exp_vjp_matches_dense_independent_reference(case):
    signature, input_layout, output_layout, values = case
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    values = values.requires_grad_(True)
    actual = algebra.bivector_exp(values, input=input_layout, output=output_layout)
    expected = bivector_exp_cpu_reference(
        algebra,
        values,
        input_layout=input_layout,
        output_layout=output_layout,
    )
    cotangent = torch.linspace(0.5, 1.5, actual.numel(), dtype=torch.float64).reshape_as(actual)
    actual_vjp = torch.autograd.grad(actual, values, cotangent, retain_graph=True)[0]
    expected_vjp = torch.autograd.grad(expected, values, cotangent)[0]

    assert torch.allclose(actual_vjp, expected_vjp, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("route", ("closed_simple", "closed_biquadratic", "spectral_local", "left_matrix_exp"))
@CORE_NUMERIC_SETTINGS
@given(data=st.data())
def test_forced_bivector_exp_routes_match_dense_reference_and_vjp(route, data):
    if route == "closed_simple":
        signature = data.draw(signature_strategy(min_n=2, max_n=3))
    elif route == "closed_biquadratic":
        signature = data.draw(signature_strategy(min_n=4, max_n=5))
    elif route == "spectral_local":
        signature = (6, 0, 0)
    else:
        signature = data.draw(signature_strategy(min_n=2, max_n=5))
    algebra = configured_algebra(*signature, device="cpu", dtype=torch.float64, planning_policy=_force_exp_route(route))
    input_layout = algebra.layout((2,))
    output_layout = algebra.layout(range(0, algebra.n + 1, 2))
    if route == "spectral_local":
        raw = data.draw(_well_conditioned_spectral_bivectors())
    else:
        raw = 0.1 * data.draw(tensor_with_shape((1, input_layout.dim)))
    values = raw.clone().requires_grad_(True)
    reference_values = raw.clone().requires_grad_(True)
    executor = algebra._planner.bivector_exp_executor(input_layout=input_layout, output_layout=output_layout)

    actual = executor(values)
    expected = bivector_exp_cpu_reference(
        algebra,
        reference_values,
        input_layout=input_layout,
        output_layout=output_layout,
    )
    cotangent = torch.linspace(0.5, 1.5, actual.numel(), dtype=torch.float64).reshape_as(actual)
    actual_vjp = torch.autograd.grad(actual, values, cotangent)[0]
    expected_vjp = torch.autograd.grad(expected, reference_values, cotangent)[0]

    assert executor.executor_family == route
    assert torch.allclose(actual, expected, atol=1e-8, rtol=1e-8)
    assert torch.allclose(actual_vjp, expected_vjp, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("signature", ((6, 0, 0), (4, 0, 2), (2, 2, 2)))
def test_forced_spectral_exp_identity_vjp_matches_dense_reference(signature):
    algebra = configured_algebra(
        *signature, device="cpu", dtype=torch.float64, planning_policy=_force_exp_route("spectral_local")
    )
    input_layout = algebra.layout((2,))
    output_layout = algebra.layout(range(0, algebra.n + 1, 2))
    values = torch.zeros(1, input_layout.dim, dtype=torch.float64, requires_grad=True)
    reference_values = values.detach().clone().requires_grad_(True)
    actual = algebra.bivector_exp(values, input=input_layout, output=output_layout)
    expected = bivector_exp_cpu_reference(
        algebra,
        reference_values,
        input_layout=input_layout,
        output_layout=output_layout,
    )
    cotangent = torch.linspace(0.5, 1.5, actual.numel(), dtype=torch.float64).reshape_as(actual)

    assert torch.equal(actual, expected)
    assert torch.allclose(
        torch.autograd.grad(actual, values, cotangent)[0],
        torch.autograd.grad(expected, reference_values, cotangent)[0],
        atol=1e-12,
        rtol=1e-12,
    )


from clifra.core._kernel.configuration import configured_algebra
