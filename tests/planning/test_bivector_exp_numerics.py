# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from clifra.core._kernel.execution.action import BivectorVectorGeneratorExecutor
from clifra.core._kernel.execution.exp import (
    _filtered_eigenvalue_cauchy_inverse,
    _filtered_symmetric_eigh_op,
    _spectral_local_nilpotent_coefficients_impl,
    _spectral_local_sinc_impl,
    _symmetric_eigh_diagonal_perturbation,
)
from clifra.core._kernel.planning.policy import FormulaPolicy, Polynomial, RouteRule
from clifra.core.algebra import AlgebraContext
from tests.helpers.bivector_exp_oracle import bivector_exp_cpu_reference

pytestmark = pytest.mark.unit

DEVICE = "cpu"
FORCE_SPECTRAL_POLICY = FormulaPolicy(
    rules=(
        RouteRule("bivector_exp", "closed_simple"),
        RouteRule("bivector_exp", "closed_biquadratic"),
        RouteRule("bivector_exp", "spectral_local", score=Polynomial(constant=-1.0)),
        RouteRule("bivector_exp", "left_matrix_exp"),
        RouteRule("bivector_exp", "cpu_matrix_exp", score=Polynomial(constant=1.0)),
    )
)


def _mps_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def test_bivector_exp_closed_simple_matches_cpu_reference_on_basis_point():
    context = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float64)
    bivector_layout = context.layout((2,))
    rotor_layout = context.layout((0, 2))
    bivectors = torch.zeros(1, bivector_layout.dim, dtype=torch.float64)
    e12_position = bivector_layout.basis_indices.index(3)
    bivectors[0, e12_position] = 0.25
    actual = context.bivector_exp(bivectors, input=bivector_layout, output=rotor_layout)
    expected = bivector_exp_cpu_reference(
        context,
        bivectors,
        input_layout=bivector_layout,
        output_layout=rotor_layout,
    )

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def _nondegenerate_generator_to_bivector_map(algebra: AlgebraContext, bivector_layout) -> torch.Tensor:
    nondegenerate_dim = algebra.p + algebra.q
    output = torch.zeros(nondegenerate_dim, nondegenerate_dim, bivector_layout.dim, dtype=torch.float64)
    input_positions = {index: position for position, index in enumerate(bivector_layout.basis_indices)}
    for i in range(nondegenerate_dim):
        for j in range(i + 1, nondegenerate_dim):
            bivector_position = input_positions.get((1 << i) | (1 << j))
            if bivector_position is not None:
                output[j, i, bivector_position] = 1.0 if i < algebra.p else -1.0
    return output


@pytest.mark.parametrize("signature", [(4, 0, 0), (5, 0, 0), (2, 2, 0), (3, 0, 1)])
def test_bivector_exp_closed_biquadratic_matches_cpu_reference(signature):
    algebra = AlgebraContext(*signature, device=DEVICE, dtype=torch.float64)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout(range(0, algebra.n + 1, 2))
    generator = torch.Generator(device=DEVICE).manual_seed(283)
    values = torch.randn(5, bivector_layout.dim, dtype=torch.float64, generator=generator) * 0.25
    executor = algebra._planner.bivector_exp_executor(
        input_layout=bivector_layout,
        output_layout=even_layout,
        dtype=torch.float64,
        device=DEVICE,
    )

    actual = executor(values)
    expected = bivector_exp_cpu_reference(
        algebra,
        values,
        input_layout=bivector_layout,
        output_layout=even_layout,
    )

    assert executor.executor_family == "closed_biquadratic"
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize(
    ("dtype", "power_start", "power_stop", "atol"),
    [
        (torch.float32, 3, 20, torch.finfo(torch.float32).eps),
        (torch.float64, 23, 31, 1e-14),
    ],
)
def test_bivector_exp_closed_biquadratic_resolves_degenerate_derivative_limit(
    dtype,
    power_start,
    power_stop,
    atol,
):
    algebra = AlgebraContext(2, 0, 2, device=DEVICE, dtype=dtype)
    bivector_layout = algebra.layout((2,))
    deltas = torch.tensor([2.0**-power for power in range(power_start, power_stop)], dtype=dtype)
    values = torch.zeros(deltas.numel(), bivector_layout.dim, dtype=dtype)
    positions = {index: position for position, index in enumerate(bivector_layout.basis_indices)}
    values[:, positions[3]] = deltas
    values[:, positions[6]] = 1.0
    values[:, positions[9]] = 2.0

    actual = algebra.bivector_exp(values, input=bivector_layout, output=bivector_layout)

    delta_sq = deltas.square()
    sinhc = 1.0 - delta_sq / 6.0 + delta_sq.square() / 120.0
    sinhc_derivative = 1.0 / 6.0 - delta_sq / 60.0 + delta_sq.square() / 1680.0
    expected = torch.zeros_like(actual)
    expected[:, positions[3]] = deltas * sinhc
    expected[:, positions[6]] = sinhc
    expected[:, positions[9]] = 2.0 * sinhc
    expected[:, positions[12]] = -4.0 * deltas * sinhc_derivative

    assert torch.allclose(actual, expected, atol=atol, rtol=atol)


@pytest.mark.parametrize(
    ("dtype", "power_start", "power_stop", "atol"),
    [
        (torch.float32, 3, 20, torch.finfo(torch.float32).eps),
        (torch.float64, 23, 31, 1e-14),
    ],
)
def test_bivector_exp_closed_biquadratic_resolves_coalescing_complex_roots(
    dtype,
    power_start,
    power_stop,
    atol,
):
    algebra = AlgebraContext(3, 1, 0, device=DEVICE, dtype=dtype)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout((0, 2, 4))
    deltas = torch.tensor([2.0**-power for power in range(power_start, power_stop)], dtype=dtype)
    values = torch.zeros(deltas.numel(), bivector_layout.dim, dtype=dtype)
    input_positions = {index: position for position, index in enumerate(bivector_layout.basis_indices)}
    output_positions = {index: position for position, index in enumerate(even_layout.basis_indices)}
    values[:, input_positions[3]] = 1.0
    values[:, input_positions[12]] = deltas

    actual = algebra.bivector_exp(values, input=bivector_layout, output=even_layout)

    expected = torch.zeros_like(actual)
    expected[:, output_positions[0]] = torch.cos(torch.ones_like(deltas)) * torch.cosh(deltas)
    expected[:, output_positions[3]] = torch.sin(torch.ones_like(deltas)) * torch.cosh(deltas)
    expected[:, output_positions[12]] = torch.cos(torch.ones_like(deltas)) * torch.sinh(deltas)
    expected[:, output_positions[15]] = torch.sin(torch.ones_like(deltas)) * torch.sinh(deltas)

    assert torch.allclose(actual, expected, atol=atol, rtol=atol)


def test_bivector_exp_closed_biquadratic_coalescing_complex_vjp_matches_reference():
    algebra = AlgebraContext(1, 3, 1, device=DEVICE, dtype=torch.float64)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout((0, 2, 4))
    positions = {index: position for position, index in enumerate(bivector_layout.basis_indices)}
    raw = torch.zeros(1, bivector_layout.dim, dtype=torch.float64)
    raw[0, positions[3]] = 0.1
    raw[0, positions[12]] = 1.0850786415217417e-13
    raw[0, positions[24]] = 0.1
    values = raw.clone().requires_grad_(True)
    reference_values = raw.clone().requires_grad_(True)

    actual = algebra.bivector_exp(values, input=bivector_layout, output=even_layout)
    expected = bivector_exp_cpu_reference(
        algebra,
        reference_values,
        input_layout=bivector_layout,
        output_layout=even_layout,
    )
    cotangent = torch.linspace(0.5, 1.5, actual.numel(), dtype=torch.float64).reshape_as(actual)

    assert torch.allclose(actual, expected, atol=1e-14, rtol=1e-14)
    assert torch.allclose(
        torch.autograd.grad(actual, values, cotangent)[0],
        torch.autograd.grad(expected, reference_values, cotangent)[0],
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    ("signature", "blades"),
    [((2, 2, 0), (5, 10)), ((3, 1, 0), (3, 12))],
)
@pytest.mark.parametrize(
    "limit_name",
    ["cosh_divided_difference_limit", "sinhc_divided_difference_limit"],
)
def test_bivector_exp_divided_difference_switch_matches_reference_through_third_derivative(
    signature,
    blades,
    limit_name,
):
    algebra = AlgebraContext(*signature, device=DEVICE, dtype=torch.float64)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout((0, 2, 4))
    executor = algebra._planner.bivector_exp_executor(input_layout=bivector_layout, output_layout=even_layout)
    positions = {index: position for position, index in enumerate(bivector_layout.basis_indices)}
    base = torch.zeros(1, bivector_layout.dim, dtype=torch.float64)
    direction = torch.zeros_like(base)
    base[0, positions[blades[0]]] = 1.0
    direction[0, positions[blades[1]]] = 1.0
    cotangent = torch.linspace(0.5, 1.5, even_layout.dim, dtype=torch.float64).unsqueeze(0)
    limit = getattr(executor, limit_name)

    for factor in (1.0 - 2.0**-8, 1.0 + 2.0**-8):
        parameter = torch.tensor(0.5 * limit * factor, dtype=torch.float64, requires_grad=True)
        values = base + parameter * direction
        actual = (executor(values) * cotangent).sum()
        expected = (
            bivector_exp_cpu_reference(
                algebra,
                values,
                input_layout=bivector_layout,
                output_layout=even_layout,
            )
            * cotangent
        ).sum()

        for order, atol in enumerate((1e-13, 1e-11, 2e-9, 1e-7)):
            assert torch.allclose(actual, expected, atol=atol, rtol=atol)
            if order < 3:
                actual = torch.autograd.grad(actual, parameter, create_graph=True)[0]
                expected = torch.autograd.grad(expected, parameter, create_graph=True)[0]


@pytest.mark.parametrize(
    ("dtype", "power_start", "power_stop", "atol"),
    [
        (torch.float32, 3, 12, torch.finfo(torch.float32).eps),
        (torch.float64, 9, 26, 1e-14),
    ],
)
def test_bivector_exp_closed_biquadratic_resolves_coalescing_real_roots(
    dtype,
    power_start,
    power_stop,
    atol,
):
    algebra = AlgebraContext(4, 1, 0, device=DEVICE, dtype=dtype)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout((0, 2, 4))
    deltas = torch.tensor([2.0**-power for power in range(power_start, power_stop)], dtype=dtype)
    simple_norms = 0.5 * deltas
    null_weights = torch.sqrt(1.0 - simple_norms.square())
    values = torch.zeros(deltas.numel(), bivector_layout.dim, dtype=dtype)
    input_positions = {index: position for position, index in enumerate(bivector_layout.basis_indices)}
    output_positions = {index: position for position, index in enumerate(even_layout.basis_indices)}
    values[:, input_positions[3]] = 1.0
    values[:, input_positions[12]] = 1.0
    values[:, input_positions[20]] = null_weights

    actual = algebra.bivector_exp(values, input=bivector_layout, output=even_layout)

    simple_norms_sq = simple_norms.square()
    simple_sinc = 1.0 - simple_norms_sq / 6.0 + simple_norms_sq.square() / 120.0
    cos_one = torch.cos(torch.ones_like(deltas))
    sin_one = torch.sin(torch.ones_like(deltas))
    cos_simple = torch.cos(simple_norms)
    expected = torch.zeros_like(actual)
    expected[:, output_positions[0]] = cos_one * cos_simple
    expected[:, output_positions[3]] = sin_one * cos_simple
    expected[:, output_positions[12]] = cos_one * simple_sinc
    expected[:, output_positions[20]] = cos_one * null_weights * simple_sinc
    expected[:, output_positions[15]] = sin_one * simple_sinc
    expected[:, output_positions[23]] = sin_one * null_weights * simple_sinc

    assert torch.allclose(actual, expected, atol=atol, rtol=atol)


def test_bivector_exp_closed_paths_have_finite_zero_gradients():
    for signature in [(3, 0, 0), (5, 0, 0)]:
        algebra = AlgebraContext(*signature, device=DEVICE, dtype=torch.float64)
        bivector_layout = algebra.layout((2,))
        even_layout = algebra.layout(range(0, algebra.n + 1, 2))
        executor = algebra._planner.bivector_exp_executor(
            input_layout=bivector_layout,
            output_layout=even_layout,
            dtype=torch.float64,
            device=DEVICE,
        )
        values = torch.zeros(3, bivector_layout.dim, dtype=torch.float64, requires_grad=True)

        executor(values).sum().backward()

        assert values.grad is not None
        assert torch.isfinite(values.grad).all()


@pytest.mark.parametrize(("dtype", "atol"), [(torch.float32, 1e-6), (torch.float64, 1e-14)])
def test_spectral_local_nilpotent_coefficients_resolve_small_angle_limits(dtype, atol):
    theta = torch.tensor(
        [2.0**-power for power in range(4, 25)] + [0.10, 0.14, 0.20, 0.30, 0.50],
        dtype=dtype,
    )
    theta_sq = theta.square()
    theta_fourth = theta_sq.square()
    theta_sixth = theta_fourth * theta_sq
    theta_eighth = theta_fourth.square()
    theta_tenth = theta_eighth * theta_sq
    theta_twelfth = theta_sixth.square()
    sinc = _spectral_local_sinc_impl(theta)

    f2, g1, g2 = _spectral_local_nilpotent_coefficients_impl(theta, sinc, torch.cos(theta))

    expected_sinc = (
        1.0
        - theta_sq / 6.0
        + theta_fourth / 120.0
        - theta_sixth / 5040.0
        + theta_eighth / 362880.0
        - theta_tenth / 39916800.0
        + theta_twelfth / 6227020800.0
    )
    expected_f2 = (
        1.0 / 24.0
        - theta_sq / 240.0
        + theta_fourth / 6720.0
        - theta_sixth / 362880.0
        + theta_eighth / 31933440.0
        - theta_tenth / 4151347200.0
        + theta_twelfth / 747242496000.0
    )
    expected_g1 = (
        1.0 / 6.0
        - theta_sq / 60.0
        + theta_fourth / 1680.0
        - theta_sixth / 90720.0
        + theta_eighth / 7983360.0
        - theta_tenth / 1037836800.0
        + theta_twelfth / 186810624000.0
    )
    expected_g2 = (
        1.0 / 120.0
        - theta_sq / 1680.0
        + theta_fourth / 60480.0
        - theta_sixth / 3991680.0
        + theta_eighth / 415134720.0
        - theta_tenth / 62270208000.0
        + theta_twelfth / 12703122432000.0
    )

    assert torch.allclose(sinc, expected_sinc, atol=atol, rtol=atol)
    assert torch.allclose(f2, expected_f2, atol=atol, rtol=atol)
    assert torch.allclose(g1, expected_g1, atol=atol, rtol=atol)
    assert torch.allclose(g2, expected_g2, atol=atol, rtol=atol)


def test_filtered_symmetric_eigh_backward_matches_torch_for_distinct_spectrum():
    base = torch.tensor(
        [
            [2.0, 0.10, -0.20, 0.05],
            [0.10, 3.0, 0.15, -0.10],
            [-0.20, 0.15, 5.0, 0.20],
            [0.05, -0.10, 0.20, 8.0],
        ],
        dtype=torch.float64,
    )
    matrix = base.clone().requires_grad_(True)
    reference_matrix = base.clone().requires_grad_(True)
    tolerances = torch.tensor([0.0, 0.0, torch.finfo(torch.float64).eps ** 0.5], dtype=torch.float64)
    grad_eigenvalues = torch.randn(4, dtype=torch.float64, generator=torch.Generator(device=DEVICE).manual_seed(383))
    grad_eigenvectors = torch.randn(
        4, 4, dtype=torch.float64, generator=torch.Generator(device=DEVICE).manual_seed(389)
    )

    eigenvalues, eigenvectors = _filtered_symmetric_eigh_op(matrix, tolerances)
    reference_values, reference_vectors = torch.linalg.eigh(
        0.5 * (reference_matrix + reference_matrix.transpose(-1, -2))
    )
    (eigenvalues * grad_eigenvalues).sum().backward(retain_graph=True)
    (eigenvectors * grad_eigenvectors).sum().backward()
    (reference_values * grad_eigenvalues).sum().backward(retain_graph=True)
    (reference_vectors * grad_eigenvectors).sum().backward()

    assert torch.allclose(matrix.grad, reference_matrix.grad, atol=1e-10, rtol=1e-10)


def test_filtered_symmetric_eigh_backward_filters_repeated_roots():
    matrix = torch.diag(torch.tensor([1.0, 1.0, 2.0, 2.0], dtype=torch.float64)).requires_grad_(True)
    tolerances = torch.tensor([0.0, 0.0, torch.finfo(torch.float64).eps ** 0.5], dtype=torch.float64)

    eigenvalues, eigenvectors = _filtered_symmetric_eigh_op(matrix, tolerances)
    (eigenvalues.sum() + eigenvectors.sum()).backward()

    assert matrix.grad is not None
    assert torch.isfinite(matrix.grad).all()


def test_filtered_symmetric_eigh_cauchy_filter_zeroes_repeated_denominators():
    eigenvalues = torch.tensor([2.0, 2.0, 5.0, 7.0], dtype=torch.float64)
    tolerances = torch.tensor([0.0, 0.0, torch.finfo(torch.float64).eps ** 0.5], dtype=torch.float64)

    cauchy = _filtered_eigenvalue_cauchy_inverse(eigenvalues, tolerances)

    assert torch.isfinite(cauchy).all()
    assert cauchy[0, 1] == 0.0
    assert cauchy[1, 0] == 0.0
    assert cauchy[0, 2] != 0.0


def test_filtered_symmetric_eigh_static_perturbation_tracks_dtype_scale():
    matrix64 = torch.eye(4, dtype=torch.float64)
    matrix32 = torch.eye(4, dtype=torch.float32)

    perturb64 = _symmetric_eigh_diagonal_perturbation(matrix64).diagonal(dim1=-2, dim2=-1)
    perturb32 = _symmetric_eigh_diagonal_perturbation(matrix32).diagonal(dim1=-2, dim2=-1)

    assert torch.allclose(perturb64, -torch.flip(perturb64, dims=(-1,)))
    assert torch.allclose(perturb32, -torch.flip(perturb32, dims=(-1,)))
    assert torch.isclose(perturb64.abs().amax(), perturb64.new_tensor(torch.finfo(torch.float64).eps * 4.0))
    assert torch.isclose(perturb32.abs().amax(), perturb32.new_tensor(torch.finfo(torch.float32).eps * 4.0))


@pytest.mark.parametrize("signature", [(6, 0, 0), (0, 6, 0), (7, 0, 0)])
def test_bivector_exp_spectral_local_matches_cpu_reference_with_low_transition(signature):
    algebra = configured_algebra(*signature, device=DEVICE, dtype=torch.float64, planning_policy=FORCE_SPECTRAL_POLICY)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout(range(0, algebra.n + 1, 2))
    values = (
        torch.randn(
            4,
            bivector_layout.dim,
            dtype=torch.float64,
            generator=torch.Generator(device=DEVICE).manual_seed(307),
        )
        * 0.2
    )
    executor = algebra._planner.bivector_exp_executor(
        spectral_tol_abs=0.0, spectral_tol_rel=0.0, input_layout=bivector_layout, output_layout=even_layout
    )

    actual = executor(values)
    expected = bivector_exp_cpu_reference(
        algebra,
        values,
        input_layout=bivector_layout,
        output_layout=even_layout,
    )

    assert executor.executor_family == "spectral_local"
    assert executor.spectral_max_planes == (algebra.p + algebra.q) // 2
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_bivector_exp_meso_cpu_defaults_to_matrix_exp_reference():
    algebra = AlgebraContext(6, 0, 0, device=DEVICE, dtype=torch.float64)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout(range(0, algebra.n + 1, 2))
    values = (
        torch.randn(
            3,
            bivector_layout.dim,
            dtype=torch.float64,
            generator=torch.Generator(device=DEVICE).manual_seed(347),
        )
        * 0.1
    )
    executor = algebra._planner.bivector_exp_executor(input_layout=bivector_layout, output_layout=even_layout)

    expected = bivector_exp_cpu_reference(
        algebra,
        values,
        input_layout=bivector_layout,
        output_layout=even_layout,
    )

    assert executor.executor_family == "left_matrix_exp"
    assert torch.allclose(executor(values), expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("signature", [(4, 0, 2), (0, 4, 2), (6, 0, 2), (2, 0, 4)])
def test_bivector_exp_spectral_local_degenerate_block_matches_cpu_reference(signature):
    algebra = configured_algebra(*signature, device=DEVICE, dtype=torch.float64, planning_policy=FORCE_SPECTRAL_POLICY)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout((0, 2, 4))
    values = (
        torch.randn(
            3,
            bivector_layout.dim,
            dtype=torch.float64,
            generator=torch.Generator(device=DEVICE).manual_seed(359),
        )
        * 0.08
    )
    executor = algebra._planner.bivector_exp_executor(
        spectral_tol_abs=0.0, spectral_tol_rel=0.0, input_layout=bivector_layout, output_layout=even_layout
    )

    actual = executor(values)
    expected = bivector_exp_cpu_reference(
        algebra,
        values,
        input_layout=bivector_layout,
        output_layout=even_layout,
    )

    assert executor.executor_family == "spectral_local"
    assert executor.ideal_dim == algebra.r
    assert executor.spectral_local_axis_count == algebra.n
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_bivector_exp_spectral_local_degenerate_block_handles_pure_mixed_kernel():
    algebra = configured_algebra(4, 0, 2, device=DEVICE, dtype=torch.float64, planning_policy=FORCE_SPECTRAL_POLICY)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout(range(0, algebra.n + 1, 2))
    values = torch.zeros(2, bivector_layout.dim, dtype=torch.float64)
    for axes, coefficient in [((0, 4), 0.30), ((1, 5), -0.20), ((3, 4), 0.10)]:
        values[0, bivector_layout.basis_indices.index(sum(1 << axis for axis in axes))] = coefficient
    values[1] = -0.5 * values[0]
    executor = algebra._planner.bivector_exp_executor(
        spectral_tol_abs=0.0, spectral_tol_rel=0.0, input_layout=bivector_layout, output_layout=even_layout
    )
    expected = bivector_exp_cpu_reference(
        algebra,
        values,
        input_layout=bivector_layout,
        output_layout=even_layout,
    )

    assert executor.executor_family == "spectral_local"
    assert torch.allclose(executor(values), expected, atol=1e-12, rtol=1e-12)


def test_bivector_exp_spectral_local_degenerate_block_keeps_r4_ideal_square_term():
    algebra = configured_algebra(2, 0, 4, device=DEVICE, dtype=torch.float64, planning_policy=FORCE_SPECTRAL_POLICY)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout(range(0, algebra.n + 1, 2))
    values = torch.zeros(1, bivector_layout.dim, dtype=torch.float64)
    for axes, coefficient in [((2, 3), 0.40), ((4, 5), 0.60)]:
        values[0, bivector_layout.basis_indices.index(sum(1 << axis for axis in axes))] = coefficient
    executor = algebra._planner.bivector_exp_executor(
        spectral_tol_abs=0.0, spectral_tol_rel=0.0, input_layout=bivector_layout, output_layout=even_layout
    )
    expected = bivector_exp_cpu_reference(
        algebra,
        values,
        input_layout=bivector_layout,
        output_layout=even_layout,
    )

    assert executor.executor_family == "spectral_local"
    assert torch.allclose(executor(values), expected, atol=1e-12, rtol=1e-12)


def test_bivector_exp_spectral_local_truncates_odd_degenerate_kernel():
    algebra = configured_algebra(5, 0, 1, device=DEVICE, dtype=torch.float64, planning_policy=FORCE_SPECTRAL_POLICY)
    bivector_layout = algebra.layout((2,))
    output_layout = algebra.layout((0, 2, 4))
    values = torch.zeros(1, bivector_layout.dim, dtype=torch.float64)
    truncated = torch.zeros_like(values)
    kept_terms = [
        ((0, 1), 0.20),
        ((2, 3), 0.13),
        ((0, 5), 0.05),
        ((3, 5), -0.04),
    ]
    omitted_terms = [
        ((4, 5), 0.30),
    ]
    for axes, coefficient in kept_terms + omitted_terms:
        values[0, bivector_layout.basis_indices.index(sum(1 << axis for axis in axes))] = coefficient
    for axes, coefficient in kept_terms:
        truncated[0, bivector_layout.basis_indices.index(sum(1 << axis for axis in axes))] = coefficient
    executor = algebra._planner.bivector_exp_executor(
        spectral_tol_abs=0.0, spectral_tol_rel=0.0, input_layout=bivector_layout, output_layout=output_layout
    )
    expected = bivector_exp_cpu_reference(
        algebra,
        truncated,
        input_layout=bivector_layout,
        output_layout=output_layout,
    )

    assert executor.executor_family == "spectral_local"
    assert executor.spectral_max_planes == 2
    assert executor.spectral_local_axis_count == 5
    assert torch.allclose(executor(values), expected, atol=1e-12, rtol=1e-12)


def test_bivector_exp_spectral_local_truncates_uncovered_degenerate_rank():
    algebra = configured_algebra(10, 0, 1, device=DEVICE, dtype=torch.float64, planning_policy=FORCE_SPECTRAL_POLICY)
    bivector_layout = algebra.layout((2,))
    output_layout = algebra.layout((0, 2, 4))
    values = torch.zeros(1, bivector_layout.dim, dtype=torch.float64)
    truncated = torch.zeros_like(values)
    kept_terms = [
        ((0, 1), 0.20),
        ((2, 3), 0.13),
        ((4, 5), 0.07),
        ((6, 7), 0.03),
        ((0, 10), 0.02),
        ((7, 10), -0.04),
    ]
    omitted_terms = [
        ((8, 9), 0.005),
        ((8, 10), 0.25),
    ]
    for axes, coefficient in kept_terms + omitted_terms:
        values[0, bivector_layout.basis_indices.index(sum(1 << axis for axis in axes))] = coefficient
    for axes, coefficient in kept_terms:
        truncated[0, bivector_layout.basis_indices.index(sum(1 << axis for axis in axes))] = coefficient
    executor = algebra._planner.bivector_exp_executor(
        spectral_max_planes=8,
        spectral_tol_abs=0.0,
        spectral_tol_rel=0.0,
        input_layout=bivector_layout,
        output_layout=output_layout,
    )
    expected = bivector_exp_cpu_reference(
        algebra,
        truncated,
        input_layout=bivector_layout,
        output_layout=output_layout,
    )

    assert executor.executor_family == "spectral_local"
    assert executor.spectral_max_planes == 4
    assert executor.spectral_local_axis_count == 9
    assert torch.allclose(executor(values), expected, atol=1e-12, rtol=1e-12)


def test_bivector_exp_spectral_local_uses_cl8_kernel_for_four_planes():
    algebra = configured_algebra(8, 0, 0, device=DEVICE, dtype=torch.float64, planning_policy=FORCE_SPECTRAL_POLICY)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout((0, 2))
    values = torch.zeros(1, bivector_layout.dim, dtype=torch.float64)
    planes = [((0, 1), 0.20), ((2, 3), 0.13), ((4, 5), 0.07), ((6, 7), 0.03)]
    for axes, coefficient in planes:
        values[0, bivector_layout.basis_indices.index(sum(1 << axis for axis in axes))] = coefficient
    executor = algebra._planner.bivector_exp_executor(
        spectral_tol_abs=0.0, spectral_tol_rel=0.0, input_layout=bivector_layout, output_layout=even_layout
    )

    actual = executor(values)
    expected = bivector_exp_cpu_reference(
        algebra,
        values,
        input_layout=bivector_layout,
        output_layout=even_layout,
    )

    assert executor.executor_family == "spectral_local"
    assert executor.spectral_max_planes == 4
    assert executor.spectral_local_product_table.shape == (128, 128, 128)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_bivector_exp_spectral_local_explicit_cap_matches_when_tail_is_zero():
    algebra = configured_algebra(10, 0, 0, device=DEVICE, dtype=torch.float64, planning_policy=FORCE_SPECTRAL_POLICY)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout((0, 2))
    values = torch.zeros(1, bivector_layout.dim, dtype=torch.float64)
    planes = [((0, 1), 0.20), ((2, 3), 0.13), ((4, 5), 0.07), ((6, 7), 0.03)]
    for axes, coefficient in planes:
        values[0, bivector_layout.basis_indices.index(sum(1 << axis for axis in axes))] = coefficient
    executor = algebra._planner.bivector_exp_executor(
        spectral_max_planes=4,
        spectral_tol_abs=0.0,
        spectral_tol_rel=0.0,
        input_layout=bivector_layout,
        output_layout=even_layout,
    )
    expected = bivector_exp_cpu_reference(
        algebra,
        values,
        input_layout=bivector_layout,
        output_layout=even_layout,
    )

    assert executor.executor_family == "spectral_local"
    assert torch.allclose(executor(values), expected, atol=1e-10, rtol=1e-10)


def test_bivector_exp_spectral_local_dominant_plane_threshold_masks_small_planes():
    algebra = configured_algebra(6, 0, 0, device=DEVICE, dtype=torch.float64, planning_policy=FORCE_SPECTRAL_POLICY)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout(range(0, algebra.n + 1, 2))
    values = torch.zeros(1, bivector_layout.dim, dtype=torch.float64)
    truncated = torch.zeros_like(values)
    planes = [
        ((1 << 0) | (1 << 1), 0.20),
        ((1 << 2) | (1 << 3), 0.009),
        ((1 << 4) | (1 << 5), 0.004),
    ]
    for index, coefficient in planes:
        values[0, bivector_layout.basis_indices.index(index)] = coefficient
    truncated[0, bivector_layout.basis_indices.index(planes[0][0])] = planes[0][1]
    executor = algebra._planner.bivector_exp_executor(
        spectral_tol_abs=0.0,
        spectral_tol_rel=0.0,
        spectral_dominant_rel=0.05,
        input_layout=bivector_layout,
        output_layout=even_layout,
    )
    expected = bivector_exp_cpu_reference(
        algebra,
        truncated,
        input_layout=bivector_layout,
        output_layout=even_layout,
    )

    assert executor.spectral_dominant_rel == 0.05
    assert torch.allclose(executor(values), expected, atol=1e-12, rtol=1e-12)


def test_bivector_exp_spectral_local_handles_repeated_rotated_angles():
    algebra = configured_algebra(6, 0, 0, device=DEVICE, dtype=torch.float64, planning_policy=FORCE_SPECTRAL_POLICY)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout(range(0, algebra.n + 1, 2))
    executor = algebra._planner.bivector_exp_executor(
        spectral_tol_abs=0.0, spectral_tol_rel=0.0, input_layout=bivector_layout, output_layout=even_layout
    )
    base = torch.zeros(1, bivector_layout.dim, dtype=torch.float64)
    for index, coefficient in [
        ((1 << 0) | (1 << 1), 0.20),
        ((1 << 2) | (1 << 3), 0.20),
        ((1 << 4) | (1 << 5), 0.07),
    ]:
        base[0, bivector_layout.basis_indices.index(index)] = coefficient
    generator = BivectorVectorGeneratorExecutor(
        bivector_layout=bivector_layout,
        dtype=torch.float64,
        device=DEVICE,
    ).execute(base)
    q, _ = torch.linalg.qr(
        torch.randn(
            algebra.n,
            algebra.n,
            dtype=torch.float64,
            generator=torch.Generator(device=DEVICE).manual_seed(317),
        )
    )
    rotated_generator = q.unsqueeze(0) @ generator @ q.T.unsqueeze(0)
    generator_to_bivector = _nondegenerate_generator_to_bivector_map(algebra, bivector_layout)
    values = torch.matmul(
        rotated_generator.reshape(1, -1),
        generator_to_bivector.reshape(-1, bivector_layout.dim),
    )

    actual = executor(values)
    expected = bivector_exp_cpu_reference(
        algebra,
        values,
        input_layout=bivector_layout,
        output_layout=even_layout,
    )

    assert executor.executor_family == "spectral_local"
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_bivector_exp_spectral_local_repeated_angle_gradient_is_filtered_finite():
    algebra = configured_algebra(6, 0, 0, device=DEVICE, dtype=torch.float64, planning_policy=FORCE_SPECTRAL_POLICY)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout(range(0, algebra.n + 1, 2))
    executor = algebra._planner.bivector_exp_executor(
        spectral_tol_abs=0.0, spectral_tol_rel=0.0, input_layout=bivector_layout, output_layout=even_layout
    )
    values = torch.zeros(1, bivector_layout.dim, dtype=torch.float64)
    for index, coefficient in [
        ((1 << 0) | (1 << 1), 0.20),
        ((1 << 2) | (1 << 3), 0.20),
        ((1 << 4) | (1 << 5), 0.07),
    ]:
        values[0, bivector_layout.basis_indices.index(index)] = coefficient
    values.requires_grad_(True)
    weights = torch.randn(
        1,
        even_layout.dim,
        dtype=torch.float64,
        generator=torch.Generator(device=DEVICE).manual_seed(331),
    )

    actual = executor(values)
    expected = bivector_exp_cpu_reference(
        algebra,
        values.detach(),
        input_layout=bivector_layout,
        output_layout=even_layout,
    )
    (actual * weights).sum().backward()

    assert executor.executor_family == "spectral_local"
    assert executor.left_product is None
    assert torch.allclose(actual.detach(), expected, atol=1e-10, rtol=1e-10)
    assert torch.isfinite(values.grad).all()
    assert values.grad.abs().sum() > 0.0


def test_bivector_exp_spectral_local_degenerate_gradient_matches_cpu_reference():
    algebra = configured_algebra(4, 0, 2, device=DEVICE, dtype=torch.float64, planning_policy=FORCE_SPECTRAL_POLICY)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout(range(0, algebra.n + 1, 2))
    values = (
        torch.randn(
            1,
            bivector_layout.dim,
            dtype=torch.float64,
            generator=torch.Generator(device=DEVICE).manual_seed(367),
        )
        * 0.05
    )
    values.requires_grad_(True)
    reference_values = values.detach().clone().requires_grad_(True)
    weights = torch.randn(
        1,
        even_layout.dim,
        dtype=torch.float64,
        generator=torch.Generator(device=DEVICE).manual_seed(373),
    )

    actual = algebra.bivector_exp(values, input=bivector_layout, output=even_layout)
    expected = bivector_exp_cpu_reference(
        algebra,
        reference_values,
        input_layout=bivector_layout,
        output_layout=even_layout,
    )
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()

    assert torch.isfinite(values.grad).all()
    assert torch.allclose(values.grad, reference_values.grad, atol=1e-10, rtol=1e-10)


def test_bivector_exp_spectral_local_gradcheck_smoke():
    algebra = configured_algebra(6, 0, 0, device=DEVICE, dtype=torch.float64, planning_policy=FORCE_SPECTRAL_POLICY)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout((0, 2))
    executor = algebra._planner.bivector_exp_executor(
        spectral_max_planes=1, input_layout=bivector_layout, output_layout=even_layout
    )
    values = (
        torch.randn(
            1,
            bivector_layout.dim,
            dtype=torch.float64,
            generator=torch.Generator(device=DEVICE).manual_seed(353),
        )
        * 0.05
    )
    values.requires_grad_(True)

    assert torch.autograd.gradcheck(
        executor,
        (values,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("output_grades", [(0,), tuple(range(7))])
def test_bivector_exp_spectral_local_public_exp_matches_cpu_reference_for_output_layouts(output_grades):
    algebra = configured_algebra(6, 0, 0, device=DEVICE, dtype=torch.float64, planning_policy=FORCE_SPECTRAL_POLICY)
    bivector_layout = algebra.layout((2,))
    output_layout = algebra.layout(output_grades)
    values = (
        torch.randn(
            3,
            bivector_layout.dim,
            dtype=torch.float64,
            generator=torch.Generator(device=DEVICE).manual_seed(337),
        )
        * 0.15
    )

    actual = algebra.bivector_exp(values, input=bivector_layout, output=output_layout)
    expected = bivector_exp_cpu_reference(
        algebra,
        values,
        input_layout=bivector_layout,
        output_layout=output_layout,
    )

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_bivector_exp_spectral_local_respects_static_plane_cap_and_tail_tolerance():
    algebra = configured_algebra(6, 0, 0, device=DEVICE, dtype=torch.float64, planning_policy=FORCE_SPECTRAL_POLICY)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout(range(0, algebra.n + 1, 2))
    values = torch.zeros(1, bivector_layout.dim, dtype=torch.float64)
    truncated = torch.zeros_like(values)
    planes = [
        ((1 << 0) | (1 << 1), 0.20),
        ((1 << 2) | (1 << 3), 0.13),
        ((1 << 4) | (1 << 5), 0.07),
    ]
    for index, coefficient in planes:
        values[0, bivector_layout.basis_indices.index(index)] = coefficient
    for index, coefficient in planes[:2]:
        truncated[0, bivector_layout.basis_indices.index(index)] = coefficient

    capped = algebra._planner.bivector_exp_executor(
        spectral_max_planes=2,
        spectral_tol_abs=0.0,
        spectral_tol_rel=0.0,
        input_layout=bivector_layout,
        output_layout=even_layout,
    )
    tolerance_masked = algebra._planner.bivector_exp_executor(
        spectral_tol_abs=0.08, spectral_tol_rel=0.0, input_layout=bivector_layout, output_layout=even_layout
    )
    expected = bivector_exp_cpu_reference(
        algebra,
        truncated,
        input_layout=bivector_layout,
        output_layout=even_layout,
    )

    assert capped.spectral_max_planes == 2
    assert tolerance_masked.spectral_max_planes == 3
    assert torch.allclose(capped(values), expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(tolerance_masked(values), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_bivector_exp_spectral_local_compiles_fullgraph_with_aot_eager():
    algebra = configured_algebra(6, 0, 0, device=DEVICE, dtype=torch.float32, planning_policy=FORCE_SPECTRAL_POLICY)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout((0, 2))
    executor = algebra._planner.bivector_exp_executor(
        spectral_max_planes=1,
        spectral_tol_abs=0.0,
        spectral_tol_rel=0.0,
        input_layout=bivector_layout,
        output_layout=even_layout,
    )
    values = torch.zeros(1, bivector_layout.dim, dtype=torch.float32)
    values[0, bivector_layout.basis_indices.index((1 << 0) | (1 << 1))] = 0.10

    compiled = torch.compile(executor, backend="aot_eager", fullgraph=True)

    expected = bivector_exp_cpu_reference(
        algebra,
        values,
        input_layout=bivector_layout,
        output_layout=even_layout,
    )
    actual = compiled(values)

    assert executor.executor_family == "spectral_local"
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_bivector_exp_spectral_local_degenerate_compiles_fullgraph_with_aot_eager():
    algebra = configured_algebra(5, 0, 1, device=DEVICE, dtype=torch.float32, planning_policy=FORCE_SPECTRAL_POLICY)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout((0, 2))
    executor = algebra._planner.bivector_exp_executor(
        spectral_max_planes=1,
        spectral_tol_abs=0.0,
        spectral_tol_rel=0.0,
        input_layout=bivector_layout,
        output_layout=even_layout,
    )
    values = torch.zeros(1, bivector_layout.dim, dtype=torch.float32)
    for axes, coefficient in [((0, 1), 0.08), ((0, 5), 0.02), ((1, 5), -0.03)]:
        values[0, bivector_layout.basis_indices.index(sum(1 << axis for axis in axes))] = coefficient

    compiled = torch.compile(executor, backend="aot_eager", fullgraph=True)

    expected = bivector_exp_cpu_reference(
        algebra,
        values,
        input_layout=bivector_layout,
        output_layout=even_layout,
    )
    actual = compiled(values)

    assert executor.executor_family == "spectral_local"
    assert executor.ideal_dim == 1
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not _mps_available(), reason="MPS not available")
@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_mps_closed_biquadratic_bivector_exp_executor_compiles_fullgraph():
    algebra = AlgebraContext(5, 0, device="mps", dtype=torch.float32)
    input_layout = algebra.layout((2,))
    output_layout = algebra.layout((0, 2, 4))
    executor = algebra._planner.bivector_exp_executor(input_layout=input_layout, output_layout=output_layout)
    values = (
        torch.randn(
            3,
            input_layout.dim,
            dtype=torch.float32,
            generator=torch.Generator(device="cpu").manual_seed(293),
        ).to("mps")
        * 0.1
    )

    compiled = torch.compile(executor, backend="aot_eager", fullgraph=True)

    expected = bivector_exp_cpu_reference(
        algebra,
        values,
        input_layout=input_layout,
        output_layout=output_layout,
    )
    actual = compiled(values)

    assert executor.executor_family == "closed_biquadratic"
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


from clifra.core._kernel.configuration import configured_algebra
