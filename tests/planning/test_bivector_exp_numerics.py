# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from clifra.core._kernel.configuration import configured_algebra
from clifra.core.algebra import AlgebraContext
from tests.helpers.bivector_exp_oracle import bivector_exp_cpu_reference
from tests.helpers.policy import PreferRoute

pytestmark = pytest.mark.unit
DEVICE = "cpu"


def _mps_available():
    return torch.backends.mps.is_available()


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


@pytest.mark.parametrize("signature", [(4, 0, 0), (5, 0, 0), (2, 2, 0), (3, 0, 1)])
def test_bivector_exp_closed_biquadratic_matches_cpu_reference(signature):
    algebra = configured_algebra(
        *signature, device=DEVICE, dtype=torch.float64, planning_policy=PreferRoute("bivector_exp", "closed")
    )
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

    assert executor.route == "closed"
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
    algebra = configured_algebra(
        2, 0, 2, device=DEVICE, dtype=dtype, planning_policy=PreferRoute("bivector_exp", "closed")
    )
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
    algebra = configured_algebra(
        3, 1, 0, device=DEVICE, dtype=dtype, planning_policy=PreferRoute("bivector_exp", "closed")
    )
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

    assert executor.route == "closed"
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("n", [3, 4, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("angle", [128.0, 1024.0])
def test_closed_large_elliptic_angle_has_correct_finite_vjp(n, dtype, angle):
    a = configured_algebra(n, dtype=dtype, planning_policy=PreferRoute("bivector_exp", "closed"))
    layout = a.layout((2,))
    output = a.layout(range(0, n + 1, 2))
    values = torch.zeros(1, layout.dim, dtype=dtype)
    values[..., 0] = angle
    values.requires_grad_()
    actual = a.bivector_exp(values, input=layout, output=output)
    expected = bivector_exp_cpu_reference(a, values.double(), input_layout=layout, output_layout=output)
    cotangent = torch.linspace(-1, 1, output.dim, dtype=torch.float64)
    actual_vjp = torch.autograd.grad((actual * cotangent).sum(), values)[0]
    expected_vjp = torch.autograd.grad((expected * cotangent).sum(), values)[0]
    tolerance = 2e-6 if dtype == torch.float32 else 2e-11
    torch.testing.assert_close(actual.double(), expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual_vjp, expected_vjp, atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(not _mps_available(), reason="MPS unavailable")
@pytest.mark.parametrize("signature", [(6, 0, 0), (3, 3, 0), (3, 1, 3)])
def test_small_mps_matrix_execution_and_compiled_vjp(signature):
    a = configured_algebra(
        *signature,
        device="mps",
        dtype=torch.float32,
        planning_policy=PreferRoute("bivector_exp", "left_matrix_exp"),
    )
    layout = a.layout((2,))
    output = a.layout((0, 2))
    f = a._planner.bivector_exp_executor(input_layout=layout, output_layout=output)
    assert f.route == "left_matrix_exp"
    compiled = torch.compile(f, backend="aot_eager", fullgraph=True)
    values = (torch.randn(2, layout.dim, device="mps") * 0.2).requires_grad_()
    actual = compiled(values)
    reference = bivector_exp_cpu_reference(
        a,
        values.cpu().double(),
        input_layout=layout,
        output_layout=output,
    )
    cotangent = torch.randn_like(actual)
    grad = torch.autograd.grad((actual * cotangent).sum(), values)[0]
    expected_grad = torch.autograd.grad((reference * cotangent.cpu().double()).sum(), values)[0]
    torch.testing.assert_close(actual.cpu().double(), reference, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(grad, expected_grad, atol=3e-6, rtol=3e-6)
