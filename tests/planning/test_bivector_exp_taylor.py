# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from clifra.core._kernel.configuration import configured_algebra
from clifra.core._kernel.planning.layouts import ProductRequest
from clifra.core._kernel.planning.resources import ResourceLimits
from tests.helpers.bivector_exp_oracle import bivector_exp_cpu_reference
from tests.helpers.policy import PreferRoute

pytestmark = pytest.mark.unit


def executor(signature=(7, 0, 0), dtype=torch.float64, device="cpu", grades=None):
    a = configured_algebra(
        *signature, dtype=dtype, device=device, planning_policy=PreferRoute("bivector_exp", "taylor")
    )
    l = a.layout((2,))
    out = a.layout(range(0, a.n + 1, 2) if grades is None else grades)
    return a, a._planner.bivector_exp_executor(input_layout=l, output_layout=out)


@pytest.mark.parametrize(
    "signature,dtype,norm",
    [
        ((6, 0, 0), torch.float64, 0.0),
        ((0, 7, 0), torch.float32, 1e-9),
        ((3, 3, 0), torch.float64, 0.5),
        ((3, 1, 3), torch.float64, 1.0),
        ((0, 0, 7), torch.float32, 2.0),
        ((6, 0, 0), torch.float32, 32.0),
        ((3, 3, 0), torch.float64, 128.0),
    ],
)
def test_taylor_forward_vjp_and_inverse(signature, dtype, norm):
    a, f = executor(signature, dtype)
    torch.manual_seed(91)
    b = torch.randn(2, f.input_layout.dim, dtype=dtype)
    b = (b / b.abs().sum(-1, keepdim=True) * norm).requires_grad_()
    actual = f(b)
    # float64 matrix exponential is the reference even for float32 execution.
    ref = bivector_exp_cpu_reference(a, b.double(), input_layout=f.input_layout, output_layout=f.output_layout)
    v = torch.randn_like(ref)
    grad = torch.autograd.grad((actual * v).sum(), b)[0]
    expected_grad = torch.autograd.grad((ref * v).sum(), b)[0]
    tol = 8e-5 if dtype == torch.float32 else 2e-11
    torch.testing.assert_close(actual.double(), ref, atol=tol * (1 + ref.detach().abs().max().item()), rtol=tol)
    torch.testing.assert_close(grad, expected_grad, atol=tol * (1 + expected_grad.abs().max().item()), rtol=tol)
    # Hyperbolic inverses at large norms are ill conditioned in coefficient arithmetic.
    if norm <= 2:
        request = ProductRequest.compact(
            a.spec,
            op="geometric_product",
            left_layout=f.output_layout,
            right_layout=f.output_layout,
            output_layout=f.output_layout,
            dtype=dtype,
            device="cpu",
        )
        identity = a._planner.product_executor(request).forward_compact(actual, f(-b))
        expected = torch.zeros_like(identity)
        expected[..., 0] = 1
        torch.testing.assert_close(identity, expected, atol=tol, rtol=tol)


@pytest.mark.parametrize(
    "signature,delta",
    [
        ((6, 0, 0), 0.0),
        ((7, 0, 0), 1e-10),
        ((3, 3, 0), 1e-6),
        ((2, 2, 3), 0.01),
        ((2, 2, 3), 0.0),
    ],
)
def test_taylor_coalescence_and_null_components(signature, delta):
    a, f = executor(signature)
    b = torch.zeros(1, f.input_layout.dim, dtype=torch.float64)
    for k in range(3):
        b[0, f.input_layout.basis_indices.index(3 << (2 * k))] = 0.7 + k * delta
    b.requires_grad_()

    def ref(x):
        return bivector_exp_cpu_reference(a, x, input_layout=f.input_layout, output_layout=f.output_layout)

    torch.testing.assert_close(f(b), ref(b), atol=2e-13, rtol=2e-13)
    direction = torch.randn_like(b)
    actual_jvp = torch.func.jvp(f, (b,), (direction,))[1]
    expected_jvp = torch.func.jvp(ref, (b,), (direction,))[1]
    torch.testing.assert_close(actual_jvp, expected_jvp, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize(
    "grades,norm",
    [((0,), 0.5), ((0, 2), 4.0), ((4,), 0.5), ((0, 4), 4.0), ((1, 3), 0.5), (tuple(range(8)), 4.0)],
)
def test_taylor_output_pruning_matches_selected_polynomial(grades, norm):
    a, f = executor(grades=grades)
    _, full = executor()
    b = torch.randn(2, f.input_layout.dim, dtype=torch.float64)
    b = b / b.abs().sum(-1, keepdim=True) * norm
    actual = f(b)
    expected = full(b)
    positions = {i: k for k, i in enumerate(full.output_layout.basis_indices)}
    projected = actual.new_zeros(actual.shape)
    for k, i in enumerate(f.output_layout.basis_indices):
        if i in positions:
            projected[..., k] = expected[..., positions[i]]
    torch.testing.assert_close(actual, projected, atol=2e-14, rtol=2e-14)
    ref = bivector_exp_cpu_reference(a, b, input_layout=f.input_layout, output_layout=f.output_layout)
    torch.testing.assert_close(f(b), ref, atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_taylor_device_and_compiled_backward(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    a, f = executor((3, 1, 3), torch.float32, device, grades=(0, 2))
    compiled = torch.compile(f, backend="aot_eager", fullgraph=True)
    for scale in [0.01, 0.3]:
        b = (torch.randn(2, f.input_layout.dim, device=device) * scale).requires_grad_()
        y = compiled(b)
        eager = f(b)
        torch.testing.assert_close(y, eager)
        g = torch.autograd.grad(y.sum(), b)[0]
        g2 = torch.autograd.grad(eager.sum(), b)[0]
        torch.testing.assert_close(g, g2)
        ref = bivector_exp_cpu_reference(
            a, b.detach().cpu().double(), input_layout=f.input_layout, output_layout=f.output_layout
        )
        torch.testing.assert_close(y.cpu().double(), ref, atol=2e-5, rtol=2e-5)


def test_taylor_gradcheck_and_second_derivatives():
    _, f = executor(grades=(0, 2))
    b = (torch.randn(1, f.input_layout.dim, dtype=torch.float64) * 0.1).requires_grad_()
    assert torch.autograd.gradcheck(f, (b,), fast_mode=True)
    assert torch.autograd.gradgradcheck(f, (b,), fast_mode=True)


def test_taylor_numerical_envelope_is_explicit():
    _, f = executor()
    with pytest.raises(RuntimeError, match="L1 norm"):
        f(torch.full((f.input_layout.dim,), 100000.0, dtype=torch.float64))


@pytest.mark.parametrize("delta", [0.0, 1e-12, 1e-6])
def test_taylor_finite_null_carrier_and_transverse_vjp(delta):
    a, f = executor((1, 5, 1))
    b = torch.zeros(1, f.input_layout.dim, dtype=torch.float64)
    b[..., f.input_layout.basis_indices.index(3)] = 8.0
    b[..., f.input_layout.basis_indices.index(6)] = 8.0 + delta
    b.requires_grad_()
    y = f(b)
    ref = bivector_exp_cpu_reference(a, b, input_layout=f.input_layout, output_layout=f.output_layout)
    v = torch.linspace(-1, 1, y.numel(), dtype=b.dtype).reshape_as(y)
    torch.testing.assert_close(y, ref, atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(
        torch.autograd.grad((y * v).sum(), b)[0], torch.autograd.grad((ref * v).sum(), b)[0], atol=2e-10, rtol=2e-10
    )


def test_taylor_mixed_batch_scaling_and_empty_batch():
    _, f = executor()
    b = torch.randn(3, f.input_layout.dim, dtype=torch.float64)
    b = b / b.abs().sum(-1, keepdim=True) * torch.tensor([[0.0], [0.1], [32.0]])
    torch.testing.assert_close(f(b), torch.cat([f(row[None]) for row in b]), atol=2e-13, rtol=2e-13)
    assert f(b[:0]).shape == (0, f.output_layout.dim)


@pytest.mark.parametrize("signature", [(6, 0, 0), (3, 2, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("crossing", [1, 16])
@pytest.mark.parametrize("scaled_norm", [2.0, 8.0])
def test_taylor_large_mixed_batch_matches_independent_reference_and_vjp(signature, dtype, crossing, scaled_norm):
    algebra, f = executor(signature, dtype)
    torch.manual_seed(621)
    values = torch.randn(4, 8, f.input_layout.dim, dtype=dtype)
    values = values / values.abs().sum(-1, keepdim=True)
    scales = torch.full((32, 1), 0.5, dtype=dtype)
    scales[:crossing] = scaled_norm
    values = (values.reshape(32, -1) * scales).reshape(4, 8, -1).requires_grad_()
    actual = f(values)
    reference_values = values.detach().double().requires_grad_()
    expected = bivector_exp_cpu_reference(
        algebra,
        reference_values,
        input_layout=f.input_layout,
        output_layout=f.output_layout,
    )
    direction = torch.randn_like(actual)
    actual_gradient = torch.autograd.grad((actual * direction).sum(), values)[0]
    expected_gradient = torch.autograd.grad((expected * direction.double()).sum(), reference_values)[0]
    tolerance = 8e-5 if dtype == torch.float32 else 2e-11
    torch.testing.assert_close(actual.double(), expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual_gradient.double(), expected_gradient, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("batch", [16, 24])
@pytest.mark.parametrize("grades", [(0,), (0, 2, 4, 6)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_taylor_sub_32_mixed_batch_matches_reference_and_vjp(batch, grades, dtype):
    algebra, f = executor((6, 0, 0), dtype, grades=grades)
    torch.manual_seed(190 + batch)
    values = torch.randn(batch, f.input_layout.dim, dtype=dtype)
    values = values / values.abs().sum(-1, keepdim=True)
    scales = torch.full((batch, 1), 0.5, dtype=dtype)
    scales[0] = 8.0
    values = (values * scales).requires_grad_()
    actual = f(values)
    reference_values = values.detach().double().requires_grad_()
    expected = bivector_exp_cpu_reference(
        algebra, reference_values, input_layout=f.input_layout, output_layout=f.output_layout
    )
    direction = torch.randn_like(actual)
    actual_gradient = torch.autograd.grad((actual * direction).sum(), values)[0]
    expected_gradient = torch.autograd.grad((expected * direction.double()).sum(), reference_values)[0]
    tolerance = 8e-5 if dtype == torch.float32 else 2e-11
    torch.testing.assert_close(actual.double(), expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual_gradient.double(), expected_gradient, atol=tolerance, rtol=tolerance)


def test_taylor_scalar_inductor_forward_backward_and_envelope():
    _, f = executor(dtype=torch.float32, grades=(0,))
    compiled = torch.compile(f, backend="inductor", fullgraph=True)
    for scale in (0.01, 0.3):
        values = (torch.randn(1, f.input_layout.dim) * scale).requires_grad_()
        actual, expected = compiled(values), f(values)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual.sum(), values)[0], torch.autograd.grad(expected.sum(), values)[0]
        )
    with pytest.raises(RuntimeError, match="L1 norm"):
        compiled(torch.full((1, f.input_layout.dim), 100000.0, requires_grad=True))


def test_taylor_dimension_twelve_keeps_all_six_planes():
    a = configured_algebra(
        12,
        dtype=torch.float32,
        planning_policy=PreferRoute("bivector_exp", "taylor"),
        resource_limits=ResourceLimits(max_pairs=12_000_000),
    )
    layout = a.layout((2,))
    f = a._planner.bivector_exp_executor(input_layout=layout, output_layout=a.layout(range(0, 13, 2)))
    angles = torch.linspace(0.03, 0.08, 6, requires_grad=True)
    embedding = torch.zeros(6, f.input_layout.dim)
    for plane in range(6):
        embedding[plane, f.input_layout.basis_indices.index(3 << (2 * plane))] = 1
    actual = f(angles @ embedding)
    expected = []
    for index in f.output_layout.basis_indices:
        factors = []
        for plane, angle in enumerate(angles):
            bits = (index >> (2 * plane)) & 3
            factors.append(angle.cos() if bits == 0 else angle.sin() if bits == 3 else angle * 0)
        expected.append(torch.stack(factors).prod())
    expected = torch.stack(expected)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), angles)[0],
        torch.autograd.grad(expected.sum(), angles)[0],
        atol=2e-6,
        rtol=2e-6,
    )


@pytest.mark.parametrize("grades", [(0,), tuple(range(8))])
def test_planned_taylor_dtype_movement_updates_degree_without_mutating_cached_plan(grades):
    a, _ = executor(dtype=torch.float32, grades=grades)
    layout, out = a.layout((2,)), a.layout(grades)
    moved = a.plan_bivector_exp(input=layout, output=out)
    original = a.plan_bivector_exp(input=layout, output=out)
    values = torch.zeros(1, layout.dim, dtype=torch.float64)
    values[..., 0] = 1.0
    moved.to(dtype=torch.float64)
    reference = bivector_exp_cpu_reference(a, values, input_layout=layout, output_layout=out)
    torch.testing.assert_close(moved(values), reference, atol=2e-14, rtol=2e-14)
    assert original(torch.zeros_like(values, dtype=torch.float32)).dtype == torch.float32
    moved.to(dtype=torch.float32)
    torch.testing.assert_close(moved(values.float()), original(values.float()))


def test_planned_taylor_movement_fails_explicitly_when_selected_route_is_unsupported():
    _, operation = executor(dtype=torch.float32, grades=(0, 2))
    values = torch.zeros(1, operation.input_layout.dim)
    before = operation(values)

    with pytest.raises(ValueError, match="requires float32 or float64"):
        operation.to(dtype=torch.float16)

    assert operation.route == "taylor"
    torch.testing.assert_close(operation(values), before)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_planned_matrix_movement_preserves_cpu_execution():
    a, _ = executor((6, 0, 0))
    # Use the ordinary CPU default rather than the helper's forced Taylor policy.
    a._planner.policy = PreferRoute("bivector_exp", "left_matrix_exp")
    a._planner.clear_cache()
    operation = a.plan_bivector_exp(input=a.layout((2,)), output=a.layout((0, 2)))
    values = torch.randn(1, 15) * 0.1
    expected = operation(values.double()).float()
    operation.to(device="mps", dtype=torch.float32)
    torch.testing.assert_close(operation(values.to("mps")).cpu(), expected, atol=2e-6, rtol=2e-6)
