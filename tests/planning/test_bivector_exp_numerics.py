# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from clifra.core.execution.action import BivectorVectorGeneratorExecutor
from clifra.core.execution.exp import (
    _filtered_eigenvalue_cauchy_inverse,
    _filtered_symmetric_eigh_op,
    _symmetric_eigh_diagonal_perturbation,
)
from clifra.core.runtime.algebra import AlgebraContext
from tests.helpers.bivector_exp_oracle import bivector_exp_cpu_reference

pytestmark = pytest.mark.unit

DEVICE = "cpu"


def _mps_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def test_bivector_exp_closed_simple_matches_cpu_reference_on_basis_point():
    context = AlgebraContext(3, 0, 0, device=DEVICE, dtype=torch.float64)
    bivector_layout = context.layout((2,))
    rotor_layout = context.layout((0, 2))
    bivectors = torch.zeros(1, bivector_layout.dim, dtype=torch.float64)
    e12_position = bivector_layout.basis_indices.index(3)
    bivectors[0, e12_position] = 0.25
    actual = context.bivector_exp(bivectors, input_layout=bivector_layout, output_layout=rotor_layout)
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
    executor = algebra.planner.bivector_exp_executor_for_layouts(
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


def test_bivector_exp_closed_paths_have_finite_zero_gradients():
    for signature in [(3, 0, 0), (5, 0, 0)]:
        algebra = AlgebraContext(*signature, device=DEVICE, dtype=torch.float64)
        bivector_layout = algebra.layout((2,))
        even_layout = algebra.layout(range(0, algebra.n + 1, 2))
        executor = algebra.planner.bivector_exp_executor_for_layouts(
            input_layout=bivector_layout,
            output_layout=even_layout,
            dtype=torch.float64,
            device=DEVICE,
        )
        values = torch.zeros(3, bivector_layout.dim, dtype=torch.float64, requires_grad=True)

        executor(values).sum().backward()

        assert values.grad is not None
        assert torch.isfinite(values.grad).all()


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
    tolerances = torch.tensor([0.0, 0.0, torch.finfo(torch.float64).eps**0.5], dtype=torch.float64)
    grad_eigenvalues = torch.randn(4, dtype=torch.float64, generator=torch.Generator(device=DEVICE).manual_seed(383))
    grad_eigenvectors = torch.randn(4, 4, dtype=torch.float64, generator=torch.Generator(device=DEVICE).manual_seed(389))

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
    tolerances = torch.tensor([0.0, 0.0, torch.finfo(torch.float64).eps**0.5], dtype=torch.float64)

    eigenvalues, eigenvectors = _filtered_symmetric_eigh_op(matrix, tolerances)
    (eigenvalues.sum() + eigenvectors.sum()).backward()

    assert matrix.grad is not None
    assert torch.isfinite(matrix.grad).all()


def test_filtered_symmetric_eigh_cauchy_filter_zeroes_repeated_denominators():
    eigenvalues = torch.tensor([2.0, 2.0, 5.0, 7.0], dtype=torch.float64)
    tolerances = torch.tensor([0.0, 0.0, torch.finfo(torch.float64).eps**0.5], dtype=torch.float64)

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
    algebra = AlgebraContext(*signature, device=DEVICE, dtype=torch.float64)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout(range(0, algebra.n + 1, 2))
    values = torch.randn(
        4,
        bivector_layout.dim,
        dtype=torch.float64,
        generator=torch.Generator(device=DEVICE).manual_seed(307),
    ) * 0.2
    executor = algebra.plan_bivector_exp(
        input_layout=bivector_layout,
        output_layout=even_layout,
        dtype=torch.float64,
        device=DEVICE,
        spectral_tol_abs=0.0,
        spectral_tol_rel=0.0,
        spectral_transition_n=6,
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
    values = torch.randn(
        3,
        bivector_layout.dim,
        dtype=torch.float64,
        generator=torch.Generator(device=DEVICE).manual_seed(347),
    ) * 0.1
    executor = algebra.plan_bivector_exp(
        input_layout=bivector_layout,
        output_layout=even_layout,
        dtype=torch.float64,
        device=DEVICE,
    )

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
    algebra = AlgebraContext(*signature, device=DEVICE, dtype=torch.float64)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout((0, 2, 4))
    values = torch.randn(
        3,
        bivector_layout.dim,
        dtype=torch.float64,
        generator=torch.Generator(device=DEVICE).manual_seed(359),
    ) * 0.08
    executor = algebra.plan_bivector_exp(
        input_layout=bivector_layout,
        output_layout=even_layout,
        dtype=torch.float64,
        device=DEVICE,
        spectral_tol_abs=0.0,
        spectral_tol_rel=0.0,
        spectral_transition_n=6,
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
    algebra = AlgebraContext(4, 0, 2, device=DEVICE, dtype=torch.float64)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout(range(0, algebra.n + 1, 2))
    values = torch.zeros(2, bivector_layout.dim, dtype=torch.float64)
    for axes, coefficient in [((0, 4), 0.30), ((1, 5), -0.20), ((3, 4), 0.10)]:
        values[0, bivector_layout.basis_indices.index(sum(1 << axis for axis in axes))] = coefficient
    values[1] = -0.5 * values[0]
    executor = algebra.plan_bivector_exp(
        input_layout=bivector_layout,
        output_layout=even_layout,
        dtype=torch.float64,
        device=DEVICE,
        spectral_tol_abs=0.0,
        spectral_tol_rel=0.0,
        spectral_transition_n=6,
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
    algebra = AlgebraContext(2, 0, 4, device=DEVICE, dtype=torch.float64)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout(range(0, algebra.n + 1, 2))
    values = torch.zeros(1, bivector_layout.dim, dtype=torch.float64)
    for axes, coefficient in [((2, 3), 0.40), ((4, 5), 0.60)]:
        values[0, bivector_layout.basis_indices.index(sum(1 << axis for axis in axes))] = coefficient
    executor = algebra.plan_bivector_exp(
        input_layout=bivector_layout,
        output_layout=even_layout,
        dtype=torch.float64,
        device=DEVICE,
        spectral_tol_abs=0.0,
        spectral_tol_rel=0.0,
        spectral_transition_n=6,
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
    algebra = AlgebraContext(5, 0, 1, device=DEVICE, dtype=torch.float64)
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
    executor = algebra.plan_bivector_exp(
        input_layout=bivector_layout,
        output_layout=output_layout,
        dtype=torch.float64,
        device=DEVICE,
        spectral_tol_abs=0.0,
        spectral_tol_rel=0.0,
        spectral_transition_n=6,
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
    algebra = AlgebraContext(10, 0, 1, device=DEVICE, dtype=torch.float64)
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
    executor = algebra.plan_bivector_exp(
        input_layout=bivector_layout,
        output_layout=output_layout,
        dtype=torch.float64,
        device=DEVICE,
        spectral_max_planes=8,
        spectral_tol_abs=0.0,
        spectral_tol_rel=0.0,
        spectral_transition_n=6,
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
    algebra = AlgebraContext(8, 0, 0, device=DEVICE, dtype=torch.float64)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout((0, 2))
    values = torch.zeros(1, bivector_layout.dim, dtype=torch.float64)
    planes = [((0, 1), 0.20), ((2, 3), 0.13), ((4, 5), 0.07), ((6, 7), 0.03)]
    for axes, coefficient in planes:
        values[0, bivector_layout.basis_indices.index(sum(1 << axis for axis in axes))] = coefficient
    executor = algebra.plan_bivector_exp(
        input_layout=bivector_layout,
        output_layout=even_layout,
        dtype=torch.float64,
        device=DEVICE,
        spectral_tol_abs=0.0,
        spectral_tol_rel=0.0,
        spectral_transition_n=6,
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
    algebra = AlgebraContext(10, 0, 0, device=DEVICE, dtype=torch.float64)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout((0, 2))
    values = torch.zeros(1, bivector_layout.dim, dtype=torch.float64)
    planes = [((0, 1), 0.20), ((2, 3), 0.13), ((4, 5), 0.07), ((6, 7), 0.03)]
    for axes, coefficient in planes:
        values[0, bivector_layout.basis_indices.index(sum(1 << axis for axis in axes))] = coefficient
    executor = algebra.plan_bivector_exp(
        input_layout=bivector_layout,
        output_layout=even_layout,
        dtype=torch.float64,
        device=DEVICE,
        spectral_max_planes=4,
        spectral_tol_abs=0.0,
        spectral_tol_rel=0.0,
        spectral_transition_n=6,
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
    algebra = AlgebraContext(6, 0, 0, device=DEVICE, dtype=torch.float64)
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
    executor = algebra.plan_bivector_exp(
        input_layout=bivector_layout,
        output_layout=even_layout,
        dtype=torch.float64,
        device=DEVICE,
        spectral_tol_abs=0.0,
        spectral_tol_rel=0.0,
        spectral_transition_n=6,
        spectral_dominant_rel=0.05,
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
    algebra = AlgebraContext(6, 0, 0, device=DEVICE, dtype=torch.float64)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout(range(0, algebra.n + 1, 2))
    executor = algebra.plan_bivector_exp(
        input_layout=bivector_layout,
        output_layout=even_layout,
        dtype=torch.float64,
        device=DEVICE,
        spectral_tol_abs=0.0,
        spectral_tol_rel=0.0,
        spectral_transition_n=6,
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
    algebra = AlgebraContext(6, 0, 0, device=DEVICE, dtype=torch.float64)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout(range(0, algebra.n + 1, 2))
    executor = algebra.plan_bivector_exp(
        input_layout=bivector_layout,
        output_layout=even_layout,
        dtype=torch.float64,
        device=DEVICE,
        spectral_tol_abs=0.0,
        spectral_tol_rel=0.0,
        spectral_transition_n=6,
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
    algebra = AlgebraContext(4, 0, 2, device=DEVICE, dtype=torch.float64)
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

    actual = algebra.bivector_exp(
        values,
        input_layout=bivector_layout,
        output_layout=even_layout,
        spectral_transition_n=6,
    )
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
    algebra = AlgebraContext(6, 0, 0, device=DEVICE, dtype=torch.float64)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout((0, 2))
    executor = algebra.plan_bivector_exp(
        input_layout=bivector_layout,
        output_layout=even_layout,
        dtype=torch.float64,
        device=DEVICE,
        spectral_max_planes=1,
        spectral_transition_n=6,
    )
    values = torch.randn(
        1,
        bivector_layout.dim,
        dtype=torch.float64,
        generator=torch.Generator(device=DEVICE).manual_seed(353),
    ) * 0.05
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
    algebra = AlgebraContext(6, 0, 0, device=DEVICE, dtype=torch.float64)
    bivector_layout = algebra.layout((2,))
    output_layout = algebra.layout(output_grades)
    values = torch.randn(
        3,
        bivector_layout.dim,
        dtype=torch.float64,
        generator=torch.Generator(device=DEVICE).manual_seed(337),
    ) * 0.15

    actual = algebra.bivector_exp(
        values,
        input_layout=bivector_layout,
        output_layout=output_layout,
        spectral_transition_n=6,
    )
    expected = bivector_exp_cpu_reference(
        algebra,
        values,
        input_layout=bivector_layout,
        output_layout=output_layout,
    )

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_bivector_exp_spectral_local_respects_static_plane_cap_and_tail_tolerance():
    algebra = AlgebraContext(6, 0, 0, device=DEVICE, dtype=torch.float64)
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

    capped = algebra.plan_bivector_exp(
        input_layout=bivector_layout,
        output_layout=even_layout,
        dtype=torch.float64,
        device=DEVICE,
        spectral_max_planes=2,
        spectral_tol_abs=0.0,
        spectral_tol_rel=0.0,
        spectral_transition_n=6,
    )
    tolerance_masked = algebra.plan_bivector_exp(
        input_layout=bivector_layout,
        output_layout=even_layout,
        dtype=torch.float64,
        device=DEVICE,
        spectral_tol_abs=0.08,
        spectral_tol_rel=0.0,
        spectral_transition_n=6,
        cache=False,
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
    algebra = AlgebraContext(6, 0, 0, device=DEVICE, dtype=torch.float32)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout((0, 2))
    executor = algebra.plan_bivector_exp(
        input_layout=bivector_layout,
        output_layout=even_layout,
        dtype=torch.float32,
        device=DEVICE,
        spectral_max_planes=1,
        spectral_tol_abs=0.0,
        spectral_tol_rel=0.0,
        spectral_transition_n=6,
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
    algebra = AlgebraContext(5, 0, 1, device=DEVICE, dtype=torch.float32)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout((0, 2))
    executor = algebra.plan_bivector_exp(
        input_layout=bivector_layout,
        output_layout=even_layout,
        dtype=torch.float32,
        device=DEVICE,
        spectral_max_planes=1,
        spectral_tol_abs=0.0,
        spectral_tol_rel=0.0,
        spectral_transition_n=6,
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
    executor = algebra.plan_bivector_exp(input_layout=input_layout, output_layout=output_layout, dtype=torch.float32, device="mps")
    values = torch.randn(
        3,
        input_layout.dim,
        dtype=torch.float32,
        generator=torch.Generator(device="cpu").manual_seed(293),
    ).to("mps") * 0.1

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
