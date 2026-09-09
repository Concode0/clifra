# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Independent determinant references for execution-local induced actions."""

from itertools import permutations

import pytest
import torch

from clifra.core.layout import AlgebraSpec
from tests.planning._grade_plan_helpers import _planned_full_sandwich, _planned_graded_action

pytestmark = pytest.mark.unit
DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def _reference(matrix, input_layout, output_layout):
    rows = []
    for output in output_layout.basis_indices:
        coefficients = []
        for source in input_layout.basis_indices:
            if source.bit_count() != output.bit_count():
                coefficients.append(matrix.new_zeros(matrix.shape[0]))
            elif source == 0:
                coefficients.append(matrix.new_ones(matrix.shape[0]))
            else:
                rb = [bit for bit in range(input_layout.spec.n) if output & (1 << bit)]
                cb = [bit for bit in range(input_layout.spec.n) if source & (1 << bit)]
                # Leibniz is independent of the specialized cofactor formulas
                # and has well-defined derivatives at singular matrices.
                terms = []
                for permutation in permutations(range(len(rb))):
                    inversions = sum(a > b for i, a in enumerate(permutation) for b in permutation[i + 1 :])
                    term = matrix.new_ones(matrix.shape[0])
                    for i, j in enumerate(permutation):
                        term = term * matrix[:, rb[i], cb[j]]
                    terms.append((-1) ** inversions * term)
                coefficients.append(torch.stack(terms).sum(0))
        rows.append(torch.stack(coefficients, -1))
    return torch.stack(rows, -2)


@pytest.mark.parametrize(
    "device,grades,output_grades,kind",
    [
        ("cpu", (1,), (1,), "random"),
        ("cpu", (2,), (2,), "singular"),
        ("cpu", (3,), (3,), "zero"),
        ("cpu", (4,), (4,), "random"),
        ("cpu", (0, 1, 2, 3), (0, 2, 3), "random"),
        *([("mps", (2,), (2,), "singular")] if "mps" in DEVICES else []),
    ],
)
def test_induced_action_coefficients_and_gradients(device, grades, output_grades, kind):
    torch.manual_seed(73)
    spec = AlgebraSpec(4, 1, 1)
    input_layout, output_layout = spec.layout(grades), spec.layout(output_grades)
    executor = _planned_graded_action(input_layout, output_layout, device=device)
    matrix = torch.randn(1, spec.n, spec.n)
    if kind == "singular":
        matrix[:, 1] = matrix[:, 0]
    elif kind == "zero":
        matrix.zero_()
    matrix = matrix.transpose(-1, -2).to(device).requires_grad_()
    reference_matrix = matrix.detach().cpu().double().requires_grad_()
    actual = executor.coefficients(matrix)
    expected = _reference(reference_matrix, input_layout, output_layout)
    torch.testing.assert_close(actual.cpu().double(), expected, atol=2e-5, rtol=2e-5)
    seed = torch.randn_like(expected)
    (actual_grad,) = torch.autograd.grad(actual, matrix, seed.to(actual))
    (expected_grad,) = torch.autograd.grad(expected, reference_matrix, seed)
    torch.testing.assert_close(actual_grad.cpu().double(), expected_grad, atol=5e-5, rtol=2e-5)


def test_induced_grade4_action_gradcheck_and_gradgradcheck():
    spec = AlgebraSpec(5)
    layout = spec.layout((4,))
    executor = _planned_graded_action(layout, layout, dtype=torch.float64)

    matrix = (torch.eye(5, dtype=torch.float64) + 0.1 * torch.randn(5, 5, dtype=torch.float64)).requires_grad_()

    fn = executor.coefficients

    assert torch.autograd.gradcheck(fn, (matrix,))
    assert torch.autograd.gradgradcheck(fn, (matrix,))


@pytest.mark.parametrize("grades", [(1,), (2,), (1, 2)])
def test_induced_action_single_and_multi_compile(grades):
    spec = AlgebraSpec(6)
    layout = spec.layout(grades)
    executor = _planned_graded_action(layout, layout, dtype=torch.float64)
    matrices = torch.randn(2, 6, 6, dtype=torch.float64, requires_grad=True)
    values = torch.randn(3, 2, layout.dim, dtype=torch.float64, requires_grad=True)
    for values in (values, values.unsqueeze(-2)):
        method = executor.execute
        compiled = torch.compile(method, fullgraph=True, backend="aot_eager")
        expected = method(values, matrices)
        actual = compiled(values, matrices)
        torch.testing.assert_close(actual, expected)
        seed = torch.randn_like(expected)
        actual_grad = torch.autograd.grad(actual, (values, matrices), seed)
        expected_grad = torch.autograd.grad(expected, (values, matrices), seed)
        for actual, expected in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, expected)


def test_induced_action_explicit_minors_gradcheck():
    layout = AlgebraSpec(6).layout((2, 3))
    executor = _planned_graded_action(layout, layout, dtype=torch.float64)
    matrix = torch.randn(1, 6, 6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(executor.coefficients, (matrix,), fast_mode=True)


@pytest.mark.parametrize("n,dtype", [(3, torch.float32), (6, torch.float64), (8, torch.float32)])
def test_full_sandwich_flat_gather_preserves_matrices_and_gradients(n, dtype):
    spec = AlgebraSpec(n - 2, 1, 1)
    executor = _planned_full_sandwich(spec.full_layout(), dtype=dtype)
    left = torch.randn(2, spec.dim * 2, dtype=dtype)[:, ::2].requires_grad_()
    right = torch.randn(2, spec.dim * 2, dtype=dtype)[:, ::2].requires_grad_()

    def reference(a, b):
        la = a[:, executor.cayley_indices].transpose(-1, -2) * executor.left_sign_t
        ra = b[:, executor.cayley_indices].transpose(-1, -2) * executor.geometric_product_sign_t
        return torch.bmm(ra, la)

    tolerance = 2e-5 if dtype == torch.float32 else 1e-10
    for fn in (
        executor.action_matrices_unchecked,
        torch.compile(executor.action_matrices_unchecked, fullgraph=True, backend="aot_eager"),
    ):
        actual, expected = fn(left, right), reference(left, right)
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        seed = torch.randn_like(expected)
        actual_grad = torch.autograd.grad(actual, (left, right), seed)
        expected_grad = torch.autograd.grad(expected, (left, right), seed)
        if dtype == torch.float32:
            # Parallel gather reductions use different summation orders. Check
            # both implementations against double precision instead of treating
            # the original float32 cancellation error as the exact reference.
            doubles = tuple(value.detach().double().requires_grad_() for value in (left, right))
            precise_grad = torch.autograd.grad(reference(*doubles), doubles, seed.double())
            for actual, expected, precise in zip(actual_grad, expected_grad, precise_grad):
                bound = 1e-6 * precise.norm().clamp_min(1.0)
                assert (actual.double() - precise).norm() <= bound
                assert (expected.double() - precise).norm() <= bound
        else:
            for actual, expected in zip(actual_grad, expected_grad):
                torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
