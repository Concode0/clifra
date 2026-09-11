"""Public action regressions for the temporary singleton matrix exponential fix."""

import pytest
import torch

from clifra.core._kernel.configuration import configured_algebra
from clifra.core._kernel.numerics import _matrix_exp_singleton_workaround
from tests.helpers.policy import PreferRoute

PLACEMENTS = [("cpu", torch.float32), ("cpu", torch.float64)] + (
    [("mps", torch.float32)] if torch.backends.mps.is_available() else []
)


@pytest.mark.parametrize("device,dtype", PLACEMENTS)
@pytest.mark.parametrize("leading", [(), (1,), (1, 1), (2,), (2, 1)])
def test_single_plane_action_output_and_vjp(device, dtype, leading):
    algebra = configured_algebra(2, device=device, dtype=dtype, planning_policy=PreferRoute("action", "vector_matrix"))
    vector, bivector = algebra.layout((1,)), algebra.layout((2,))
    operation = algebra.plan_versor_action(grade=2, input=vector, parameter=bivector, output=vector)
    values = torch.tensor([0.5, 1.0], device=device, dtype=dtype).expand(*leading, 2).clone().requires_grad_()
    angle = 0.04 if dtype == torch.float64 else 0.5
    weights = torch.full((*leading, 1), angle, device=device, dtype=dtype, requires_grad=True)
    x, b = (v.detach().cpu().double().requires_grad_() for v in (values, weights))
    c, s = b[..., 0].cos(), b[..., 0].sin()
    expected = torch.stack((c * x[..., 0] - s * x[..., 1], s * x[..., 0] + c * x[..., 1]), -1)
    rotor_algebra = configured_algebra(2, dtype=torch.float64, planning_policy=PreferRoute("action", "rotor_product"))
    rotor = rotor_algebra.plan_versor_action(
        grade=2, input=rotor_algebra.layout((1,)), parameter=rotor_algebra.layout((2,))
    )
    torch.testing.assert_close(rotor(x, b), expected, atol=1e-12, rtol=1e-10)
    actual = operation(values, weights)
    atol, rtol = (1e-12, 1e-10) if dtype == torch.float64 else (1e-5, 1e-4)
    torch.testing.assert_close(actual.cpu().double(), expected, atol=atol, rtol=rtol)
    seed = torch.tensor([0.5, -1.0], device=device, dtype=dtype).expand_as(actual)
    got = torch.autograd.grad(actual, (values, weights), seed)
    want = torch.autograd.grad(expected, (x, b), seed.cpu().double())
    for a, e in zip(got, want):
        torch.testing.assert_close(a.cpu().double(), e, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "shape,duplicated",
    [
        ((2, 2), True),
        ((1, 1, 2, 2), True),
        ((2, 2, 2), False),
        ((2, 1, 2, 2), False),
        ((0, 2, 2), False),
        ((1, 1), False),
    ],
)
def test_only_singleton_matrices_are_duplicated(monkeypatch, shape, duplicated):
    original = torch.matrix_exp
    seen = []

    def record(matrix):
        seen.append(matrix)
        return original(matrix)

    matrix = torch.zeros(shape)
    monkeypatch.setattr(torch, "matrix_exp", record)
    result = _matrix_exp_singleton_workaround(matrix)
    assert result.shape == matrix.shape
    assert len(seen) == 1
    if duplicated:
        assert seen[0].shape == (2, 2, 2)
        torch.testing.assert_close(seen[0][0], seen[0][1])
    else:
        assert seen[0] is matrix


@pytest.mark.parametrize("device,dtype", PLACEMENTS)
def test_singleton_action_compiles_with_gradients(device, dtype):
    algebra = configured_algebra(2, device=device, dtype=dtype, planning_policy=PreferRoute("action", "vector_matrix"))
    operation = algebra.plan_versor_action(grade=2, input=algebra.layout((1,)), parameter=algebra.layout((2,)))
    compiled = torch.compile(operation, backend="aot_eager", fullgraph=True)
    x = torch.tensor([0.5, 1.0], device=device, dtype=dtype, requires_grad=True)
    b = torch.tensor([0.04 if dtype == torch.float64 else 0.5], device=device, dtype=dtype, requires_grad=True)
    expected, actual = operation(x, b), compiled(x, b)
    torch.testing.assert_close(actual, expected)
    for a, e in zip(torch.autograd.grad(actual.sum(), (x, b)), torch.autograd.grad(expected.sum(), (x, b))):
        torch.testing.assert_close(a, e)


def test_materialized_matrix_exp_singleton_matches_closed_route():
    def operation(route):
        a = configured_algebra(2, dtype=torch.float64, planning_policy=PreferRoute("bivector_exp", route))
        return a.plan_bivector_exp(input=a.layout((2,)), output=a.layout((0, 2)))

    matrix, closed = operation("left_matrix_exp"), operation("closed")
    b = torch.tensor([0.04], dtype=torch.float64, requires_grad=True)
    actual, expected = matrix(b), closed(b)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), b)[0], torch.autograd.grad(expected.sum(), b)[0], atol=1e-12, rtol=1e-10
    )
