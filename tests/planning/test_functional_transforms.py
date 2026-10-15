"""Public plans preserve mathematical values and derivatives under torch.func."""

from itertools import permutations

import pytest
import torch

from clifra import make_algebra


def _check_transforms(call, reference, point, batch):
    def scalar(value):
        return call(value).sum()

    def reference_scalar(value):
        return reference(value).sum()

    jacobian = torch.autograd.functional.jacobian(reference, point)
    hessian = torch.autograd.functional.hessian(reference_scalar, point)
    torch.testing.assert_close(torch.func.grad(scalar)(point), jacobian.sum(0))
    torch.testing.assert_close(torch.func.jacrev(call)(point), jacobian)
    torch.testing.assert_close(torch.func.jacfwd(call)(point), jacobian)
    direction = torch.linspace(-0.3, 0.5, point.numel(), dtype=point.dtype).reshape_as(point)
    result, derivative = torch.func.jvp(call, (point,), (direction,))
    torch.testing.assert_close(result, reference(point))
    torch.testing.assert_close(derivative, jacobian.flatten(1) @ direction.flatten())
    torch.testing.assert_close(torch.func.hessian(scalar)(point), hessian)
    hvp = torch.func.jvp(torch.func.grad(scalar), (point,), (direction,))[1]
    torch.testing.assert_close(hvp.flatten(), hessian.reshape(point.numel(), -1) @ direction.flatten())
    torch.testing.assert_close(torch.vmap(torch.vmap(call))(batch), reference(batch))
    expected_gradients = torch.stack(
        [torch.autograd.functional.jacobian(reference_scalar, value) for value in batch.flatten(0, 1)]
    ).reshape_as(batch)
    torch.testing.assert_close(torch.vmap(torch.vmap(torch.func.grad(scalar)))(batch), expected_gradients)


@pytest.mark.parametrize("n", [4, 5])
def test_compound_action_transforms_at_singular_maps(n):
    algebra = make_algebra(n, dtype=torch.float64)
    full, top = algebra.layout(), algebra.layout((n,))
    operation = algebra.plan_linear_action(input=full, output=top)
    values = torch.linspace(0.1, 0.7, full.dim, dtype=torch.float64)
    orders = tuple(permutations(range(n)))
    indices = torch.tensor(orders)
    signs = torch.tensor(
        [(-1) ** sum(a > b for i, a in enumerate(order) for b in order[i + 1 :]) for order in orders],
        dtype=torch.float64,
    )

    def call(matrix):
        return operation(values, matrix)

    def reference(matrix):
        # Independent Leibniz polynomial, including derivatives at singular maps.
        terms = signs * matrix[..., 0, indices[:, 0]]
        for row in range(1, n):
            terms = terms * matrix[..., row, indices[:, row]]
        return terms.sum(-1, keepdim=True) * values[-1]

    regular = torch.eye(n, dtype=torch.float64) + torch.arange(n * n, dtype=torch.float64).reshape(n, n) / (10 * n * n)
    singular = regular.clone()
    singular[-1] = singular[0]
    batch = torch.stack((regular, singular)).unsqueeze(0)
    _check_transforms(call, reference, singular, batch)


@pytest.mark.parametrize("grades", [(0,), (0, 2, 4, 6)])
def test_taylor_transforms_match_commuting_plane_exponentials(grades):
    algebra = make_algebra(6, dtype=torch.float64)
    bivector, output = algebra.layout((2,)), algebra.layout(grades)
    operation = algebra.plan_bivector_exp(input=bivector, output=output)
    embedding = torch.zeros(3, bivector.dim, dtype=torch.float64)
    for plane in range(3):
        embedding[plane, bivector.basis_indices.index(3 << (2 * plane))] = 1

    def call(angles):
        return operation(angles @ embedding)

    def reference(angles):
        lanes = []
        for index in output.basis_indices:
            result = torch.ones_like(angles[..., 0])
            for plane in range(3):
                bits = (index >> (2 * plane)) & 3
                factor = angles[..., plane].cos() if bits == 0 else angles[..., plane].sin() if bits == 3 else 0
                result = result * factor
            lanes.append(result)
        return torch.stack(lanes, dim=-1)

    batch = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.1, -0.2, 0.05]], [[2.0, 1.0, -0.25], [-3.0, 0.4, 1.0]]], dtype=torch.float64
    )
    _check_transforms(call, reference, batch[1, 0], batch)
    with pytest.raises(RuntimeError, match="L1 norm"):
        invalid = batch.clone()
        invalid[0, 1, 0] = 100000.0
        torch.vmap(torch.vmap(call))(invalid)


@pytest.mark.parametrize("family", ["taylor", "compound"])
def test_mapped_operations_compile_with_gradients(family):
    if family == "taylor":
        algebra = make_algebra(6, dtype=torch.float64)
        operation = algebra.plan_bivector_exp(input=algebra.layout((2,)), output=algebra.layout((0,)))
        values = torch.zeros(2, 15, dtype=torch.float64)
        values[:, 0] = torch.tensor([0.1, 2.0])

        def call(value):
            return operation(value)
    else:
        algebra = make_algebra(4, dtype=torch.float64)
        operation = algebra.plan_linear_action(input=algebra.layout(), output=algebra.layout((4,)))
        operand = torch.ones(16, dtype=torch.float64)
        values = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        values[0, -1] = 0

        def call(matrix):
            return operation(operand, matrix)

    values.requires_grad_()
    mapped = torch.vmap(call)
    compiled = torch.compile(mapped, backend="aot_eager", fullgraph=True)
    actual, expected = compiled(values), mapped(values)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), values)[0], torch.autograd.grad(expected.sum(), values)[0]
    )


def test_mapped_large_finite_hyperbolic_exponential_keeps_finite_derivatives():
    algebra = make_algebra(1, 5, dtype=torch.float64)
    bivector = algebra.layout((2,))
    operation = algebra.plan_bivector_exp(input=bivector, output=algebra.layout((0,)))
    embedding = torch.zeros(bivector.dim, dtype=torch.float64)
    embedding[0] = 1

    def scalar(angle):
        return operation(angle * embedding).squeeze(-1)

    angles = torch.tensor([0.1, 400.0], dtype=torch.float64)
    torch.testing.assert_close(torch.vmap(scalar)(angles), angles.cosh(), atol=0, rtol=2e-12)
    torch.testing.assert_close(torch.vmap(torch.func.grad(scalar))(angles), angles.sinh(), atol=0, rtol=2e-12)
