# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from clifra.core._kernel.configuration import configured_algebra
from clifra.core.algebra import AlgebraContext
from tests.helpers.policy import PreferRoute

pytestmark = pytest.mark.unit


def test_blade_inverse_preserves_negative_signature_denominator():
    algebra = AlgebraContext(0, 1, 0, device="cpu", dtype=torch.float64)
    blade = torch.zeros(1, algebra.dim, dtype=torch.float64)
    blade[0, 1] = 1.0

    inverse = algebra.blade_inverse(blade)
    product = algebra.geometric_product(blade, inverse)

    assert torch.allclose(inverse[0, 1], torch.tensor(-1.0, dtype=torch.float64))
    assert torch.allclose(product[0, 0], torch.tensor(1.0, dtype=torch.float64))


def _geometry_result(algebra, name, values, blade, layout):
    if name.endswith("blade_inverse"):
        return getattr(algebra, name)(blade, input=layout)
    if name.endswith("versor_product"):
        return getattr(algebra, name)(blade, values, input=layout, versor=layout)
    keyword = "normal" if name.endswith("reflect") else "blade"
    return getattr(algebra, name)(values, blade, input=layout, **{keyword: layout})


@pytest.mark.parametrize("signature,blade", [((2, 0, 0), [0.0, 0.0]), ((1, 1, 0), [1.0, 1.0]), ((1, 0, 1), [0.0, 1.0])])
@pytest.mark.parametrize("name", ["blade_inverse", "blade_project", "blade_reject", "reflect", "versor_product"])
def test_existing_geometry_operations_remain_finite_at_singular_inputs(signature, blade, name):
    algebra = AlgebraContext(*signature, dtype=torch.float64)
    layout = algebra.layout((1,))
    blade = torch.tensor(blade, dtype=torch.float64, requires_grad=True)
    values = torch.arange(1, layout.dim + 1, dtype=torch.float64, requires_grad=True)

    result = _geometry_result(algebra, name, values, blade, layout)

    assert torch.isfinite(result).all()
    gradients = torch.autograd.grad(result.sum(), (values, blade), allow_unused=True)
    assert all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("signature,blade", [((2, 0, 0), [0.0, 0.0]), ((1, 1, 0), [1.0, 1.0]), ((1, 0, 1), [0.0, 1.0])])
@pytest.mark.parametrize(
    "name",
    [
        "strict_blade_inverse",
        "strict_blade_project",
        "strict_blade_reject",
        "strict_reflect",
        "strict_versor_product",
    ],
)
def test_strict_geometry_rejects_exact_singular_blades(signature, blade, name):
    algebra = AlgebraContext(*signature, dtype=torch.float64)
    layout = algebra.layout((1,))
    blade = torch.tensor(blade, dtype=torch.float64)
    values = torch.arange(1, layout.dim + 1, dtype=torch.float64)

    with pytest.raises(ValueError, match="undefined for a zero denominator"):
        _geometry_result(algebra, name, values, blade, layout)


@pytest.mark.parametrize("route", ["vector_matrix", "full_action_matrix"])
@pytest.mark.parametrize("signature,normal", [((1, 1, 0), [1.0, 1.0]), ((1, 0, 1), [0.0, 1.0])])
def test_existing_planned_reflection_routes_remain_finite_for_null_normals(route, signature, normal):
    algebra = configured_algebra(
        *signature,
        dtype=torch.float64,
        planning_policy=PreferRoute("action", route),
    )
    layout = algebra.layout()
    vector = algebra.layout((1,))
    action = algebra.plan_versor_action(grade=1, input=layout, parameter=vector)

    result = action(torch.ones(layout.dim, dtype=torch.float64), torch.tensor(normal, dtype=torch.float64))

    assert torch.isfinite(result).all()


def test_near_null_inverse_exposes_default_regularization_and_strict_denominator():
    algebra = AlgebraContext(1, 1, dtype=torch.float64)
    layout = algebra.layout((1,))
    blade = torch.tensor([1.0e-12, 1.0e-12 * (1.0 - 1.0e-8)], dtype=torch.float64)
    denominator = blade[0].square() - blade[1].square()

    default = algebra.blade_inverse(blade, input=layout)
    strict = algebra.strict_blade_inverse(blade, input=layout)

    torch.testing.assert_close(default, blade / algebra.eps_sq)
    torch.testing.assert_close(strict, blade / denominator)
    assert not torch.allclose(default, strict)


@pytest.mark.parametrize(
    "base_name",
    ["blade_inverse", "blade_project", "blade_reject", "reflect", "versor_product"],
)
@pytest.mark.parametrize(
    "signature,blade_coefficients",
    [((3, 0, 0), [2.0, 0.5, 0.25]), ((1, 2, 0), [2.0, 0.5, 0.25]), ((1, 1, 1), [2.0, 0.5, 0.25])],
)
def test_strict_and_default_geometry_agree_with_gradients_away_from_regularization(
    base_name, signature, blade_coefficients
):
    algebra = AlgebraContext(*signature, dtype=torch.float64)
    layout = algebra.layout((1,))
    default_values = torch.tensor([0.2, -0.4, 0.6], dtype=torch.float64, requires_grad=True)
    strict_values = default_values.detach().clone().requires_grad_()
    default_blade = torch.tensor(blade_coefficients, dtype=torch.float64, requires_grad=True)
    strict_blade = default_blade.detach().clone().requires_grad_()

    default = _geometry_result(algebra, base_name, default_values, default_blade, layout)
    strict = _geometry_result(algebra, f"strict_{base_name}", strict_values, strict_blade, layout)
    default_gradients = torch.autograd.grad(default.square().sum(), (default_values, default_blade), allow_unused=True)
    strict_gradients = torch.autograd.grad(strict.square().sum(), (strict_values, strict_blade), allow_unused=True)

    torch.testing.assert_close(strict, default)
    for actual, expected in zip(strict_gradients, default_gradients):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected)
