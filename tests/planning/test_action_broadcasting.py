"""Mathematical action contracts, independent references and allocation limits."""

import pytest
import torch

from clifra.core import AlgebraContext, TensorContract
from clifra.core._kernel.configuration import configured_algebra
from clifra.core._kernel.planning.policy import NoAvailableRouteError, PolicyEvaluation
from clifra.core._kernel.planning.resources import ResourceLimits
from clifra.core._kernel.providers import BuiltinProvider, action_execution_request
from tests.helpers.bivector_exp_oracle import bivector_exp_cpu_reference
from tests.helpers.policy import PreferRoute
from tests.helpers.small_oracle import SmallCliffordOracle
from tests.planning._grade_plan_helpers import _planned_full_sandwich

DEVICES = (
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []) + (["mps"] if torch.backends.mps.is_available() else [])
)


@pytest.mark.parametrize(
    "signature,grades,output,zero",
    [
        ((4, 0, 0), (0,), (0,), True),
        ((0, 4, 0), (1,), (1,), False),
        ((2, 2, 0), (2,), (2,), False),
        ((1, 1, 2), (3,), (3,), False),
        ((0, 0, 4), (4,), (4,), False),
        ((2, 1, 1), (0, 1, 2, 3, 4), (0, 2, 4), False),
    ],
)
def test_direct_induced_action_matches_rotor_products_and_gradients(signature, grades, output, zero):
    algebra = configured_algebra(
        *signature, dtype=torch.float64, planning_policy=PreferRoute("action", "vector_matrix")
    )
    inputs, outputs, parameters = algebra.layout(grades), algebra.layout(output), algebra.layout((2,))
    action = algebra.plan_versor_action(grade=2, input=inputs, output=outputs, parameter=parameters)
    assert action._kernel.route == "vector_matrix"
    assert action._kernel.rotor_layout is None
    assert action._kernel.bivector_exp is None
    values = torch.randn(2, 1, inputs.dim, dtype=torch.float64, requires_grad=True)
    weights = ((0 if zero else 0.15) * torch.randn(3, parameters.dim, dtype=torch.float64)).requires_grad_()
    oracle = SmallCliffordOracle(*signature)
    full = algebra.spec.full_layout()
    rotor = algebra.bivector_exp(-0.5 * weights, input=parameters, output=full)
    expected = outputs.compact(oracle.product(oracle.product(rotor, inputs.full(values)), oracle.reverse(rotor)))
    actual = action(values, weights)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    seed = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, (values, weights), seed)
    expected_grad = torch.autograd.grad(expected, (values, weights), seed)
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("route", ["vector_matrix", "rotor_product", "full_action_matrix"])
def test_versor_broadcasting_compiles_with_gradients(device, route):
    torch.manual_seed(81)
    dtype = torch.float32 if device == "mps" else torch.float64
    algebra = configured_algebra(2, 1, 1, dtype=dtype, device=device, planning_policy=PreferRoute("action", route))
    layout, parameters = algebra.layout(), algebra.layout((2,))
    action = algebra.plan_versor_action(grade=2, input=layout, parameter=parameters)
    assert action._kernel.route == route
    for batch_values, batch_weights in [((), ()), ((2, 1, 1), (3, 1)), ((2, 1), (3,))]:
        values = torch.randn(*batch_values, layout.dim, device=device, dtype=dtype, requires_grad=True)
        weights = (0.1 * torch.randn(*batch_weights, parameters.dim, device=device, dtype=dtype)).requires_grad_()
        expected = action(values, weights)
        actual = torch.compile(action, backend="aot_eager", fullgraph=True)(values, weights)
        torch.testing.assert_close(actual, expected)
        seed = torch.randn_like(actual)
        actual_grad = torch.autograd.grad(actual, (values, weights), seed)
        expected_grad = torch.autograd.grad(expected, (values, weights), seed)
        for a, b in zip(actual_grad, expected_grad):
            torch.testing.assert_close(a, b)
        cpu_values = values.detach().cpu().double().requires_grad_()
        cpu_weights = weights.detach().cpu().double().requires_grad_()
        rotor = bivector_exp_cpu_reference(algebra, -0.5 * cpu_weights, input_layout=parameters, output_layout=layout)
        oracle = SmallCliffordOracle(2, 1, 1)
        reference = oracle.product(oracle.product(rotor, cpu_values), oracle.reverse(rotor))
        torch.testing.assert_close(expected.cpu().double(), reference, atol=2e-5, rtol=2e-5)
        reference_grad = torch.autograd.grad(reference, (cpu_values, cpu_weights), seed.cpu().double())
        for actual, reference in zip(actual_grad, reference_grad):
            torch.testing.assert_close(actual.cpu().double(), reference, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("canonical", [False, True])
def test_linear_action_arbitrary_batches_and_explicit_storage(canonical):
    algebra = AlgebraContext(2, 1, 1, dtype=torch.float64)
    layout = algebra.layout((0, 2, 4))
    contract = TensorContract.canonical(layout) if canonical else TensorContract.compact(layout)
    action = algebra.plan_linear_action(input=contract, output=contract)
    values = torch.randn(2, 1, layout.dim, dtype=torch.float64)
    matrix = torch.randn(3, 4, 4, dtype=torch.float64)
    actual = action(layout.full(values) if canonical else values, matrix)
    single = algebra.plan_linear_action(input=layout)
    expected = torch.stack([torch.stack([single(v[0], m) for m in matrix]) for v in values])
    torch.testing.assert_close(actual, layout.full(expected) if canonical else expected)
    with pytest.raises(ValueError, match="matrix trailing shape"):
        single(values, torch.randn(3, 4, 3))


def test_weighted_action_and_independent_rotors_are_explicit_compositions():
    algebra = AlgebraContext(2, 1, 1, dtype=torch.float64)
    layout, parameters, even = algebra.layout((1, 2)), algebra.layout((2,)), algebra.layout((0, 2, 4))
    action = algebra.plan_versor_action(grade=2, input=layout, parameter=algebra.layout((2,)))
    exp = algebra.plan_bivector_exp(input=parameters, output=even)
    reverse = algebra.plan_unary(op="reverse", input=even)
    sandwich = algebra.plan_sandwich_action(left=even, input=layout, right=even, output=layout)

    def composed(values, weights, other, mix, indices):
        transformed = action(values.unsqueeze(-2), weights)
        weighted = torch.einsum("ck,...cko->...co", mix, transformed)
        left, right = exp(-0.5 * weights), reverse(exp(-0.5 * other))
        return weighted + sandwich(left[indices], values, right[indices])

    values = torch.randn(2, 3, layout.dim, dtype=torch.float64, requires_grad=True)
    weights = (0.1 * torch.randn(4, parameters.dim, dtype=torch.float64)).requires_grad_()
    other = torch.randn_like(weights, requires_grad=True) * 0.1
    mix = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    indices = torch.tensor([0, 2, 0])
    args = values, weights, other, mix
    expected = composed(*args, indices)
    actual = torch.compile(composed, backend="aot_eager", fullgraph=True)(*args, indices)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(torch.autograd.grad(actual.sum(), args), torch.autograd.grad(expected.sum(), args)):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("route", ["vector_matrix", "full_action_matrix"])
def test_reflection_preserves_scalars_and_induces_each_grade(route):
    algebra = configured_algebra(2, 1, 1, dtype=torch.float64, planning_policy=PreferRoute("action", route))
    layout = algebra.layout()
    action = algebra.plan_versor_action(grade=1, input=layout, parameter=algebra.layout((1,)))
    values = torch.randn(layout.dim, dtype=torch.float64, requires_grad=True)
    normal = torch.tensor([1.0, 0.2, 0.1, 0.3], dtype=torch.float64, requires_grad=True)
    oracle = SmallCliffordOracle(2, 1, 1)
    vector = algebra.layout((1,)).full(normal)
    involuted = values * torch.tensor([(-1) ** i.bit_count() for i in range(algebra.dim)])
    norm = (normal.square() * normal.new_tensor([1, 1, -1, 0])).sum()
    expected = oracle.product(oracle.product(vector, involuted), vector / norm)
    actual = action(values, normal)
    torch.testing.assert_close(actual, expected)
    seed = torch.randn_like(actual)
    for a, b in zip(
        torch.autograd.grad(actual, (values, normal), seed), torch.autograd.grad(expected, (values, normal), seed)
    ):
        torch.testing.assert_close(a, b)


def test_full_sandwich_broadcasts_independent_factor_shapes():
    algebra = AlgebraContext(1, 1, 1, dtype=torch.float64)
    executor = _planned_full_sandwich(algebra.layout(), dtype=torch.float64)
    left, values, right = (torch.randn(*shape, algebra.dim, dtype=torch.float64) for shape in [(2, 1), (), (3,)])
    oracle = SmallCliffordOracle(1, 1, 1)
    torch.testing.assert_close(executor(left, values, right), oracle.product(oracle.product(left, values), right))


class _DirectOnly:
    def evaluate(self, candidate):
        return PolicyEvaluation(0 if candidate.route in {"vector_matrix", "graded_linear"} else None)


@pytest.mark.parametrize("operation", ["linear", "versor"])
def test_minor_resource_limit_rejects_before_allocating_indices(monkeypatch, operation):
    algebra = configured_algebra(8, resource_limits=ResourceLimits(max_pairs=10000), planning_policy=_DirectOnly())
    layout = algebra.layout((4,))  # 4900 coefficients, but 78400 determinant entries.

    def fail(*args, **kwargs):
        raise AssertionError("resource rejection must precede index allocation")

    monkeypatch.setattr("clifra.core._kernel.planning.action._graded_action_plan_tensors", fail)
    with pytest.raises(NoAvailableRouteError, match="max_pairs"):
        if operation == "linear":
            algebra.plan_linear_action(input=layout)
        else:
            algebra.plan_versor_action(grade=2, input=layout, parameter=algebra.layout((2,)))


def test_direct_route_works_when_rotor_layout_exceeds_lane_limit():
    algebra = configured_algebra(10, resource_limits=ResourceLimits(max_lanes=100))
    layout = algebra.layout((0, 1, 2))
    action = algebra.plan_versor_action(grade=2, input=layout, parameter=algebra.layout((2,)))
    assert action._kernel.route == "vector_matrix"
    values = torch.randn(layout.dim)
    torch.testing.assert_close(action(values, torch.zeros(45)), values)


def test_capability_accepts_mixed_layout_independently_of_policy():
    algebra = AlgebraContext(3, 1, 1)
    layout = algebra.layout((0, 2, 3, 5))
    request = action_execution_request(
        algebra, "versor", grade=2, input_layout=layout, parameter_layout=algebra.layout((2,))
    )
    assessment = BuiltinProvider(("action", "vector_matrix")).assess(request)
    assert assessment.preparation is not None
    assert assessment.pairs >= 10 * 10 * 3 * 3


def test_high_dimensional_action_assessment_uses_static_layout_facts(monkeypatch):
    algebra = AlgebraContext(24, 0, 1, dtype=torch.float64)
    vector, bivector = algebra.layout((1,)), algebra.layout((2,))

    def fail(*args, **kwargs):
        raise AssertionError("assessment must not materialize basis tensors")

    monkeypatch.setattr("clifra.core.layout.basis_index_tuple_for_grades", fail)
    request = action_execution_request(
        algebra,
        "versor",
        grade=2,
        input_layout=vector,
        output_layout=vector,
        parameter_layout=bivector,
    )
    selection = algebra._planner.router.select(request, algebra._planner.policy, algebra._planner.limits)
    assert selection.route == "vector_matrix"


def test_action_default_policy_is_structural_and_device_independent():
    algebra = AlgebraContext(4)
    layout = algebra.layout((0, 2, 4))
    request = action_execution_request(
        algebra,
        "versor",
        grade=2,
        input_layout=layout,
        output_layout=layout,
        parameter_layout=algebra.layout((2,)),
    )
    from dataclasses import replace

    routes = {
        algebra._planner.router.select(
            replace(request, device=torch.device(device)), algebra._planner.policy, algebra._planner.limits
        ).route
        for device in ("cpu", "mps", "cuda")
    }
    assert len(routes) == 1


@pytest.mark.parametrize("role", ["input_layout", "output_layout", "parameter_layout"])
def test_action_rejects_foreign_layouts(role):
    algebra, foreign = AlgebraContext(3), AlgebraContext(0, 3)
    layouts = {
        "input_layout": algebra.layout((1,)),
        "output_layout": algebra.layout((1,)),
        "parameter_layout": algebra.layout((2,)),
    }
    layouts[role] = foreign.layout((2,) if role == "parameter_layout" else (1,))
    with pytest.raises(ValueError, match="signature"):
        algebra.plan_versor_action(
            grade=2,
            input=layouts["input_layout"],
            output=layouts["output_layout"],
            parameter=layouts["parameter_layout"],
        )
