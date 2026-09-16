"""Prepared kernels must remain usable after planning is disabled."""

import ast
from pathlib import Path

import pytest
import torch

from clifra.core import _kernel
from clifra.core._kernel.configuration import configured_algebra
from clifra.core._kernel.providers import BuiltinProvider
from clifra.core._kernel.routing import ExecutorRouter
from tests.helpers.policy import PreferRoute


def test_execution_has_no_routing_policy_or_provider_imports():
    root = Path(_kernel.__file__).parent
    forbidden = {"routing", "providers", "policy", "resources", "planner"}
    for path in (root / "execution").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                modules = [node.module or "", *(alias.name for alias in node.names)]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            assert not any(forbidden.intersection(module.split(".")) for module in modules), path


def test_planning_does_not_import_concrete_executors():
    root = Path(_kernel.__file__).parent
    for path in (root / "planning").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                modules = [node.module or "", *(alias.name for alias in node.names)]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            assert not any("execution" in module.split(".") for module in modules), path


@pytest.mark.parametrize(
    "family,route",
    [
        ("product", "sparse"),
        ("product", "full_table"),
        ("bivector_exp", "closed"),
        ("bivector_exp", "taylor"),
        ("bivector_exp", "left_matrix_exp"),
        ("action", "vector_matrix"),
        ("action", "rotor_product"),
        ("action", "full_action_matrix"),
    ],
)
def test_prepared_forward_and_backward_do_not_reenter_planning(monkeypatch, family, route):
    algebra = configured_algebra(4, dtype=torch.float64, planning_policy=PreferRoute(family, route))
    bivector = algebra.layout((2,))
    full = algebra.spec.full_layout()
    if family == "product":
        operation = algebra.plan_product(left=full, right=full, output=full)
        widths = (full.dim, full.dim)
    elif family == "bivector_exp":
        operation = algebra.plan_bivector_exp(input=bivector, output=full)
        widths = (bivector.dim,)
    else:
        layout = algebra.layout((1,)) if route == "vector_matrix" else full
        operation = algebra.plan_versor_action(grade=2, input=layout, output=layout, parameter=bivector)
        widths = (layout.dim, bivector.dim)
    assert operation._kernel.route == route
    values = tuple((0.1 * torch.randn(2, 2, width, dtype=torch.float64)).requires_grad_() for width in widths)
    if family == "action":
        values = (values[0], values[1][0].detach().requires_grad_())
    expected = operation(*values)
    expected_grad = torch.autograd.grad(expected.square().sum(), values)

    def fail(*args, **kwargs):
        raise AssertionError("prepared execution reentered planning")

    monkeypatch.setattr(ExecutorRouter, "select", fail)
    monkeypatch.setattr(BuiltinProvider, "assess", fail)
    monkeypatch.setattr(BuiltinProvider, "build", fail)
    algebra._planner.clear_cache()
    actual = operation(*values)
    actual_grad = torch.autograd.grad(actual.square().sum(), values)
    torch.testing.assert_close(actual, expected)
    for actual_value, expected_value in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual_value, expected_value)


def test_prepared_composed_sandwich_does_not_reenter_planning(monkeypatch):
    algebra = configured_algebra(4, dtype=torch.float64, planning_policy=PreferRoute("action", "composed_products"))
    left, inputs, right, output = (
        algebra.layout((0, 2, 4)),
        algebra.layout((1, 2)),
        algebra.layout((0, 2)),
        algebra.layout((1, 3)),
    )
    operation = algebra.plan_sandwich_action(left=left, input=inputs, right=right, output=output)
    values = tuple(
        (0.1 * torch.randn(2, width, dtype=torch.float64)).requires_grad_()
        for width in (left.dim, inputs.dim, right.dim)
    )
    expected = operation(*values)
    expected_grad = torch.autograd.grad(expected.square().sum(), values)

    def fail(*args, **kwargs):
        raise AssertionError("prepared execution reentered planning")

    monkeypatch.setattr(ExecutorRouter, "select", fail)
    monkeypatch.setattr(BuiltinProvider, "assess", fail)
    monkeypatch.setattr(BuiltinProvider, "build", fail)
    algebra._planner.clear_cache()
    actual = operation(*values)
    actual_grad = torch.autograd.grad(actual.square().sum(), values)
    torch.testing.assert_close(actual, expected)
    for actual_value, expected_value in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual_value, expected_value)
