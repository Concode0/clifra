# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Execution-local shortcuts must preserve the original tensor formulas."""

import pytest
import torch

from clifra.core._kernel.execution.metric import SignatureNormSquaredExecutor
from clifra.core._kernel.execution.permutation import PseudoscalarProductExecutor
from clifra.core._kernel.execution.product import GradeProductExecutor
from clifra.core._kernel.execution.unary import GradeUnaryExecutor
from clifra.core._kernel.planning.metric import build_signature_norm_squared_plan
from clifra.core._kernel.planning.permutation import build_pseudoscalar_product_plan
from clifra.core._kernel.planning.product import build_grade_product_plan
from clifra.core._kernel.planning.unary import UnaryRequest, build_unary_plan_from_request
from clifra.core.layout import AlgebraSpec

pytestmark = pytest.mark.unit
DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def _check(function, reference, *values):
    actual, expected = function(*values), reference(*values)
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == expected.dtype
    seed = torch.randn_like(expected)
    actual_grad = torch.autograd.grad(actual, values, seed, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, values, seed, retain_graph=True)
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("op", ["geometric_product", "wedge", "symmetric_product", "left_contraction"])
@pytest.mark.parametrize("scalar_left", [False, True])
@pytest.mark.parametrize("output_grades", [(1, 2), (1,)])
def test_scalar_product_preserves_projection_broadcast_and_gradients(device, op, scalar_left, output_grades):
    left_grades, right_grades = ((0,), (1, 2)) if scalar_left else ((1, 2), (0,))
    plan = build_grade_product_plan(
        2,
        1,
        1,
        op=op,
        left_grades=left_grades,
        right_grades=right_grades,
        output_grades=output_grades,
        device=device,
    )
    executor = GradeProductExecutor(plan)
    for canonical in (False, True):
        left_dim = plan.dim if canonical else plan.left_layout.dim
        right_dim = plan.dim if canonical else plan.right_layout.dim
        left = torch.randn(2, 1, 2 * left_dim, device=device)[..., ::2].requires_grad_()
        right = torch.randn(1, 3, 2 * right_dim, device=device)[..., ::2].requires_grad_()
        lp = plan.left_indices if canonical else plan.left_compact_positions
        rp = plan.right_indices if canonical else plan.right_compact_positions

        def reference(a, b):
            terms = a.index_select(-1, lp) * b.index_select(-1, rp) * plan.coefficients
            if plan.pair_count == 0:
                return terms.sum(-1, keepdim=True).expand(*terms.shape[:-1], plan.output_dim)
            return terms.new_zeros(*terms.shape[:-1], plan.output_dim).index_add(-1, plan.output_positions, terms)

        _check(executor.forward if canonical else executor.forward_compact, reference, left, right)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("op", ["identity", "reverse", "grade_involution", "clifford_conjugation"])
@pytest.mark.parametrize(
    "grades,output_grades", [((1,), (1,)), ((2,), (2,)), ((0, 1, 2, 3), (0, 1, 2, 3)), ((1, 2), (2,))]
)
def test_unary_specializations_preserve_storage_and_gradients(device, op, grades, output_grades):
    spec = AlgebraSpec(2, 1)
    request = UnaryRequest.compact(
        spec,
        op=op,
        input_layout=spec.layout(grades),
        output_layout=spec.layout(output_grades),
        dtype=torch.float32,
        device=device,
    )
    plan = build_unary_plan_from_request(request)
    executor = GradeUnaryExecutor(plan)
    for canonical in (False, True):
        width = spec.dim if canonical else plan.input_layout.dim
        values = torch.randn(3, width * 2, device=device)[..., ::2].requires_grad_()
        indices = plan.output_indices if canonical else plan.input_positions
        _check(
            executor.forward if canonical else executor.forward_compact,
            lambda x: x.index_select(-1, indices) * plan.signs,
            values,
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("signature", [(3, 0, 0), (6, 0, 0), (4, 1, 1)])
@pytest.mark.parametrize("grades", [None, (1, 2), (0,)])
def test_metric_and_permutation_specializations(device, signature, grades):
    spec = AlgebraSpec(*signature)
    layout = spec.full_layout() if grades is None else spec.layout(grades)
    values = torch.randn(3, layout.dim * 2, device=device)[..., ::2].requires_grad_()
    metric = SignatureNormSquaredExecutor(build_signature_norm_squared_plan(spec, input_layout=layout, device=device))
    permutation = PseudoscalarProductExecutor(build_pseudoscalar_product_plan(spec, input_layout=layout, device=device))
    _check(metric, lambda x: (x * x * metric.signs).sum(-1, keepdim=True), values)
    _check(permutation, lambda x: x.index_select(-1, permutation.input_positions) * permutation.signs, values)


def test_shortcuts_preserve_coefficient_dtype_promotion():
    spec = AlgebraSpec(3)
    layout = spec.layout((1,))
    values = torch.randn(2, 3, dtype=torch.float32, requires_grad=True)
    unary = GradeUnaryExecutor(
        build_unary_plan_from_request(
            UnaryRequest.compact(
                spec,
                op="identity",
                input_layout=layout,
                output_layout=layout,
                dtype=torch.float64,
                device="cpu",
            )
        )
    )
    metric = SignatureNormSquaredExecutor(
        build_signature_norm_squared_plan(spec, input_layout=layout, dtype=torch.float64)
    )
    permutation = PseudoscalarProductExecutor(
        build_pseudoscalar_product_plan(spec, input_layout=layout, dtype=torch.float64)
    )
    _check(unary.forward_compact, lambda x: x * unary.signs, values)
    _check(metric, lambda x: (x * x * metric.signs).sum(-1, keepdim=True), values)
    _check(permutation, lambda x: x.index_select(-1, permutation.input_positions) * permutation.signs, values)
    plan = build_grade_product_plan(3, op="geometric_product", left_grades=(0,), right_grades=(1,), dtype=torch.float64)
    product = GradeProductExecutor(plan)
    scalar = torch.randn(2, 1, requires_grad=True)
    _check(product.forward_compact, lambda a, b: a * b * plan.coefficients, scalar, values)


def test_scalar_product_compiles_with_broadcast_gradients():
    plan = build_grade_product_plan(3, op="geometric_product", left_grades=(0,), right_grades=(1, 2))
    executor = GradeProductExecutor(plan)
    compiled = torch.compile(executor.forward_compact, backend="aot_eager", fullgraph=True)
    left = torch.randn(2, 1, 1, requires_grad=True)
    right = torch.randn(1, 3, 6, requires_grad=True)
    _check(compiled, executor.forward_compact, left, right)
