import pytest
import torch
from correctness_reference import (
    dense_reference_product,
    qualify_case,
    qualify_product,
    reference_product,
    reference_product_with_gradients,
)

from benchmarks.adapters import construct_case, plan_case
from benchmarks.cases import BenchmarkCase, ExecutionPlacement, LayoutCase, OrdinaryTensorCase, smoke_cases


def test_independent_oracle_qualifies_broadcast_forward_and_gradients():
    case = BenchmarkCase(
        case_id="product.correctness.broadcast",
        operation="geometric_product",
        signature=(2, 1, 0),
        inputs=(LayoutCase((0, 1, 2, 3), leading_shape=(2, 1)), LayoutCase((0, 1, 2, 3), leading_shape=(1, 3))),
        output=LayoutCase((0, 1, 2, 3)),
    )
    placement = ExecutionPlacement("float64")
    algebra, inputs, output, values = construct_case(case, placement)
    operation = plan_case(algebra, inputs, output, case)

    qualification = qualify_product(operation, values, case, placement, check_gradients=True)

    assert qualification["status"] == "passed"
    assert qualification["gradient_checked"] is True


def test_oracle_respects_canonical_storage_contracts():
    case = BenchmarkCase(
        case_id="product.correctness.canonical",
        operation="wedge",
        signature=(3, 0, 0),
        inputs=(LayoutCase((1,), "canonical", (2,)), LayoutCase((1,), "canonical", (2,))),
        output=LayoutCase((2,), "canonical"),
    )
    placement = ExecutionPlacement("float64")
    algebra, inputs, output, values = construct_case(case, placement)
    operation = plan_case(algebra, inputs, output, case)

    qualification = qualify_product(operation, values, case, placement, check_gradients=False)

    assert qualification["status"] == "passed"
    assert qualification["max_abs_error"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "operation",
    [
        "geometric_product",
        "wedge",
        "left_contraction",
        "right_contraction",
        "symmetric_product",
        "commutator_product",
        "anti_commutator_product",
    ],
)
def test_oracle_covers_every_declared_product_operation(operation):
    case = BenchmarkCase(
        case_id=f"product.correctness.{operation}",
        operation=operation,
        signature=(1, 1, 1),
        inputs=(LayoutCase((0, 1, 2, 3), leading_shape=(2,)),) * 2,
        output=LayoutCase((0, 1, 2, 3)),
    )
    placement = ExecutionPlacement("float64")
    algebra, inputs, output, values = construct_case(case, placement)

    qualification = qualify_product(
        plan_case(algebra, inputs, output, case), values, case, placement, check_gradients=True
    )

    assert qualification["status"] == "passed"


@pytest.mark.parametrize(
    "signature",
    [(3, 0, 0), (0, 3, 0), (1, 1, 1)],
)
@pytest.mark.parametrize("storage", ["compact", "canonical"])
@pytest.mark.parametrize(
    "operation",
    [
        "geometric_product",
        "wedge",
        "left_contraction",
        "right_contraction",
        "symmetric_product",
        "commutator_product",
        "anti_commutator_product",
    ],
)
def test_sparse_reference_matches_original_dense_formulation(signature, storage, operation):
    case = BenchmarkCase(
        case_id=f"product.correctness.reference-cross-check.{''.join(map(str, signature))}.{storage}.{operation}",
        operation=operation,
        signature=signature,
        inputs=(
            LayoutCase((0, 1, 2), storage, (2, 1)),
            LayoutCase((1, 2, 3), storage, (1, 3)),
        ),
        output=LayoutCase((0, 1, 2, 3), storage),
    )
    _, _, _, values = construct_case(case)

    actual = reference_product(values, case)
    expected = dense_reference_product(values, case)

    torch.testing.assert_close(actual, expected)


def test_sparse_reference_analytical_gradients_match_dense_autograd():
    case = BenchmarkCase(
        case_id="product.correctness.gradient-cross-check",
        operation="geometric_product",
        signature=(2, 1, 0),
        inputs=(LayoutCase((0, 1, 2), leading_shape=(2, 1)), LayoutCase((1, 2, 3), leading_shape=(1, 3))),
        output=LayoutCase((0, 1, 2, 3)),
    )
    _, _, _, values = construct_case(case)
    dense_inputs = tuple(value.detach().double().requires_grad_() for value in values)

    output, gradients = reference_product_with_gradients(values, case)
    dense_output = dense_reference_product(dense_inputs, case)
    dense_gradients = torch.autograd.grad(dense_output.square().sum(), dense_inputs)

    torch.testing.assert_close(output, dense_output)
    for actual, expected in zip(gradients, dense_gradients):
        torch.testing.assert_close(actual, expected)


def test_sparse_reference_does_not_scan_full_high_dimensional_space():
    case = BenchmarkCase(
        case_id="product.correctness.high-dimensional-compact",
        operation="wedge",
        signature=(32, 0, 0),
        inputs=(LayoutCase((1,)), LayoutCase((1,))),
        output=LayoutCase((2,)),
    )
    _, _, _, values = construct_case(case)

    result = reference_product(values, case)

    assert result.shape == (496,)


def test_large_leading_cpu_sparse_strategy_preserves_broadcast_gradients_and_output_layout():
    case = BenchmarkCase(
        case_id="product.correctness.large-leading-sparse",
        operation="geometric_product",
        signature=(8, 0, 0),
        inputs=(
            LayoutCase((1,), leading_shape=(128, 1)),
            LayoutCase((1,), leading_shape=(1, 2)),
        ),
        output=LayoutCase((0, 2)),
    )
    placement = ExecutionPlacement("float64")
    algebra, inputs, output, values = construct_case(case, placement)
    operation = plan_case(algebra, inputs, output, case)

    result = operation(*values)
    qualification = qualify_product(operation, values, case, placement, check_gradients=True)

    assert operation._kernel.route == "sparse"
    assert result.shape == (128, 2, output.lane_dim)
    assert result.is_contiguous()
    assert qualification["status"] == "passed"


@pytest.mark.parametrize("case", smoke_cases()[2:])
def test_exp_and_action_references_check_forward_and_gradients(case):
    placement = ExecutionPlacement("float64")
    algebra, inputs, output, values = construct_case(case, placement)
    operation = plan_case(algebra, inputs, output, case)

    qualification = qualify_case(operation, values, output, case, placement, check_gradients=True)

    assert qualification["status"] == "passed"
    assert qualification["gradient_checked"] is True
    assert qualification["claim"]


def test_linear_reference_accepts_an_ordinary_matrix_contract():
    case = BenchmarkCase(
        case_id="action.correctness.linear",
        family="action",
        operation="linear",
        signature=(3, 0, 0),
        inputs=(LayoutCase((0, 2), leading_shape=(2, 1)),),
        ordinary_inputs=(OrdinaryTensorCase((1, 3), (3, 3)),),
        output=LayoutCase((0, 2)),
        generator_structure="near_identity_matrix",
        value_scale=0.1,
    )
    placement = ExecutionPlacement("float64")
    algebra, inputs, output, values = construct_case(case, placement)
    qualification = qualify_case(
        plan_case(algebra, inputs, output, case), values, output, case, placement, check_gradients=True
    )

    assert qualification["status"] == "passed"
