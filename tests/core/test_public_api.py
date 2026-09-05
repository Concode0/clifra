"""Contract-level regression tests independent of concrete executor families."""

import pytest
import torch

import clifra.core as core
from clifra.core import (
    AlgebraConfig,
    AlgebraSpec,
    Layout,
    PlannedOperation,
    TensorContract,
    make_algebra,
    make_algebra_from_config,
)


def test_stable_exports_and_construction_reject_removed_tuning():
    assert all(hasattr(core, name) for name in core.__all__)
    assert not {"GradePlanner", "PlanFacts", "GradeProductExecutor", "AlgebraLike"}.intersection(core.__all__)
    for keyword in ("default_grades", "planning_policy", "resource_limits", "bivector_exp_options"):
        with pytest.raises(TypeError):
            make_algebra(3, **{keyword: None})
        with pytest.raises(TypeError):
            make_algebra_from_config({"p": 3, keyword: None})
    assert make_algebra_from_config(AlgebraConfig(2, 1)).spec == AlgebraSpec(2, 1)


def test_geometric_product_identifier_and_full_basis_default():
    algebra = make_algebra(3)
    a, b = torch.randn(2, 8), torch.randn(2, 8)
    expected = algebra.geometric_product(a, b)
    operation = algebra.plan_product(op="geometric_product")
    assert operation.inputs == (TensorContract(algebra.layout()),) * 2
    assert torch.equal(operation(a, b), expected)
    with pytest.raises(ValueError, match="Unsupported grade product op"):
        algebra.plan_product(op="gp")
    with pytest.raises(TypeError):
        algebra.geometric_product(a, b, left_layout=algebra.layout())


@pytest.mark.parametrize("signature", [(3, 0, 0), (1, 1, 1)])
def test_explicit_storage_preserves_values_and_gradients(signature):
    algebra = make_algebra(*signature, dtype=torch.float64)
    vector, output = algebra.layout((1,)), algebra.layout((0, 2))
    a = torch.randn(2, vector.dim, dtype=torch.float64, requires_grad=True)
    b = torch.randn_like(a, requires_grad=True)
    compact = algebra.geometric_product(a, b, left=vector, right=vector, output=output)
    operation = algebra.plan_product(
        left=TensorContract.canonical(vector), right=vector, output=TensorContract.canonical(output)
    )
    full = operation(vector.full(a), b)
    assert torch.allclose(full, output.full(compact))
    expected_grad = torch.autograd.grad(compact.square().sum(), (a, b), retain_graph=True)
    actual_grad = torch.autograd.grad(full.square().sum(), (a, b))
    assert all(torch.allclose(x, y) for x, y in zip(actual_grad, expected_grad))
    with pytest.raises(ValueError, match="last dimension"):
        algebra.geometric_product(a, b)
    with pytest.raises(ValueError, match="last dimension"):
        algebra.geometric_product(vector.full(a), b, left=vector, right=vector)


def test_preplanned_calls_do_not_resolve_or_plan(monkeypatch):
    algebra = make_algebra(3)
    vector = algebra.layout((1,))
    operation = algebra.plan_product(left=vector, right=vector, pairwise=True)
    a, b = torch.randn(2, 3), torch.randn(4, 3)
    expected = operation(a, b)

    def forbidden(*args, **kwargs):
        raise AssertionError("planning or resolution during a preplanned call")

    monkeypatch.setattr(algebra, "_contract", forbidden)
    monkeypatch.setattr(algebra, "layout", forbidden)
    monkeypatch.setattr(algebra._planner, "product_executor", forbidden)
    assert torch.equal(operation(a, b), expected)
    assert not hasattr(operation, "route")
    assert not hasattr(operation, "executor")
    with pytest.raises(AttributeError):
        operation.output = TensorContract(vector)
    with pytest.raises(ValueError, match="expected 2 tensor"):
        operation(a)


def test_plan_dtype_move_does_not_mutate_other_plans_or_algebra():
    algebra = make_algebra(3)
    vector = algebra.layout((1,))
    first = algebra.plan_product(left=vector, right=vector)
    second = algebra.plan_product(left=vector, right=vector)
    first.to(dtype=torch.float64)
    a = torch.randn(2, 3)
    assert first(a.double(), a.double()).dtype == torch.float64
    assert second(a, a).dtype == torch.float32
    assert algebra.dtype == torch.float32
    algebra.to(dtype=torch.float64)
    assert second(a, a).dtype == torch.float32
    assert algebra.plan_product(left=vector, right=vector)(a.double(), a.double()).dtype == torch.float64


def test_layout_sets_lookup_masks_and_disjoint_conversion_gradients():
    algebra = make_algebra(3)
    vectors, bivectors = algebra.layout((1,)), algebra.layout((2,))
    assert isinstance(vectors, Layout)
    assert vectors.union(bivectors).grades == (1, 2)
    assert vectors.union(bivectors).difference(vectors) == bivectors
    assert vectors.intersection(bivectors).dim == 0
    assert vectors.positions_for_basis((4, 1, 4)).tolist() == [2, 0, 2]
    with pytest.raises(ValueError, match="absent"):
        vectors.positions_for_basis((0,))
    with pytest.raises(ValueError, match="signatures"):
        vectors.union(make_algebra(0, 3).layout((1,)))
    assert vectors.grade_mask((2,)).tolist() == [False] * 3
    empty = vectors.difference(vectors)
    values = torch.randn(2, vectors.dim, requires_grad=True)
    converted = empty.convert(values, vectors)
    assert converted.shape == (2, 0)
    converted.sum().backward()
    assert torch.equal(values.grad, torch.zeros_like(values))
    assert empty.full(converted).shape == (2, 8)


def test_grade_energy_helpers_have_no_batch_reduction_and_handle_empty_input():
    algebra = make_algebra(3)
    layout = algebra.layout((0, 1))
    values = torch.tensor([[2.0, 1.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    energy = algebra.lane_grade_energy(values, input=layout)
    assert torch.equal(energy, torch.tensor([[4.0, 9.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]))
    assert torch.equal(algebra.lane_energy(values, input=layout, grades=(1,)), energy[:, 1:2])
    assert torch.equal(algebra.lane_energy(values, input=layout, grades=()), torch.zeros(2, 1))
    assert torch.isfinite(algebra.lane_grade_distribution(values, input=layout)).all()
    assert torch.equal(algebra.lane_grade_distribution(values, input=layout)[1], torch.zeros(4))
    empty = torch.empty(2, 0, requires_grad=True)
    result = algebra.lane_grade_energy(empty, input=algebra.layout(()))
    result.sum().backward()
    assert result.shape == (2, 4)
    assert empty.grad is not None


@pytest.mark.parametrize("kind", ["product", "sandwich", "linear"])
def test_stable_plans_compile_fullgraph(kind):
    algebra = make_algebra(3)
    vector = algebra.layout((1,))
    if kind == "product":
        operation = algebra.plan_product(left=vector, right=vector)
        args = (torch.randn(2, 3), torch.randn(2, 3))
    elif kind == "sandwich":
        operation = algebra.plan_sandwich_action()
        args = (torch.randn(2, 8), torch.randn(2, 8), torch.randn(2, 8))
    else:
        operation = algebra.plan_linear_action(input=vector)
        args = (torch.randn(2, 1, 3), torch.eye(3).unsqueeze(0))
    assert isinstance(operation, PlannedOperation)
    expected = operation(*args)
    compiled = torch.compile(operation, backend="aot_eager", fullgraph=True)
    assert torch.allclose(compiled(*args), expected)
