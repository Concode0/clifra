"""Layout-first tests for bivector decomposition primitives."""

import pytest
import torch

from clifra.core.runtime.algebra import AlgebraContext
from clifra.core.runtime.decomposition import ExpPolicy, differentiable_invariant_decomposition, ga_power_iteration

pytestmark = pytest.mark.unit


def _basis_vector(algebra: AlgebraContext, index: int, *, dtype=torch.float32) -> torch.Tensor:
    vector_layout = algebra.layout((1,))
    values = torch.zeros(vector_layout.dim, dtype=dtype)
    values[vector_layout.basis_indices.index(1 << index)] = 1.0
    return vector_layout.full(values)


def test_wedge_is_antisymmetric_for_vectors():
    algebra = AlgebraContext(3, 0, device="cpu")
    e1 = _basis_vector(algebra, 0)
    e2 = _basis_vector(algebra, 1)

    wedge_12 = algebra.wedge(e1, e2)
    wedge_21 = algebra.wedge(e2, e1)

    assert torch.allclose(wedge_12, -wedge_21, atol=1e-6)


def test_power_iteration_recovers_simple_bivector_norm():
    algebra = AlgebraContext(3, 0, device="cpu")
    e1 = _basis_vector(algebra, 0)
    e2 = _basis_vector(algebra, 1)
    bivector = algebra.wedge(e1, e2)

    component, vector = ga_power_iteration(algebra, bivector, threshold=1e-6, max_iterations=100)

    assert vector.shape == e1.shape
    assert torch.allclose(component.norm(), bivector.norm(), atol=1e-4)


def test_bivector_decomposition_reconstructs_sum_of_components():
    algebra = AlgebraContext(3, 0, device="cpu")
    e1 = _basis_vector(algebra, 0)
    e2 = _basis_vector(algebra, 1)
    e3 = _basis_vector(algebra, 2)
    bivector_sum = algebra.wedge(e1, e2) + algebra.wedge(e1, e3)

    components, vectors = differentiable_invariant_decomposition(algebra, bivector_sum, k=2, threshold=1e-6)

    assert len(components) >= 1
    assert len(vectors) >= 1
    reconstructed = sum(components)
    assert torch.allclose(bivector_sum, reconstructed, atol=1e-4)


def test_bivector_decomposition_preserves_gradient_path():
    algebra = AlgebraContext(3, 0, device="cpu")
    bivector = torch.zeros(algebra.dim)
    bivector[3] = 1.0
    bivector.requires_grad_(True)

    components, _ = differentiable_invariant_decomposition(algebra, bivector, k=1)

    assert components[0].requires_grad


def test_exp_policy_setter_replans_exp_iteration_budget():
    algebra = AlgebraContext(4, 0, device="cpu")

    assert algebra.exp_policy == ExpPolicy.BALANCED
    balanced_iterations = algebra._exp_fixed_iterations
    algebra.exp_policy = ExpPolicy.PRECISE

    assert algebra.exp_policy == ExpPolicy.PRECISE
    assert algebra._exp_fixed_iterations > balanced_iterations
