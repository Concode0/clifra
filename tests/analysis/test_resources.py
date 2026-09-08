# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from clifra.analysis import (
    CommutatorAnalyzer,
    SpectralAnalyzer,
    TransformationDiagnosticsAnalyzer,
    commutator,
    spectral,
    transformation,
)
from clifra.analysis._resources import check_full_matrix_budget, check_full_product_budget
from clifra.core._kernel.planning.resources import ResourceLimits
from clifra.core.config import make_algebra

pytestmark = pytest.mark.unit


def test_matrix_and_product_caps_do_not_plan(monkeypatch):
    algebra = make_algebra(9)

    def no_planning(*args, **kwargs):
        pytest.fail("feasibility must use immutable metadata without planning")

    monkeypatch.setattr(algebra, "layout", no_planning)
    matrix = check_full_matrix_budget(
        algebra, role="test", matrix_kind="eigensolver", limits=ResourceLimits(max_lanes=256)
    )
    assert not matrix and matrix.reason == "eigensolver_matrix_cap"
    assert matrix.details["matrix_entries"] == 512**2
    product = check_full_product_budget(
        algebra, role="test", op="geometric_product", limits=ResourceLimits(max_pairs=1)
    )
    assert not product and product.reason == "product_pair_cap"


def test_spectral_caps_apply_to_aggregate_and_direct_method(monkeypatch):
    algebra = make_algebra(3)
    data = torch.randn(5, algebra.dim)
    monkeypatch.setattr(spectral, "_LEFT_MULTIPLICATION_LIMITS", ResourceLimits(max_pairs=1))
    analyzer = SpectralAnalyzer(algebra)
    result = analyzer.analyze(data)
    assert result.left_multiplication_eigenvalue_magnitudes is None
    assert analyzer.left_multiplication_eigenvalue_magnitudes(data) is None
    assert result.skipped["left_multiplication_eigenvalue_magnitudes"]["reason"] == "eigensolver_matrix_cap"
    assert torch.isfinite(result.grade_coefficient_energy).all()


def test_commutator_product_caps_never_substitute_vector_measurements(monkeypatch):
    algebra = make_algebra(3)
    data = torch.zeros(2, algebra.dim)
    data[0, 3], data[1, 5] = 1, 1
    monkeypatch.setattr(commutator, "_PRODUCT_LIMITS", ResourceLimits(max_pairs=1))
    analyzer = CommutatorAnalyzer(algebra)
    result = analyzer.analyze(data)
    for name in ("mean_commutator_norm", "adjoint_eigenvalue_magnitudes", "basis_bivector_commutator_norm_ratios"):
        assert getattr(result, name) is None
        assert getattr(analyzer, name)(data) is None
        assert result.skipped[name]["reason"] == "product_pair_cap"


def test_adjoint_matrix_cap_keeps_other_measurements(monkeypatch):
    algebra = make_algebra(3)
    monkeypatch.setattr(commutator, "_ADJOINT_LIMITS", ResourceLimits(max_pairs=1))
    result = CommutatorAnalyzer(algebra).analyze(torch.randn(4, algebra.dim))
    assert result.adjoint_eigenvalue_magnitudes is None
    assert result.mean_commutator_norm is not None
    assert result.basis_bivector_commutator_norm_ratios is not None


def test_reflection_caps_apply_to_aggregate_and_direct_method(monkeypatch):
    algebra = make_algebra(3)
    data = torch.randn(5, algebra.dim)
    monkeypatch.setattr(transformation, "_REFLECTION_LIMITS", ResourceLimits(max_pairs=1))
    analyzer = TransformationDiagnosticsAnalyzer(algebra)
    result = analyzer.analyze(data)
    assert result.basis_reflection_marginal_scores is None
    assert analyzer.basis_reflection_marginal_scores(data) is None
    assert result.skipped["basis_reflection_marginal_scores"]["reason"] == "product_pair_cap"
    assert torch.isfinite(result.vector_coefficient_energy).all()


def test_reflections_use_product_cost_not_an_arbitrary_dimension_cutoff():
    algebra = make_algebra(9)
    data = algebra.layout((1,)).full(torch.randn(3, 9))
    scores = TransformationDiagnosticsAnalyzer(algebra).basis_reflection_marginal_scores(data)
    assert scores is not None and len(scores) == 9


def test_spectrum_skip_records_algebra_normalized_dtype(monkeypatch):
    algebra = make_algebra(3, dtype=torch.float32)
    monkeypatch.setattr(spectral, "_LEFT_MULTIPLICATION_LIMITS", ResourceLimits(max_pairs=1))
    data = torch.randn(4, algebra.dim, dtype=torch.float64)
    result = SpectralAnalyzer(algebra).analyze(data)
    details = result.skipped["left_multiplication_eigenvalue_magnitudes"]["checks"]["eigensolver_matrix"]["details"]
    assert details["dtype"] == "float32"
    assert details["estimated_bytes"] == algebra.dim**2 * torch.empty((), dtype=algebra.dtype).element_size()
