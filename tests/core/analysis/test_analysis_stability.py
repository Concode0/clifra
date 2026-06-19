# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from clifra.core.analysis._types import CONSTANTS, AnalysisConfig
from clifra.core.analysis._utils import full_matrix_feasibility, full_product_feasibility
from clifra.core.analysis.commutator import CommutatorAnalyzer
from clifra.core.analysis.geodesic import GeodesicFlow
from clifra.core.analysis.pipeline import GeometricAnalyzer
from clifra.core.analysis.signature import MetricSearch
from clifra.core.analysis.spectral import SpectralAnalyzer
from clifra.core.analysis.symmetry import SymmetryDetector
from clifra.core.config import make_algebra

pytestmark = pytest.mark.unit


def test_geodesic_flow_handles_single_point_without_nan():
    algebra = make_algebra(3, 0, device="cpu", dtype=torch.float64)
    flow = GeodesicFlow(algebra, k=4)
    mv = algebra.embed_vector(torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64))

    coherence = flow._coherence_tensor(mv)
    curvature = flow._curvature_tensor(mv)
    per_point = flow.per_point_coherence(mv)
    bivectors = flow.flow_bivectors(mv)

    assert torch.isfinite(coherence)
    assert torch.isfinite(curvature)
    assert torch.isfinite(per_point).all()
    assert torch.isfinite(bivectors).all()


def test_analysis_modules_accept_context_without_reference_fallback():
    algebra = make_algebra(
        4,
        0,
        device="cpu",
        dtype=torch.float32,
        default_grades=(1,),
    )
    mv = algebra.embed_vector(torch.randn(16, algebra.n))

    spectral = SpectralAnalyzer(algebra).analyze(mv)
    symmetry = SymmetryDetector(algebra).analyze(mv)
    commutator = CommutatorAnalyzer(algebra).analyze(mv)

    assert spectral.grade_energy.shape == (algebra.n + 1,)
    assert len(symmetry.reflection_symmetries) == algebra.n
    assert commutator.commutativity_matrix.shape == (algebra.n, algebra.n)


def test_geometric_analyzer_preserves_configured_dtype_for_embedding():
    cfg = AnalysisConfig(
        device="cpu",
        dtype=torch.float64,
        run_dimension=False,
        run_signature=False,
        run_spectral=True,
        run_symmetry=False,
        run_commutator=False,
    )
    report = GeometricAnalyzer(cfg).analyze(torch.randn(20, 3, dtype=torch.float32))

    assert report.spectral is not None
    assert report.spectral.grade_energy.dtype == torch.float64


def test_analysis_feasibility_reports_eigensolver_matrix_cap():
    algebra = make_algebra(9, 0, device="cpu", dtype=torch.float32)

    feasibility = full_matrix_feasibility(
        algebra,
        role="adjoint_exchange_spectrum",
        max_entries=CONSTANTS.adjoint_matrix_entries,
        matrix_kind="eigensolver",
    )

    assert not feasibility
    assert feasibility.reason == "eigensolver_matrix_cap"
    assert feasibility.details["matrix_entries"] == algebra.dim * algebra.dim


def test_analysis_feasibility_reports_product_pair_cap_without_planning():
    algebra = make_algebra(4, 0, device="cpu", dtype=torch.float32)

    feasibility = full_product_feasibility(
        algebra,
        role="test_full_gp",
        op="gp",
        max_pairs=1,
    )

    assert not feasibility
    assert feasibility.reason == "product_pair_cap"
    assert feasibility.details["estimated_pairs"] == algebra.dim * algebra.dim


def test_metric_search_lift_reports_action_matrix_cap():
    searcher = MetricSearch(device="cpu")
    data = torch.randn(2, 11)

    with pytest.warns(UserWarning, match="action-matrix entries"):
        with pytest.raises(ValueError, match="action matrix"):
            searcher._lift_data(data)


def test_reflection_analysis_uses_product_feasibility_not_dimension_cap():
    algebra = make_algebra(9, 0, device="cpu", dtype=torch.float32)
    mv = algebra.embed_vector(torch.randn(3, algebra.n))

    result = SymmetryDetector(algebra).detect_reflection_symmetries(mv)

    assert len(result) == algebra.n


def test_spectral_result_reports_skipped_gp_spectrum(monkeypatch):
    algebra = make_algebra(3, 0, device="cpu", dtype=torch.float32)
    mv = algebra.embed_vector(torch.randn(6, algebra.n))
    monkeypatch.setattr(CONSTANTS, "gp_spectrum_matrix_entries", 1)

    result = SpectralAnalyzer(algebra).analyze(mv)

    assert result.gp_eigenvalues is None
    assert result.skipped["gp_operator_spectrum"]["reason"] == "eigensolver_matrix_cap"


def test_commutator_result_reports_skipped_exchange_spectrum(monkeypatch):
    algebra = make_algebra(3, 0, device="cpu", dtype=torch.float32)
    mv = algebra.embed_vector(torch.randn(6, algebra.n))
    monkeypatch.setattr(CONSTANTS, "adjoint_matrix_entries", 1)

    result = CommutatorAnalyzer(algebra).analyze(mv)

    assert result.exchange_spectrum.numel() == 0
    assert result.skipped["exchange_spectrum"]["reason"] == "eigensolver_matrix_cap"


def test_symmetry_result_reports_skipped_reflections_and_continuous_symmetries(monkeypatch):
    algebra = make_algebra(3, 0, device="cpu", dtype=torch.float32)
    mv = algebra.embed_vector(torch.randn(6, algebra.n))
    monkeypatch.setattr(CONSTANTS, "reflection_product_pairs", 1)
    monkeypatch.setattr(CONSTANTS, "continuous_symmetry_product_pairs", 1)

    result = SymmetryDetector(algebra).analyze(mv)

    assert result.reflection_symmetries == []
    assert result.continuous_symmetry_dim == 0
    assert result.skipped["reflection_symmetries"]["reason"] == "product_pair_cap"
    assert result.skipped["continuous_symmetries"]["reason"] == "product_pair_cap"
