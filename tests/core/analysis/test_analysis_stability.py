# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from clifra.analysis._types import CONSTANTS, AnalysisConfig
from clifra.analysis._utils import full_matrix_feasibility, full_product_feasibility
from clifra.analysis.commutator import CommutatorAnalyzer
from clifra.analysis.geodesic import NeighborhoodBivectorFlow
from clifra.analysis.pipeline import GeometricAnalyzer
from clifra.analysis.signature import RotorProbeSignatureEstimator
from clifra.analysis.spectral import SpectralAnalyzer
from clifra.analysis.symmetry import TransformationDiagnosticsAnalyzer
from clifra.core.config import make_algebra

pytestmark = pytest.mark.unit


def test_geodesic_flow_handles_single_point_without_nan():
    algebra = make_algebra(3, 0, device="cpu", dtype=torch.float64)
    flow = NeighborhoodBivectorFlow(algebra, k=4)
    mv = algebra.embed_vector(torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64))

    connection_alignment = flow._connection_alignment_tensor(mv)
    connection_dissimilarity = flow._connection_dissimilarity_tensor(mv)
    per_point = flow.per_point_connection_alignment(mv)
    bivectors = flow.flow_bivectors(mv)

    assert torch.isfinite(connection_alignment)
    assert torch.isfinite(connection_dissimilarity)
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
    symmetry = TransformationDiagnosticsAnalyzer(algebra).analyze(mv)
    commutator = CommutatorAnalyzer(algebra).analyze(mv)

    assert spectral.grade_energy.shape == (algebra.n + 1,)
    assert len(symmetry.basis_reflection_scores) == algebra.n
    assert commutator.pairwise_commutator_norms.shape == (algebra.n, algebra.n)


def test_geometric_analyzer_preserves_configured_dtype_for_embedding():
    cfg = AnalysisConfig(
        device="cpu",
        dtype=torch.float64,
        run_dimension=False,
        run_signature_estimation=False,
        run_spectral=True,
        run_transformation_diagnostics=False,
        run_commutator=False,
    )
    report = GeometricAnalyzer(cfg).analyze(torch.randn(20, 3, dtype=torch.float32))

    assert report.spectral is not None
    assert report.spectral.grade_energy.dtype == torch.float64


def test_analysis_feasibility_reports_eigensolver_matrix_cap():
    algebra = make_algebra(9, 0, device="cpu", dtype=torch.float32)

    feasibility = full_matrix_feasibility(
        algebra,
        role="adjoint_eigenvalue_magnitudes",
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


def test_signature_probe_lift_reports_action_matrix_cap():
    searcher = RotorProbeSignatureEstimator(device="cpu")
    data = torch.randn(2, 11)

    with pytest.warns(UserWarning, match="action-matrix entries"):
        with pytest.raises(ValueError, match="action matrix"):
            searcher._lift_data(data)


def test_reflection_analysis_uses_product_feasibility_not_dimension_cap():
    algebra = make_algebra(9, 0, device="cpu", dtype=torch.float32)
    mv = algebra.embed_vector(torch.randn(3, algebra.n))

    result = TransformationDiagnosticsAnalyzer(algebra).basis_reflection_scores(mv)

    assert len(result) == algebra.n


def test_spectral_result_reports_skipped_gp_spectrum(monkeypatch):
    algebra = make_algebra(3, 0, device="cpu", dtype=torch.float32)
    mv = algebra.embed_vector(torch.randn(6, algebra.n))
    monkeypatch.setattr(CONSTANTS, "gp_spectrum_matrix_entries", 1)

    result = SpectralAnalyzer(algebra).analyze(mv)

    assert result.gp_action_eigenvalue_magnitudes is None
    assert result.skipped["gp_action_eigenvalue_magnitudes"]["reason"] == "eigensolver_matrix_cap"


def test_commutator_result_reports_skipped_adjoint_eigenvalue_magnitudes(monkeypatch):
    algebra = make_algebra(3, 0, device="cpu", dtype=torch.float32)
    mv = algebra.embed_vector(torch.randn(6, algebra.n))
    monkeypatch.setattr(CONSTANTS, "adjoint_matrix_entries", 1)

    result = CommutatorAnalyzer(algebra).analyze(mv)

    assert result.adjoint_eigenvalue_magnitudes.numel() == 0
    assert result.skipped["adjoint_eigenvalue_magnitudes"]["reason"] == "eigensolver_matrix_cap"


def test_symmetry_result_reports_skipped_reflections_and_near_commuting_modes(monkeypatch):
    algebra = make_algebra(3, 0, device="cpu", dtype=torch.float32)
    mv = algebra.embed_vector(torch.randn(6, algebra.n))
    monkeypatch.setattr(CONSTANTS, "reflection_product_pairs", 1)
    monkeypatch.setattr(CONSTANTS, "near_commuting_mode_product_pairs", 1)

    result = TransformationDiagnosticsAnalyzer(algebra).analyze(mv)

    assert result.basis_reflection_scores == []
    assert result.near_commuting_mode_count == 0
    assert result.skipped["basis_reflection_scores"]["reason"] == "product_pair_cap"
    assert result.skipped["near_commuting_modes"]["reason"] == "product_pair_cap"
