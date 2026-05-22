import pytest
import torch

from clifra.core.analysis import AnalysisConfig, CommutatorAnalyzer, GeodesicFlow, GeometricAnalyzer, SpectralAnalyzer
from clifra.core.analysis.symmetry import SymmetryDetector
from clifra.core.config import make_algebra

pytestmark = pytest.mark.unit


def test_geodesic_flow_handles_single_point_without_nan():
    algebra = make_algebra(3, 0, kernel="dense", device="cpu", dtype=torch.float64)
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


def test_analysis_modules_accept_context_without_dense_fallback():
    algebra = make_algebra(
        4,
        0,
        kernel="context",
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
