# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Tests for the learned RotorProbeSignatureEstimator probe approach."""

import pytest
import torch

from clifra.core.runtime.algebra import AlgebraContext

pytestmark = pytest.mark.slow
from clifra.analysis.geodesic import NeighborhoodBivectorFlow
from clifra.analysis.signature import RotorProbeSignatureEstimator, _apply_biased_init, _SignatureProbe


@pytest.fixture(scope="module")
def small_searcher():
    """Create a small RotorProbeSignatureEstimator instance for testing."""
    return RotorProbeSignatureEstimator(
        device="cpu",
        probe_epochs=60,
        num_probes=2,
        probe_channels=2,
        k=4,
    )


@pytest.fixture(scope="module")
def alg_conformal():
    """Create a Cl(3,1) algebra -- conformal lift of 2D data."""
    return AlgebraContext(3, 1, 0, device="cpu")


class TestRotorProbeSignatureEstimatorAPI:
    """Tests for RotorProbeSignatureEstimator public API."""

    def test_search_returns_3_tuple(self, small_searcher):
        """Verify search returns (p, q, r) signature tuple."""
        torch.manual_seed(0)
        data = torch.randn(16, 2)
        result = small_searcher.estimate(data)
        assert isinstance(result, tuple)
        assert len(result) == 3
        p, q, r = result
        assert isinstance(p, int)
        assert isinstance(q, int)
        assert isinstance(r, int)

    def test_search_euclidean_data(self, small_searcher):
        """Verify search identifies Euclidean data correctly.

        Uses a circle (clear 2D Euclidean manifold) rather than random Gaussian
        noise, which has no structural signal for the probe to latch onto.
        """
        # 2D circle: unambiguous Euclidean manifold structure
        theta = torch.linspace(0, 2 * 3.141592653589793, 33)[:-1]
        data = torch.stack([theta.cos(), theta.sin()], dim=-1)
        p, q, r = small_searcher.estimate(data)
        # Euclidean data should have p >= 1 and p dominates
        assert p >= 1, f"Euclidean data should have p>=1, got p={p}"
        assert p >= q, f"Euclidean data should have p>=q, got p={p}, q={q}"

    def test_search_detailed_keys(self, small_searcher):
        """Verify search_detailed returns all expected diagnostic keys."""
        torch.manual_seed(2)
        data = torch.randn(16, 2)
        result = small_searcher.estimate_detailed(data)
        for key in (
            "estimated_signature",
            "connection_alignment",
            "connection_dissimilarity",
            "energy_breakdown",
            "per_probe_results",
        ):
            assert key in result, f"Missing key: {key}"

    def test_sequential_small_probes(self):
        """Verify sequential path works with num_probes=1."""
        searcher = RotorProbeSignatureEstimator(
            device="cpu",
            probe_epochs=10,
            num_probes=1,
            probe_channels=2,
            k=4,
        )
        torch.manual_seed(3)
        data = torch.randn(12, 2)
        p, q, r = searcher.estimate(data)
        assert p + q + r <= 2


class TestSignatureProbe:
    """Tests for _SignatureProbe."""

    def test_forward_shape(self, alg_conformal):
        """Verify forward pass output shape."""
        probe = _SignatureProbe(alg_conformal, channels=2)
        x = torch.randn(8, 1, alg_conformal.dim)
        out = probe(x)
        assert out.shape == (8, 1, alg_conformal.dim)

    def test_get_rotor_layers(self, alg_conformal):
        """Verify retrieval of rotor layers."""
        probe = _SignatureProbe(alg_conformal, channels=2)
        rotors = probe.get_rotor_layers()
        assert len(rotors) >= 1


class TestConformalLifting:
    """Tests for RotorProbeSignatureEstimator._lift_data."""

    def test_lifting_shape(self):
        """Verify data lifting to higher-dimensional multivector space."""
        searcher = RotorProbeSignatureEstimator(device="cpu")
        data = torch.randn(10, 3)
        mv, algebra = searcher._lift_data(data)
        # 3D data -> Cl(4, 1) -> dim = 2^5 = 32
        assert algebra.n == 5
        assert algebra.p == 4
        assert algebra.q == 1
        assert mv.shape == (10, 1, 32)

    def test_lifting_2d(self):
        """Verify 2D data lifting to Cl(3,1)."""
        searcher = RotorProbeSignatureEstimator(device="cpu")
        data = torch.randn(8, 2)
        mv, algebra = searcher._lift_data(data)
        # 2D data -> Cl(3, 1) -> dim = 2^4 = 16
        assert algebra.n == 4
        assert mv.shape == (8, 1, 16)


class TestBiasedInit:
    """Tests for _apply_biased_init."""

    def test_euclidean_bias(self, alg_conformal):
        """Verify Euclidean bias prioritizes elliptic bivectors."""
        probe = _SignatureProbe(alg_conformal, channels=2)
        _apply_biased_init(probe, alg_conformal, "euclidean")
        bv_sq = alg_conformal.bivector_squared_signs(device=alg_conformal.device, dtype=alg_conformal.dtype)
        for rotor in probe.get_rotor_layers():
            weights = rotor.bivector_weights.detach()
            # Elliptic bivectors (bv_sq < -0.5) should have larger weights
            elliptic_mask = bv_sq < -0.5
            if elliptic_mask.any():
                elliptic_energy = weights[:, elliptic_mask].abs().mean()
                non_elliptic_mask = ~elliptic_mask
                if non_elliptic_mask.any():
                    other_energy = weights[:, non_elliptic_mask].abs().mean()
                    assert elliptic_energy > other_energy

    def test_all_bias_types_run(self, alg_conformal):
        """Verify all bias types run without error."""
        for bias_type in ("euclidean", "minkowski", "projective", "random"):
            probe = _SignatureProbe(alg_conformal, channels=2)
            _apply_biased_init(probe, alg_conformal, bias_type)
            # Should not raise


class TestDifferentiableMethods:
    """Tests for _connection_alignment_tensor and _connection_dissimilarity_tensor."""

    def test_connection_alignment_tensor_differentiable(self):
        """Verify connection_alignment calculation is differentiable."""
        alg = AlgebraContext(3, 0, device="cpu")
        data = torch.randn(16, 3, requires_grad=True)
        mv = alg.embed_vector(data)
        gf = NeighborhoodBivectorFlow(alg, k=4)
        coh = gf._connection_alignment_tensor(mv)
        assert isinstance(coh, torch.Tensor)
        assert coh.dim() == 0  # scalar
        # Should have grad_fn (differentiable)
        assert coh.grad_fn is not None

    def test_connection_dissimilarity_tensor_differentiable(self):
        """Verify connection_dissimilarity calculation is differentiable."""
        alg = AlgebraContext(3, 0, device="cpu")
        data = torch.randn(16, 3, requires_grad=True)
        mv = alg.embed_vector(data)
        gf = NeighborhoodBivectorFlow(alg, k=4)
        curv = gf._connection_dissimilarity_tensor(mv)
        assert isinstance(curv, torch.Tensor)
        assert curv.dim() == 0
        assert curv.grad_fn is not None

    def test_connection_alignment_tensor_matches_connection_alignment(self):
        """Verify _connection_alignment_tensor matches connection_alignment float value."""
        alg = AlgebraContext(3, 0, device="cpu")
        torch.manual_seed(10)
        data = torch.randn(16, 3)
        mv = alg.embed_vector(data)
        gf = NeighborhoodBivectorFlow(alg, k=4)
        coh_float = gf.connection_alignment(mv)
        coh_tensor = gf._connection_alignment_tensor(mv).item()
        assert abs(coh_float - coh_tensor) < 1e-6


class TestBivectorEnergyAnalysis:
    """Tests for RotorProbeSignatureEstimator._analyze_bivector_energy."""

    def test_returns_valid_signature(self):
        """Verify energy analysis returns valid (p, q, r) and breakdown dict."""
        alg = AlgebraContext(3, 1, 0, device="cpu")
        probe = _SignatureProbe(alg, channels=2)
        searcher = RotorProbeSignatureEstimator(device="cpu")
        (p, q, r), breakdown = searcher._analyze_bivector_energy(probe, alg, 2)
        assert isinstance(p, int)
        assert isinstance(q, int)
        assert isinstance(r, int)
        assert p + q + r <= 2
        assert "per_bivector_energy" in breakdown
        assert "bivector_squared_signs" in breakdown
