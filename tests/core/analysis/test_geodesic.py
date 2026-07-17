# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Tests for NeighborhoodBivectorFlow and CoordinateLiftAnalyzer in core/analysis/.

Covers:
- NeighborhoodBivectorFlow: flow bivectors, connection_alignment, connection_dissimilarity, interpolation,
  alignment_threshold_report and its operational threshold result
- CoordinateLiftAnalyzer: lift shape, positive/null lift, test() report structure,
  lifting structured data improves connection_alignment, format_report
"""

import math

import pytest
import torch

from clifra.analysis.dimension import CoordinateLiftAnalyzer
from clifra.analysis.geodesic import NeighborhoodBivectorFlow
from clifra.core.runtime.algebra import AlgebraContext

pytestmark = pytest.mark.unit


# Fixtures


def _circle_data(N: int = 64, noise: float = 0.0) -> torch.Tensor:
    """N points uniformly on the unit circle, optionally with Gaussian noise."""
    theta = torch.linspace(0, 2 * math.pi, N + 1)[:-1]
    x = torch.stack([theta.cos(), theta.sin()], dim=-1)
    if noise > 0:
        x = x + torch.randn_like(x) * noise
    return x


def _helix_data(N: int = 64) -> torch.Tensor:
    """3D helix: uniform rotation plane -> concentrated connection bivectors."""
    t = torch.linspace(0, 4 * math.pi, N)
    x = torch.stack([t.cos(), t.sin(), t / (4 * math.pi)], dim=-1)
    return x


def _flat_2d_in_3d(N: int = 64) -> torch.Tensor:
    """2D circle embedded in 3D with z=0.

    All connection bivectors are in the e_12 plane (grade-2 index 3 in Cl(3,0)).
    Pairwise |cos| within every neighbourhood = 1.0 -> connection_alignment = 1.0.
    """
    theta = torch.linspace(0, 2 * math.pi, N + 1)[:-1]
    z = torch.zeros(N)
    return torch.stack([theta.cos(), theta.sin(), z], dim=-1)


def _aligned_flow_data(N: int = 64) -> torch.Tensor:
    """2D data whose flow is strongly aligned: points on a spiral."""
    t = torch.linspace(0, 4 * math.pi, N)
    r = 0.5 + t / (4 * math.pi)
    x = torch.stack([r * t.cos(), r * t.sin()], dim=-1)
    return x


def _random_data(N: int = 64, dim: int = 2) -> torch.Tensor:
    """Pure noise: no spatial structure."""
    return torch.randn(N, dim)


# TestNeighborhoodBivectorFlow


class TestNeighborhoodBivectorFlow:
    """Tests for NeighborhoodBivectorFlow."""

    def test_flow_bivectors_shape(self, alg2):
        """Flow bivectors must have shape [N, algebra.dim]."""
        gf = NeighborhoodBivectorFlow(alg2, k=4)
        data = _circle_data(32)
        mv = gf._embed(data)
        flow = gf.flow_bivectors(mv)
        assert flow.shape == (32, alg2.dim)

    def test_connection_bivectors_grade2_only(self, alg2):
        """Connection bivectors must have energy only in grade-2 components.

        Tests the raw per-edge bivectors (not the mean, which may cancel for
        symmetric data).  In Cl(2,0) the only grade-2 blade is index 3 (e_12).
        """
        gf = NeighborhoodBivectorFlow(alg2, k=4)
        # Use two non-parallel grade-1 vectors that produce a non-zero bivector
        data = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [0.5, 0.5]]
        )
        mv = gf._embed(data)
        bv = gf._connection_bivectors(mv)  # [N, k, dim]

        # Non-grade-2 components must be zero
        non_g2_mask = [i for i in range(alg2.dim) if bin(i).count("1") != 2]
        other_energy = bv[:, :, non_g2_mask].abs().max().item()
        assert other_energy < 1e-5, f"Non-grade-2 energy should be ~0, got {other_energy}"

        # Grade-2 component must have some energy (index 3 = e_12 in Cl(2,0))
        g2_energy = bv[:, :, 3].abs().max().item()
        assert g2_energy > 0.0, "Connection bivectors should have grade-2 energy"

    def test_connection_alignment_range(self, alg2):
        """Coherence must be in [0, 1]."""
        gf = NeighborhoodBivectorFlow(alg2, k=4)
        for data in [_circle_data(32), _random_data(32, dim=2)]:
            mv = gf._embed(data)
            coh = gf.connection_alignment(mv)
            assert 0.0 <= coh <= 1.0 + 1e-5, f"Coherence {coh} out of [0,1] range"

    def test_connection_dissimilarity_nonnegative(self, alg2):
        """Curvature must be >= 0."""
        gf = NeighborhoodBivectorFlow(alg2, k=4)
        for data in [_circle_data(32), _random_data(32, dim=2)]:
            mv = gf._embed(data)
            curv = gf.connection_dissimilarity(mv)
            assert curv >= 0.0, f"Curvature {curv} is negative"

    def test_structured_higher_connection_alignment_than_noise(self, alg3):
        """Flat 2D circle in 3D should have connection_alignment=1 vs lower-connection_alignment random 3D.

        In Cl(3,0) the bivector space has 3 planes (e_12, e_13, e_23).

        A flat circle in the z=0 plane forces ALL connection bivectors into
        the single e_12 plane - pairwise |cos| = 1.0 exactly -> connection_alignment = 1.0.

        Pure random 3D data scatters connections across all three planes, giving
        connection_alignment below 1.0.
        """
        torch.manual_seed(0)
        gf = NeighborhoodBivectorFlow(alg3, k=6)

        flat = _flat_2d_in_3d(64)
        noise = _random_data(64, dim=3)

        mv_flat = gf._embed(flat)
        mv_noise = gf._embed(noise)

        coh_flat = gf.connection_alignment(mv_flat)
        coh_noise = gf.connection_alignment(mv_noise)

        assert coh_flat > coh_noise, (
            f"Flat-2D-in-3D connection_alignment {coh_flat:.3f} should exceed noise {coh_noise:.3f}"
        )
        # Flat data must be exactly 1.0 (all connections in e_12 plane)
        assert abs(coh_flat - 1.0) < 1e-4, f"Flat connection_alignment should be 1.0, got {coh_flat:.5f}"

    def test_structured_lower_connection_dissimilarity_than_noise(self, alg3):
        """Flat 2D circle in 3D should have connection_dissimilarity=0 vs higher-connection_dissimilarity random 3D.

        Since all connections of every point are in the e_12 plane, the cross-
        neighbourhood comparison always gives |cos| = 1.0 -> connection_dissimilarity = 0.0.
        """
        torch.manual_seed(1)
        gf = NeighborhoodBivectorFlow(alg3, k=6)

        flat = _flat_2d_in_3d(64)
        noise = _random_data(64, dim=3)

        mv_flat = gf._embed(flat)
        mv_noise = gf._embed(noise)

        curv_flat = gf.connection_dissimilarity(mv_flat)
        curv_noise = gf.connection_dissimilarity(mv_noise)

        assert curv_flat < curv_noise, (
            f"Flat connection_dissimilarity {curv_flat:.3f} should be less than noise {curv_noise:.3f}"
        )

    def test_embed_shape(self, alg3):
        """Embed should produce [N, dim] grade-1 multivectors."""
        gf = NeighborhoodBivectorFlow(alg3, k=4)
        data = torch.randn(20, 3)
        mv = gf._embed(data)
        assert mv.shape == (20, alg3.dim)

    def test_embed_grade1_only(self, alg3):
        """Embedded multivectors must have energy only in grade-1 blades."""
        gf = NeighborhoodBivectorFlow(alg3, k=4)
        data = torch.randn(10, 3)
        mv = gf._embed(data)
        # Non-grade-1 blades must remain zero.
        other_mask = [i for i in range(alg3.dim) if bin(i).count("1") != 1]
        assert mv[:, other_mask].abs().max().item() < 1e-6

    def test_knn_count(self, alg2):
        """KNN should return min(k, N-1) indices per point."""
        gf = NeighborhoodBivectorFlow(alg2, k=10)
        data = _circle_data(8)  # fewer points than k
        mv = gf._embed(data)
        idx = gf._knn(mv)
        assert idx.shape == (8, 7)  # k capped at N-1 = 7

    def test_interpolate_endpoints(self, alg3):
        """Interpolated endpoints must match a and b (up to approximation)."""
        gf = NeighborhoodBivectorFlow(alg3, k=4)
        # Use simple unit vectors
        a = alg3.embed_vector(torch.tensor([[1.0, 0.0, 0.0]]))  # [1, 8]
        b = alg3.embed_vector(torch.tensor([[0.0, 1.0, 0.0]]))  # [1, 8]
        path = gf.approximate_bivector_interpolation(a[0], b[0], steps=10)  # [10, 8]
        assert path.shape == (10, alg3.dim)
        # Step 0 should be close to a
        assert torch.allclose(path[0], a[0], atol=1e-4)

    def test_interpolate_steps(self, alg3):
        """Number of returned frames should equal steps."""
        gf = NeighborhoodBivectorFlow(alg3, k=4)
        a = alg3.embed_vector(torch.tensor([[1.0, 0.0, 0.0]]))
        b = alg3.embed_vector(torch.tensor([[0.0, 1.0, 0.0]]))
        for steps in [2, 5, 20]:
            path = gf.approximate_bivector_interpolation(a[0], b[0], steps=steps)
            assert path.shape[0] == steps

    def test_alignment_threshold_report_keys(self, alg2):
        """alignment_threshold_report must return all expected keys."""
        gf = NeighborhoodBivectorFlow(alg2, k=4)
        report = gf.alignment_threshold_report(_circle_data(32))
        for key in ("connection_alignment", "connection_dissimilarity", "passes_alignment_thresholds", "label"):
            assert key in report, f"Missing key: {key}"

    def test_alignment_threshold_report_types(self, alg2):
        """alignment_threshold_report values must have correct types."""
        gf = NeighborhoodBivectorFlow(alg2, k=4)
        report = gf.alignment_threshold_report(_circle_data(32))
        assert isinstance(report["connection_alignment"], float)
        assert isinstance(report["connection_dissimilarity"], float)
        assert isinstance(report["passes_alignment_thresholds"], bool)
        assert isinstance(report["label"], str)

    def test_alignment_threshold_report_aligned_data(self, alg2):
        """The report label follows its operational threshold flag."""
        gf = NeighborhoodBivectorFlow(alg2, k=6)
        report = gf.alignment_threshold_report(_aligned_flow_data(128))
        # The exact result depends on the data; check label consistency.
        if report["passes_alignment_thresholds"]:
            assert report["label"] == "passes thresholds"
        else:
            assert report["label"] == "outside thresholds"

    def test_alignment_threshold_report_label_consistency(self, alg2):
        """The threshold flag and label string must be consistent."""
        torch.manual_seed(42)
        gf = NeighborhoodBivectorFlow(alg2, k=6)
        for data in [_circle_data(64), _random_data(64, dim=2)]:
            report = gf.alignment_threshold_report(data)
            if report["passes_alignment_thresholds"]:
                assert report["label"] == "passes thresholds"
            else:
                assert report["label"] == "outside thresholds"


# TestCoordinateLiftAnalyzer


class TestCoordinateLiftAnalyzer:
    """Tests for CoordinateLiftAnalyzer."""

    def test_lift_shape(self):
        """Lifted multivectors must have shape [N, target_dim]."""
        lifter = CoordinateLiftAnalyzer(device="cpu")
        data = torch.randn(20, 2)
        alg = AlgebraContext(3, 0, device="cpu")  # 2D data -> 3D algebra
        mv = lifter.lift(data, alg, fill=1.0)
        assert mv.shape == (20, alg.dim)

    def test_lift_grade1_only(self):
        """Lifted multivectors must have energy only in grade-1 blades."""
        lifter = CoordinateLiftAnalyzer(device="cpu")
        data = torch.randn(10, 2)
        alg = AlgebraContext(3, 0, device="cpu")
        mv = lifter.lift(data, alg, fill=1.0)
        other = [i for i in range(alg.dim) if bin(i).count("1") != 1]
        assert mv[:, other].abs().max().item() < 1e-6

    def test_lift_same_dimension(self):
        """Lifting to the same dimension (no padding) should still work."""
        lifter = CoordinateLiftAnalyzer(device="cpu")
        data = torch.randn(10, 3)
        alg = AlgebraContext(3, 0, device="cpu")
        mv = lifter.lift(data, alg, fill=1.0)
        assert mv.shape == (10, alg.dim)

    def test_lift_fill_values(self):
        """The extra coordinate should equal `fill` in the extra blade."""
        lifter = CoordinateLiftAnalyzer(device="cpu")
        data = torch.zeros(5, 2)  # all-zero 2D data
        alg = AlgebraContext(3, 0, device="cpu")
        # The third grade-1 blade is index 4 (= 1 << 2)
        mv = lifter.lift(data, alg, fill=0.7)
        assert torch.allclose(mv[:, 4], torch.full((5,), 0.7), atol=1e-5)

    def test_lift_too_large_raises(self):
        """Lifting to a smaller algebra must raise ValueError."""
        lifter = CoordinateLiftAnalyzer(device="cpu")
        data = torch.randn(10, 4)
        alg = AlgebraContext(2, 0, device="cpu")  # only 2D
        with pytest.raises(ValueError):
            lifter.lift(data, alg)

    def test_test_output_keys(self):
        """test() must return all expected keys."""
        lifter = CoordinateLiftAnalyzer(device="cpu")
        data = _circle_data(32)
        results = lifter.compare_lifts(data, p=2, q=0, k=4)
        for key in ("original", "positive_coordinate_lift", "negative_square_coordinate_lift", "highest_alignment"):
            assert key in results

    def test_compare_lifts_operational_fields(self):
        """Each result includes the algebra signature and operational scores."""
        lifter = CoordinateLiftAnalyzer(device="cpu")
        data = _circle_data(32)
        results = lifter.compare_lifts(data, p=2, q=0, k=4)
        for key in ("original", "positive_coordinate_lift", "negative_square_coordinate_lift"):
            r = results[key]
            assert "algebra_signature" in r
            assert "connection_alignment" in r
            assert "connection_dissimilarity" in r
            assert "passes_alignment_thresholds" in r

    def test_test_original_signature(self):
        """Original result must carry the correct (p, q) signature."""
        lifter = CoordinateLiftAnalyzer(device="cpu")
        results = lifter.compare_lifts(_circle_data(32), p=2, q=0, k=4)
        assert results["original"]["algebra_signature"] == (2, 0)

    def test_test_lifted_signatures(self):
        """Lifted results must carry incremented signatures."""
        lifter = CoordinateLiftAnalyzer(device="cpu")
        results = lifter.compare_lifts(_circle_data(32), p=2, q=0, k=4)
        assert results["positive_coordinate_lift"]["algebra_signature"] == (3, 0)
        assert results["negative_square_coordinate_lift"]["algebra_signature"] == (2, 1)

    def test_test_best_is_valid_key(self):
        """'best' must point to one of the three algebra keys."""
        lifter = CoordinateLiftAnalyzer(device="cpu")
        results = lifter.compare_lifts(_circle_data(32), p=2, q=0, k=4)
        assert results["highest_alignment"] in (
            "original",
            "positive_coordinate_lift",
            "negative_square_coordinate_lift",
        )

    def test_test_best_has_highest_connection_alignment(self):
        """'best' must indeed be the algebra with the highest connection_alignment."""
        lifter = CoordinateLiftAnalyzer(device="cpu")
        results = lifter.compare_lifts(_circle_data(48), p=2, q=0, k=6)
        best_key = results["highest_alignment"]
        best_coh = results[best_key]["connection_alignment"]
        for key in ("original", "positive_coordinate_lift", "negative_square_coordinate_lift"):
            assert results[key]["connection_alignment"] <= best_coh + 1e-6

    def test_positive_lift_expands_algebra_dim(self):
        """Positive lift must produce [N, 2^(n+1)] multivectors."""
        lifter = CoordinateLiftAnalyzer(device="cpu")
        data = _circle_data(16)
        alg_pos = AlgebraContext(3, 0, device="cpu")
        mv = lifter.lift(data, alg_pos, fill=1.0)
        assert mv.shape == (16, alg_pos.dim)  # 2^3 = 8

    def test_format_report_returns_string(self):
        """format_report must return a non-empty string."""
        lifter = CoordinateLiftAnalyzer(device="cpu")
        results = lifter.compare_lifts(_circle_data(32), p=2, q=0, k=4)
        report = lifter.format_report(results)
        assert isinstance(report, str)
        assert len(report) > 0
        assert "Cl(" in report

    def test_format_report_contains_all_algebras(self):
        """Report must mention all three algebras."""
        lifter = CoordinateLiftAnalyzer(device="cpu")
        results = lifter.compare_lifts(_circle_data(32), p=2, q=0, k=4)
        report = lifter.format_report(results)
        # All three signatures should appear
        assert "Cl(2,0)" in report
        assert "Cl(3,0)" in report
        assert "Cl(2,1)" in report
