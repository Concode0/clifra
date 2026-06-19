# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from clifra.core.runtime.algebra import AlgebraContext
from clifra.layers.adapters.conformal import ConformalEmbedding
from clifra.layers.adapters.projective import ProjectiveEmbedding

pytestmark = pytest.mark.unit


class TestExtensions:
    def test_cga_null_property(self):
        """
        Test that points embedded in CGA are null vectors (P * P = 0).
        """
        algebra = AlgebraContext(p=4, q=1, device="cpu")
        embed = ConformalEmbedding(algebra, euclidean_dim=3)

        # Random Euclidean points
        x = torch.randn(5, 3, device="cpu")

        # Embed
        P = embed.embed(x)

        # Geometric Product P * P
        # Should be scalar 0
        sq = algebra.geometric_product(P, P)

        # Check norm of the result (should be 0)
        assert torch.allclose(sq, torch.zeros_like(sq), atol=1e-5), f"P^2 should be 0, got {sq[0, :5]}..."

        # Check reconstruction
        x_recon = embed.extract(P)
        assert torch.allclose(x, x_recon, atol=1e-5), "Reconstructed x should match original"

    def test_cga_compact_grade1_roundtrip(self):
        """CGA embedding can use compact grade-1 lanes only."""
        algebra = AlgebraContext(p=4, q=1, device="cpu")
        layout = algebra.layout((1,))
        embed = ConformalEmbedding(algebra, euclidean_dim=3, layout=layout)

        x = torch.randn(5, 3, device="cpu")
        P = embed.embed(x)

        assert P.shape == (5, layout.dim)
        assert torch.allclose(embed.extract(P), x, atol=1e-5)

    def test_pga_embed_extract_roundtrip(self):
        """Points embedded in PGA can be extracted back."""
        algebra = AlgebraContext(p=3, q=0, r=1, device="cpu")
        embed = ProjectiveEmbedding(algebra, euclidean_dim=3)

        x = torch.randn(5, 3, device="cpu")
        P = embed.embed(x)

        # Should be grade-1 only
        g1_mask = torch.zeros(algebra.dim, dtype=torch.bool)
        g1_mask[algebra.layout((1,)).indices_tensor()] = True
        assert torch.allclose(P[:, ~g1_mask], torch.zeros_like(P[:, ~g1_mask]))

        # e_0 coefficient should be 1.0
        assert torch.allclose(P[:, embed._idx_e0], torch.ones(5))

        # Roundtrip
        x_recon = embed.extract(P)
        assert torch.allclose(x, x_recon, atol=1e-5)

    def test_pga_compact_grade1_roundtrip(self):
        """PGA embedding can use compact grade-1 lanes only."""
        algebra = AlgebraContext(p=3, q=0, r=1, device="cpu")
        layout = algebra.layout((1,))
        embed = ProjectiveEmbedding(algebra, euclidean_dim=3, layout=layout)

        x = torch.randn(5, 3, device="cpu")
        P = embed.embed(x)

        assert P.shape == (5, layout.dim)
        assert torch.allclose(P[:, embed._idx_e0], torch.ones(5))
        assert torch.allclose(embed.extract(P), x, atol=1e-5)

    def test_pga_direction_has_no_e0(self):
        """Directions (ideal points) have e_0 = 0."""
        algebra = AlgebraContext(p=3, q=0, r=1, device="cpu")
        embed = ProjectiveEmbedding(algebra, euclidean_dim=3)

        v = torch.tensor([[1.0, 0.0, 0.0]])
        P = embed.embed_direction(v)
        assert P[0, embed._idx_e0].item() == 0.0

    def test_pga_rotation_via_rotor(self):
        """Rotors in PGA correctly rotate embedded points."""
        import math

        algebra = AlgebraContext(p=3, q=0, r=1, device="cpu")
        embed = ProjectiveEmbedding(algebra, euclidean_dim=3)

        # Embed (1, 0, 0)
        x = torch.tensor([[1.0, 0.0, 0.0]])
        P = embed.embed(x)

        # 90-degree rotation in e1-e2 plane: R = exp(-pi/4 * e12)
        B = torch.zeros(1, algebra.dim)
        B[0, 3] = -math.pi / 4  # e12 bivector
        R = algebra.exp(B)
        R_rev = algebra.reverse(R)

        # Sandwich product
        RP = algebra.geometric_product(R, P)
        P_rot = algebra.geometric_product(RP, R_rev)

        # Extract: should be (0, 1, 0)
        x_out = embed.extract(P_rot)
        assert torch.allclose(x_out, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-4)

        # e_0 component preserved (degenerate dimension unaffected by rotor)
        assert abs(P_rot[0, embed._idx_e0].item() - 1.0) < 1e-5
