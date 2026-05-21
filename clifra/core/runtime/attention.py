"""Core geometric attention score routing for dense and compact storage."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from clifra.core.foundation.layout import GradeLayout
from clifra.core.foundation.module import AlgebraLike, require_dense_kernel_host


class GeometricAttentionScorer(nn.Module):
    """Exact geometric-product attention scores for dense or compact layouts.

    Layers own projection and value aggregation. This scorer owns the algebraic
    score route so dense and compact optimization policy stays in core.
    """

    def __init__(
        self,
        algebra: AlgebraLike,
        *,
        head_channels: int,
        bivector_weight: float,
        layout: GradeLayout = None,
        score_blade_chunk_size: int = 16,
        score_precompute_limit: int = 8_000_000,
    ):
        super().__init__()
        self.algebra = algebra
        self.head_channels = int(head_channels)
        self.bivector_weight = float(bivector_weight)
        self.layout = layout
        self.score_blade_chunk_size = max(1, int(score_blade_chunk_size))
        self.score_precompute_limit = max(0, int(score_precompute_limit))
        self._precompute_score_tables()

    def _precompute_score_tables(self) -> None:
        alg = self.algebra

        if self.layout is not None and self.layout.dim != alg.dim:
            if self.layout.grades != (1,):
                raise ValueError(
                    "Compact geometric attention scoring currently requires a grade-1 input layout. "
                    "Use dense CliffordAlgebra mode for full multivector attention."
                )
            self._score_mode = "compact"
            self.score_output_layout = alg.layout((0, 2))
            scalar_positions = self.score_output_layout.positions_for_grades((0,), device=alg.device)
            bivector_positions = self.score_output_layout.positions_for_grades((2,), device=alg.device)
            self.register_buffer("_score_scalar_positions", scalar_positions)
            self.register_buffer("_score_bivector_positions", bivector_positions)
            return

        require_dense_kernel_host(alg, "geometric attention dense scoring")
        self._score_mode = "dense"
        dense_dim = alg.dim

        metric_rev = alg.gp_signs[:, 0].to(dtype=alg.dtype) * alg.rev_signs.to(dtype=alg.dtype)
        self.register_buffer("_metric_rev", metric_rev)

        g2_blades = [index for index in range(dense_dim) if index.bit_count() == 2]
        self.n_g2 = len(g2_blades)
        self.register_buffer("_g2_blades", torch.tensor(g2_blades, dtype=torch.long, device=alg.device))
        self.register_buffer("_basis_indices", torch.arange(dense_dim, dtype=torch.long, device=alg.device))

    def forward(self, q_head: torch.Tensor, k_head: torch.Tensor) -> torch.Tensor:
        """Return attention scores for heads shaped ``[B, H, L, Hc, D]``."""
        if self._score_mode == "compact":
            return self._score_compact(q_head, k_head)
        return self._score_dense(q_head, k_head)

    def _score_compact(self, q_head: torch.Tensor, k_head: torch.Tensor) -> torch.Tensor:
        B, H, Lq, Hc, lane_dim = q_head.shape
        Lk = k_head.shape[2]
        q_by_channel = q_head.permute(0, 1, 3, 2, 4).reshape(B, H, Hc, Lq, lane_dim)
        k_by_channel = k_head.permute(0, 1, 3, 2, 4).reshape(B, H, Hc, Lk, lane_dim)
        product = self.algebra.geometric_product(
            q_by_channel,
            k_by_channel,
            left_layout=self.layout,
            right_layout=self.layout,
            output_layout=self.score_output_layout,
            compact_output=True,
            pairwise=True,
        )

        scalar = torch.index_select(product, -1, self._score_scalar_positions.to(device=product.device))
        score_g0 = scalar.sum(dim=(-1, 2))

        bivectors = torch.index_select(product, -1, self._score_bivector_positions.to(device=product.device))
        if bivectors.shape[-1] > 0:
            score_g2 = bivectors.pow(2).sum(dim=(-1, 2)).clamp_min(0.0).sqrt()
        else:
            score_g2 = torch.zeros_like(score_g0)

        scale = math.sqrt(self.head_channels * self.algebra.dim)
        return (score_g0 + self.bivector_weight * score_g2) / scale

    def _score_dense(self, q_head: torch.Tensor, k_head: torch.Tensor) -> torch.Tensor:
        B, H, Lq, Hc, D = q_head.shape
        Lk = k_head.shape[2]
        n_g2 = self.n_g2

        q_weighted = q_head * self._metric_rev
        q_flat = q_weighted.reshape(B, H, Lq, Hc * D)
        k_flat = k_head.reshape(B, H, Lk, Hc * D)
        score_g0 = torch.matmul(q_flat, k_flat.transpose(-2, -1))

        if n_g2 > 0:
            q_2d = q_head.permute(0, 1, 3, 2, 4).reshape(B * H * Hc, Lq, D)

            full_k_g2_elements = B * H * Lk * Hc * n_g2 * D
            if full_k_g2_elements <= self.score_precompute_limit:
                score_g2_sq = self._dense_score_g2_precomputed(q_2d, k_head, B, H, Hc, Lq, Lk, D, n_g2)
            else:
                k_2d = k_head.permute(0, 1, 3, 2, 4).reshape(B * H * Hc, Lk, D)
                score_g2_sq = self._dense_score_g2_chunked(q_2d, k_2d, B, H, Hc, Lq, Lk, D, n_g2)
            score_g2 = score_g2_sq.clamp_min(0.0).sqrt()
        else:
            score_g2 = torch.zeros_like(score_g0)

        scale = math.sqrt(self.head_channels * self.algebra.dim)
        return (score_g0 + self.bivector_weight * score_g2) / scale

    def _dense_score_g2_precomputed(self, q_2d, k_head, B, H, Hc, Lq, Lk, D, n_g2):
        r_vals = self._g2_blades
        b_idx = self._basis_indices.unsqueeze(0) ^ r_vals.unsqueeze(1)
        rev_b = self.algebra.rev_signs[b_idx].to(dtype=k_head.dtype)
        gp_ar = self.algebra.gp_signs[:, r_vals].T.to(dtype=k_head.dtype)
        g2_sign = rev_b * gp_ar

        k_g2 = k_head[..., b_idx] * g2_sign
        k_g2_2d = k_g2.permute(0, 1, 3, 2, 4, 5).reshape(B * H * Hc, Lk * n_g2, D)
        comp = torch.bmm(q_2d, k_g2_2d.transpose(-2, -1))
        comp_sq = comp.reshape(B * H * Hc, Lq, Lk, n_g2).pow(2).sum(-1)
        return comp_sq.reshape(B, H, Hc, Lq, Lk).sum(2)

    def _dense_score_g2_chunked(self, q_2d, k_2d, B, H, Hc, Lq, Lk, D, n_g2):
        score_g2_sq = q_2d.new_zeros(B, H, Lq, Lk)
        for start in range(0, n_g2, self.score_blade_chunk_size):
            end = min(start + self.score_blade_chunk_size, n_g2)
            r_vals = self._g2_blades[start:end]
            b_idx = self._basis_indices.unsqueeze(0) ^ r_vals.unsqueeze(1)
            rev_b = self.algebra.rev_signs[b_idx].to(dtype=k_2d.dtype)
            gp_ar = self.algebra.gp_signs[:, r_vals].T.to(dtype=k_2d.dtype)
            g2_sign = rev_b * gp_ar

            k_shifted = torch.index_select(k_2d, -1, b_idx.reshape(-1))
            k_shifted = k_shifted * g2_sign.reshape(-1)
            k_g2_2d = k_shifted.reshape(B * H * Hc, Lk * (end - start), D)
            comp = torch.bmm(q_2d, k_g2_2d.transpose(-2, -1))
            comp_sq = comp.reshape(B * H * Hc, Lq, Lk, end - start).pow(2).sum(-1)
            score_g2_sq = score_g2_sq + comp_sq.reshape(B, H, Hc, Lq, Lk).sum(2)
        return score_g2_sq
