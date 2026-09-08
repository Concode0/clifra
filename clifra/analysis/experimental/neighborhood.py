# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Exploratory coefficient-space neighborhoods and coordinate-lift comparisons."""

import torch

from clifra.core._kernel.numerics import eps_like
from clifra.core.algebra import AlgebraContext
from clifra.core.config import make_algebra
from clifra.core.tensors import TensorContract

__all__ = ["NeighborhoodBivectorAnalyzer", "compare_coordinate_lifts"]


class NeighborhoodBivectorAnalyzer:
    """Explore normalized vector wedges over Euclidean coefficient neighbors.

    Accepts canonical [N, dim] tensors. Wedges use their grade-1 parts;
    neighbor selection uses all supplied coefficients, including probe outputs.
    Alignment and dissimilarity are research scores in coefficient space.
    """

    def __init__(self, algebra: AlgebraContext, k: int = 8):
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        self.algebra = algebra
        self.k = k

    def _validate(self, mv):
        if mv.ndim != 2 or mv.shape[0] == 0 or mv.shape[1] != self.algebra.dim:
            raise ValueError(f"expected nonempty canonical [N, {self.algebra.dim}] data")
        if not mv.dtype.is_floating_point:
            raise TypeError("neighborhood data must be floating point")

    def _bivector_alignment_tensor(self, mv: torch.Tensor) -> torch.Tensor:
        return self.per_point_bivector_alignment(mv).mean()

    def neighbor_bivector_alignment(self, mv: torch.Tensor) -> float:
        """Return mean within-neighborhood absolute coefficient dot product of normalized wedges."""
        return self._bivector_alignment_tensor(mv).item()

    def _knn(self, mv: torch.Tensor) -> torch.Tensor:
        """Returns k-nearest neighbor indices in multivector coefficient space.

        Args:
            mv (torch.Tensor): ``[N, dim]`` multivectors.

        Returns:
            torch.Tensor: ``[N, k]`` neighbor indices.
        """
        N = mv.shape[0]
        k = min(self.k, N - 1)
        if k <= 0:
            return torch.empty(N, 0, dtype=torch.long, device=mv.device)
        dists = torch.cdist(mv, mv)  # [N, N]
        dists.fill_diagonal_(float("inf"))
        _, idx = dists.topk(k, dim=-1, largest=False)
        return idx  # [N, k]

    def _normalized_neighbor_bivectors(self, mv: torch.Tensor, *, compact_output: bool = False) -> torch.Tensor:
        """Normalize wedges of grade-1 parts over coefficient-space neighbors.

        Returns [N, k, dim] canonical bivectors, or compact grade-2 coefficients.
        Each wedge is divided by max(coefficient norm, epsilon); zero stays zero.
        """
        self._validate(mv)
        N, D = mv.shape
        k = min(self.k, N - 1)
        if self.algebra.n < 2:
            width = 0 if compact_output else D
            return mv.new_zeros(N, 0, width)
        layout = self.algebra.layout((2,))
        if k <= 0 or layout.dim == 0:
            width = layout.dim if compact_output else D
            return mv.new_zeros(N, 0, width)
        nn_idx = self._knn(mv)

        neighbors = mv[nn_idx]  # [N, k, dim]
        xi = mv.unsqueeze(1).expand(N, k, D).reshape(N * k, D)
        xj = neighbors.reshape(N * k, D)

        # Wedge only the grade-1 parts of each point-neighbor pair.
        vectors = TensorContract.canonical(self.algebra.layout((1,)))
        bv_raw = self.algebra.wedge(xi, xj, left=vectors, right=vectors, output=layout)
        bv_norm = bv_raw.norm(dim=-1, keepdim=True).clamp_min(eps_like(bv_raw))
        compact = (bv_raw / bv_norm).reshape(N, k, layout.dim)
        if compact_output:
            return compact
        return layout.full(compact)  # [N, k, dim]

    def mean_neighbor_bivectors(self, mv: torch.Tensor) -> torch.Tensor:
        """Return [N, dim] mean normalized neighbor wedges at each point.

        Wedges use grade-1 parts; neighbor selection uses all coefficients.
        Oppositely oriented wedges may cancel in the mean.
        """
        bv = self._normalized_neighbor_bivectors(mv)  # [N, k, dim]
        if bv.shape[1] == 0:
            return mv.new_zeros(mv.shape[0], mv.shape[-1])
        return bv.mean(dim=1)  # [N, dim]

    def _bivector_dissimilarity_tensor(self, mv: torch.Tensor) -> torch.Tensor:
        """Return differentiable neighbor-bivector dissimilarity.

        Args:
            mv (torch.Tensor): ``[N, dim]`` multivectors.

        Returns:
            torch.Tensor: Scalar neighbor_bivector_dissimilarity in [0, 1].
        """
        bv = self._normalized_neighbor_bivectors(mv, compact_output=True)  # [N, k, grade2_dim]
        N, k, D = bv.shape
        if k == 0 or D == 0:
            return mv.new_zeros(())
        nn_idx = self._knn(mv)  # [N, k_nn]

        bi = bv.unsqueeze(2)  # [N, k, 1, dim]
        bj = bv[nn_idx[:, 0]]  # first neighboring neighborhood
        bj = bj.unsqueeze(1)  # [N, 1, k, dim]

        cross_cos = (bi * bj).sum(dim=-1).abs()  # [N, k, k]
        alignment = cross_cos.mean(dim=(-1, -2))  # [N]

        return (1.0 - alignment.mean()).clamp_min(0.0)

    def neighbor_bivector_dissimilarity(self, mv: torch.Tensor) -> float:
        """Compare each neighborhood with its first coefficient-space neighbor.

        Returns one minus the mean cross-set absolute coefficient dot product of their
        normalized bivectors. The first-neighbor choice is part of this score.
        """
        return self._bivector_dissimilarity_tensor(mv).item()

    def per_point_bivector_alignment(self, mv: torch.Tensor) -> torch.Tensor:
        """Return per-point within-neighborhood normalized-bivector alignment.

        Returns a scalar neighbor_bivector_alignment value per data point, measuring how
        well-aligned that point's normalized neighbor bivectors are.

        Args:
            mv (torch.Tensor): ``[N, dim]`` multivectors.

        Returns:
            torch.Tensor: ``[N]`` neighbor_bivector_alignment scores in [0, 1].
        """
        bv = self._normalized_neighbor_bivectors(mv, compact_output=True)  # [N, k, grade2_dim]
        N, k, D = bv.shape
        if k < 2 or D == 0:
            return mv.new_zeros(N)

        bi = bv.unsqueeze(2)  # [N, k, 1, dim]
        bj = bv.unsqueeze(1)  # [N, 1, k, dim]
        abs_cos = (bi * bj).sum(dim=-1).abs()  # [N, k, k]

        mask = ~torch.eye(k, dtype=torch.bool, device=mv.device)  # [k, k]
        # Mean over off-diagonal pairs per point
        off_diag = abs_cos[:, mask].reshape(N, -1)  # [N, k*(k-1)]
        return off_diag.mean(dim=1) if off_diag.shape[-1] > 0 else mv.new_zeros(N)  # [N]


def compare_coordinate_lifts(data: torch.Tensor, p: int, q: int, *, k: int = 8) -> dict:
    """Compare coordinate lifts preserving the original positive/negative directions.

    Input has shape [N, p+q]; its floating dtype and device are preserved.
    The positive-count extension inserts 1 after the p positive coordinates
    and uses Cl(p+1,q). The negative-count extension appends 0 and uses Cl(p,q+1).
    Returns signatures and neighborhood scores for the three embeddings.
    There is no threshold verdict or automatic preferred representation.
    """
    for name, count in (("p", p), ("q", q)):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] != p + q:
        raise ValueError("data must have nonempty shape [N, p+q]")
    if not data.dtype.is_floating_point:
        raise TypeError("coordinate data must be floating point")
    results = {}
    for key, positive, negative, fill in (
        ("original", p, q, None),
        ("positive_count_extension", p + 1, q, 1.0),
        ("negative_count_extension", p, q + 1, 0.0),
    ):
        algebra = make_algebra(positive, negative, device=data.device, dtype=data.dtype)
        if fill is None:
            coordinates = data
        else:
            insertion = p if positive > p else p + q
            coordinates = torch.cat(
                (data[:, :insertion], data.new_full((len(data), 1), fill), data[:, insertion:]), dim=-1
            )
        mv = algebra.layout((1,)).full(coordinates)
        neighborhood = NeighborhoodBivectorAnalyzer(algebra, k=k)
        results[key] = {
            "algebra_signature": (positive, negative),
            "neighbor_bivector_alignment": neighborhood.neighbor_bivector_alignment(mv),
            "neighbor_bivector_dissimilarity": neighborhood.neighbor_bivector_dissimilarity(mv),
        }
    return results
