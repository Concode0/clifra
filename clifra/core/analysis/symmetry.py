# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Symmetry and null-direction detection in Clifford algebras.

Detects null directions, grade-involution symmetry, reflection
symmetries, and continuous symmetries (via commutator analysis).
"""

from typing import Dict, List, Optional, Tuple

import torch

from clifra.core.foundation.basis import operation_coefficient
from clifra.core.foundation.module import AlgebraLike
from clifra.core.foundation.numerics import eps_like
from clifra.core.runtime.tensors import LaneStorage

from ._types import CONSTANTS, CommutatorResult, SymmetryResult
from ._utils import (
    full_grades,
    full_layout_for_analysis,
    grade_layout_for_analysis,
    product_feasibility,
)
from .policy import feasibility_record


class SymmetryDetector:
    """Detect symmetries, null directions, and invariances.

    Args:
        algebra: Layout-first algebra host.
        null_threshold: Energy threshold below which a direction is
            considered effectively null.
    """

    def __init__(
        self,
        algebra: AlgebraLike,
        null_threshold: float = CONSTANTS.symmetry_null_threshold,
    ):
        self.algebra = algebra
        self.null_threshold = null_threshold

    def analyze(
        self,
        mv_data: torch.Tensor,
        commutator_result: Optional[CommutatorResult] = None,
    ) -> SymmetryResult:
        """Full symmetry analysis.

        Args:
            mv_data: Multivector data.  Accepted shapes:

                * ``[N, dim]`` -- single-channel batch.
                * ``[N, C, dim]`` -- multi-channel batch (channels
                  averaged for scalar-valued analyses).

            commutator_result: Pre-computed commutator results for
                continuous-symmetry refinement.

        Returns:
            :class:`SymmetryResult`.
        """
        if mv_data.ndim == 3:
            flat = mv_data.mean(dim=1)
        else:
            flat = mv_data  # [N, dim]

        null_dirs, null_scores = self.detect_null_directions(flat)
        inv_sym = self.detect_involution_symmetry(flat)
        refl_syms, reflection_skipped = self._reflection_symmetries_with_skips(flat)
        cont_dim, continuous_skipped = self._continuous_symmetries_with_skips(
            flat, commutator_result=commutator_result
        )
        skipped = {}
        skipped.update(reflection_skipped)
        skipped.update(continuous_skipped)

        return SymmetryResult(
            null_directions=null_dirs,
            null_scores=null_scores,
            involution_symmetry=inv_sym,
            reflection_symmetries=refl_syms,
            continuous_symmetry_dim=cont_dim,
            skipped=skipped,
        )

    def detect_null_directions(self, mv_data: torch.Tensor) -> Tuple[List[int], torch.Tensor]:
        """Detect effectively null basis-vector directions.

        For each basis vector ``e_i``, computes the mean squared
        projection energy of the data onto that direction.  Directions
        below :attr:`null_threshold` are flagged.

        Args:
            mv_data: ``[N, dim]`` multivector data.

        Returns:
            ``(null_indices, scores)`` where *scores* is ``[n]`` and
            *null_indices* lists those with score < threshold.
        """
        g1_idx = self.algebra.grade_indices((1,), device=mv_data.device)

        # Energy on each grade-1 component
        g1_coeffs = mv_data[:, g1_idx]  # [N, n]
        scores = (g1_coeffs**2).mean(dim=0)  # [n]

        # Normalize so max is 1
        smax = scores.max()
        if smax > self.algebra.eps_sq:
            scores = scores / smax

        null_dirs = (scores < self.null_threshold).nonzero(as_tuple=True)[0]
        return null_dirs.tolist(), scores

    def detect_involution_symmetry(self, mv_data: torch.Tensor) -> float:
        """Measure grade-involution symmetry of the data distribution.

        Computes the fraction of total energy in odd-grade components:

            score = E[ ||x_odd||^2 / ||x||^2 ]

        where ``x_odd = (x - alpha(x)) / 2`` and ``alpha`` is the grade
        involution (flips odd-grade components).

        Returns:
            Score in ``[0, 1]``.  0 means the data lives entirely in
            the even sub-algebra; 1 means entirely odd grades.
        """
        alpha = self.algebra.grade_involution(
            mv_data,
            input_grades=full_grades(self.algebra),
            output_storage=LaneStorage.COMPACT,
        )  # [N, dim]
        odd_part = (mv_data - alpha) / 2.0
        odd_energy = (odd_part**2).sum(dim=-1)
        total_energy = (mv_data**2).sum(dim=-1).clamp_min(eps_like(mv_data))
        return (odd_energy / total_energy).mean().item()

    def detect_reflection_symmetries(self, mv_data: torch.Tensor) -> List[Dict]:
        """Test reflection symmetry along each basis-vector direction.

        For each basis vector ``e_i``, reflects the data
        ``x' = -e_i x e_i^{-1}`` and compares the reflected
        distribution to the original.

        Returns:
            List of dicts ``{"direction": i, "score": float}`` sorted
            by score ascending (lower = more symmetric).
        """
        results, _ = self._reflection_symmetries_with_skips(mv_data)
        return results

    def _reflection_symmetries_with_skips(self, mv_data: torch.Tensor) -> tuple[List[Dict], dict[str, dict]]:
        """Return reflection symmetries plus feasibility skip metadata."""
        n = self.algebra.n
        dim = self.algebra.dim
        N = mv_data.shape[0]
        skipped = {}
        vector_layout = grade_layout_for_analysis(self.algebra, (1,))
        full_layout = full_layout_for_analysis(self.algebra)
        left_feasible = product_feasibility(
            self.algebra,
            role="basis_reflection",
            op="gp",
            left_layout=vector_layout,
            right_layout=full_layout,
            output_layout=full_layout,
            max_pairs=CONSTANTS.reflection_product_pairs,
        )
        right_feasible = product_feasibility(
            self.algebra,
            role="basis_reflection",
            op="gp",
            left_layout=full_layout,
            right_layout=vector_layout,
            output_layout=full_layout,
            max_pairs=CONSTANTS.reflection_product_pairs,
        )
        if not left_feasible or not right_feasible:
            skipped["reflection_symmetries"] = {
                "reason": _first_skip_reason(left_feasible, right_feasible),
                "checks": {
                    "left_vector_full_product": feasibility_record(left_feasible),
                    "right_full_vector_product": feasibility_record(right_feasible),
                },
            }
            return [], skipped

        reflected, valid = self._planned_basis_reflections(mv_data)

        # Distributional distance for all directions at once
        orig_sorted = mv_data.sort(dim=0).values.unsqueeze(0).expand(n, N, dim)
        refl_sorted = reflected.sort(dim=1).values  # sort along N
        dist = ((orig_sorted - refl_sorted) ** 2).sum(dim=-1).mean(dim=-1)  # [n]
        norm = (mv_data**2).sum(dim=-1).mean().clamp_min(eps_like(mv_data))
        scores = dist / norm  # [n]
        scores = torch.where(valid, scores, torch.full_like(scores, float("inf")))

        results = [{"direction": i, "score": scores[i].item()} for i in range(n)]
        results.sort(key=lambda r: r["score"])
        return results, skipped

    def _planned_basis_reflections(
        self,
        mv_data: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reflect by compact basis vectors through planned products."""
        n = self.algebra.n
        dim = self.algebra.dim
        N = mv_data.shape[0]
        vector_layout = self.algebra.layout((1,))
        full_layout = self.algebra.layout(full_grades(self.algebra))
        basis_vecs = torch.eye(n, device=mv_data.device, dtype=mv_data.dtype)
        blade_indices = vector_layout.indices_tensor(device=mv_data.device)
        signs = torch.tensor(
            [
                operation_coefficient(int(index), int(index), self.algebra.p, self.algebra.q, self.algebra.r, "gp")
                for index in blade_indices.tolist()
            ],
            device=mv_data.device,
            dtype=mv_data.dtype,
        )
        valid = signs.abs() > eps_like(signs)
        inv_basis = basis_vecs * signs.view(-1, 1)

        left = basis_vecs.unsqueeze(1).expand(n, N, vector_layout.dim).reshape(n * N, vector_layout.dim)
        values = mv_data.unsqueeze(0).expand(n, N, dim).reshape(n * N, dim)
        right = inv_basis.unsqueeze(1).expand(n, N, vector_layout.dim).reshape(n * N, vector_layout.dim)
        first = self.algebra.geometric_product(
            left,
            values,
            left_layout=vector_layout,
            right_layout=full_layout,
            output_layout=full_layout,
        )
        reflected = -self.algebra.geometric_product(
            first,
            right,
            left_layout=full_layout,
            right_layout=vector_layout,
            output_layout=full_layout,
        ).reshape(n, N, dim)
        return reflected, valid

    def detect_continuous_symmetries(
        self,
        mv_data: torch.Tensor,
        commutator_result: Optional[CommutatorResult] = None,
        threshold: Optional[float] = None,
    ) -> int:
        """Estimate the dimension of the continuous symmetry group.

        A bivector ``B_j`` generates a continuous symmetry if
        ``E[||[B_j, x_i]||]`` is near zero for all data points.

        If a :class:`CommutatorResult` is provided, its exchange
        spectrum is used directly (eigenvalues near zero -> symmetry
        generators).  Otherwise the computation is done from scratch.

        Args:
            mv_data: ``[N, dim]`` multivector data.
            commutator_result: Pre-computed commutator analysis.
            threshold: Normalized commutator norm below which a bivector
                is considered a symmetry generator.

        Returns:
            Estimated dimension of the continuous symmetry group.
        """
        dim, _ = self._continuous_symmetries_with_skips(
            mv_data,
            commutator_result=commutator_result,
            threshold=threshold,
        )
        return dim

    def _continuous_symmetries_with_skips(
        self,
        mv_data: torch.Tensor,
        commutator_result: Optional[CommutatorResult] = None,
        threshold: Optional[float] = None,
    ) -> tuple[int, dict[str, dict]]:
        """Return continuous-symmetry dimension plus feasibility skip metadata."""
        if threshold is None:
            threshold = CONSTANTS.continuous_symmetry_threshold
        skipped = {}

        if commutator_result is not None:
            spec = commutator_result.exchange_spectrum
            if spec.numel() > 0:
                max_val = spec.abs().max().clamp_min(eps_like(spec))
                return int((spec.abs() / max_val < threshold).sum().item()), skipped

        # Compute from scratch: test each basis bivector
        N = mv_data.shape[0]
        bivector_layout = grade_layout_for_analysis(self.algebra, (2,))
        full_layout = full_layout_for_analysis(self.algebra)
        commutator_feasible = product_feasibility(
            self.algebra,
            role="continuous_symmetry_basis_bivectors",
            op="commutator_product",
            left_layout=bivector_layout,
            right_layout=full_layout,
            output_layout=full_layout,
            max_pairs=CONSTANTS.continuous_symmetry_product_pairs,
        )
        if not commutator_feasible:
            skipped["continuous_symmetries"] = feasibility_record(commutator_feasible)
            return 0, skipped

        if bivector_layout.dim == 0:
            skipped["continuous_symmetries"] = {
                "reason": "grade_absent",
                "details": {"n": int(self.algebra.n), "required_grade": 2},
            }
            return 0, skipped

        n_bv = bivector_layout.dim
        bv_bases = torch.eye(n_bv, device=mv_data.device, dtype=mv_data.dtype)

        # Batch commutator: [n_bv, N, dim]
        bv_exp = bv_bases.unsqueeze(1).expand(n_bv, N, bivector_layout.dim)
        mv_exp = mv_data.unsqueeze(0).expand(n_bv, N, full_layout.dim)
        comm = self.algebra.commutator_product(
            bv_exp,
            mv_exp,
            left_layout=bivector_layout,
            right_layout=full_layout,
            output_layout=full_layout,
        )

        comm_norms = comm.norm(dim=-1).mean(dim=-1)  # [n_bv]
        data_norm = mv_data.norm(dim=-1).mean().clamp_min(eps_like(mv_data))

        return int((comm_norms / data_norm < threshold).sum().item()), skipped


def _first_skip_reason(*checks) -> str:
    for check in checks:
        if not check:
            return check.reason
    return "ok"
