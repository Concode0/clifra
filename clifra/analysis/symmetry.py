# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Operational transformation diagnostics for multivector data.

Computes low-energy grade-1 directions, odd-grade energy fraction, reflection
scores, and near-zero commutator modes.
"""

from typing import Dict, List, Optional, Tuple

import torch

from clifra.core.foundation.basis import operation_coefficient
from clifra.core.foundation.module import AlgebraLike
from clifra.core.foundation.numerics import eps_like
from clifra.core.runtime.tensors import LaneStorage

from ._types import CONSTANTS, CommutatorResult, TransformationDiagnosticsResult
from ._utils import (
    full_grades,
    full_layout_for_analysis,
    grade_layout_for_analysis,
    product_feasibility,
)
from .policy import feasibility_record


class TransformationDiagnosticsAnalyzer:
    """Compute experimental coefficient and transformation diagnostics.

    The methods report explicitly defined energy, distribution-distance, and
    commutator statistics. These quantities characterize the implemented
    transformations without establishing metric nullity or invariance under a
    continuous symmetry group.

    Args:
        algebra: Layout-first algebra host.
        low_energy_threshold: Normalized grade-1 coefficient-energy threshold below
            which a basis direction is reported.
    """

    def __init__(
        self,
        algebra: AlgebraLike,
        low_energy_threshold: float = CONSTANTS.low_energy_vector_threshold,
    ):
        self.algebra = algebra
        self.low_energy_threshold = low_energy_threshold

    def analyze(
        self,
        mv_data: torch.Tensor,
        commutator_result: Optional[CommutatorResult] = None,
    ) -> TransformationDiagnosticsResult:
        """Compute the complete operational transformation report.

        Args:
            mv_data: Multivector data.  Accepted shapes:

                * ``[N, dim]`` -- single-channel batch.
                * ``[N, C, dim]`` -- multi-channel batch (channels
                  averaged for scalar-valued analyses).

            commutator_result: Pre-computed commutator results for
                optional adjoint-mode counting.

        Returns:
            :class:`TransformationDiagnosticsResult`.
        """
        if mv_data.ndim == 3:
            flat = mv_data.mean(dim=1)
        else:
            flat = mv_data  # [N, dim]

        low_energy_directions, normalized_vector_energy = self.low_energy_vector_directions(flat)
        odd_fraction = self.odd_grade_energy_fraction(flat)
        reflection_scores, reflection_score_skipped = self._basis_reflection_scores_with_skips(flat)
        near_commuting_count, commuting_skipped = self._near_commuting_modes_with_skips(
            flat, commutator_result=commutator_result
        )
        skipped = {}
        skipped.update(reflection_score_skipped)
        skipped.update(commuting_skipped)

        return TransformationDiagnosticsResult(
            low_energy_vector_directions=low_energy_directions,
            normalized_vector_energy=normalized_vector_energy,
            odd_grade_energy_fraction=odd_fraction,
            basis_reflection_scores=reflection_scores,
            near_commuting_mode_count=near_commuting_count,
            skipped=skipped,
        )

    def low_energy_vector_directions(self, mv_data: torch.Tensor) -> Tuple[List[int], torch.Tensor]:
        """Return low-energy grade-1 coordinate directions and their energies.

        For each basis vector ``e_i``, computes the mean squared
        projection energy of the data onto that direction.  Directions
        below :attr:`low_energy_threshold` are reported. This is an observed-data
        diagnostic and is unrelated to null generators in the algebra's
        signature.

        Args:
            mv_data: ``[N, dim]`` multivector data.

        Returns:
            ``(indices, normalized_energy)`` with one energy per vector lane.
        """
        g1_idx = self.algebra.grade_indices((1,), device=mv_data.device)

        # Energy on each grade-1 component
        g1_coeffs = mv_data[:, g1_idx]  # [N, n]
        scores = (g1_coeffs**2).mean(dim=0)  # [n]

        # Normalize so max is 1
        smax = scores.max()
        if smax > self.algebra.eps_sq:
            scores = scores / smax

        low_energy_directions = (scores < self.low_energy_threshold).nonzero(as_tuple=True)[0]
        return low_energy_directions.tolist(), scores

    def odd_grade_energy_fraction(self, mv_data: torch.Tensor) -> float:
        """Measure the fraction of coefficient energy in odd grades.

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

    def basis_reflection_scores(self, mv_data: torch.Tensor) -> List[Dict]:
        """Compute coefficient-distribution distance after each basis reflection.

        For each basis vector ``e_i``, reflects the data
        ``x' = -e_i x e_i^{-1}`` and compares the reflected
        distribution to the original.

        Returns:
            List of dicts ``{"direction": i, "score": float}`` sorted
            by score ascending. Lower means the independently sorted
            coefficient distributions are closer under this statistic.
        """
        results, _ = self._basis_reflection_scores_with_skips(mv_data)
        return results

    def _basis_reflection_scores_with_skips(self, mv_data: torch.Tensor) -> tuple[List[Dict], dict[str, dict]]:
        """Return basis-reflection scores plus feasibility skip metadata."""
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
            skipped["basis_reflection_scores"] = {
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

    def near_commuting_mode_count(
        self,
        mv_data: torch.Tensor,
        commutator_result: Optional[CommutatorResult] = None,
        threshold: Optional[float] = None,
    ) -> int:
        """Count modes below a normalized commutator threshold.

        Without a precomputed commutator result, the method counts basis
        bivectors whose mean normalized commutator norm is below ``threshold``.

        If a :class:`CommutatorResult` is provided, the method instead counts
        near-zero values in its full adjoint eigenvalue magnitudes. The two branches do
        not necessarily count modes in spaces of the same dimension.

        Args:
            mv_data: ``[N, dim]`` multivector data.
            commutator_result: Pre-computed commutator analysis.
            threshold: Normalized commutator norm below which a bivector
                is counted as near-commuting.

        Returns:
            Number of modes below the selected threshold.
        """
        dim, _ = self._near_commuting_modes_with_skips(
            mv_data,
            commutator_result=commutator_result,
            threshold=threshold,
        )
        return dim

    def _near_commuting_modes_with_skips(
        self,
        mv_data: torch.Tensor,
        commutator_result: Optional[CommutatorResult] = None,
        threshold: Optional[float] = None,
    ) -> tuple[int, dict[str, dict]]:
        """Return the near-commuting mode count plus feasibility metadata."""
        if threshold is None:
            threshold = CONSTANTS.near_commuting_mode_threshold
        skipped = {}

        if commutator_result is not None:
            spec = commutator_result.adjoint_eigenvalue_magnitudes
            if spec.numel() > 0:
                max_val = spec.abs().max().clamp_min(eps_like(spec))
                return int((spec.abs() / max_val < threshold).sum().item()), skipped

        # Compute from scratch: test each basis bivector
        N = mv_data.shape[0]
        bivector_layout = grade_layout_for_analysis(self.algebra, (2,))
        full_layout = full_layout_for_analysis(self.algebra)
        commutator_feasible = product_feasibility(
            self.algebra,
            role="near_commuting_mode_basis_bivectors",
            op="commutator_product",
            left_layout=bivector_layout,
            right_layout=full_layout,
            output_layout=full_layout,
            max_pairs=CONSTANTS.near_commuting_mode_product_pairs,
        )
        if not commutator_feasible:
            skipped["near_commuting_modes"] = feasibility_record(commutator_feasible)
            return 0, skipped

        if bivector_layout.dim == 0:
            skipped["near_commuting_modes"] = {
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
