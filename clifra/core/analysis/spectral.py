# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Spectral analysis of multivector data in a Clifford algebra.

Computes grade-energy spectrum, mean-bivector magnitude, and optionally the
eigenvalue spectrum of the geometric-product operator.
"""

from typing import Optional

import torch

from clifra.core.foundation.module import AlgebraLike
from clifra.core.runtime.metric import hermitian_grade_spectrum

from ._types import CONSTANTS, SpectralResult
from ._utils import (
    declared_full_product_kwargs,
    feasibility_record,
    full_grades,
    full_matrix_feasibility,
    full_product_feasibility,
)


class SpectralAnalyzer:
    """Spectral analysis of multivector data.

    Three independent analyses are combined:

    1. **Grade energy spectrum** -- population-level distribution of
       Hermitian grade energy across all grades.
    2. **Bivector field spectrum** -- norm of the mean bivector field.
    3. **GP operator spectrum** -- eigenvalues of the left-multiplication
       operator :math:`L_x(y) = x \\cdot y` (only for small algebras).
    """

    def __init__(self, algebra: AlgebraLike):
        self.algebra = algebra

    def analyze(self, mv_data: torch.Tensor) -> SpectralResult:
        """Full spectral analysis.

        Args:
            mv_data: Multivector data.  Accepted shapes:

                * ``[N, dim]`` -- single-channel batch.
                * ``[N, C, dim]`` -- multi-channel batch (channels are
                  averaged before bivector / GP analysis).

        Returns:
            :class:`SpectralResult`.
        """
        if mv_data.ndim == 2:
            mv_data = mv_data.unsqueeze(1)  # [N, 1, dim]

        grade_energy = self.grade_energy_spectrum(mv_data)
        bv_spectrum, simple_comps, skipped = self._bivector_field_spectrum_with_skips(mv_data)

        gp_eigs = None
        gp_matrix = full_matrix_feasibility(
            self.algebra,
            role="gp_operator_spectrum",
            max_entries=CONSTANTS.gp_spectrum_matrix_entries,
            matrix_kind="eigensolver",
        )
        gp_product = full_product_feasibility(
            self.algebra,
            role="gp_operator_spectrum",
            op="gp",
            max_pairs=CONSTANTS.gp_spectrum_product_pairs,
        )
        if gp_matrix and gp_product:
            gp_eigs = self.gp_operator_spectrum(mv_data)
        else:
            skipped["gp_operator_spectrum"] = {
                "reason": _first_skip_reason(gp_matrix, gp_product),
                "checks": {
                    "eigensolver_matrix": feasibility_record(gp_matrix),
                    "product": feasibility_record(gp_product),
                },
            }

        return SpectralResult(
            grade_energy=grade_energy,
            bivector_spectrum=bv_spectrum,
            simple_components=simple_comps,
            gp_eigenvalues=gp_eigs,
            skipped=skipped,
        )

    def grade_energy_spectrum(self, mv_data: torch.Tensor) -> torch.Tensor:
        """Mean Hermitian grade energy across the batch.

        Args:
            mv_data: ``[N, C, dim]`` multivector data.

        Returns:
            ``[n+1]`` tensor of mean grade energies.
        """
        # hermitian_grade_spectrum expects [..., dim]
        # Average over channels first to get [N, dim]
        flat = mv_data.mean(dim=1)  # [N, dim]
        spectrum = hermitian_grade_spectrum(self.algebra, flat, grades=full_grades(self.algebra))  # [N, n+1]
        return spectrum.mean(dim=0)  # [n+1]

    def bivector_field_spectrum(self, mv_data: torch.Tensor) -> tuple:
        """Return the mean-bivector magnitude and representative component.

        Args:
            mv_data: ``[N, C, dim]`` multivector data.

        Returns:
            ``(spectrum, components)`` where *spectrum* is a 1-D tensor
            containing the mean-bivector norm and *components* contains the
            corresponding full-layout mean bivector.
        """
        spectrum, components, _ = self._bivector_field_spectrum_with_skips(mv_data)
        return spectrum, components

    def _bivector_field_spectrum_with_skips(self, mv_data: torch.Tensor) -> tuple:
        """Return bivector spectrum plus feasibility skip metadata."""
        flat = mv_data.mean(dim=1)  # [N, dim]
        mean_mv = flat.mean(dim=0)  # [dim]
        skipped = {}

        if self.algebra.n < 2:
            skipped["bivector_field_spectrum"] = {
                "reason": "grade_absent",
                "details": {"n": int(self.algebra.n), "required_grade": 2},
            }
            return (
                torch.zeros(1, device=mv_data.device),
                [torch.zeros_like(mean_mv)],
                skipped,
            )

        # Extract grade-2 (bivector) part
        bv_layout = self.algebra.layout((2,))
        mean_bv_compact = self.algebra.grade_projection(mean_mv, 2, active_output=True)
        mean_bv = bv_layout.full(mean_bv_compact)

        bv_norm = mean_bv_compact.norm()
        if bv_norm <= torch.finfo(mean_bv_compact.dtype).eps:
            return (
                torch.zeros(1, device=mv_data.device),
                [torch.zeros_like(mean_bv)],
                skipped,
            )
        return bv_norm.reshape(1), [mean_bv], skipped

    def gp_operator_spectrum(self, mv_data: torch.Tensor, n_samples: Optional[int] = None) -> torch.Tensor:
        """Eigenvalue magnitudes of the left-multiplication operator.

        For a subsample of data points, constructs the explicit matrix
        representation of :math:`L_x(y) = x \\cdot y` and computes
        eigenvalues.

        Args:
            mv_data: ``[N, C, dim]`` multivector data.
            n_samples: Number of data points to sample.

        Returns:
            Sorted (descending) eigenvalue magnitudes.
        """
        if n_samples is None:
            n_samples = CONSTANTS.gp_spectrum_n_samples
        dim = self.algebra.dim
        flat = mv_data.mean(dim=1)  # [N, dim]
        N = flat.shape[0]

        k = min(N, n_samples)
        if k < N:
            idx = torch.randperm(N, device=flat.device)[:k]
            flat = flat[idx]

        # Build GP matrix: L[:, j] = gp(mean_x, e_j)
        basis = torch.eye(dim, device=flat.device, dtype=flat.dtype)
        mean_x = flat.mean(dim=0)  # [dim]
        # Batched GP: expand mean_x to [dim, dim], basis is [dim, dim]
        # Result[j, :] = gp(mean_x, e_j) = L[:, j], so transpose
        L = self.algebra.geometric_product(
            mean_x.unsqueeze(0).expand(dim, -1),
            basis,
            **declared_full_product_kwargs(self.algebra),
        ).T

        eigvals = torch.linalg.eigvals(L)  # complex
        magnitudes = eigvals.abs()
        return magnitudes.sort(descending=True).values


def _first_skip_reason(*checks) -> str:
    for check in checks:
        if not check:
            return check.reason
    return "ok"
