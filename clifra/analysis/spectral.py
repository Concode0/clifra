# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Grade measurements and sample-mean multiplication spectra."""

from dataclasses import dataclass, field

import torch

from clifra.core._kernel.planning.resources import ResourceLimits
from clifra.core.algebra import AlgebraContext
from clifra.utils.mps import safe_linalg_eigvals

from ._observations import canonical_observations
from ._resources import budget_check_record, check_full_matrix_budget, check_full_product_budget, first_skip_reason

__all__ = ["SpectralAnalyzer", "SpectralResult"]

_LEFT_MULTIPLICATION_LIMITS = ResourceLimits(max_lanes=1 << 10, max_pairs=1 << 20)


@dataclass
class SpectralResult:
    """Measurements of canonical observations.

    ``grade_coefficient_energy`` is the mean positive coefficient energy per grade.
    ``mean_bivector`` is the canonical grade-2 projection of the sample mean.
    The optional eigenvalue magnitudes describe left multiplication by that
    same mean, sorted descending. Unavailable measurements have skip records.
    """

    grade_coefficient_energy: torch.Tensor
    mean_bivector: torch.Tensor
    left_multiplication_eigenvalue_magnitudes: torch.Tensor | None
    skipped: dict[str, dict] = field(default_factory=dict)


class SpectralAnalyzer:
    """Measure nonempty canonical [N, dim] floating-point observations in a declared algebra."""

    def __init__(self, algebra: AlgebraContext):
        self.algebra = algebra

    def analyze(self, mv_data: torch.Tensor) -> SpectralResult:
        flat = canonical_observations(mv_data, self.algebra)
        spectrum, skipped = self._left_multiplication_eigenvalue_magnitudes(flat)
        return SpectralResult(self.grade_coefficient_energy(flat), self.mean_bivector(flat), spectrum, skipped)

    def grade_coefficient_energy(self, mv_data: torch.Tensor) -> torch.Tensor:
        """Return [n+1] mean positive coefficient energies."""
        flat = canonical_observations(mv_data, self.algebra)
        return self.algebra.lane_grade_energy(flat, input=self.algebra.layout()).mean(dim=0)

    def mean_bivector(self, mv_data: torch.Tensor) -> torch.Tensor:
        """Return the canonical [dim] grade-2 sample mean, without thresholding."""
        flat = canonical_observations(mv_data, self.algebra)
        mean = flat.mean(dim=0)
        if self.algebra.n < 2:
            return torch.zeros_like(mean)
        layout = self.algebra.layout((2,))
        return layout.full(self.algebra.grade_projection(mean, output=layout))

    def left_multiplication_eigenvalue_magnitudes(self, mv_data: torch.Tensor) -> torch.Tensor | None:
        """Return sorted sample-mean left-multiplication eigenvalue magnitudes, or None."""
        spectrum, _ = self._left_multiplication_eigenvalue_magnitudes(canonical_observations(mv_data, self.algebra))
        return spectrum

    def _left_multiplication_eigenvalue_magnitudes(
        self, flat: torch.Tensor
    ) -> tuple[torch.Tensor | None, dict[str, dict]]:
        role = "left_multiplication_eigenvalue_magnitudes"
        matrix = check_full_matrix_budget(
            self.algebra, role=role, limits=_LEFT_MULTIPLICATION_LIMITS, matrix_kind="eigensolver", dtype=flat.dtype
        )
        product = check_full_product_budget(
            self.algebra, role=role, op="geometric_product", limits=_LEFT_MULTIPLICATION_LIMITS
        )
        if not matrix or not product:
            return None, {
                role: {
                    "reason": first_skip_reason(matrix, product),
                    "checks": {
                        "eigensolver_matrix": budget_check_record(matrix),
                        "product": budget_check_record(product),
                    },
                }
            }
        basis = torch.eye(self.algebra.dim, device=flat.device, dtype=flat.dtype)
        layout = self.algebra.layout()
        operator = self.algebra.geometric_product(
            flat.mean(dim=0).expand_as(basis),
            basis,
            left=layout,
            right=layout,
            output=layout,
        ).T
        return safe_linalg_eigvals(operator).abs().sort(descending=True).values, {}
