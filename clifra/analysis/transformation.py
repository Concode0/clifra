# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Declared-grade energies and basis-reflection distribution measurements."""

from dataclasses import dataclass, field

import torch

from clifra.core._kernel.basis import operation_coefficient
from clifra.core._kernel.numerics import eps_like
from clifra.core._kernel.planning.resources import ResourceLimits
from clifra.core.algebra import AlgebraContext

from ._observations import canonical_observations
from ._resources import budget_check_record, check_product_budget, first_skip_reason

__all__ = ["TransformationDiagnosticsAnalyzer", "TransformationDiagnosticsResult"]

_REFLECTION_LIMITS = ResourceLimits(max_lanes=1 << 12, max_pairs=1 << 16)


@dataclass
class TransformationDiagnosticsResult:
    """Measurements of canonical observations, without inferred labels.

    Vector energy is mean squared grade-1 coefficients, shape [n]. Odd-grade
    fraction averages each observation's energy fraction (zero contributes zero).
    Reflection scores compare independently sorted coefficient marginals after
    grade-extended basis reflection, normalized by mean input coefficient energy.
    A null direction has score None; a resource-skipped collection is None.
    """

    vector_coefficient_energy: torch.Tensor
    odd_grade_energy_fraction: float
    basis_reflection_marginal_scores: list[dict] | None
    skipped: dict[str, dict] = field(default_factory=dict)


class TransformationDiagnosticsAnalyzer:
    """Measure nonempty canonical [N, dim] floating-point observations."""

    def __init__(self, algebra: AlgebraContext):
        self.algebra = algebra

    def analyze(self, mv_data: torch.Tensor) -> TransformationDiagnosticsResult:
        flat = canonical_observations(mv_data, self.algebra)
        scores, skipped = self._reflection_marginal_scores(flat)
        return TransformationDiagnosticsResult(
            self.vector_coefficient_energy(flat), self.odd_grade_energy_fraction(flat), scores, skipped
        )

    def vector_coefficient_energy(self, mv_data: torch.Tensor) -> torch.Tensor:
        """Return [n] mean squared vector coefficients, without normalization."""
        flat = canonical_observations(mv_data, self.algebra)
        if self.algebra.n == 0:
            return flat.new_empty(0)
        return self.algebra.layout((1,)).compact(flat).square().mean(dim=0)

    def odd_grade_energy_fraction(self, mv_data: torch.Tensor) -> float:
        """Mean odd/total coefficient energy per observation; zero contributes zero."""
        flat = canonical_observations(mv_data, self.algebra)
        layout = self.algebra.layout()
        odd = (flat - self.algebra.grade_involution(flat, input=layout, output=layout)) / 2
        total = flat.square().sum(dim=-1)
        denominator = torch.where(total > 0, total, torch.ones_like(total))
        return (odd.square().sum(dim=-1) / denominator).mean().item()

    def basis_reflection_marginal_scores(self, mv_data: torch.Tensor) -> list[dict] | None:
        """Return normalized squared marginal-difference scores, or None at resource limits.

        Invertible directions are sorted by score; noninvertible directions follow
        with score None. Reflection is e_i alpha(x) e_i^{-1} on all grades.
        """
        return self._reflection_marginal_scores(canonical_observations(mv_data, self.algebra))[0]

    def _reflection_marginal_scores(self, flat):
        if self.algebra.n == 0:
            return [], {}
        vector = self.algebra.spec.layout((1,))
        full = self.algebra.spec.full_layout()
        left = check_product_budget(
            role="basis_reflection",
            op="geometric_product",
            left_layout=vector,
            right_layout=full,
            output_layout=full,
            limits=_REFLECTION_LIMITS,
        )
        right = check_product_budget(
            role="basis_reflection",
            op="geometric_product",
            left_layout=full,
            right_layout=vector,
            output_layout=full,
            limits=_REFLECTION_LIMITS,
        )
        if not left or not right:
            return None, {
                "basis_reflection_marginal_scores": {
                    "reason": first_skip_reason(left, right),
                    "checks": {
                        "left_vector_full_product": budget_check_record(left),
                        "right_full_vector_product": budget_check_record(right),
                    },
                }
            }
        reflected, valid = self._planned_basis_reflections(flat)
        marginal_squared_difference = (
            (flat.sort(dim=0).values.unsqueeze(0) - reflected.sort(dim=1).values).square().sum(dim=-1).mean(dim=-1)
        )
        denominator = flat.square().sum(dim=-1).mean().clamp_min(eps_like(flat))
        scores = marginal_squared_difference / denominator
        results = [{"direction": i, "score": scores[i].item() if valid[i] else None} for i in range(self.algebra.n)]
        results.sort(
            key=lambda item: (item["score"] is None, item["score"] if item["score"] is not None else item["direction"])
        )
        invalid = (~valid).nonzero(as_tuple=True)[0].tolist()
        skipped = {}
        if invalid:
            skipped["basis_reflection_marginal_scores"] = {
                "reason": "noninvertible_directions",
                "details": {"directions": invalid},
            }
        return results, skipped

    def _planned_basis_reflections(self, flat):
        vector = self.algebra.layout((1,))
        full = self.algebra.layout()
        basis = torch.eye(self.algebra.n, device=flat.device, dtype=flat.dtype)
        signs = flat.new_tensor(
            [
                operation_coefficient(i, i, self.algebra.p, self.algebra.q, self.algebra.r, "geometric_product")
                for i in vector.basis_indices
            ]
        )
        valid = signs != 0
        inverse = basis * signs[:, None]
        alpha = self.algebra.grade_involution(flat, input=full, output=full)
        first = self.algebra.geometric_product(
            basis[:, None, :], alpha[None, :, :], left=vector, right=full, output=full
        )
        reflected = self.algebra.geometric_product(first, inverse[:, None, :], left=full, right=vector, output=full)
        return reflected, valid
