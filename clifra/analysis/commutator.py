# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Commutator measurements in an explicitly declared algebra."""

from dataclasses import dataclass, field
from numbers import Integral

import torch

from clifra.core._kernel.numerics import eps_like
from clifra.core._kernel.planning.resources import ResourceLimits
from clifra.core.algebra import AlgebraContext
from clifra.utils.mps import safe_linalg_eigvals

from ._observations import canonical_observations
from ._resources import (
    budget_check_record,
    check_full_matrix_budget,
    check_full_product_budget,
    check_product_budget,
    first_skip_reason,
)

__all__ = ["CommutatorAnalyzer", "CommutatorResult"]

_ADJOINT_LIMITS = ResourceLimits(max_lanes=1 << 8, max_pairs=1 << 16)
_PRODUCT_LIMITS = ResourceLimits(max_lanes=1 << 12, max_pairs=1 << 16)
_CLOSURE_MAX_BIVECTORS = 15


@dataclass
class CommutatorResult:
    """Measurements of canonical observations; None denotes a skipped measurement.

    Pairwise norms have shape [n,n] for vector-coordinate pairs. Adjoint
    eigenvalue magnitudes describe x -> [mean(data), x]. The mean norm uses full data.
    Normalized basis-bivector norms follow the core grade-2 layout order and
    divide mean bracket norms by max(mean data coefficient norm, epsilon).
    """

    vector_pair_commutator_norms: torch.Tensor
    adjoint_eigenvalue_magnitudes: torch.Tensor | None
    mean_commutator_norm: float | None
    basis_bivector_commutator_norm_ratios: torch.Tensor | None
    skipped: dict[str, dict] = field(default_factory=dict)


class CommutatorAnalyzer:
    """Measure nonempty canonical [N, dim] floating-point observations."""

    def __init__(self, algebra: AlgebraContext):
        self.algebra = algebra

    def analyze(self, mv_data: torch.Tensor) -> CommutatorResult:
        flat = canonical_observations(mv_data, self.algebra)
        spectrum, skipped = self._adjoint_eigenvalue_magnitudes(flat)
        mean, mean_skips = self._mean_norm(flat)
        norms, basis_skips = self._basis_norms(flat)
        skipped.update(mean_skips)
        skipped.update(basis_skips)
        return CommutatorResult(self.vector_pair_commutator_norms(flat), spectrum, mean, norms, skipped)

    def vector_pair_commutator_norms(self, mv_data: torch.Tensor) -> torch.Tensor:
        """Return mean commutator norms for pairs of vector coordinates.

        For each pair ``(i, j)`` of the *n* basis-vector directions,
        computes ``E[||[x_i, x_j]||]`` where ``x_i`` is the data
        projected onto ``e_i``.

        Args:
            mv_data: ``[N, dim]`` multivector data.

        Returns:
            ``[n, n]`` symmetric matrix of mean commutator norms.
        """
        mv_data = canonical_observations(mv_data, self.algebra)
        n = self.algebra.n
        device = mv_data.device
        dtype = mv_data.dtype

        N = mv_data.shape[0]

        # All (i, j) pairs with i < j
        i_idx, j_idx = torch.triu_indices(n, n, offset=1, device=device)
        if i_idx.numel() == 0:
            return torch.zeros(n, n, device=device, dtype=dtype)

        g1_idx = self.algebra.layout((1,)).indices_tensor(device=device)
        coeffs = mv_data[:, g1_idx]  # [N, n]
        left = torch.zeros(i_idx.numel(), N, n, device=device, dtype=dtype)
        right = torch.zeros_like(left)
        left.scatter_(-1, i_idx.view(-1, 1, 1).expand(-1, N, 1), coeffs[:, i_idx].T.unsqueeze(-1))
        right.scatter_(-1, j_idx.view(-1, 1, 1).expand(-1, N, 1), coeffs[:, j_idx].T.unsqueeze(-1))

        # Batched compact commutator: [n_pairs, N, grade2_dim]
        comm = self.algebra.commutator_product(
            left,
            right,
            left=self.algebra.layout((1,)),
            right=self.algebra.layout((1,)),
            output=self.algebra.layout((2,)),
        )
        vals = comm.norm(dim=-1).mean(dim=-1)  # [n_pairs]

        matrix = torch.zeros(n, n, device=device, dtype=dtype)
        matrix[i_idx, j_idx] = vals
        matrix[j_idx, i_idx] = vals

        return matrix

    def adjoint_eigenvalue_magnitudes(self, mv_data: torch.Tensor) -> torch.Tensor | None:
        """Sorted full sample-mean adjoint eigenvalue magnitudes, or None at resource limits."""
        return self._adjoint_eigenvalue_magnitudes(canonical_observations(mv_data, self.algebra))[0]

    def _adjoint_eigenvalue_magnitudes(self, flat):
        role = "adjoint_eigenvalue_magnitudes"
        matrix = check_full_matrix_budget(
            self.algebra, role=role, limits=_ADJOINT_LIMITS, matrix_kind="eigensolver", dtype=flat.dtype
        )
        product = check_full_product_budget(self.algebra, role=role, op="commutator_product", limits=_PRODUCT_LIMITS)
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
        layout = self.algebra.layout()
        basis = torch.eye(layout.dim, device=flat.device, dtype=flat.dtype)
        operator = self.algebra.commutator_product(
            flat.mean(dim=0).expand_as(basis),
            basis,
            left=layout,
            right=layout,
            output=layout,
        ).T
        return safe_linalg_eigvals(operator).abs().sort(descending=True).values, {}

    def mean_commutator_norm(self, mv_data: torch.Tensor) -> float | None:
        """Mean full-multivector ||[x, mean(x)]||, or None at resource limits."""
        return self._mean_norm(canonical_observations(mv_data, self.algebra))[0]

    def _mean_norm(self, flat):
        role = "mean_commutator_norm"
        verdict = check_full_product_budget(self.algebra, role=role, op="commutator_product", limits=_PRODUCT_LIMITS)
        if not verdict:
            return None, {role: budget_check_record(verdict)}
        layout = self.algebra.layout()
        brackets = self.algebra.commutator_product(
            flat,
            flat.mean(dim=0).expand_as(flat),
            left=layout,
            right=layout,
            output=layout,
        )
        return brackets.norm(dim=-1).mean().item(), {}

    def basis_bivector_commutator_norm_ratios(self, mv_data: torch.Tensor) -> torch.Tensor | None:
        """Return normalized norms in canonical grade-2 order, or None when skipped."""
        return self._basis_norms(canonical_observations(mv_data, self.algebra))[0]

    def _basis_norms(self, flat):
        role = "basis_bivector_commutator_norm_ratios"
        if self.algebra.n < 2:
            return flat.new_empty(0), {}
        bv = self.algebra.spec.layout((2,))
        full = self.algebra.spec.full_layout()
        verdict = check_product_budget(
            role=role,
            op="commutator_product",
            left_layout=bv,
            right_layout=full,
            output_layout=full,
            limits=_PRODUCT_LIMITS,
        )
        if not verdict:
            return None, {role: budget_check_record(verdict)}
        basis = torch.eye(bv.dim, device=flat.device, dtype=flat.dtype)
        brackets = self.algebra.commutator_product(
            basis[:, None, :],
            flat[None, :, :],
            left=bv,
            right=full,
            output=full,
        )
        denominator = flat.norm(dim=-1).mean().clamp_min(eps_like(flat))
        return brackets.norm(dim=-1).mean(dim=-1) / denominator, {}

    def bivector_bracket_closure(self, blade_indices) -> dict:
        """Measure closure of explicitly supplied coordinate bivectors.

        Indices are unique canonical blade indices of grade 2 (at most 15).
        Returns projected bracket coefficients, mean relative residual over
        nonzero brackets, and the supplied index order. No data selects the span.
        """
        selected_indices = list(blade_indices)
        if any(isinstance(i, bool) or not isinstance(i, Integral) for i in selected_indices):
            raise ValueError("blade_indices must be integer grade-2 blade indices")
        if len(set(selected_indices)) != len(selected_indices):
            raise ValueError("blade_indices must be unique")
        if len(selected_indices) > _CLOSURE_MAX_BIVECTORS:
            raise ValueError(f"at most {_CLOSURE_MAX_BIVECTORS} bivectors may be selected")
        device, dtype = self.algebra.device, self.algebra.dtype
        indices = self.algebra.spec.layout((2,)).basis_indices if self.algebra.n >= 2 else ()
        if any(i not in indices for i in selected_indices):
            raise ValueError("blade_indices must select grade-2 blades in the declared algebra")
        k = len(selected_indices)
        if k < 2:
            return {
                "projected_bracket_coefficients": torch.zeros(k, k, k, device=device, dtype=dtype),
                "mean_relative_closure_residual": 0.0,
                "blade_indices": selected_indices,
            }
        bv_blade_indices = torch.tensor(indices, device=device)
        positions = torch.tensor([indices.index(i) for i in selected_indices], device=device)
        # Build compact basis bivectors.
        B = torch.zeros(k, bv_blade_indices.numel(), device=device, dtype=dtype)
        B[torch.arange(k, device=device), positions] = 1.0

        # Compute coefficients of brackets projected onto the selected span.
        a_idx, b_idx = torch.triu_indices(k, k, offset=1, device=device)

        # Batched compact commutator and grade-2 projection.
        brackets_bv = self.algebra.commutator_product(
            B[a_idx],
            B[b_idx],
            left=self.algebra.layout((2,)),
            right=self.algebra.layout((2,)),
            output=self.algebra.layout((2,)),
        )  # [n_pairs, grade2_dim]

        # Project onto basis: coeffs[p, c] = <bracket_bv_p, B_c>
        coeffs = brackets_bv @ B.T  # [n_pairs, k]

        structure = torch.zeros(k, k, k, device=device, dtype=dtype)
        structure[a_idx, b_idx, :] = coeffs
        structure[b_idx, a_idx, :] = -coeffs  # antisymmetry

        # Closure residuals: residual norm / bracket norm
        projected = coeffs @ B  # [n_pairs, grade2_dim]
        residuals = brackets_bv - projected
        res_norms = residuals.norm(dim=-1)  # [n_pairs]
        bracket_norms = brackets_bv.norm(dim=-1)  # [n_pairs]

        valid = bracket_norms > eps_like(bracket_norms)
        if valid.any():
            mean_relative_closure_residual = (res_norms[valid] / bracket_norms[valid]).mean().item()
        else:
            mean_relative_closure_residual = 0.0

        return {
            "projected_bracket_coefficients": structure,
            "mean_relative_closure_residual": mean_relative_closure_residual,
            "blade_indices": selected_indices,
        }
