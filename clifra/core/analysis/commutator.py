# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Commutator (Lie bracket) analysis of multivector data.

Computes pairwise commutativity indices, the exchange spectrum of the
adjoint operator, and tests for Lie-subalgebra closure among data
bivectors.
"""

from typing import Dict

import torch

from clifra.core.foundation.module import AlgebraLike
from clifra.core.foundation.numerics import eps_like
from clifra.core.runtime.tensors import LaneStorage
from clifra.utils.mps import safe_linalg_eigvals

from ._types import CONSTANTS, CommutatorResult
from ._utils import declared_full_product_kwargs, full_matrix_feasibility, full_product_feasibility
from .policy import feasibility_record


class CommutatorAnalyzer:
    """Analyze algebraic exchange properties via commutators.

    The commutator ``[A, B] = AB - BA`` measures non-commutativity.
    In Clifford algebras, commutators of grade-1 elements yield grade-2
    elements (bivectors), directly related to rotation planes.

    Args:
        algebra: Layout-first algebra host.
        max_bivectors: Maximum number of bivectors for Lie-bracket
            closure analysis (guards combinatorial cost).
    """

    def __init__(
        self,
        algebra: AlgebraLike,
        max_bivectors: int = CONSTANTS.commutator_max_bivectors,
    ):
        self.algebra = algebra
        self.max_bivectors = max_bivectors

    def analyze(self, mv_data: torch.Tensor) -> CommutatorResult:
        """Full commutator analysis.

        Args:
            mv_data: Multivector data.  Accepted shapes:

                * ``[N, dim]`` -- single-channel batch.
                * ``[N, C, dim]`` -- multi-channel batch (channels
                  averaged).

        Returns:
            :class:`CommutatorResult`.
        """
        if mv_data.ndim == 3:
            flat = mv_data.mean(dim=1)
        else:
            flat = mv_data  # [N, dim]

        comm_matrix = self.commutativity_matrix(flat)
        ex_spectrum, skipped = self._exchange_spectrum_with_skips(flat)
        mcn = self.mean_commutator_norm(flat)
        lie_struct = self.lie_bracket_closure(flat)

        return CommutatorResult(
            commutativity_matrix=comm_matrix,
            exchange_spectrum=ex_spectrum,
            mean_commutator_norm=mcn,
            lie_bracket_structure=lie_struct,
            skipped=skipped,
        )

    def commutativity_matrix(self, mv_data: torch.Tensor) -> torch.Tensor:
        """Pairwise commutativity index for input dimensions.

        For each pair ``(i, j)`` of the *n* basis-vector directions,
        computes ``E[||[x_i, x_j]||]`` where ``x_i`` is the data
        projected onto ``e_i``.

        Args:
            mv_data: ``[N, dim]`` multivector data.

        Returns:
            ``[n, n]`` symmetric matrix of commutativity indices.
        """
        n = self.algebra.n
        device = mv_data.device
        dtype = mv_data.dtype

        g1_idx = self.algebra.grade_indices((1,), device=device)
        N = mv_data.shape[0]

        # All (i, j) pairs with i < j
        i_idx, j_idx = torch.triu_indices(n, n, offset=1, device=device)
        if i_idx.numel() == 0:
            return torch.zeros(n, n, device=device, dtype=dtype)

        coeffs = mv_data[:, g1_idx]  # [N, n]
        left = torch.zeros(i_idx.numel(), N, n, device=device, dtype=dtype)
        right = torch.zeros_like(left)
        left.scatter_(-1, i_idx.view(-1, 1, 1).expand(-1, N, 1), coeffs[:, i_idx].T.unsqueeze(-1))
        right.scatter_(-1, j_idx.view(-1, 1, 1).expand(-1, N, 1), coeffs[:, j_idx].T.unsqueeze(-1))

        # Batched compact commutator: [n_pairs, N, grade2_dim]
        comm = self.algebra.commutator_product(
            left,
            right,
            left_grades=(1,),
            right_grades=(1,),
            output_grades=(2,),
            left_storage=LaneStorage.COMPACT,
            right_storage=LaneStorage.COMPACT,
            output_storage=LaneStorage.COMPACT,
        )
        vals = comm.norm(dim=-1).mean(dim=-1)  # [n_pairs]

        matrix = torch.zeros(n, n, device=device, dtype=dtype)
        matrix[i_idx, j_idx] = vals
        matrix[j_idx, i_idx] = vals

        return matrix

    def exchange_spectrum(self, mv_data: torch.Tensor) -> torch.Tensor:
        """Eigenvalue magnitudes of the adjoint operator ``ad_mu``.

        Constructs the explicit matrix for ``ad_mu(x) = [mu, x]``
        where ``mu = E[x]`` and diagonalizes it.

        Args:
            mv_data: ``[N, dim]`` multivector data.

        Returns:
            Eigenvalue magnitudes sorted descending.  Returns an
            empty tensor if the algebra is too large.
        """
        spectrum, _ = self._exchange_spectrum_with_skips(mv_data)
        return spectrum

    def _exchange_spectrum_with_skips(self, mv_data: torch.Tensor) -> tuple[torch.Tensor, dict[str, dict]]:
        """Return exchange spectrum plus feasibility skip metadata."""
        dim = self.algebra.dim
        device = mv_data.device
        dtype = mv_data.dtype
        skipped = {}

        matrix_feasible = full_matrix_feasibility(
            self.algebra,
            role="adjoint_exchange_spectrum",
            max_entries=CONSTANTS.adjoint_matrix_entries,
            matrix_kind="eigensolver",
        )
        product_feasible = full_product_feasibility(
            self.algebra,
            role="adjoint_exchange_spectrum",
            op="commutator_product",
            max_pairs=CONSTANTS.analysis_product_pairs,
        )
        if not matrix_feasible or not product_feasible:
            skipped["exchange_spectrum"] = {
                "reason": _first_skip_reason(matrix_feasible, product_feasible),
                "checks": {
                    "eigensolver_matrix": feasibility_record(matrix_feasible),
                    "product": feasibility_record(product_feasible),
                },
            }
            return torch.tensor([], device=device, dtype=dtype), skipped

        mu = mv_data.mean(dim=0)  # [dim]
        basis = torch.eye(dim, device=device, dtype=dtype)

        # Batched commutator: [dim, dim] x [dim, dim] -> [dim, dim], transpose.
        ad_mu = self.algebra.commutator_product(
            mu.unsqueeze(0).expand(dim, -1),
            basis,
            **declared_full_product_kwargs(self.algebra),
        ).T

        eigvals = safe_linalg_eigvals(ad_mu)  # complex
        magnitudes = eigvals.abs()
        return magnitudes.sort(descending=True).values, skipped

    def mean_commutator_norm(self, mv_data: torch.Tensor) -> float:
        """``E[||[x_i, mu]||_2]`` -- scalar non-commutativity summary.

        Generalizes the *Geometric Uncertainty Index* from
        :func:`clifra.core.analysis.compute_uncertainty_and_alignment`.

        Args:
            mv_data: ``[N, dim]`` multivector data.

        Returns:
            Mean commutator norm (float).
        """
        full_product = full_product_feasibility(
            self.algebra,
            role="mean_commutator_norm",
            op="commutator_product",
            max_pairs=CONSTANTS.analysis_product_pairs,
        )
        if not full_product:
            layout = self.algebra.layout((1,))
            values = layout.compact(mv_data)
            mu = values.mean(dim=0, keepdim=True)
            comm = self.algebra.commutator_product(
                values,
                mu.expand_as(values),
                left_grades=(1,),
                right_grades=(1,),
                output_grades=(2,),
                left_storage=LaneStorage.COMPACT,
                right_storage=LaneStorage.COMPACT,
                output_storage=LaneStorage.COMPACT,
            )
            return comm.norm(dim=-1).mean().item()

        mu = mv_data.mean(dim=0, keepdim=True)  # [1, dim]
        comm = self.algebra.commutator_product(mv_data, mu.expand_as(mv_data), **declared_full_product_kwargs(self.algebra))
        return comm.norm(dim=-1).mean().item()

    def lie_bracket_closure(self, mv_data: torch.Tensor) -> Dict:
        """Test whether the data bivectors close under the Lie bracket.

        Selects the top-*k* energetic bivectors from the batch,
        computes all pairwise brackets ``[B_i, B_j]``, and measures
        how well the results lie in the span of the original set.

        Args:
            mv_data: ``[N, dim]`` multivector data.

        Returns:
            Dict with ``"structure_constants"`` (``[k, k, k]``),
            ``"closure_error"`` (scalar), and ``"basis_indices"`` (list
            of multivector-coefficient indices of the chosen bivectors).
        """
        n = self.algebra.n
        device = mv_data.device
        dtype = mv_data.dtype

        if n < 2:
            return {
                "structure_constants": torch.zeros(0, 0, 0, device=device, dtype=dtype),
                "closure_error": 0.0,
                "basis_indices": [],
            }

        # Extract compact grade-2 part of mean per-sample
        bv_data = self.algebra.grade_projection(mv_data, 2, output_storage=LaneStorage.COMPACT)  # [N, grade2_dim]
        mean_bv = bv_data.mean(dim=0)  # [grade2_dim]

        bv_blade_indices = self.algebra.grade_indices((2,), device=device)

        if bv_blade_indices.numel() == 0:
            return {
                "structure_constants": torch.zeros(0, 0, 0, device=device, dtype=dtype),
                "closure_error": 0.0,
                "basis_indices": [],
            }

        # Pick top-k by energy in the mean bivector
        energies = mean_bv.abs()
        k = min(self.max_bivectors, int(bv_blade_indices.numel()))
        topk_pos = energies.topk(k).indices
        selected_indices = bv_blade_indices[topk_pos].tolist()

        # Build compact basis bivectors.
        B = torch.zeros(k, bv_blade_indices.numel(), device=device, dtype=dtype)
        B[torch.arange(k, device=device), topk_pos] = 1.0

        # Compute structure constants c_{a,b,c} such that [B_a, B_b] ~= Sum_c c_{abc} B_c
        a_idx, b_idx = torch.triu_indices(k, k, offset=1, device=device)

        # Batched compact commutator and grade-2 projection.
        brackets_bv = self.algebra.commutator_product(
            B[a_idx],
            B[b_idx],
            left_grades=(2,),
            right_grades=(2,),
            output_grades=(2,),
            left_storage=LaneStorage.COMPACT,
            right_storage=LaneStorage.COMPACT,
            output_storage=LaneStorage.COMPACT,
        )  # [n_pairs, grade2_dim]

        # Project onto basis: coeffs[p, c] = <bracket_bv_p, B_c>
        coeffs = brackets_bv @ B.T  # [n_pairs, k]

        structure = torch.zeros(k, k, k, device=device, dtype=dtype)
        structure[a_idx, b_idx, :] = coeffs
        structure[b_idx, a_idx, :] = -coeffs  # antisymmetry

        # Closure errors: residual norm / bracket norm
        projected = coeffs @ B  # [n_pairs, grade2_dim]
        residuals = brackets_bv - projected
        res_norms = residuals.norm(dim=-1)  # [n_pairs]
        bracket_norms = brackets_bv.norm(dim=-1)  # [n_pairs]

        valid = bracket_norms > eps_like(bracket_norms)
        if valid.any():
            closure_error = (res_norms[valid] / bracket_norms[valid]).mean().item()
        else:
            closure_error = 0.0

        return {
            "structure_constants": structure,
            "closure_error": closure_error,
            "basis_indices": selected_indices,
        }


def compute_uncertainty_and_alignment(algebra: AlgebraLike, data_tensor: torch.Tensor):
    """Compute Geometric Uncertainty Index (U) and Procrustes Alignment (V).

    Args:
        algebra: Layout-first algebra host.
        data_tensor: ``[N, D]`` tensor of raw features.

    Returns:
        Tuple ``(U, V)`` where *U* is a float (mean commutator norm) and
        *V* is a ``[D, D]`` Procrustes alignment matrix from SVD.
    """
    N, D = data_tensor.shape
    n = algebra.n

    # Lift data to grade-1 for commutator analysis
    if D < n:
        pad = torch.zeros(N, n - D, device=data_tensor.device, dtype=data_tensor.dtype)
        x_n = torch.cat([data_tensor, pad], dim=-1)
    else:
        x_n = data_tensor[:, :n]

    # Mean grade-1 vector and compact commutator [x_i, mu]
    mu = x_n.mean(dim=0, keepdim=True)  # [1, n]
    comm = algebra.commutator_product(
        x_n,
        mu.expand_as(x_n),
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(2,),
        left_storage=LaneStorage.COMPACT,
        right_storage=LaneStorage.COMPACT,
        output_storage=LaneStorage.COMPACT,
    )

    U = torch.norm(comm, p=2, dim=-1).mean().item()

    # Procrustes alignment via SVD
    x_c = data_tensor - data_tensor.mean(dim=0, keepdim=True)
    try:
        _, _, Vh = torch.linalg.svd(x_c, full_matrices=False)
        V = Vh.mH
    except RuntimeError:
        V = torch.eye(D, device=data_tensor.device, dtype=data_tensor.dtype)

    return U, V


def _first_skip_reason(*checks) -> str:
    for check in checks:
        if not check:
            return check.reason
    return "ok"
