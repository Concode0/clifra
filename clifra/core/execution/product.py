# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Product executors for static Clifford product plans."""

from __future__ import annotations

import torch
import torch.nn as nn

from clifra.core.planning.product import FullTableProductPlan, GradeProductPlan


class GradeProductExecutor(nn.Module):
    """Compile-friendly grade-restricted product using a static interaction plan.

    ``forward`` returns compact output lanes ordered by ``active_output_indices``.
    ``forward_full`` materializes into the canonical all-grades layout.
    """

    def __init__(self, plan: GradeProductPlan):
        super().__init__()
        self.executor_family = "sparse"
        self.p = plan.p
        self.q = plan.q
        self.r = plan.r
        self.n = plan.n
        self.dim = plan.dim
        self.op = plan.op
        self.left_grades = plan.left_grades
        self.right_grades = plan.right_grades
        self.output_grades = plan.output_grades
        self.left_layout = plan.left_layout
        self.right_layout = plan.right_layout
        self.output_layout = plan.output_layout
        self._output_dim = plan.output_dim
        self._pair_count = plan.pair_count
        self.register_buffer("left_indices", plan.left_indices, persistent=False)
        self.register_buffer("right_indices", plan.right_indices, persistent=False)
        self.register_buffer("output_indices", plan.output_indices, persistent=False)
        self.register_buffer("output_positions", plan.output_positions, persistent=False)
        self.register_buffer("coefficients", plan.coefficients, persistent=False)
        self.register_buffer("active_output_indices", plan.active_output_indices, persistent=False)
        self.register_buffer("left_active_positions", plan.left_active_positions, persistent=False)
        self.register_buffer("right_active_positions", plan.right_active_positions, persistent=False)
        self._pairwise_contract_left = plan.pairwise_contract_left
        self.register_buffer("pairwise_gather_positions", plan.pairwise_gather_positions, persistent=False)
        self.register_buffer("pairwise_coefficients", plan.pairwise_coefficients, persistent=False)

    @property
    def output_dim(self) -> int:
        """Return the compact output lane count."""
        return self._output_dim

    @property
    def pair_count(self) -> int:
        """Return the number of planned basis interactions."""
        return self._pair_count

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Return compact grade-lane output for full-layout input tensors."""
        if left.shape[-1] != self.dim:
            raise ValueError(f"left last dimension must be {self.dim}, got {left.shape[-1]}")
        if right.shape[-1] != self.dim:
            raise ValueError(f"right last dimension must be {self.dim}, got {right.shape[-1]}")

        left_terms = torch.index_select(left, -1, self.left_indices)
        right_terms = torch.index_select(right, -1, self.right_indices)
        left_terms, right_terms = torch.broadcast_tensors(left_terms, right_terms)
        terms = left_terms * right_terms * self._coefficients_for(left_terms, right_terms)

        output = terms.new_zeros(*terms.shape[:-1], self.output_dim)
        return output.index_add(-1, self.output_positions, terms)

    def forward_compact(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Return compact output for inputs already stored in this plan's compact layouts."""
        if left.shape[-1] != self.left_layout.dim:
            raise ValueError(f"left compact dimension must be {self.left_layout.dim}, got {left.shape[-1]}")
        if right.shape[-1] != self.right_layout.dim:
            raise ValueError(f"right compact dimension must be {self.right_layout.dim}, got {right.shape[-1]}")

        left_terms = torch.index_select(left, -1, self.left_active_positions)
        right_terms = torch.index_select(right, -1, self.right_active_positions)
        left_terms, right_terms = torch.broadcast_tensors(left_terms, right_terms)
        terms = left_terms * right_terms * self._coefficients_for(left_terms, right_terms)

        output = terms.new_zeros(*terms.shape[:-1], self.output_dim)
        return output.index_add(-1, self.output_positions, terms)

    def forward_pairwise_compact(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Pairwise compact product for sequence-style bilinear scoring.

        ``left`` is ``[..., left_items, left_layout.dim]`` and ``right`` is
        ``[..., right_items, right_layout.dim]``. The result is
        ``[..., left_items, right_items, output_layout.dim]``.
        """
        if left.shape[-1] != self.left_layout.dim:
            raise ValueError(f"left compact dimension must be {self.left_layout.dim}, got {left.shape[-1]}")
        if right.shape[-1] != self.right_layout.dim:
            raise ValueError(f"right compact dimension must be {self.right_layout.dim}, got {right.shape[-1]}")

        prefix = torch.broadcast_shapes(left.shape[:-2], right.shape[:-2])
        left = left.expand(*prefix, *left.shape[-2:])
        right = right.expand(*prefix, *right.shape[-2:])

        if self._pairwise_contract_left:
            flat_positions = self.pairwise_gather_positions.reshape(-1)
            right_gathered = torch.index_select(right, -1, flat_positions).reshape(
                *right.shape[:-1],
                self.left_layout.dim,
                self.output_dim,
            )
            weighted_right = right_gathered * self.pairwise_coefficients
            return torch.einsum("...li,...rik->...lrk", left, weighted_right)

        flat_positions = self.pairwise_gather_positions.reshape(-1)
        left_gathered = torch.index_select(left, -1, flat_positions).reshape(
            *left.shape[:-1],
            self.right_layout.dim,
            self.output_dim,
        )
        weighted_left = left_gathered * self.pairwise_coefficients
        return torch.einsum("...ljk,...rj->...lrk", weighted_left, right)

    def forward_full(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Return a full ``[..., 2**n]`` lane tensor."""
        compact = self.forward(left, right)
        output = compact.new_zeros(*compact.shape[:-1], self.dim)
        return output.index_copy(-1, self.active_output_indices, compact)

    def _coefficients_for(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.coefficients


class FullTableProductExecutor(nn.Module):
    """Planner-owned full-layout Cayley-table product executor.

    The public protocol matches :class:`GradeProductExecutor` where possible so
    hosts and layers can call executor handles without caring which family the
    planner selected.
    """

    def __init__(self, plan: FullTableProductPlan):
        super().__init__()
        self.executor_family = "full_table"
        self.p = plan.p
        self.q = plan.q
        self.r = plan.r
        self.n = plan.n
        self.dim = plan.dim
        self.op = plan.op
        self.left_grades = plan.left_grades
        self.right_grades = plan.right_grades
        self.output_grades = plan.output_grades
        self.left_layout = plan.left_layout
        self.right_layout = plan.right_layout
        self.output_layout = plan.output_layout
        self._output_dim = plan.output_dim
        self._pair_count = plan.pair_count
        self.register_buffer("cayley_indices", plan.cayley_indices, persistent=False)
        self.register_buffer("signs", plan.signs, persistent=False)

    @property
    def output_dim(self) -> int:
        """Return the full output lane count."""
        return self._output_dim

    @property
    def pair_count(self) -> int:
        """Return the number of nonzero full-table interactions."""
        return self._pair_count

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Return full-layout product lanes."""
        return self.forward_compact(left, right)

    def forward_compact(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Return full-layout product lanes for full-layout active values."""
        if left.shape[-1] != self.dim:
            raise ValueError(f"left full dimension must be {self.dim}, got {left.shape[-1]}")
        if right.shape[-1] != self.dim:
            raise ValueError(f"right full dimension must be {self.dim}, got {right.shape[-1]}")

        right_gathered = right[..., self.cayley_indices]
        return torch.matmul(left.unsqueeze(-2), right_gathered * self._signs_for(left, right)).squeeze(-2)

    def forward_pairwise_compact(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Pairwise full-layout product for item-axis operands."""
        if left.shape[-1] != self.dim:
            raise ValueError(f"left full dimension must be {self.dim}, got {left.shape[-1]}")
        if right.shape[-1] != self.dim:
            raise ValueError(f"right full dimension must be {self.dim}, got {right.shape[-1]}")

        prefix = torch.broadcast_shapes(left.shape[:-2], right.shape[:-2])
        left = left.expand(*prefix, *left.shape[-2:])
        right = right.expand(*prefix, *right.shape[-2:])
        right_gathered = right[..., self.cayley_indices]
        weighted_right = right_gathered * self._signs_for(left, right)
        return torch.einsum("...li,...rik->...lrk", left, weighted_right)

    def forward_full(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Return full-layout product lanes."""
        return self.forward_compact(left, right)

    def _signs_for(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.signs


__all__ = ["FullTableProductExecutor", "GradeProductExecutor"]
