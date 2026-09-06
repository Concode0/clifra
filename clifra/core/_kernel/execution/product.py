# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Product executors for static Clifford product plans."""

from __future__ import annotations

import torch
import torch.nn as nn

from clifra.core._kernel.planning.product import FullTableProductPlan, GradeProductPlan
from clifra.core.tensors import TensorContract


class GradeProductExecutor(nn.Module):
    """Compile-friendly grade-restricted product using a static interaction plan.

    ``forward`` returns compact output lanes ordered by ``output_basis_indices``.
    ``forward_full`` materializes into the canonical all-grades layout.
    """

    def __init__(self, plan: GradeProductPlan):
        super().__init__()
        self.route = "sparse"
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
        self.left_contract = plan.left_contract
        self.right_contract = plan.right_contract
        self.output_contract = plan.output_contract
        self.left_canonical_contract = TensorContract.canonical(self.left_layout)
        self.right_canonical_contract = TensorContract.canonical(self.right_layout)
        self._output_dim = plan.output_dim
        self._pair_count = plan.pair_count
        self._empty_product = len(plan.tree.paths) == 0
        scalar_product = self.op in {"geometric_product", "wedge", "symmetric_product"}
        self._scalar_left = scalar_product and self.left_grades == (0,) and self.right_layout == self.output_layout
        self._scalar_right = scalar_product and self.right_grades == (0,) and self.left_layout == self.output_layout
        self._vector_scalar = (
            self.left_grades == self.right_grades == (1,)
            and self.output_grades == (0,)
            and self.op
            in {
                "geometric_product",
                "left_contraction",
                "right_contraction",
                "symmetric_product",
                "anti_commutator_product",
            }
        )
        self.register_buffer("left_indices", plan.left_indices, persistent=False)
        self.register_buffer("right_indices", plan.right_indices, persistent=False)
        self.register_buffer("output_indices", plan.output_indices, persistent=False)
        self.register_buffer("output_positions", plan.output_positions, persistent=False)
        self.register_buffer("coefficients", plan.coefficients, persistent=False)
        self.register_buffer("output_basis_indices", plan.output_basis_indices, persistent=False)
        self.register_buffer("left_compact_positions", plan.left_compact_positions, persistent=False)
        self.register_buffer("right_compact_positions", plan.right_compact_positions, persistent=False)
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
        self.left_canonical_contract.validate(left, name="left")
        self.right_canonical_contract.validate(right, name="right")

        if self._scalar_multiply(left, right):
            if self._scalar_left:
                return left[..., :1] * torch.index_select(right, -1, self.output_basis_indices)
            return torch.index_select(left, -1, self.output_basis_indices) * right[..., :1]

        left_terms = torch.index_select(left, -1, self.left_indices)
        right_terms = torch.index_select(right, -1, self.right_indices)
        terms = left_terms * right_terms * self.coefficients

        if self._vector_scalar:
            return terms.sum(-1, keepdim=True)
        if self._empty_product:
            return terms.sum(-1, keepdim=True).expand(*terms.shape[:-1], self.output_dim)
        output = terms.new_zeros(*terms.shape[:-1], self.output_dim)
        return output.index_add(-1, self.output_positions, terms)

    def forward_compact(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Return compact output for inputs already stored in this plan's compact layouts."""
        self.left_contract.validate(left, name="left")
        self.right_contract.validate(right, name="right")
        if self._scalar_multiply(left, right):
            return left * right
        if self._vector_scalar:
            # Null directions have no planned diagonal interaction. Do not
            # multiply them by zero: that would introduce NaNs for infinite lanes.
            terms = left[..., : self.p + self.q] * right[..., : self.p + self.q] * self.coefficients
            return terms.sum(-1, keepdim=True)
        left_terms = torch.index_select(left, -1, self.left_compact_positions)
        right_terms = torch.index_select(right, -1, self.right_compact_positions)
        terms = left_terms * right_terms * self.coefficients

        if self._empty_product:
            return terms.sum(-1, keepdim=True).expand(*terms.shape[:-1], self.output_dim)
        output = terms.new_zeros(*terms.shape[:-1], self.output_dim)
        return output.index_add(-1, self.output_positions, terms)

    def _scalar_multiply(self, left: torch.Tensor, right: torch.Tensor) -> bool:
        # Preserve coefficient promotion and device checks for directly used modules.
        return (
            (self._scalar_left or self._scalar_right)
            and left.dtype == right.dtype == self.coefficients.dtype
            and left.device == right.device == self.coefficients.device
        )

    def forward_pairwise_compact(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Pairwise compact product for sequence-style bilinear scoring.

        ``left`` is ``[..., left_items, left_layout.dim]`` and ``right`` is
        ``[..., right_items, right_layout.dim]``. The result is
        ``[..., left_items, right_items, output_layout.dim]``.
        """
        self.left_contract.validate(left, name="left")
        self.right_contract.validate(right, name="right")
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

    def forward_pairwise_compact_right_signed(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        right_signs: torch.Tensor,
    ) -> torch.Tensor:
        """Pairwise compact product with a diagonal sign applied to right lanes."""
        self.left_contract.validate(left, name="left")
        self.right_contract.validate(right, name="right")
        if right_signs.shape != (self.right_layout.dim,):
            raise ValueError(f"right_signs shape must be {(self.right_layout.dim,)}, got {tuple(right_signs.shape)}")
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
            right_signs_gathered = torch.index_select(right_signs, 0, flat_positions).reshape(
                self.left_layout.dim,
                self.output_dim,
            )
            weighted_right = right_gathered * (self.pairwise_coefficients * right_signs_gathered)
            return torch.einsum("...li,...rik->...lrk", left, weighted_right)

        flat_positions = self.pairwise_gather_positions.reshape(-1)
        left_gathered = torch.index_select(left, -1, flat_positions).reshape(
            *left.shape[:-1],
            self.right_layout.dim,
            self.output_dim,
        )
        weighted_left = left_gathered * (self.pairwise_coefficients * right_signs.unsqueeze(-1))
        return torch.einsum("...ljk,...rj->...lrk", weighted_left, right)

    def forward_full(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Return a full ``[..., 2**n]`` lane tensor."""
        compact = self.forward(left, right)
        output = compact.new_zeros(*compact.shape[:-1], self.dim)
        return output.index_copy(-1, self.output_basis_indices, compact)


class FullTableProductExecutor(nn.Module):
    """Planner-owned full-layout Cayley-table product executor.

    The public protocol matches :class:`GradeProductExecutor` where possible so
    hosts and layers can call executor handles without caring which family the
    planner selected.
    """

    def __init__(self, plan: FullTableProductPlan):
        super().__init__()
        self.route = "full_table"
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
        self.left_contract = plan.left_contract
        self.right_contract = plan.right_contract
        self.output_contract = plan.output_contract
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
        """Return full-layout product lanes for full-layout compact values."""
        self.left_contract.validate(left, name="left")
        self.right_contract.validate(right, name="right")
        right_gathered = self._gather_right(right)
        return torch.matmul(left.unsqueeze(-2), right_gathered * self.signs).squeeze(-2)

    def forward_pairwise_compact(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Pairwise full-layout product for item-axis operands."""
        self.left_contract.validate(left, name="left")
        self.right_contract.validate(right, name="right")
        prefix = torch.broadcast_shapes(left.shape[:-2], right.shape[:-2])
        left = left.expand(*prefix, *left.shape[-2:])
        right = right.expand(*prefix, *right.shape[-2:])
        right_gathered = self._gather_right(right)
        weighted_right = right_gathered * self.signs
        return torch.einsum("...li,...rik->...lrk", left, weighted_right)

    def _gather_right(self, right: torch.Tensor) -> torch.Tensor:
        # Larger float32 CPU tables benefit from flattened indexing. Keep the
        # other dtypes, accelerators, and tiny tables on their existing path.
        if self.dim >= 64 and right.device.type == "cpu" and right.dtype == torch.float32:
            return right.index_select(-1, self.cayley_indices.reshape(-1)).reshape(
                *right.shape[:-1], self.dim, self.dim
            )
        return right[..., self.cayley_indices]

    def forward_full(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Return full-layout product lanes."""
        return self.forward_compact(left, right)


__all__ = ["FullTableProductExecutor", "GradeProductExecutor"]


def assess_product_routes(
    algebra,
    *,
    op: str,
    left_layout,
    right_layout,
    output_layout,
    dtype: torch.dtype,
    device,
):
    """Declare product capabilities and conservative costs without execution buffers."""
    from clifra.core._kernel.planning.policy import PlanCandidate, PlanFacts, environment_extensions
    from clifra.core._kernel.planning.resources import ResourceRequirements
    from clifra.core._kernel.planning.tree import build_grade_plan_tree

    tree = build_grade_plan_tree(
        left_layout.spec,
        op=op,
        left_grades=left_layout.grades,
        right_grades=right_layout.grades,
        output_grades=output_layout.grades,
    )
    backend = _device_backend(device)
    dtype_bytes = torch.finfo(dtype).bits // 8
    full_table_pairs = left_layout.dim * right_layout.dim
    sparse_pairs = tree.estimated_pairs
    full_table_bytes = full_table_pairs * (8 + dtype_bytes)
    sparse_bytes = sparse_pairs * (24 + dtype_bytes)
    full_grades = tuple(range(left_layout.spec.n + 1))
    full_table_supported = (
        left_layout.grades == full_grades and right_layout.grades == full_grades and output_layout.grades == full_grades
    )

    def candidate(route: str, pair_count: int, peak_bytes: int, unavailable_reason=None) -> PlanCandidate:
        extensions = {
            **environment_extensions(left_layout.spec, backend, dtype_bytes),
            "layout.left_lanes": left_layout.dim,
            "layout.right_lanes": right_layout.dim,
            "layout.output_lanes": output_layout.dim,
        }
        return PlanCandidate(
            "product",
            route,
            PlanFacts(
                forward_work=pair_count,
                backward_work=pair_count * 2,
                peak_bytes=peak_bytes,
                compile_work=tree.path_count,
                extensions=extensions,
                resources=ResourceRequirements(
                    max(left_layout.dim, right_layout.dim, output_layout.dim),
                    max(pair_count, min(left_layout.dim, right_layout.dim) * output_layout.dim)
                    if route == "sparse"
                    else pair_count,
                ),
            ),
            unavailable_reason,
        )

    return (
        candidate(
            "full_table",
            full_table_pairs,
            full_table_bytes,
            None if full_table_supported else "requires_canonical_full_layouts",
        ),
        candidate("sparse", sparse_pairs, sparse_bytes),
    )


def _device_backend(device) -> str:
    if device is None:
        return "cpu"
    device_type = torch.device(device).type
    return device_type if device_type in {"cpu", "mps"} else "other"
