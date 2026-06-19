# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Public hot-path handles for preplanned executor calls."""

from __future__ import annotations

import torch
import torch.nn as nn


class ProductPlanHandle(nn.Module):
    """Active-lane product handle returned by algebra planning APIs."""

    def __init__(self, executor: nn.Module):
        super().__init__()
        self.executor = executor
        self.executor_family = getattr(executor, "executor_family", "unknown")
        self.op = executor.op
        self.left_layout = executor.left_layout
        self.right_layout = executor.right_layout
        self.output_layout = executor.output_layout
        self.left_grades = executor.left_grades
        self.right_grades = executor.right_grades
        self.output_grades = executor.output_grades
        self.output_dim = executor.output_dim
        self.pair_count = executor.pair_count

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Execute the product for tensors already stored in the declared layouts."""
        return self.executor.forward_compact(left, right)

    def pairwise(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Execute pairwise item-axis product for compact-lane tensors."""
        return self.executor.forward_pairwise_compact(left, right)


class UnaryPlanHandle(nn.Module):
    """Active-lane unary handle returned by algebra planning APIs."""

    def __init__(self, executor: nn.Module):
        super().__init__()
        self.executor = executor
        self.executor_family = getattr(executor, "executor_family", "unary_sign")
        self.op = executor.op
        self.input_layout = executor.input_layout
        self.output_layout = executor.output_layout
        self.input_grades = executor.input_layout.grades
        self.output_grades = executor.output_layout.grades
        self.output_dim = executor.output_dim

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Execute the unary operation for values already stored in ``input_layout``."""
        return self.executor.forward_compact(values)

    def full_input(self, values: torch.Tensor) -> torch.Tensor:
        """Execute the same plan for full-lane input values."""
        return self.executor(values)

    def full_output(self, values: torch.Tensor) -> torch.Tensor:
        """Execute the same plan and materialize the output into full lanes."""
        return self.executor.forward_full(values)


class FullSandwichActionHandle(nn.Module):
    """Full-layout sandwich action handle returned by algebra planning APIs."""

    def __init__(self, executor: nn.Module):
        super().__init__()
        self.executor = executor
        self.executor_family = executor.executor_family
        self.op = executor.op
        self.layout = executor.layout
        self.dim = executor.dim

    def forward(self, left: torch.Tensor, values: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Apply one sandwich action per channel in ``values``."""
        return self.executor.per_channel_unchecked(left, values, right)

    def action_matrices(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Return full-layout action matrices for the supplied left/right factors."""
        return self.executor.action_matrices_unchecked(left, right)

    def checked_action_matrices(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Return action matrices with eager validation."""
        return self.executor.action_matrices(left, right)

    def per_channel(self, left: torch.Tensor, values: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Apply one sandwich action per channel in ``values``."""
        return self.executor.per_channel_unchecked(left, values, right)

    def checked_per_channel(self, left: torch.Tensor, values: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Apply one sandwich action per channel with eager validation."""
        return self.executor.per_channel(left, values, right)

    def batched(self, left: torch.Tensor, values: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Apply one sandwich action per leading batch item."""
        return self.executor.batched_unchecked(left, values, right)

    def checked_batched(self, left: torch.Tensor, values: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Apply one sandwich action per leading batch item with eager validation."""
        return self.executor.batched(left, values, right)

    def multi(self, left: torch.Tensor, values: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Apply every sandwich action to every channel."""
        return self.executor.multi_unchecked(left, values, right)

    def checked_multi(self, left: torch.Tensor, values: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Apply every sandwich action to every channel with eager validation."""
        return self.executor.multi(left, values, right)

    def routed(
        self,
        left: torch.Tensor,
        values: torch.Tensor,
        right: torch.Tensor,
        channel_to_pair: torch.Tensor,
    ) -> torch.Tensor:
        """Apply actions selected by channel index."""
        return self.executor.routed_unchecked(left, values, right, channel_to_pair)

    def checked_routed(
        self,
        left: torch.Tensor,
        values: torch.Tensor,
        right: torch.Tensor,
        channel_to_pair: torch.Tensor,
    ) -> torch.Tensor:
        """Apply actions selected by channel index with eager validation."""
        return self.executor.routed(left, values, right, channel_to_pair)


class VersorActionHandle(nn.Module):
    """Grade-1 or grade-2 versor action handle for compact-lane values."""

    def __init__(self, executor: nn.Module):
        super().__init__()
        self.executor = executor
        self.executor_family = "versor_action"
        self.op = "versor_action"
        self.grade = executor.grade
        self.input_layout = executor.input_layout
        self.output_layout = executor.output_layout
        self.parameter_layout = executor.parameter_layout
        self.use_full_action = executor.use_full_action

    def forward(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Apply one grade-1 or grade-2 versor per channel."""
        return self.executor.execute(values, weights)

    def checked(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Apply one grade-1 or grade-2 versor per channel with eager validation."""
        return self.executor(values, weights)


class MultiVersorActionHandle(nn.Module):
    """Weighted multi-versor action handle for compact-lane values."""

    def __init__(self, executor: nn.Module):
        super().__init__()
        self.executor = executor
        self.executor_family = "multi_versor_action"
        self.op = "multi_versor_action"
        self.grade = executor.grade
        self.input_layout = executor.input_layout
        self.output_layout = executor.output_layout
        self.parameter_layout = executor.parameter_layout
        self.use_full_action = executor.use_full_action

    def forward(self, values: torch.Tensor, weights: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
        """Apply a weighted superposition of grade-1 or grade-2 versor actions."""
        return self.executor.execute(values, weights, mix)

    def checked(self, values: torch.Tensor, weights: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
        """Apply a weighted superposition with eager validation."""
        return self.executor(values, weights, mix)


class PairedBivectorActionHandle(nn.Module):
    """Independent left/right bivector action handle for compact-lane values."""

    def __init__(self, executor: nn.Module):
        super().__init__()
        self.executor = executor
        self.executor_family = "paired_bivector_action"
        self.op = "paired_bivector_action"
        self.input_layout = executor.input_layout
        self.output_layout = executor.output_layout
        self.parameter_layout = executor.parameter_layout
        self.rotor_layout = executor.rotor_layout
        self.middle_layout = executor.middle_layout
        self.use_full_action = executor.use_full_action

    def forward(
        self,
        values: torch.Tensor,
        left_weights: torch.Tensor,
        right_weights: torch.Tensor,
        channel_to_pair: torch.Tensor,
    ) -> torch.Tensor:
        """Apply independent left/right bivector rotor pairs routed by channel."""
        return self.executor.execute(values, left_weights, right_weights, channel_to_pair)

    def checked(
        self,
        values: torch.Tensor,
        left_weights: torch.Tensor,
        right_weights: torch.Tensor,
        channel_to_pair: torch.Tensor,
    ) -> torch.Tensor:
        """Apply routed paired-bivector actions with eager validation."""
        return self.executor(values, left_weights, right_weights, channel_to_pair)
