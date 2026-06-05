# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Rotor-based linear transformation layer (Generalized Rotor Gadget).

Implements Section 4.2 from Pence et al. (2025), "Composing Linear Layers
from Irreducibles." Replaces standard linear layers with parameter-efficient
rotor-sandwich transformations.

Reference:
    Pence, T., Yamada, D., & Singh, V. (2025). "Composing Linear Layers
    from Irreducibles." arXiv:2507.11688v1, Section 4.2, Equation 6
"""

from typing import Literal

import torch
import torch.nn as nn

from clifra.core.foundation.layout import GradeLayout
from clifra.core.foundation.manifold import MANIFOLD_SPIN, tag_manifold
from clifra.core.foundation.module import AlgebraLike, CliffordModule
from clifra.core.foundation.validation import check_channels, check_multivector
from clifra.core.storage import LayerLayout, resolve_layer_layout, resolve_layer_layout_contract

from ._utils import (
    channel_mix,
    pair_mean,
    require_choice,
    require_positive_int,
)

ROTOR_GADGET_INIT_STD = 0.01


class RotorGadget(CliffordModule):
    """Rotor-based linear transformation (Generalized Rotor Gadget).

    Replaces standard linear layers with parameter-efficient rotor-sandwich
    transformations. Instead of using O(in_channels x out_channels) parameters,
    this uses O(num_rotor_pairs x n(n-1)/2) parameters where n is the number
    of basis vectors in the Clifford algebra.

    Architecture:
        Partition input channels into blocks, apply rotor sandwiches to each
        pair, then aggregate the results to output channels.

    The transformation is: psi(x) = r.x.s.H where r, s are rotors (bivector exponentials).

    Attributes:
        algebra: Planner-capable algebra host
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_rotor_pairs: Number of rotor pairs to use
        aggregation: Aggregation method ('mean', 'sum', or 'learned')
    """

    def __init__(
        self,
        algebra: AlgebraLike,
        in_channels: int,
        out_channels: int,
        num_rotor_pairs: int = 4,
        aggregation: Literal["mean", "sum", "learned"] = "mean",
        shuffle: Literal["none", "fixed", "random"] = "none",
        bias: bool = False,
        *,
        grades=None,
        layout: GradeLayout = None,
        input_grades=None,
        output_grades=None,
        input_layout: GradeLayout = None,
        output_layout: GradeLayout = None,
    ):
        """Initialize rotor gadget layer.

        Args:
            algebra: Planner-capable algebra host
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_rotor_pairs: Number of rotor pairs (higher = more expressive)
            aggregation: How to pool rotor outputs ('mean', 'sum', 'learned')
            shuffle: Input channel shuffle strategy:
                - 'none': No shuffle, sequential block assignment (default)
                - 'fixed': Random permutation at initialization (fixed during training)
                - 'random': Random permutation each forward pass (regularization)
            bias: Whether to include bias term (applied after transformation)
        """
        super().__init__(algebra)

        self.in_channels = require_positive_int(in_channels, "in_channels")
        self.out_channels = require_positive_int(out_channels, "out_channels")
        self.num_rotor_pairs = require_positive_int(num_rotor_pairs, "num_rotor_pairs")
        self.aggregation = require_choice(aggregation, "aggregation", ("mean", "sum", "learned"))
        self.shuffle = require_choice(shuffle, "shuffle", ("none", "fixed", "random"))
        if input_layout is None:
            input_layout = layout
        if input_grades is None:
            input_grades = grades

        if algebra.num_grades <= 2:
            raise ValueError(f"Algebra has no bivectors. RotorGadget requires at least one bivector for rotation.")
        self.input_contract = resolve_layer_layout_contract(algebra, layout=input_layout, grades=input_grades)
        resolved_output_layout = (
            resolve_layer_layout(algebra, layout=output_layout, grades=output_grades)
            if output_layout is not None or output_grades is not None
            else None
        )
        self.parameter_layout = algebra.layout((2,))
        self.action_plan = algebra.planner.paired_bivector_action_plan(
            input_layout=self.input_contract.layout,
            output_layout=resolved_output_layout,
            parameter_layout=self.parameter_layout,
        )
        self.output_contract = LayerLayout(algebra, self.action_plan.output_layout)
        self.input_layout = self.input_contract.layout
        self.output_layout = self.output_contract.layout
        self.rotor_layout = self.action_plan.rotor_layout
        self.middle_layout = self.action_plan.middle_layout
        self.input_lane_dim = self.input_contract.lane_dim
        self.output_lane_dim = self.output_contract.lane_dim
        self.num_bivectors = self.parameter_layout.dim
        self.action = algebra.plan_paired_bivector_action(
            input_layout=self.input_layout,
            output_layout=self.output_layout,
            parameter_layout=self.parameter_layout,
        )

        # Rotor parameters: bivector coefficients for exponential map
        # Left rotors: [num_rotor_pairs, num_bivectors]
        self.bivector_left = nn.Parameter(torch.empty(self.num_rotor_pairs, self.num_bivectors))
        tag_manifold(self.bivector_left, MANIFOLD_SPIN)
        # Right rotors: [num_rotor_pairs, num_bivectors]
        self.bivector_right = nn.Parameter(torch.empty(self.num_rotor_pairs, self.num_bivectors))
        tag_manifold(self.bivector_right, MANIFOLD_SPIN)
        self.reset_parameters()

        # Channel routing: block diagonal partitioning (paper style)
        # Each rotor pair processes a subset of input channels
        self._setup_channel_routing()

        # Aggregation weights (if learned)
        if self.aggregation == "learned":
            self.agg_weights = nn.Parameter(torch.ones(self.num_rotor_pairs, self.out_channels) / self.num_rotor_pairs)
        else:
            self.register_buffer("agg_weights", None)

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels, self.output_lane_dim))
        else:
            self.register_buffer("bias", None)

    def reset_parameters(self) -> None:
        """Initialize paired bivector parameters near the identity action."""
        nn.init.normal_(self.bivector_left, std=ROTOR_GADGET_INIT_STD)
        nn.init.normal_(self.bivector_right, std=ROTOR_GADGET_INIT_STD)

    def _setup_channel_routing(self):
        """Set up block diagonal channel routing with optional shuffle.

        Partitions input and output channels into blocks, where each rotor
        pair operates on a specific block. Optionally shuffles input channels
        before routing for regularization.
        """
        in_assignment = torch.div(
            torch.arange(self.in_channels) * self.num_rotor_pairs,
            self.in_channels,
            rounding_mode="floor",
        ).clamp_max(self.num_rotor_pairs - 1)
        out_assignment = torch.div(
            torch.arange(self.out_channels) * self.num_rotor_pairs,
            self.out_channels,
            rounding_mode="floor",
        ).clamp_max(self.num_rotor_pairs - 1)

        in_indices = []
        out_indices = []
        for i in range(self.num_rotor_pairs):
            in_members = (in_assignment == i).nonzero(as_tuple=False).flatten()
            out_members = (out_assignment == i).nonzero(as_tuple=False).flatten()
            if in_members.numel() == 0:
                in_indices.append((self.in_channels, self.in_channels))
            else:
                in_indices.append((int(in_members[0]), int(in_members[-1]) + 1))
            if out_members.numel() == 0:
                out_indices.append((self.out_channels, self.out_channels))
            else:
                out_indices.append((int(out_members[0]), int(out_members[-1]) + 1))

        self.in_indices = in_indices
        self.out_indices = out_indices

        ch2pair = in_assignment.long()
        self.register_buffer("_ch2pair", ch2pair)
        self.register_buffer("_channel_mix_mean", channel_mix(self.in_channels, self.out_channels, normalize=True))
        self.register_buffer("_channel_mix_sum", channel_mix(self.in_channels, self.out_channels, normalize=False))
        self.register_buffer("_pair_mean", pair_mean(ch2pair, self.num_rotor_pairs))

        # Set up channel shuffle permutation
        if self.shuffle == "fixed":
            # Create fixed random permutation at initialization
            perm = torch.randperm(self.in_channels)
            self.register_buffer("channel_permutation", perm)
        elif self.shuffle == "random":
            # Random shuffle each forward pass - no fixed permutation
            self.register_buffer("channel_permutation", None)
        else:  # 'none'
            # No shuffle - identity permutation
            self.register_buffer("channel_permutation", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotor-based transformation.

        Uses batched geometric products - all rotor pairs are applied in
        parallel via a single pair of GP calls.

        Args:
            x: Input tensor of shape [Batch, In_Channels, Dim]

        Returns:
            Output tensor of shape [Batch, Out_Channels, Dim]
        """
        if self.input_layout.dim == self.algebra.dim:
            check_multivector(x, self.algebra, "RotorGadget input")
            check_channels(x, self.in_channels, "RotorGadget input")
        else:
            self.input_contract.validate_input(
                x,
                channels=self.in_channels,
                name="RotorGadget input",
            )

        # Apply input channel shuffle if enabled
        if self.shuffle == "fixed":
            x = x.index_select(-2, self.channel_permutation)
        elif self.shuffle == "random":
            perm = torch.randperm(self.in_channels, device=x.device)
            x = x.index_select(-2, perm)

        concat_out = self.action(
            x,
            self.bivector_left,
            self.bivector_right,
            self._ch2pair,
        )

        # Map to output channels
        out = self._aggregate_to_output_channels(concat_out)

        if self.bias is not None:
            bias_shape = (1,) * (out.ndim - 2) + (self.out_channels, self.output_lane_dim)
            out = out + self.bias.view(bias_shape)

        return out

    def _aggregate_to_output_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate rotor pair outputs to match output channel count.

        Args:
            x: Concatenated outputs from rotor pairs [B, total_channels, dim]

        Returns:
            Aggregated output [B, out_channels, dim]
        """
        if self.aggregation == "learned":
            pair_values = torch.einsum("ki,...id->...kd", self._pair_mean, x)
            return torch.einsum("ko,...kd->...od", self.agg_weights, pair_values)

        mix = self._channel_mix_sum if self.aggregation == "sum" else self._channel_mix_mean
        return torch.einsum("oi,...id->...od", mix, x)

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"num_rotor_pairs={self.num_rotor_pairs}, "
            f"aggregation={self.aggregation}, "
            f"shuffle={self.shuffle}, "
            f"bias={self.bias is not None}"
        )
