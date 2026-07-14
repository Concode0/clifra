# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Clifford-valued channel-mixing layers."""

from typing import Literal

import torch
import torch.nn as nn

from clifra.core.foundation.layout import GradeLayout
from clifra.core.foundation.module import AlgebraLike, CliffordModule
from clifra.core.runtime.tensors import resolve_contract

from ._utils import require_choice, require_positive_int


class CliffordLinear(CliffordModule):
    """Fully connected layer with optional rotor-based backend.

    The traditional backend uses a scalar channel-mixing matrix and a
    multivector bias. The rotor backend delegates to :class:`RotorGadget`,
    whose total parameter count depends on the rotor-pair count, aggregation
    mode, and bias.

    Attributes:
        in_channels (int): Input features.
        out_channels (int): Output features.
        backend (str): 'traditional' or 'rotor'
        weight (torch.nn.Parameter | None): Weights [Out, In] (traditional backend only).
        bias (torch.nn.Parameter | None): Bias multivector [Out, Dim] (traditional backend only).
        gadget (nn.Module | None): Rotor transformation (rotor backend only).
    """

    def __init__(
        self,
        algebra: AlgebraLike,
        in_channels: int,
        out_channels: int,
        backend: Literal["traditional", "rotor"] = "traditional",
        num_rotor_pairs: int = 4,
        aggregation: Literal["mean", "sum", "learned"] = "mean",
        shuffle: Literal["none", "fixed", "random"] = "none",
        grades=None,
        layout: GradeLayout = None,
    ):
        """Initialize Clifford Linear.

        Args:
            algebra: Planner-capable algebra host.
            in_channels (int): Input size.
            out_channels (int): Output size.
            backend (str): 'traditional' for standard linear layer,
                          'rotor' for rotor-based transformation
            num_rotor_pairs (int): Number of rotor pairs (rotor backend only)
            aggregation (str): Aggregation method (rotor backend only)
            shuffle (str): Input channel shuffle strategy (rotor backend only):
                - 'none': No shuffle (default)
                - 'fixed': Fixed random permutation
                - 'random': Random permutation each forward pass
        """
        super().__init__(algebra)
        self.in_channels = require_positive_int(in_channels, "in_channels")
        self.out_channels = require_positive_int(out_channels, "out_channels")
        self.backend = require_choice(backend, "backend", ("traditional", "rotor"))
        self.layout_contract = resolve_contract(algebra, layout=layout, grades=grades)
        self.layout = self.layout_contract.layout
        self.lane_dim = self.layout_contract.lane_dim
        self.output_layout = self.layout
        self.output_lane_dim = self.lane_dim

        if self.backend == "traditional":
            self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels))
            self.bias = nn.Parameter(torch.Tensor(self.out_channels, self.lane_dim))
            self.reset_parameters()
            self.gadget = None

        elif self.backend == "rotor":
            from .rotor_gadget import RotorGadget

            self.gadget = RotorGadget(
                algebra=algebra,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                num_rotor_pairs=num_rotor_pairs,
                aggregation=aggregation,
                shuffle=shuffle,
                bias=True,  # Include bias in rotor gadget
                layout=self.layout,
            )
            self.output_layout = self.gadget.output_layout
            self.output_lane_dim = self.gadget.output_lane_dim
            self.weight = None
            self.bias = None

    def reset_parameters(self):
        """Initialize weights with Xavier uniform and zero bias."""
        if self.backend == "traditional":
            nn.init.xavier_uniform_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel-mixing linear transformation.

        Args:
            x (torch.Tensor): Input [Batch, In, Dim].

        Returns:
            torch.Tensor: Output [Batch, Out, Dim].
        """
        self.layout_contract.validate_input(
            x,
            channels=self.in_channels,
            name="CliffordLinear input",
        )

        if self.backend == "traditional":
            out = torch.einsum("oi,...id->...od", self.weight, x)
            bias_shape = (1,) * (x.ndim - 2) + (self.out_channels, self.lane_dim)
            out = out + self.bias.view(bias_shape)
            return out
        return self.gadget(x)

    def extra_repr(self) -> str:
        """String representation for debugging.

        Returns:
            str: Layer parameters description
        """
        parts = [f"in_channels={self.in_channels}", f"out_channels={self.out_channels}", f"backend={self.backend}"]
        if self.layout.dim != self.algebra.dim:
            parts.append(f"grades={self.layout.grades}")
        if self.output_layout != self.layout:
            parts.append(f"output_grades={self.output_layout.grades}")
        return ", ".join(parts)
