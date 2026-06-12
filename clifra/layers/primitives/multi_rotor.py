# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Multi-versor superposition layers with universal grade parameterization.

Implements versor-based transformations using weighted sums of sandwich products.
"""

import torch
import torch.nn as nn

from clifra.core.foundation.layout import GradeLayout
from clifra.core.foundation.manifold import MANIFOLD_SPIN, tag_manifold
from clifra.core.foundation.module import AlgebraLike, CliffordModule
from clifra.core.storage import resolve_layer_layout_contract

from ._utils import (
    grade_indices,
    require_positive_int,
)


class MultiRotorLayer(CliffordModule):
    """Multi-versor layer with weighted superposition: x' = sum_k w_k hat(V_k) x V_k^{-1}.

    For grade=2 (default): each V_k = exp(-B_k/2) is a rotor, reducing to
    x' = sum_k w_k R_k x R~_k.
    For grade=k: each V_k is a grade-k versor applied via the general versor product.

    Bivector exponentials are planned by the core exp executor family:
    closed formulas, matrix exp, or spectral-local execution for eligible
    high-dimensional signatures.

    Attributes:
        channels (int): Input features.
        num_rotors (int): Number of overlapping versors.
        grade (int): Grade of the learnable parameters. Default 2 (rotors).
        rotor_grade_weights (nn.Parameter): Grade-k coefficients [num_rotors, num_grade_elements].
        weights (nn.Parameter): Mixing weights [channels, num_rotors].
    """

    def __init__(
        self,
        algebra: AlgebraLike,
        channels: int,
        num_rotors: int = 8,
        grade: int = 2,
        *,
        input_grades=None,
        output_grades=None,
        input_layout: GradeLayout = None,
        output_layout: GradeLayout = None,
    ):
        """Initialize a multi-rotor layer.

        Args:
            algebra: Planner-capable algebra host.
            channels (int): Input features.
            num_rotors (int): Number of parallel versor heads.
            grade (int): Grade of the learnable parameter.
                grade=2 (default): bivectors → rotors via exp(-B/2), Spin group.
                grade=k: general grade-k versor product.
        """
        super().__init__(algebra)
        self.channels = require_positive_int(channels, "channels")
        self.num_rotors = require_positive_int(num_rotors, "num_rotors")
        self.grade = int(grade)
        self.input_contract = resolve_layer_layout_contract(algebra, layout=input_layout, grades=input_grades)
        self.output_contract = (
            resolve_layer_layout_contract(algebra, layout=output_layout, grades=output_grades)
            if output_layout is not None or output_grades is not None
            else self.input_contract
        )
        self.input_layout = self.input_contract.layout
        self.output_layout = self.output_contract.layout

        self.register_buffer("grade_indices", grade_indices(algebra, self.grade))
        self.num_grade_elements = self.grade_indices.numel()
        self.parameter_layout = algebra.layout((self.grade,))
        self.action = algebra.plan_multi_versor_action(
            grade=self.grade,
            input_layout=self.input_layout,
            output_layout=self.output_layout,
            parameter_layout=self.parameter_layout,
        )

        self.rotor_grade_weights = nn.Parameter(torch.Tensor(self.num_rotors, self.num_grade_elements))
        if self.grade == 2:
            tag_manifold(self.rotor_grade_weights, MANIFOLD_SPIN)

        # Mixing weights (Euclidean — intentionally untagged)
        self.weights = nn.Parameter(torch.Tensor(self.channels, self.num_rotors))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize with small transforms and uniform mixing weights."""
        nn.init.normal_(self.rotor_grade_weights, std=0.01)
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x: torch.Tensor, return_invariants: bool = False) -> torch.Tensor:
        """Apply weighted multi-versor superposition.

        Args:
            x (torch.Tensor): Input [Batch, Channels, Dim].
            return_invariants (bool): If True, returns per-grade norms instead of output.

        Returns:
            torch.Tensor: Transformed output [Batch, Channels, Dim].
        """
        values = x if x.shape[-1] == self.input_layout.dim else self.input_layout.compact(x)
        out = self.action(values, self.rotor_grade_weights, self.weights)

        if return_invariants:
            return self.algebra.grade_norms(out, layout=self.output_layout)

        return out

    def sparsity_loss(self) -> torch.Tensor:
        """Compute L1 sparsity loss for versor weights and mixing weights."""
        return torch.norm(self.rotor_grade_weights, p=1) + torch.norm(self.weights, p=1)
