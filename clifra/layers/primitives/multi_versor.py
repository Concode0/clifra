# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Multi-versor superposition layers for grade-1 and grade-2 parameters.

Implements versor-based transformations using weighted sums of sandwich products.
"""

import torch
import torch.nn as nn

from clifra.core.foundation.layout import GradeLayout
from clifra.core.foundation.manifold import MANIFOLD_SPIN, tag_manifold
from clifra.core.foundation.module import AlgebraLike, CliffordModule
from clifra.core.runtime.tensors import resolve_contract

from ._utils import (
    grade_indices,
    require_positive_int,
)


class MultiVersorLayer(CliffordModule):
    """Weighted superposition of planned reflection or rotor actions.

    With ``grade=2`` (default), each parameter is exponentiated to a rotor.
    With ``grade=1``, each parameter defines a reflection. These are the
    currently supported parameter grades.

    Bivector exponentials are planned by the core exp executor family:
    closed formulas, matrix exp, or spectral-local execution for eligible
    high-dimensional signatures.

    Attributes:
        channels (int): Input features.
        num_versors (int): Number of overlapping versors.
        grade (int): Grade of the learnable parameters. Default 2 (rotors).
        grade_weights (nn.Parameter): Parameter coefficients with shape
            ``[num_versors, num_grade_elements]``.
        weights (nn.Parameter): Mixing weights [channels, num_versors].
    """

    def __init__(
        self,
        algebra: AlgebraLike,
        channels: int,
        num_versors: int = 8,
        grade: int = 2,
        *,
        input_grades=None,
        output_grades=None,
        input_layout: GradeLayout = None,
        output_layout: GradeLayout = None,
    ):
        """Initialize a multi-versor layer.

        Args:
            algebra: Planner-capable algebra host.
            channels (int): Input features.
            num_versors (int): Number of parallel versor heads.
            grade (int): Grade of the learnable parameter. The supported values
                are ``1`` for vector reflections and ``2`` for bivector rotor
                actions. Defaults to ``2``.
        """
        super().__init__(algebra)
        self.channels = require_positive_int(channels, "channels")
        self.num_versors = require_positive_int(num_versors, "num_versors")
        self.grade = int(grade)
        self.input_contract = resolve_contract(algebra, layout=input_layout, grades=input_grades)
        self.output_contract = (
            resolve_contract(algebra, layout=output_layout, grades=output_grades)
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

        self.grade_weights = nn.Parameter(torch.Tensor(self.num_versors, self.num_grade_elements))
        if self.grade == 2:
            tag_manifold(self.grade_weights, MANIFOLD_SPIN)

        # Mixing weights (Euclidean — intentionally untagged)
        self.weights = nn.Parameter(torch.Tensor(self.channels, self.num_versors))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize with small transforms and uniform mixing weights."""
        nn.init.normal_(self.grade_weights, std=0.01)
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
        out = self.action(values, self.grade_weights, self.weights)

        if return_invariants:
            return self.algebra.grade_norms(out, layout=self.output_layout)

        return out

    def sparsity_loss(self) -> torch.Tensor:
        """Compute L1 sparsity loss for versor weights and mixing weights."""
        return torch.norm(self.grade_weights, p=1) + torch.norm(self.weights, p=1)
