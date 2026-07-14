# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Versor layers that act through learned grade-1 or grade-2 parameters."""

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


class VersorLayer(CliffordModule):
    """Learnable versor layer for reflection or rotor actions.

    ``grade=2`` (default) learns ``R = exp(-B/2)`` and applies the rotor
    action ``x' = R x reverse(R)``. ``grade=1`` learns a vector and applies
    the corresponding reflection action. These are the currently supported
    parameter grades.

    Preserves origin. For grade=2, also preserves lengths and angles (isometry).

    Bivector exponentials are planned by the core exp executor family:
    closed formulas for low dimensions, matrix exp by default in higher
    dimensions, and spectral-local execution for eligible high-dimensional
    signatures.

    Attributes:
        channels (int): Number of versors.
        grade (int): Grade of the learnable parameter. Default 2 (bivector → rotor).
        grade_weights (nn.Parameter): Learnable coefficients with shape
            ``[channels, num_grade_elements]``.
    """

    def __init__(
        self,
        algebra: AlgebraLike,
        channels: int,
        grade: int = 2,
        *,
        input_grades=None,
        output_grades=None,
        input_layout: GradeLayout = None,
        output_layout: GradeLayout = None,
    ):
        """Initialize the versor layer.

        Args:
            algebra: Planner-capable algebra host.
            channels (int): Number of features.
            grade (int): Grade of the learnable parameter. The supported values
                are ``1`` for vector reflections and ``2`` for bivector rotor
                actions. Defaults to ``2``.
        """
        super().__init__(algebra)
        self.channels = require_positive_int(channels, "channels")
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
        self.action = algebra.plan_versor_action(
            grade=self.grade,
            input_layout=self.input_layout,
            output_layout=self.output_layout,
            parameter_layout=self.parameter_layout,
        )

        self.grade_weights = nn.Parameter(torch.Tensor(self.channels, self.num_grade_elements))
        if self.grade == 2:
            tag_manifold(self.grade_weights, MANIFOLD_SPIN)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize with near-identity transform (small weights)."""
        nn.init.normal_(self.grade_weights, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the planned reflection or rotor action.

        Args:
            x (torch.Tensor): Input [Batch, Channels, Dim].

        Returns:
            torch.Tensor: Transformed input [Batch, Channels, Dim].
        """
        values = x if x.shape[-1] == self.input_layout.dim else self.input_layout.compact(x)
        return self.action(values, self.grade_weights)

    def prune_weights(self, threshold: float = 1e-4) -> int:
        """Zero out grade weights below threshold.

        Args:
            threshold (float): Cutoff magnitude.

        Returns:
            int: Number of pruned parameters.
        """
        with torch.no_grad():
            mask = torch.abs(self.grade_weights) >= threshold
            num_pruned = (~mask).sum().item()
            self.grade_weights.data.mul_(mask.type_as(self.grade_weights))
        return num_pruned

    def sparsity_loss(self) -> torch.Tensor:
        """Compute L1 sparsity regularization on grade weights."""
        return torch.norm(self.grade_weights, p=1)
