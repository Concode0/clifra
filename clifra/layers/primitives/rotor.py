# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Rotor layers that act through learned bivector parameters."""

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


class RotorLayer(CliffordModule):
    """Learnable versor layer with universal grade parameterization.

    For grade=2 (default): learns R = exp(-B/2) and applies the isometry x' = RxR~.
    For grade=k: learns a grade-k element V and applies the versor product
    x' = hat(V) x V^{-1}, where hat denotes grade involution.

    Preserves origin. For grade=2, also preserves lengths and angles (isometry).

    The exp strategy (closed-form vs decomposition) is controlled by
    ``algebra.exp_policy`` -- see :class:`clifra.core.runtime.decomposition.ExpPolicy`.

    Attributes:
        channels (int): Number of versors.
        grade (int): Grade of the learnable parameter. Default 2 (bivector → rotor).
        grade_weights (nn.Parameter): Learnable grade-k coefficients [channels, num_grade_elements].
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
            grade (int): Grade of the learnable parameter.
                grade=2 (default): bivectors → rotors via exp(-B/2), Spin group.
                grade=1: vectors → reflections via hat(n) x n^{-1}, Pin group.
                grade=k: general grade-k versor product.
        """
        super().__init__(algebra)
        self.channels = require_positive_int(channels, "channels")
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
        """Apply versor product x' = hat(V) x V^{-1} (= RxR~ for grade=2).

        Args:
            x (torch.Tensor): Input [Batch, Channels, Dim].

        Returns:
            torch.Tensor: Transformed input [Batch, Channels, Dim].
        """
        values = x if x.shape[-1] == self.input_layout.dim else self.input_layout.compact(x)
        return self.action(values, self.grade_weights)

    def prune_bivectors(self, threshold: float = 1e-4) -> int:
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
