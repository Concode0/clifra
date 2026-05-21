# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import torch
import torch.nn as nn

from clifra.core.foundation.layout import GradeLayout
from clifra.core.foundation.manifold import MANIFOLD_SPIN, tag_manifold
from clifra.core.foundation.module import CliffordModule
from clifra.core.runtime.actions import dense_versor_factors
from clifra.core.runtime.algebra import CliffordAlgebra
from clifra.core.storage import resolve_layer_storage

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
        algebra: CliffordAlgebra,
        channels: int,
        grade: int = 2,
        *,
        input_grades=None,
        output_grades=None,
        input_layout: GradeLayout = None,
        output_layout: GradeLayout = None,
        compact_output: bool = True,
    ):
        """Initialize the versor layer.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            channels (int): Number of features.
            grade (int): Grade of the learnable parameter.
                grade=2 (default): bivectors → rotors via exp(-B/2), Spin group.
                grade=1: vectors → reflections via hat(n) x n^{-1}, Pin group.
                grade=k: general grade-k versor product.
        """
        super().__init__(algebra)
        self.channels = require_positive_int(channels, "channels")
        self.grade = int(grade)
        self.input_storage = resolve_layer_storage(algebra, layout=input_layout, grades=input_grades)
        self.output_storage = (
            resolve_layer_storage(algebra, layout=output_layout, grades=output_grades)
            if output_layout is not None or output_grades is not None
            else self.input_storage
        )
        self.input_layout = self.input_storage.layout
        self.output_layout = self.output_storage.layout
        self.compact_output = bool(compact_output)

        self.register_buffer("grade_indices", grade_indices(algebra, self.grade))
        self.num_grade_elements = self.grade_indices.numel()
        self.parameter_layout = algebra.layout((self.grade,))

        self.grade_weights = nn.Parameter(torch.Tensor(self.channels, self.num_grade_elements))
        if self.grade == 2:
            tag_manifold(self.grade_weights, MANIFOLD_SPIN)

        # Versor cache for eval mode
        self._cached_V_left = None
        self._cached_V_right = None

        self.reset_parameters()

    # --- Backward-compat aliases (grade == 2 usage) ---

    @property
    def bivector_indices(self):
        return self.grade_indices

    @property
    def num_bivectors(self):
        return self.num_grade_elements

    @property
    def bivector_weights(self):
        return self.grade_weights

    # ---------------------------------------------------

    def reset_parameters(self):
        """Initialize with near-identity transform (small weights)."""
        nn.init.normal_(self.grade_weights, std=0.01)

    def _build_grade_element(self, device, dtype):
        """Scatter grade_weights into full multivector dimension [channels, dim]."""
        weights = self.grade_weights.to(device=device, dtype=dtype)
        return self.parameter_layout.dense(weights)

    def _compute_versors(self, device, dtype):
        """Compute left and right factors for per_channel_sandwich.

        For grade=2: left = R = exp(-B/2), right = R~ (reverse).
        For grade=k: left = hat(V) (grade involution), right = V^{-1} (blade inverse).
          V is L2-normalized per channel before inversion so that blade_inverse
          remains exact (norm_sq is purely scalar for unit-norm grade-k elements).

        Returns:
            Tuple[Tensor, Tensor]: (V_left [C, dim], V_right [C, dim])
        """
        weights = self.grade_weights.to(device=device, dtype=dtype)
        return dense_versor_factors(
            self.algebra,
            weights,
            grade=self.grade,
            parameter_layout=self.parameter_layout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply versor product x' = hat(V) x V^{-1} (= RxR~ for grade=2).

        Caches versors during eval mode for faster inference.

        Args:
            x (torch.Tensor): Input [Batch, Channels, Dim].

        Returns:
            torch.Tensor: Transformed input [Batch, Channels, Dim].
        """
        cache = (
            (self._cached_V_left, self._cached_V_right)
            if not self.training and self._cached_V_left is not None and self._cached_V_right is not None
            else None
        )
        out, next_cache = self.algebra.versor_action(
            x,
            self.grade_weights,
            grade=self.grade,
            input_layout=self.input_layout,
            output_layout=self.output_layout,
            parameter_layout=self.parameter_layout,
            compact_output=self.compact_output,
            channels=self.channels,
            name="RotorLayer input",
            dense_cache=cache,
            cache_dense=not self.training,
            return_cache=True,
        )
        if not self.training and next_cache is not None:
            self._cached_V_left, self._cached_V_right = next_cache
        return out

    def train(self, mode: bool = True):
        """Invalidate versor cache when switching to train mode."""
        if mode:
            self._cached_V_left = None
            self._cached_V_right = None
        return super().train(mode)

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
            self.grade_weights.data.mul_(mask.to(dtype=self.grade_weights.dtype))
        return num_pruned

    def sparsity_loss(self) -> torch.Tensor:
        """Compute L1 sparsity regularization on grade weights."""
        return torch.norm(self.grade_weights, p=1)
