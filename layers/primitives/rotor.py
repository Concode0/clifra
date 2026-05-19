# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import torch
import torch.nn as nn

from core.foundation.layout import GradeLayout
from core.foundation.manifold import MANIFOLD_SPIN, tag_manifold
from core.foundation.module import CliffordModule
from core.runtime.actions import compact_versor_action
from core.runtime.algebra import CliffordAlgebra
from core.runtime.layers import resolve_layer_storage

from ._utils import (
    cache_matches,
    dense_from_indices,
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
    ``algebra.exp_policy`` -- see :class:`core.runtime.decomposition.ExpPolicy`.

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
        return dense_from_indices(weights, self.grade_indices, self.algebra.dim)

    def _compute_versors(self, device, dtype):
        """Compute left and right factors for per_channel_sandwich.

        For grade=2: left = R = exp(-B/2), right = R~ (reverse).
        For grade=k: left = hat(V) (grade involution), right = V^{-1} (blade inverse).
          V is L2-normalized per channel before inversion so that blade_inverse
          remains exact (norm_sq is purely scalar for unit-norm grade-k elements).

        Returns:
            Tuple[Tensor, Tensor]: (V_left [C, dim], V_right [C, dim])
        """
        V = self._build_grade_element(device, dtype)
        if self.grade == 2:
            R = self.algebra.exp(-0.5 * V)
            return R, self.algebra.reverse(R)
        else:
            # Normalize per channel so blade_inverse is exact.
            # For a unit-norm grade-k element, V * V_rev = scalar everywhere.
            norm = V.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            V = V / norm
            return self.algebra.grade_involution(V), self.algebra.blade_inverse(V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply versor product x' = hat(V) x V^{-1} (= RxR~ for grade=2).

        Caches versors during eval mode for faster inference.

        Args:
            x (torch.Tensor): Input [Batch, Channels, Dim].

        Returns:
            torch.Tensor: Transformed input [Batch, Channels, Dim].
        """
        is_compact = self.input_storage.validate_input(
            x,
            channels=self.channels,
            name="RotorLayer input",
            allow_dense=self.input_layout is None or self.input_layout.dim == self.algebra.dim,
        )
        if is_compact:
            return self._forward_compact(x)
        if not hasattr(self.algebra, "per_channel_sandwich"):
            raise ValueError(
                "RotorLayer dense execution requires CliffordAlgebra; declare input_grades for compact use."
            )

        cache = (
            (self._cached_V_left, self._cached_V_right)
            if self._cached_V_left is not None and self._cached_V_right is not None
            else None
        )
        if not self.training and cache_matches(cache, x):
            V_left, V_right = self._cached_V_left, self._cached_V_right
        else:
            V_left, V_right = self._compute_versors(x.device, x.dtype)
            if not self.training:
                self._cached_V_left = V_left
                self._cached_V_right = V_right

        return self.algebra.per_channel_sandwich(V_left, x, V_right)

    def _forward_compact(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a compact grade-preserving versor action."""
        if self.input_layout is None:
            raise ValueError("RotorLayer compact input requires input_layout or input_grades")
        if self.output_layout is None:
            raise ValueError("RotorLayer compact output requires output_layout or output_grades")
        return compact_versor_action(
            self.algebra,
            x,
            self.grade_weights,
            grade=self.grade,
            input_layout=self.input_layout,
            output_layout=self.output_layout,
            parameter_layout=self.parameter_layout,
            compact_output=self.compact_output,
        )

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
