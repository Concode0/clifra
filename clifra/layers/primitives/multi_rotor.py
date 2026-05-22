# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Multi-versor superposition layers with universal grade parameterization.

Implements versor-based transformations using weighted sums of sandwich products.
"""

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


class MultiRotorLayer(CliffordModule):
    """Multi-versor layer with weighted superposition: x' = sum_k w_k hat(V_k) x V_k^{-1}.

    For grade=2 (default): each V_k = exp(-B_k/2) is a rotor, reducing to
    x' = sum_k w_k R_k x R~_k.
    For grade=k: each V_k is a grade-k versor applied via the general versor product.

    The exp strategy is controlled by ``algebra.exp_policy``.

    Attributes:
        channels (int): Input features.
        num_rotors (int): Number of overlapping versors.
        grade (int): Grade of the learnable parameters. Default 2 (rotors).
        rotor_grade_weights (nn.Parameter): Grade-k coefficients [num_rotors, num_grade_elements].
        weights (nn.Parameter): Mixing weights [channels, num_rotors].
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        channels: int,
        num_rotors: int = 8,
        grade: int = 2,
        *,
        input_grades=None,
        output_grades=None,
        input_layout: GradeLayout = None,
        output_layout: GradeLayout = None,
    ):
        """Initialize Multi-Versor Layer.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
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
        self.input_storage = resolve_layer_storage(algebra, layout=input_layout, grades=input_grades)
        self.output_storage = (
            resolve_layer_storage(algebra, layout=output_layout, grades=output_grades)
            if output_layout is not None or output_grades is not None
            else self.input_storage
        )
        self.input_layout = self.input_storage.layout
        self.output_layout = self.output_storage.layout

        self.register_buffer("grade_indices", grade_indices(algebra, self.grade))
        self.num_grade_elements = self.grade_indices.numel()
        self.parameter_layout = algebra.layout((self.grade,))

        self.rotor_grade_weights = nn.Parameter(torch.Tensor(self.num_rotors, self.num_grade_elements))
        if self.grade == 2:
            tag_manifold(self.rotor_grade_weights, MANIFOLD_SPIN)

        # Mixing weights (Euclidean — intentionally untagged)
        self.weights = nn.Parameter(torch.Tensor(self.channels, self.num_rotors))

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
    def rotor_bivectors(self):
        return self.rotor_grade_weights

    # ---------------------------------------------------

    def reset_parameters(self):
        """Initialize with small transforms and uniform mixing weights."""
        nn.init.normal_(self.rotor_grade_weights, std=0.01)
        nn.init.xavier_uniform_(self.weights)

    def _compute_versors(self, device, dtype):
        """Compute left and right factors for all K versors.

        For grade=2: left = R_k = exp(-B_k/2), right = R~_k.
        For grade=k: left = hat(V_k), right = V_k^{-1}.

        Returns:
            Tuple[Tensor, Tensor]: (V_left [K, dim], V_right [K, dim])
        """
        weights = self.rotor_grade_weights.to(device=device, dtype=dtype)
        return dense_versor_factors(
            self.algebra,
            weights,
            grade=self.grade,
            parameter_layout=self.parameter_layout,
        )

    def forward(self, x: torch.Tensor, return_invariants: bool = False) -> torch.Tensor:
        """Apply weighted multi-versor superposition.

        Caches versors during eval mode for faster inference.

        Args:
            x (torch.Tensor): Input [Batch, Channels, Dim].
            return_invariants (bool): If True, returns per-grade norms instead of output.

        Returns:
            torch.Tensor: Transformed output [Batch, Channels, Dim].
        """
        cache = (
            (self._cached_V_left, self._cached_V_right)
            if not self.training and self._cached_V_left is not None and self._cached_V_right is not None
            else None
        )
        out, next_cache = self.algebra.multi_versor_action(
            x,
            self.rotor_grade_weights,
            self.weights,
            grade=self.grade,
            input_layout=self.input_layout,
            output_layout=self.output_layout,
            parameter_layout=self.parameter_layout,
            compact_output=self.output_layout is not None,
            channels=self.channels,
            name="MultiRotorLayer input",
            dense_cache=cache,
            cache_dense=not self.training,
            return_cache=True,
        )
        if not self.training and next_cache is not None:
            self._cached_V_left, self._cached_V_right = next_cache

        if return_invariants:
            return self.algebra.grade_norms(out, layout=self.output_layout)

        return out

    def train(self, mode: bool = True):
        """Invalidate versor cache when switching to train mode."""
        if mode:
            self._cached_V_left = None
            self._cached_V_right = None
        return super().train(mode)

    def sparsity_loss(self) -> torch.Tensor:
        """Compute L1 sparsity loss for versor weights and mixing weights."""
        return torch.norm(self.rotor_grade_weights, p=1) + torch.norm(self.weights, p=1)
