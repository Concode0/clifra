# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import torch
import torch.nn as nn

from clifra.core.foundation.layout import GradeLayout
from clifra.core.foundation.manifold import MANIFOLD_SPHERE, tag_manifold
from clifra.core.foundation.module import CliffordModule
from clifra.core.foundation.numerics import eps_like
from clifra.core.runtime.algebra import CliffordAlgebra
from clifra.core.storage import resolve_layer_layout_contract

from ._utils import (
    grade_indices,
    require_positive_int,
)


class ReflectionLayer(CliffordModule):
    """Learnable reflection layer via unit vectors.

    Each channel learns a unit vector n_c and applies the reflection
    x'_c = -n_c x_c n_c^{-1}. This is the fundamental odd versor
    transformation - rotors (even versors) are compositions of two
    reflections.

    The learned vectors are projected to unit norm before each reflection.
    For Euclidean signature, the projection is simple L2 normalization.
    For mixed signature, the projection normalizes by |<n ~n>_0|.

    Attributes:
        channels (int): Number of reflection vectors.
        vector_weights (nn.Parameter): Learnable grade-1 coefficients [C, n].
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        channels: int,
        *,
        input_grades=None,
        output_grades=None,
        input_layout: GradeLayout = None,
        output_layout: GradeLayout = None,
    ):
        """Initialize the reflection layer.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            channels (int): Number of features.
        """
        super().__init__(algebra)
        self.channels = require_positive_int(channels, "channels")
        self.input_contract = resolve_layer_layout_contract(algebra, layout=input_layout, grades=input_grades)
        self.output_contract = (
            resolve_layer_layout_contract(algebra, layout=output_layout, grades=output_grades)
            if output_layout is not None or output_grades is not None
            else self.input_contract
        )
        self.input_layout = self.input_contract.layout
        self.output_layout = self.output_contract.layout

        self.register_buffer("vector_indices", grade_indices(algebra, 1, name="vector grade"))
        self.num_vectors = self.vector_indices.numel()
        self.vector_layout = algebra.layout((1,))

        self.vector_weights = nn.Parameter(torch.Tensor(self.channels, self.num_vectors))
        tag_manifold(self.vector_weights, MANIFOLD_SPHERE)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize with random unit-ish vectors."""
        nn.init.normal_(self.vector_weights, std=1.0)
        # Normalize to unit vectors
        with torch.no_grad():
            norms = self.vector_weights.norm(dim=-1, keepdim=True).clamp_min(eps_like(self.vector_weights))
            self.vector_weights.div_(norms)

    def _build_vectors(self, device, dtype):
        """Build full multivectors from vector weights and normalize.

        Returns:
            Tuple of (n, n_inv) each [C, dim].
        """
        weights = self.vector_weights.to(device=device, dtype=dtype)
        n = self.vector_layout.dense(weights)

        # Normalize: n_hat = n / sqrt(|<n ~n>_0|)
        n_sq = self.algebra.norm_sq(n)  # [C, 1]
        scale = n_sq.abs().clamp_min(eps_like(n_sq)).sqrt()
        n = n / scale

        n_inv = self.algebra.blade_inverse(n)
        return n, n_inv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-channel reflections: x'_c = -n_c x_c n_c^{-1}.

        Args:
            x (torch.Tensor): Input [Batch, Channels, Dim].

        Returns:
            torch.Tensor: Reflected input [Batch, Channels, Dim].
        """
        return self.algebra.versor_action(
            x,
            self.vector_weights,
            grade=1,
            input_layout=self.input_layout,
            output_layout=self.output_layout,
            parameter_layout=self.vector_layout,
            channels=self.channels,
            name="ReflectionLayer input",
        )

    def sparsity_loss(self) -> torch.Tensor:
        """Compute L1 sparsity regularization on vector weights."""
        return torch.norm(self.vector_weights, p=1)
