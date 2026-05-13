# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

from typing import Iterable, Optional

import torch
import torch.nn as nn

from core.foundation.module import CliffordModule

from ..planning import lane_count, resolve_layer_layout


class MultivectorEmbedding(CliffordModule):
    """Token embedding as multivectors.

    Each token maps to a [channels, dim] multivector. Initializes
    content in grade-1 (vector) subspace only - semantic content
    starts as directed quantities before rotors act on them.

    Attributes:
        vocab_size (int): Number of tokens.
        channels (int): Number of multivector channels.
        embedding (nn.Embedding): Underlying embedding table.
    """

    optimization_operators = ("embed",)
    optimization_input_grades = None

    def __init__(
        self,
        algebra,
        vocab_size: int,
        channels: int,
        grades: Optional[Iterable[int]] = None,
    ):
        """Sets up the multivector embedding.

        Args:
            algebra: Clifford algebra instance.
            vocab_size: Vocabulary size.
            channels: Number of multivector channels per token.
            grades: Optional declared output grades. When set, the
                embedding table stores compact lanes only.
        """
        super().__init__(algebra)
        self.vocab_size = vocab_size
        self.channels = channels
        self.layout = resolve_layer_layout(algebra, grades)
        self.basis_dim = lane_count(algebra, self.layout)

        # Single flat embedding: vocab_size -> channels * active basis lanes
        self.embedding = nn.Embedding(vocab_size, channels * self.basis_dim)
        self._init_grade1()

    def _init_grade1(self):
        """Initializes only grade-1 components; zeros out all others."""
        with torch.no_grad():
            channels = self.channels
            dim = self.basis_dim

            if self.layout is None:
                grade1_flat = [i for i in range(dim) if bin(i).count("1") == 1]
            else:
                grade1_flat = [pos for pos, index in enumerate(self.layout.basis_indices) if bin(index).count("1") == 1]

            # Zero everything
            self.embedding.weight.zero_()

            # Fill grade-1 slots with small normal values
            for ch in range(channels):
                for idx in grade1_flat:
                    flat_idx = ch * dim + idx
                    self.embedding.weight[:, flat_idx].normal_(std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Maps token ids to multivector embeddings.

        Args:
            token_ids: Token indices [B, L].

        Returns:
            Multivector embeddings [B, L, channels, dim].
        """
        B, L = token_ids.shape
        flat = self.embedding(token_ids)  # [B, L, channels * dim]
        return flat.reshape(B, L, self.channels, self.basis_dim)


class RotaryBivectorPE(CliffordModule):
    """Positional encoding via bivector rotors.

    Each position learns a bivector B_p. The rotor R_p = exp(-B_p/2)
    is applied as a sandwich product to the token embedding. This is
    the GA analog of rotary positional embeddings (RoPE), but acting
    on the full multivector instead of scalar pairs.

    Initialization: sinusoidal - B_pos[p, k] = p * 10000^(-2k/num_bv) * 0.01
    so that early training starts close to identity.

    Attributes:
        num_bivectors (int): Number of bivector basis elements.
        bivector_weights (nn.Parameter): [max_seq_len, num_bivectors] or fixed.
        bivector_indices (torch.Tensor): Indices of grade-2 basis elements.
    """

    optimization_operators = ("dense_sandwich",)
    optimization_parameter_grades = (2,)
    optimization_dense_only_reason = "positional rotor path still materializes dense multivectors"

    def __init__(
        self,
        algebra,
        channels: int,
        max_seq_len: int,
        learnable: bool = True,
    ):
        """Sets up rotary bivector positional encoding.

        Args:
            algebra: Clifford algebra instance.
            channels: Unused (kept for API consistency with other layers).
            max_seq_len: Maximum sequence length supported.
            learnable: If True, bivector weights are learned parameters.
        """
        super().__init__(algebra)
        self.max_seq_len = max_seq_len
        self.learnable = learnable

        # Identify grade-2 basis elements through the planner layout.
        if algebra.n >= 2:
            bivector_indices = algebra.planner.layout((2,)).indices_tensor(device=algebra.device)
        else:
            bivector_indices = torch.zeros(0, dtype=torch.long, device=algebra.device)
        self.register_buffer("bivector_indices", bivector_indices)
        self.num_bivectors = int(bivector_indices.numel())

        # Sinusoidal initialization
        init = self._sinusoidal_init(max_seq_len, self.num_bivectors)

        if learnable:
            self.bivector_weights = nn.Parameter(init)
            self.bivector_weights._manifold = "spin"
        else:
            self.register_buffer("bivector_weights", init)

    def _sinusoidal_init(self, L: int, num_bv: int) -> torch.Tensor:
        """Sinusoidal initialization: B[p, k] = p * 10000^(-2k/num_bv) * 0.01."""
        # Compute in float32 for numerical precision, then cast to algebra dtype.
        positions = torch.arange(L, dtype=torch.float32).unsqueeze(1)  # [L, 1]
        freqs = torch.pow(10000.0, -2.0 * torch.arange(num_bv, dtype=torch.float32) / max(num_bv, 1)).unsqueeze(
            0
        )  # [1, num_bv]
        return (positions * freqs * 0.01).to(dtype=self.algebra.dtype)  # [L, num_bv]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies position-dependent rotor rotations.

        Args:
            x: Token multivectors [B, L, C, D].

        Returns:
            Positionally encoded multivectors [B, L, C, D].
        """
        B, L, C, D = x.shape
        device = x.device
        dtype = x.dtype

        # Build bivectors for each position: [L, D]
        B_pos = torch.zeros(L, D, device=device, dtype=dtype)
        bv_idx = self.bivector_indices  # [num_bv]
        weights = self.bivector_weights[:L]  # [L, num_bv]
        B_pos.scatter_(1, bv_idx.unsqueeze(0).expand(L, -1), weights)

        # Compute rotors: [L, D]
        R = self.algebra.exp(-0.5 * B_pos)
        R_rev = self.algebra.reverse(R)

        # Broadcast for sandwich product over B and C
        # R: [L, D] -> [1, L, 1, D]
        R = R.unsqueeze(0).unsqueeze(2)
        R_rev = R_rev.unsqueeze(0).unsqueeze(2)

        # Flatten B*C for geometric_product: needs [..., D]
        x_flat = x.reshape(B * L * C, D)
        R_flat = R.expand(B, L, C, D).reshape(B * L * C, D)
        R_rev_flat = R_rev.expand(B, L, C, D).reshape(B * L * C, D)

        Rx = self.algebra.geometric_product(R_flat, x_flat)
        RxRr = self.algebra.geometric_product(Rx, R_rev_flat)

        return RxRr.reshape(B, L, C, D)
