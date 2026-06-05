# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Geometric activation layers."""

import torch
import torch.nn as nn

from clifra.core.foundation.layout import GradeLayout
from clifra.core.foundation.module import AlgebraLike, CliffordModule
from clifra.core.storage import resolve_layer_layout_contract
from clifra.functional.activation import geometric_gelu, geometric_square, grade_swish

from ._utils import require_positive_int


class GeometricGELU(CliffordModule):
    """Magnitude-gated GELU that preserves multivector direction."""

    def __init__(self, algebra: AlgebraLike, channels: int = 1, *, grades=None, layout: GradeLayout = None):
        """Initialize Geometric GELU."""
        super().__init__(algebra)
        self.channels = require_positive_int(channels, "channels")
        self.layout_contract = resolve_layer_layout_contract(algebra, layout=layout, grades=grades)
        self.layout = self.layout_contract.layout
        self.bias = nn.Parameter(torch.zeros(self.channels))

    def _validate_input(self, x: torch.Tensor, name: str) -> None:
        channels = x.shape[-2] if self.channels == 1 else self.channels
        self.layout_contract.validate_input(x, channels=channels, name=name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply geometric GELU activation."""
        self._validate_input(x, "GeometricGELU input")
        return geometric_gelu(x, bias=self.bias)


class GeometricSquare(CliffordModule):
    """Gated geometric self-product: ``x + gate * (x * x)``."""

    def __init__(self, algebra: AlgebraLike, channels: int = 1, *, grades=None, layout: GradeLayout = None):
        """Initialize gated geometric self-product."""
        super().__init__(algebra)
        self.channels = require_positive_int(channels, "channels")
        self.layout_contract = resolve_layer_layout_contract(algebra, layout=layout, grades=grades)
        self.layout = self.layout_contract.layout
        self.gate_logit = nn.Parameter(torch.full((self.channels,), -2.0))

    def _validate_input(self, x: torch.Tensor, name: str) -> None:
        channels = x.shape[-2] if self.channels == 1 else self.channels
        self.layout_contract.validate_input(x, channels=channels, name=name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated geometric square."""
        self._validate_input(x, "GeometricSquare input")
        return geometric_square(self.algebra, x, gate=torch.sigmoid(self.gate_logit), layout=self.layout)


class GradeSwish(CliffordModule):
    """Per-grade sigmoid gate based on grade norms."""

    def __init__(self, algebra: AlgebraLike, channels: int = 1, *, grades=None, layout: GradeLayout = None):
        """Initialize grade-wise Swish activation."""
        super().__init__(algebra)
        self.channels = require_positive_int(channels, "channels")
        self.layout_contract = resolve_layer_layout_contract(algebra, layout=layout, grades=grades)
        self.layout = self.layout_contract.layout
        self.n_grades = self.algebra.n + 1

        self.grade_weights = nn.Parameter(torch.ones(self.n_grades))
        self.grade_biases = nn.Parameter(torch.zeros(self.n_grades))
        self.register_buffer("_grade_index", self.layout.grade_indices_tensor(device=algebra.device))

    def _validate_input(self, x: torch.Tensor, name: str) -> None:
        channels = x.shape[-2] if self.channels == 1 else self.channels
        self.layout_contract.validate_input(x, channels=channels, name=name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-grade gating."""
        self._validate_input(x, "GradeSwish input")
        return grade_swish(
            x,
            grade_index=self._grade_index,
            grade_weights=self.grade_weights,
            grade_biases=self.grade_biases,
            n_grades=self.n_grades,
        )
