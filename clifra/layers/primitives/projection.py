# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Projection and neutralization primitives for multivector channels."""

import torch
import torch.nn as nn

from clifra.core.foundation.layout import GradeLayout
from clifra.core.foundation.module import AlgebraLike, CliffordModule
from clifra.core.foundation.numerics import covariance_regularizer
from clifra.core.storage import resolve_layer_layout_contract
from clifra.utils.mps import safe_linalg_solve

from ._utils import require_positive_int


class BladeSelector(CliffordModule):
    """Blade Selector. Filters insignificant components.

    Learns to weigh geometric grades, suppressing less relevant ones.

    Attributes:
        weights (nn.Parameter): Gate logits [Channels, Dim].
    """

    def __init__(self, algebra: AlgebraLike, channels: int, *, grades=None, layout: GradeLayout = None):
        """Sets up the selector.

        Args:
            algebra: Planner-capable algebra host.
            channels (int): Input features.
        """
        super().__init__(algebra)
        self.channels = require_positive_int(channels, "channels")
        self.layout_contract = resolve_layer_layout_contract(algebra, layout=layout, grades=grades)
        self.layout = self.layout_contract.layout
        self.lane_dim = self.layout_contract.lane_dim

        self.weights = nn.Parameter(torch.Tensor(self.channels, self.lane_dim))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize logits so the selector starts as pass-through."""
        nn.init.zeros_(self.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gates the grades.

        The gate is ``2 * sigmoid(weights)`` so zero logits preserve the input.

        Args:
            x (torch.Tensor): Input [Batch, Channels, Dim].

        Returns:
            torch.Tensor: Filtered input.
        """
        self.layout_contract.validate_input(
            x,
            channels=self.channels,
            name="BladeSelector input",
        )
        gate_shape = (1,) * (x.ndim - 2) + (self.channels, self.lane_dim)
        gate = 2.0 * torch.sigmoid(self.weights).view(gate_shape)
        return x * gate


class GeometricNeutralizer(CliffordModule):
    """Geometric Neutralization. Orthogonalizes Grade-0 against Grade-2 in real-time.

    Removes the component of the Grade-0 (scalar) signal that is parallel to the
    Grade-2 (bivector) subspace.

    Uses Exponential Moving Average (EMA) to maintain stable covariance statistics
    across batches, ensuring batch-independent behavior during inference.

    Attributes:
        algebra: Planner-capable algebra host.
        momentum (float): EMA momentum.
    """

    def __init__(
        self,
        algebra: AlgebraLike,
        channels: int,
        momentum: float = 0.1,
        *,
        grades=None,
        layout: GradeLayout = None,
    ):
        """Initialize the neutralizer.

        Args:
            algebra: Planner-capable algebra host.
            channels (int): Number of multivector channels.
            momentum (float): EMA momentum for covariance tracking.
        """
        super().__init__(algebra)
        self.channels = require_positive_int(channels, "channels")
        self.momentum = momentum
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")
        self.layout_contract = resolve_layer_layout_contract(algebra, layout=layout, grades=grades)
        self.layout = self.layout_contract.layout
        self.lane_dim = self.layout_contract.lane_dim

        self.register_buffer("g0_idx", self.layout_contract.grade_positions(0))
        self.register_buffer("g2_idx", self.layout_contract.grade_positions(2))
        if self.g0_idx.numel() == 0 or self.g2_idx.numel() == 0:
            raise ValueError("GeometricNeutralizer layout must include grades 0 and 2")

        self.d0 = self.g0_idx.numel()
        self.d2 = self.g2_idx.numel()

        # EMA Buffers for each channel
        # We track:
        #   - Mean of scalar (Grade-0): [C, D0]
        #   - Mean of bivector (Grade-2): [C, D2]
        #   - Covariance(bivector, bivector): [C, D2, D2]
        #   - Covariance(bivector, scalar): [C, D2, D0]
        self.register_buffer("running_mean_scalar", torch.zeros(self.channels, self.d0))
        self.register_buffer("running_mean_bivec", torch.zeros(self.channels, self.d2))
        self.register_buffer("running_cov_bb", torch.eye(self.d2).unsqueeze(0).repeat(self.channels, 1, 1))
        self.register_buffer("running_cov_bs", torch.zeros(self.channels, self.d2, self.d0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Neutralizes the multivector signal using EMA statistics.

        Args:
            x (torch.Tensor): Input [Batch, Channels, Dim].

        Returns:
            torch.Tensor: Neutralized multivector.
        """
        self.layout_contract.validate_input(
            x,
            channels=self.channels,
            name="GeometricNeutralizer input",
        )

        x_flat = x.reshape(-1, self.channels, self.lane_dim)
        scalar = x_flat[..., self.g0_idx]
        bivec = x_flat[..., self.g2_idx]
        batch = scalar.shape[0]

        if self.training:
            batch_mean_s = scalar.mean(dim=0)
            batch_mean_b = bivec.mean(dim=0)

            s_centered = scalar - batch_mean_s.unsqueeze(0)
            b_centered = bivec - batch_mean_b.unsqueeze(0)
            denom = max(batch - 1, 1)

            batch_cov_bb = torch.einsum("bci,bcj->cij", b_centered, b_centered) / denom
            batch_cov_bs = torch.einsum("bci,bcj->cij", b_centered, s_centered) / denom

            with torch.no_grad():
                self.running_mean_scalar.lerp_(batch_mean_s.detach(), self.momentum)
                self.running_mean_bivec.lerp_(batch_mean_b.detach(), self.momentum)
                self.running_cov_bb.lerp_(batch_cov_bb.detach(), self.momentum)
                self.running_cov_bs.lerp_(batch_cov_bs.detach(), self.momentum)

            cur_mean_b = batch_mean_b
            cur_cov_bb = batch_cov_bb
            cur_cov_bs = batch_cov_bs
        else:
            # Use EMA stats during inference
            cur_mean_b = self.running_mean_bivec
            cur_cov_bb = self.running_cov_bb
            cur_cov_bs = self.running_cov_bs

        reg = covariance_regularizer(cur_cov_bb) * torch.eye(
            self.d2,
            device=cur_cov_bb.device,
            dtype=cur_cov_bb.dtype,
        ).unsqueeze(0)
        weights = safe_linalg_solve(cur_cov_bb + reg, cur_cov_bs)

        b_centered = bivec - cur_mean_b.unsqueeze(0)
        projection = torch.einsum("bci,cij->bcj", b_centered, weights)
        scalar_n = scalar - projection

        delta = torch.zeros_like(x_flat)
        delta[..., self.g0_idx] = scalar_n - scalar
        return (x_flat + delta).reshape_as(x)
