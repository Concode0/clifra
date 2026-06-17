# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Numerical guard helpers shared across clifra core and layers."""

from __future__ import annotations

import torch

DEFAULT_EPS_MULTIPLIER = 32.0


def eps_for(dtype: torch.dtype, *, multiplier: float = 1.0, min_value: float = 0.0) -> float:
    """Return a floating-point epsilon scaled for ``dtype``."""
    try:
        eps = torch.finfo(dtype).eps
    except TypeError:
        eps = torch.finfo(torch.float32).eps
    return max(float(eps) * float(multiplier), float(min_value))


def eps_like(values: torch.Tensor, *, multiplier: float = 1.0, min_value: float = 0.0) -> float:
    """Return a dtype-aware epsilon for ``values``."""
    return eps_for(values.dtype, multiplier=multiplier, min_value=min_value)


def signed_clamp_min(values: torch.Tensor, eps: float | torch.Tensor) -> torch.Tensor:
    """Clamp magnitude while preserving denominator sign."""
    eps_tensor = torch.as_tensor(eps, device=values.device, dtype=values.dtype)
    magnitude = values.abs().clamp_min(eps_tensor)
    return torch.where(values < 0, -magnitude, magnitude)


def covariance_regularizer(covariance: torch.Tensor, *, multiplier: float = DEFAULT_EPS_MULTIPLIER) -> torch.Tensor:
    """Return a scale-aware diagonal regularizer for covariance matrices."""
    if covariance.ndim < 2:
        raise ValueError(f"covariance must have matrix axes, got shape {tuple(covariance.shape)}")
    scale = covariance.diagonal(dim1=-2, dim2=-1).abs().mean(dim=-1, keepdim=True)
    scale = scale.clamp_min(1.0).unsqueeze(-1)
    return eps_like(covariance, multiplier=multiplier) * scale
