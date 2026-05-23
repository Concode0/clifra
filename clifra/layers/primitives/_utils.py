# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Shared implementation helpers for primitive layers."""

from __future__ import annotations

from typing import Iterable

import torch


def require_positive_int(value: int, name: str) -> int:
    """Validate a positive integer layer dimension."""
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def require_choice(value: str, name: str, choices: Iterable[str]) -> str:
    """Validate a string option and return it unchanged."""
    options = tuple(choices)
    if value not in options:
        supported = ", ".join(repr(option) for option in options)
        raise ValueError(f"{name} must be one of {supported}, got {value!r}")
    return value


def grade_indices(algebra, grade: int, *, name: str = "grade") -> torch.Tensor:
    """Return dense basis indices for a grade with consistent errors."""
    grade = int(grade)
    if grade < 0 or grade >= algebra.num_grades:
        raise ValueError(f"{name} must be in [0, {algebra.num_grades - 1}], got {grade}")
    indices = algebra.grade_indices((grade,))
    if indices.numel() == 0:
        raise ValueError(f"{name}={grade} has no basis elements in this algebra")
    return indices


def dense_from_indices(coefficients: torch.Tensor, indices: torch.Tensor, dense_dim: int) -> torch.Tensor:
    """Scatter active coefficients into full-basis multivector lanes."""
    dense = coefficients.new_zeros(*coefficients.shape[:-1], dense_dim)
    index = indices.to(device=coefficients.device).expand(*coefficients.shape[:-1], -1)
    return dense.scatter(-1, index, coefficients)


def channel_mix(in_channels: int, out_channels: int, *, normalize: bool) -> torch.Tensor:
    """Build a deterministic channel routing matrix [out_channels, in_channels].

    Compression assigns every input channel to one output bin. Expansion repeats
    input channels across output bins. The normalized form averages each output
    bin; the unnormalized form sums it.
    """
    in_channels = require_positive_int(in_channels, "in_channels")
    out_channels = require_positive_int(out_channels, "out_channels")
    mix = torch.zeros(out_channels, in_channels)

    if in_channels >= out_channels:
        source = torch.arange(in_channels)
        target = torch.div(source * out_channels, in_channels, rounding_mode="floor").clamp_max(out_channels - 1)
        mix[target, source] = 1.0
    else:
        target = torch.arange(out_channels)
        source = torch.div(target * in_channels, out_channels, rounding_mode="floor").clamp_max(in_channels - 1)
        mix[target, source] = 1.0

    if normalize:
        mix = mix / mix.sum(dim=1, keepdim=True).clamp_min(1.0)
    return mix


def pair_mean(ch2pair: torch.Tensor, num_pairs: int) -> torch.Tensor:
    """Build [num_pairs, in_channels] means from a channel-to-pair map."""
    mix = torch.zeros(num_pairs, ch2pair.numel())
    source = torch.arange(ch2pair.numel())
    mix[ch2pair, source] = 1.0
    return mix / mix.sum(dim=1, keepdim=True).clamp_min(1.0)
