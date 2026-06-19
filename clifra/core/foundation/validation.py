# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Lightweight input validation for clifra tensors.

Assertion helpers use ``assert`` so they are free under ``python -O``.
Set ``VALIDATE = False`` to disable even without the -O flag.

Public boundary validators raise ``ValueError``. They intentionally remain
enabled outside the hot unchecked executor paths.
"""

import torch

VALIDATE = True


def check_multivector(x: torch.Tensor, algebra, name: str = "x") -> None:
    """Assert *x* looks like a multivector for *algebra*.

    Checks ``x.ndim >= 1`` and ``x.shape[-1] == algebra.dim``.
    """
    if not VALIDATE:
        return
    assert x.ndim >= 1, f"{name}: expected ndim >= 1, got shape {tuple(x.shape)}"
    assert x.shape[-1] == algebra.dim, (
        f"{name}: last dim should be {algebra.dim} (algebra dim), got {x.shape[-1]} (shape {tuple(x.shape)})"
    )


def check_channels(x: torch.Tensor, expected: int, name: str = "x") -> None:
    """Assert the channel dimension of *x* equals *expected*.

    Assumes layout ``[Batch, Channels, Dim]`` (ndim >= 3, channel at -2).
    """
    if not VALIDATE:
        return
    assert x.ndim >= 3, f"{name}: expected ndim >= 3 for channel check, got shape {tuple(x.shape)}"
    assert x.shape[-2] == expected, f"{name}: expected {expected} channels, got {x.shape[-2]} (shape {tuple(x.shape)})"


def validate_layout_lanes(values: torch.Tensor, layout, name: str = "values") -> None:
    """Validate that the trailing lane axis matches ``layout``."""
    if values.ndim < 1:
        raise ValueError(f"{name}: expected ndim >= 1, got shape {tuple(values.shape)}")
    if values.shape[-1] != layout.dim:
        raise ValueError(f"{name}: last dim must be {layout.dim} for grades {layout.grades}, got {values.shape[-1]}")


def validate_channel_values(values: torch.Tensor, layout, channels: int, name: str = "values") -> None:
    """Validate ``[..., channels, lanes]`` tensors against a grade layout."""
    if values.ndim < 3:
        raise ValueError(f"{name}: expected ndim >= 3, got shape {tuple(values.shape)}")
    if values.shape[-2] != channels:
        raise ValueError(f"{name}: expected {channels} channels, got {values.shape[-2]}")
    validate_layout_lanes(values, layout, name)
