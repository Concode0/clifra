"""Shared manifold metadata for framework parameters."""

from __future__ import annotations

import torch.nn as nn

MANIFOLD_SPIN = "spin"
MANIFOLD_SPHERE = "sphere"
MANIFOLD_EUCLIDEAN = "euclidean"

MANIFOLD_ORDER = (MANIFOLD_SPIN, MANIFOLD_SPHERE, MANIFOLD_EUCLIDEAN)
VALID_MANIFOLDS = frozenset(MANIFOLD_ORDER)


def format_valid_manifolds() -> str:
    """Return a stable, user-facing list of supported manifold names."""
    return ", ".join(repr(name) for name in MANIFOLD_ORDER)


def validate_manifold(manifold: str) -> str:
    """Validate and return a manifold tag."""
    if manifold not in VALID_MANIFOLDS:
        raise ValueError(f"Unknown manifold {manifold!r}. Must be one of {format_valid_manifolds()}")
    return manifold


def tag_manifold(param: nn.Parameter, manifold: str) -> nn.Parameter:
    """Tag a parameter with its manifold type and return the parameter."""
    param._manifold = validate_manifold(manifold)
    return param
