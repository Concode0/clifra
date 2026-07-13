# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Optimizer dispatch tags for framework parameters.

The three accepted labels are deliberately small in scope. They select the
retraction applied by clifra's optimizers; they are not a taxonomy of geometric
objects supported by the algebra, planner, or layer systems.
"""

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
    """Validate and return an optimizer manifold tag."""
    if manifold not in VALID_MANIFOLDS:
        raise ValueError(f"Unknown manifold {manifold!r}. Must be one of {format_valid_manifolds()}")
    return manifold


def tag_manifold(param: nn.Parameter, manifold: str) -> nn.Parameter:
    """Tag a parameter for optimizer retraction dispatch and return it."""
    param._manifold = validate_manifold(manifold)
    return param
