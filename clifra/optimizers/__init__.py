# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Tag-aware optimizers for geometric algebra neural networks.

The optimizers dispatch post-update handling for parameter groups marked
``spin``, ``sphere``, or ``euclidean``. Their ``from_model()`` constructors
build those groups from parameter tags. The public class names are retained for
API compatibility.
"""

from .riemannian import (
    MANIFOLD_EUCLIDEAN,
    MANIFOLD_SPHERE,
    MANIFOLD_SPIN,
    ExponentialSGD,
    RiemannianAdam,
    exponential_retraction,
    group_parameters_by_manifold,
    make_riemannian_optimizer,
    project_to_tangent_space,
    tag_manifold,
)

__all__ = [
    "ExponentialSGD",
    "RiemannianAdam",
    "project_to_tangent_space",
    "exponential_retraction",
    "tag_manifold",
    "group_parameters_by_manifold",
    "make_riemannian_optimizer",
    "MANIFOLD_SPIN",
    "MANIFOLD_SPHERE",
    "MANIFOLD_EUCLIDEAN",
]
