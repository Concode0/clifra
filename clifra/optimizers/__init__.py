# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Explicit coefficient adjustments and rotor update building blocks."""

from .updates import (
    PostUpdateSGD,
    clip_coefficients_,
    exponential_update,
    normalize_coefficients_,
    post_update,
    project_to_rotor_tangent_space,
)

__all__ = [
    "clip_coefficients_",
    "normalize_coefficients_",
    "post_update",
    "PostUpdateSGD",
    "project_to_rotor_tangent_space",
    "exponential_update",
]
