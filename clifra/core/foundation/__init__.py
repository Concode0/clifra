# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Foundation value objects, basis utilities, validation, and device helpers."""

from .basis import (
    GradeProductOp,
    basis_count_for_grades,
    basis_index_tuple_for_grades,
    basis_indices_for_grades,
    basis_product,
    expand_output_grades,
    geometric_product_output_grades,
    normalize_grades,
    operation_coefficient,
    operation_may_be_nonzero,
    product_output_grades,
    reverse_sign,
)
from .device import DeviceConfig, dtype_name, optional_dtype, resolve_device, resolve_dtype
from .layout import AlgebraSpec, GradeLayout
from .manifold import (
    MANIFOLD_EUCLIDEAN,
    MANIFOLD_ORDER,
    MANIFOLD_SPHERE,
    MANIFOLD_SPIN,
    VALID_MANIFOLDS,
    format_valid_manifolds,
    tag_manifold,
    validate_manifold,
)
from .module import AlgebraLike, CliffordModule
from .numerics import covariance_regularizer, eps_for, eps_like, signed_clamp_min
from .validation import check_channels, check_multivector

__all__ = [
    "AlgebraLike",
    "AlgebraSpec",
    "CliffordModule",
    "DeviceConfig",
    "GradeLayout",
    "GradeProductOp",
    "MANIFOLD_EUCLIDEAN",
    "MANIFOLD_ORDER",
    "MANIFOLD_SPHERE",
    "MANIFOLD_SPIN",
    "VALID_MANIFOLDS",
    "basis_count_for_grades",
    "basis_index_tuple_for_grades",
    "basis_indices_for_grades",
    "basis_product",
    "check_channels",
    "check_multivector",
    "covariance_regularizer",
    "dtype_name",
    "eps_for",
    "eps_like",
    "expand_output_grades",
    "format_valid_manifolds",
    "geometric_product_output_grades",
    "normalize_grades",
    "operation_coefficient",
    "operation_may_be_nonzero",
    "optional_dtype",
    "product_output_grades",
    "reverse_sign",
    "resolve_device",
    "resolve_dtype",
    "signed_clamp_min",
    "tag_manifold",
    "validate_manifold",
]
