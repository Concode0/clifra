# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Runtime algebra hosts and full-basis reference operations."""

from ..storage import LayerLayout, resolve_layer_layout, resolve_layer_layout_contract
from .accessors import active_values, as_multivector, grade_indices, hermitian_signs, materialize_full, resolve_layout
from .actions import (
    apply_multi_versor_action,
    apply_versor_action,
    compact_multi_versor_action,
    compact_versor_action,
    dense_versor_factors,
    grade_norms,
    versor_vector_matrix,
)
from .algebra import CliffordAlgebra
from .context import AlgebraContext
from .multivector import Multivector
from .projected import AlgebraRuntimeMixin

__all__ = [
    "AlgebraContext",
    "AlgebraRuntimeMixin",
    "CliffordAlgebra",
    "Multivector",
    "as_multivector",
    "active_values",
    "apply_multi_versor_action",
    "apply_versor_action",
    "compact_multi_versor_action",
    "compact_versor_action",
    "dense_versor_factors",
    "grade_indices",
    "grade_norms",
    "hermitian_signs",
    "materialize_full",
    "resolve_layer_layout",
    "resolve_layer_layout_contract",
    "resolve_layout",
    "LayerLayout",
    "versor_vector_matrix",
]
