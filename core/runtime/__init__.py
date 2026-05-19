# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Runtime algebra hosts and dense reference operations."""

from .accessors import as_multivector, compact_values, grade_indices, hermitian_signs, materialize_dense, resolve_layout
from .actions import compact_multi_versor_action, compact_versor_action, versor_vector_matrix
from .algebra import CliffordAlgebra
from .context import AlgebraContext
from .layers import LayerStorage, resolve_layer_layout, resolve_layer_storage
from .multivector import Multivector
from .projected import AlgebraRuntimeMixin

__all__ = [
    "AlgebraContext",
    "AlgebraRuntimeMixin",
    "CliffordAlgebra",
    "Multivector",
    "as_multivector",
    "compact_values",
    "compact_multi_versor_action",
    "compact_versor_action",
    "grade_indices",
    "hermitian_signs",
    "materialize_dense",
    "resolve_layer_layout",
    "resolve_layer_storage",
    "resolve_layout",
    "LayerStorage",
    "versor_vector_matrix",
]
