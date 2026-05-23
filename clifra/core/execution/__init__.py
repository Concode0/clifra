# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Compile-friendly tensor executors produced by Clifra planners."""

from .action import (
    GradedLinearActionExecutor,
    MultiVersorActionExecutor,
    PairedBivectorActionExecutor,
    VersorActionExecutor,
    apply_graded_linear_action,
    apply_multi_graded_linear_action,
    bivector_vector_generator,
    dense_paired_bivector_factors,
    paired_bivector_factors,
    reflection_vector_matrix,
)
from .attention import GeometricAttentionScoreExecutor

__all__ = [
    "GeometricAttentionScoreExecutor",
    "GradedLinearActionExecutor",
    "MultiVersorActionExecutor",
    "PairedBivectorActionExecutor",
    "VersorActionExecutor",
    "apply_graded_linear_action",
    "apply_multi_graded_linear_action",
    "bivector_vector_generator",
    "dense_paired_bivector_factors",
    "paired_bivector_factors",
    "reflection_vector_matrix",
]
