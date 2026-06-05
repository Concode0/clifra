# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Compile-friendly tensor executors produced by Clifra planners."""

from .action import (
    FullSandwichActionExecutor,
    GradedLinearActionExecutor,
    MultiVersorActionExecutor,
    PairedBivectorActionExecutor,
    VersorActionExecutor,
    apply_graded_linear_action,
    apply_multi_graded_linear_action,
    bivector_vector_generator,
    full_paired_bivector_factors,
    full_versor_factors,
    paired_bivector_factors,
    reflection_vector_matrix,
)
from .attention import GeometricAttentionScoreExecutor
from .exp import BivectorExpExecutor
from .handles import (
    FullSandwichActionHandle,
    MultiVersorActionHandle,
    PairedBivectorActionHandle,
    ProductPlanHandle,
    UnaryPlanHandle,
    VersorActionHandle,
)
from .metric import NormSquaredExecutor
from .permutation import DualExecutor
from .product import FullTableProductExecutor, GradeProductExecutor
from .unary import GradeUnaryExecutor

__all__ = [
    "BivectorExpExecutor",
    "FullTableProductExecutor",
    "DualExecutor",
    "GeometricAttentionScoreExecutor",
    "GradeProductExecutor",
    "GradeUnaryExecutor",
    "FullSandwichActionExecutor",
    "GradedLinearActionExecutor",
    "MultiVersorActionExecutor",
    "PairedBivectorActionExecutor",
    "VersorActionExecutor",
    "apply_graded_linear_action",
    "apply_multi_graded_linear_action",
    "bivector_vector_generator",
    "full_paired_bivector_factors",
    "full_versor_factors",
    "paired_bivector_factors",
    "reflection_vector_matrix",
    "FullSandwichActionHandle",
    "ProductPlanHandle",
    "UnaryPlanHandle",
    "VersorActionHandle",
    "MultiVersorActionHandle",
    "PairedBivectorActionHandle",
    "NormSquaredExecutor",
]
