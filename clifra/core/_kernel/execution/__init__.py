# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Compile-friendly tensor executors produced by clifra planners."""

from .action import (
    FullSandwichActionExecutor,
    GradedLinearActionExecutor,
    MultiVersorActionExecutor,
    PairedBivectorActionExecutor,
    VersorActionExecutor,
    full_versor_factors,
)
from .attention import GeometricAttentionScoreExecutor
from .exp import BivectorExpExecutor
from .metric import SignatureNormSquaredExecutor
from .permutation import PseudoscalarProductExecutor
from .product import FullTableProductExecutor, GradeProductExecutor
from .unary import GradeUnaryExecutor

__all__ = [
    "BivectorExpExecutor",
    "FullSandwichActionExecutor",
    "FullTableProductExecutor",
    "PseudoscalarProductExecutor",
    "GeometricAttentionScoreExecutor",
    "GradeProductExecutor",
    "GradeUnaryExecutor",
    "GradedLinearActionExecutor",
    "MultiVersorActionExecutor",
    "PairedBivectorActionExecutor",
    "VersorActionExecutor",
    "full_versor_factors",
    "FullSandwichActionExecutor",
    "MultiVersorActionExecutor",
    "PairedBivectorActionExecutor",
    "SignatureNormSquaredExecutor",
]
