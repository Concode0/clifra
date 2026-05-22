"""Clifra - Clifford geometric algebra deep learning framework for PyTorch."""

__version__ = "1.0.0"

from clifra.core.config import AlgebraConfig, make_algebra, make_algebra_from_config
from clifra.core.formatting import Multivector, format_multivector
from clifra.core.foundation.module import CliffordModule
from clifra.core.runtime.algebra import AlgebraContext, CliffordAlgebra
from clifra.layers import CliffordLinear, RotorLayer

__all__ = [
    "__version__",
    "AlgebraConfig",
    "AlgebraContext",
    "CliffordAlgebra",
    "CliffordModule",
    "Multivector",
    "format_multivector",
    "make_algebra",
    "make_algebra_from_config",
    "RotorLayer",
    "CliffordLinear",
]
