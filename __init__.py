"""Versor - Geometric Algebra Deep Learning framework for PyTorch."""

__version__ = "1.0.0"

from core.config import AlgebraConfig, make_algebra, make_algebra_from_config
from core.foundation.module import CliffordModule
from core.runtime.algebra import CliffordAlgebra
from core.runtime.context import AlgebraContext
from layers import CliffordLinear, RotorLayer

__all__ = [
    "__version__",
    "AlgebraConfig",
    "AlgebraContext",
    "CliffordAlgebra",
    "CliffordModule",
    "make_algebra",
    "make_algebra_from_config",
    "RotorLayer",
    "CliffordLinear",
]
