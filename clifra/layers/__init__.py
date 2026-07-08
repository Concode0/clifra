# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Neural network layers built on Clifford algebra.

The framework namespace exports primitives and reusable blocks. Example
adapters live under :mod:`clifra.layers.adapters` and are imported explicitly.
"""

from clifra.core.foundation.module import CliffordModule

from .blocks.attention import EntropyGatedAttention, GeometricProductAttention
from .primitives.activation import GeometricGELU, GeometricSquare, GradeSwish
from .primitives.linear import CliffordLinear
from .primitives.multi_versor import MultiVersorLayer
from .primitives.normalization import CliffordLayerNorm
from .primitives.product import (
    AntiCommutatorLayer,
    AntiCommutatorProductLayer,
    CommutatorLayer,
    CommutatorProductLayer,
    GeometricProductLayer,
    InnerProductLayer,
    LeftContractionLayer,
    ProductLayer,
    RightContractionLayer,
    SymmetricProductLayer,
    WedgeLayer,
)
from .primitives.projection import BladeSelector, GeometricNeutralizer
from .primitives.reflection import ReflectionLayer
from .primitives.rotor_gadget import RotorGadget
from .primitives.versor import VersorLayer

__all__ = [
    "CliffordModule",
    "GeometricGELU",
    "GeometricSquare",
    "GradeSwish",
    "VersorLayer",
    "MultiVersorLayer",
    "CliffordLinear",
    "RotorGadget",
    "CliffordLayerNorm",
    "ProductLayer",
    "GeometricProductLayer",
    "WedgeLayer",
    "SymmetricProductLayer",
    "CommutatorProductLayer",
    "AntiCommutatorProductLayer",
    "LeftContractionLayer",
    "RightContractionLayer",
    "BladeSelector",
    "GeometricNeutralizer",
    "ReflectionLayer",
    "EntropyGatedAttention",
    "GeometricProductAttention",
]
