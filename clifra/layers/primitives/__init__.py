# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Primitive Clifford neural network layers."""

from .activation import GeometricGELU, GeometricSquare, GradeSwish
from .linear import CliffordLinear
from .multi_versor import MultiVersorLayer
from .normalization import CliffordLayerNorm
from .product import (
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
from .projection import BladeSelector, GeometricNeutralizer
from .reflection import ReflectionLayer
from .rotor_gadget import RotorGadget
from .versor import VersorLayer

__all__ = [
    "GeometricGELU",
    "GeometricSquare",
    "GradeSwish",
    "CliffordLinear",
    "MultiVersorLayer",
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
    "VersorLayer",
    "RotorGadget",
]
