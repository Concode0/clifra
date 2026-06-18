# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Primitive Clifford neural network layers."""

from .activation import GeometricGELU, GeometricSquare, GradeSwish
from .linear import CliffordLinear
from .multi_versor import MultiVersorLayer
from .normalization import CliffordLayerNorm
from .product import (
    AntiCommutatorLayer,
    CommutatorLayer,
    GeometricProductLayer,
    InnerProductLayer,
    ProductLayer,
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
    "InnerProductLayer",
    "CommutatorLayer",
    "AntiCommutatorLayer",
    "BladeSelector",
    "GeometricNeutralizer",
    "ReflectionLayer",
    "VersorLayer",
    "RotorGadget",
]
