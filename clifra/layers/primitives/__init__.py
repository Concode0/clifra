"""Primitive Clifford neural network layers."""

from .activation import GeometricGELU, GeometricSquare, GradeSwish
from .linear import CliffordLinear
from .multi_rotor import MultiRotorLayer
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
from .rotor import RotorLayer
from .rotor_gadget import RotorGadget

__all__ = [
    "GeometricGELU",
    "GeometricSquare",
    "GradeSwish",
    "CliffordLinear",
    "MultiRotorLayer",
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
    "RotorLayer",
    "RotorGadget",
]
