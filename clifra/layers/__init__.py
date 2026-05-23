"""Neural network layers built on Clifford algebra.

The framework namespace exports primitives and reusable blocks. Example
adapters live under :mod:`clifra.layers.adapters` and are imported explicitly.
"""

from clifra.core.foundation.module import CliffordModule

from .blocks.attention import EntropyGatedAttention, GeometricProductAttention
from .blocks.multi_rotor_ffn import MultiRotorFFN
from .blocks.transformer import GeometricTransformerBlock
from .primitives.linear import CliffordLinear
from .primitives.multi_rotor import MultiRotorLayer
from .primitives.normalization import CliffordLayerNorm
from .primitives.product import (
    AntiCommutatorLayer,
    CommutatorLayer,
    GeometricProductLayer,
    InnerProductLayer,
    ProductLayer,
    WedgeLayer,
)
from .primitives.projection import BladeSelector, GeometricNeutralizer
from .primitives.reflection import ReflectionLayer
from .primitives.rotor import RotorLayer
from .primitives.rotor_gadget import RotorGadget

__all__ = [
    "CliffordModule",
    "RotorLayer",
    "MultiRotorLayer",
    "CliffordLinear",
    "RotorGadget",
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
    "EntropyGatedAttention",
    "GeometricProductAttention",
    "MultiRotorFFN",
    "GeometricTransformerBlock",
]
