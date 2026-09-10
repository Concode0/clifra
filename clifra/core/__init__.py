# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0
"""Stable Clifford algebra API. Concrete planning and execution are private."""

from clifra.core.executors import ExecutorRegistry

from .algebra import AlgebraContext
from .config import AlgebraConfig, make_algebra, make_algebra_from_config
from .formatting import Multivector, basis_blade_label, format_multivector
from .layout import AlgebraSpec, GradeLayout, Layout
from .module import CliffordModule
from .operation import PlannedOperation
from .resources import ResourceLimits
from .tensors import LaneStorage, TensorContract

__all__ = [
    "ExecutorRegistry",
    "ResourceLimits",
    "AlgebraContext",
    "AlgebraConfig",
    "make_algebra",
    "make_algebra_from_config",
    "AlgebraSpec",
    "Layout",
    "GradeLayout",
    "LaneStorage",
    "TensorContract",
    "PlannedOperation",
    "CliffordModule",
    "Multivector",
    "basis_blade_label",
    "format_multivector",
]
