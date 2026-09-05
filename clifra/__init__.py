# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Layout-first Clifford algebra tools for PyTorch.

``make_algebra`` constructs one concrete mathematical API. Layouts and tensor
contracts describe coefficient storage; planned operations execute those
contracts independently of the private planner and kernel implementations.
"""

__version__ = "1.4.0"

from clifra.core.algebra import AlgebraContext
from clifra.core.config import AlgebraConfig, make_algebra, make_algebra_from_config
from clifra.core.formatting import Multivector, format_multivector
from clifra.core.layout import AlgebraSpec, GradeLayout, Layout
from clifra.core.module import CliffordModule
from clifra.core.operation import PlannedOperation
from clifra.core.tensors import LaneStorage, TensorContract

__all__ = [
    "__version__",
    "AlgebraConfig",
    "AlgebraContext",
    "CliffordModule",
    "AlgebraSpec",
    "GradeLayout",
    "Layout",
    "LaneStorage",
    "TensorContract",
    "PlannedOperation",
    "Multivector",
    "format_multivector",
    "make_algebra",
    "make_algebra_from_config",
]
