# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Layout-first Clifford algebra tools for PyTorch.

Clifra exposes one planner-owned algebra host through ``make_algebra``. Tensors
can use full lanes or declared compact ``GradeLayout`` lanes; planning resolves
the static graph, execution modules own the compiled tensor work, and layers
consume those layout contracts without depending on a second algebra host.
"""

__version__ = "1.0.0"

from clifra.core.config import AlgebraConfig, make_algebra, make_algebra_from_config
from clifra.core.formatting import Multivector, format_multivector
from clifra.core.foundation.module import CliffordModule
from clifra.core.runtime.algebra import AlgebraContext
from clifra.layers import CliffordLinear, RotorLayer

__all__ = [
    "__version__",
    "AlgebraConfig",
    "AlgebraContext",
    "CliffordModule",
    "Multivector",
    "format_multivector",
    "make_algebra",
    "make_algebra_from_config",
    "RotorLayer",
    "CliffordLinear",
]
