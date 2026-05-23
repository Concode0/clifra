# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Layout-first adapter examples.

The adapter package intentionally exposes only the concrete geometry examples
that map between Euclidean data and declared Clifford layouts.
"""

from .conformal import ConformalEmbedding
from .projective import ProjectiveEmbedding

__all__ = [
    "ConformalEmbedding",
    "ProjectiveEmbedding",
]
