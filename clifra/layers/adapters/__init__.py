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
