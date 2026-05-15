"""Stateless functional operations for geometric algebra networks.

Includes activation functions, loss functions, and orthogonality enforcement.
"""

from .activation import GeometricGELU, GeometricSquare, GradeSwish
from .loss import (
    BivectorRegularization,
    ChamferDistance,
    ConservativeLoss,
    GeometricMSELoss,
    HermitianGradeRegularization,
    IsometryLoss,
    PhysicsInformedLoss,
    SubspaceLoss,
)
from .orthogonality import OrthogonalitySettings, StrictOrthogonality
from .products import (
    anti_commutator,
    clifford_conjugation,
    commutator,
    dual,
    embed_vector,
    geometric_product,
    grade_involution,
    grade_projection,
    inner_product,
    norm_sq,
    product,
    projected_product,
    reverse,
    wedge,
)

__all__ = [
    # activations
    "GeometricGELU",
    "GeometricSquare",
    "GradeSwish",
    # products
    "product",
    "projected_product",
    "geometric_product",
    "wedge",
    "inner_product",
    "commutator",
    "anti_commutator",
    "grade_projection",
    "reverse",
    "grade_involution",
    "clifford_conjugation",
    "dual",
    "norm_sq",
    "embed_vector",
    # losses
    "GeometricMSELoss",
    "SubspaceLoss",
    "IsometryLoss",
    "BivectorRegularization",
    "HermitianGradeRegularization",
    "ChamferDistance",
    "ConservativeLoss",
    "PhysicsInformedLoss",
    # orthogonality
    "StrictOrthogonality",
    "OrthogonalitySettings",
]
