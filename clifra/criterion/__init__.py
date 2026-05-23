"""Training criteria for Clifra models."""

from .loss import (
    AsymmetryLoss,
    BivectorRegularization,
    ChamferDistance,
    ConservativeLoss,
    GeometricMSELoss,
    HermitianGradeRegularization,
    InvolutionConsistencyLoss,
    IsometryLoss,
    PhysicsInformedLoss,
    SubspaceLoss,
)
from .orthogonality import OrthogonalitySettings, StrictOrthogonality

__all__ = [
    "GeometricMSELoss",
    "SubspaceLoss",
    "IsometryLoss",
    "BivectorRegularization",
    "HermitianGradeRegularization",
    "ChamferDistance",
    "ConservativeLoss",
    "PhysicsInformedLoss",
    "AsymmetryLoss",
    "InvolutionConsistencyLoss",
    "StrictOrthogonality",
    "OrthogonalitySettings",
]
