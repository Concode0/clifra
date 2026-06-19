# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Training criteria for clifra models."""

from .loss import (
    AsymmetryLoss,
    BivectorRegularization,
    ChamferDistance,
    ConservativeLoss,
    GeometricMSELoss,
    GradeEnergyRegularization,
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
    "GradeEnergyRegularization",
    "ChamferDistance",
    "ConservativeLoss",
    "PhysicsInformedLoss",
    "AsymmetryLoss",
    "InvolutionConsistencyLoss",
    "StrictOrthogonality",
    "OrthogonalitySettings",
]
