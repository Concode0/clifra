# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Loss modules built from pure functional formulas."""

import torch
import torch.nn as nn

from clifra.core.foundation.module import CliffordModule
from clifra.functional.loss import (
    asymmetry_penalty,
    bivector_regularization,
    chamfer_distance,
    conservative_force_loss,
    geometric_mse,
    hermitian_grade_regularization,
    involution_consistency_loss,
    isometry_loss,
    physics_informed_loss,
    subspace_penalty,
)


class GeometricMSELoss(CliffordModule):
    """Geometric MSE: standard MSE on multivector coefficients."""

    def __init__(self, algebra):
        """Initialize the geometric MSE loss."""
        super().__init__(algebra)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return coefficient MSE."""
        return geometric_mse(pred, target)


class SubspaceLoss(CliffordModule):
    """Penalty for energy in forbidden basis lanes."""

    def __init__(self, algebra, target_indices: list = None, exclude_indices: list = None):
        """Initialize grade constraint penalties."""
        super().__init__(algebra)

        if target_indices is not None:
            mask = torch.ones(self.algebra.dim, dtype=torch.bool)
            mask[target_indices] = False
        elif exclude_indices is not None:
            mask = torch.zeros(self.algebra.dim, dtype=torch.bool)
            mask[exclude_indices] = True
        else:
            raise ValueError("Must provide target_indices or exclude_indices")

        self.register_buffer("penalty_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return forbidden-lane energy."""
        return subspace_penalty(x, self.penalty_mask)


class IsometryLoss(CliffordModule):
    """Metric norm preservation loss."""

    def __init__(self, algebra):
        """Initialize isometry loss with metric diagonal."""
        super().__init__(algebra)
        self.register_buffer("metric_diag", self._compute_metric_diagonal())

    def _compute_metric_diagonal(self):
        """Return metric signs for canonical basis lanes."""
        basis = torch.eye(self.algebra.dim, device=self.algebra.device, dtype=self.algebra.dtype)
        sq = self.algebra.geometric_product(basis, basis)
        return sq[:, 0]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compare metric norms."""
        return isometry_loss(pred, target, self.metric_diag)


class BivectorRegularization(CliffordModule):
    """Regularize multivectors toward one target grade."""

    def __init__(self, algebra, grade=2):
        """Initialize grade regularization."""
        super().__init__(algebra)
        self.grade = grade

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Penalize energy outside ``grade``."""
        return bivector_regularization(self.algebra, x, grade=self.grade)


class HermitianGradeRegularization(CliffordModule):
    """Regularize Hermitian grade spectrum toward a target distribution."""

    def __init__(self, algebra, target_spectrum=None):
        """Initialize grade regularization."""
        super().__init__(algebra)
        n_grades = self.algebra.n + 1
        if target_spectrum is None:
            target = torch.ones(n_grades) / n_grades
        else:
            target = torch.tensor(target_spectrum, dtype=torch.float32)
            target = target / (target.sum() + 1e-8)
        self.register_buffer("target", target)

    def forward(self, features):
        """Compute grade spectrum regularization."""
        return hermitian_grade_regularization(self.algebra, features, self.target)


class ChamferDistance(nn.Module):
    """Symmetric Chamfer distance between two point clouds."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Chamfer distance."""
        return chamfer_distance(pred, target)


class ConservativeLoss(nn.Module):
    """Enforce ``F = -grad(E)`` conservative force consistency."""

    def forward(self, energy: torch.Tensor, force_pred: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Compute conservative force loss."""
        return conservative_force_loss(energy, force_pred, pos)


class PhysicsInformedLoss(nn.Module):
    """MSE plus a simple conservation penalty."""

    def __init__(self, physics_weight: float = 0.1):
        super().__init__()
        self.physics_weight = physics_weight

    def forward(self, forecast: torch.Tensor, target: torch.Tensor, lat_weights: torch.Tensor = None) -> torch.Tensor:
        """Compute physics-informed loss."""
        return physics_informed_loss(forecast, target, lat_weights=lat_weights, physics_weight=self.physics_weight)


class AsymmetryLoss(nn.Module):
    """Penalize symmetric behavior in asymmetric predictions."""

    def __init__(self, margin: float = 0.1):
        """Initialize asymmetry loss."""
        super().__init__()
        self.margin = margin

    def forward(self, logits_fwd: torch.Tensor, logits_rev: torch.Tensor) -> torch.Tensor:
        """Compute asymmetry penalty."""
        return asymmetry_penalty(logits_fwd, logits_rev, margin=self.margin)


class InvolutionConsistencyLoss(nn.Module):
    """Enforce consistency between grade involution and negation."""

    def forward(self, features: torch.Tensor, features_neg: torch.Tensor, algebra) -> torch.Tensor:
        """Compute involution consistency loss."""
        return involution_consistency_loss(features, features_neg, algebra)
