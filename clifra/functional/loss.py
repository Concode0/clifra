# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Pure loss and regularization formulas."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from clifra.core.runtime.metric import hermitian_grade_spectrum


def geometric_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return coefficient-wise mean squared error."""
    return F.mse_loss(pred, target, reduction="mean")


def subspace_penalty(values: torch.Tensor, penalty_mask: torch.Tensor) -> torch.Tensor:
    """Return mean squared energy in masked coefficient lanes."""
    penalty_components = values[..., penalty_mask]
    return (penalty_components**2).sum(dim=-1).mean()


def isometry_loss(pred: torch.Tensor, target: torch.Tensor, metric_diag: torch.Tensor) -> torch.Tensor:
    """Return MSE between metric norms of ``pred`` and ``target``."""
    pred_norm = ((pred**2) * metric_diag).sum(dim=-1)
    target_norm = ((target**2) * metric_diag).sum(dim=-1)
    return F.mse_loss(pred_norm, target_norm)


def bivector_regularization(algebra, values: torch.Tensor, *, grade: int = 2) -> torch.Tensor:
    """Penalize energy outside one target grade."""
    target_part = algebra.grade_projection(values, grade)
    residual = values - target_part
    return (residual**2).sum(dim=-1).mean()


def hermitian_grade_regularization(algebra, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return MSE between actual and target Hermitian grade distributions."""
    flat = features.reshape(-1, features.shape[-1])
    spectrum = hermitian_grade_spectrum(algebra, flat)
    dist = spectrum / (spectrum.sum(dim=-1, keepdim=True) + 1e-8)
    return F.mse_loss(dist.mean(dim=0), target)


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return symmetric squared Chamfer distance between point clouds."""
    diff = pred.unsqueeze(2) - target.unsqueeze(1)
    dist_sq = (diff**2).sum(dim=-1)
    min_dist_pred = dist_sq.min(dim=2)[0].mean(dim=1)
    min_dist_target = dist_sq.min(dim=1)[0].mean(dim=1)
    return (min_dist_pred + min_dist_target).mean()


def conservative_force_loss(energy: torch.Tensor, force_pred: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Return MSE between predicted forces and ``-grad(energy, pos)``."""
    force_from_energy = -torch.autograd.grad(energy.sum(), pos, create_graph=True, retain_graph=True)[0]
    return F.mse_loss(force_pred, force_from_energy)


def physics_informed_loss(
    forecast: torch.Tensor,
    target: torch.Tensor,
    *,
    lat_weights: torch.Tensor = None,
    physics_weight: float = 0.1,
) -> torch.Tensor:
    """Return forecast MSE plus a weighted global-conservation penalty."""
    mse_loss = F.mse_loss(forecast, target)

    if lat_weights is not None and forecast.dim() == 4:
        weights = lat_weights.view(1, -1, 1, 1).to(forecast.device)
        forecast_mean = (forecast * weights).sum(dim=[1, 2]) / weights.sum()
        target_mean = (target * weights).sum(dim=[1, 2]) / weights.sum()
    else:
        forecast_mean = forecast.mean(dim=list(range(1, forecast.dim() - 1)))
        target_mean = target.mean(dim=list(range(1, target.dim() - 1)))

    conservation_loss = F.mse_loss(forecast_mean, target_mean)
    return mse_loss + physics_weight * conservation_loss


def asymmetry_penalty(logits_fwd: torch.Tensor, logits_rev: torch.Tensor, *, margin: float = 0.1) -> torch.Tensor:
    """Return penalty for correlation above ``margin`` between forward and reversed logits."""
    forward = logits_fwd.detach().flatten()
    reverse = logits_rev.flatten()
    forward_centered = forward - forward.mean()
    reverse_centered = reverse - reverse.mean()
    corr = (forward_centered * reverse_centered).sum() / (
        forward_centered.norm() * reverse_centered.norm() + 1e-8
    )
    return F.relu(corr - margin)


def involution_consistency_loss(features: torch.Tensor, features_neg: torch.Tensor, algebra) -> torch.Tensor:
    """Return MSE between grade-involuted features and negated-input features."""
    return F.mse_loss(algebra.grade_involution(features), features_neg)
